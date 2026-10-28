"""The frame table: output file, ledger and resume.

Only the parent process opens an output file for writing. Rows are addressed
by label: :meth:`FrameTableWriter.write_block`
checks every frame's trainId and every train's frame count against the plan
before anything is stored, so a dropped train can never shift frames onto the
wrong train.

There are no NaN sentinels anywhere in a file written here. A frame that was
not integrated carries a status code and zeros; ``status`` is what
distinguishes the two, and pooling must select on it.

What differs between passes is only the *schema*: which per-frame scalars a
detector produces, and which groups it stores its operator and masks in. Those
are class attributes, so the ledger, the resume logic, the label checks and the
pooling reducer are written once.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import h5py
import numpy as np

from analysis.common.cpu import package_versions
from analysis.common.plan import Block, RunPlan
from analysis.common.status import UNFINISHED, FrameStatus

__all__ = [
    "GATING_PACKAGES",
    "ConfigHashMismatch",
    "FrameRows",
    "FrameTableWriter",
    "IncompleteRun",
    "OperatorMismatch",
    "SchemaMismatch",
    "PassConfig",
    "as_handle",
    "config_payload",
    "per_label",
    "pooled_per_train",
    "q_centers",
    "status_counts",
    "version_changes",
]

log = logging.getLogger(__name__)


#: Packages whose version must match for a file to be resumed or counted as
#: complete. pyFAI builds both operators and integrates every JUNGFRAU frame, so
#: a file written under another version would be a mix of two integrators.
GATING_PACKAGES: tuple[str, ...] = ("pyFAI",)

#: Compares the operator stored in an open output file with the one a pass built
#: now, returning one reason per difference. Each pass supplies its own, since
#: only it knows its operator's layout and which of its arrays are exact.
OperatorCheck = Callable[[h5py.File], list[str]]

#: How to reprocess a file this pass refuses to resume into.
_REPROCESS_HINT = (
    "move it aside with scripts/reprocess_safe.py and reprocess, or pass "
    "overwrite=True to replace it"
)


class ConfigHashMismatch(RuntimeError):
    """An existing output file was written with a different configuration."""


class OperatorMismatch(ConfigHashMismatch):
    """An existing output file was integrated by another operator or pyFAI.

    The config hash covers the inputs, not what pyFAI makes of them, so it can
    match while the operator does not: after a pyFAI upgrade, say. Resuming
    would mix rows from two integrators, and finalising would then stamp the
    new version on all of them.

    The operator is compared array by array, not by sha256, because the same
    build differs by one ULP in the solid angle between AVX2 and AVX-512 nodes,
    and the ``allcpu`` partition mixes both.
    """


class SchemaMismatch(RuntimeError):
    """An existing output file has a different frame table from this writer's.

    A column added to a pass does not change ``config_hash``, which covers the
    config and not the layout. Without this check a file written before the
    column would be reopened for resume and then fail on the first write,
    part-processed. The check is derived from ``FRAME_VECTORS``/``FRAME_MATRICES``
    rather than from a version number, so it cannot fall behind the schema.
    """


class IncompleteRun(RuntimeError):
    """Some frames did not reach ``OK`` and ``allow_incomplete`` is False."""


@runtime_checkable
class PassConfig(Protocol):
    """What :class:`FrameTableWriter` needs of a pass's config.

    Both passes satisfy this with a frozen dataclass; :func:`config_payload`
    additionally requires that, since it records every field in provenance.
    """

    npt: int
    overwrite: bool

    @property
    def output_file(self) -> Path: ...

    @property
    def operational_fields(self) -> frozenset[str]: ...

    def config_hash(self) -> str: ...


def config_payload(cfg: Any) -> dict[str, Any]:
    """Every config field, JSON-ready, for the provenance record.

    ``frozenset`` fields are sorted so the rendering is stable, and ``method``
    becomes a list because a tuple round-trips through JSON as one anyway.
    """
    payload: dict[str, Any] = {
        key: (sorted(value) if isinstance(value, frozenset) else value)
        for key, value in asdict(cfg).items()
    }
    if "method" in payload:
        payload["method"] = list(cfg.method)
    return payload


@dataclass(slots=True)
class FrameRows:
    """One block's rows of a frame table, keyed by the file's dataset names.

    A worker fills it and the parent's writer stores it. The ``trainId``
    column holds the labels the worker read; the writer checks them against
    the plan and stores the plan's.
    """

    columns: dict[str, np.ndarray]
    bits_present: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.columns["status"].size)

    @property
    def status(self) -> np.ndarray:
        return self.columns["status"]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    def put(self, row: int | slice, values: Mapping[str, Any]) -> None:
        """Store ``values`` into ``row`` of the named columns."""
        for name, value in values.items():
            self.columns[name][row] = value


class FrameTableWriter:
    """The single writer for one run's output file.

    Subclasses declare the schema and add whatever ``store_*`` methods their
    pass needs for its operator and masks. The schema is the only copy:
    :meth:`empty_rows` allocates a worker's rows from it.
    """

    #: Per-frame ``(n,)`` columns, dataset name to dtype. ``trainId`` is
    #: required: it is the label every row is addressed by.
    FRAME_VECTORS: dict[str, Any] = {}
    #: The ``(n, npt)`` float32 sufficient statistics.
    FRAME_MATRICES: tuple[str, ...] = ("signal", "normalization", "variance")
    #: Per-frame ``(n, k)`` columns, dataset name to ``(k, dtype)``.
    #: Zero-filled for frames never written, like every column.
    FRAME_ARRAYS: dict[str, tuple[int, Any]] = {}
    #: Groups created empty at layout time for the pass to fill later.
    EXTRA_GROUPS: tuple[str, ...] = ()
    #: What ``signal`` counts, recorded as ``provenance.attrs["units"]`` when set.
    #: A file whose stored value differs, or that has none, is refused rather
    #: than resumed, so rows of two units never share a file.
    UNITS: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.FRAME_VECTORS and "trainId" not in cls.FRAME_VECTORS:
            raise TypeError(f"{cls.__name__}.FRAME_VECTORS has no 'trainId' column")

    @classmethod
    def columns(cls) -> tuple[str, ...]:
        """Every per-frame dataset this writer stores."""
        return (*cls.FRAME_VECTORS, *cls.FRAME_MATRICES, *cls.FRAME_ARRAYS)

    @classmethod
    def empty_rows(cls, n: int, npt: int) -> FrameRows:
        """``n`` rows of zeros in this schema, every one ``NOT_PROCESSED``."""
        columns = {
            name: np.zeros(n, dtype=dtype) for name, dtype in cls.FRAME_VECTORS.items()
        }
        columns["status"] = np.full(n, FrameStatus.NOT_PROCESSED, dtype=np.uint8)
        for name in cls.FRAME_MATRICES:
            columns[name] = np.zeros((n, npt), dtype=np.float32)
        for name, (width, dtype) in cls.FRAME_ARRAYS.items():
            columns[name] = np.zeros((n, width), dtype=dtype)
        return FrameRows(columns)

    def __init__(self, handle: h5py.File, cfg: PassConfig, plan: RunPlan) -> None:
        self._f = handle
        self._cfg = cfg
        self._plan = plan

    @classmethod
    def open_or_create(
        cls,
        cfg: PassConfig,
        plan: RunPlan,
        path: Path | None = None,
        *,
        operator_check: OperatorCheck | None = None,
    ) -> FrameTableWriter:
        """Open the run's output file, creating and initialising it if needed.

        An existing file is resumed into only if :meth:`check_resumable`
        allows it, unless ``cfg.overwrite``, which recreates it.

        :param operator_check: see :meth:`check_resumable`.
        """
        path = Path(path) if path is not None else cfg.output_file
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not cfg.overwrite:
            cls.check_resumable(cfg, path, operator_check=operator_check)
            return cls(h5py.File(path, "r+"), cfg, plan)

        handle = h5py.File(path, "w")
        cls._create_layout(handle, cfg, plan, cfg.config_hash())
        return cls(handle, cfg, plan)

    @classmethod
    def check_resumable(
        cls,
        cfg: PassConfig,
        path: Path,
        *,
        operator_check: OperatorCheck | None = None,
    ) -> None:
        """Refuse an existing file that resuming would turn into a mixture.

        Read-only, so a pass can call it before any expensive setup as well as
        when it opens the file. A missing file, or ``cfg.overwrite``, passes.

        :param operator_check: compares the stored operator with the one this
            run built. ``None`` skips it, for a caller with no operator.
        :raises ConfigHashMismatch: the config hash differs.
        :raises SchemaMismatch: the frame table lacks a column, or stores
            other units than this writer (``UNITS``).
        :raises OperatorMismatch: the operator, or the version of a
            :data:`GATING_PACKAGES` package, differs.
        """
        path = Path(path)
        if cfg.overwrite or not path.exists():
            return
        config_hash = cfg.config_hash()
        with h5py.File(path, "r") as existing:
            stored = existing["provenance"].attrs.get("config_hash")
            stored_units = existing["provenance"].attrs.get("units")
            missing = cls._missing_columns(existing)
            versions = existing["provenance"].attrs.get("package_versions")
            # An empty /operator is a file that died before storing one, with
            # no rows to mix; there is nothing to compare.
            has_operator = "operator" in existing and len(existing["operator"]) > 0
            differences = (
                operator_check(existing) if operator_check and has_operator else []
            )
        if stored != config_hash:
            raise ConfigHashMismatch(
                f"{path} was written with config hash {stored}, "
                f"this run has {config_hash}; {_REPROCESS_HINT}"
            )
        if cls.UNITS is not None and stored_units != cls.UNITS:
            raise SchemaMismatch(
                f"{path} stores {stored_units or 'no units attribute'}, this "
                f"pass writes {cls.UNITS}; rows of two units must never share "
                f"a file; {_REPROCESS_HINT}"
            )
        if missing:
            raise SchemaMismatch(
                f"{path} has no {missing} in its frame table, so it predates "
                f"a column this pass now stores; {_REPROCESS_HINT}"
            )
        changed = version_changes(versions) + differences
        if changed:
            raise OperatorMismatch(
                f"{path} was integrated differently from this run "
                f"({'; '.join(changed)}), so resuming into it would mix the "
                f"two; {_REPROCESS_HINT}"
            )

    @classmethod
    def _missing_columns(cls, handle: h5py.File) -> list[str]:
        """Frame-table datasets this writer stores that ``handle`` does not have."""
        frames = handle.get("frames")
        wanted = cls.columns()
        if frames is None:
            return sorted(wanted)
        return sorted(name for name in wanted if name not in frames)

    @classmethod
    def _create_layout(
        cls, handle: h5py.File, cfg: PassConfig, plan: RunPlan, config_hash: str
    ) -> None:
        n, npt = plan.n_frames, cfg.npt
        frames = handle.create_group("frames")
        # One chunk per train keeps a block's writes contiguous; the largest
        # train in the run sets the chunk length.
        chunk_rows = max(
            (train.n_frames for train in plan.trains if train.n_frames), default=1
        )
        for name, dtype in cls.FRAME_VECTORS.items():
            fill = FrameStatus.NOT_PROCESSED if name == "status" else 0
            frames.create_dataset(
                name,
                shape=(n,),
                dtype=dtype,
                fillvalue=fill,
                chunks=(min(chunk_rows, n),) if n else None,
                compression="gzip",
                compression_opts=1,
            )
        for name in cls.FRAME_MATRICES:
            frames.create_dataset(
                name,
                shape=(n, npt),
                dtype=np.float32,
                fillvalue=0.0,
                chunks=(min(chunk_rows, n), npt) if n else None,
                compression="gzip",
                compression_opts=1,
            )
        for name, (width, dtype) in cls.FRAME_ARRAYS.items():
            frames.create_dataset(
                name,
                shape=(n, width),
                dtype=dtype,
                fillvalue=0,
                chunks=(min(chunk_rows, n), width) if n else None,
                compression="gzip",
                compression_opts=1,
            )

        trains = handle.create_group("trains")
        trains["trainId"] = np.array([t.train_id for t in plan.trains], dtype=np.uint64)
        trains["first"] = np.array([t.first_row for t in plan.trains], dtype=np.uint64)
        trains["count"] = np.array([t.n_frames for t in plan.trains], dtype=np.uint32)
        trains["status"] = np.array(
            [int(t.status) for t in plan.trains], dtype=np.uint8
        )

        provenance = handle.create_group("provenance")
        provenance.attrs["config_hash"] = config_hash
        provenance.attrs["config"] = json.dumps(
            config_payload(cfg), sort_keys=True, default=str
        )
        # Which fields the hash covers, so a stored file explains its own
        # compatibility rules rather than requiring the reader to have the
        # matching source version to hand. Asked of the config rather than
        # taken from the shared constant: the two passes exclude different
        # sets, and a provenance record that says otherwise is worse than none.
        provenance.attrs["config_operational_fields"] = json.dumps(
            sorted(cfg.operational_fields)
        )
        provenance.attrs["detector_name"] = plan.detector_name
        provenance.attrs["run_checks"] = json.dumps(plan.checks, default=str)
        # Recorded now as well as at finalise, so that a file whose run died
        # before finishing still says which pyFAI wrote its rows.
        provenance.attrs["package_versions"] = json.dumps(package_versions())
        # At layout time, not at finalise: a file the pass never finished still
        # says what its rows count.
        if cls.UNITS is not None:
            provenance.attrs["units"] = cls.UNITS
        for name in cls.EXTRA_GROUPS:
            handle.create_group(name)
        handle.flush()

    @property
    def handle(self) -> h5py.File:
        """The open file, for a reducer to read the rows written so far."""
        return self._f

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> FrameTableWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def block_complete(self, block: Block) -> bool:
        """A block is complete when none of its frames is :data:`UNFINISHED`.

        So a rerun retries ``WORKER_ERROR`` blocks as well as unprocessed ones,
        and leaves ``LABEL_MISMATCH`` and ``DATA_CHECK_FAILED`` alone: those
        would come out the same.
        """
        rows = block.rows()
        if rows.size == 0:
            return True
        status = self._f["frames/status"][rows[0] : rows[-1] + 1]
        return not bool(np.isin(status, [int(code) for code in UNFINISHED]).any())

    def write_block(self, block: Block, result: FrameRows) -> None:
        """Validate the block's labels, then store its rows.

        :raises ValueError: the result's shape does not match the block, which
            is a programming fault. A label disagreement is not an error. It is
            stored as ``LABEL_MISMATCH``.
        """
        rows = block.rows()
        if result.n_frames != rows.size:
            raise ValueError(
                f"block {block.index} expects {rows.size} frames, "
                f"result carries {result.n_frames}"
            )

        integrated = result.status == FrameStatus.OK
        expected_trains = (
            np.concatenate(
                [
                    np.full(count, train_id, dtype=np.uint64)
                    for train_id, count in zip(
                        block.train_ids, block.expected_frames, strict=True
                    )
                ]
            )
            if rows.size
            else np.zeros(0, dtype=np.uint64)
        )
        mislabelled = integrated & (result["trainId"] != expected_trains)
        if mislabelled.any():
            log.error(
                "block %d: %d frames carry a trainId the plan does not expect",
                block.index,
                int(mislabelled.sum()),
            )
            result.status[mislabelled] = FrameStatus.LABEL_MISMATCH

        start, stop = int(rows[0]), int(rows[-1]) + 1
        frames = self._f["frames"]
        frames["trainId"][start:stop] = expected_trains
        for name in self.columns():
            if name != "trainId":
                frames[name][start:stop] = result[name]
        self._resolve_block_error(block)
        self._f.flush()

    def _resolve_block_error(self, block: Block) -> None:
        """Move a retried block's earlier error to ``block_errors_resolved``.

        Kept rather than dropped: a block that failed once and then succeeded
        is worth knowing about when a node or the file system misbehaves.
        """
        attrs = self._f["provenance"].attrs
        if "block_errors" not in attrs:
            return
        errors = json.loads(attrs["block_errors"])
        message = errors.pop(str(block.index), None)
        if message is None:
            return
        resolved = json.loads(attrs.get("block_errors_resolved", "{}"))
        resolved[str(block.index)] = message
        attrs["block_errors_resolved"] = json.dumps(resolved)
        if errors:
            attrs["block_errors"] = json.dumps(errors)
        else:
            del attrs["block_errors"]
        log.info(
            "block %d succeeded on retry (it had failed: %s)", block.index, message
        )

    def mark(self, block: Block, status: FrameStatus, message: str = "") -> None:
        """Record a status for every frame of a block, with no data."""
        rows = block.rows()
        if rows.size == 0:
            return
        self._f["frames/status"][int(rows[0]) : int(rows[-1]) + 1] = status
        if message:
            errors = self._f["provenance"].attrs.get("block_errors", "{}")
            recorded = json.loads(errors)
            recorded[str(block.index)] = message
            self._f["provenance"].attrs["block_errors"] = json.dumps(recorded)
        self._f.flush()

    def mark_remaining(self, status: FrameStatus) -> None:
        """Stamp every still-unprocessed frame, e.g. after a broken pool."""
        current = self._f["frames/status"][:]
        current[current == FrameStatus.NOT_PROCESSED] = status
        self._f["frames/status"][:] = current
        self._f.flush()

    def finalise(self, provenance: dict[str, Any]) -> None:
        group = self._f["provenance"]
        for key, value in provenance.items():
            group.attrs[key] = (
                value
                if isinstance(value, (str, int, float))
                else json.dumps(value, default=str)
            )
        self._f.flush()

    def status_summary(self) -> dict[str, int]:
        status = self._f["frames/status"][:]
        return {
            code.name: int((status == code).sum())
            for code in FrameStatus
            if (status == code).any()
        }


def version_changes(stored: str | None) -> list[str]:
    """How the :data:`GATING_PACKAGES` installed now differ from a record.

    :param stored: a ``package_versions`` provenance attribute, JSON. A file
        without one cannot be checked and reports nothing.
    :returns: one ``"<package> <stored> -> <installed>"`` entry per difference.
    """
    if not stored:
        return []
    recorded = json.loads(stored)
    installed = package_versions(GATING_PACKAGES)
    return [
        f"{name} {recorded.get(name)} -> {installed[name]}"
        for name in GATING_PACKAGES
        if recorded.get(name) != installed[name]
    ]


# The reducers below take the output file rather than a live writer, so the same
# code serves the run that produced it and any later analysis of it. They
# read the ``/trains`` table for the row-to-train map: ``frames/trainId`` is
# only filled for blocks that were actually written, so a run with an
# unprocessed block has rows carrying trainId 0.


def status_counts(
    status: np.ndarray,
    *,
    exclude: FrameStatus | None = None,
    nonzero_only: bool = False,
) -> dict[str, int]:
    """Count a status array by code name.

    :param status: the ledger column, or any slice of it.
    :param exclude: a code to leave out, e.g. ``OK`` when reporting failures.
    :param nonzero_only: drop codes that do not occur, so a clean run reports
        an empty dict rather than a row of zeros.
    :returns: ``{code name: count}``.
    """
    return {
        code.name: int(total)
        for code in FrameStatus
        if code is not exclude
        and ((total := (status == code).sum()) or not nonzero_only)
    }


@contextmanager
def as_handle(source: Any) -> Any:
    """Accept an open file or a path, and yield an open file either way."""
    if isinstance(source, h5py.File):
        yield source
    else:
        with h5py.File(source, "r") as handle:
            yield handle


def q_centers(handle: h5py.File, npt: int) -> np.ndarray:
    """The stored q axis, or bin indices for a file written without one."""
    if "q" in handle:
        return np.asarray(handle["q/centers"][:])
    return np.arange(npt, dtype=np.float64)


def pooled_per_train(source: Any) -> Any:
    """Per-train pooled ``I(q)`` and ``σ(q)`` over the ``OK`` frames.

    ``I(q) = Σ S / Σ N`` and ``σ(q) = sqrt(Σ V) / Σ N`` over each train's
    integrated frames. Trains with no integrated frame are returned as zeros
    with ``n_frames == 0``; there are no NaN sentinels, so callers select on
    ``n_frames``.
    """
    import xarray as xr

    with as_handle(source) as handle:
        frames = handle["frames"]
        status = frames["status"][:]
        npt = int(frames["signal"].shape[1])
        q = q_centers(handle, npt)
        train_ids = handle["trains/trainId"][:]
        first = handle["trains/first"][:]
        count = handle["trains/count"][:]
        # Inside the ``with``: an AttributeManager outlives its file silently.
        units = handle["provenance"].attrs.get("units")

        intensity = np.zeros((train_ids.size, npt), dtype=np.float64)
        sigma = np.zeros((train_ids.size, npt), dtype=np.float64)
        counts = np.zeros(train_ids.size, dtype=np.uint32)
        negative_variance = 0

        for index in range(train_ids.size):
            n = int(count[index])
            if n == 0:
                continue
            rows = slice(int(first[index]), int(first[index]) + n)
            ok = status[rows] == FrameStatus.OK
            counts[index] = int(ok.sum())
            if not ok.any():
                continue
            signal = frames["signal"][rows][ok].sum(axis=0)
            normalization = frames["normalization"][rows][ok].sum(axis=0)
            variance = frames["variance"][rows][ok].sum(axis=0)
            valid = normalization > 0
            intensity[index, valid] = signal[valid] / normalization[valid]
            # A pass may store an unbiased variance that individual frames can
            # drive negative (the JUNGFRAU error model does). sqrt of that is
            # not a number, and there are no NaN sentinels here, so such a bin
            # gets sigma 0 and is counted instead of being quietly rooted.
            positive = valid & (variance > 0)
            negative_variance += int((valid & (variance < 0)).sum())
            sigma[index, positive] = (
                np.sqrt(variance[positive]) / normalization[positive]
            )

    result = xr.Dataset(
        {
            "intensity": (("trainId", "q"), intensity),
            "sigma": (("trainId", "q"), sigma),
            "n_frames": (("trainId",), counts),
        },
        coords={"trainId": train_ids, "q": q},
    )
    result.attrs["negative_variance_bins"] = negative_variance
    if units is not None:
        result.attrs["units"] = str(units)
    return result


def per_label(
    source: Any,
    *,
    label_column: str,
    label_dim: str,
    dtype: Any = np.float32,
    chunk_rows: int = 20_000,
    carry_attrs: tuple[str, ...] = (),
) -> Any:
    """Per-frame ``I(q)`` as a ``(trainId, <label>, q)`` DataArray.

    Every frame is placed by its *stored* trainId and label, never by its row
    position, so a dropped or short train cannot slide frames onto the wrong
    train. Only ``OK`` frames carry trustworthy labels, so anything else is
    counted into ``attrs["unplaced"]`` rather than guessed onto a slot; empty
    slots are zeros with ``n_frames == 0``, never NaN.

    :param source: an open handle or a path to a finished output file.
    :param label_column: the ``/frames`` column holding the second axis's label,
        e.g. ``"reader_pulseId"`` or ``"cellId"``.
    :param label_dim: name of that axis on the result, e.g. ``"pulseId"``.
    :param dtype: storage for the intensity grid. f4 on a 3000-train,
        155-pulse run at ``npt`` 500 costs ~0.9 GB; f8 doubles it.
    :param chunk_rows: rows read per pass, to bound peak memory on the large
        AGIPD grids.
    :param carry_attrs: provenance attributes copied onto the result, for
        things a DAMNIT user needs on the object rather than only in the file.
    :returns: the grid, with ``n_frames`` as a non-dimension coordinate. That
        keeps it a DataArray, which DAMNIT renders as its shape rather than as a
        total size.
    """
    import xarray as xr

    noun = label_dim.removesuffix("Id")
    with as_handle(source) as handle:
        frames = handle["frames"]
        status = frames["status"][:]
        row_train = frames["trainId"][:]
        row_label = frames[label_column][:]
        npt = int(frames["signal"].shape[1])
        q = q_centers(handle, npt)
        train_ids = handle["trains/trainId"][:]
        if not np.all(np.diff(train_ids.astype(np.int64)) > 0):
            raise ValueError("the train table is not strictly increasing")
        # Read inside the ``with``: an AttributeManager kept past it returns
        # the default rather than raising.
        carried = {
            name: handle["provenance"].attrs.get(name, "") for name in carry_attrs
        }

        placeable = status == FrameStatus.OK
        labels = np.unique(row_label[placeable])
        unplaced = status_counts(status, exclude=FrameStatus.OK, nonzero_only=True)

        intensity = np.zeros((train_ids.size, labels.size, npt), dtype=dtype)
        n_frames = np.zeros((train_ids.size, labels.size), dtype=np.uint8)

        for start in range(0, status.size, chunk_rows):
            stop = min(start + chunk_rows, status.size)
            keep = placeable[start:stop]
            if not keep.any():
                continue
            trains = row_train[start:stop][keep]
            # searchsorted returns size for anything past the last train, so
            # the result is clamped before it is used as an index: an unknown
            # trainId must reach the check below, not raise IndexError first.
            train_index = np.clip(
                np.searchsorted(train_ids, trains), 0, train_ids.size - 1
            )
            if not np.array_equal(train_ids[train_index], trains):
                raise ValueError(
                    "a frame carries a trainId that is not in the train table"
                )
            label_index = np.searchsorted(labels, row_label[start:stop][keep])

            signal = frames["signal"][start:stop][keep]
            normalization = frames["normalization"][start:stop][keep]
            value = np.zeros_like(signal)
            np.divide(signal, normalization, out=value, where=normalization > 0)
            intensity[train_index, label_index] = value
            n_frames[train_index, label_index] = 1

    placed = int(n_frames.sum())
    if placed != int(placeable.sum()):
        raise ValueError(
            f"{int(placeable.sum()) - placed} frame(s) shared a (train, {noun}) "
            f"slot with another; the {noun} ids do not identify frames uniquely"
        )

    result = xr.DataArray(
        intensity,
        dims=("trainId", label_dim, "q"),
        coords={
            "trainId": train_ids,
            label_dim: labels,
            "q": q,
            "n_frames": (("trainId", label_dim), n_frames),
        },
        name="intensity",
    )
    result.attrs["unplaced"] = json.dumps(unplaced, sort_keys=True)
    result.attrs["n_placed"] = placed
    result.attrs.update(carried)
    return result
