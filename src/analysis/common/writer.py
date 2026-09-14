"""The frame table: output file, ledger and resume.

Only the parent process opens an output file for writing (AGIPD context file §3
rule 8). Rows are addressed by label: :meth:`FrameTableWriter.write_block`
checks every frame's trainId and every train's frame count against the plan
before anything is stored, so a dropped train can never shift frames onto the
wrong train (CLAUDE.md pitfall 4).

There are no NaN sentinels anywhere in a file written here. A frame that was
not integrated carries a status code and zeros (§3 rule 7); ``status`` is what
distinguishes the two, and pooling must select on it.

What differs between passes is only the *schema*: which per-frame scalars a
detector produces, and which groups it stores its operator and masks in. Those
are class attributes, so the ledger, the resume logic, the label checks and the
pooling reducer are written once.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import h5py
import numpy as np

from analysis.common.plan import Block, RunPlan
from analysis.common.status import FrameStatus

__all__ = [
    "ConfigHashMismatch",
    "FrameTableWriter",
    "IncompleteRun",
    "SchemaMismatch",
    "PassConfig",
    "config_payload",
    "pooled_per_train",
]

log = logging.getLogger(__name__)


class ConfigHashMismatch(RuntimeError):
    """An existing output file was written with a different configuration."""


class SchemaMismatch(RuntimeError):
    """An existing output file has a different frame table from this writer's.

    A column added to a pass does not change ``config_hash`` — the hash covers
    the config, not the layout — so without this check a file written before
    the column would be reopened for resume and then fail on the first write,
    part-processed. Derived from ``FRAME_VECTORS``/``FRAME_MATRICES`` rather
    than from a version number, so it cannot fall behind the schema it guards.
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


class FrameTableWriter:
    """The single writer for one run's output file.

    Subclasses declare the schema and add whatever ``store_*`` methods their
    pass needs for its operator and masks.
    """

    #: Per-frame ``(n,)`` columns, dataset name to dtype. ``trainId`` is
    #: required: it is the label every row is addressed by.
    FRAME_VECTORS: dict[str, Any] = {}
    #: Where each ``FRAME_VECTORS`` column comes from on a block result. Every
    #: column except ``trainId`` needs an entry; ``trainId`` is filled from the
    #: plan, not from the worker, which is what makes the label check bite.
    FRAME_VECTOR_SOURCES: dict[str, str] = {}
    #: The ``(n, npt)`` sufficient statistics, read off the result by name.
    FRAME_MATRICES: tuple[str, ...] = ("signal", "normalization", "variance")
    #: Groups created empty at layout time for the pass to fill later.
    EXTRA_GROUPS: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.FRAME_VECTORS:
            return
        if "trainId" not in cls.FRAME_VECTORS:
            raise TypeError(f"{cls.__name__}.FRAME_VECTORS has no 'trainId' column")
        missing = set(cls.FRAME_VECTORS) - set(cls.FRAME_VECTOR_SOURCES) - {"trainId"}
        if missing:
            raise TypeError(
                f"{cls.__name__} declares columns with no source on the block "
                f"result: {sorted(missing)}"
            )
        unknown = set(cls.FRAME_VECTOR_SOURCES) - set(cls.FRAME_VECTORS)
        if unknown:
            raise TypeError(
                f"{cls.__name__} maps sources for columns it does not create: "
                f"{sorted(unknown)}"
            )

    def __init__(self, handle: h5py.File, cfg: PassConfig, plan: RunPlan) -> None:
        self._f = handle
        self._cfg = cfg
        self._plan = plan

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def open_or_create(
        cls, cfg: PassConfig, plan: RunPlan, path: Path | None = None
    ) -> FrameTableWriter:
        """Open the run's output file, creating and initialising it if needed.

        An existing file whose stored config hash differs is refused unless
        ``cfg.overwrite``: resuming into a file built under a different
        configuration would mix incompatible rows.
        """
        path = Path(path) if path is not None else cfg.output_file
        path.parent.mkdir(parents=True, exist_ok=True)
        config_hash = cfg.config_hash()

        if path.exists() and not cfg.overwrite:
            with h5py.File(path, "r") as existing:
                stored = existing["provenance"].attrs.get("config_hash")
                missing = cls._missing_columns(existing)
            if stored != config_hash:
                raise ConfigHashMismatch(
                    f"{path} was written with config hash {stored}, "
                    f"this run has {config_hash}; pass overwrite=True to replace it"
                )
            if missing:
                raise SchemaMismatch(
                    f"{path} has no {missing} in its frame table, so it predates "
                    "a column this pass now stores; pass overwrite=True to "
                    "reprocess it rather than resuming into it"
                )
            return cls(h5py.File(path, "r+"), cfg, plan)

        handle = h5py.File(path, "w")
        cls._create_layout(handle, cfg, plan, config_hash)
        return cls(handle, cfg, plan)

    @classmethod
    def _missing_columns(cls, handle: h5py.File) -> list[str]:
        """Frame-table datasets this writer stores that ``handle`` does not have."""
        frames = handle.get("frames")
        wanted = (*cls.FRAME_VECTORS, *cls.FRAME_MATRICES)
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
        for name in cls.EXTRA_GROUPS:
            handle.create_group(name)
        handle.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> FrameTableWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── resume ───────────────────────────────────────────────────────────────
    def block_complete(self, block: Block) -> bool:
        """A block is complete when none of its frames is ``NOT_PROCESSED``."""
        rows = block.rows()
        if rows.size == 0:
            return True
        status = self._f["frames/status"][rows[0] : rows[-1] + 1]
        return bool((status != FrameStatus.NOT_PROCESSED).all())

    # ── writing ──────────────────────────────────────────────────────────────
    def write_block(self, block: Block, result: Any) -> None:
        """Validate the block's labels, then store its rows.

        :raises ValueError: the result's shape does not match the block. Label
            disagreements are not an error — they are recorded as
            ``LABEL_MISMATCH`` by the worker and stored as such — but a result
            of the wrong length is a programming fault.
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
        mislabelled = integrated & (result.train_id != expected_trains)
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
        for name, attribute in self.FRAME_VECTOR_SOURCES.items():
            frames[name][start:stop] = getattr(result, attribute)
        for name in self.FRAME_MATRICES:
            frames[name][start:stop] = getattr(result, name)
        self._f.flush()

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

    # ── finishing ────────────────────────────────────────────────────────────
    def finalise(self, provenance: dict[str, Any]) -> None:
        group = self._f["provenance"]
        for key, value in provenance.items():
            group.attrs[key] = (
                value
                if isinstance(value, (str, int, float))
                else json.dumps(value, default=str)
            )
        self._f.flush()

    # ── inspection ───────────────────────────────────────────────────────────
    def status_summary(self) -> dict[str, int]:
        status = self._f["frames/status"][:]
        return {
            code.name: int((status == code).sum())
            for code in FrameStatus
            if (status == code).any()
        }

    def any_not_ok(self) -> bool:
        return bool((self._f["frames/status"][:] != FrameStatus.OK).any())

    def pooled_per_train(self) -> Any:
        """Per-train pooled ``I(q)`` — see :func:`pooled_per_train`."""
        return pooled_per_train(self._f)


# ── reducers ──────────────────────────────────────────────────────────────────
# These take the output file rather than a live writer, so the same code serves
# the run that produced it and any post hoc analysis of it months later. They
# read the ``/trains`` table for the row-to-train map: ``frames/trainId`` is
# only filled for blocks that were actually written, so a run with an
# unprocessed block has rows carrying trainId 0 (CLAUDE.md pitfall 4).


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
    return result
