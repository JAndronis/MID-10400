"""Per-pixel photon sums per window of trains (context file §15).

For every window of ``cfg.pixel_sum_trains`` trains, and every pixel
``(module, slow, fast)``:

``counts``
    the sum of ``image.data`` over the frames whose ``image.mask`` is 0 for
    that pixel;
``valid_frames``
    how many such frames.

All mask bits count, whatever ``cfg.mask_bits`` says, and nothing else is
applied: no static mask, no seams, no normalisation. That is
:func:`analysis.cache.detector_sums` exactly, and every window equals it over the
window's member trains, bit for bit.

A window is a range of *train ids* from the run's first train, never a count of
trains, so a dropped or failed train cannot shift a later window (CLAUDE.md
pitfall 4). A train is summed whole or not at all; one that is left out is
recorded with the status that left it out.

The kernel and :class:`BlockSums` run in the workers, on the frames the pass has
already decompressed. :class:`PixelSumWriter` runs in the parent, which adds the
workers' partial sums and writes a window once every block touching it is back.
The readers at the bottom need nothing but the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.common.config import restore_by_name, state_by_name
from analysis.common.plan import Block, RunPlan
from analysis.common.status import FrameStatus
from analysis.common.writer import (
    ConfigHashMismatch,
    SchemaMismatch,
    as_handle,
    config_payload,
    status_counts,
)
from analysis.saxs.config import DEFAULT_PIXEL_SUMS_ROOT, N_MODULES

__all__ = [
    "FILE_NAME",
    "HASH_FIELDS",
    "BlockSums",
    "PixelSumWriter",
    "WindowPartial",
    "WindowSpec",
    "accumulate_train",
    "detector_sums",
    "pixel_sums_hash",
    "train_data_ok",
    "window_sums",
    "window_table",
    "windows_for_trains",
]

log = logging.getLogger(__name__)

#: The file's name inside ``{pixel_sums_root}/r{run:04d}/``.
FILE_NAME = "agipd_pixel_sums.h5"

#: The config fields a window's numbers depend on, and so all its hash covers.
#: ``detector_name`` enters *resolved*, so ``None`` (auto-detect) and the name
#: it detects are the same file.
HASH_FIELDS: tuple[str, ...] = (
    "proposal",
    "run",
    "detector_name",
    "min_modules",
    "pixel_sum_trains",
)

#: One window's sums, module × slow-scan × fast-scan.
SUMS_SHAPE: tuple[int, int, int] = (N_MODULES, 512, 128)

#: A train with rows that was not summed for one of these reasons makes the
#: run incomplete. ``MISSING_MODULES`` and ``NO_FRAMES`` trains own no rows and
#: are not failures, exactly as in the frame table.
FAILED = frozenset(
    {
        FrameStatus.LABEL_MISMATCH,
        FrameStatus.DATA_CHECK_FAILED,
        FrameStatus.WORKER_ERROR,
    }
)

DEFINITION = (
    "counts: sum of image.data over the frames with image.mask == 0 for the "
    "pixel; valid_frames: how many such frames. Every mask bit counts; no static "
    "mask, no seams, no normalisation. Window k holds the trains with "
    "window_origin_trainId + k*window_trains <= trainId < "
    "window_origin_trainId + (k+1)*window_trains; a train is summed whole or not "
    "at all, and /trains/status says which."
)

#: Every dataset the file must have; a file missing one predates the schema.
DATASETS: tuple[str, ...] = (
    "windows/start_trainId",
    "windows/n_trains",
    "windows/n_frames",
    "windows/written",
    "windows/counts",
    "windows/valid_frames",
    "trains/trainId",
    "trains/window",
    "trains/status",
)

_INT32_MAX = int(np.iinfo(np.int32).max)


def pixel_sums_hash(cfg: Any, detector_name: str) -> str:
    """sha256 over :data:`HASH_FIELDS`, with the detector name as resolved.

    Geometry, beam centre, photon energy, ``npt`` and every mask are left out:
    none of them changes a stored count (CLAUDE.md pitfall 12).
    """
    payload = {name: getattr(cfg, name) for name in HASH_FIELDS}
    payload["detector_name"] = detector_name
    payload["kind"] = "agipd_pixel_sums"
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ── windows ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class WindowSpec:
    """Window ``k`` holds ``origin + k·length <= trainId < origin + (k+1)·length``."""

    origin: int
    length: int
    count: int

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(f"window length must be positive, got {self.length}")
        if self.count < 1:
            raise ValueError(f"a run has at least one window, got {self.count}")

    def __getstate__(self) -> dict[str, Any]:
        """Pickle by field name — CLAUDE.md pitfall 20."""
        return state_by_name(self)

    def __setstate__(self, state: Any) -> None:
        restore_by_name(self, state)

    @classmethod
    def for_plan(cls, plan: RunPlan, length: int) -> WindowSpec:
        """Windows anchored at the first train of the plan's train table."""
        train_ids = plan.train_ids.astype(np.int64)
        if train_ids.size == 0:
            raise ValueError("a run with no trains has no windows")
        if np.any(np.diff(train_ids) <= 0):
            raise ValueError("the plan's train ids are not strictly increasing")
        origin = int(train_ids[0])
        return cls(origin, int(length), (int(train_ids[-1]) - origin) // length + 1)

    def index_of(self, train_id: int) -> int:
        index = (int(train_id) - self.origin) // self.length
        if not 0 <= index < self.count:
            raise ValueError(f"train {train_id} lies outside every window")
        return index

    def windows_of(self, train_ids: Iterable[int]) -> list[int]:
        """The windows a set of trains touches, ascending."""
        return sorted({self.index_of(t) for t in train_ids})

    def starts(self) -> np.ndarray:
        return self.origin + self.length * np.arange(self.count, dtype=np.uint64)


# ── the per-train kernel (workers) ────────────────────────────────────────────
def train_data_ok(data: np.ndarray) -> bool:
    """Whether a train may be summed: integer photon counts, none negative.

    The same condition the frame loop applies frame by frame
    (``sparse.frame_data_status``), taken over the whole train at once.
    """
    if not np.issubdtype(data.dtype, np.integer):
        return False
    return data.size == 0 or int(data.min()) >= 0


def accumulate_train(
    counts: np.ndarray, valid_frames: np.ndarray, data: np.ndarray, mask: np.ndarray
) -> None:
    """Add one train's frames into ``counts`` and ``valid_frames``, in place.

    :param counts: ``(16, 512, 128)`` int64 accumulator.
    :param valid_frames: same shape and dtype.
    :param data: ``image.data`` of the train, ``(16, n, 512, 128)``, integer.
    :param mask: ``image.mask`` of the train, same shape.

    One module at a time keeps the temporaries to ~20 MB. The per-train sums
    are int32, which a train cannot overflow (352 frames × int16).
    """
    if data.shape != mask.shape or data.ndim != 4 or data.shape[0] != N_MODULES:
        raise ValueError(
            f"data and mask must both be (16, n, 512, 128), got {data.shape} "
            f"and {mask.shape}"
        )
    if counts.shape != SUMS_SHAPE or valid_frames.shape != SUMS_SHAPE:
        raise ValueError(f"accumulators must be {SUMS_SHAPE}")
    for module in range(N_MODULES):
        valid = mask[module] == 0
        counts[module] += np.where(valid, data[module], 0).sum(axis=0, dtype=np.int32)
        valid_frames[module] += valid.sum(axis=0, dtype=np.int32)


@dataclass(slots=True)
class WindowPartial:
    """One block's contribution to one window."""

    counts: np.ndarray
    valid_frames: np.ndarray
    train_ids: list[int] = field(default_factory=list)
    n_frames: int = 0


class BlockSums:
    """A worker's partial window sums for one scheduling block."""

    def __init__(self, windows: WindowSpec) -> None:
        self.windows = windows
        self.partials: dict[int, WindowPartial] = {}
        self.excluded: dict[int, int] = {}

    def add_train(self, train_id: int, data: np.ndarray, mask: np.ndarray) -> None:
        """Sum one train whose labels have been checked, or record why not."""
        if not train_data_ok(data):
            self.exclude(train_id, FrameStatus.DATA_CHECK_FAILED)
            return
        index = self.windows.index_of(train_id)
        partial = self.partials.get(index)
        if partial is None:
            partial = WindowPartial(
                np.zeros(SUMS_SHAPE, np.int64), np.zeros(SUMS_SHAPE, np.int64)
            )
            self.partials[index] = partial
        accumulate_train(partial.counts, partial.valid_frames, data, mask)
        partial.train_ids.append(int(train_id))
        partial.n_frames += int(data.shape[1])

    def exclude(self, train_id: int, status: FrameStatus) -> None:
        self.excluded[int(train_id)] = int(status)

    def packed(self) -> dict[int, WindowPartial]:
        """The partials as int32, half the bytes to send back to the parent."""
        return {
            index: WindowPartial(
                _as_int32(p.counts, "counts"),
                _as_int32(p.valid_frames, "valid_frames"),
                list(p.train_ids),
                p.n_frames,
            )
            for index, p in self.partials.items()
        }


def _as_int32(values: np.ndarray, name: str) -> np.ndarray:
    """Cast to int32, refusing rather than wrapping a value out of range."""
    if values.size and (int(values.min()) < 0 or int(values.max()) > _INT32_MAX):
        raise OverflowError(
            f"{name} spans [{int(values.min())}, {int(values.max())}], outside int32"
        )
    return values.astype(np.int32)


# ── the file (parent) ─────────────────────────────────────────────────────────
@dataclass(slots=True)
class _Accumulator:
    counts: np.ndarray = field(default_factory=lambda: np.zeros(SUMS_SHAPE, np.int64))
    valid_frames: np.ndarray = field(
        default_factory=lambda: np.zeros(SUMS_SHAPE, np.int64)
    )
    n_frames: int = 0


def _initial_status(plan: RunPlan) -> np.ndarray:
    """A train with rows starts ``NOT_PROCESSED``; one without keeps its status."""
    return np.array(
        [
            int(FrameStatus.NOT_PROCESSED if t.status is FrameStatus.OK else t.status)
            for t in plan.trains
        ],
        dtype=np.uint8,
    )


class PixelSumWriter:
    """The single writer of one run's window-sums file.

    Workers send partial sums per scheduling block; this adds them and writes a
    window once the last block touching it has returned or failed. A written
    window is final, except one that lost trains to ``WORKER_ERROR``: such an
    error says nothing about the data, so opening the file again reopens that
    window and the run rebuilds it from scratch. The file is never truncated:
    ``cfg.overwrite`` applies to the frame table only (context file §15.3).
    """

    def __init__(self, handle: h5py.File, windows: WindowSpec) -> None:
        self._f = handle
        self.windows = windows
        self._window_of_train = handle["trains/window"][:]
        self._train_index = {
            int(t): i for i, t in enumerate(handle["trains/trainId"][:].tolist())
        }
        self._status = handle["trains/status"][:]
        self._written = handle["windows/written"][:].astype(bool)
        self._pending: dict[int, set[int]] = {}
        self._sums: dict[int, _Accumulator] = {}
        self._block_errors: dict[str, str] = {}
        self._reopen_lost_windows()

    def _reopen_lost_windows(self) -> None:
        """Mark every window holding a ``WORKER_ERROR`` train for a rebuild.

        Only memory changes: the file keeps the old window, consistent with its
        own train statuses, until the rebuilt one replaces it — so a run that
        dies part-way leaves it exactly as it found it.
        """
        lost = np.unique(
            self._window_of_train[self._status == FrameStatus.WORKER_ERROR]
        )
        if lost.size == 0:
            return
        self._written[lost] = False
        reopened = np.isin(self._window_of_train, lost) & np.isin(
            self._status, [FrameStatus.OK, FrameStatus.WORKER_ERROR]
        )
        self._status[reopened] = FrameStatus.NOT_PROCESSED
        log.info("rebuilding windows %s, which lost trains to a worker error", lost)

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def open_or_create(
        cls, cfg: Any, plan: RunPlan, windows: WindowSpec, path: Path
    ) -> PixelSumWriter:
        """Open the run's window-sums file, creating it if needed.

        :raises ConfigHashMismatch: the file was written under another hash —
            even with ``cfg.overwrite``. Remove it by hand if that is meant.
        :raises SchemaMismatch: the file lacks a dataset this writer stores.
        :raises ValueError: the run's train table or windows have changed.
        """
        path = Path(path)
        config_hash = pixel_sums_hash(cfg, plan.detector_name)
        if path.exists():
            with h5py.File(path, "r") as existing:
                stored = existing["provenance"].attrs.get("config_hash")
                missing = [name for name in DATASETS if name not in existing]
                stored_ids = (
                    existing["trains/trainId"][:] if not missing else np.zeros(0)
                )
                stored_status = (
                    existing["trains/status"][:] if not missing else np.zeros(0)
                )
                stored_windows = (
                    int(existing["provenance"].attrs.get("window_origin_trainId", -1)),
                    int(existing["provenance"].attrs.get("window_trains", -1)),
                    int(existing["provenance"].attrs.get("n_windows", -1)),
                )
            if stored != config_hash:
                raise ConfigHashMismatch(
                    f"{path} holds window sums under hash {stored}, this run has "
                    f"{config_hash}. The pass never replaces window sums, not even "
                    "with overwrite=True: once proc is on tape they cannot be "
                    "rebuilt. Move or remove the file by hand if that is meant"
                )
            if missing:
                raise SchemaMismatch(f"{path} has no {missing}")
            cls._check_same_run(path, plan, windows, stored_ids, stored_status)
            if stored_windows != (windows.origin, windows.length, windows.count):
                raise ValueError(
                    f"{path} was written with windows {stored_windows} "
                    "(origin, length, count); this run's plan gives "
                    f"{(windows.origin, windows.length, windows.count)}"
                )
            return cls(h5py.File(path, "r+"), windows)

        path.parent.mkdir(parents=True, exist_ok=True)
        handle = h5py.File(path, "w")
        cls._create_layout(handle, cfg, plan, windows, config_hash)
        return cls(handle, windows)

    @staticmethod
    def _check_same_run(
        path: Path,
        plan: RunPlan,
        windows: WindowSpec,
        stored_ids: np.ndarray,
        stored_status: np.ndarray,
    ) -> None:
        """Refuse to resume into a file built from a different train table."""
        expected = _initial_status(plan)
        rowless = expected != FrameStatus.NOT_PROCESSED
        if (
            not np.array_equal(stored_ids, plan.train_ids)
            or not np.array_equal(stored_status[rowless], expected[rowless])
            or np.isin(
                stored_status[~rowless],
                [FrameStatus.MISSING_MODULES, FrameStatus.NO_FRAMES],
            ).any()
        ):
            raise ValueError(
                f"{path} was written from a different train table than this run's "
                "plan; the proc data must have changed. Move it aside by hand"
            )

    @staticmethod
    def _create_layout(
        handle: h5py.File,
        cfg: Any,
        plan: RunPlan,
        windows: WindowSpec,
        config_hash: str,
    ) -> None:
        n_windows = windows.count
        group = handle.create_group("windows")
        group["start_trainId"] = windows.starts()
        group.create_dataset("n_trains", shape=(n_windows,), dtype=np.uint32)
        group.create_dataset("n_frames", shape=(n_windows,), dtype=np.uint32)
        group.create_dataset("written", shape=(n_windows,), dtype=np.uint8)
        for name in ("counts", "valid_frames"):
            group.create_dataset(
                name,
                shape=(n_windows, *SUMS_SHAPE),
                dtype=np.int32,
                fillvalue=0,
                chunks=(1, 1, *SUMS_SHAPE[1:]),
                shuffle=True,
                compression="gzip",
                compression_opts=1,
            )

        trains = handle.create_group("trains")
        trains["trainId"] = plan.train_ids
        trains["window"] = np.array(
            [windows.index_of(t.train_id) for t in plan.trains], dtype=np.int32
        )
        trains["status"] = _initial_status(plan)

        provenance = handle.create_group("provenance")
        provenance.attrs["config_hash"] = config_hash
        provenance.attrs["config"] = json.dumps(
            config_payload(cfg), sort_keys=True, default=str
        )
        provenance.attrs["hash_fields"] = json.dumps(list(HASH_FIELDS))
        provenance.attrs["detector_name"] = plan.detector_name
        provenance.attrs["proposal"] = int(cfg.proposal)
        provenance.attrs["run"] = int(cfg.run)
        provenance.attrs["window_trains"] = windows.length
        provenance.attrs["window_origin_trainId"] = windows.origin
        provenance.attrs["n_windows"] = n_windows
        provenance.attrs["definition"] = DEFINITION
        handle.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> PixelSumWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── what is left to do ───────────────────────────────────────────────────
    def window_written(self, index: int) -> bool:
        return bool(self._written[index])

    def blocks_to_do(self, blocks: Sequence[Block]) -> list[Block]:
        """The blocks touching a window not yet written."""
        return [
            block
            for block in blocks
            if any(
                not self._written[w] for w in self.windows.windows_of(block.train_ids)
            )
        ]

    def begin(self, todo: Sequence[Block]) -> None:
        """Register which blocks each unwritten window waits for.

        An unwritten window no block touches holds only trains without rows,
        so it is final already and written at once.
        """
        for block in todo:
            for index in self.windows.windows_of(block.train_ids):
                if not self._written[index]:
                    self._pending.setdefault(index, set()).add(block.index)
        for index in range(self.windows.count):
            if not self._written[index] and index not in self._pending:
                self._write_window(index)

    # ── writing ──────────────────────────────────────────────────────────────
    def write_block(self, block: Block, result: Any) -> None:
        """Add one block's partial sums, and write every window it completes.

        :raises ValueError: the result carries a train that is not the block's,
            or files it under the wrong window — a programming fault, never a
            data condition.
        """
        trains = set(block.train_ids)
        for index, partial in result.pixel_partials.items():
            if self._written[index]:
                continue  # final from an earlier attempt; resume ignores it
            if block.index not in self._pending.get(index, ()):
                raise ValueError(
                    f"block {block.index} returned sums for window {index}, "
                    "which was not waiting for it"
                )
            for train_id in partial.train_ids:
                if train_id not in trains or self.windows.index_of(train_id) != index:
                    raise ValueError(
                        f"block {block.index} filed train {train_id} under window "
                        f"{index}"
                    )
                self._set_status(train_id, FrameStatus.OK)
            total = self._sums.setdefault(index, _Accumulator())
            total.counts += partial.counts
            total.valid_frames += partial.valid_frames
            total.n_frames += partial.n_frames
        for train_id, status in result.pixel_excluded.items():
            if train_id not in trains:
                raise ValueError(
                    f"block {block.index} left out train {train_id}, not one of its own"
                )
            if not self._written[self.windows.index_of(train_id)]:
                self._set_status(train_id, FrameStatus(status))
        self._settle(block)

    def fail_block(self, block: Block, status: FrameStatus, message: str = "") -> None:
        """Record every train of a block that never came back as left out."""
        for train_id in block.train_ids:
            if not self._written[self.windows.index_of(train_id)]:
                self._set_status(train_id, status)
        if message:
            self._block_errors[str(block.index)] = message
        self._settle(block)

    def _set_status(self, train_id: int, status: FrameStatus) -> None:
        self._status[self._train_index[int(train_id)]] = int(status)

    def _settle(self, block: Block) -> None:
        for index in self.windows.windows_of(block.train_ids):
            waiting = self._pending.get(index)
            if waiting is None:
                continue
            waiting.discard(block.index)
            if not waiting:
                del self._pending[index]
                self._write_window(index)

    def _write_window(self, index: int) -> None:
        members = np.flatnonzero(self._window_of_train == index)
        status = self._status[members]
        if (status == FrameStatus.NOT_PROCESSED).any():
            unsettled = members[status == FrameStatus.NOT_PROCESSED]
            raise RuntimeError(
                f"window {index} is complete but {unsettled.size} of its trains "
                "were neither summed nor left out"
            )
        total = self._sums.pop(index, None) or _Accumulator()
        group = self._f["windows"]
        group["counts"][index] = _as_int32(total.counts, "counts")
        group["valid_frames"][index] = _as_int32(total.valid_frames, "valid_frames")
        group["n_trains"][index] = int((status == FrameStatus.OK).sum())
        group["n_frames"][index] = total.n_frames
        group["written"][index] = 1
        if members.size:
            # This window's rows only: another window may be mid-rebuild, and
            # its rows on disk must keep matching its sums on disk.
            rows = slice(int(members[0]), int(members[-1]) + 1)
            self._f["trains/status"][rows] = self._status[rows]
        self._written[index] = True
        self._f.flush()

    # ── finishing ────────────────────────────────────────────────────────────
    def status_summary(self) -> dict[str, int]:
        return status_counts(self._status, nonzero_only=True)

    def incomplete(self) -> bool:
        """A train with rows was left out, or a window is still unwritten."""
        failed = np.isin(self._status, [int(code) for code in FAILED]).any()
        return bool(failed or not self._written.all())

    def finalise(self, provenance: dict[str, Any]) -> None:
        group = self._f["provenance"]
        record = {
            **provenance,
            "status_summary": self.status_summary(),
            "windows_written": int(self._written.sum()),
        }
        if self._block_errors:
            record["block_errors"] = self._block_errors
        for key, value in record.items():
            group.attrs[key] = (
                value
                if isinstance(value, (str, int, float))
                else json.dumps(value, default=str)
            )
        self._f.flush()


# ── readers ───────────────────────────────────────────────────────────────────
def _path_for(run_nr: int, root: str | Path) -> Path:
    return Path(root) / f"r{int(run_nr):04d}" / FILE_NAME


def window_table(source: Any) -> Any:
    """One row per window: where its range starts, what it holds, whether final.

    :param source: an open file or a path.
    """
    import xarray as xr

    with as_handle(source) as handle:
        group = handle["windows"]
        table = {
            name: ("window", group[name][:])
            for name in ("start_trainId", "n_trains", "n_frames")
        }
        table["written"] = ("window", group["written"][:].astype(bool))
        attrs = {
            name: handle["provenance"].attrs[name]
            for name in ("run", "window_trains", "window_origin_trainId")
        }
    n_windows = table["written"][1].size
    return xr.Dataset(table, coords={"window": np.arange(n_windows)}, attrs=attrs)


def windows_for_trains(source: Any, train_ids: Iterable[int]) -> list[int]:
    """The windows whose member trains are exactly ``train_ids``.

    :raises ValueError: a train is not in the run, was not summed, or the
        trains are not a union of whole windows — the message names the windows
        that cover them, so a caller can choose those instead.
    """
    requested = sorted({int(t) for t in train_ids})
    if not requested:
        raise ValueError("no train ids given")
    with as_handle(source) as handle:
        table = handle["trains/trainId"][:]
        window = handle["trains/window"][:]
        status = handle["trains/status"][:]
        written = handle["windows/written"][:].astype(bool)
        starts = handle["windows/start_trainId"][:]
        length = int(handle["provenance"].attrs["window_trains"])

    position = np.searchsorted(table, np.asarray(requested, dtype=table.dtype))
    position = np.clip(position, 0, table.size - 1)
    unknown = [t for t, p in zip(requested, position, strict=True) if table[p] != t]
    if unknown:
        raise ValueError(f"{len(unknown)} trains are not in this run: {unknown[:10]}")

    chosen = sorted({int(w) for w in window[position]})
    unwritten = [w for w in chosen if not written[w]]
    if unwritten:
        raise ValueError(f"windows {unwritten} are not written yet")
    left_out = {
        int(table[p]): FrameStatus(int(status[p])).name
        for p in position
        if status[p] != FrameStatus.OK
    }
    if left_out:
        raise ValueError(
            f"{len(left_out)} requested trains were not summed: "
            f"{dict(list(left_out.items())[:10])}"
        )
    members = table[np.isin(window, chosen) & (status == FrameStatus.OK)]
    if members.size != len(requested):
        ranges = ", ".join(
            f"{w} [{int(starts[w])}, {int(starts[w]) + length})" for w in chosen
        )
        raise ValueError(
            f"the {len(requested)} trains are not a union of whole windows: the "
            f"windows that cover them ({ranges}) hold {members.size} summed trains. "
            "Ask for those windows, or for exactly their trains"
        )
    return chosen


def window_sums(source: Any, windows: Iterable[int]) -> Any:
    """The per-pixel sums over ``windows``, as :func:`analysis.cache.detector_sums`.

    Returns a Dataset over (module, slow, fast) with ``counts`` and
    ``valid_frames`` (int32), the member trains as ``train_id``, and ``run``
    and ``n_frames`` attributes, plus which windows and which file.
    """
    import xarray as xr

    chosen = sorted({int(w) for w in windows})
    if not chosen:
        raise ValueError("no windows given")
    counts = np.zeros(SUMS_SHAPE, np.int64)
    valid = np.zeros(SUMS_SHAPE, np.int64)
    with as_handle(source) as handle:
        group = handle["windows"]
        written = group["written"][:].astype(bool)
        outside = [w for w in chosen if not 0 <= w < written.size]
        if outside:
            raise ValueError(
                f"windows {outside} do not exist; there are {written.size}"
            )
        unwritten = [w for w in chosen if not written[w]]
        if unwritten:
            raise ValueError(f"windows {unwritten} are not written yet")
        for index in chosen:
            counts += group["counts"][index]
            valid += group["valid_frames"][index]
        n_frames = int(group["n_frames"][chosen].sum())
        trains = handle["trains"]
        members = trains["trainId"][:][
            np.isin(trains["window"][:], chosen)
            & (trains["status"][:] == FrameStatus.OK)
        ]
        provenance = handle["provenance"].attrs
        attrs = {
            "run": int(provenance["run"]),
            "n_frames": n_frames,
            "windows": np.asarray(chosen, dtype=np.int64),
            "window_trains": int(provenance["window_trains"]),
            "config_hash": str(provenance["config_hash"]),
            "source": str(handle.filename),
            "description": "per-pixel photon sums over whole windows; see "
            "analysis.saxs.pixel_sums",
        }

    dims = ("module", "slow", "fast")
    result = xr.Dataset(
        {
            "counts": (dims, _as_int32(counts, "counts")),
            "valid_frames": (dims, _as_int32(valid, "valid_frames")),
            "train_id": ("train", np.asarray(members, np.uint64)),
        },
        coords={"module": np.arange(N_MODULES)},
        attrs=attrs,
    )
    result["counts"].attrs["description"] = (
        "sum of image.data over the frames with image.mask == 0 for the pixel"
    )
    return result


def detector_sums(
    run_nr: int,
    train_ids: Iterable[int] | None = None,
    *,
    windows: Iterable[int] | None = None,
    root: str | Path = DEFAULT_PIXEL_SUMS_ROOT,
    path: str | Path | None = None,
) -> Any:
    """Per-pixel photon sums from the pass's window sums, no proc read.

    The drop-in for :func:`analysis.cache.detector_sums`, returning the same
    Dataset. Give exactly one of:

    :param train_ids: trains that must be exactly a union of whole windows.
    :param windows: window indices — see :func:`window_table`.
    :param root: ``cfg.pixel_sums_root`` the pass wrote under.
    :param path: the file itself, instead of ``root`` and ``run_nr``.
    """
    if (train_ids is None) == (windows is None):
        raise ValueError("give exactly one of train_ids and windows")
    source = Path(path) if path is not None else _path_for(run_nr, root)
    if train_ids is not None:
        windows = windows_for_trains(source, train_ids)
    return window_sums(source, windows)
