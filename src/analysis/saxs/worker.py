"""Per-block read and frame loop.

One single-threaded worker per core, spawned. The worker owns nothing but its
own reads: the operator and the masks come from files written by the parent,
and the results go back as plain arrays. Only the parent writes the output files.

The same read feeds the per-pixel window sums (context file §15) when the
parent asks for them, so they cost an add per train rather than a second read.
A block the parent needs only for its windows skips the integration.

Thread limits are inherited from the parent's environment — a
spawned child has already imported numpy by the time an initializer runs, so
setting them here would be too late for pools that already exist. They are
re-asserted anyway, which helps only libraries imported lazily.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common.cpu import set_thread_env
from analysis.common.masks import frame_bad
from analysis.common.plan import Block
from analysis.common.status import DataCheckFailed, FrameStatus
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.masks import BaseMasks, load_masks
from analysis.saxs.operator import SparseOperator, load_operator
from analysis.saxs.pixel_sums import BlockSums, WindowSpec
from analysis.saxs.sparse import integrate_frame

__all__ = [
    "BlockResult",
    "WorkerPaths",
    "init",
    "process_block",
]


@dataclass(frozen=True, slots=True)
class WorkerPaths:
    """What the parent wrote for the workers to load.

    ``run_dir`` opens a run by directory instead of by proposal and run number,
    which is how the mock run is reached; ``None`` uses ``open_run``.
    ``operator`` and ``masks`` are ``None`` for a pass that only sums pixels,
    which integrates nothing.
    """

    operator: str | None
    masks: str | None
    run_dir: str | None = None


@dataclass(slots=True)
class _WorkerState:
    cfg: AgipdSaxsConfig
    op: SparseOperator | None
    masks: BaseMasks | None
    detector: Any
    windows: WindowSpec | None = None
    skip_integration: frozenset[int] = frozenset()


_STATE: _WorkerState | None = None


def init(
    paths: WorkerPaths,
    cfg: AgipdSaxsConfig,
    windows: WindowSpec | None = None,
    skip_integration: frozenset[int] = frozenset(),
) -> None:
    """Process initializer: load the operator and masks, open the run.

    :param windows: sum pixels into these windows; ``None`` sums nothing.
    :param skip_integration: blocks to read and sum without integrating —
        those whose frames the output file already holds.
    """
    set_thread_env()

    from extra_data import RunDirectory, open_run
    from extra_data.components import AGIPD1M

    op = load_operator(Path(paths.operator)) if paths.operator else None
    masks = load_masks(Path(paths.masks)) if paths.masks else None
    if (op is None) != (masks is None):
        raise ValueError("the operator and the masks come together or not at all")
    if op is not None and masks.operator_sha256 != op.sha256:
        raise ValueError(
            f"masks were built against operator {masks.operator_sha256}, "
            f"but the loaded operator is {op.sha256}"
        )

    dc = (
        RunDirectory(paths.run_dir)
        if paths.run_dir
        else open_run(cfg.proposal, cfg.run, data="proc")
    )
    detector = AGIPD1M(dc, detector_name=cfg.detector_name, min_modules=cfg.min_modules)
    global _STATE
    _STATE = _WorkerState(
        cfg=cfg,
        op=op,
        masks=masks,
        detector=detector,
        windows=windows,
        skip_integration=frozenset(skip_integration),
    )


def init_from_detector(
    cfg: AgipdSaxsConfig,
    op: SparseOperator | None,
    masks: BaseMasks | None,
    detector: Any,
    windows: WindowSpec | None = None,
    skip_integration: frozenset[int] = frozenset(),
) -> None:
    """In-process initializer for tests and single-process runs."""
    global _STATE
    _STATE = _WorkerState(
        cfg=cfg,
        op=op,
        masks=masks,
        detector=detector,
        windows=windows,
        skip_integration=frozenset(skip_integration),
    )


@dataclass(slots=True)
class BlockResult:
    """One block's frames, addressed by label rather than by position."""

    block_index: int
    train_id: np.ndarray  # (n,) uint64
    pulse_id: np.ndarray  # (n,) uint64
    cell_id: np.ndarray  # (n,) uint16
    status: np.ndarray  # (n,) uint8
    signal: np.ndarray  # (n, npt) float32
    normalization: np.ndarray  # (n, npt) float32
    variance: np.ndarray  # (n, npt) float32
    photons_valid: np.ndarray  # (n,) float32
    n_bad_pixels: np.ndarray  # (n,) uint32
    n_frame_specific: np.ndarray  # (n,) uint32
    max_count: np.ndarray  # (n,) uint16
    bits_present: int = 0
    unseen_cells: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    #: False when the block was read only for its window sums: its frame rows
    #: are then still ``NOT_PROCESSED`` zeros and must not be written.
    integrated: bool = True
    #: Window index → this block's partial sums for it (``pixel_sums``).
    pixel_partials: dict[int, Any] = field(default_factory=dict)
    #: Train id → the ``FrameStatus`` that kept it out of the window sums.
    pixel_excluded: dict[int, int] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.train_id.size)


def _empty_result(block: Block, npt: int) -> BlockResult:
    n = block.n_frames
    return BlockResult(
        block_index=block.index,
        train_id=np.zeros(n, dtype=np.uint64),
        pulse_id=np.zeros(n, dtype=np.uint64),
        cell_id=np.zeros(n, dtype=np.uint16),
        status=np.full(n, FrameStatus.NOT_PROCESSED, dtype=np.uint8),
        signal=np.zeros((n, npt), dtype=np.float32),
        normalization=np.zeros((n, npt), dtype=np.float32),
        variance=np.zeros((n, npt), dtype=np.float32),
        photons_valid=np.zeros(n, dtype=np.float32),
        n_bad_pixels=np.zeros(n, dtype=np.uint32),
        n_frame_specific=np.zeros(n, dtype=np.uint32),
        max_count=np.zeros(n, dtype=np.uint16),
    )


def process_block(block: Block) -> BlockResult:
    """Read and integrate one block, and sum its pixels if asked to.

    Every frame's identity comes from the reader's coordinates, never from its
    position in the array. A train whose labels or frame
    count disagree with the plan is marked ``LABEL_MISMATCH`` and none of its
    frames are integrated or summed.
    """
    if _STATE is None:
        raise RuntimeError("worker.init has not run in this process")
    state = _STATE
    op, cfg, masks = state.op, state.cfg, state.masks
    integrate = block.index not in state.skip_integration
    if integrate and (op is None or masks is None):
        raise RuntimeError(
            f"block {block.index} is to be integrated, but this worker was "
            "initialised without an operator"
        )
    sums = BlockSums(state.windows) if state.windows is not None else None

    from extra_data import by_id

    result = _empty_result(block, op.npt if op is not None else 0)
    result.integrated = integrate
    timings = {"read_data": 0.0, "read_mask": 0.0}
    if integrate:
        timings["integrate"] = 0.0
    if sums is not None:
        timings["pixel_sums"] = 0.0
    bits_present = 0
    unseen_cells = 0
    offset = 0

    for train_id, expected in zip(block.train_ids, block.expected_frames, strict=True):
        rows = slice(offset, offset + expected)
        offset += expected

        selected = state.detector.select_trains(by_id[[train_id]])
        counts_key = selected["image.data"]
        mask_key = selected["image.mask"]

        started = time.perf_counter()
        data = counts_key.ndarray(decompress_threads=1)
        timings["read_data"] += time.perf_counter() - started

        started = time.perf_counter()
        mask = mask_key.ndarray(decompress_threads=1)
        timings["read_mask"] += time.perf_counter() - started

        train_ids = counts_key.train_id_coordinates()
        pulse_ids = counts_key.pulse_id_coordinates()
        cell_ids = counts_key.cell_id_coordinates()

        if (
            train_ids.size != expected
            or data.shape[1] != expected
            or mask.shape[1] != expected
            or not np.all(train_ids == train_id)
        ):
            result.status[rows] = FrameStatus.LABEL_MISMATCH
            if sums is not None:
                sums.exclude(train_id, FrameStatus.LABEL_MISMATCH)
            continue

        result.train_id[rows] = train_ids
        result.pulse_id[rows] = pulse_ids
        result.cell_id[rows] = cell_ids
        bits_present |= int(np.bitwise_or.reduce(mask, axis=None))

        if sums is not None:
            started = time.perf_counter()
            sums.add_train(train_id, data, mask)
            timings["pixel_sums"] += time.perf_counter() - started
        if not integrate:
            continue

        started = time.perf_counter()
        for frame in range(expected):
            row = rows.start + frame
            cell = int(cell_ids[frame])
            if not masks.has_cell(cell):
                unseen_cells += 1
            base_bad, base_denominator = masks.for_cell(cell)
            counts = data[:, frame].reshape(-1)
            bad = frame_bad(mask[:, frame], cfg.mask_bits, masks.static_bad)
            try:
                frame_result = integrate_frame(
                    op, counts, bad, base_bad, base_denominator
                )
            except DataCheckFailed:
                result.status[row] = FrameStatus.DATA_CHECK_FAILED
                continue
            result.signal[row] = frame_result.signal
            result.normalization[row] = frame_result.normalization
            result.variance[row] = frame_result.variance
            result.photons_valid[row] = frame_result.photons_valid
            result.n_bad_pixels[row] = frame_result.n_bad_pixels
            result.n_frame_specific[row] = frame_result.n_frame_specific
            result.max_count[row] = frame_result.max_count
            result.status[row] = FrameStatus.OK
        timings["integrate"] += time.perf_counter() - started

    result.bits_present = bits_present
    result.unseen_cells = unseen_cells
    result.timings = timings
    if sums is not None:
        result.pixel_partials = sums.packed()
        result.pixel_excluded = sums.excluded
    return result
