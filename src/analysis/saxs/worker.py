"""Per-block read and frame loop.

One single-threaded worker per core, spawned. The worker owns nothing but its
own reads: the operator and the masks come from files written by the parent,
and the results go back as plain arrays. Only the parent writes the output files.

The same read feeds the per-pixel window sums and the per-frame photon
statistics (``analysis.saxs.photon_stats``) when the parent asks for them, so
they cost an add per train and a histogram per frame rather than a second read.
A block the parent needs only for those skips the integration.

Thread limits are inherited from the parent's environment. A spawned child has
already imported numpy by the time an initializer runs, so setting them here
would be too late for pools that already exist. They are re-asserted anyway,
which helps only libraries imported lazily.
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
from analysis.common.writer import FrameRows
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.masks import BaseMasks, load_masks
from analysis.saxs.operator import SparseOperator, load_operator
from analysis.saxs.photon_stats import PhotonSpec, PhotonStatsWriter, histogram_frame
from analysis.saxs.pixel_sums import BlockSums, WindowSpec
from analysis.saxs.plan import open_detector
from analysis.saxs.sparse import frame_data_status, integrate_frame
from analysis.saxs.writer import AgipdSaxsWriter

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
    photon: PhotonSpec | None = None
    #: ``(ring_n, in_ring)`` of ``photon``, built once.
    rings: tuple[np.ndarray, np.ndarray] | None = None


_STATE: _WorkerState | None = None


def init(
    paths: WorkerPaths,
    cfg: AgipdSaxsConfig,
    windows: WindowSpec | None = None,
    skip_integration: frozenset[int] = frozenset(),
    photon: PhotonSpec | None = None,
) -> None:
    """Process initializer: load the operator and masks, open the run.

    :param windows: sum pixels into these windows; ``None`` sums nothing.
    :param skip_integration: blocks to read and sum without integrating, because
        the output file already holds their frames.
    :param photon: histogram the frames of ``photon.blocks``; ``None`` none.
    """
    set_thread_env()

    from extra_data import RunDirectory, open_run

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
    detector = open_detector(cfg, dc)
    global _STATE
    _STATE = _WorkerState(
        cfg=cfg,
        op=op,
        masks=masks,
        detector=detector,
        windows=windows,
        skip_integration=frozenset(skip_integration),
        photon=photon,
        rings=photon.ring_counts() if photon is not None else None,
    )


def init_from_detector(
    cfg: AgipdSaxsConfig,
    op: SparseOperator | None,
    masks: BaseMasks | None,
    detector: Any,
    windows: WindowSpec | None = None,
    skip_integration: frozenset[int] = frozenset(),
    photon: PhotonSpec | None = None,
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
        photon=photon,
        rings=photon.ring_counts() if photon is not None else None,
    )


@dataclass(slots=True)
class BlockResult:
    """One block's results for each product, rows addressed by label."""

    #: The frame-table rows, in ``AgipdSaxsWriter``'s schema; ``None`` when
    #: the block was read only for its window sums or statistics.
    frames: FrameRows | None = None
    #: The photon-statistics rows, when the parent asked for them.
    photon: FrameRows | None = None
    #: Window index → this block's partial sums for it (``pixel_sums``).
    pixel_partials: dict[int, Any] = field(default_factory=dict)
    #: Train id → the ``FrameStatus`` that kept it out of the window sums.
    pixel_excluded: dict[int, int] = field(default_factory=dict)
    bits_present: int = 0
    unseen_cells: int = 0
    timings: dict[str, float] = field(default_factory=dict)


def process_block(block: Block) -> BlockResult:
    """Read one block once, and feed it to each product the parent asked for.

    Every frame's identity comes from the reader's coordinates, never from its
    position in the array. A train whose frame count disagrees with the plan,
    or whose ``image.trainId`` rows in any module name another train, is marked
    ``LABEL_MISMATCH`` and none of its frames are integrated, summed or
    histogrammed. One data check per frame sets the status of both the frame
    table and the statistics, and the histogram sees the integration's bad
    pixels, so a frame has one status in both files.
    """
    if _STATE is None:
        raise RuntimeError("worker.init has not run in this process")
    state = _STATE
    op, cfg, masks, spec = state.op, state.cfg, state.masks, state.photon
    integrate = block.index not in state.skip_integration
    if integrate and (op is None or masks is None):
        raise RuntimeError(
            f"block {block.index} is to be integrated, but this worker was "
            "initialised without an operator"
        )
    n = block.n_frames
    result = BlockResult(
        frames=AgipdSaxsWriter.empty_rows(n, op.npt) if integrate else None,
        photon=(
            PhotonStatsWriter.photon_rows(n, spec.n_rings, spec.k_max + 2)
            if spec is not None and block.index in spec.blocks
            else None
        ),
    )
    tables = [t for t in (result.frames, result.photon) if t is not None]
    sums = BlockSums(state.windows) if state.windows is not None else None
    if masks is not None:
        static_bad = masks.static_bad
    elif spec is not None:
        static_bad = spec.static_bad

    from extra_data import by_id

    timings = {"read_data": 0.0, "read_mask": 0.0}
    if integrate:
        timings["integrate"] = 0.0
    if sums is not None:
        timings["pixel_sums"] = 0.0
    if result.photon is not None:
        timings["photon_stats"] = 0.0
    # The mask's time counts towards the integration when it runs, else
    # towards the histogram.
    frame_stage = "integrate" if integrate else "photon_stats"
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
        # The reader's train coordinate is the INDEX's, which placed these rows
        # under ``train_id`` by construction. Only every module's own
        # ``image.trainId`` rows can say the rows belong to another train.
        row_labels = selected["image.trainId"].ndarray(decompress_threads=1)

        if (
            train_ids.size != expected
            or data.shape[1] != expected
            or mask.shape[1] != expected
            or not np.all(row_labels == train_id)
        ):
            for table in tables:
                table.status[rows] = FrameStatus.LABEL_MISMATCH
            if sums is not None:
                sums.exclude(train_id, FrameStatus.LABEL_MISMATCH)
            continue

        cell_ids = counts_key.cell_id_coordinates()
        labels = {
            "trainId": train_ids,
            "reader_pulseId": counts_key.pulse_id_coordinates(),
            "cellId": cell_ids,
        }
        for table in tables:
            table.put(rows, labels)
        result.bits_present |= int(np.bitwise_or.reduce(mask, axis=None))

        if sums is not None:
            started = time.perf_counter()
            sums.add_train(train_id, data, mask)
            timings["pixel_sums"] += time.perf_counter() - started

        for frame in range(expected if tables else 0):
            row = rows.start + frame
            counts = data[:, frame].reshape(-1)
            started = time.perf_counter()
            bad = frame_bad(mask[:, frame], cfg.mask_bits, static_bad)
            if integrate:
                status = _integrate(
                    state, result, row, counts, bad, int(cell_ids[frame])
                )
                timings["integrate"] += time.perf_counter() - started
                started = time.perf_counter()
            else:
                status = frame_data_status(counts)
            if status is not FrameStatus.OK:
                for table in tables:
                    table.status[row] = status
                timings[frame_stage] += time.perf_counter() - started
                continue
            if result.photon is not None:
                histogram_frame(
                    result.photon,
                    row,
                    counts,
                    bad,
                    spec.labels,
                    *state.rings,
                    spec.k_max,
                )
                timings["photon_stats"] += time.perf_counter() - started

    result.timings = timings
    if result.photon is not None:
        result.photon.bits_present = result.bits_present
    if sums is not None:
        result.pixel_partials = sums.packed()
        result.pixel_excluded = sums.excluded
    return result


def _integrate(
    state: _WorkerState,
    result: BlockResult,
    row: int,
    counts: np.ndarray,
    bad: np.ndarray,
    cell: int,
) -> FrameStatus:
    """Integrate one frame into its frame-table row; its status either way."""
    masks = state.masks
    if not masks.has_cell(cell):
        result.unseen_cells += 1
    base_bad, base_denominator = masks.for_cell(cell)
    try:
        frame_result = integrate_frame(
            state.op, counts, bad, base_bad, base_denominator
        )
    except DataCheckFailed:
        return FrameStatus.DATA_CHECK_FAILED
    result.frames.put(row, frame_result.as_row())
    result.frames.status[row] = FrameStatus.OK
    return FrameStatus.OK
