"""Per-block read and frame loop for one JUNGFRAU detector.

One single-threaded worker per core, spawned. The worker owns nothing but its
own reads: it rebuilds the operator from the config — a PONI load plus one
sparse-matrix build, tens of milliseconds, once per process — and checks the
result's sha256 against the parent's, so no engine has to be pickled.

Thread limits are inherited from the parent's environment. Note that
``MultimodKeyData.ndarray`` takes no ``decompress_threads`` argument — that is
AGIPD's ``XtdfImageMultimodKeyData`` — so this read path has no threaded
decompression to disable.

**All sixteen cells are read and the lit ones selected in memory.** EXtra-data's
``roi=`` would read only the lit cells, but it slices the *array* axis while the
lit set is defined by ``data.memoryCell`` values, and the proc chunk layout spans
all sixteen cells per chunk, so it would save no I/O.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from analysis.common.cpu import set_thread_env
from analysis.common.masks import frame_bad
from analysis.common.plan import Block
from analysis.common.status import DataCheckFailed, FrameStatus
from analysis.waxs.config import CELLS_PER_TRAIN, MODULE_SHAPE, JungfrauWaxsConfig
from analysis.waxs.integrate import ErrorModel, frame_maxima, integrate_frame
from analysis.waxs.operator import WaxsOperator, build_operator

__all__ = [
    "BlockResult",
    "init",
    "init_from_detector",
    "process_block",
]


@dataclass(slots=True)
class _WorkerState:
    cfg: JungfrauWaxsConfig
    op: WaxsOperator
    ai: Any
    model: ErrorModel
    lit_cells: tuple[int, ...]
    detector: Any


_STATE: _WorkerState | None = None


def init(
    cfg: JungfrauWaxsConfig,
    operator_sha256: str,
    model: ErrorModel,
    lit_cells: tuple[int, ...],
    run_dir: str | None = None,
) -> None:
    """Process initializer: rebuild the operator, open the run.

    :param operator_sha256: what the parent built. A worker whose geometry or
        static mask differs — a mask file replaced mid-run, say — must not
        quietly integrate against a different q axis.
    """
    set_thread_env()

    from extra_data import RunDirectory, open_run

    from analysis.waxs.plan import open_detector

    op, ai = build_operator(cfg)
    if op.sha256 != operator_sha256:
        raise ValueError(
            f"this worker built operator {op.sha256}, the parent built "
            f"{operator_sha256}; the geometry or the static mask has changed "
            "underneath the run"
        )

    dc = (
        RunDirectory(run_dir)
        if run_dir
        else open_run(cfg.proposal, cfg.run, data="proc")
    )
    global _STATE
    _STATE = _WorkerState(
        cfg=cfg,
        op=op,
        ai=ai,
        model=model,
        lit_cells=tuple(lit_cells),
        detector=open_detector(cfg, dc),
    )


def init_from_detector(
    cfg: JungfrauWaxsConfig,
    op: WaxsOperator,
    ai: Any,
    model: ErrorModel,
    lit_cells: tuple[int, ...],
    detector: Any,
) -> None:
    """In-process initializer for tests and single-process runs."""
    global _STATE
    _STATE = _WorkerState(
        cfg=cfg,
        op=op,
        ai=ai,
        model=model,
        lit_cells=tuple(lit_cells),
        detector=detector,
    )


@dataclass(slots=True)
class BlockResult:
    """One block's frames, addressed by label rather than by position."""

    block_index: int
    train_id: np.ndarray  # (n,) uint64
    cell_id: np.ndarray  # (n,) uint16
    status: np.ndarray  # (n,) uint8
    signal: np.ndarray  # (n, npt) float32
    normalization: np.ndarray  # (n, npt) float32
    variance: np.ndarray  # (n, npt) float32
    energy_valid: np.ndarray  # (n,) float32
    n_bad_pixels: np.ndarray  # (n,) uint32
    n_negative_variance_bins: np.ndarray  # (n,) uint32
    max_kev: np.ndarray  # (n,) float32
    max_kev_static: np.ndarray  # (n,) float32
    n_extreme_pixels: np.ndarray  # (n,) uint32
    bits_present: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.train_id.size)


def _empty_result(block: Block, npt: int) -> BlockResult:
    n = block.n_frames
    return BlockResult(
        block_index=block.index,
        train_id=np.zeros(n, dtype=np.uint64),
        cell_id=np.zeros(n, dtype=np.uint16),
        status=np.full(n, FrameStatus.NOT_PROCESSED, dtype=np.uint8),
        signal=np.zeros((n, npt), dtype=np.float32),
        normalization=np.zeros((n, npt), dtype=np.float32),
        variance=np.zeros((n, npt), dtype=np.float32),
        energy_valid=np.zeros(n, dtype=np.float32),
        n_bad_pixels=np.zeros(n, dtype=np.uint32),
        n_negative_variance_bins=np.zeros(n, dtype=np.uint32),
        max_kev=np.zeros(n, dtype=np.float32),
        max_kev_static=np.zeros(n, dtype=np.float32),
        n_extreme_pixels=np.zeros(n, dtype=np.uint32),
    )


def _squeeze_module(array: np.ndarray, name: str) -> np.ndarray:
    """Drop the module and train axes of a one-module, one-train read."""
    if array.shape[0] != 1 or array.shape[1] != 1:
        raise ValueError(
            f"{name} has shape {array.shape}; expected one module and one train"
        )
    return array[0, 0]


def process_block(block: Block) -> BlockResult:
    """Read and integrate one block.

    Every frame's identity comes from the reader: the train from
    ``train_id_coordinates()`` and the cell from ``data.memoryCell``, never from
    array position. ``JUNGFRAU.cell_ids()`` is deliberately
    not used — it reads the first train only and asserts the rest match, which
    is an assumption this pass has not verified for this beamtime.

    A train whose labels disagree with the plan, or which is missing one of the
    lit cells, is marked ``LABEL_MISMATCH`` and none of its frames integrated.
    """
    if _STATE is None:
        raise RuntimeError("worker.init has not run in this process")
    state = _STATE
    cfg, op, ai, model, lit = (
        state.cfg,
        state.op,
        state.ai,
        state.model,
        state.lit_cells,
    )

    from extra_data import by_id

    result = _empty_result(block, op.npt)
    timings = {"read_data": 0.0, "read_mask": 0.0, "integrate": 0.0}
    bits_present = 0
    offset = 0

    for train_id, expected in zip(block.train_ids, block.expected_frames, strict=True):
        rows = slice(offset, offset + expected)
        offset += expected

        selected = state.detector.select_trains(by_id[[train_id]])
        adc_key = selected["data.adc"]
        mask_key = selected["data.mask"]

        started = time.perf_counter()
        data = adc_key.ndarray()
        timings["read_data"] += time.perf_counter() - started

        started = time.perf_counter()
        mask = mask_key.ndarray()
        timings["read_mask"] += time.perf_counter() - started

        train_ids = adc_key.train_id_coordinates()
        cell_ids = selected["data.memoryCell"].ndarray()

        if (
            train_ids.size != 1
            or int(train_ids[0]) != train_id
            or data.shape[:2] != (1, 1)
            or mask.shape != data.shape
            or data.shape[2] != CELLS_PER_TRAIN
            or data.shape[3:] != MODULE_SHAPE
        ):
            result.status[rows] = FrameStatus.LABEL_MISMATCH
            continue

        frame_data = _squeeze_module(data, "data.adc")
        frame_mask = _squeeze_module(mask, "data.mask")
        cells = np.asarray(_squeeze_module(cell_ids, "data.memoryCell")).reshape(-1)

        positions = {int(cell): index for index, cell in enumerate(cells.tolist())}
        if len(positions) != cells.size or not set(lit) <= set(positions):
            # Repeated or missing memory cells: the rows this train owns cannot
            # be filled by label, and guessing by position is what pitfall 4
            # forbids.
            result.status[rows] = FrameStatus.LABEL_MISMATCH
            continue
        if expected != len(lit):
            result.status[rows] = FrameStatus.LABEL_MISMATCH
            continue

        bits_present |= int(np.bitwise_or.reduce(frame_mask, axis=None))

        started = time.perf_counter()
        for index, cell in enumerate(lit):
            row = rows.start + index
            position = positions[cell]
            result.train_id[row] = train_id
            result.cell_id[row] = cell
            values = frame_data[position]
            bad = frame_bad(frame_mask[position], cfg.mask_bits, op.static_bad)
            try:
                frame_result = integrate_frame(
                    ai, op, model, values, bad, max_abs_kev=cfg.max_abs_kev
                )
            except DataCheckFailed:
                # This means every pixel was excluded, not that the frame
                # carried a wild value: those are masked and counted. The row is
                # still filled with what evidence there is, because a status on
                # its own explains nothing.
                result.status[row] = FrameStatus.DATA_CHECK_FAILED
                reaching, checked = frame_maxima(values, bad, op.static_bad)
                result.max_kev[row] = reaching
                result.max_kev_static[row] = checked
                result.n_bad_pixels[row] = int(bad.sum())
                continue
            result.signal[row] = frame_result.signal
            result.normalization[row] = frame_result.normalization
            result.variance[row] = frame_result.variance
            result.energy_valid[row] = frame_result.energy_valid
            result.n_bad_pixels[row] = frame_result.n_bad_pixels
            result.n_negative_variance_bins[row] = frame_result.n_negative_variance_bins
            result.max_kev[row] = frame_result.max_kev
            result.max_kev_static[row] = frame_result.max_kev_static
            result.n_extreme_pixels[row] = frame_result.n_extreme_pixels
            result.status[row] = FrameStatus.OK
        timings["integrate"] += time.perf_counter() - started

    result.bits_present = bits_present
    result.timings = timings
    return result
