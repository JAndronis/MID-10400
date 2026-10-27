"""Per-block read and frame loop for one JUNGFRAU detector.

One single-threaded spawned worker per core. Each worker rebuilds the operator
from the config once per process and checks its sha256 against the parent's,
so no engine has to be pickled. Thread limits come from the parent's
environment. ``MultimodKeyData.ndarray`` takes no ``decompress_threads``
argument, so this read path has no threaded decompression to disable.

All sixteen cells are read and the lit ones selected in memory. EXtra-data's
``roi=`` slices the array axis while the lit set is defined by
``data.memoryCell`` values, and one proc chunk spans all sixteen cells, so it
would save no I/O.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from analysis.common.cpu import set_thread_env
from analysis.common.plan import Block
from analysis.common.status import DataCheckFailed, FrameStatus
from analysis.waxs.config import CELLS_PER_TRAIN, MODULE_SHAPE, JungfrauWaxsConfig
from analysis.waxs.integrate import frame_maxima, integrate_frame
from analysis.waxs.operator import WaxsOperator, build_operator
from analysis.waxs.photons import N_ASICS, PhotonCounting

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
    counting: PhotonCounting
    lit_cells: tuple[int, ...]
    detector: Any


_STATE: _WorkerState | None = None


def init(
    cfg: JungfrauWaxsConfig,
    operator_sha256: str,
    lit_cells: tuple[int, ...],
    run_dir: str | None = None,
) -> None:
    """Process initializer: rebuild the operator, open the run.

    The photon-counting settings are rebuilt from ``cfg`` too. They are config
    fields, so a worker cannot count with anything the hash does not cover.

    :param operator_sha256: the parent's operator hash. A worker whose geometry
        or static mask differs, for example after a mask file was replaced
        mid-run, raises instead of integrating against a different q axis.
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
        counting=PhotonCounting.from_config(cfg),
        lit_cells=tuple(lit_cells),
        detector=open_detector(cfg, dc),
    )


def init_from_detector(
    cfg: JungfrauWaxsConfig,
    op: WaxsOperator,
    ai: Any,
    lit_cells: tuple[int, ...],
    detector: Any,
) -> None:
    """In-process initializer for tests and single-process runs."""
    global _STATE
    _STATE = _WorkerState(
        cfg=cfg,
        op=op,
        ai=ai,
        counting=PhotonCounting.from_config(cfg),
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
    signal: np.ndarray  # (n, npt) float32, photons
    normalization: np.ndarray  # (n, npt) float32
    variance: np.ndarray  # (n, npt) float32, photons²
    energy_valid: np.ndarray  # (n,) float32, raw keV
    n_photons: np.ndarray  # (n,) uint32
    n_bad_pixels: np.ndarray  # (n,) uint32
    n_negative_variance_bins: np.ndarray  # (n,) uint32
    max_kev: np.ndarray  # (n,) float32
    max_kev_static: np.ndarray  # (n,) float32
    n_extreme_pixels: np.ndarray  # (n,) uint32
    asic_level: np.ndarray  # (n, 8) float32, keV; NaN where an ASIC had none
    n_level_missing: np.ndarray  # (n,) uint8
    count_energy_kev: np.ndarray  # (n,) float32
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
        n_photons=np.zeros(n, dtype=np.uint32),
        n_bad_pixels=np.zeros(n, dtype=np.uint32),
        n_negative_variance_bins=np.zeros(n, dtype=np.uint32),
        max_kev=np.zeros(n, dtype=np.float32),
        max_kev_static=np.zeros(n, dtype=np.float32),
        n_extreme_pixels=np.zeros(n, dtype=np.uint32),
        asic_level=np.zeros((n, N_ASICS), dtype=np.float32),
        n_level_missing=np.zeros(n, dtype=np.uint8),
        count_energy_kev=np.zeros(n, dtype=np.float32),
    )


def process_block(block: Block) -> BlockResult:
    """Read and integrate one block.

    Every frame's identity comes from the entry's own labels: the train from
    ``data.trainId`` and the cell from ``data.memoryCell``, never from array
    position. ``JUNGFRAU.cell_ids()`` is not used because it reads the
    first train only and assumes the rest match.

    A train whose labels disagree with the plan, or which is missing one of the
    lit cells, is marked ``LABEL_MISMATCH`` and none of its frames integrated.
    """
    if _STATE is None:
        raise RuntimeError("worker.init has not run in this process")
    state = _STATE
    cfg, op, ai, counting, lit = (
        state.cfg,
        state.op,
        state.ai,
        state.counting,
        state.lit_cells,
    )

    from extra_data import by_id

    result = _empty_result(block, op.npt)
    timings = {"read_data": 0.0, "read_mask": 0.0, "photonise": 0.0, "integrate": 0.0}
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

        # The reader's train coordinate is the INDEX's, which filed this entry
        # under ``train_id`` by construction. Only the entry's own
        # ``data.trainId`` row can say it belongs to another train.
        row_labels = np.asarray(selected["data.trainId"].ndarray()).reshape(-1)
        cell_ids = selected["data.memoryCell"].ndarray()

        if (
            row_labels.size != 1
            or int(row_labels[0]) != train_id
            or data.shape[:2] != (1, 1)
            or mask.shape != data.shape
            or data.shape[2] != CELLS_PER_TRAIN
            or data.shape[3:] != MODULE_SHAPE
        ):
            result.status[rows] = FrameStatus.LABEL_MISMATCH
            continue

        frame_data, frame_mask = data[0, 0], mask[0, 0]
        cells = np.asarray(cell_ids).reshape(-1)

        positions = {int(cell): index for index, cell in enumerate(cells.tolist())}
        if len(positions) != cells.size or not set(lit) <= set(positions):
            # Repeated or missing memory cells, so the rows this train owns cannot
            # be filled by label. Filling them by array position could put a
            # frame on the wrong cell.
            result.status[rows] = FrameStatus.LABEL_MISMATCH
            continue

        bits_present |= int(np.bitwise_or.reduce(frame_mask, axis=None))

        bits = np.uint32(cfg.mask_bits)
        for index, cell in enumerate(lit):
            row = rows.start + index
            position = positions[cell]
            result.train_id[row] = train_id
            result.cell_id[row] = cell
            values = frame_data[position]
            # Kept apart from the static mask because the zero level can be
            # estimated over .edf-masked pixels as well.
            dynamic = (frame_mask[position].reshape(-1) & bits) != 0
            try:
                frame_result = integrate_frame(
                    ai,
                    op,
                    counting,
                    values,
                    dynamic,
                    max_abs_kev=cfg.max_abs_kev,
                    timings=timings,
                )
            except DataCheckFailed:
                # Every pixel was excluded. Extreme values alone are masked and
                # counted, not raised. The row still records the maxima and the
                # bad-pixel count, so the status can be explained.
                bad = dynamic | op.static_bad
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
            result.n_photons[row] = frame_result.n_photons
            result.n_bad_pixels[row] = frame_result.n_bad_pixels
            result.n_negative_variance_bins[row] = frame_result.n_negative_variance_bins
            result.max_kev[row] = frame_result.max_kev
            result.max_kev_static[row] = frame_result.max_kev_static
            result.n_extreme_pixels[row] = frame_result.n_extreme_pixels
            result.asic_level[row] = frame_result.asic_level
            result.n_level_missing[row] = frame_result.n_level_missing
            result.count_energy_kev[row] = frame_result.count_energy_kev
            result.status[row] = FrameStatus.OK

    result.bits_present = bits_present
    result.timings = timings
    return result
