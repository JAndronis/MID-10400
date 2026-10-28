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

from dataclasses import dataclass
from typing import Any

import numpy as np

from analysis.common.cpu import set_thread_env
from analysis.common.plan import Block
from analysis.common.status import DataCheckFailed, FrameStatus
from analysis.common.writer import FrameRows
from analysis.waxs.config import JungfrauWaxsConfig
from analysis.waxs.integrate import frame_maxima, integrate_frame
from analysis.waxs.operator import WaxsOperator, build_operator
from analysis.waxs.photons import PhotonCounting
from analysis.waxs.plan import open_detector, read_train
from analysis.waxs.writer import JungfrauWaxsWriter

__all__ = [
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


def process_block(block: Block) -> FrameRows:
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

    result = JungfrauWaxsWriter.empty_rows(block.n_frames, op.npt)
    timings = {"read_data": 0.0, "read_mask": 0.0, "photonise": 0.0, "integrate": 0.0}
    bits_present = 0
    offset = 0

    for train_id, expected in zip(block.train_ids, block.expected_frames, strict=True):
        rows = slice(offset, offset + expected)
        offset += expected

        train = read_train(state.detector, train_id, timings)
        if train is None or not set(lit) <= set(train.positions):
            # A lit cell missing too leaves rows this train owns that cannot
            # be filled by label.
            result.status[rows] = FrameStatus.LABEL_MISMATCH
            continue
        frame_data, frame_mask = train.data, train.mask
        bits_present |= int(np.bitwise_or.reduce(frame_mask, axis=None))

        bits = np.uint32(cfg.mask_bits)
        for index, cell in enumerate(lit):
            row = rows.start + index
            position = train.positions[cell]
            result.put(row, {"trainId": train_id, "cellId": cell})
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
                reaching, checked = frame_maxima(values, bad, op.static_bad)
                result.put(
                    row,
                    {
                        "status": FrameStatus.DATA_CHECK_FAILED,
                        "max_kev": reaching,
                        "max_kev_static": checked,
                        "n_bad_pixels": int(bad.sum()),
                    },
                )
                continue
            result.put(row, frame_result.as_row())
            result.status[row] = FrameStatus.OK

    result.bits_present = bits_present
    result.timings = timings
    return result
