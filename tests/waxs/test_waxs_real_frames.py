"""One real r0423 train per detector, read from proc, through the pass.

This is the only place the pass meets the real cell order: ``storageCellStart``
is 15, so array position 0 holds memory cell 15. The mock reproduces that
order, and this checks that the real files still have it and that the worker
places each real frame on the cell ``data.memoryCell`` names. It reads the real
PONI and ``.edf`` and skips wherever the proc files are not readable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyFAI")
pytest.importorskip("extra_data")

from analysis.common.plan import Block  # noqa: E402
from analysis.waxs import worker  # noqa: E402
from analysis.waxs.cells import CellAccumulator  # noqa: E402
from analysis.waxs.config import EXPECTED_LIT_CELLS, config_for  # noqa: E402
from analysis.waxs.integrate import integrate_frame  # noqa: E402
from analysis.waxs.operator import build_operator  # noqa: E402
from analysis.waxs.photons import PhotonCounting  # noqa: E402
from analysis.waxs.plan import open_detector  # noqa: E402
from analysis.waxs.selftest import run_selftest  # noqa: E402

PROC = Path("/gpfs/exfel/exp/MID/202601/p010400/proc/r0423")
#: A train in the science block of r0423, lit in cells 0-6 and 15.
TRAIN_ID = 2637695397

pytestmark = pytest.mark.skipif(
    not (PROC.is_dir() and os.access(PROC, os.R_OK | os.X_OK)),
    reason="the r0423 proc files are readable on Maxwell only",
)


@pytest.fixture(scope="module", params=["jf1", "jf2"])
def real(request):
    from extra_data import by_id, open_run

    cfg = config_for(10400, 423, request.param, npt=200)
    op, ai = build_operator(cfg)
    det = open_detector(cfg, open_run(10400, 423, data="proc"))
    selected = det.select_trains(by_id[[TRAIN_ID]])
    data = np.asarray(selected["data.adc"].ndarray())[0, 0]
    mask = np.asarray(selected["data.mask"].ndarray())[0, 0]
    cells = np.asarray(selected["data.memoryCell"].ndarray()).reshape(-1)
    return cfg, op, ai, det, data, mask, cells


def test_each_row_holds_the_real_frame_its_memory_cell_labels(real):
    cfg, op, ai, det, data, mask, cells = real
    assert int(cells[0]) == 15
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(data, mask, cells)
    lit = accumulator.finalise().lit
    assert lit == tuple(EXPECTED_LIT_CELLS)

    worker.init_from_detector(cfg, op, ai, lit, det)
    try:
        result = worker.process_block(Block(0, (TRAIN_ID,), (len(lit),), (0,)))
    finally:
        worker._STATE = None

    counting = PhotonCounting.from_config(cfg)
    bits = np.uint32(cfg.mask_bits)
    position = {int(cell): index for index, cell in enumerate(cells)}
    assert list(result.cell_id) == list(lit)
    for row, cell in enumerate(lit):
        at = position[cell]
        dynamic = (mask[at].reshape(-1) & bits) != 0
        alone = integrate_frame(
            ai, op, counting, data[at], dynamic, max_abs_kev=cfg.max_abs_kev
        )
        assert np.array_equal(result.signal[row], alone.signal.astype(np.float32)), cell


def test_the_nan_path_equals_the_reference_on_real_frames(real):
    """Exactly, in every bin, on dense keV data with a negative tail."""
    cfg, op, ai, det, data, mask, cells = real
    bits = np.uint32(cfg.mask_bits)
    position = {int(cell): index for index, cell in enumerate(cells)}
    frames = [
        (data[position[cell]], (mask[position[cell]].reshape(-1) & bits) != 0)
        for cell in EXPECTED_LIT_CELLS
    ]
    report = run_selftest(
        ai,
        op,
        PhotonCounting.from_config(cfg),
        frames,
        max_abs_kev=cfg.max_abs_kev,
    )
    assert report.n_frames == len(EXPECTED_LIT_CELLS)
    assert report.max_rel_signal == 0.0
    assert report.max_rel_normalization == 0.0
    assert report.max_rel_variance == 0.0
