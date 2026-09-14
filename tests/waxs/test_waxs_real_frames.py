"""The W2 gate, kept as a test: real r0423 frames, both detectors.

``data/`` is not committed (it is facility data), so this skips wherever the
exported train is absent. Where it is present, it is the only place the pass is
exercised on real JUNGFRAU values — dense float32 keV with a negative tail and
a 75.9 % / 84.4 % static mask — rather than on the mock's synthetic ones.

The exported files are one train of r0423 (2637695397) read as

    JUNGFRAU(run, detector_name=f"MID_EXP_JF500K{n}")
        .select_trains(...).get_array("data.adc").squeeze("module")

**Two things the export cannot speak to, and this file must not pretend it can.**

*It carries no ``data.memoryCell``.* The lit *array positions* are 0–7, but the
cell ids those positions hold are 0–6 and 15 (measured on the cluster,
``EXPECTED_LIT_CELLS``). Anything here that classifies cells is therefore about
positions, and says so.

*``proc_mask_jf2_r423.nc`` is a copy of jf1's mask* — ``module=1``, byte-identical
to the jf1 export. It is the same mix-up as §6 O3, which was fixed for the data
and not for the mask. So jf2's ``data.mask`` assertions are skipped here until it
is re-exported; the cluster, with the right mask, flags **all** of jf2's extreme
pixels, which is the opposite of what this file once recorded.

The geometry is *not* real either: the PONI files live on GPFS, so a synthetic
one is used and the q values here mean nothing. What is tested is the frame
arithmetic and the masks, neither of which depends on where the beam centre is.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyFAI")
pytest.importorskip("fabio")
xr = pytest.importorskip("xarray")

from waxs_mockrun import write_synthetic_poni  # noqa: E402

from analysis.common.masks import frame_bad  # noqa: E402
from analysis.waxs.cells import CellAccumulator  # noqa: E402
from analysis.waxs.config import (  # noqa: E402
    EXPECTED_BITS,
    EXPECTED_LIT_CELLS,
    JungfrauWaxsConfig,
)
from analysis.waxs.integrate import ErrorModel, integrate_frame  # noqa: E402
from analysis.waxs.operator import build_operator  # noqa: E402
from analysis.waxs.selftest import run_selftest  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"

#: Measured on r0423 train 2637695397. ``static_fraction`` comes from the
#: ``.edf``, which is correct for both detectors; the rest depends on
#: ``data.mask`` and so is trustworthy for jf1 only (see the module docstring).
MEASURED = {
    "jf1": {
        "read_noise_kev": 0.3230,
        "static_fraction": 0.759,
        "kept_per_cell": 126125,
    },
    "jf2": {"read_noise_kev": 0.3175, "static_fraction": 0.844, "kept_per_cell": 81767},
}

#: ``proc_mask_jf2_r423.nc`` is a duplicate of jf1's export, so anything that
#: reads it for jf2 is reading the wrong detector's mask.
MASK_EXPORT_TRUSTWORTHY = {"jf1": True, "jf2": False}


def files_for(detector):
    return (
        DATA / f"proc_data_{detector}_r423.nc",
        DATA / f"proc_mask_{detector}_r423.nc",
        DATA / f"{detector}_mask.edf",
    )


@pytest.fixture(params=["jf1", "jf2"])
def real(request, tmp_path):
    detector = request.param
    data_file, mask_file, edf = files_for(detector)
    for path in (data_file, mask_file, edf):
        if not path.exists():
            pytest.skip(f"{path} is absent (facility data is not committed)")

    cfg = JungfrauWaxsConfig(
        proposal=10400,
        run=423,
        detector=detector,
        poni_file=str(write_synthetic_poni(tmp_path / f"{detector}.poni")),
        static_mask_file=str(edf),
        npt=500,
    )
    op, ai = build_operator(cfg)
    data = xr.open_dataarray(data_file).values[0]
    mask = xr.open_dataarray(mask_file).values[0]
    return dataclasses.replace(cfg), op, ai, data, mask, detector


def test_the_data_is_what_claude_md_records(real):
    cfg, op, ai, data, mask, detector = real
    assert data.dtype == np.float32
    assert data.shape == (16, 512, 1024)
    assert mask.dtype == np.uint32

    bits = int(np.bitwise_or.reduce(mask, axis=None))
    assert {b for b in range(32) if bits >> b & 1} == set(EXPECTED_BITS)
    # Bit 22 NON_STANDARD_SIZE is set, which is why no seam mask is needed.
    assert bits & (1 << 22)

    expected = MEASURED[detector]
    # The .edf is correct for both detectors.
    assert op.static_bad.mean() == pytest.approx(expected["static_fraction"], abs=0.001)
    if MASK_EXPORT_TRUSTWORTHY[detector]:
        kept = int((~op.static_bad & (mask[0].reshape(-1) == 0)).sum())
        assert kept == expected["kept_per_cell"]


def test_the_lit_array_positions_are_the_first_eight(real):
    """Positions, not cell ids: the export carries no ``data.memoryCell``.

    Feeding ``arange(16)`` as the cell ids makes this a statement about where
    the lit frames sit in the array, which is 0-7. What cell ids they carry is
    a different question, and the answer is 0-6 and 15 — see the module
    docstring. Reading one off the other is CLAUDE.md pitfall 4.
    """
    cfg, op, ai, data, mask, detector = real
    positions = np.arange(16, dtype=np.uint16)
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(data, mask, positions)
    result = accumulator.finalise()

    assert result.lit == tuple(range(8))
    # The margin that makes the threshold safe, on positions: >40 % against
    # <0.1 %, four orders of magnitude either side of it.
    assert result.lit_fraction[:8].min() > 0.40
    assert result.lit_fraction[8:].max() < 0.001

    # Positions 0-7 are consecutive, so the run-invariant shape check is happy
    # with them - which is the point: it cannot catch a pitfall-4 mix-up.
    result.check_structure(16)
    # ...and so this set is *not* the expected one, which is about cell ids.
    from analysis.waxs.cells import UnexpectedLitCells

    with pytest.raises(UnexpectedLitCells):
        result.check_expected(EXPECTED_LIT_CELLS)


def test_the_read_noise_is_recovered(real):
    cfg, op, ai, data, mask, detector = real
    if not MASK_EXPORT_TRUSTWORTHY[detector]:
        pytest.skip(f"proc_mask_{detector}_r423.nc is a copy of jf1's export")
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(data, mask, np.arange(16, dtype=np.uint16))
    result = accumulator.finalise()
    assert result.read_noise_kev == pytest.approx(
        MEASURED[detector]["read_noise_kev"], abs=0.001
    )


def test_the_nan_path_equals_the_reference_on_real_frames(real):
    """The W2 gate. Measured exact on both detectors, all 500 bins."""
    cfg, op, ai, data, mask, detector = real
    model = ErrorModel(MEASURED[detector]["read_noise_kev"], cfg.photon_energy_kev)
    # Array positions 0-7, which is where the lit frames are in the export.
    frames = [
        (data[position], frame_bad(mask[position], cfg.mask_bits, op.static_bad))
        for position in range(8)
    ]
    report = run_selftest(ai, op, model, frames, max_abs_kev=cfg.max_abs_kev)

    assert report.n_frames == 8
    assert report.max_rel_signal == 0.0
    assert report.max_rel_normalization == 0.0
    assert report.max_rel_variance == 0.0
    assert len(ai.engines) == 1  # the matrix was never rebuilt


def test_a_real_frame_integrates_to_something_sensible(real):
    cfg, op, ai, data, mask, detector = real
    model = ErrorModel(MEASURED[detector]["read_noise_kev"], cfg.photon_energy_kev)
    bad = frame_bad(mask[0], cfg.mask_bits, op.static_bad)
    result = integrate_frame(ai, op, model, data[0], bad, max_abs_kev=cfg.max_abs_kev)

    assert np.isfinite(result.signal).all()
    assert (result.normalization > 0).all()
    assert result.energy_valid > 0
    # jf2's extreme pixels sit under the .edf, so the kept maximum is modest.
    assert result.max_kev < 100
    assert result.n_bad_pixels == int(bad.sum())


def test_the_value_check_catches_extremes_the_static_mask_would_have_hidden(real):
    """§3 D6, restated: the check is insurance, not a fix for an unflagged set.

    This file once claimed jf2 carried extreme pixels ``data.mask`` missed. It
    does not: that came from comparing jf2's data against jf1's mask (see the
    module docstring). On the cluster every extreme pixel on both detectors is
    flagged. What survives is the weaker, still-worth-having statement: strip
    the ``.edf`` away and the value check is what stops a 1.8e5 keV pixel
    reaching a q bin.
    """
    cfg, op, ai, data, mask, detector = real
    from analysis.common.status import FrameStatus
    from analysis.waxs.integrate import frame_data_status

    extreme = np.abs(data) > cfg.max_abs_kev
    assert extreme.any(), "the export should carry some extreme pixels"
    # Every one of them is under the .edf, so none reaches the integrator.
    assert not (extreme.reshape(16, -1)[:, ~op.static_bad]).any()

    naked = dataclasses.replace(cfg, static_mask_file=None)
    naked_op, _ = build_operator(naked, poni_file=cfg.poni_file)
    failing = [
        cell
        for cell in range(16)
        if frame_data_status(data[cell], naked_op.static_bad, cfg.max_abs_kev)
        is FrameStatus.DATA_CHECK_FAILED
    ]
    assert failing, "without the .edf the value check must fire"
