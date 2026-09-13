"""The W2 gate, kept as a test: real r0423 frames, both detectors.

``data/`` is not committed (it is facility data), so this skips wherever the
exported train is absent. Where it is present, it is the only place the pass is
exercised on real JUNGFRAU values — dense float32 keV with a negative tail, a
75.9 % / 84.4 % static mask, and jf2's unflagged extreme pixels — rather than on
the mock's synthetic ones.

The exported files are one train of r0423 (2637695397) read as

    JUNGFRAU(run, detector_name=f"MID_EXP_JF500K{n}")
        .select_trains(...).get_array("data.adc").squeeze("module")

The geometry is *not* real: the PONI files live on GPFS, so a synthetic one is
used and the q values here mean nothing. What is being tested is the frame
arithmetic, the masks and the cell classification, none of which depend on where
the beam centre is.
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
from analysis.waxs.config import EXPECTED_BITS, JungfrauWaxsConfig  # noqa: E402
from analysis.waxs.integrate import ErrorModel, integrate_frame  # noqa: E402
from analysis.waxs.operator import build_operator  # noqa: E402
from analysis.waxs.selftest import run_selftest  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"

#: Measured on r0423 train 2637695397 (CLAUDE.md, JUNGFRAU row).
MEASURED = {
    "jf1": {
        "read_noise_kev": 0.3230,
        "static_fraction": 0.759,
        "kept_per_cell": 126125,
    },
    "jf2": {"read_noise_kev": 0.3175, "static_fraction": 0.844, "kept_per_cell": 81767},
}


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
    assert op.static_bad.mean() == pytest.approx(expected["static_fraction"], abs=0.001)
    kept = int((~op.static_bad & (mask[0].reshape(-1) == 0)).sum())
    assert kept == expected["kept_per_cell"]


def test_the_lit_cells_and_read_noise_are_recovered(real):
    cfg, op, ai, data, mask, detector = real
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(data, mask, np.arange(16, dtype=np.uint16))
    result = accumulator.finalise()

    assert result.lit == tuple(range(8))
    result.check_expected(cfg.expected_lit_cells)
    assert result.read_noise_kev == pytest.approx(
        MEASURED[detector]["read_noise_kev"], abs=0.001
    )
    # The margin that makes the threshold safe: 42-53 % against <= 0.08 %.
    assert result.lit_fraction[:8].min() > 0.40
    assert result.lit_fraction[8:].max() < 0.001


def test_the_nan_path_equals_the_reference_on_real_frames(real):
    """The W2 gate. Measured exact on both detectors, all 500 bins."""
    cfg, op, ai, data, mask, detector = real
    model = ErrorModel(MEASURED[detector]["read_noise_kev"], cfg.photon_energy_kev)
    frames = [
        (data[cell], frame_bad(mask[cell], cfg.mask_bits, op.static_bad))
        for cell in range(8)
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


def test_the_extreme_pixels_are_still_caught_without_the_static_mask(real):
    """§3 D6: data.mask alone is not sufficient on jf2, and this is why."""
    cfg, op, ai, data, mask, detector = real
    unmasked = int(
        (np.abs(data) > 1e3).sum() and ((np.abs(data[0]) > 1e3) & (mask[0] == 0)).sum()
    )
    if detector == "jf1":
        assert unmasked == 0  # jf1 flags all of its own
    else:
        assert unmasked > 0  # jf2 flags none of them
        # ...and the value check catches them when the .edf does not.
        naked = dataclasses.replace(cfg, static_mask_file=None)
        naked_op, naked_ai = build_operator(naked, poni_file=cfg.poni_file)
        from analysis.common.status import FrameStatus
        from analysis.waxs.integrate import frame_data_status

        assert (
            frame_data_status(data[0], naked_op.static_bad, cfg.max_abs_kev)
            is FrameStatus.DATA_CHECK_FAILED
        )
