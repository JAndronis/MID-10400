"""Scaling one detector onto the other over the overlap (§6 O5)."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.waxs.combine import (
    NoOverlap,
    combine_curves,
    mean_curve,
    scale_to_overlap,
)

# The populated ranges after masking: jf1 11.5-23.7, jf2 9.8-18.5 nm^-1.
Q_REF = np.linspace(11.5, 23.7, 500)
Q_OTHER = np.linspace(9.8, 18.5, 500)
TRUE_FACTOR = 2.5


def featureless(q):
    return 100.0 / np.asarray(q) ** 2 + 0.5


def featured(q):
    q = np.asarray(q)
    return featureless(q) + 3.0 * np.exp(-0.5 * ((q - 14.0) / 0.25) ** 2)


def curves(shape=featureless, factor=TRUE_FACTOR, q_shift=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    i_ref = shape(Q_REF)
    i_other = shape(Q_OTHER * q_shift) / factor
    if noise:
        i_ref = i_ref * (1 + rng.normal(0, noise, i_ref.size))
        i_other = i_other * (1 + rng.normal(0, noise, i_other.size))
    return i_ref, np.abs(0.01 * i_ref), i_other, np.abs(0.01 * i_other)


def test_the_factor_is_recovered_exactly_without_noise():
    i_ref, s_ref, i_other, s_other = curves()
    scaling = scale_to_overlap(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    assert scaling.factor == pytest.approx(TRUE_FACTOR, rel=1e-5)
    assert scaling.residual_rms < 1e-5


def test_the_overlap_is_the_shared_range():
    i_ref, s_ref, i_other, s_other = curves()
    scaling = scale_to_overlap(Q_REF, i_ref, Q_OTHER, i_other)
    assert scaling.q_low == pytest.approx(11.5, abs=0.05)
    assert scaling.q_high == pytest.approx(18.5, abs=0.05)
    assert scaling.n_bins > 200


def test_noise_moves_the_factor_within_its_stated_error():
    i_ref, s_ref, i_other, s_other = curves(noise=0.01, seed=1)
    scaling = scale_to_overlap(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    assert abs(scaling.factor - TRUE_FACTOR) < 5 * scaling.factor_error
    assert 0.2 < scaling.reduced_chi2 < 5.0


def test_the_median_method_agrees_and_ignores_the_errors():
    i_ref, s_ref, i_other, s_other = curves(noise=0.01, seed=2)
    wls = scale_to_overlap(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    median = scale_to_overlap(Q_REF, i_ref, Q_OTHER, i_other, method="median")
    assert median.factor == pytest.approx(wls.factor, rel=0.02)
    assert np.isnan(median.factor_error)


def test_a_q_mismatch_is_caught_when_the_overlap_has_a_feature():
    """The cross-check on the two PONIs, on a run with Bragg peaks."""
    for shift, floor in ((1.01, 10.0), (1.05, 100.0)):
        i_ref, s_ref, i_other, s_other = curves(shape=featured, q_shift=shift)
        scaling = scale_to_overlap(
            Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
        )
        assert scaling.reduced_chi2 > floor, shift
        assert scaling.residual_rms > 0.1


def test_a_q_mismatch_is_invisible_on_a_featureless_curve():
    """Measured, and the reason the check must not be oversold.

    ``I(1.05 q) = 1.05**-2 I(q)`` for a power law, so a 5 % q error is exactly a
    9.3 % scale error and the fit simply absorbs it. On r0423 — no Bragg
    reflections — a good chi2 here means the two detectors agree in *shape*, not
    that either q axis is right.
    """
    i_ref, s_ref, i_other, s_other = curves(shape=featureless, q_shift=1.05)
    scaling = scale_to_overlap(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    # The fit still looks healthy...
    assert scaling.reduced_chi2 < 1.0
    assert scaling.residual_rms < 0.01
    # ...while the q error has been absorbed into a factor that is simply wrong,
    # by far more than its own stated precision. Nothing in the output says so.
    assert abs(scaling.factor - TRUE_FACTOR) / TRUE_FACTOR > 0.02
    assert abs(scaling.factor - TRUE_FACTOR) > 10 * scaling.factor_error


def test_curves_that_do_not_meet_are_refused():
    with pytest.raises(NoOverlap, match="do not overlap"):
        scale_to_overlap(
            np.linspace(1, 2, 100),
            np.ones(100),
            np.linspace(5, 6, 100),
            np.ones(100),
        )


def test_too_thin_an_overlap_is_refused():
    """A factor from three bins is a number, not a measurement."""
    with pytest.raises(NoOverlap, match="fewer than"):
        scale_to_overlap(
            np.linspace(10.0, 20.0, 50),
            featureless(np.linspace(10.0, 20.0, 50)),
            np.linspace(19.9, 30.0, 50),
            featureless(np.linspace(19.9, 30.0, 50)),
        )


def test_an_unknown_method_is_refused():
    i_ref, _, i_other, _ = curves()
    with pytest.raises(ValueError, match="method must be one of"):
        scale_to_overlap(Q_REF, i_ref, Q_OTHER, i_other, method="eyeball")


# ── the merged curve ─────────────────────────────────────────────────────────
def test_the_combined_curve_spans_both_and_marks_its_sources():
    i_ref, s_ref, i_other, s_other = curves()
    combined, scaling = combine_curves(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    q = combined["q"].values
    source = combined["source"].values

    assert q.min() == pytest.approx(Q_OTHER.min())
    assert q.max() == pytest.approx(Q_REF.max())
    assert set(np.unique(source)) == {0, 1, 2}
    # below the reference's range: the scaled other only
    assert (source[q < Q_REF.min()] == 1).all()
    # above the other's range: the reference only
    assert (source[q > Q_OTHER.max()] == 0).all()
    assert np.isfinite(combined["intensity"].values).all()
    assert (combined["intensity"].values > 0).all()


def test_the_combined_curve_recovers_the_underlying_shape():
    i_ref, s_ref, i_other, s_other = curves()
    combined, _ = combine_curves(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    q = combined["q"].values
    got = combined["intensity"].values
    assert got == pytest.approx(featureless(q), rel=1e-3)


def test_the_reference_is_the_curve_that_does_not_move():
    i_ref, s_ref, i_other, s_other = curves()
    combined, scaling = combine_curves(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    q = combined["q"].values
    reference_only = combined["source"].values == 0
    expected = np.interp(q[reference_only], Q_REF, i_ref)
    assert combined["intensity"].values[reference_only] == pytest.approx(expected)


def test_the_scaling_is_recorded_on_the_result():
    i_ref, s_ref, i_other, s_other = curves()
    combined, scaling = combine_curves(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    assert combined.attrs["scaling_factor"] == pytest.approx(scaling.factor)
    assert combined.attrs["scaling_n_bins"] == scaling.n_bins
    assert combined.attrs["scaling_method"] == "wls"


# ── the adapters ─────────────────────────────────────────────────────────────
def test_mean_curve_from_a_per_cell_grid(cfg, mock_run_factory, tmp_path):
    from analysis.waxs.run import run_jungfrau_waxs

    mock, dc = mock_run_factory()
    grid = run_jungfrau_waxs(
        cfg, dc=dc, run_dir=mock.path, output_path=tmp_path / "c.h5", reduce="per_cell"
    )
    q, intensity, sigma = mean_curve(grid)
    assert q.shape == intensity.shape == sigma.shape == (cfg.npt,)
    assert np.isfinite(intensity).all()


def test_mean_curve_from_pooled_carries_real_errors(cfg, mock_run_factory, tmp_path):
    from analysis.waxs.run import run_jungfrau_waxs

    mock, dc = mock_run_factory()
    pooled = run_jungfrau_waxs(
        cfg, dc=dc, run_dir=mock.path, output_path=tmp_path / "p.h5", reduce="pooled"
    )
    q, intensity, sigma = mean_curve(pooled)
    assert (sigma[intensity > 0] > 0).all()


def test_combine_files_reads_two_finished_runs(
    config_for_detector, mock_run_factory, tmp_path
):
    from analysis.waxs.combine import combine_files
    from analysis.waxs.config import DETECTOR_MODNOS, DETECTOR_NAMES
    from analysis.waxs.run import run_jungfrau_waxs

    paths = {}
    for detector in ("jf1", "jf2"):
        cfg = config_for_detector(detector, output_root=str(tmp_path / detector))
        mock, dc = mock_run_factory(
            detector_name=DETECTOR_NAMES[detector], module=DETECTOR_MODNOS[detector]
        )
        run_jungfrau_waxs(cfg, dc=dc, run_dir=mock.path, reduce="none")
        paths[detector] = cfg.output_file

    combined, scaling = combine_files(paths["jf1"], paths["jf2"], min_bins=4)
    assert scaling.factor > 0
    assert scaling.n_bins >= 4
    assert np.isfinite(combined["intensity"].values).all()


# ── telling a normalisation offset from a q-dependent one ────────────────────
def test_a_pure_normalisation_offset_leaves_a_flat_residual():
    """The factor absorbs it, so there is nothing left to trend."""
    i_ref, s_ref, i_other, s_other = curves(factor=3.7)
    scaling = scale_to_overlap(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    assert scaling.factor == pytest.approx(3.7, rel=1e-5)
    assert abs(scaling.residual_slope_per_nm) < 1e-6
    assert abs(scaling.residual_at_q_low) < 1e-5
    assert abs(scaling.residual_at_q_high) < 1e-5


def test_a_q_dependent_difference_tilts_the_residual():
    """A correction applied to neither detector, growing with scattering angle.

    Modelled on the polarisation factor, whose azimuthal amplitude goes as
    sin^2(2theta) and so grows through the overlap. No single factor can remove
    it, and the slope is what says so.
    """
    i_ref, s_ref, i_other, s_other = curves()
    tilt = 1.0 + 0.01 * (Q_OTHER - Q_OTHER.min()) / np.ptp(Q_OTHER)
    scaling = scale_to_overlap(
        Q_REF,
        i_ref,
        Q_OTHER,
        i_other * tilt,
        sigma_ref=s_ref,
        sigma_other=s_other,
    )
    assert abs(scaling.residual_slope_per_nm) > 1e-4
    # ...and the two ends of the overlap straddle zero, because the factor fits
    # the middle: that shape is the signature, not the magnitude.
    assert scaling.residual_at_q_low * scaling.residual_at_q_high < 0


def test_sigma_over_intensity_separates_disagreement_from_tight_errors():
    """chi2 alone cannot say whether curves differ or errors are understated."""
    i_ref, s_ref, i_other, s_other = curves(shape=featureless, q_shift=1.02)

    generous = scale_to_overlap(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    tight = scale_to_overlap(
        Q_REF,
        i_ref,
        Q_OTHER,
        i_other,
        sigma_ref=s_ref / 10,
        sigma_other=s_other / 10,
    )
    # The same curves, so the same physical disagreement...
    assert tight.residual_rms == pytest.approx(generous.residual_rms, rel=1e-6)
    # ...but a hundredfold worse chi2 purely from the error bars.
    assert tight.reduced_chi2 == pytest.approx(100 * generous.reduced_chi2, rel=0.01)
    # sigma_over_intensity recovers what was claimed, in both cases.
    assert tight.sigma_over_intensity == pytest.approx(
        generous.sigma_over_intensity / 10, rel=0.01
    )


def test_the_trend_is_reported_on_the_combined_dataset():
    i_ref, s_ref, i_other, s_other = curves()
    combined, scaling = combine_curves(
        Q_REF, i_ref, Q_OTHER, i_other, sigma_ref=s_ref, sigma_other=s_other
    )
    assert combined.attrs["scaling_residual_slope_per_nm"] == pytest.approx(
        scaling.residual_slope_per_nm
    )
