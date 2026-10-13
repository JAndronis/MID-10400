"""XPCS SNR (context/run-triage.md §4.6) against synthetic runs with known truth.

The synthetic TTCF carries contrast β, rate Γ and, per pair, exactly the noise the
analytical formula predicts. So the jackknife must find that noise (ρ ≈ 1), the fit
must find β and Γ, and the SNR must be the formula evaluated on the run's own k̄,
N_pix, φ and train count.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from triage_fakes import FIRST_TRAIN, REP_RATE_HZ, write_xpcs_run

from analysis.triage.xpcs import (
    OFF_SUBTRACTION,
    Beta,
    DecayFit,
    QGridMismatch,
    beta_eligible,
    crl_state,
    diagonal_sums,
    fit_decay,
    jackknife_g2,
    measure_beta,
    measure_run,
    read_npix_per_bin,
    read_saxs_kbar,
    roi_bin_masks,
    roi_edges_nm,
    roi_kinds,
    slice_index,
    snr,
    summarise,
)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("xpcs")
    truth = write_xpcs_run(
        d / "damnit.h5", d / "scratch.h5", np.random.default_rng(5), nan_frames={6: 10}
    )
    m = measure_run(423, d / "damnit.h5", d / "scratch.h5", REP_RATE_HZ)
    return truth, m


def test_rois_split_into_plain_and_bragg():
    edges = roi_edges_nm()
    np.testing.assert_allclose(edges[0], [0.225, 0.35])
    assert roi_kinds(edges) == [
        "plain",
        "plain",
        "plain",
        "bragg",
        "bragg",
        "bragg",
        "plain",
    ]


def test_diagonal_sums_skip_non_finite_entries():
    t = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    t[0, 0, 0, 1] = np.nan
    sums, counts = diagonal_sums(t)
    assert counts[0, 0].tolist() == [2, 2, 1]  # lag 1 lost (0, 1)
    assert sums[0, 0].tolist() == [6.0 + 11.0, 2.0 + 7.0, 3.0]


def test_jackknife_finds_iid_noise():
    rng = np.random.default_rng(0)
    sums = rng.normal(0.0, 1.0, (500, 1, 3))
    g2, sigma, _ = jackknife_g2(sums, np.ones_like(sums, dtype=np.int32), 50)
    np.testing.assert_allclose(sigma, 1 / np.sqrt(500), rtol=0.25)


def test_fit_recovers_contrast_and_rate():
    tau = np.arange(1, 60) / REP_RATE_HZ
    rate = 1.0 / (2 * 8 * tau[0])
    g2 = 0.03 * np.exp(-2 * rate * tau) + 0.001
    fit = fit_decay(tau, g2, np.full_like(g2, 1e-4))
    assert fit.converged
    assert fit.amplitude == pytest.approx(0.03, rel=1e-3)
    assert fit.rate == pytest.approx(rate, rel=1e-3)


def test_measure_run_recovers_the_truth(synthetic):
    truth, m = synthetic
    assert m.n_usable == 200 and m.n_frames == 40
    np.testing.assert_allclose(m.kbar, truth["kbar"], rtol=1e-6)
    np.testing.assert_allclose(m.npix, truth["npix_roi"], rtol=1e-6)
    assert m.phi[0] == pytest.approx(1.0)
    assert m.phi[6] == pytest.approx(29 / 39)  # pairs with both frames >= 10
    for r in (0, 1, 2):  # within three of the fit's own jackknife errors
        fit = m.fits[r]
        assert abs(fit.amplitude - truth["beta"]) < 3 * fit.sigma_amplitude
        assert abs(fit.rate - truth["rate"]) < 3 * fit.sigma_rate


def test_the_noise_check_is_one_when_the_formula_holds(synthetic):
    _, m = synthetic
    np.testing.assert_allclose(m.rho, 1.0, atol=0.2)  # jackknife ~10 %, twice that


def test_our_g2_agrees_with_damnits(synthetic):
    _, m = synthetic
    assert np.nanmax(m.damnit_max_z) < 5


def test_beta_is_the_median_amplitude_of_eligible_runs(synthetic):
    truth, m = synthetic
    beta = measure_beta([m])
    for r in (0, 1, 2, 6):
        assert beta[r].n_eligible == 1
        assert beta[r].value == pytest.approx(truth["beta"], rel=0.1)


def test_beta_eligibility():
    tau1, tau_max = 1.0, 100.0
    good = DecayFit(0.03, 0.002, 1 / (2 * 10.0), 0.0, 0.0, True)
    assert beta_eligible(good, tau1, tau_max)
    imprecise = DecayFit(0.03, 0.005, 1 / (2 * 10.0), 0.0, 0.0, True)  # A/σ = 6
    fast = DecayFit(0.03, 0.002, 1 / (2 * 2.0), 0.0, 0.0, True)  # 1/(2Γ) = 2 τ₁
    slow = DecayFit(0.03, 0.002, 1 / (2 * 200.0), 0.0, 0.0, True)  # beyond the window
    assert not any(beta_eligible(f, tau1, tau_max) for f in (imprecise, fast, slow))


def test_snr_is_the_formula(synthetic):
    truth, m = synthetic
    beta = measure_beta([m])
    s = summarise(m, beta)
    b0 = beta[0].value
    expected = b0 * m.kbar[0] * np.sqrt(m.npix[0] * 39 * 1.0 * 200) / OFF_SUBTRACTION
    assert s["snr_run"][float(m.roi_q_A[0])] == pytest.approx(expected)
    assert s["n_slices"] == 1  # 200 trains: one part-slice of >= 150
    assert s["extractable"] == (s["best_snr"] >= 3)
    assert s["t_needed_s"] == pytest.approx(30.0 * (3 / s["best_snr"]) ** 2)


def test_an_unknown_beta_is_not_a_verdict(synthetic):
    _, m = synthetic
    nan = float("nan")
    unknown = {r: Beta(nan, 0, nan, nan, ()) for r in range(len(m.kinds))}
    s = summarise(m, unknown)
    assert s["extractable"] is None
    assert s["n_extractable_slices"] == 0
    assert np.isnan(s["best_snr"])


def test_snr_scales_with_the_square_root_of_trains_and_linearly_with_kbar():
    one = snr(0.03, 0.01, 1e5, 155, 1.0, 300)
    assert snr(0.03, 0.01, 1e5, 155, 1.0, 1200) == pytest.approx(2 * one)
    assert snr(0.03, 0.02, 1e5, 155, 1.0, 300) == pytest.approx(2 * one)


def test_slices_follow_train_ids_and_absorb_a_short_tail():
    tids = np.arange(1000, 1000 + 700)
    idx = slice_index(tids, 1000, 1699)  # 700 trains: 300 + 300 + a 100-train tail
    assert idx.max() == 1 and (idx == 1).sum() == 400
    tids = np.arange(1000, 1000 + 760)  # tail of 160 >= 150 stays its own slice
    assert slice_index(tids, 1000, 1759).max() == 2
    gap = np.r_[np.arange(1000, 1100), np.arange(1350, 1460)]  # trains dropped
    assert slice_index(gap, 1000, 1459).tolist() == [0] * 100 + [1] * 110


def test_kbar_weights_bins_by_pixels_and_skips_missing_frames(tmp_path):
    q = np.linspace(0.0813, 1.0570, 400)
    edges = roi_edges_nm()
    npix = np.where(q < 0.3, 1.0, 3.0)
    intensity = np.broadcast_to(np.where(q < 0.3, 0.01, 0.02), (2, 4, 400)).copy()
    intensity[:, 3] = 99.0  # a slot with no frame must not count
    n_frames = np.ones((2, 4), np.uint8)
    n_frames[:, 3] = 0
    with h5py.File(tmp_path / "f.h5", "w") as f:
        g = f.create_group("agipd_saxs")
        g["q"], g["trainId"] = q, FIRST_TRAIN + np.arange(2, dtype=np.uint64)
        g["intensity"], g["n_frames"] = intensity.astype(np.float32), n_frames
    k = read_saxs_kbar(tmp_path / "f.h5", edges, npix)
    m0 = roi_bin_masks(q, edges)[0]
    w = npix[m0]
    expected = (np.where(q < 0.3, 0.01, 0.02)[m0] * w).sum() / w.sum()
    assert k.kbar[0, 0] == pytest.approx(expected, rel=1e-6)
    assert k.n_frames.tolist() == [3, 3]


def test_npix_refuses_another_q_grid(tmp_path):
    with h5py.File(tmp_path / "s.h5", "w") as f:
        f["q/centers"] = np.linspace(0.1, 1.0, 10)
        f["frames/status"] = np.zeros(3, np.uint8)
        f["frames/normalization"] = np.ones((3, 10))
    with pytest.raises(QGridMismatch):
        read_npix_per_bin(tmp_path / "s.h5", np.linspace(0.1, 1.0, 11))
    np.testing.assert_allclose(
        read_npix_per_bin(tmp_path / "s.h5", np.linspace(0.1, 1.0, 10)), 1.0
    )


class _FakeRaw:
    """Just enough of an EXtra-data run for `crl_state`."""

    control_sources = {"SA2_XTD1_CRL/MDL/CRL", "MID_XTD6_CRL/MDL/CRL"}

    def __init__(self, inserted):
        self.inserted = inserted

    def __getitem__(self, key):
        source, k = key
        lens = int(k.split("Lens")[1].split(".")[0])
        values = self.inserted.get((source, lens), [0])

        class _K:
            def ndarray(self_inner):
                return np.asarray(values)

        return _K()


def test_crl_state_reads_lens_insertion_and_marks_movement():
    xtd1, xtd6 = "SA2_XTD1_CRL/MDL/CRL", "MID_XTD6_CRL/MDL/CRL"
    raw = _FakeRaw({(xtd1, 3): [1, 1], (xtd1, 5): [1], (xtd6, 2): [0, 1]})
    assert crl_state(raw) == "XTD1:0010100000 XTD6:0~00000000"
