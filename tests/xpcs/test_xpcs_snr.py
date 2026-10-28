"""XPCS SNR against synthetic runs with known truth.

The synthetic TTCF carries contrast β, rate Γ and, per pair, exactly the noise the
analytical formula predicts. So the jackknife must find that noise (ρ ≈ 1), the fit
must find β and Γ, and the SNR must be the formula evaluated on the run's own k̄,
N_pix, φ and train count.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from triage_fakes import FIRST_TRAIN, hand_run, write_hit_results

from analysis.xpcs.snr import (
    ES_REFERENCE_BETA,
    REFERENCE_BETA_RUNS,
    Beta,
    beta_for,
    fit_decay,
    fixed_beta,
    measure_run,
    slice_index,
    snr,
    summarise,
)


def test_fit_recovers_contrast_and_rate():
    tau = np.arange(1, 60) * 444e-9  # one 2.25 MHz pulse period per lag
    rate = 1.0 / (2 * 8 * tau[0])
    g2 = 0.03 * np.exp(-2 * rate * tau) + 0.001
    fit = fit_decay(tau, g2, np.full_like(g2, 1e-4))
    assert fit.converged
    assert fit.amplitude == pytest.approx(0.03, rel=1e-3)
    assert fit.rate == pytest.approx(rate, rel=1e-3)


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


# ── the hit-train results file ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hit_results(tmp_path_factory):
    return write_hit_results(tmp_path_factory.mktemp("hits") / "xpcs_hits.h5")


@pytest.fixture(scope="module")
def measured(hit_results):
    """`measure_run` of `hit_results`, with five jackknife blocks to keep it cheap."""
    return measure_run(423, hit_results, n_blocks=5)


def test_measure_run_reduces_the_results_file(hit_results, measured):
    h, m = hit_results, measured
    with_baseline = h.train_ids[h.usable & h.has_off]
    assert m.n_usable == with_baseline.size > 0
    assert m.n_frames == 14
    # k̄ is photons per pixel per frame, the density times the mean photon count
    np.testing.assert_allclose(
        m.kbar[:4], 2 * np.array([0.2, 0.1, 0.01, 0.005]), rtol=0.2
    )
    # pulse ids 200, 202, …: two slots of the 4.5 MHz pulse clock, 444 ns
    assert m.tau[0] == pytest.approx(444e-9, rel=0.005)
    assert m.tau.size == 13


def test_a_ring_below_the_kbar_cut_loses_its_pairs(hit_results, measured):
    m = measured
    assert m.phi[0] > 0.98  # k̄ 0.4: only an unlit frame drops a pair
    assert m.phi[-1] < 0.01  # k̄ 2e-4: nearly every pair is dropped
    everything = measure_run(423, hit_results, kbar_min=10.0, n_blocks=5)
    assert (everything.phi == 0).all()
    assert np.isnan(everything.g2).all()


def test_select_narrows_the_trains_and_unknown_ones_are_refused(hit_results, measured):
    some = hit_results.train_ids[hit_results.has_off][:20]
    assert measure_run(423, hit_results, select=some, n_blocks=5).n_usable == 20
    assert measured.n_usable > 20
    other = np.array([FIRST_TRAIN + 10_000], dtype=np.uint64)
    assert measure_run(423, hit_results, select=other).n_usable == 0


def test_the_slices_are_anchored_to_the_first_usable_hit_or_a_given_span(
    hit_results, measured
):
    m = measured
    assert [s["slice"] for s in m.slices] == [0]
    assert m.slices[0]["n_usable"] == m.n_usable
    first = int(hit_results.train_ids[hit_results.has_off][0])
    span = (first - 300, first + 500)
    shifted = measure_run(423, hit_results, span=span, n_blocks=5)
    assert [s["slice"] for s in shifted.slices] == [1]


@pytest.fixture(scope="module")
def poisson_hits(tmp_path_factory):
    """Speckle-free Poisson photons: four rings of about 300 pixels, 100 trains."""
    return write_hit_results(
        tmp_path_factory.mktemp("poisson") / "xpcs_hits.h5",
        n_trains=100,
        n_env=1200,
        densities=(0.2, 0.1, 0.05, 0.03),
        poisson=True,
        seed=0,
    )


def test_measure_run_noise_is_the_analytical_noise_on_poisson_photons(poisson_hits):
    """ρ ≈ 1 pins √2 · 1/(k̄ √(N_pix (F − 1) φ n)) against what the jackknife sees."""
    m = measure_run(423, poisson_hits, n_blocks=20)
    assert m.n_usable > 90
    # 20 blocks give each ring's σ to about 16 %, the mean over four to about 8 %
    assert np.all((m.rho > 0.7) & (m.rho < 1.4))
    assert m.rho.mean() == pytest.approx(1.0, abs=0.12)


def test_snr_is_the_analytical_formula_on_hand_picked_numbers():
    # β k̄ √(N_pix (F − 1) φ n) = 0.02 · 0.05 · √(2000 · 100 · 0.5 · 40) = 2, over √2
    assert snr(0.02, 0.05, 2000, 101, 0.5, 40) == pytest.approx(np.sqrt(2.0))


def test_the_verdict_takes_the_best_plain_roi_and_the_time_it_needs():
    nan = float("nan")
    beta = {0: Beta(0.02, 1, nan, nan, ()), 1: Beta(0.5, 1, nan, nan, ())}
    s = summarise(hand_run(), beta)
    # the plain ROI's SNR is √2 (above); the Bragg ROI's 25√2 does not count
    assert s["snr_run"] == {0.02875: pytest.approx(np.sqrt(2.0))}
    assert s["best_snr"] == pytest.approx(np.sqrt(2.0)) and s["best_roi_q_A"] == 0.02875
    assert s["extractable"] is False
    # SNR grows as √t: 3 needs (3/√2)² = 4.5 slices of 30 s
    assert s["t_needed_s"] == pytest.approx(135.0)


def test_an_unknown_beta_is_not_a_verdict():
    nan = float("nan")
    s = summarise(hand_run(), {r: Beta(nan, 0, nan, nan, ()) for r in (0, 1)})
    assert s["extractable"] is None
    assert s["n_extractable_slices"] == 0
    assert np.isnan(s["best_snr"])


def test_n_pix_is_a_typical_trains_and_heavily_masked_trains_do_not_move_it(
    tmp_path,
):
    """Every sixth train loses 90 % of its pixels; N_pix stays the unmasked count."""
    h = write_hit_results(
        tmp_path / "xpcs_hits.h5", masked={i: 0.9 for i in range(0, 60, 6)}
    )
    with h5py.File(h.path, "r") as f:
        ring_size = np.bincount(f["rings/label"][:], minlength=h.n_rings)
    m = measure_run(423, h, n_blocks=5)
    # a pixel survives a train unless one of its 14 frames flags it (2 % each)
    np.testing.assert_allclose(m.npix, ring_size * 0.98**14, rtol=0.08)


def test_beta_is_looked_up_by_roi_centre():
    nan = float("nan")
    near, far = Beta(0.03, 3, 0.02, 0.04, (1,)), Beta(0.01, 2, nan, nan, (2,))
    beta = beta_for(np.array([0.072, 0.0505, 0.07200000000000001, 0.04]),
                    {0.04: near, 0.072: far})  # fmt: skip
    assert (beta[0], beta[2], beta[3]) == (far, far, near)
    assert np.isnan(beta[1].value) and beta[1].n_eligible == 0


def test_the_fixed_beta_is_the_reference_table_from_every_reference_run():
    beta = fixed_beta(np.array([0.04, 0.3]))
    value, p16, p84 = ES_REFERENCE_BETA[0.04]
    assert (beta[0].value, beta[0].p16, beta[0].p84) == (value, p16, p84)
    assert beta[0].runs == REFERENCE_BETA_RUNS
    assert beta[0].n_eligible == len(REFERENCE_BETA_RUNS)
    assert np.isnan(beta[1].value)
