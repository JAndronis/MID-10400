"""The per-ring KWW fit, on g2 curves made from the model itself."""

import numpy as np
import pytest
import xarray as xr

from analysis.xpcs.kww import GAMMA_BOUNDS, fit_rings
from analysis.xpcs.snr import REFERENCE_BETA

RINGS_A = np.array([0.02475, 0.03375, 0.04275, 0.05175])
TAU = 4.44e-7 * np.arange(1, 301)


def g2_from(gamma, stretch=1.0, base=0.0, beta=None, noise=1e-4, seed=0):
    """A pooled g2 whose c2 follows the KWW model exactly, plus Gaussian noise."""
    beta = np.array([REFERENCE_BETA[q][0] for q in RINGS_A]) if beta is None else beta
    gamma = np.broadcast_to(gamma, RINGS_A.shape)
    c2 = beta[:, None] * np.exp(-2 * (gamma[:, None] * TAU) ** stretch) + base
    c2 = c2 + np.random.default_rng(seed).normal(0, noise, c2.shape)
    return xr.Dataset(
        {
            "c2": (("roi_q", "tau"), c2),
            "sigma": (("roi_q", "tau"), np.full(c2.shape, noise)),
        },
        coords={"roi_q": RINGS_A, "tau": TAU},
    )


def test_recovers_rates_exponent_and_offset_with_beta_fixed():
    gamma = np.array([2e4, 4e4, 6e4, 8e4])
    fit = fit_rings(g2_from(gamma, stretch=0.8, base=2e-3))
    np.testing.assert_allclose(fit.gamma, gamma, rtol=0.03)
    np.testing.assert_allclose(fit.stretch, 0.8, atol=0.03)
    np.testing.assert_allclose(fit.base, 2e-3, atol=3e-4)
    assert fit.resolved.all()
    assert not fit.failed


def test_diffusion_gives_q_squared_and_its_coefficient():
    q_nm = 10 * RINGS_A
    d_um2_per_s = 0.5
    fit = fit_rings(g2_from(d_um2_per_s * 1e6 * q_nm**2, noise=5e-5))
    record = fit.summary()
    assert record["exponent_n"] == pytest.approx(2, abs=0.05)
    assert record["D_um2_per_s"] == pytest.approx(d_um2_per_s, rel=0.03)
    assert record["n_resolved"] == RINGS_A.size


def test_bounded_beta_measures_the_contrast_lost_before_the_first_lag():
    beta_ref = np.array([REFERENCE_BETA[q][0] for q in RINGS_A])
    fit = fit_rings(g2_from(3e4, beta=0.6 * beta_ref), beta_mode="bounded")
    np.testing.assert_allclose(fit.beta, 0.6 * beta_ref, rtol=0.05)
    np.testing.assert_allclose(fit.summary()["lost"], 0.4, atol=0.03)


def test_a_decay_faster_than_the_first_lag_is_not_resolved():
    fit = fit_rings(g2_from(GAMMA_BOUNDS[1]))
    assert not fit.resolved.any()
    assert "exponent_n" not in fit.summary()


def test_a_ring_without_reference_or_points_is_left_out_not_raised():
    g2 = g2_from(3e4)
    g2 = g2.assign_coords(roi_q=[0.5, *RINGS_A[1:]])
    g2["c2"][1, 3:] = np.nan
    fit = fit_rings(g2)
    assert set(fit.failed) == {0, 1}
    assert np.isnan(fit.gamma[:2]).all()
    assert np.isfinite(fit.gamma[2:]).all()
