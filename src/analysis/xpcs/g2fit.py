"""extra-speckle's single-exponential fit of g2 and its dispersion relation.

The DAMNIT ``XPCS g2 fit`` column fits this model; `analysis.xpcs.kww` fits a
stretched exponential per ring with β fixed, for comparing droplets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

__all__ = ["G2Fit", "fit_g2"]

#: ``xpcs.fit_dr``'s constants: kB [J/K], 25 °C, viscosity 1 mPa s in N s Å⁻².
_KB, _T_K, _ETA = 1.381e-23, 273.15 + 25, 1e-23


@dataclass(frozen=True, slots=True)
class G2Fit:
    """extra-speckle's single-exponential fit and its dispersion relation.

    ``g2(τ) = β exp(−2 Γ τ) + baseline`` per ring, ``β`` free for every ring
    (``beta='local'``). ``Γ`` in s⁻¹, rings that could not be fitted are NaN.
    ``dr_a`` is the slope of ``Γ = a q²`` in Å² s⁻¹ and ``rh_A`` the
    Stokes-Einstein radius in Å at 25 °C and 1 mPa s, as ``fit_dr`` computes it.
    """

    q: np.ndarray
    tau: np.ndarray
    curve: np.ndarray  # (R, T) fitted g2
    gamma: np.ndarray
    baseline: np.ndarray
    beta: np.ndarray
    err_gamma: np.ndarray
    err_baseline: np.ndarray
    err_beta: np.ndarray
    dr_a: float
    dr_a_err: float
    rh_A: float
    rh_err_A: float


def fit_g2(g2: xr.DataArray, *, min_points: int = 4) -> G2Fit:
    """``extra_speckle.xpcs.fit(g2, model='exp', beta='local', dr=True)``.

    One joint ``scipy.optimize.curve_fit`` over all rings, started at
    ``Γ = 1 / linspace(max τ / 10, min τ × 10)``, baseline at the global
    minimum and ``β = 0.1``, as there. The dispersion fit is ``fit_dr``'s
    ``'weightedaverage'``: ``a = average(Γ/q², weights=err_Γ/q²)`` (the weights
    are the *errors*, as in extra-speckle, not their inverses) and
    ``err_a = std(Γ/q²)``, ``R_H = kB T / (6 π η a) × 1e10`` with
    ``η = 1e-23 N s Å⁻²``.

    One difference is deliberate. extra-speckle fitted every ring it was given,
    after dropping noisy ones itself; a ring here with fewer than
    ``min_points`` finite points (no photons, for instance) is left out of the
    fit and returns NaN, rather than being fitted to nothing.

    :param g2: ``(roi_q, tau)``, e.g. ``hits.g2()["c2"]``.
    :raises ValueError: no ring can be fitted, or the optimiser fails.
    """
    from scipy.optimize import curve_fit

    q_all = g2["roi_q"].values.astype(float)
    tau = g2["tau"].values.astype(float)
    data = g2.values.astype(float)
    fit_rings = np.flatnonzero(np.isfinite(data).sum(axis=1) >= min_points)
    if fit_rings.size == 0:
        raise ValueError("no ring has enough finite g2 points to fit")
    cc = data[fit_rings].T  # (tau, rings), as extra-speckle's ``cc_exp``
    valid = np.isfinite(cc)
    n = fit_rings.size

    def model(p):
        gamma, baseline, beta = np.reshape(p, (3, n))
        return beta[None, :] * np.exp(np.outer(-2 * tau, gamma)) + baseline[None, :]

    p0 = np.concatenate(
        [
            1 / np.linspace(tau.max() / 10, tau.min() * 10, n),
            np.full(n, np.nanmin(cc)),
            0.1 * np.ones(n),
        ]
    )
    try:
        popt, pcov = curve_fit(
            lambda _, *p: model(p)[valid], np.arange(valid.sum()), cc[valid], p0=p0
        )
    except RuntimeError as error:
        raise ValueError(f"the g2 fit did not converge: {error}") from error
    gamma, baseline, beta = np.reshape(popt, (3, n))
    err_gamma, err_baseline, err_beta = np.reshape(np.sqrt(np.diag(pcov)), (3, n))

    q = q_all[fit_rings]
    try:
        a = float(np.average(gamma / q / q, weights=err_gamma / q / q))
    except ZeroDivisionError as error:
        raise ValueError("the fit has no errors to weight its dispersion") from error
    err_a = float(np.std(gamma / q / q))
    radius = _KB * _T_K / 6 / np.pi / _ETA / a * 1e10

    def spread(values):
        out = np.full(q_all.size, np.nan)
        out[fit_rings] = values
        return out

    curve = np.full((q_all.size, tau.size), np.nan)
    curve[fit_rings] = model(popt).T
    return G2Fit(
        q=q_all,
        tau=tau,
        curve=curve,
        gamma=spread(gamma),
        baseline=spread(baseline),
        beta=spread(beta),
        err_gamma=spread(err_gamma),
        err_baseline=spread(err_baseline),
        err_beta=spread(err_beta),
        dr_a=a,
        dr_a_err=err_a,
        rh_A=float(radius),
        rh_err_A=float(radius * err_a / a),
    )
