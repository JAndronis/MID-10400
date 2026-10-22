"""A weighted per-ring KWW fit of a pooled g2, for comparing droplets at a given t/t*.

Each ring's baseline-subtracted ``c2`` is fitted with ``β exp(−2 (Γτ)^α) + b``
(errors ``sigma``), β fixed at the ring's reference contrast
(`analysis.triage.xpcs.REFERENCE_BETA`, the τ → 0 intercept of r450 and r493-496)
and Γ, α and b free and bounded. With β free the fit degenerates wherever part of
the decay lies below the first lag: β, Γ and α trade off and α runs to its floor.
Fixed, contrast lost before the first lag reads as a fast Γ. ``beta_mode="bounded"``
fits β too, in [0, the reference's p84]: β is then the amplitude of the decay inside
the lag window and 1 − β/β_ref the fraction of the contrast gone before the first lag.

τ = 1/Γ is the KWW relaxation time, not the mean one, τ Γ(1/α)/α. This is not
`analysis.xpcs.figures.fit_g2`, the DAMNIT column's: that is one unweighted joint fit
over all rings, so one ring that does not converge fails them all.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from analysis.triage.xpcs import fixed_beta

__all__ = [
    "AT_BOUND",
    "BASE_BOUNDS",
    "GAMMA_BOUNDS",
    "MIN_POINTS",
    "STRETCH_BOUNDS",
    "RingFits",
    "fit_rings",
]

#: Γ in s⁻¹: 1/Γ from 10 ms down to 10 ns, two decades past either end of the
#: 0.44-135 µs lag window.
GAMMA_BOUNDS = (1e2, 1e8)
#: α, from strongly stretched (0.2, a generous floor) to compressed up to 2, the
#: ballistic limit seen in jammed and gelling soft matter.
STRETCH_BOUNDS = (0.2, 2.0)
#: The offset b of the baseline-subtracted c2.
BASE_BOUNDS = (-1.0, 1.0)
#: A fitted parameter this close to a bound, as a fraction of the bound interval
#: (log Γ, α, b, β), marks the ring unresolved: the bound, not the data, set it.
AT_BOUND = 0.01
MIN_POINTS = 6

BetaMode = Literal["fixed", "bounded"]


def _kww(t, log_gamma, stretch, base, beta):
    return beta * np.exp(-2 * (np.exp(log_gamma) * t) ** stretch) + base


@dataclass
class RingFits:
    """Per-ring parameters and their errors, NaN where a ring was not fitted.

    ``resolved``: no parameter within AT_BOUND of a bound, 1/Γ inside the lags and
    σ_Γ < Γ. ``visible``: the measured c2 at the first lag over β_ref, the fraction
    of the contrast left there. ``failed``: ring index → why it was not fitted.
    """

    beta_mode: BetaMode
    q: np.ndarray  # Å⁻¹, as the results file's roi_q
    tau: np.ndarray
    gamma: np.ndarray
    err_gamma: np.ndarray
    stretch: np.ndarray
    err_stretch: np.ndarray
    base: np.ndarray
    beta: np.ndarray
    err_beta: np.ndarray
    beta_ref: np.ndarray
    visible: np.ndarray
    err_visible: np.ndarray
    at_bound: np.ndarray
    curve: np.ndarray
    failed: dict[int, str] = field(default_factory=dict)

    @property
    def q_nm(self) -> np.ndarray:
        return 10 * self.q

    @property
    def resolved(self) -> np.ndarray:
        relax = 1 / self.gamma
        with np.errstate(invalid="ignore"):
            return (
                ~self.at_bound
                & (relax >= self.tau.min())
                & (relax <= self.tau.max())
                & (self.err_gamma < self.gamma)
            )

    def summary(self) -> dict:
        """The fit as JSON, with what is derived from the resolved rings: D = Γ/q²
        and the mean α, each weighted by its inverse variance; the slope n of
        Γ ∝ qⁿ (three resolved rings or more), weighted by Γ's relative errors; and
        with β bounded, ``lost`` = 1 − β/β_ref."""
        resolved = self.resolved
        record = {
            "q_nm": np.round(self.q_nm, 4).tolist(),
            "gamma_per_s": self.gamma.tolist(),
            "err_gamma_per_s": self.err_gamma.tolist(),
            "tau_s": (1 / self.gamma).tolist(),
            "beta": self.beta.tolist(),
            "err_beta": self.err_beta.tolist(),
            "beta_ref": self.beta_ref.tolist(),
            "stretch": self.stretch.tolist(),
            "err_stretch": self.err_stretch.tolist(),
            "base": self.base.tolist(),
            "visible": self.visible.tolist(),
            "err_visible": self.err_visible.tolist(),
            "at_bound": self.at_bound.tolist(),
            "resolved": resolved.tolist(),
            "n_resolved": int(resolved.sum()),
        }
        if self.failed:
            record["ring_fit_failed"] = {str(k): v for k, v in self.failed.items()}
        if resolved.any():
            q2 = self.q_nm[resolved] ** 2
            # nm²/s → µm²/s
            d = self.gamma[resolved] / q2 * 1e-6
            w = (self.err_gamma[resolved] / q2 * 1e-6) ** -2
            w_alpha = self.err_stretch[resolved] ** -2
            record |= {
                "D_um2_per_s": float(np.average(d, weights=w)),
                "D_err_um2_per_s": float(w.sum() ** -0.5),
                "stretch_mean": float(
                    np.average(self.stretch[resolved], weights=w_alpha)
                ),
                "stretch_mean_err": float(w_alpha.sum() ** -0.5),
            }
        if self.beta_mode == "bounded":
            lost = 1 - self.beta / self.beta_ref
            record["lost"] = lost.tolist()
            if resolved.any():
                record["lost_mean"] = float(lost[resolved].mean())
        if resolved.sum() >= 3:
            (slope, _), cov = np.polyfit(
                np.log(self.q_nm[resolved]),
                np.log(self.gamma[resolved]),
                1,
                w=self.gamma[resolved] / self.err_gamma[resolved],
                cov="unscaled",
            )
            record |= {
                "exponent_n": float(slope),
                "exponent_err": float(cov[0, 0] ** 0.5),
            }
        return record


def fit_rings(g2: xr.Dataset, beta_mode: BetaMode = "fixed") -> RingFits:
    """Fit every ring of `g2` (``c2`` and ``sigma`` over ``roi_q`` in Å⁻¹ and ``tau``
    in s). A ring without a reference contrast, with fewer than MIN_POINTS finite
    points, or whose fit raises is left NaN and listed in ``failed``."""
    if beta_mode not in ("fixed", "bounded"):
        raise ValueError(f"beta_mode must be 'fixed' or 'bounded', got {beta_mode!r}")
    q = g2.roi_q.values
    tau = g2.tau.values
    n = q.size

    def nan(*shape):
        return np.full(shape or (n,), np.nan)

    fit = RingFits(
        beta_mode=beta_mode,
        q=q,
        tau=tau,
        gamma=nan(),
        err_gamma=nan(),
        stretch=nan(),
        err_stretch=nan(),
        base=nan(),
        beta=nan(),
        err_beta=nan(),
        beta_ref=nan(),
        visible=nan(),
        err_visible=nan(),
        at_bound=np.zeros(n, dtype=bool),
        curve=nan(n, tau.size),
    )
    # Γ is fitted as log Γ, so that its bound check is relative
    low = (np.log(GAMMA_BOUNDS[0]), STRETCH_BOUNDS[0], BASE_BOUNDS[0])
    high = (np.log(GAMMA_BOUNDS[1]), STRETCH_BOUNDS[1], BASE_BOUNDS[1])
    for r, ref in fixed_beta(q).items():
        fit.beta_ref[r] = ref.value
        if beta_mode == "fixed":
            fit.beta[r], fit.err_beta[r] = ref.value, (ref.p84 - ref.p16) / 2
        y, e = g2.c2.values[r], g2.sigma.values[r]
        ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
        if not np.isfinite(ref.value):
            fit.failed[r] = "no reference contrast for this ring"
            continue
        if ok.sum() < MIN_POINTS:
            fit.failed[r] = f"fewer than {MIN_POINTS} points"
            continue
        fit.visible[r], fit.err_visible[r] = y[0] / ref.value, e[0] / ref.value
        p0 = (-np.log(float(np.median(tau))), 1.0, 0.0)
        if beta_mode == "fixed":
            model = functools.partial(_kww, beta=ref.value)
            lo, hi = low, high
        else:
            model = _kww
            lo, hi = (*low, 0.0), (*high, ref.p84)
            p0 = (*p0, ref.value)
        try:
            popt, pcov = curve_fit(
                model,
                tau[ok],
                y[ok],
                p0=p0,
                sigma=e[ok],
                absolute_sigma=True,
                bounds=(lo, hi),
                maxfev=20000,
            )
        except (RuntimeError, ValueError) as error:
            fit.failed[r] = str(error)
            continue
        err = np.sqrt(np.diag(pcov))
        span = np.subtract(hi, lo)
        fit.at_bound[r] = bool(
            np.any((popt - lo) < AT_BOUND * span)
            or np.any((hi - popt) < AT_BOUND * span)
        )
        fit.gamma[r], fit.stretch[r], fit.base[r] = np.exp(popt[0]), popt[1], popt[2]
        fit.err_gamma[r], fit.err_stretch[r] = fit.gamma[r] * err[0], err[1]
        if beta_mode == "bounded":
            fit.beta[r], fit.err_beta[r] = popt[3], err[3]
        fit.curve[r] = model(tau, *popt)
    return fit
