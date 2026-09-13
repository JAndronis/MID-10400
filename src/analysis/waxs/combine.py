"""Putting the two detectors on one scale (WAXS context file §6 O5).

After masking, jf1 populates ~11.5–23.7 nm⁻¹ and jf2 ~9.8–18.5, so they share a
wide overlap. This module fits a single multiplicative factor for one detector
against the other **over that overlap**, and merges the two into one curve.

**What the factor absorbs, and what it cannot.** O5 warned that pooling the two
onto a common grid needs both `N` arrays on the same absolute scale, which is
untested: different solid-angle coverage, different masks, two PONIs. A *fitted*
factor sidesteps that by construction — whatever the relative normalisation is,
the fit absorbs it. What it cannot absorb is a difference in *shape*, which is
what :attr:`OverlapScaling.reduced_chi2` and :attr:`OverlapScaling.residual_rms`
report. A factor is a result; a factor plus a bad χ² is a warning about the
geometry.

**When that cross-check is blind, measured.** A relative error in the two q axes
is *degenerate* with a scale factor whenever the overlap is featureless. For a
power law, `I(1.05·q) = 1.05⁻² · I(q)` exactly, so a 5 % q error is
indistinguishable from a 9.3 % scale error — and it shows: on a smooth
`100/q² + 0.5`, a 5 % q shift leaves χ²ᵣ at 0.22, which no one would look at
twice. Add a Bragg-like peak to the overlap and the same 5 % shift gives χ²ᵣ
776, and even 1 % gives 77.

So: **this is a real cross-check on the two PONIs only on a run whose overlap
carries a feature.** On r0426 (crystallised) it bites; on r0423 (no Bragg
reflections) a good χ² here says the two detectors are mutually consistent in
*shape*, not that either q axis is right. Do not read more into it than that.

Everything here works on stored sums or on pooled curves — nothing reprocesses
frames — so it is post hoc in the sense of the AGIPD §9 contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "NoOverlap",
    "OverlapScaling",
    "combine_curves",
    "combine_files",
    "mean_curve",
    "scale_to_overlap",
]

#: How the scale factor is estimated.
METHODS = ("wls", "median")


class NoOverlap(ValueError):
    """The two curves share no q range with data in both."""


@dataclass(frozen=True, slots=True)
class OverlapScaling:
    """One multiplicative factor, and how well it actually worked."""

    #: ``I_other * factor`` is on the reference's scale.
    factor: float
    #: Standard error on the factor, from the weighted fit; ``nan`` for the
    #: median method, which has no closed form here.
    factor_error: float
    q_low: float
    q_high: float
    n_bins: int
    method: str
    #: Reduced χ² of ``I_ref − factor·I_other`` over the overlap. Near 1 means
    #: the two agree within their errors; ≫ 1 means they do not lie on top of
    #: each other and no factor will make them.
    reduced_chi2: float
    #: Fractional RMS residual, which is readable without trusting the errors.
    residual_rms: float
    #: Slope of the *fractional* residual against q, per nm⁻¹, from an
    #: unweighted straight-line fit. This is what separates the two things a
    #: bad χ² can mean. A leftover **normalisation** difference is flat in q and
    #: the factor absorbs it, leaving slope ≈ 0. Anything **q-dependent** — a
    #: relative error between the two q axes, or a correction applied to neither
    #: detector that varies with scattering angle — tilts the residual, and no
    #: single factor can take that out. Unweighted on purpose: when χ² is this
    #: far above 1 the errors are no longer what limits the comparison, so
    #: weighting by them would just re-import the assumption under test.
    residual_slope_per_nm: float
    #: The fitted fractional residual at each end of the overlap, which is the
    #: slope in units anyone can act on.
    residual_at_q_low: float
    residual_at_q_high: float

    def summary(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "factor_error": self.factor_error,
            "q_low": self.q_low,
            "q_high": self.q_high,
            "n_bins": self.n_bins,
            "method": self.method,
            "reduced_chi2": self.reduced_chi2,
            "residual_rms": self.residual_rms,
            "residual_slope_per_nm": self.residual_slope_per_nm,
            "residual_at_q_low": self.residual_at_q_low,
            "residual_at_q_high": self.residual_at_q_high,
        }

    @property
    def sigma_over_intensity(self) -> float:
        """Roughly what fractional error the χ² implies the inputs claimed.

        ``residual_rms / sqrt(reduced_chi2)``. Useful for telling "the curves
        disagree" from "the error bars are too small": on r0423 the residual is
        0.74 % against a claimed 0.10 %, so the disagreement is real and
        systematic but sub-percent, and χ² alone would have made it sound
        catastrophic.
        """
        if not np.isfinite(self.reduced_chi2) or self.reduced_chi2 <= 0:
            return float("nan")
        return self.residual_rms / np.sqrt(self.reduced_chi2)


def mean_curve(source: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(q, I, σ)`` from a per-cell grid or a pooled dataset.

    Accepts what the pass actually hands around:

    * the ``(trainId, cellId, q)`` DataArray the DAMNIT variable returns —
      averaged over every slot that holds a frame, with σ from the scatter
      between those slots;
    * the ``pooled_per_train`` Dataset — averaged over trains that have frames,
      with σ propagated from the stored per-train σ.

    Slots with no frame are stored as zeros, never NaN, so they are selected out
    by ``n_frames`` rather than by ``isnan`` (AGIPD context file §3 rule 7).
    """
    import xarray as xr

    q = np.asarray(source["q"].values, dtype=np.float64)

    if isinstance(source, xr.Dataset) and "intensity" in source:
        present = np.asarray(source["n_frames"].values) > 0
        intensity = np.asarray(source["intensity"].values, dtype=np.float64)
        sigma = np.asarray(source["sigma"].values, dtype=np.float64)
        if not present.any():
            raise ValueError("no train in this dataset has an integrated frame")
        mean = intensity[present].mean(axis=0)
        # Independent trains, so the mean's variance is the mean of the
        # variances divided by the count.
        count = int(present.sum())
        error = np.sqrt((sigma[present] ** 2).sum(axis=0)) / count
        return q, mean, error

    present = np.asarray(source["n_frames"].values) > 0
    values = np.asarray(source.values, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])[present.reshape(-1)]
    if flat.size == 0:
        raise ValueError("no slot in this grid holds an integrated frame")
    mean = flat.mean(axis=0)
    # No stored σ on the per-cell grid, so the scatter between slots is the
    # honest estimate; with one slot there is none and the weights fall back
    # to uniform in the fit.
    error = (
        flat.std(axis=0, ddof=1) / np.sqrt(flat.shape[0])
        if flat.shape[0] > 1
        else np.full(mean.shape, np.nan)
    )
    return q, mean, error


def _overlap(
    q_ref: np.ndarray,
    i_ref: np.ndarray,
    q_other: np.ndarray,
    i_other: np.ndarray,
    sigma_ref: np.ndarray | None,
    sigma_other: np.ndarray | None,
    min_bins: int,
) -> tuple[np.ndarray, ...]:
    """Restrict to the shared q range and put ``other`` on the reference's bins.

    Only ``other`` is interpolated; the reference keeps its own bins, so the
    result is reported on a real q axis rather than an invented one. The
    variance is interpolated too, which ignores the correlation interpolation
    introduces — acceptable for a weight, and the reason ``residual_rms`` is
    reported alongside ``reduced_chi2``.
    """
    usable_ref = np.isfinite(i_ref) & (i_ref > 0)
    usable_other = np.isfinite(i_other) & (i_other > 0)
    if not usable_ref.any() or not usable_other.any():
        raise NoOverlap("one of the curves has no positive, finite intensity")

    low = max(q_ref[usable_ref].min(), q_other[usable_other].min())
    high = min(q_ref[usable_ref].max(), q_other[usable_other].max())
    if not (high > low):
        raise NoOverlap(
            f"the curves do not overlap: reference spans "
            f"{q_ref[usable_ref].min():.3g}–{q_ref[usable_ref].max():.3g} and the "
            f"other {q_other[usable_other].min():.3g}–"
            f"{q_other[usable_other].max():.3g}"
        )

    keep = usable_ref & (q_ref >= low) & (q_ref <= high)
    if int(keep.sum()) < min_bins:
        raise NoOverlap(
            f"only {int(keep.sum())} reference bins lie in the overlap "
            f"{low:.3g}–{high:.3g}, fewer than the {min_bins} required"
        )

    order = np.argsort(q_other[usable_other])
    q_src = q_other[usable_other][order]
    interpolated = np.interp(q_ref[keep], q_src, i_other[usable_other][order])

    def lift(sigma, mask):
        if sigma is None:
            return None
        values = np.asarray(sigma, dtype=np.float64)
        if not np.isfinite(values).any():
            return None
        return values[mask]

    sigma_r = lift(sigma_ref, keep)
    sigma_o = lift(sigma_other, usable_other)
    if sigma_o is not None:
        sigma_o = np.sqrt(np.interp(q_ref[keep], q_src, (sigma_o[order]) ** 2))
    return q_ref[keep], i_ref[keep], interpolated, sigma_r, sigma_o


def scale_to_overlap(
    q_ref: np.ndarray,
    i_ref: np.ndarray,
    q_other: np.ndarray,
    i_other: np.ndarray,
    *,
    sigma_ref: np.ndarray | None = None,
    sigma_other: np.ndarray | None = None,
    method: str = "wls",
    min_bins: int = 8,
) -> OverlapScaling:
    """Fit ``a`` such that ``a·I_other ≈ I_ref`` over the shared q range.

    :param method: ``"wls"`` is a weighted least-squares fit through the origin,
        which is what a pure scale factor is; ``"median"`` takes the median of
        the per-bin ratios, which ignores the errors and is the one to reach for
        when a few bins are wild.
    :param min_bins: refuse a fit resting on fewer reference bins than this. A
        factor from three bins is a number, not a measurement.

    :raises NoOverlap: the curves share no usable q range, or too little of one.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    q, reference, other, sigma_r, sigma_o = _overlap(
        np.asarray(q_ref, dtype=np.float64),
        np.asarray(i_ref, dtype=np.float64),
        np.asarray(q_other, dtype=np.float64),
        np.asarray(i_other, dtype=np.float64),
        sigma_ref,
        sigma_other,
        min_bins,
    )

    if method == "median":
        factor = float(np.median(reference / other))
        factor_error = float("nan")
    else:
        # Start unweighted, then weight by the combined error at that estimate.
        # One refinement is enough: the weights depend on `a` only through the
        # other curve's contribution, which is second order here.
        first = float((reference * other).sum() / (other**2).sum())
        variance = np.zeros_like(reference)
        if sigma_r is not None:
            variance += sigma_r**2
        if sigma_o is not None:
            variance += (first * sigma_o) ** 2
        if np.all(variance > 0):
            weight = 1.0 / variance
        else:
            weight = np.ones_like(reference)
        denominator = float((weight * other**2).sum())
        factor = float((weight * reference * other).sum() / denominator)
        factor_error = float(np.sqrt(1.0 / denominator))

    residual = reference - factor * other
    variance = np.zeros_like(reference)
    if sigma_r is not None:
        variance += sigma_r**2
    if sigma_o is not None:
        variance += (factor * sigma_o) ** 2
    degrees = max(residual.size - 1, 1)
    reduced_chi2 = (
        float((residual**2 / variance).sum() / degrees)
        if np.all(variance > 0)
        else float("nan")
    )

    fractional = residual / np.where(reference != 0, reference, np.nan)
    usable = np.isfinite(fractional)
    if usable.sum() >= 2:
        slope, intercept = np.polyfit(q[usable], fractional[usable], 1)
    else:
        slope = intercept = float("nan")

    return OverlapScaling(
        factor=factor,
        factor_error=factor_error,
        q_low=float(q.min()),
        q_high=float(q.max()),
        n_bins=int(q.size),
        method=method,
        reduced_chi2=reduced_chi2,
        residual_rms=float(np.sqrt((residual**2).mean()) / np.abs(reference).mean()),
        residual_slope_per_nm=float(slope),
        residual_at_q_low=float(slope * q.min() + intercept),
        residual_at_q_high=float(slope * q.max() + intercept),
    )


def combine_curves(
    q_ref: np.ndarray,
    i_ref: np.ndarray,
    q_other: np.ndarray,
    i_other: np.ndarray,
    *,
    sigma_ref: np.ndarray | None = None,
    sigma_other: np.ndarray | None = None,
    method: str = "wls",
    min_bins: int = 8,
) -> tuple[Any, OverlapScaling]:
    """One curve from two, with ``other`` scaled onto the reference.

    Returns a Dataset over a q axis that is the sorted union of both, carrying
    ``intensity``, ``sigma`` and ``source`` (0 = reference only, 1 = scaled
    other only, 2 = both). Where the two overlap they are combined by inverse
    variance if both carry errors, and averaged otherwise.

    The reference keeps its own values untouched: only the other detector moves.
    """
    import xarray as xr

    scaling = scale_to_overlap(
        q_ref,
        i_ref,
        q_other,
        i_other,
        sigma_ref=sigma_ref,
        sigma_other=sigma_other,
        method=method,
        min_bins=min_bins,
    )

    q_ref = np.asarray(q_ref, dtype=np.float64)
    q_other = np.asarray(q_other, dtype=np.float64)
    i_ref = np.asarray(i_ref, dtype=np.float64)
    scaled = np.asarray(i_other, dtype=np.float64) * scaling.factor
    s_ref = None if sigma_ref is None else np.asarray(sigma_ref, dtype=np.float64)
    s_other = (
        None
        if sigma_other is None
        else np.asarray(sigma_other, dtype=np.float64) * scaling.factor
    )

    grid = np.unique(np.concatenate([q_ref, q_other]))

    def onto(q, values, sigma, lo, hi):
        """Interpolate a curve onto the union grid, blank outside its range."""
        usable = np.isfinite(values) & (values > 0)
        inside = (grid >= lo) & (grid <= hi)
        out = np.zeros(grid.size)
        err = np.full(grid.size, np.nan)
        order = np.argsort(q[usable])
        out[inside] = np.interp(grid[inside], q[usable][order], values[usable][order])
        if sigma is not None and np.isfinite(sigma[usable]).any():
            err[inside] = np.sqrt(
                np.interp(grid[inside], q[usable][order], sigma[usable][order] ** 2)
            )
        return out, err, inside

    usable_ref = np.isfinite(i_ref) & (i_ref > 0)
    usable_other = np.isfinite(scaled) & (scaled > 0)
    a_values, a_error, a_inside = onto(
        q_ref, i_ref, s_ref, q_ref[usable_ref].min(), q_ref[usable_ref].max()
    )
    b_values, b_error, b_inside = onto(
        q_other,
        scaled,
        s_other,
        q_other[usable_other].min(),
        q_other[usable_other].max(),
    )

    intensity = np.zeros(grid.size)
    sigma = np.full(grid.size, np.nan)
    source = np.zeros(grid.size, dtype=np.uint8)

    only_a = a_inside & ~b_inside
    only_b = b_inside & ~a_inside
    both = a_inside & b_inside

    intensity[only_a] = a_values[only_a]
    sigma[only_a] = a_error[only_a]
    source[only_a] = 0

    intensity[only_b] = b_values[only_b]
    sigma[only_b] = b_error[only_b]
    source[only_b] = 1

    source[both] = 2

    weighted = both & np.isfinite(a_error) & np.isfinite(b_error)
    weighted &= (a_error > 0) & (b_error > 0)
    plain = both & ~weighted
    intensity[plain] = 0.5 * (a_values[plain] + b_values[plain])
    if weighted.any():
        wa = 1.0 / a_error[weighted] ** 2
        wb = 1.0 / b_error[weighted] ** 2
        intensity[weighted] = (wa * a_values[weighted] + wb * b_values[weighted]) / (
            wa + wb
        )
        sigma[weighted] = 1.0 / np.sqrt(wa + wb)

    combined = xr.Dataset(
        {
            "intensity": (("q",), intensity),
            "sigma": (("q",), sigma),
            "source": (("q",), source),
        },
        coords={"q": grid},
    )
    combined.attrs.update(
        {f"scaling_{key}": value for key, value in scaling.summary().items()}
    )
    return combined, scaling


def combine_files(
    reference_file: str | Path,
    other_file: str | Path,
    *,
    method: str = "wls",
    min_bins: int = 8,
) -> tuple[Any, OverlapScaling]:
    """:func:`combine_curves` straight off two finished output files.

    The stored sums carry real per-bin errors, which the per-cell grid does not,
    so this is the path that gives a meaningful ``reduced_chi2``.
    """
    from analysis.waxs.writer import pooled_per_train

    q_ref, i_ref, s_ref = mean_curve(pooled_per_train(Path(reference_file)))
    q_other, i_other, s_other = mean_curve(pooled_per_train(Path(other_file)))
    return combine_curves(
        q_ref,
        i_ref,
        q_other,
        i_other,
        sigma_ref=s_ref,
        sigma_other=s_other,
        method=method,
        min_bins=min_bins,
    )
