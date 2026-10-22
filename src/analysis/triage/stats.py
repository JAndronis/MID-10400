"""Small statistics the triage derives its thresholds with.

No threshold in the triage is a constant. Each one is either a multiple-testing cut,
which fixes the expected number of false positives over the whole triage, or the
centre of an empty band in a measured distribution. Both are computed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm, t

__all__ = [
    "EXPECTED_FALSE",
    "MAD_EFFICIENCY",
    "MAD_TO_SIGMA",
    "EmptyBand",
    "empty_band",
    "mad_dof",
    "robust_sd",
    "welch_dof",
    "z_for_tests",
]

#: MAD -> standard deviation for normally distributed data.
MAD_TO_SIGMA = 1.4826
#: Gaussian efficiency of the MAD as a scale estimator (Rousseeuw & Croux 1993,
#: JASA 88:1273): a MAD from n values is as precise as a standard deviation from
#: 0.37 n, which sets the degrees of freedom of a statistic studentised by it.
MAD_EFFICIENCY = 0.37
#: Expected false positives over a whole family of tests. At 0.1, a flag raised
#: anywhere in the triage is very unlikely to be noise.
EXPECTED_FALSE = 0.1


def robust_sd(values: ArrayLike) -> float:
    """1.4826 · MAD of the finite values; NaN for fewer than three."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))))


def mad_dof(n_values: int) -> float:
    """Degrees of freedom of a `robust_sd` from ``n_values`` values."""
    return MAD_EFFICIENCY * max(n_values - 1, 1)


def welch_dof(var_a: float, dof_a: float, var_b: float, dof_b: float) -> float:
    """Welch–Satterthwaite degrees of freedom of ``sqrt(var_a + var_b)``.

    An infinite ``dof`` marks a variance taken as known; it adds nothing to the
    denominator.
    """
    den = sum(v * v / d for v, d in ((var_a, dof_a), (var_b, dof_b)) if np.isfinite(d))
    return float((var_a + var_b) ** 2 / den) if den > 0 else float("inf")


def z_for_tests(
    n_tests: int,
    expected_false: float = EXPECTED_FALSE,
    *,
    two_sided: bool = False,
    dof: float | None = None,
) -> float:
    """The cut that ``expected_false`` of `n_tests` tests of pure noise pass.

    α = expected_false / n_tests, split over both tails for a statistic in absolute
    value. With ``dof`` the statistic is taken as studentised by an estimated
    scale and the cut is Student's t, which is what keeps a noise estimate from a
    few dozen values from inflating the false count; without it, the normal.
    """
    if n_tests < 1:
        raise ValueError(f"need at least one test, got {n_tests}")
    tail = expected_false / n_tests / (2 if two_sided else 1)
    if dof is None or not np.isfinite(dof):
        return float(norm.isf(tail))
    return float(t.isf(tail, dof))


@dataclass(frozen=True, slots=True)
class EmptyBand:
    """The gap between two neighbouring values of a sorted sample."""

    lower: float
    upper: float

    @property
    def centre(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower


def empty_band(
    values: ArrayLike, *, lo: float = -np.inf, hi: float = np.inf
) -> EmptyBand | None:
    """The widest gap between neighbouring finite values inside ``[lo, hi]``.

    Binning-free: the band's edges are two measured values, so the threshold it
    yields (its centre) is reproducible to the last digit. None for fewer than two
    values.
    """
    x = np.asarray(values, dtype=np.float64)
    x = np.sort(x[np.isfinite(x) & (x >= lo) & (x <= hi)])
    if x.size < 2:
        return None
    i = int(np.argmax(np.diff(x)))
    return EmptyBand(float(x[i]), float(x[i + 1]))
