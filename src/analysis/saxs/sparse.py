"""Gather/bincount kernels and the pure per-frame integration.

Kept here rather than in ``worker.py`` so that the per-frame computation stays
a pure function with no I/O and no EXtra-data import, which is what lets the
self-test and the unit tests reach it without a run. ``worker.py`` is the
read-and-loop wrapper around it.

Everything here works on the flattened pixel order module × ss × fs, i.e. a
1-D array of ``NPIX`` entries, and accumulates in float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.saxs.operator import SparseOperator
from analysis.saxs.status import DataCheckFailed, FrameStatus

__all__ = [
    "FrameResult",
    "denominator",
    "frame_data_status",
    "gather",
    "integrate_frame",
]


def gather(
    op: SparseOperator,
    cols: np.ndarray,
    weights: np.ndarray,
    *,
    squared: bool = False,
) -> np.ndarray:
    """Accumulate ``Σ_j c_ij^p · w_j`` over the CSC columns ``cols``.

    :param cols: flattened pixel indices to sum over, one per entry of
        ``weights``.
    :param weights: per-pixel weight, same length as ``cols``.
    :param squared: square the split coefficients (``p = 2``), which turns
        ``Σ c·x`` into the ``Σ c²·x`` variance.
    :returns: float64 array of length ``op.npt``.

    Index arithmetic is done in int64 throughout: ``indptr`` is int32 and the
    per-column offsets would otherwise be at risk of overflow for larger
    detectors.
    """
    cols = np.asarray(cols)
    weights = np.asarray(weights, dtype=np.float64)
    if cols.shape != weights.shape:
        raise ValueError(
            f"cols and weights must have the same shape, "
            f"got {cols.shape} and {weights.shape}"
        )
    if cols.size == 0:
        return np.zeros(op.npt, dtype=np.float64)

    starts = op.indptr[cols].astype(np.int64)
    counts = op.indptr[cols + 1].astype(np.int64) - starts
    total = int(counts.sum())
    if total == 0:
        # Every selected pixel falls outside the radial range: no entries.
        return np.zeros(op.npt, dtype=np.float64)

    offsets = starts - (np.cumsum(counts) - counts)
    idx = np.repeat(offsets, counts) + np.arange(total, dtype=np.int64)
    coeff = op.coef[idx].astype(np.float64)
    if squared:
        coeff *= coeff
    return np.bincount(
        op.bins[idx],
        weights=coeff * np.repeat(weights, counts),
        minlength=op.npt,
    )


def denominator(op: SparseOperator, bad: np.ndarray) -> np.ndarray:
    """``Σ c·Ω`` over the pixels *not* flagged by ``bad``.

    The ``D`` of the per-cell base masks, and :func:`integrate_frame`'s
    ``base_denominator``.
    """
    good = np.flatnonzero(~bad)
    return gather(op, good, op.omega[good])


def frame_data_status(x: np.ndarray) -> FrameStatus:
    """Classify a frame's counts as ``OK`` or ``DATA_CHECK_FAILED``.

    The ledger-facing counterpart of the check inside :func:`integrate_frame`,
    for callers that need the status code rather than an exception.
    """
    if not np.issubdtype(x.dtype, np.integer):
        return FrameStatus.DATA_CHECK_FAILED
    nz = np.flatnonzero(x)
    if nz.size and x[nz].min() < 0:
        return FrameStatus.DATA_CHECK_FAILED
    return FrameStatus.OK


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Per-frame sufficient statistics and scalars."""

    signal: np.ndarray  # float64 (npt,)  Σ c·x
    normalization: np.ndarray  # float64 (npt,)  Σ c·Ω over valid pixels
    variance: np.ndarray  # float64 (npt,)  Σ c²·x
    photons_valid: float
    n_bad_pixels: int
    n_frame_specific: int
    max_count: int


def integrate_frame(
    op: SparseOperator,
    x: np.ndarray,
    bad: np.ndarray,
    base_bad: np.ndarray,
    base_denominator: np.ndarray,
) -> FrameResult:
    """Integrate one frame over its photon hits.

    :param x: flattened integer photon counts, ``(NPIX,)``.
    :param bad: this frame's bad-pixel mask, ``((m & mask_bits) != 0) |
        static_bad``.
    :param base_bad: the memory cell's base mask, a superset of ``static_bad``.
    :param base_denominator: ``Σ c·Ω`` over ``~base_bad``, from
        :func:`denominator`.

    ``normalization`` is corrected off ``base_denominator`` by the pixels where
    this frame disagrees with its cell, in either direction: a pixel the cell
    calls bad but the frame calls good is added, one the cell calls good but
    the frame calls bad is subtracted. The result is exact, not approximate.

    :raises DataCheckFailed: ``x`` is not an integer dtype, or carries negative
        counts. There is no dense fallback and no NaN sentinel, so a failing
        frame never yields arrays.
    """
    npix = op.indptr.size - 1
    for name, array in (("x", x), ("bad", bad), ("base_bad", base_bad)):
        if array.shape != (npix,):
            raise ValueError(f"{name} must have shape ({npix},), got {array.shape}")
    if base_denominator.shape != (op.npt,):
        raise ValueError(
            f"base_denominator must have shape ({op.npt},), "
            f"got {base_denominator.shape}"
        )

    if not np.issubdtype(x.dtype, np.integer):
        raise DataCheckFailed(
            f"image.data must be an integer photon count, got dtype {x.dtype}"
        )
    nz = np.flatnonzero(x)
    if nz.size and x[nz].min() < 0:
        raise DataCheckFailed(
            f"negative photon counts in frame: min {int(x[nz].min())}"
        )

    keep = nz[~bad[nz]]
    xs = x[keep].astype(np.float64)
    signal = gather(op, keep, xs)
    variance = gather(op, keep, xs, squared=True)

    diff = np.flatnonzero(bad != base_bad)
    sign = np.where(base_bad[diff], 1.0, -1.0)
    normalization = base_denominator + gather(op, diff, sign * op.omega[diff])

    return FrameResult(
        signal=signal,
        normalization=normalization,
        variance=variance,
        photons_valid=float(xs.sum()),
        n_bad_pixels=int(bad.sum()),
        n_frame_specific=int(diff.size),
        max_count=int(x[nz].max()) if nz.size else 0,
    )
