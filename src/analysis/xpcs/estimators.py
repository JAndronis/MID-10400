"""The XPCS estimators: pure functions from per-train integer sums to g2.

Nothing here reads a file. The pooled estimates divide summed numerators by
summed denominators, so they pool in any order, and a quantity with no photons
behind it is NaN. extra-speckle's per-train normalisation and its log binning
are here too, for the ``Averaging.ES`` view.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "N_BLOCKS",
    "SKIP_LAST_LAGS",
    "block_jackknife",
    "diagonal_means",
    "es_g2",
    "lag_frames",
    "normalised",
    "pooled_g2",
    "safe_divide",
]

#: Lags extra-speckle drops from the end of g2, when 20 or more frames remain.
SKIP_LAST_LAGS = 10
#: Jackknife blocks; the variance estimate is then good to sqrt(1/(2·49)) ≈ 10 %.
N_BLOCKS = 50
#: extra-speckle bins g2 only above this many lags, in this many log bins after
#: this many unbinned ones (``g2_to_bins(50, 20, 10)``).
BIN_THRESHOLD, BIN_COUNT, BIN_SKIP = 50, 20, 10


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator / denominator``, NaN where there is nothing behind the ratio."""
    out = np.full(np.shape(numerator), np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def outer(s: np.ndarray, t: np.ndarray, count: np.ndarray) -> np.ndarray:
    """``s_a t_b / count`` over the last axis pair; zero where ``count`` is."""
    out = np.zeros((*s.shape, s.shape[-1]))
    np.divide(
        s[..., :, None] * t[..., None, :],
        count[..., None, None],
        out=out,
        where=(count > 0)[..., None, None],
    )
    return out


def lag_sums(matrix: np.ndarray, lags: np.ndarray, both: bool = False):
    """Sum of each ``lags`` diagonal of the last two axes, plus ``−lag`` if ``both``."""
    if lags.size == 0:
        return np.zeros((*matrix.shape[:-2], 0))
    return np.stack(
        [
            np.diagonal(matrix, k, -2, -1).sum(-1)
            + (np.diagonal(matrix, -k, -2, -1).sum(-1) if both else 0)
            for k in lags
        ],
        axis=-1,
    )


def block_jackknife(
    statistic: Callable[..., np.ndarray],
    *terms: np.ndarray,
    n_blocks: int = N_BLOCKS,
    blocks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``statistic`` of the summed ``terms``, its leave-one-block-out error, and the
    leave-one-out values.

    Each term is summed over its first axis (the rows), so ``statistic`` sees
    totals. The blocks are contiguous in row order unless given, so a slow
    drift inflates the error instead of hiding in it. With fewer than two
    blocks the error is NaN, and it is NaN wherever the value is not finite.

    :param blocks: row indices of each block, e.g. every frame of a group of
        trains; ``n_blocks`` contiguous blocks of rows when None.
    :returns: ``(value, sigma, loo)``, ``loo`` stacked over the blocks.
    """
    n = terms[0].shape[0]
    totals = [term.sum(0) for term in terms]
    value = statistic(*totals)
    if blocks is None:
        blocks = np.array_split(np.arange(n), min(n_blocks, n)) if n else []
    count = len(blocks)
    if count < 2:
        return value, np.full_like(value, np.nan), np.empty((0, *np.shape(value)))
    parts = [np.stack([term[b].sum(0) for b in blocks]) for term in terms]
    loo = statistic(*(total - part for total, part in zip(totals, parts, strict=True)))
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(
            (count - 1) / count * np.nansum((loo - np.nanmean(loo, 0)) ** 2, 0)
        )
    sigma[~np.isfinite(value)] = np.nan
    return value, sigma, loo


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """The pooled ratio of two summed terms."""
    return safe_divide(numerator, denominator)


def difference_of_ratios(on, den_on, off, den_off) -> np.ndarray:
    """``c2 = g2 − baseline`` from the four summed lag terms."""
    return safe_divide(on, den_on) - safe_divide(off, den_off)


def lag_frames(n_frames: int, skip_last: int = SKIP_LAST_LAGS) -> np.ndarray:
    """Lags (in frames) g2 uses, ``1 … P − 1 − skip``; skip if ``P ≥ 20 + skip``."""
    skip = skip_last if n_frames - skip_last >= 20 else 0
    return np.arange(1, n_frames - skip)


def pooled_g2(
    on: np.ndarray,
    den_on: np.ndarray,
    off: np.ndarray,
    den_off: np.ndarray,
    has_off: np.ndarray,
    *,
    lags: np.ndarray,
    period: float | None,
    roi_q: np.ndarray,
    binned: bool = True,
    n_blocks: int = N_BLOCKS,
) -> xr.Dataset:
    """The pooled g2 Dataset of per-row lag terms, as ``XpcsHits.g2`` makes it.

    ``on, den_on, off, den_off`` are ``(rows, rings, P − 1)`` over the lags
    ``1 … P − 1`` (``PooledSums``); ``lags`` picks the columns used. A row without a
    baseline has ``has_off`` False and its baseline terms are not read. Rows pool
    as a ratio of sums, and the jackknife blocks are contiguous in row order.
    """
    rows = on.shape[0]
    on, den_on, off, den_off = (a[..., lags - 1] for a in (on, den_on, off, den_off))
    groups = _lag_bins(lags) if binned else [np.array([i]) for i in range(lags.size)]
    pooled = lambda a: np.stack([a[..., g].sum(-1) for g in groups], -1)  # noqa: E731
    on, den_on, off, den_off = (pooled(a) for a in (on, den_on, off, den_off))
    paired = (on[has_off], den_on[has_off], off[has_off], den_off[has_off])
    g2, sigma_g2, _ = block_jackknife(ratio, on, den_on, n_blocks=n_blocks)
    g2_paired = ratio(paired[0].sum(0), paired[1].sum(0))
    baseline = ratio(paired[2].sum(0), paired[3].sum(0))
    c2, sigma, _ = block_jackknife(difference_of_ratios, *paired, n_blocks=n_blocks)
    centre = np.array([lags[g].mean() for g in groups])
    dims = ("roi_q", "tau")
    dataset = xr.Dataset(
        {
            "g2": (dims, g2),
            "sigma_g2": (dims, sigma_g2),
            "g2_paired": (dims, g2_paired),
            "baseline": (dims, baseline),
            "c2": (dims, c2),
            "sigma": (dims, sigma),
        },
        coords={"roi_q": roi_q, "tau": centre * period, "lag": ("tau", centre)},
    )
    dataset.attrs.update(
        averaging="pooled",
        n_trains=int(rows),
        n_with_baseline=int(has_off.sum()),
        period_s=period,
        binned=int(binned and lags.size > BIN_THRESHOLD),
    )
    return dataset


def _lag_bins(lags: np.ndarray) -> list[np.ndarray]:
    """Index groups of ``lags`` for binning, or one group per lag when few.

    As extra-speckle's ``g2_to_bins``: above ``BIN_THRESHOLD`` lags the first
    ``BIN_SKIP`` stay alone and the rest fall into ``BIN_COUNT`` logarithmic bins
    (the last one closed, so the final lag is not lost).
    """
    if lags.size <= BIN_THRESHOLD:
        return [np.array([i]) for i in range(lags.size)]
    edges = np.logspace(np.log10(lags[BIN_SKIP]), np.log10(lags[-1]), BIN_COUNT + 1)
    groups = [np.array([i]) for i in range(BIN_SKIP)]
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        inside = (lags >= lo) & ((lags < hi) | (b == BIN_COUNT - 1))
        if inside.any():
            groups.append(np.flatnonzero(inside))
    return groups


def normalised(c: dict[str, Any], has_off: np.ndarray, zero_norm: float):
    """Each train's own normalised maps, extra-speckle's ``convert_TTCF``.

    ``c`` holds the float sums of some rows (``on``, ``off``, ``s_on``,
    ``s_off_cur``, ``s_off_prev``, ``cnt``, ``cnt_pair``). A frame pair whose
    norm is zero takes the norm ``zero_norm``: extra-speckle's 1 turns a frame
    with no photons into a 0 entry, NaN leaves it undefined.

    :returns: ``(raw, off, kbar)``; ``off`` is symmetrised, and NaN for a row
        whose ``has_off`` is False.
    """
    cnt, cnt_pair = c["cnt"], c["cnt_pair"]
    with np.errstate(invalid="ignore", divide="ignore"):
        kbar = c["s_on"] / cnt[..., None]
        norm = kbar[..., :, None] * kbar[..., None, :]
        norm[norm == 0] = zero_norm
        raw = c["on"] / (norm * cnt[..., None, None])
        cur = c["s_off_cur"] / cnt_pair[..., None]
        prev = c["s_off_prev"] / cnt_pair[..., None]
        norm_off = cur[..., :, None] * prev[..., None, :]
        norm_off[norm_off == 0] = zero_norm
        off = c["off"] / (norm_off * cnt_pair[..., None, None])
        off = 0.5 * (off + off.swapaxes(-1, -2))
    off[~has_off] = np.nan
    return raw, off, kbar


def diagonal_means(maps: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """The NaN-mean of each ``lags`` diagonal of every ring's map, ``(R, K)``.

    As extra-speckle's ``reduce_ttcf``; a diagonal with no finite entry is NaN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack(
            [np.nanmean(np.diagonal(maps, k, -2, -1), axis=-1) for k in lags],
            axis=-1,
        )


def es_g2(
    mean: np.ndarray,
    roi_q: np.ndarray,
    lags: np.ndarray,
    period: float,
    binned: bool,
    nan_thresh: float | None,
) -> xr.Dataset:
    """extra-speckle's g2 of a mean baseline-subtracted map ``(R, P, P)``.

    A ring with more than ``nan_thresh`` of its map NaN is left out
    (``reduce_ttcf(mask_noisy=True)``); each lag is the NaN-mean of its diagonal,
    and above ``BIN_THRESHOLD`` lags the tail is the median over log bins.
    """
    if nan_thresh is not None:
        noisy = np.isnan(mean).sum(axis=(1, 2)) > nan_thresh * mean[0].size
        mean, roi_q = mean[~noisy], roi_q[~noisy]
    diag = diagonal_means(mean, lags)
    tau = lags * period
    if binned and lags.size > BIN_THRESHOLD:
        edges = np.logspace(np.log10(tau[BIN_SKIP]), np.log10(tau[-1]), BIN_COUNT + 1)
        tail = [
            np.nanmedian(diag[..., (tau >= lo) & (tau < hi)], axis=-1)
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
        diag = np.concatenate([diag[..., :BIN_SKIP], np.column_stack(tail)], -1)
        tau = np.concatenate([tau[:BIN_SKIP], 0.5 * (edges[:-1] + edges[1:])])
    return xr.Dataset(
        {"g2": (("roi_q", "tau"), diag)},
        coords={"roi_q": roi_q, "tau": tau},
        attrs={"averaging": "es", "period_s": period},
    )


@dataclass(frozen=True)
class TtcfDiagonals:
    """Per train and ROI, the sum and count of finite TTCF entries at each lag."""

    train_ids: np.ndarray  # (T,) uint64, sorted
    roi_q_A: np.ndarray  # (R,)
    sums: np.ndarray  # (T, R, F − 1) float64; lag k at index k − 1
    counts: np.ndarray  # (T, R, F − 1) int32

    @property
    def n_frames(self) -> int:
        return self.sums.shape[2] + 1


def diagonal_sums(ttcf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n, R, F, F)`` TTCF stack → sums and counts of its finite entries per lag."""
    n, r, f, _ = ttcf.shape
    sums = np.empty((n, r, f - 1), dtype=np.float64)
    counts = np.empty((n, r, f - 1), dtype=np.int32)
    for k in range(1, f):
        diag = np.diagonal(ttcf, offset=k, axis1=2, axis2=3)
        finite = np.isfinite(diag)
        sums[..., k - 1] = np.where(finite, diag, 0.0).sum(axis=-1, dtype=np.float64)
        counts[..., k - 1] = finite.sum(axis=-1)
    return sums, counts
