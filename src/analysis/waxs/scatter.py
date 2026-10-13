"""Is the stored σ the scatter the frames actually show? (W4 gate E.)

The pass stores a variance per frame and bin — Poisson on the counts since
photon counting, ``σ_read² + E·x`` in keV² before it. Pooling trusts that number,
so it is worth checking against the data, and the frames of one train are the
natural comparison: eight pulses through the same droplet a few hundred
nanoseconds apart.

What differs between them besides counting noise is mostly the amount of light
— pulse energy, and cell 15 running ~5 % low — so each frame is first scaled by
its own mean intensity over the bins every frame of the train populates. The
scatter left between the scaled frames is then compared with the model's σ
scaled the same way, **as variances summed over trains** (a sample variance is
unbiased where a sample σ is not: the median σ of eight Poisson frames sits ~5 %
below the truth).

A ratio near 1 says the stored σ is what the frames show. W7 measured D = 0.93–0.95
for the per-pixel counts, which predicts a ratio of ~0.97 for a pure counting
process; anything well above that is real frame-to-frame structure, or noise the
model leaves out (an uncorrected zero level, for one).

Works on any output file of the pass, keV or photons, off the stored sums only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.common.status import FrameStatus
from analysis.common.writer import as_handle, q_centers

__all__ = ["frame_scatter"]

#: q bands the ratio is reported over, besides its median over all bins.
N_BANDS = 10


def frame_scatter(
    source: Any, *, q_range: tuple[float, float] | None = None
) -> dict[str, Any]:
    """Empirical frame-to-frame scatter against the stored σ, and train spread.

    :param source: an open handle or a path to a finished output file.
    :param q_range: restrict to bins with centres inside it; ``None`` takes all.
    :returns: ``sigma_ratio_median`` — the median over bins of
        ``sqrt(Σ_trains empirical variance / Σ_trains model variance)`` — its 5th
        and 95th percentiles and its value per q band; ``relative_std_between_
        trains``, the spread of each train's pooled mean intensity; and the
        counts behind them. Only trains whose frames are all ``OK`` and number
        at least two take part.
    """
    with as_handle(source) as handle:
        frames = handle["frames"]
        status = frames["status"][:]
        signal = frames["signal"][:].astype(np.float64)
        normalization = frames["normalization"][:].astype(np.float64)
        variance = frames["variance"][:].astype(np.float64)
        q = q_centers(handle, signal.shape[1])
        first = handle["trains/first"][:]
        count = handle["trains/count"][:]

    chosen = np.ones(q.size, dtype=bool)
    if q_range is not None:
        chosen = (q >= q_range[0]) & (q <= q_range[1])

    empirical = np.zeros(q.size)
    model = np.zeros(q.size)
    n_terms = np.zeros(q.size, dtype=np.int64)
    train_means = []
    for start, n in zip(first.tolist(), count.tolist(), strict=True):
        if n < 2:
            continue
        rows = slice(int(start), int(start) + int(n))
        if not (status[rows] == FrameStatus.OK).all():
            continue
        s, w, v = signal[rows], normalization[rows], variance[rows]
        common = chosen & (w > 0).all(axis=0)
        if common.sum() < 2:
            continue
        s, w, v = s[:, common], w[:, common], v[:, common]
        scale = s.sum(axis=1) / w.sum(axis=1)  # each frame's mean intensity
        if not (scale > 0).all():
            continue
        shape = (s / w) / scale[:, None]
        model_variance = (v / w**2) / scale[:, None] ** 2
        empirical[common] += shape.var(axis=0, ddof=1)
        model[common] += model_variance.mean(axis=0)
        n_terms[common] += 1
        train_means.append(s.sum() / w.sum())

    usable = (n_terms > 0) & (model > 0)
    ratio = np.full(q.size, np.nan)
    ratio[usable] = np.sqrt(empirical[usable] / model[usable])
    means = np.asarray(train_means)
    bands = np.array_split(np.flatnonzero(usable), N_BANDS) if usable.any() else []
    return {
        "n_trains": int(means.size),
        "n_bins": int(usable.sum()),
        "sigma_ratio_median": float(np.nanmedian(ratio)) if usable.any() else None,
        "sigma_ratio_p05_p95": (
            [float(v) for v in np.nanpercentile(ratio, [5, 95])]
            if usable.any()
            else None
        ),
        "sigma_ratio_by_q_band": [
            {
                "q_low": float(q[band[0]]),
                "q_high": float(q[band[-1]]),
                "sigma_ratio": float(
                    np.sqrt(empirical[band].sum() / model[band].sum())
                ),
            }
            for band in bands
            if band.size
        ],
        "relative_std_between_trains": (
            float(means.std(ddof=1) / means.mean()) if means.size > 1 else None
        ),
        "q_range": list(q_range) if q_range is not None else None,
    }
