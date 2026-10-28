"""extra-speckle's ``xpcs_offline`` data columns, made from the hit-train results file.

Each function reproduces one DAMNIT data column of extra-speckle from
:class:`~analysis.xpcs.reader.XpcsHits` (exact per-train sums). The default
estimator is :attr:`~analysis.xpcs.reader.Averaging.POOLED` (ratios of summed
sums); ``Averaging.ES`` gives extra-speckle's mean of per-train ratios.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from analysis.xpcs.reader import Averaging, XpcsHits

__all__ = ["legacy_mean_dataset", "legacy_ttcf", "mean_intensity", "saxs_rings"]


def legacy_ttcf(
    hits: XpcsHits, trains: np.ndarray | None = None, chunk: int = 128
) -> xr.DataArray:
    """The ``XPCS TTCF`` column: each train's own ``TTCF_raw − TTCF_off``.

    Same layout as extra-speckle's per-train files, ``(trainId, roi_q, frame_1,
    frame_2)`` float32, one row per usable hit train. A train with no baseline
    (the first of a chain, or after a gap) has a NaN ``TTCF_off`` and so a NaN
    row, as in extra-speckle's files. It holds 4 P² R bytes a train, so pass
    ``trains`` to take a part of a long run.

    :param chunk: trains read at a time, so the peak is one float32 copy, not three.
    """
    ids = hits.train_ids[hits.usable]
    if trains is not None:
        ids = ids[np.isin(ids, np.asarray(trains, dtype=np.uint64))]
    out = np.empty((ids.size, hits.n_rings, hits.n_frames, hits.n_frames), np.float32)
    for start in range(0, ids.size, chunk):
        part = hits.per_train(ids[start : start + chunk])
        out[start : start + chunk] = part["TTCF_raw"].values - part["TTCF_off"].values
    frames = np.arange(hits.n_frames)
    return xr.DataArray(
        out,
        dims=("trainId", "roi_q", "frame_1", "frame_2"),
        coords={
            "trainId": ids,
            "roi_q": hits.roi_q,
            "frame_1": frames,
            "frame_2": frames,
        },
        name="TTCF",
    )


def legacy_mean_dataset(
    hits: XpcsHits, averaging: Averaging = Averaging.POOLED
) -> xr.Dataset:
    """The ``XPCS mean dataset`` column: ``ttcf_mean`` and ``g2``.

    ``ttcf_mean`` is the baseline-subtracted mean map ``(roi_q, frame_1,
    frame_2)`` and ``g2`` the binned diagonal reduction ``(roi_q, tau)``, both
    float32 as extra-speckle wrote them. With ``Averaging.POOLED`` they are ``c2`` of
    :meth:`XpcsHits.ttcf` and :meth:`XpcsHits.g2` and ``g2_sigma`` (the
    jackknife error) is added; ``Averaging.ES`` gives extra-speckle's own, with
    its noisy-ring rule, so rings can be missing from ``g2`` and not from
    ``ttcf_mean``.
    """
    maps = hits.ttcf(averaging)["c2"]
    g2 = hits.g2(averaging)
    pooled = averaging is Averaging.POOLED
    data = {
        "ttcf_mean": maps.astype(np.float32),
        "g2": g2["c2" if pooled else "g2"].astype(np.float32),
    }
    if pooled:
        data["g2_sigma"] = g2["sigma"].astype(np.float32)
    out = xr.Dataset(data)
    out.attrs.update(averaging=averaging.value, **{k: v for k, v in maps.attrs.items()})
    return out


def mean_intensity(hits: XpcsHits) -> xr.DataArray:
    """Mean photons per ring pixel and frame of each usable train, ``(trainId,)``.

    extra-speckle's was the mean over the whole masked detector; this one is
    over the ring pixels, because the store holds only the annulus. Both track
    the droplet's total scattering per train, on different scales.
    """
    rows, s_on, cnt = hits.ring_sums()
    total_pixels = cnt.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = s_on.sum(axis=(1, 2)) / (total_pixels * hits.n_frames)
    return xr.DataArray(
        value,
        dims="trainId",
        coords={"trainId": hits.train_ids[rows]},
        name="mean_intensity",
    )


def saxs_rings(hits: XpcsHits) -> xr.DataArray:
    """I per frame and ring, ``(frame, q)`` in photons per pixel, from the results file.

    extra-speckle's ``XPCS SAXS`` column was a 300-point I(q) per pulse. The
    sums resolve nothing finer than the rings, so this is that curve sampled at
    the ring centres, pooled over the usable trains. For the full curve use
    :func:`analysis.xpcs.detector.q_curves`.
    """
    _, s_on, cnt = hits.ring_sums()
    with np.errstate(invalid="ignore", divide="ignore"):
        value = s_on.sum(0) / cnt.sum(0)[:, None]
    out = xr.DataArray(
        value.T,
        dims=("frame", "q"),
        coords={"frame": np.arange(hits.n_frames), "q": hits.roi_q},
        name="saxs",
    )
    out.attrs.update(resolution="rings", n_trains=int(cnt.shape[0]))
    return out
