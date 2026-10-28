"""The photon store's ring pixels, averaged over hit trains, for the detector figures.

:func:`detector_mean` reads the photons of some hit trains from the photon
store, which holds the annulus the rings span and nothing else of the detector,
and the curves here bin them in q and in azimuthal cones as extra-speckle's
``saxs.get`` and ``get_SAXS_var`` did. Nothing here reads proc.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from analysis.saxs.config import SHAPE
from analysis.xpcs.config import XpcsConfig
from analysis.xpcs.reader import XpcsHits
from analysis.xpcs.rings import sha256_of_array
from analysis.xpcs.store import RingPhotonStore

__all__ = [
    "N_CONES",
    "OUTLIER_THRESHOLDS",
    "Q_BIN_A",
    "DetectorMean",
    "cone_curves",
    "cone_index",
    "detector_mean",
    "outlier_ratio",
    "q_curves",
    "run_label",
]

#: extra-speckle's outlier thresholds, as ``xpcs_offline`` was called with them.
OUTLIER_THRESHOLDS = (0.75, 1.75)
#: Width in Å⁻¹ of extra-speckle's ``saxs.get`` bins, 300 over the detector's
#: q range.
Q_BIN_A = 3.24e-4
N_CONES = 8


def run_label(hits: XpcsHits) -> str:
    """The run number, from the recorded config, else from ``r####`` in the path."""
    try:
        return str(json.loads(hits.attrs["config"])["saxs"]["run"])
    except (KeyError, TypeError, ValueError):
        match = re.search(r"r(\d{4})", str(hits.path))
        return str(int(match.group(1))) if match else "?"


@dataclass(frozen=True, slots=True)
class DetectorMean:
    """Per-frame mean photons of the ring pixels over some hit trains.

    Built by :func:`detector_mean` from the photon store, so it holds the
    *annulus* the rings span and nothing else of the detector, and only the
    ring pixels that survive the static mask (the results file's ring labels).
    """

    #: ``(P, n_good)`` float32: photons per frame summed over ``n_trains`` trains.
    total: np.ndarray
    n_trains: int
    train_ids: np.ndarray
    #: flat detector index ``(n_good,)`` into ``(16·512, 128)``.
    pixel: np.ndarray
    ring: np.ndarray  # int8 (n_good,)
    q_A: np.ndarray  # Å⁻¹ at the pixel centre
    chi_deg: np.ndarray
    ring_bounds_A: np.ndarray  # (R, 2), (0, 2) for rings from a masks file
    geometry: Any
    beam_center: tuple[float, float] | None
    run: str
    #: Flagged in some frame of every train read: ``MaskPolicy.ANY_FRAME`` keeps
    #: such a pixel out of every one of those trains' sums. ``None``: not known.
    always_flagged: np.ndarray | None = None
    #: The pass's applied outlier mask and its statistic, per pixel; ``None``
    #: when the results file has none. ``pixel`` includes the outliers.
    outlier: np.ndarray | None = None
    outlier_ratio: np.ndarray | None = None
    outlier_thresholds: tuple[float, float] | None = None
    outlier_trains: int = 0

    @property
    def n_frames(self) -> int:
        return int(self.total.shape[0])

    @property
    def mean_image(self) -> np.ndarray:
        """Mean photons per pixel and frame, ``(n_good,)``."""
        return self.total.sum(0) / (self.n_trains * self.n_frames)

    def per_frame(self) -> np.ndarray:
        """``(P, n_good)`` mean photons per pixel and frame over the trains."""
        return self.total / self.n_trains


def detector_mean(
    hits: XpcsHits,
    cfg: XpcsConfig,
    *,
    max_trains: int | None = 200,
    geometry: Any = None,
) -> DetectorMean:
    """Read the photons of up to ``max_trains`` hit trains from the store.

    The trains are spread evenly over the usable ones. Only the static ring
    labels apply, not the per-frame ``image.mask`` flags, because pixels flagged
    in single frames are a small fraction of the ring pixels.

    :param cfg: gives the store path and the geometry (beam centre, distance).
    :param geometry: an AGIPD geometry object to use instead of
        ``cfg.saxs.geometry_file``, such as a synthetic one in tests.
    :raises FileNotFoundError: no photon store for this run.
    :raises ValueError: the store's envelope is not the one the results used.
    """
    from analysis.saxs.operator import build_operator, geometry_from_config

    if not cfg.store_file.exists():
        raise FileNotFoundError(f"no photon store at {cfg.store_file}")
    usable = hits.train_ids[hits.usable]
    if max_trains is not None and usable.size > max_trains:
        usable = usable[np.linspace(0, usable.size - 1, max_trains).astype(int)]
    label = hits.ring_label().astype(np.int16)
    applied = hits.outlier()
    if applied is not None:
        label = applied[1].astype(np.int16)
    geom = geometry if geometry is not None else geometry_from_config(cfg.saxs)
    _, ai = build_operator(geom, cfg.saxs)
    q_all = np.asarray(ai.array_from_unit(SHAPE, "center", "q_A^-1")).ravel()
    chi_all = np.asarray(ai.array_from_unit(SHAPE, "center", "chi_deg")).ravel()
    with RingPhotonStore.open(cfg.store_file) as store:
        envelope = store.envelope()
        if sha256_of_array(envelope) != hits.attrs.get("envelope_sha256"):
            raise ValueError(
                f"{cfg.store_file} holds another envelope than {hits.path}"
            )
        if label.size != envelope.size:
            raise ValueError("ring labels and store envelope differ in size")
        good = np.flatnonzero(label >= 0)
        to_good = np.full(envelope.size, -1, np.int64)
        to_good[good] = np.arange(good.size)
        total = np.zeros((hits.n_frames, good.size), np.float32)
        always_flagged = np.ones(good.size, bool)
        n_read = 0
        for train in usable:
            photons = store.get(int(train))
            if photons.n_frames != hits.n_frames:
                continue
            n_read += 1
            always_flagged &= photons.flagged_any(envelope.size)[good]
            frames = photons.frames(envelope.size)
            rows = np.repeat(np.arange(hits.n_frames), np.diff(frames.indptr))
            cols = to_good[frames.indices]
            keep = cols >= 0
            total[rows[keep], cols[keep]] += frames.data[keep]
    pixel = envelope[good]
    centre = (
        (cfg.saxs.beam_center_px, cfg.saxs.beam_center_py)
        if cfg.saxs.beam_center_px is not None
        else None
    )
    return DetectorMean(
        total=total,
        n_trains=int(usable.size),
        train_ids=usable,
        pixel=pixel.astype(np.int64),
        ring=label[good].astype(np.int8),
        q_A=q_all[pixel],
        chi_deg=chi_all[pixel],
        ring_bounds_A=hits.bounds_nm / 10.0,
        geometry=geom,
        beam_center=centre,
        run=run_label(hits),
        always_flagged=always_flagged if n_read else np.zeros(good.size, bool),
        outlier=None if applied is None else applied[0].mask[good],
        outlier_ratio=None if applied is None else applied[0].ratio[good],
        outlier_thresholds=None if applied is None else applied[0].thresholds,
        outlier_trains=0 if applied is None else applied[0].n_trains,
    )


def q_curves(
    det: DetectorMean,
    *,
    dq_A: float = Q_BIN_A,
    pixels: np.ndarray | None = None,
) -> xr.DataArray:
    """I(q) per frame by binning the ring pixels in q: ``(frame, q)``, photons/px.

    This is extra-speckle's ``saxs.get`` on the annulus, a plain mean of the
    pixels in each ``dq_A`` bin without the pixel splitting pyFAI does. Bins at
    the edge of the annulus hold fewer pixels, and a bin with none is NaN.

    :param pixels: boolean ``(n_good,)`` selecting the pixels that count. The
        default is all but the applied outliers, as extra-speckle made
        ``SAXS.nc`` after ``mask_outlier``.
    """
    if pixels is not None:
        keep = np.asarray(pixels)
    elif det.outlier is not None:
        keep = ~det.outlier
    else:
        keep = np.ones(det.pixel.size, bool)
    edges = _q_edges(det, dq_A)
    index = np.digitize(det.q_A, edges) - 1
    keep = keep & (index >= 0) & (index < edges.size - 1)
    counts = np.bincount(index[keep], minlength=edges.size - 1).astype(np.float64)
    per_frame = det.per_frame()
    sums = np.stack(
        [
            np.bincount(index[keep], weights=row[keep], minlength=edges.size - 1)
            for row in per_frame
        ]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(counts > 0, sums / counts, np.nan)
    out = xr.DataArray(
        value,
        dims=("frame", "q"),
        coords={"frame": np.arange(det.n_frames), "q": 0.5 * (edges[:-1] + edges[1:])},
        name="saxs",
    )
    out.attrs.update(resolution="q bins", dq_A=dq_A, n_trains=det.n_trains)
    return out


def _q_edges(det: DetectorMean, dq_A: float) -> np.ndarray:
    lo, hi = float(det.q_A.min()), float(det.q_A.max())
    return np.arange(lo, hi + dq_A, dq_A)


def outlier_ratio(
    det: DetectorMean, *, dq_A: float = Q_BIN_A, thresholds=OUTLIER_THRESHOLDS
) -> np.ndarray:
    """extra-speckle's ``mask_outlier`` statistic for every ring pixel.

    ``sqrt(mean over frames of (per-frame mean)²)`` of a pixel, over the median
    of that quantity in its q bin; a pixel outside ``thresholds`` is an
    outlier. A pixel outside every q bin gets extra-speckle's fill, the mean of
    the thresholds, and a bin whose median is zero gives NaN.

    It is not the mask the pass applies (`analysis.xpcs.outliers.outlier_mask`):
    that one leaves each frame's flagged pixels out of the per-frame mean and
    bins on the q centres of the configured range, while this one has only the
    store's photons of ``det.n_trains`` trains, no per-frame flags, and the
    figures' ``dq_A`` bins from the annulus's own q range. The figures use it
    only for a results file made without the mask.
    """
    per_frame = det.per_frame()
    rms = np.sqrt(np.mean(per_frame.astype(np.float64) ** 2, axis=0))
    edges = _q_edges(det, dq_A)
    index = np.digitize(det.q_A, edges) - 1
    ratio = np.full(rms.size, float(np.mean(thresholds)))
    for b in np.unique(index[(index >= 0) & (index < edges.size - 1)]):
        inside = index == b
        median = np.median(rms[inside])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio[inside] = rms[inside] / median if median > 0 else np.nan
    return ratio


def cone_index(det: DetectorMean) -> np.ndarray:
    """Each ring pixel's azimuthal cone, ``0 … N_CONES − 1``, by its χ."""
    return np.floor((det.chi_deg % 360.0) / (360.0 / N_CONES)).astype(int)


def cone_curves(
    det: DetectorMean, q_bounds: tuple[float, float], *, dq_A: float = Q_BIN_A
) -> xr.DataArray:
    """I(q) in ``N_CONES`` azimuthal cones, ``(q_slice, q)``, over ``q_bounds`` (Å⁻¹).

    The cones are extra-speckle's ``get_SAXS_var`` ones, equal azimuthal sectors
    of 360/``N_CONES`` degrees.
    """
    cone = cone_index(det)
    mean = det.mean_image
    edges = np.arange(q_bounds[0], q_bounds[1] + dq_A, dq_A)
    index = np.digitize(det.q_A, edges) - 1
    inside = (index >= 0) & (index < edges.size - 1)
    out = np.full((N_CONES, edges.size - 1), np.nan)
    for c in range(N_CONES):
        sel = inside & (cone == c)
        counts = np.bincount(index[sel], minlength=edges.size - 1)
        sums = np.bincount(index[sel], weights=mean[sel], minlength=edges.size - 1)
        out[c] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return xr.DataArray(
        out,
        dims=("q_slice", "q"),
        coords={"q_slice": np.arange(N_CONES), "q": 0.5 * (edges[:-1] + edges[1:])},
    )
