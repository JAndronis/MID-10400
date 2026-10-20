"""extra-speckle's ``xpcs_offline`` figures and data columns, from the hit store.

The old DAMNIT columns (``src/amore/context.py`` ``xpcs_*``) came from one
``xpcs_offline`` call that read every train of a run. Everything it showed is
reproduced here from the two files the hit-train pass keeps: the results file
(:class:`~analysis.xpcs.result.XpcsHits`: exact per-train sums) and, for what is
a *detector* picture, the photon store (raw counts over the annulus the rings
span). Nothing here reads proc.

How faithfully each old product is reproduced:

========================= ==============================================================
old column                here
========================= ==============================================================
XPCS TTCF                 :func:`legacy_ttcf` (exact, per hit train, float32)
XPCS mean dataset         :func:`legacy_mean_dataset` (``ttcf_mean`` and ``g2``)
XPCS SAXS                 :func:`saxs_curves` (store), :func:`saxs_rings` (results)
q-rings                   :func:`figure_q_rings` (annulus only, see there)
beamcenter                :func:`figure_beamcenter` (a check, not a fit, see there)
outliers                  :func:`figure_outliers` (statistic of N hit trains)
intensity outliers        :func:`figure_intensity_outliers` (ring pixels only)
kbar                      :func:`figure_kbar` (exact)
correction                :func:`figure_correction` (exact up to the pooled estimator)
TTCF's                    :func:`figure_ttcfs` (exact up to the pooled estimator)
g2                        :func:`figure_g2`
g2 fit                    :func:`figure_g2_fit` (extra-speckle's model, :func:`fit_g2`)
SAXS overview             :func:`figure_saxs`
========================= ==============================================================

"Up to the pooled estimator" means the default view is
:attr:`~analysis.xpcs.result.Averaging.POOLED` (ratios of summed sums), the
production estimate; pass ``Averaging.ES`` for extra-speckle's mean of per-train
ratios, which the old columns held.

All figures are built on :class:`matplotlib.figure.Figure` directly, not through
``pyplot``, so no backend is needed and no figure leaks into global state. They
are returned open for DAMNIT or a notebook to show or save.
"""

from __future__ import annotations

import itertools
import json
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.saxs.config import SHAPE
from analysis.xpcs.config import XpcsConfig
from analysis.xpcs.result import ES_KBAR_MIN, Averaging, XpcsHits
from analysis.xpcs.rings import sha256_of_array
from analysis.xpcs.store import RingPhotonStore

__all__ = [
    "DetectorMean",
    "G2Fit",
    "cone_curves",
    "detector_mean",
    "figure_beamcenter",
    "figure_correction",
    "figure_g2",
    "figure_g2_fit",
    "figure_intensity_outliers",
    "figure_kbar",
    "figure_outliers",
    "figure_q_rings",
    "figure_saxs",
    "figure_ttcfs",
    "fit_g2",
    "legacy_mean_dataset",
    "legacy_ttcf",
    "mean_intensity",
    "outlier_ratio",
    "q_curves",
    "saxs_curves",
    "saxs_rings",
]

#: The old defaults (``xpcs_offline`` and the DAMNIT variable's arguments).
OUTLIER_THRESHOLDS = (0.75, 1.75)
INTENSITY_OUTLIER_FRACTION = 0.2
#: extra-speckle's ``saxs.get`` used 300 radial bins over the detector's q range
#: (0.0082 to 0.1051 Å⁻¹ in r0493's ``SAXS.nc``): 3.24e-4 Å⁻¹ each.
Q_BIN_A = 3.24e-4
N_CONES = 8
#: ``xpcs.fit_dr``'s constants: kB [J/K], 25 °C, viscosity 1 mPa s in N s Å⁻².
_KB, _T_K, _ETA = 1.381e-23, 273.15 + 25, 1e-23

_MARKERS = ("o", "v", "^", "<", ">", "H", "D", "*")


# ── small shared helpers ────────────────────────────────────────────────────────
def _run_label(hits: XpcsHits) -> str:
    """The run number, from the recorded config, else from ``r####`` in the path."""
    try:
        return str(json.loads(hits.attrs["config"])["saxs"]["run"])
    except (KeyError, TypeError, ValueError):
        match = re.search(r"r(\d{4})", str(hits.path))
        return str(int(match.group(1))) if match else "?"


def _ring_label(roi: float) -> str:
    return rf"{np.around(roi, 3)} $\AA^{{-1}}$"


def _ring_sums(hits: XpcsHits) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, s_on (n, R, P), cnt (n, R))`` of the usable trains, as float64."""
    rows = np.flatnonzero(hits.usable)
    with h5py.File(hits.path, "r") as f:
        s_on = f["sums/s_on"][rows.tolist()].astype(np.float64)
        cnt = f["sums/cnt"][rows.tolist()].astype(np.float64)
    return rows, s_on, cnt


def _jet(i: int, n: int):
    return colormaps["jet"](i / n)


def _quantile_limits(data: np.ndarray, lo: float = 0.02, hi: float = 0.90):
    """extra-speckle's TTCF colour limits: quantiles of the strict upper triangle."""
    values = data[np.triu_indices_from(data, k=1)]
    if not np.isfinite(values).any():
        return None, None
    return float(np.nanquantile(values, lo)), float(np.nanquantile(values, hi))


def _ttcf_panel(ax, data, title, frames, vmin=None, vmax=None, cmap="jet") -> None:
    """``extra_speckle.xpcs.plot.plot_ttcf_single`` plus its axis labels."""
    auto_min, auto_max = _quantile_limits(data)
    vmin = auto_min if vmin is None else vmin
    vmax = auto_max if vmax is None else vmax
    lo, hi = float(frames.min()), float(frames.max())
    image = ax.imshow(
        data,
        vmin=vmin,
        vmax=vmax,
        extent=[lo, hi, lo, hi],
        origin="lower",
        interpolation="none",
        cmap=cmap,
    )
    ax.set_title(title)
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(image, cax=cax)
    ax.set_xlabel("frame")
    ax.set_ylabel("frame")


def _g2_panel(
    ax,
    roi_q: np.ndarray,
    tau: np.ndarray,
    values: np.ndarray,
    *,
    title: str | None = None,
    fit: np.ndarray | None = None,
    errors: np.ndarray | None = None,
) -> None:
    """``extra_speckle.xpcs.plot.plot_g2``: jet colours, a marker per ring, log τ."""
    n = len(roi_q)
    markers = itertools.cycle(_MARKERS)
    for i, q in enumerate(roi_q):
        colour, marker = _jet(i, n), next(markers)
        kwargs = dict(
            ms=6,
            marker=marker,
            linewidth=0.0 if fit is not None else 1.0,
            color=colour,
            alpha=1 - i * 0.50 / n,
            label=rf"$q = {q:.4f} \ \mathrm{{\AA}}^{{-1}}$",
        )
        if errors is not None:
            ax.errorbar(tau, values[i], errors[i], capsize=0, elinewidth=0.6, **kwargs)
        else:
            ax.plot(tau, values[i], **kwargs)
        if fit is not None:
            ax.plot(tau, fit[i], color=colour)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau,  s$", fontsize="x-large")
    ax.set_ylabel(r"$g_{2}(q, tau)$", fontsize="x-large")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    if title is not None:
        ax.set_title(title)


def _diagonal_curves(maps: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """``reduce_ttcf``: the NaN-mean of each diagonal of every ring's map."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack(
            [[np.nanmean(np.diagonal(ring, int(k))) for k in lags] for ring in maps]
        )


# ── data products that come from the results file ─────────────────────────────
def legacy_ttcf(
    hits: XpcsHits, trains: np.ndarray | None = None, chunk: int = 128
) -> xr.DataArray:
    """The old ``XPCS TTCF`` column: each train's own ``TTCF_raw − TTCF_off``.

    Same layout as extra-speckle's per-train files, ``(trainId, roi_q, frame_1,
    frame_2)`` float32, one row per usable hit train. A train with no baseline
    (the first of a chain, or after a gap) has a NaN ``TTCF_off`` and so a NaN
    row, as in the old files. **Large** (about 0.65 MB a train a ring set at
    P = 155, R = 7): pass ``trains`` to take a part.

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
    """The old ``XPCS mean dataset`` column: ``ttcf_mean`` and ``g2``.

    ``ttcf_mean`` is the baseline-subtracted mean map ``(roi_q, frame_1,
    frame_2)`` and ``g2`` the binned diagonal reduction ``(roi_q, tau)``, both
    float32 as before. With ``Averaging.POOLED`` they are ``c2`` of
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

    The old column was the mean over the *whole masked detector*; this is over
    the ring pixels (the store holds only the annulus). It tracks the same
    thing, the droplet's total scattering per train, on another scale.
    """
    rows, s_on, cnt = _ring_sums(hits)
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
    """I per frame and ring, ``(frame, q)`` in photons per pixel: results file only.

    The old ``XPCS SAXS`` column was a 300-point I(q) per pulse. With the
    sums alone the finest resolution is the rings, so this is that curve
    sampled at the seven ring centres (pooled over the usable trains). For the
    full curve use :func:`saxs_curves`.
    """
    _, s_on, cnt = _ring_sums(hits)
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


# ── what needs the photon store ───────────────────────────────────────────────
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
    #: The pass's applied outlier mask and its statistic, per pixel above
    #: (``None``: the results file has none). The pixels include the outliers.
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

    The trains are spread evenly over the usable ones. Dynamic (per-frame)
    ``image.mask`` flags are not applied, only the static ring labels: a pixel
    flagged in one frame only is a small fraction of the ring pixels (§7 of
    ``context/xpcs-hits.md``).

    :param cfg: gives the store path and the geometry (beam centre, distance).
    :param geometry: an AGIPD geometry object instead of ``cfg.saxs.geometry_file``
        (the tests' synthetic one).
    :raises FileNotFoundError: no photon store for this run.
    :raises ValueError: the store's envelope is not the one the results used.
    """
    from analysis.saxs.operator import build_operator, geometry_from_config

    if not cfg.store_file.exists():
        raise FileNotFoundError(f"no photon store at {cfg.store_file}")
    usable = hits.train_ids[hits.usable]
    if max_trains is not None and usable.size > max_trains:
        usable = usable[np.linspace(0, usable.size - 1, max_trains).astype(int)]
    with h5py.File(hits.path, "r") as f:
        label = f["rings/label"][()].astype(np.int16)
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
        for train in usable:
            photons = store.get(int(train))
            if photons.n_frames != hits.n_frames:
                continue
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
        run=_run_label(hits),
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

    This is ``saxs.get`` of the old pipeline on the annulus: a plain mean of the
    pixels in each ``dq_A`` bin (no pixel splitting, which pyFAI's does), so
    bins at the edge of the annulus hold fewer pixels and a bin with none is NaN.

    :param pixels: boolean ``(n_good,)``: which pixels count (default all but
        the applied outliers, as ``SAXS.nc`` was made after ``mask_outlier``).
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


def saxs_curves(det: DetectorMean, *, dq_A: float = Q_BIN_A) -> xr.DataArray:
    """The old ``XPCS SAXS`` column on the annulus: I(q) per frame, ``(frame, q)``."""
    return q_curves(det, dq_A=dq_A)


def outlier_ratio(
    det: DetectorMean, *, dq_A: float = Q_BIN_A, thresholds=OUTLIER_THRESHOLDS
) -> np.ndarray:
    """extra-speckle's ``mask_outlier`` statistic for every ring pixel.

    ``sqrt(mean over frames of (per-frame mean)²)`` of a pixel, over the median
    of that quantity in its q bin; a pixel outside ``thresholds`` is an
    outlier. Pixels in a bin of their own (no q bin) keep the old fill, the
    mean of the thresholds.
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


def cone_curves(
    det: DetectorMean, q_bounds: tuple[float, float], *, dq_A: float = Q_BIN_A
) -> xr.DataArray:
    """I(q) in ``N_CONES`` azimuthal cones, ``(q_slice, q)``, over ``q_bounds`` (Å⁻¹).

    ``get_SAXS_var``'s cones: equal azimuthal sectors of 360/``N_CONES`` degrees.
    """
    cone = (np.floor((det.chi_deg % 360.0) / (360.0 / N_CONES))).astype(int)
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


# ── detector figures ──────────────────────────────────────────────────────────
def _assemble(det: DetectorMean, values: np.ndarray, fill: float = np.nan):
    """Assemble per-ring-pixel ``values`` into the detector picture.

    Module pixels without a value get ``fill``; the gaps between modules are NaN.
    """
    image = np.full(SHAPE[0] * SHAPE[1], fill, np.float32)
    image[det.pixel] = values
    return det.geometry.position_modules(image.reshape(16, 512, 128))


def _draw_image(ax, image, centre, det, *, title, norm=None, cmap="jet", **kwargs):
    """``Setup.plot_data``: front view, NaN gray, the geometry and beam crosses."""
    colours = colormaps[cmap] if isinstance(cmap, str) else cmap
    colours = colours.copy()
    colours.set_bad(color="0.25", alpha=1.0)
    ny, nx = image.shape
    handle = ax.imshow(
        image,
        origin="lower",
        interpolation="none",
        extent=(-0.5, nx + 0.5, -0.5, ny + 0.5),
        cmap=colours,
        norm=norm,
        **kwargs,
    )
    ctr_y, ctr_x = centre
    for colour, width, z in (("black", 3, 10), ("white", 1, 11)):
        ax.hlines(
            ctr_y, ctr_x - 20, ctr_x + 20, colors=colour, linewidths=width, zorder=z
        )
        ax.vlines(
            ctr_x, ctr_y - 20, ctr_y + 20, colors=colour, linewidths=width, zorder=z
        )
    if det.beam_center is not None:
        px, py = det.beam_center
        ax.hlines(py, px - 10, px + 10, colors="yellow", linewidths=1)
        ax.vlines(px, py - 10, py + 10, colors="yellow", linewidths=1)
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    ax.set_title(title)
    ax.invert_xaxis()
    return handle


def figure_q_rings(det: DetectorMean, *, remove_outliers: bool = True) -> Figure:
    """``XPCS q-rings`` (``Setup.show_image_qroi``): mean image, then the ring map.

    Left, the mean detector image (log colours from 1e-4, jet, NaN gray, front
    view, black/white cross at the geometry centre and yellow at the beam
    centre); right, the same image under the ring ROIs at alpha 0.4.

    **Differs from the old figure** in what is shown: the photon store holds the
    annulus the rings span (plus the envelope margin) and nothing of the
    rest of the detector, so only ring pixels that survive the static mask are
    coloured, and the mean is over ``det.n_trains`` hit trains instead of the
    whole run. The ring layout itself is the stored one, not a recomputation.

    :param remove_outliers: drop the pixels :func:`outlier_ratio` flags before
        averaging, as the old figure did (it showed ``mean_pixelcell`` after
        ``mask_outlier``). Without it a handful of hot pixels set the colour
        scale.
    """
    mean = det.mean_image.copy()
    if remove_outliers and det.outlier is not None:
        mean[det.outlier] = np.nan
    elif remove_outliers:
        ratio = outlier_ratio(det)
        mean[(ratio < OUTLIER_THRESHOLDS[0]) | (ratio > OUTLIER_THRESHOLDS[1])] = np.nan
    image, centre = _assemble(det, mean)
    fig = Figure(figsize=(14, 6))
    ax = fig.subplots(1, 2)
    norm = LogNorm(vmin=1e-4)
    handle = _draw_image(
        ax[0], image, centre, det, title="Mean detector image", norm=norm
    )
    cax = make_axes_locatable(ax[0]).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(handle, cax=cax)

    n_rings = int(det.ring.max()) + 1 if det.ring.size else 0
    _draw_image(ax[1], image, centre, det, title="")
    rings, _ = _assemble(det, det.ring.astype(np.float32) + 1.0, fill=0.0)
    colours = []
    for name in ("Paired", "tab20", "Set3"):
        colours += list(colormaps[name].colors)
    while n_rings > len(colours):
        colours += colours
    cmap = ListedColormap(["white", *colours[:n_rings]])
    cmap.set_bad(color="0.25", alpha=1.0)
    overlay = ax[1].imshow(
        rings,
        origin="lower",
        interpolation="none",
        extent=(-0.5, image.shape[1] + 0.5, -0.5, image.shape[0] + 0.5),
        cmap=cmap,
        norm=BoundaryNorm(np.arange(-0.5, n_rings + 1.5, 1), cmap.N),
        alpha=0.40,
    )
    ax[1].set_title(f"Rois in {n_rings} bins")
    cax = make_axes_locatable(ax[1]).append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(overlay, cax=cax, ticks=np.arange(1, n_rings + 1))
    cbar.minorticks_off()
    fig.suptitle(
        f"run {det.run}: {det.n_trains} hit trains, ring pixels of the photon store",
        fontsize=9,
    )
    return fig


def _saxs_axes(ax, saxs: xr.DataArray, bounds_A, title, threshold=None) -> None:
    """``extra_speckle.saxs.plot.plot`` for an ``(frame, q)`` series."""
    cmap = colormaps["jet"]
    frames = saxs["frame"].values
    norm = Normalize(vmin=float(frames.min()), vmax=float(frames.max()))
    for frame in frames:
        ax.plot(saxs["q"].values, saxs.sel(frame=frame).values, color=cmap(norm(frame)))
    bounds = np.asarray(bounds_A).reshape(-1, 2)
    if len(bounds) > 1:
        for i, (lo, hi) in enumerate(bounds):
            ax.axvspan(lo, hi, facecolor=cmap(i / float(len(bounds) + 1)), alpha=0.5)
    elif len(bounds) == 1:
        ax.vlines(
            bounds[0],
            np.nanmin(saxs.values),
            np.nanmax(saxs.values),
            ls="--",
            color="gray",
        )
    if threshold is not None and threshold > 0:
        ax.axhline(y=threshold, ls="--", color="gray")
    ax.set_title(title)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.set_xlabel(r"$q$ / [$\AA^{-1}$]", fontsize="x-large")
    ax.set_ylabel(r"$I(q)$ / [ph/px]", fontsize="x-large")
    ax.set_xscale("log")
    ax.set_yscale("log")
    colourbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colourbar.set_label("frame")


def figure_outliers(
    det: DetectorMean,
    *,
    thresholds: tuple[float, float] = OUTLIER_THRESHOLDS,
    dq_A: float = Q_BIN_A,
) -> Figure:
    """``XPCS outliers`` (``mask_outlier``): I(q) before and after, and the statistic.

    Left and middle, the per-frame I(q) with every ring pixel, then with the
    pixels outside ``thresholds`` removed; right, the histogram of
    :func:`outlier_ratio` with the thresholds dashed.

    **Differs from the old figure:** the statistic comes from ``det.n_trains``
    hit trains, not the run's 3000. Per-pixel means at 0.01 photon a frame are
    Poisson-dominated, so with few trains the histogram is wider than the old
    one and the same thresholds flag more pixels: read the title's train count,
    and do not compare the number of removed pixels with the old log.

    When the results file carries the pass's applied mask (``det.outlier``),
    that mask and its statistic (over every usable hit train) are what is
    shown, and ``thresholds`` and ``dq_A`` only bin the curves. Its NaN ratios
    are the pixels ``image.mask`` flags in every frame of every hit train:
    ``MaskPolicy.ANY_FRAME`` keeps them out of every sum, so neither panel shows
    them (the store's photons carry no flags, and they were r0424's spikes).
    """
    if det.outlier is not None:
        ratio, removed = det.outlier_ratio, det.outlier
        thresholds, n_trains, source = (
            det.outlier_thresholds,
            det.outlier_trains,
            "applied",
        )
    else:
        ratio = outlier_ratio(det, dq_A=dq_A, thresholds=thresholds)
        removed = (ratio < thresholds[0]) | (ratio > thresholds[1])
        n_trains, source = det.n_trains, "not applied, recomputed"
    lower, higher = thresholds
    correlated = (
        np.isfinite(ratio) if det.outlier is not None else np.ones_like(removed)
    )
    fig = Figure(figsize=(16, 6))
    ax = fig.subplots(1, 3)
    before = q_curves(det, dq_A=dq_A, pixels=correlated)
    after = q_curves(det, dq_A=dq_A, pixels=correlated & ~removed)
    _saxs_axes(ax[0], before, det.ring_bounds_A, "before outlier removal")
    _saxs_axes(ax[1], after, det.ring_bounds_A, "after outlier removal")
    ax[2].hist(ratio[np.isfinite(ratio)], bins=np.logspace(-1, 3, 1000))
    ax[2].vlines(x=thresholds, ymin=1, ymax=10**5, ls="--", color="gray")
    ax[2].set_yscale("log")
    ax[2].set_xscale("log")
    ax[2].set_title("Distribution of std_of_mean and outlier threshold")
    fig.suptitle(
        f"run {det.run}: {int(removed.sum())} of {ratio.size} ring pixels outside "
        f"({lower}, {higher}), from {n_trains} hit trains ({source}); "
        f"{int((~correlated).sum())} flagged in every train not shown",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def figure_beamcenter(det: DetectorMean, *, dq_A: float = Q_BIN_A) -> Figure:
    """``XPCS beamcenter``: the cone test of ``fit_saxs_beamcenter``, without the fit.

    **Not reproducible as a fit.** The old figure was produced only when
    ``refine_beamcenter`` was on, which the production call never set
    (``args["refine_beamcenter"] = False``), so the old column was empty. A fit
    moves the centre and re-integrates the *whole* detector image at every
    step; the store holds the annulus only and the geometry's beam centre
    (``cfg.saxs.beam_center_*``, open task 4). What the store does support is
    the old fit's merit function at the configured centre, which is also the
    decisive check of open task 4: I(q) in eight azimuthal cones should agree if
    the centre is right.

    Left, I(q) per cone over the ring range, coloured by cone (the old "before
    refinement" panel, titled with the centre). Right, per ring the relative
    deviation of each cone's mean from the ring mean: flat when the centre is
    right, sinusoidal in the cone number when it is off. The title carries the
    old merit, ``mean over q of var over cones of log I``.
    """
    q_bounds = (float(det.ring_bounds_A[0, 0]), float(det.ring_bounds_A[-1, 1]))
    cones = cone_curves(det, q_bounds, dq_A=5 * dq_A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        merit = float(np.nanmean(np.nanvar(np.log(cones.values), axis=0)))
    fig = Figure(figsize=(14, 5.5))
    ax = fig.subplots(1, 2)
    cmap = colormaps["jet"]
    norm = Normalize(0, N_CONES - 1)
    for c in range(N_CONES):
        ax[0].plot(cones["q"].values, cones.values[c], color=cmap(norm(c)))
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$q$ / [$\AA^{-1}$]", fontsize="x-large")
    ax[0].set_ylabel(r"$I(q)$ / [ph/px]", fontsize="x-large")
    ax[0].grid(True, which="both", alpha=0.3)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax[0])
    cb.set_label("q_slice")
    centre = (
        f"({det.beam_center[0]:.2f}, {det.beam_center[1]:.2f})"
        if det.beam_center
        else "geometry origin"
    )
    ax[0].set_title(f"At beam center {centre}")

    cone = np.floor((det.chi_deg % 360.0) / (360.0 / N_CONES)).astype(int)
    mean = det.mean_image
    for r in range(int(det.ring.max()) + 1):
        in_ring = det.ring == r
        ring_mean = mean[in_ring].mean()
        deviation = [
            mean[in_ring & (cone == c)].mean() / ring_mean - 1.0
            if (in_ring & (cone == c)).any()
            else np.nan
            for c in range(N_CONES)
        ]
        ax[1].plot(
            np.arange(N_CONES),
            deviation,
            "o-",
            color=_jet(r, int(det.ring.max()) + 1),
            label=_ring_label(det.ring_bounds_A[r].mean())
            if len(det.ring_bounds_A) > r
            else f"ring {r}",
        )
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("azimuthal cone")
    ax[1].set_ylabel("mean / ring mean − 1")
    ax[1].legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle(
        f"run {det.run}: cone merit {merit:.3g} (old fit minimised this); "
        f"{det.n_trains} hit trains, no refinement",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


# ── figures that come from the results file ───────────────────────────────────
def figure_intensity_outliers(
    hits: XpcsHits,
    xgm: xr.DataArray | np.ndarray | None = None,
    *,
    out_thresh: float = INTENSITY_OUTLIER_FRACTION,
) -> Figure:
    """``XPCS intensity outliers``: the mean intensity of each hit train.

    Without ``xgm`` this is the old ``plot_intensity`` (what the production call
    drew, since ``intensity_masking`` was off): blue dots against train
    position. **The y axis differs:** the old one was the mean of the whole
    masked detector, here it is the mean over ring pixels
    (:func:`mean_intensity`); the train axis runs over the hit trains, not
    every train of the run.

    With ``xgm`` (per hit train, aligned by position or as a ``trainId``
    DataArray) it is ``mask_intensity_outliers``: intensity over XGM, trains
    further than ``out_thresh`` from the median in red, the ±20 % band dashed.
    """
    intensity = mean_intensity(hits)
    fig = Figure()
    ax = fig.subplots()
    run = _run_label(hits)
    if xgm is None:
        ax.plot(intensity.values, "o", c="blue", ms=2)
        ax.set_xlabel("train")
        ax.set_ylabel("intensity, a.u.")
        ax.set_title(f"Run {run}")
        ax.grid()
        return fig
    if isinstance(xgm, xr.DataArray):
        xgm = xgm.reindex(trainId=intensity["trainId"]).values
    norm = intensity.values / np.asarray(xgm, dtype=float)
    median = np.nanmedian(norm)
    bad = np.abs(norm - median) / median > out_thresh
    index = np.arange(norm.size)
    ax.scatter(index, np.where(bad, np.nan, norm), c="blue", s=5, label="good trains")
    ax.scatter(index, np.where(bad, norm, np.nan), c="red", s=5, label="removed trains")
    ax.axhline(y=(1 - out_thresh) * median, ls="--", color="gray")
    ax.axhline(y=(1 + out_thresh) * median, ls="--", color="gray")
    ax.set_xlim(0, max(index.max(), 1))
    ax.set_xlabel("train index")
    ax.set_ylabel("normalized intensity")
    ax.set_title(f"run {run}")
    ax.legend()
    return fig


def figure_kbar(hits: XpcsHits, threshold: float | None = ES_KBAR_MIN) -> Figure:
    """``XPCS kbar`` (``plot_kbar``): k̄ against train, and its histogram per ring.

    Left, every frame's k̄ of every ring against train id (log y, the
    threshold as a thick dashed line); right, a step histogram per ring on
    1000 log bins. Exact reproduction of the old figure's content: k̄ is the
    mean photons per good ring pixel and frame, as extra-speckle's ``kbar``.
    The default threshold is extra-speckle's ``kbar_threshold`` (3e-3).
    """
    kbar = hits.kbar()
    ids = kbar["trainId"].values
    values = kbar.values
    fig = Figure(figsize=(10, 5))
    ax1, ax2 = fig.subplots(1, 2)
    for frame in range(values.shape[1]):
        ax1.plot(
            ids, values[:, frame, :], marker=".", linestyle="None", rasterized=True
        )
    if threshold is not None and ids.size:
        ax1.hlines(threshold, ids.min(), ids.max(), color="k", linewidth=3, ls="--")
    ax1.set_yscale("log")
    ax1.set_xlabel("trainId", fontsize="x-large")
    ax1.set_ylabel(r"$\langle k \rangle$", fontsize="x-large")

    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if positive.size:
        bins = np.logspace(np.log10(positive.min()), np.log10(finite.max()), 1000)
        for i, q in enumerate(kbar["roi_q"].values):
            ax2.hist(
                values[:, :, i].ravel(),
                bins=bins,
                label=rf"$q = {q:.4f} \ \mathrm{{\AA}}^{{-1}}$",
                histtype="step",
            )
    if threshold is not None:
        ax2.vlines(threshold, *ax2.get_ylim(), color="k", linewidth=3, ls="--")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\langle k \rangle$", fontsize="x-large")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


def _correction_data(hits: XpcsHits, averaging: Averaging):
    """The three maps and three diagonal curves of the baseline figure."""
    maps = hits.ttcf(averaging)
    lags = hits.lags()
    if averaging is Averaging.POOLED:
        g2 = hits.g2(averaging, binned=False)
        curves = (g2["g2"].values, g2["baseline"].values, g2["c2"].values)
        tau = g2["tau"].values
    else:
        curves = tuple(
            _diagonal_curves(maps[n].values, lags) for n in ("raw", "off", "c2")
        )
        tau = lags * hits.period_s
    return maps, curves, tau


def figure_correction(
    hits: XpcsHits, averaging: Averaging = Averaging.POOLED
) -> Figure:
    """``XPCS correction`` (``baseline_subtraction_and_averaging``): 2 × 3 panels.

    Top, the two-time map of the middle ring for the raw map, the baseline
    (``off``) and their difference; bottom, g2 of every ring from the raw
    map, the baseline estimation and the corrected map. Same layout, colour
    scales (quantiles 0.02 and 0.90 of the upper triangle), titles and axes as
    extra-speckle's.

    With ``Averaging.POOLED`` the curves are :meth:`XpcsHits.g2` unbinned
    (``g2``, ``baseline``, ``c2``) and the maps those of :meth:`XpcsHits.ttcf`
    (``raw``, ``off``, ``c2``), so *corrected* is ``raw_paired − off`` over the
    trains that have a baseline, not ``raw − off`` of two different train sets.
    With ``Averaging.ES`` both come from extra-speckle's mean of per-train maps.
    """
    maps, curves, tau = _correction_data(hits, averaging)
    roi = hits.roi_q
    mid = len(roi) // 2
    frames = maps["frame_1"].values
    fig = Figure(figsize=(18, 8))
    ax = fig.subplots(2, 3)
    top = (("(raw)", "raw"), ("(off)", "off"), ("(raw-off)", "c2"))
    for col, (title, name) in enumerate(top):
        _ttcf_panel(
            ax[0, col],
            maps[name].values[mid],
            f"{title}, {_ring_label(roi[mid])}",
            frames,
        )
    for col, (title, values) in enumerate(
        zip(("(raw)", "baseline estimation", "corrected"), curves, strict=True)
    ):
        _g2_panel(ax[1, col], roi, tau, values, title=title)
    fig.tight_layout()
    return fig


def figure_ttcfs(
    hits: XpcsHits,
    averaging: Averaging = Averaging.POOLED,
    maps: xr.Dataset | None = None,
) -> Figure:
    """``XPCS TTCF's`` (``xpcs.plot`` of the corrected TTCF): one panel per ring.

    The baseline-subtracted two-time map ``c2`` of every ring on a grid of up to
    three columns, jet from 0 (``vmin=0``) to the 0.90 quantile of the upper
    triangle, titled ``TTCF run N, q Å⁻¹``.

    :param maps: an already computed :meth:`XpcsHits.ttcf`.
    """
    maps = maps if maps is not None else hits.ttcf(averaging)
    roi = maps["roi_q"].values
    frames = maps["frame_1"].values
    ncols = min(len(roi), 3)
    nrows = int(np.ceil(len(roi) / ncols))
    fig = Figure(figsize=(ncols * 4, nrows * 4))
    ax = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
    run = _run_label(hits)
    for i, q in enumerate(roi):
        _ttcf_panel(
            ax[i],
            maps["c2"].values[i],
            f"TTCF run {run}, {_ring_label(q)}",
            frames,
            vmin=0,
        )
    for extra in ax[len(roi) :]:
        extra.axis("off")
    fig.tight_layout()
    return fig


def _g2_series(g2: xr.Dataset) -> tuple[np.ndarray, np.ndarray | None]:
    """Baseline-subtracted g2 and its error, whichever averaging made ``g2``."""
    if "c2" in g2:
        return g2["c2"].values, g2["sigma"].values
    return g2["g2"].values, None


def figure_g2(
    hits: XpcsHits,
    averaging: Averaging = Averaging.POOLED,
    *,
    g2: xr.Dataset | None = None,
    errors: bool = False,
) -> Figure:
    """``XPCS g2`` (``xpcs.plot`` of g2): baseline-subtracted g2 against τ.

    One curve per ring, jet colours fading with ring number, a marker per ring,
    log τ in seconds, legend outside to the right, titled ``run N``; the binning
    is extra-speckle's ``g2_to_bins(50, 20, 10)`` (:meth:`XpcsHits.g2`).

    :param errors: add the jackknife error bars (not in the old figure; pooled
        only, ES has no error).
    :param g2: an already computed :meth:`XpcsHits.g2`.
    """
    g2 = g2 if g2 is not None else hits.g2(averaging)
    values, sigma = _g2_series(g2)
    fig = Figure(figsize=(10, 7))
    ax = fig.subplots()
    _g2_panel(
        ax,
        g2["roi_q"].values,
        g2["tau"].values,
        values,
        title=f"run {_run_label(hits)}",
        errors=sigma if errors else None,
    )
    fig.tight_layout()
    return fig


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

    **One deliberate difference.** extra-speckle fitted every ring it was given,
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


def figure_g2_fit(
    hits: XpcsHits,
    averaging: Averaging = Averaging.POOLED,
    *,
    g2: xr.Dataset | None = None,
) -> Figure:
    """``XPCS g2 fit`` (``xpcs.fit(dr=True)``): data and fit, β(q), Γ(q²).

    Left, g2 as markers with the fit as lines (titled ``(fitted) run N``);
    middle, ``β`` against q with its fit error; right, ``Γ`` against q² with
    the ``Γ = a q²`` line and ``R_H`` in the legend. The model, the start values
    and the dispersion fit are extra-speckle's (:func:`fit_g2`); it has no KWW
    model (open task 10) and neither does this.

    :raises ValueError: the fit failed. The old pipeline caught that and left
        the column empty, so callers turn it into "no figure".
    """
    g2 = g2 if g2 is not None else hits.g2(averaging)
    values, _ = _g2_series(g2)
    fit = fit_g2(
        xr.DataArray(
            values,
            dims=("roi_q", "tau"),
            coords={"roi_q": g2["roi_q"].values, "tau": g2["tau"].values},
        )
    )
    fig = Figure(figsize=(13.5, 5))
    ax = fig.subplots(1, 3)
    _g2_panel(
        ax[0],
        fit.q,
        fit.tau,
        values,
        title=f"(fitted) run {_run_label(hits)}",
        fit=fit.curve,
    )
    ax[1].errorbar(
        fit.q, fit.beta, fit.err_beta, linestyle="None", marker="o", color="k"
    )
    ax[1].set_xlabel(r"$q, \mathrm{\AA}^{-1}$", fontsize=13)
    ax[1].set_ylabel(r"$\beta $", fontsize=13)
    ax[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    q2 = fit.q**2
    ax[2].errorbar(
        q2, fit.gamma, yerr=fit.err_gamma, linestyle="None", marker="o", color="k"
    )
    fitted = np.isfinite(q2)
    rr = np.linspace(0.9 * np.nanmin(q2[fitted]), 1.1 * np.nanmax(q2[fitted]), 101)
    ax[2].plot(
        rr,
        rr * fit.dr_a,
        label=rf"$R_H = {fit.rh_A:.1f} +- {fit.rh_err_A:.1f} \ \mathrm{{\AA}}$",
    )
    ax[2].legend(loc=4, fontsize=13, frameon=False)
    ax[2].set_ylabel(r"$\Gamma, \mathrm{s}^{-1}$", fontsize=13)
    ax[2].set_xlabel(r"$q^2, \mathrm{\AA}^{-2}$", fontsize=13)
    fig.tight_layout()
    return fig


def figure_saxs(
    hits: XpcsHits,
    det: DetectorMean | None = None,
    *,
    threshold: float | None = ES_KBAR_MIN,
) -> Figure:
    """``XPCS SAXS overview`` (``saxs.plot``): I(q) per frame, log-log, jet by frame.

    The ring edges are shaded as the old ROIs and the k̄ threshold is a
    dashed line, titled ``I(q) per pulseId``. With ``det`` (the photon store's
    annulus) the curves are :func:`saxs_curves`, q-binned, but cover only the
    annulus ``q = 0.02`` to ``0.09`` Å⁻¹ and not the detector's 0.008 to 0.105;
    without it they are the seven ring means of :func:`saxs_rings`, which is all
    the results file can give. The title says which.
    """
    saxs = saxs_rings(hits) if det is None else saxs_curves(det)
    fig = Figure()
    ax = fig.subplots()
    _saxs_axes(
        ax,
        saxs,
        hits.bounds_nm / 10.0,
        f"I(q) per pulseId ({saxs.attrs['resolution']})",
        threshold,
    )
    fig.tight_layout()
    return fig


#: The figure of each old column, for callers that loop over them.
FIGURES: dict[str, Callable[..., Figure]] = {
    "q_rings": figure_q_rings,
    "beamcenter": figure_beamcenter,
    "outliers": figure_outliers,
    "intensity_outliers": figure_intensity_outliers,
    "kbar": figure_kbar,
    "correction": figure_correction,
    "ttcfs": figure_ttcfs,
    "g2": figure_g2,
    "g2_fit": figure_g2_fit,
    "saxs": figure_saxs,
}
