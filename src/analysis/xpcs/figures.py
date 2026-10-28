"""extra-speckle's ``xpcs_offline`` figures, drawn from the hit-train XPCS products.

Each function reproduces one extra-speckle figure from what the hit-train pass
keeps: the results file (:class:`~analysis.xpcs.reader.XpcsHits`) and, for
detector pictures, the photon store's mean over hit trains
(:class:`~analysis.xpcs.detector.DetectorMean`). The numbers come from
`analysis.xpcs.reader`, `analysis.xpcs.legacy`, `analysis.xpcs.detector` and
`analysis.xpcs.g2fit`; this module only draws. Figures are built on
:class:`matplotlib.figure.Figure` without ``pyplot``, so no backend is needed
and no figure enters global state; they are returned open for the caller to
show or save.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import xarray as xr
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.saxs.config import SHAPE
from analysis.xpcs.detector import (
    N_CONES,
    OUTLIER_THRESHOLDS,
    Q_BIN_A,
    DetectorMean,
    cone_curves,
    cone_index,
    outlier_ratio,
    q_curves,
    run_label,
)
from analysis.xpcs.estimators import diagonal_means
from analysis.xpcs.g2fit import fit_g2
from analysis.xpcs.legacy import mean_intensity, saxs_rings
from analysis.xpcs.reader import ES_KBAR_MIN, Averaging, XpcsHits

__all__ = [
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
]

#: extra-speckle's intensity-outlier fraction, as ``xpcs_offline`` was called
#: with it.
INTENSITY_OUTLIER_FRACTION = 0.2

_MARKERS = ("o", "v", "^", "<", ">", "H", "D", "*")


def _ring_label(roi: float) -> str:
    return rf"{np.around(roi, 3)} $\AA^{{-1}}$"


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


def _assemble(det: DetectorMean, values: np.ndarray, fill: float = np.nan):
    """Assemble per-ring-pixel ``values`` into the detector picture.

    Module pixels without a value get ``fill``; the gaps between modules are NaN.
    """
    image = np.full(SHAPE[0] * SHAPE[1], fill, np.float32)
    image[det.pixel] = values
    return det.geometry.position_modules(image.reshape(16, 512, 128))


def _draw_image(ax, image, centre, det, *, title, norm=None, cmap="jet", **kwargs):
    """As ``Setup.plot_data``: front view, NaN gray, geometry and beam crosses."""
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

    It shows less than extra-speckle's figure. The photon store holds the
    annulus the rings span (plus the envelope margin) and nothing of the rest
    of the detector, so only ring pixels that survive the static mask are
    coloured, and the mean is over ``det.n_trains`` hit trains instead of the
    whole run. The ring layout is the stored one, not a recomputation.

    :param remove_outliers: drop the outlier pixels before averaging (the
        applied mask, else :func:`outlier_ratio` recomputed), as extra-speckle
        showed ``mean_pixelcell`` after ``mask_outlier``. Without it a few hot
        pixels set the colour scale.
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

    The statistic comes from ``det.n_trains`` hit trains, not every train of
    the run. Per-pixel means at 0.01 photon a frame are Poisson-dominated, so
    with few trains the histogram is wider than extra-speckle's and the same
    thresholds flag more pixels. Do not compare the number of removed pixels
    with extra-speckle's log without the train count in the title.

    When the results file carries the pass's applied mask (``det.outlier``),
    that mask and its statistic (over every usable hit train) are shown, and
    ``thresholds`` and ``dq_A`` only bin the curves. Neither curve panel
    includes ``det.always_flagged``. ``MaskPolicy.ANY_FRAME`` keeps those
    pixels out of every sum, but the store's photons carry no flags, so they
    would show as spikes. The histogram leaves out NaN ratios, which come from
    a q bin whose median is zero.
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
        ~det.always_flagged
        if det.always_flagged is not None
        else np.ones(removed.size, bool)
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

    A fit moves the centre and re-integrates the whole detector image at every
    step, and the store holds the annulus only, so no fit is made. The store
    does support the fit's merit function at the configured centre
    (``cfg.saxs.beam_center_*``): I(q) in ``N_CONES`` azimuthal cones agrees
    between cones when the centre is right.

    Left, I(q) per cone over the ring range, coloured by cone (extra-speckle's
    "before refinement" panel, titled with the centre). Right, per ring the
    relative deviation of each cone's mean from the ring mean, flat when the
    centre is right and sinusoidal in the cone number when it is off. The title
    carries the merit, ``mean over q of var over cones of log I``.
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

    cone = cone_index(det)
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


def figure_intensity_outliers(
    hits: XpcsHits,
    xgm: xr.DataArray | np.ndarray | None = None,
    *,
    out_thresh: float = INTENSITY_OUTLIER_FRACTION,
) -> Figure:
    """``XPCS intensity outliers``: the mean intensity of each hit train.

    Without ``xgm`` this is extra-speckle's ``plot_intensity``, blue dots
    against train position. The y axis is the mean over ring pixels
    (:func:`mean_intensity`) where extra-speckle used the whole masked detector,
    and the train axis runs over the hit trains only.

    With ``xgm`` (per hit train, aligned by position or as a ``trainId``
    DataArray) it is ``mask_intensity_outliers``: intensity over XGM, trains
    further than ``out_thresh`` from the median in red, and the ``±out_thresh``
    band dashed.
    """
    intensity = mean_intensity(hits)
    fig = Figure()
    ax = fig.subplots()
    run = run_label(hits)
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
    1000 log bins. The content is exactly extra-speckle's: k̄ is the mean
    photons per good ring pixel and frame, as its ``kbar``.
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
            diagonal_means(maps[n].values, lags) for n in ("raw", "off", "c2")
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
    run = run_label(hits)
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

    :param errors: add the jackknife error bars, which extra-speckle did not
        draw (pooled only; ES has no error).
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
        title=f"run {run_label(hits)}",
        errors=sigma if errors else None,
    )
    fig.tight_layout()
    return fig


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
    and the dispersion fit are extra-speckle's (:func:`fit_g2`), which has no
    KWW model.

    :raises ValueError: the fit failed. Callers show no figure, as extra-speckle
        left its column empty.
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
        title=f"(fitted) run {run_label(hits)}",
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

    The ring bounds are shaded as extra-speckle's ROIs and the k̄ threshold is a
    dashed line, titled ``I(q) per pulseId``. With ``det`` (the photon store's
    annulus) the curves are :func:`~analysis.xpcs.detector.q_curves`, q-binned
    but covering only the annulus, not the whole detector; without it they are
    the ring means of :func:`~analysis.xpcs.legacy.saxs_rings`, all the results
    file can give. The title says which.
    """
    saxs = saxs_rings(hits) if det is None else q_curves(det)
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
