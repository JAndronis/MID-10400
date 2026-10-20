"""The DAMNIT cells of the diagnostics: frames used (phase D), droplet videos (E).

Each ``*_frames_used`` variable reads a pass's frame table straight from the
path its config names. It never depends on the pass's DAMNIT variable, so it
cannot relaunch the cluster job (CLAUDE.md pitfall 26). The cost is that a cell
shows the file as it was when the cell ran. After a reprocess, refresh it with
``damnit reprocess --match "frames used"`` (``--match`` compares titles, not
names, in damnit 0.4.0).

The context-file wrappers are an import plus a call, like the integrators'.
DAMNIT's ``Cell`` and ``Skip`` are imported inside the function, so this module
imports without DAMNIT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.common.status import FrameStatus

__all__ = [
    "COLORS",
    "DETECTORS",
    "diagnostics_overview",
    "droplet_first_image",
    "droplet_video",
    "frames_used",
    "frames_used_cell",
    "output_file",
    "photon_statistics",
    "photon_statistics_overview",
    "xpcs_score_cell",
]

#: Cell backgrounds (p010646's context, transfer_context/templates).
COLORS = {"good": "#33e32a", "warning": "#f2d40f", "error": "#ec0839"}

#: The passes with a frame table, by the detector name the wrappers use.
DETECTORS = ("agipd", "jf1", "jf2")

#: A frame with one of these codes means the pass went wrong, not the data.
FAILED = (FrameStatus.WORKER_ERROR, FrameStatus.LABEL_MISMATCH)

#: HDF5's words for a file another process holds open for writing.
_LOCKED = ("unable to lock file", "already open", "resource temporarily unavailable")


def output_file(proposal: int, run_no: int, detector: str, **overrides: Any) -> Path:
    """The frame table the pass writes for this run, from its own config.

    :param detector: ``agipd``, ``jf1`` or ``jf2``.
    :param overrides: config fields, e.g. ``output_root``.
    """
    if detector == "agipd":
        from analysis.saxs.damnit import config_for

        return config_for(proposal, run_no, **overrides).output_file
    if detector not in DETECTORS:
        raise ValueError(f"detector must be one of {DETECTORS}, got {detector!r}")
    from analysis.waxs.config import config_for

    return config_for(proposal, run_no, detector, **overrides).output_file


def frames_used(proposal: int, run_no: int, detector: str, **overrides: Any) -> Any:
    """The ``Cell`` of a ``*_frames_used`` variable: OK frames over all rows."""
    return frames_used_cell(output_file(proposal, run_no, detector, **overrides))


def frames_used_cell(path: str | Path) -> Any:
    """``used / total (%)`` of a frame table, coloured by its status codes.

    - green: every frame OK;
    - red: any WORKER_ERROR or LABEL_MISMATCH, or a file HDF5 cannot read;
    - amber: anything else (NOT_PROCESSED, i.e. incomplete; MISSING_MODULES;
      NO_FRAMES; DATA_CHECK_FAILED), or a file another process holds open.

    When not green, the count is followed by the code that decided the colour
    (the commonest failure on red, the commonest non-OK code on amber).

    "busy" needs HDF5 file locking, and inside DAMNIT it is off: the context
    imports ``extra_speckle``, whose ``xcca.tools`` sets
    ``HDF5_USE_FILE_LOCKING=FALSE`` for the whole process. There a file being
    written reads as it stands (amber, NOT_PROCESSED rows), or red if HDF5 trips
    over it mid-write; refresh the cell once the pass has finished.

    :raises Skip: no file, i.e. the pass has not written this run.
    """
    from damnit.context import Cell, Skip

    path = Path(path)
    if not path.exists():
        raise Skip(f"no frame table at {path}")
    try:
        with h5py.File(path, "r") as f:
            status = f["frames/status"][:]
    except OSError as error:
        text = str(error)
        if any(word in text.lower() for word in _LOCKED):
            return Cell(None, summary_value="busy", background=COLORS["warning"])
        return Cell(text, summary_value="unreadable", background=COLORS["error"])

    total = int(status.size)
    used = int((status == FrameStatus.OK).sum())
    pct = 100.0 * used / total if total else 0.0
    summary = f"{used} / {total} ({pct:.1f}%)"
    if used == total and total:
        return Cell(used, summary_value=summary, background=COLORS["good"])
    failed = np.isin(status, np.asarray(FAILED, dtype=status.dtype))
    shown = status[failed] if failed.any() else status[status != FrameStatus.OK]
    if shown.size:
        codes, counts = np.unique(shown, return_counts=True)
        summary += f" · {_name(codes[counts.argmax()])} {int(counts.max())}"
    return Cell(
        used,
        summary_value=summary,
        background=COLORS["error"] if failed.any() else COLORS["warning"],
    )


def _name(code) -> str:
    try:
        return FrameStatus(int(code)).name
    except ValueError:
        return f"code {int(code)}"


#: Threads preparing video frames in the DAMNIT job (explicit, pitfall 10).
VIDEO_WORKERS = 16


def droplet_video(run, run_no: int, *, root: str | Path | None = None,
                  workers: int = VIDEO_WORKERS) -> Any:  # fmt: skip
    """The ``droplet_video`` cell: write the run's CAM5 video and check it.

    The video is decoded again after writing; a frame count that differs from
    the number of trains with an image raises.

    :param run: DAMNIT's raw run (the camera is a raw source).
    :returns: a ``Cell`` with the path and "N frames, G gaps".
    :raises Skip: the run has no image of the camera.
    """
    from damnit.context import Cell, Skip

    from analysis import camera

    facts = camera.camera_facts(run)
    if not facts["n_images"]:
        raise Skip(f"no images of {camera.CAMERA_SOURCE}")
    path = camera.video_path(run_no, root=root or camera.VIDEO_ROOT)
    info = camera.write_camera_video(run, path, label=f"r{run_no:04d}", workers=workers)
    n = camera.count_frames(path)
    if n != info["frames"]:
        raise RuntimeError(f"{path} decodes to {n} frames, {info['frames']} written")
    return Cell(str(path), summary_value=f"{n} frames, {info['gaps']} gaps")


def droplet_first_image(run) -> Any:
    """The run's first CAM5 image, reading only that image.

    :returns: a (y, x) DataArray with its ``trainId`` and ``source`` in attrs,
        which DAMNIT shows as a thumbnail.
    :raises Skip: the run has no image of the camera.
    """
    import xarray as xr
    from damnit.context import Skip

    from analysis import camera

    if camera.CAMERA_SOURCE not in run.all_sources:
        raise Skip(f"no {camera.CAMERA_SOURCE} in this run")
    try:
        train_id, image = camera.first_image(run)
    except ValueError as error:
        raise Skip(str(error)) from None
    return xr.DataArray(
        image,
        dims=("y", "x"),
        attrs={"trainId": train_id, "source": camera.CAMERA_SOURCE},
    )


# ── photon statistics (agipd_preprocessing writes them; xpcs-quality.md) ──────


def photon_statistics(proposal: int, run_no: int, saxs, xgm, transmission) -> Any:
    """The ``photon_statistics`` column: k̄ and β_sf per XPCS ROI and ring, over hits.

    Reads the statistics ``agipd_preprocessing`` wrote. The droplet hits are
    classified from DAMNIT's values in memory (``agipd_saxs``, ``xgm_intensity``,
    ``total_transmission``), because inside the cluster job this run's values
    are not in DAMNIT's file yet. The per-pixel weights come from the window
    sums: every window for a static run, the windows wholly inside the hits for
    a mesh run; without window sums or masks the estimator is unweighted, and
    ``attrs["weighted"]`` says so.

    :raises Skip: no statistics file, or no hit train.
    """
    from damnit.context import Skip

    from analysis.diagnostics.photon_stats import run_weights, summarise
    from analysis.saxs.damnit import config_for
    from analysis.triage.classify import hits, signals_from

    cfg = config_for(proposal, run_no)
    path = cfg.photon_stats_file
    if not path.exists():
        raise Skip(f"no photon statistics at {path}: see agipd_preprocessing")
    h = hits(run_no, proposal=proposal, signals=signals_from(saxs, xgm, transmission))
    trains = h["train_ids"]
    if not trains.size:
        raise Skip(f"r{run_no}: no droplet hit")
    masks = cfg.output_file.with_name("masks.npz")
    weights = None
    if cfg.pixel_sums_file.exists() and masks.exists():
        try:
            weights = run_weights(
                path,
                pixel_sums_path=cfg.pixel_sums_file,
                masks_path=masks,
                trains=trains if h["kind"] == "mesh" else None,
            )
        except ValueError:  # no window wholly inside the hits
            weights = None
    out = summarise(path, weights, trains)
    out.attrs.update(
        kind=h["kind"],
        block=h["block"] or "",
        n_hits=int(trains.size),
        weights_windows=int(weights.attrs["n_windows"]) if weights is not None else 0,
    )
    return out


def photon_statistics_overview(stats) -> Any:
    """Three panels from the ``photon_statistics`` value.

    β_sf per ring against q, with the fixed reference contrast per ROI; the
    pooled P(k) of each ROI against the negative binomial at its β_sf and k̄;
    k̄ per ring against q, with the 3e-3 photon/pixel/frame threshold.
    """
    import matplotlib.pyplot as plt

    from analysis.diagnostics.xpcs import KBAR_MIN
    from analysis.triage.xpcs import roi_edges_nm

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    q = 0.5 * (stats.ring_lo.values + stats.ring_hi.values)
    top = max(0.08, float(np.nanmax(stats.reference_beta)) * 1.6)
    beta = stats.ring_beta.values
    shown = beta <= top
    ax1.errorbar(q[shown], beta[shown], stats.ring_sigma.values[shown], fmt="o",
                 ms=3, label="β_sf")  # fmt: skip
    if (~shown).any():  # outlier rings: marked at the edge, value in the label
        values = ", ".join(f"{b:.2g}" for b in beta[~shown])
        edge = np.full((~shown).sum(), 0.97 * top)
        ax1.plot(q[~shown], edge, "^", color="C3", label=f"off scale: {values}")
    edges = {round(float(lo + hi) / 20.0, 6): (lo, hi) for lo, hi in roi_edges_nm()}
    for i, roi in enumerate(stats.roi.values):
        lo, hi = edges.get(round(float(roi), 6), (np.nan, np.nan))
        ax1.hlines(stats.reference_beta.values[i], lo, hi, colors="k",
                   label="reference" if i == 0 else None)  # fmt: skip
    ax1.set_xlabel("q [nm$^{-1}$]")
    ax1.set_ylabel("single-frame contrast β")
    ax1.set_ylim(0, top)
    ax1.legend(fontsize=8)

    k = stats.k.values
    lowest = 1.0
    rings_of = {}
    q_ring = 0.5 * (stats.ring_lo.values + stats.ring_hi.values)
    for roi in stats.roi.values:
        lo, hi = edges.get(round(float(roi), 6), (np.nan, np.nan))
        rings_of[roi] = np.flatnonzero((q_ring >= lo) & (q_ring < hi))
    for i, roi in enumerate(stats.roi.values):
        hist = stats.ring_hist.values[rings_of[roi]].sum(axis=0).astype(float)
        if hist.sum() == 0:
            continue
        p = hist[:-1] / hist.sum()  # leave out the overflow bin
        lowest = min(lowest, float(p[p > 0].min()))
        (line,) = ax2.semilogy(k[:-1], np.where(p > 0, p, np.nan), "o", ms=3,
                               label=f"{roi:.4g} Å$^{{-1}}$")  # fmt: skip
        beta, kbar = float(stats.beta[i]), float(stats.kbar[i])
        if beta > 0 and kbar > 0:
            m = 1.0 / beta
            ks = np.arange(k.size - 1)
            from scipy.special import gammaln

            nb = np.exp(
                gammaln(ks + m) - gammaln(ks + 1) - gammaln(m)
                + ks * np.log(kbar / (kbar + m)) + m * np.log(m / (kbar + m))
            )  # fmt: skip
            ax2.semilogy(ks, nb, "-", color=line.get_color(), lw=0.8)
    ax2.set_xlim(-0.3, 6.3)
    ax2.set_ylim(lowest / 3, 2.0)
    ax2.set_xlabel("photons per pixel k")
    ax2.set_ylabel("P(k)")
    ax2.legend(fontsize=7)

    ax3.semilogy(q, stats.ring_kbar, "o", ms=3)
    ax3.axhline(KBAR_MIN, color="r", lw=0.8, label=f"{KBAR_MIN:g}")
    ax3.set_xlabel("q [nm$^{-1}$]")
    ax3.set_ylabel("k̄ [photons / pixel / frame]")
    ax3.legend(fontsize=8)
    a = stats.attrs
    weighted = "weighted" if a.get("weighted") else "unweighted"
    fig.suptitle(
        f"{a.get('kind', '')} run, {a.get('n_hits', '?')} hit trains, "
        f"{a.get('n_frames', '?')} frames, {weighted}",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# ── XPCS score and run diagnostics ────────────────────────────────────────────


def xpcs_score_cell(proposal: int, run_no: int) -> Any:
    """The ``xpcs_score`` cell: hits, ROIs at k̄ ≥ 3e-3, best 30 s SNR.

    Reads the run's hit-train results file (``diagnostics.xpcs.xpcs_score``), so it
    shows the run as of its last processing; refresh with
    ``damnit reprocess --match "XPCS score"``. The run is scored only when every
    plain ROI of its results file meets the k̄ threshold (the SNR is taken over
    the plain ROIs only, so a Bragg-adjacent ROI below it does not matter).
    Otherwise it is amber; when they do, SNR determines green, red or amber. The
    ROI counts are the file's, so they follow the rings it was merged with.

    :raises Skip: the run has no hit-train results file yet.
    """
    from damnit.context import Cell, Skip

    from analysis.diagnostics.xpcs import xpcs_score

    try:
        score = xpcs_score(run_no, proposal=proposal)
    except (LookupError, FileNotFoundError) as error:
        raise Skip(str(error)) from None
    snr = score["snr"] or {}
    best = snr.get("best_snr")
    ok = score["n_rois_kbar_ok"]
    plain = score["n_plain_rois_kbar_ok"]
    n_rois, n_plain = score["n_rois"], score["n_plain_rois"]
    parts = [f"{score['n_hits']} hits"]
    if ok is not None:
        parts.append(f"k̄ ok {ok}/{n_rois} ({plain}/{n_plain} plain)")
    parts.append(
        "SNR —" if best is None or not np.isfinite(best) else f"SNR {best:.1f}"
    )
    extractable = snr.get("extractable")
    plain_ok = plain is not None and plain == n_plain
    colour = (
        COLORS["warning"]
        if not plain_ok or extractable is None
        else COLORS["good"]
        if extractable
        else COLORS["error"]
    )
    record = {k: v for k, v in score.items() if k != "kbar"}
    record["snr"] = {k: v for k, v in snr.items() if k != "slices"}
    return Cell(
        json.dumps(record, default=_jsonable),
        summary_value=" · ".join(parts),
        background=colour,
    )


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def diagnostics_overview(proposal: int, run_no: int, saxs, xgm, transmission) -> Any:
    """One figure of the AGIPD run diagnostics over its droplet hits.

    SVD of the per-train curves (divided by the flux and the median curve)
    against the noise floor; χ² of each train against the run median; the
    0.3–0.5 nm⁻¹ band against the flux, before and after dividing by it.
    Inputs as ``photon_statistics``: the frame table, and this run's hits from
    DAMNIT's values in memory.
    """
    import matplotlib.pyplot as plt
    from damnit.context import Skip

    from analysis.diagnostics.curves import (
        band_mean,
        chi2_vs_median,
        flux_from,
        svd,
        xgm_scatter,
    )
    from analysis.diagnostics.frames import per_train
    from analysis.saxs.damnit import config_for
    from analysis.triage.classify import MID_BAND_NM, hits, signals_from

    path = config_for(proposal, run_no).output_file
    if not path.exists():
        raise Skip(f"no frame table at {path}: see agipd_preprocessing")
    h = hits(run_no, proposal=proposal, signals=signals_from(saxs, xgm, transmission))
    trains = h["train_ids"]
    if trains.size < 3:
        raise Skip(f"r{run_no}: {trains.size} droplet hits, too few to diagnose")
    curves = per_train(path).sel(trainId=trains)
    flux = flux_from(xgm, transmission).flux.reindex(trainId=trains)
    decomposition = svd(curves, k=5, flux=flux)
    chi2 = chi2_vs_median(curves, flux=flux)
    signal = band_mean(curves, MID_BAND_NM)
    scatter = xgm_scatter(signal, flux)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    s = decomposition.singular_value.values
    ax.semilogy(np.arange(s.size), s, "o-")
    floor = decomposition.attrs.get("noise_floor")
    if floor is not None and np.isfinite(floor):
        ax.axhline(floor, color="r", lw=0.8, label="noise floor")
        ax.legend(fontsize=8)
    ax.set_xlabel("component")
    ax.set_ylabel("singular value")
    ax.set_title("SVD of curves / flux / median", fontsize=9)

    ax = axes[1]
    t = (chi2.trainId.values - chi2.trainId.values[0]) / 10.0
    ax.plot(t, chi2.chi2.values, ".", ms=2)
    ax.set_yscale("log")
    ax.set_xlabel("time since first hit [s]")
    ax.set_ylabel("reduced χ² against the median")
    ax.set_title("each train against the run median", fontsize=9)

    ax = axes[2]
    ax.plot(flux.values, signal.values, ".", ms=2)
    ax.set_xlabel("flux = XGM × transmission [µJ]")
    ax.set_ylabel(f"I in {MID_BAND_NM[0]}–{MID_BAND_NM[1]} nm$^{{-1}}$")
    ratio = scatter.get("robust_ratio_detrended", scatter.get("robust_ratio"))
    ax.set_title(f"flux division: robust spread ×{float(ratio):.2f}", fontsize=9)
    fig.suptitle(f"r{run_no:04d}, {h['kind']} run, {trains.size} hit trains",
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    return fig
