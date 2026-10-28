"""Make the XPCS figures of every droplet after r240, one folder per droplet.

The droplets are `analysis.droplets.DROPLETS` with ``figures`` set,
and each gets ``figures/droplets/xpcs/<id>_<sample>_r<first>-<last>/``, named as its
SAXS folder. A droplet's usable hit trains (every train of a static run that hit the
droplet, the "on" trains of a mesh run) are split into contiguous t/t* windows
WINDOW_WIDTH wide, t* the SAXS curves' (`analysis.cache.droplet_timeline` of the
droplet's runs, checked against its ``curves.nc``). Per window the trains of all its
runs are pooled, g2 by `analysis.cache.pool_g2` and the two-time map by summing each
run's `XpcsHits.pooled_sums` before dividing (photon-weighted, as `XpcsHits.ttcf`).
The pooled windows are saved in the droplet's folder (``xpcs_windows.nc``, with what
they were made from), so a refit or a new figure need not pool again. A file made
from other inputs is refused, and ``--refresh`` pools again. ``run.json`` holds the
droplet's status, and ``index.json`` beside the folders gathers every ``run.json``.

Droplets can run as parallel jobs (a Slurm array task each), and ``--matched`` and
``--mesh-vs-static`` beside them. A job writes only its droplet's folder, the
windows files under a temporary name renamed into place, and ``index.json``, which
it gathers under a lock.

Each ring's baseline-subtracted c2 is fitted with `analysis.xpcs.kww.fit_rings`: a KWW
``β exp(−2(Γτ)^α) + b`` with β fixed at the ring's reference contrast
(`analysis.triage.xpcs.REFERENCE_BETA`) and Γ, α, b free. Every fitted ring is drawn
as (c2 − b)/β_ref; a ring is "resolved" when no parameter sits at a bound, 1/Γ lies
inside the lags and σ_Γ < Γ, and only resolved rings enter n, D and the mean α.

Per droplet:

``g2_norm_by_tstar.png``
    a panel per window, every ring as (c2 − b)/β_ref with its fit, coloured by q
``ttc_t<lo>-<hi>.png``
    a window's two-time maps c2 = raw − off, a panel per ring
``fit_params_by_tstar.png``
    a row per window: Γ(q) with the q² and q slopes, c2(τ₀)/β_ref (the contrast still
    visible at the first lag) and α against q
``evolution_g2.png``
    a panel per ring, the normalised g2 of every window coloured by t/t*
``evolution_params.png``
    against t/t*: Γ, c2(τ₀)/β_ref and α per ring, n of Γ ∝ qⁿ, D = Γ/q², trains
``xpcs_fits.json``
    every window: its trains, runs, t/t*, ferritin concentration and fit
``xpcs_windows.nc``
    the pooled g2 (c2, sigma, g2, baseline) and two-time c2 of every window

With ``--mesh-vs-static``, each condition's mesh and static representatives
(`analysis.droplets.REPRESENTATIVE`) on the same axes from the same matched windows,
into ``figures/overview/xpcs_mesh_vs_static/``: ``xpcs_mesh_vs_static_*.png`` and
``xpcs_mesh_vs_static_fits.json``
(``--beta bounded``: β free in [0, β_ref's p84], names ending in ``_beta_bounded``).

With ``--matched``, each condition's droplets side by side at matched t/t* instead
(MATCH_T ± MATCH_TOLERANCE, mesh and static apart, saved per droplet as
``xpcs_windows_matched.nc``), into ``figures/overview/xpcs_matched_tstar/``:
``xpcs_matched_tstar_<sample>[_mesh]_{g2,gamma,params}.png`` and
``xpcs_matched_tstar_fits.json``.

Run with the shared environment and a complete TeX (the "science" style sets text in
LaTeX):

    module load texlive/2022
    PY=/gpfs/exfel/exp/MID/202601/p010400/usr/Software/MID-10400/.venv/bin/python
    $PY scripts/damnit_xpcs_figures.py              # every droplet
    $PY scripts/damnit_xpcs_figures.py s443 s394    # some droplets
    $PY scripts/damnit_xpcs_figures.py --refresh    # pool the windows again
    $PY scripts/damnit_xpcs_figures.py --matched    # droplets at matched t/t*
    $PY scripts/damnit_xpcs_figures.py --mesh-vs-static --beta bounded
"""

import argparse
import datetime
import json
import math
import subprocess
import sys
import time
import traceback

from _common import REPO, droplets_dir, overview_dir, write_index

from analysis import plotting
from analysis.droplets import (
    CONDITIONS,
    DROPLETS,
    MATCH_T,
    MATCH_TOLERANCE,
    PROPOSAL,
    REPRESENTATIVE,
    by_id,
    condition_title,
    groups,
)
from analysis.plotting import DROPLET_COLOURS, KIND_COLOURS
from analysis.xpcs.kww import BASE_BOUNDS, GAMMA_BOUNDS, STRETCH_BOUNDS, fit_rings

# t/t* bin width. The SAXS blocks are 0.13-0.18 apart in t/t*, so a window pools one
# or two of them.
WINDOW_WIDTH = 0.25
# what xpcs_windows.nc holds; a change to the pooling or the file's layout bumps it
WINDOWS_SCHEMA = 1
# the rings of the matched comparison's g2 figure, as the mesh-vs-static one's
MATCHED_RINGS = (0, 3, 7, 11)
# q colours: plasma without its yellow end, too faint on white
Q_CMAP_RANGE = (0.0, 0.85)


def outdir(droplet):
    return droplets_dir("xpcs") / droplet.name


def load_droplet(droplet, db):
    """Every usable hit train of `droplet`'s runs with its g2 lag terms, t/t* and
    ferritin concentration, plus each run's results file and the timeline attributes.
    Runs may differ in pulses per train, so the lag terms stay per run. The pulse
    period and the rings must agree.
    Returns (parts by run, t_over_tstar, ferritin, run_of_train, paths, info)."""
    import numpy as np
    import xarray as xr

    from analysis import cache
    from analysis.common.status import FrameStatus
    from analysis.xpcs.damnit import result_file

    droplet_id = droplet.id
    runs = droplet.kept_runs
    tl = cache.droplet_timeline(runs, db)
    t_star = float(tl.attrs["t_star_s"])
    tau = tl.elapsed_s / t_star
    conc = droplet.ferritin_mg_per_ml(tl)
    terms = (
        "status",
        "train_g2_num",
        "train_g2_den",
        "train_baseline_num",
        "train_baseline_den",
    )
    parts, paths, sources, no_hits = {}, {}, {}, []
    for run_nr in runs:
        ds = cache.xpcs_run(run_nr, db, v0_run=runs[0])
        sources[run_nr] = ds.attrs.get("hits_source")
        ok = ds.trainId[ds.status == int(FrameStatus.OK)].values
        # a train needs a droplet fit to have a t/t*
        ok = ok[np.isfinite(tau.reindex(trainId=ok).values)]
        if ok.size == 0:
            no_hits.append(run_nr)
            ds.close()
            continue
        part = ds[list(terms)].sel(trainId=ok).load()
        part.attrs = {"period_s": ds.attrs.get("period_s")}
        part.coords["run"] = ("trainId", np.full(ok.size, run_nr))
        parts[run_nr] = part
        paths[run_nr] = result_file(PROPOSAL, run_nr)
        ds.close()
    shapes = {(p.attrs["period_s"], tuple(p.roi_q.values)) for p in parts.values()}
    if len(shapes) != 1:
        raise ValueError(f"{droplet_id}: runs differ in pulse period or rings")
    frames = {r: p.sizes["lag_frame"] + 1 for r, p in parts.items()}
    info = {
        "t_star_s": t_star,
        "t_star_source": tl.attrs.get("t_star_source"),
        "hit_sources": {str(r): s for r, s in sources.items()},
        "runs_without_usable_hits": no_hits,
        "frames_per_train": {str(r): n for r, n in frames.items()},
        "period_s": float(next(iter(parts.values())).attrs["period_s"]),
    }
    saxs = droplets_dir("saxs") / droplet.name / "curves.nc"
    if saxs.exists():
        saxs_t_star = float(xr.load_dataset(saxs).attrs["t_star_s"])
        info["saxs_t_star_s"] = saxs_t_star
        if abs(saxs_t_star - t_star) > 1e-6 * t_star:
            print(
                f"WARNING {droplet_id}: t* {t_star:.1f} s here, "
                f"{saxs_t_star:.1f} s in the SAXS curves.nc",
                flush=True,
            )
    run_of_train = xr.concat([p.run for p in parts.values()], "trainId")
    t = tau.reindex(trainId=run_of_train.trainId)
    c = conc.reindex(trainId=run_of_train.trainId)
    return parts, t, c, run_of_train, paths, info


def pooled_ttc(paths, run_of_train, chosen, n_frames):
    """The two-time maps of the trains `chosen`, pooled over their runs: each run's
    `XpcsHits.pooled_sums` summed before dividing, as `XpcsHits.ttcf` does within a
    run, over the first `n_frames` frames (the same pulses in every run, whatever
    its pattern's length). Returns c2 = raw_paired − off, (rings, n_frames,
    n_frames), NaN where nothing is behind it."""
    import numpy as np

    from analysis.xpcs.reader import XpcsHits

    names = ("pair_num", "pair_den", "off_num", "off_den")
    total = None
    runs = run_of_train.sel(trainId=chosen).values
    for run_nr in np.unique(runs):
        sums = XpcsHits(paths[int(run_nr)]).pooled_sums(chosen[runs == run_nr])
        maps = {n: getattr(sums, n)[:, :n_frames, :n_frames] for n in names}
        if total is None:
            total = {n: m.copy() for n, m in maps.items()}
        else:
            for n in names:
                total[n] += maps[n]

    def divide(num, den):
        out = np.full_like(num, np.nan)
        np.divide(num, den, out=out, where=den > 0)
        return out

    paired = divide(total["pair_num"], total["pair_den"])
    off = divide(
        total["off_num"] + total["off_num"].swapaxes(-1, -2),
        total["off_den"] + total["off_den"].swapaxes(-1, -2),
    )
    return paired - off


def pool_windows(droplet, db, matched=False):
    """`droplet`'s windows, pooled: per WINDOW_WIDTH bin of t/t* with a usable hit
    train, (record, pooled g2, two-time c2). With `matched`, the windows are instead
    MATCH_T ± MATCH_TOLERANCE, the SAXS matched comparison's, without the two-time
    maps (None). A window whose runs differ in pulses per train is pooled over the
    frames they share: the lag terms and the two-time sums of those frames are the
    same pulses in every run, so truncating them is exact. Returns the windows and
    the droplet's info."""
    import numpy as np
    import xarray as xr

    from analysis.cache import pool_g2

    parts, t, conc, run_of_train, paths, info = load_droplet(droplet, db)
    t_values = t.values
    all_trains = run_of_train.trainId.values
    if matched:
        bounds = [
            (t - MATCH_TOLERANCE, t + MATCH_TOLERANCE, {"t_target": t}) for t in MATCH_T
        ]
    else:
        lows = np.arange(math.ceil(t_values.max() / WINDOW_WIDTH)) * WINDOW_WIDTH
        bounds = [(lo, lo + WINDOW_WIDTH, {}) for lo in lows]
    out = []
    for lo, hi, label in bounds:
        chosen = all_trains[(t_values >= lo) & (t_values < hi)]
        if chosen.size == 0:
            continue
        runs = sorted({int(r) for r in run_of_train.sel(trainId=chosen).values})
        n_lags = min(parts[r].sizes["lag_frame"] for r in runs)
        pool = xr.concat(
            [
                parts[r]
                .isel(lag_frame=slice(0, n_lags))
                .sel(trainId=np.intersect1d(chosen, parts[r].trainId.values))
                for r in runs
            ],
            "trainId",
            combine_attrs="override",
        )
        record = label | {
            "t_range": [round(float(lo), 3), round(float(hi), 3)],
            "n_trains": int(chosen.size),
            "t_over_tstar_mean": round(float(t.sel(trainId=chosen).mean()), 3),
            "ferritin_mg_per_ml": round(float(conc.sel(trainId=chosen).mean()), 1),
            "runs": runs,
            "frames_per_train": n_lags + 1,
        }
        g2 = pool_g2(pool, chosen)[["c2", "sigma", "g2", "baseline"]]
        ttc = None if matched else pooled_ttc(paths, run_of_train, chosen, n_lags + 1)
        out.append((record, g2, ttc))
        print(f"  t/t* {lo:.2f}-{hi:.2f}: {chosen.size} trains pooled", flush=True)
    return out, info


def saved_windows(droplet, db, refresh=False, matched=False):
    """`pool_windows` of `droplet`, from ``xpcs_windows.nc`` in its folder
    (``xpcs_windows_matched.nc`` for the `matched` windows) when that exists, else
    pooled and saved there (``refresh`` pools again). Windows differ in
    lags and frames where their runs' pulse patterns differ, so the file pads them
    with NaN: ``c2``, ``sigma``, ``g2``, ``baseline`` over (window, roi_q, lag) with
    their ``tau`` over (window, lag), ``ttc`` (float32) over (window, roi_q, frame_1,
    frame_2). It records what it was made from: WINDOWS_SCHEMA, WINDOW_WIDTH, the
    runs, the PEG %, `analysis.cache.CACHE_VERSION`, and the size and mtime of every
    run's XPCS results file and DAMNIT file (the sources of `cache.xpcs_run` and
    `cache.droplet_timeline`). A file made from anything else raises."""
    import numpy as np
    import xarray as xr

    from analysis import cache
    from analysis.xpcs.damnit import result_file

    droplet_id = droplet.id
    runs = droplet.kept_runs
    inputs = {
        "schema": WINDOWS_SCHEMA,
        "window_width": WINDOW_WIDTH,
        "runs": runs,
        "peg_percent_wv": droplet.peg_percent_wv,
        "cache_version": cache.CACHE_VERSION,
    }
    if matched:
        inputs |= {"match_t": list(MATCH_T), "match_tolerance": MATCH_TOLERANCE}
    sources = [cache.file_source(result_file(PROPOSAL, r)) for r in runs]
    sources += [cache.file_source(db[r].file) for r in runs]
    path = outdir(droplet) / f"xpcs_windows{'_matched' if matched else ''}.nc"
    if path.exists() and not refresh:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            # read every attribute inside the block, while the file is still open
            stale = cache.stale(ds.attrs, inputs=inputs, sources=sources)
            if stale:
                raise ValueError(
                    f"{path} was made from other {' and '.join(stale)} than "
                    f"{droplet_id}'s now; remake it with --refresh"
                )
            records = json.loads(ds.attrs["windows"])
            info = json.loads(ds.attrs["info"])
            ds = ds.load()
        windows = []
        for k, record in enumerate(records):
            w = ds.isel(window=k)
            lags = np.isfinite(w.tau.values)
            frames = record["frames_per_train"]
            g2 = xr.Dataset(
                {
                    name: (("roi_q", "tau"), w[name].values[:, lags])
                    for name in ("c2", "sigma", "g2", "baseline")
                },
                coords={"roi_q": ds.roi_q.values, "tau": w.tau.values[lags]},
            )
            ttc = (
                w.ttc.values[:, :frames, :frames].astype(np.float64)
                if "ttc" in w
                else None
            )
            windows.append((record, g2, ttc))
        return windows, info

    windows, info = pool_windows(droplet, db, matched)
    if not windows:  # no hit train near any target: nothing to save
        return windows, info
    outdir(droplet).mkdir(parents=True, exist_ok=True)
    n_lag = max(g2.sizes["tau"] for _, g2, _ in windows)
    n_rings = windows[0][1].sizes["roi_q"]

    def padded(values, shape):
        full = np.full(shape, np.nan, dtype=values.dtype)
        full[tuple(slice(0, n) for n in values.shape)] = values
        return full

    ds = xr.Dataset(
        {
            **{
                name: (
                    ("window", "roi_q", "lag"),
                    np.stack(
                        [
                            padded(g2[name].values, (n_rings, n_lag))
                            for _, g2, _ in windows
                        ]
                    ),
                )
                for name in ("c2", "sigma", "g2", "baseline")
            },
            "tau": (
                ("window", "lag"),
                np.stack([padded(g2.tau.values, (n_lag,)) for _, g2, _ in windows]),
            ),
        },
        coords={
            "roi_q": windows[0][1].roi_q.values,
            "t_lo": ("window", [r["t_range"][0] for r, _, _ in windows]),
        },
    )
    if not matched:
        n_frame = max(ttc.shape[-1] for _, _, ttc in windows)
        ds["ttc"] = (
            ("window", "roi_q", "frame_1", "frame_2"),
            np.stack(
                [
                    padded(ttc.astype(np.float32), (n_rings, n_frame, n_frame))
                    for _, _, ttc in windows
                ]
            ),
        )
    ds.attrs = {
        "inputs": json.dumps(inputs, sort_keys=True),
        "sources": json.dumps(sources, sort_keys=True),
        "windows": json.dumps([r for r, _, _ in windows]),
        "info": json.dumps(info),
        "description": "pooled g2 and two-time c2 of the hit trains of each t/t* "
        "window; see scripts/damnit_xpcs_figures.py saved_windows",
    }
    cache.write_dataset(ds, path)
    return windows, info


def fit_windows(windows, beta_mode="fixed"):
    """`fit_rings` on every window. Returns (record, g2, fit, ttc) per window, the
    record extended by the fit's summary."""
    out = []
    for record, g2, ttc in windows:
        fit = fit_rings(g2, beta_mode)
        out.append((record | fit.summary(), g2, fit, ttc))
    return out


def matched_windows(droplet, db, refresh=False, beta_mode="fixed"):
    """`droplet`'s MATCH_T windows (`saved_windows(matched=True)`) fitted, by target:
    {t_target: (record, g2, fit, None)} for the targets with a hit train, and the
    droplet's info."""
    pooled, info = saved_windows(droplet, db, refresh, matched=True)
    return {w[0]["t_target"]: w for w in fit_windows(pooled, beta_mode)}, info


def window_label(record):
    lo, hi = record["t_range"]
    return f"{lo:.2f}-{hi:.2f}"


def window_title(record):
    lo, hi = record["t_range"]
    return (
        f"$t/t^*$ {lo:.2f}--{hi:.2f}: {record['n_trains']} trains, "
        f"{record['ferritin_mg_per_ml']:.0f} mg/ml"
    )


def make_figures(droplet, windows, info):
    """Every figure of one droplet into its folder. Returns the files written."""
    import matplotlib
    import numpy as np
    from matplotlib.colors import Normalize

    plt = plotting.pyplot()
    out = outdir(droplet)
    runs = droplet.kept_runs
    head = (
        f"{droplet.id}: {CONDITIONS[droplet.sample]}, r{runs[0]}--{runs[-1]}"
        + (" (mesh scans, hit trains)" if droplet.mesh else "")
        + f", $t^*$ = {info['t_star_s']:.0f} s"
    )
    q_nm = windows[0][2].q_nm
    fitted = [np.isfinite(w[2].gamma) for w in windows]
    n_rings = q_nm.size
    q_norm = Normalize(q_nm.min(), q_nm.max())
    lo_c, hi_c = Q_CMAP_RANGE

    def q_colour(r):
        return plt.get_cmap("plasma")(lo_c + (hi_c - lo_c) * q_norm(q_nm[r]))

    q_mappable = matplotlib.cm.ScalarMappable(
        norm=q_norm,
        cmap=matplotlib.colors.ListedColormap(
            plt.get_cmap("plasma")(np.linspace(lo_c, hi_c, 256))
        ),
    )
    t_norm = Normalize(0, max(w[0]["t_range"][1] for w in windows))
    t_cmap = plt.get_cmap("viridis")
    t_mappable = matplotlib.cm.ScalarMappable(norm=t_norm, cmap=t_cmap)
    kww_text = (
        r"$(c_2 - b)/\beta_\mathrm{ref}$, "
        r"$c_2 = \beta_\mathrm{ref}\, e^{-2(\Gamma\tau)^\alpha} + b$, "
        r"$\beta_\mathrm{ref}$ fixed per ring"
    )
    unresolved_text = (
        r"faint points, dashed fit: not resolved (a parameter at a bound, "
        r"$1/\Gamma$ outside the lags or $\sigma_\Gamma \geq \Gamma$)"
    )
    names = []

    def save(fig, name):
        names.append(plotting.save(fig, out / name))

    def normalised(g2, fit, r):
        y = (g2.c2.values[r] - fit.base[r]) / fit.beta[r]
        e = g2.sigma.values[r] / fit.beta[r]
        return y, e

    def model(fit, r):
        return np.exp(-2 * (fit.gamma[r] * tau_fine) ** fit.stretch[r])

    def look(fit, r):
        "points' alpha and the fit's line style: faint and dashed when unresolved"
        return (1.0, "-") if fit.resolved[r] else (0.3, "--")

    def tau_axis(ax):
        ax.set_xscale("log")
        ax.set_ylim(-0.3, 1.3)
        ax.axhline(0, color="0.6", lw=0.5)

    tau_fine = np.geomspace(
        min(float(w[1].tau.min()) for w in windows),
        max(float(w[1].tau.max()) for w in windows),
        200,
    )

    # normalised g2, a panel per window, a line per ring
    ncols = min(4, len(windows))
    nrows = math.ceil(len(windows) / ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols + 1.0, 2.8 * nrows + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, (record, g2, fit, _), ok in zip(axs.flat, windows, fitted, strict=False):
        for r in np.flatnonzero(ok):
            y, e = normalised(g2, fit, r)
            alpha, ls = look(fit, r)
            ax.errorbar(
                g2.tau,
                y,
                e,
                color=q_colour(r),
                marker="o",
                ms=1.5,
                lw=0,
                elinewidth=0.4,
                alpha=alpha,
            )
            ax.plot(tau_fine, model(fit, r), color=q_colour(r), lw=0.8, ls=ls)
        tau_axis(ax)
        ax.set_title(
            window_title(record) + f"\n{record['n_resolved']}/{n_rings} rings resolved",
            fontsize=7,
        )
    for ax in axs.flat[len(windows) :]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel(r"$\tau$ (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$(c_2 - b)/\beta_\mathrm{ref}$")
    fig.colorbar(q_mappable, ax=axs, label=r"$q$ (nm$^{-1}$)", shrink=0.9)
    fig.suptitle(f"{head}\n{kww_text}; {unresolved_text}", fontsize=9)
    save(fig, "g2_norm_by_tstar.png")

    # two-time maps, a file per window, a panel per ring
    period_us = info["period_s"] * 1e6
    ttc_cols = 4
    ttc_rows = math.ceil(n_rings / ttc_cols)
    for record, _, _, c2 in windows:
        fig, axs = plt.subplots(
            ttc_rows,
            ttc_cols,
            figsize=(3.2 * ttc_cols, 3.0 * ttc_rows + 0.6),
            squeeze=False,
        )
        extent = [0, c2.shape[-1] * period_us, 0, c2.shape[-1] * period_us]
        for r, ax in enumerate(axs.flat[:n_rings]):
            upper = c2[r][np.triu_indices_from(c2[r], k=1)]
            vmax = float(np.nanquantile(upper, 0.90)) if np.isfinite(upper).any() else 1
            image = ax.imshow(
                c2[r],
                vmin=0,
                vmax=vmax,
                extent=extent,
                origin="lower",
                interpolation="none",
                cmap="jet",
            )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"$q$ = {q_nm[r]:.3f} nm$^{{-1}}$", fontsize=8)
            ax.set_xlabel(r"$t_1$ ($\mu$s)", fontsize=7)
            ax.set_ylabel(r"$t_2$ ($\mu$s)", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axs.flat[n_rings:]:
            ax.set_visible(False)
        fig.suptitle(
            f"{head}\n{window_title(record)}; two-time $c_2$ = raw $-$ off, "
            r"pooled over the trains; colour 0 to the 0.90 quantile above the diagonal",
            fontsize=9,
        )
        fig.tight_layout()
        save(fig, f"ttc_t{window_label(record)}.png")

    visible_label = r"$c_2(\tau_0)/\beta_\mathrm{ref}$"
    # fit parameters against q, a row per window
    fig, axs = plt.subplots(
        len(windows),
        3,
        figsize=(10.5, 2.3 * len(windows) + 0.8),
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    for row, (record, _, fit, _) in zip(axs, windows, strict=True):
        ok = fit.resolved
        for ax, value, err in (
            (row[0], fit.gamma, fit.err_gamma),
            (row[1], fit.visible, fit.err_visible),
            (row[2], fit.stretch, fit.err_stretch),
        ):
            for mask, face in ((ok, "C0"), (~ok, "none")):
                ax.errorbar(
                    q_nm[mask],
                    value[mask],
                    err[mask],
                    color="C0",
                    mfc=face,
                    marker="o",
                    ms=3,
                    lw=0,
                    elinewidth=0.6,
                )
        if ok.any():
            k = np.flatnonzero(ok)[ok.sum() // 2]
            qq = np.geomspace(q_nm.min(), q_nm.max(), 20)
            for power, style in ((2, "--"), (1, ":")):
                row[0].plot(
                    qq,
                    fit.gamma[k] * (qq / q_nm[k]) ** power,
                    color="0.5",
                    ls=style,
                    lw=0.8,
                )
        if "exponent_n" in record:
            row[0].text(
                0.04,
                0.94,
                f"$n$ = {record['exponent_n']:.2f} $\\pm$ {record['exponent_err']:.2f}",
                transform=row[0].transAxes,
                fontsize=7,
                va="top",
            )
        row[0].set_yscale("log")
        row[0].set_ylim(1e3, 1e7)
        row[1].set_ylim(-0.1, 1.3)
        row[1].axhline(1, color="0.5", lw=0.6)
        row[2].set_ylim(*STRETCH_BOUNDS)
        row[2].axhline(1, color="0.5", lw=0.6)
        row[0].set_ylabel(r"$\Gamma$ (s$^{-1}$)")
        row[1].set_ylabel(visible_label)
        row[2].set_ylabel(r"KWW $\alpha$")
        row[1].set_title(window_title(record), fontsize=7)
    for ax in axs[-1]:
        plotting.log_q_axis(ax, lim=None, ticks=(0.2, 0.3, 0.5, 0.8))
        ax.set_xlabel(r"$q$ (nm$^{-1}$)")
    fig.suptitle(
        f"{head}\nKWW fit per ring, $\\beta$ fixed at $\\beta_\\mathrm{{ref}}$; "
        "filled: resolved, open: not; "
        r"dashed $\propto q^2$ (diffusion), dotted $\propto q$ (shear)",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "fit_params_by_tstar.png")

    # normalised g2 against τ, a panel per ring, a line per window
    fig, axs = plt.subplots(
        ttc_rows,
        ttc_cols,
        figsize=(3.2 * ttc_cols + 1.0, 2.6 * ttc_rows + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for r, ax in enumerate(axs.flat[:n_rings]):
        for (record, g2, fit, _), ok in zip(windows, fitted, strict=True):
            if not ok[r]:
                continue
            colour = t_cmap(t_norm(record["t_over_tstar_mean"]))
            y, _ = normalised(g2, fit, r)
            alpha, ls = look(fit, r)
            ax.plot(g2.tau, y, color=colour, marker="o", ms=1.5, lw=0, alpha=alpha)
            ax.plot(tau_fine, model(fit, r), color=colour, lw=0.8, ls=ls)
        tau_axis(ax)
        ax.set_title(f"$q$ = {q_nm[r]:.3f} nm$^{{-1}}$", fontsize=8)
    for ax in axs.flat[n_rings:]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel(r"$\tau$ (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$(c_2 - b)/\beta_\mathrm{ref}$")
    fig.colorbar(t_mappable, ax=axs, label="$t/t^*$", shrink=0.9)
    fig.suptitle(
        f"{head}\n{kww_text}, a line per {WINDOW_WIDTH:g}-wide $t/t^*$ window; "
        f"{unresolved_text}",
        fontsize=9,
    )
    save(fig, "evolution_g2.png")

    # fit parameters against t/t*; a window without trains or an unresolved ring is a
    # NaN, so the lines break there instead of bridging it
    slots = [round(w[0]["t_range"][0] / WINDOW_WIDTH) for w in windows]

    def by_slot(values):
        full = np.full(max(slots) + 1, np.nan)
        full[slots] = values
        return full

    records = [w[0] for w in windows]
    t = by_slot([r["t_over_tstar_mean"] for r in records])
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key, label, only_resolved in (
        (axs[0, 0], "gamma", r"$\Gamma$ (s$^{-1}$)", True),
        (axs[0, 1], "visible", visible_label, False),
        (axs[0, 2], "stretch", r"KWW $\alpha$", True),
    ):
        for r in range(n_rings):
            # the visible contrast is measured, so it is drawn for every fitted ring
            ok = np.array(
                [
                    (w[2].resolved if only_resolved else ok)[r]
                    for w, ok in zip(windows, fitted, strict=True)
                ]
            )
            if not ok.any():
                continue
            value = np.array([getattr(w[2], key)[r] for w in windows])
            err = np.array([getattr(w[2], f"err_{key}")[r] for w in windows])
            ax.errorbar(
                t,
                by_slot(np.where(ok, value, np.nan)),
                by_slot(np.where(ok, err, np.nan)),
                color=q_colour(r),
                marker="o",
                ms=3,
                lw=0.6,
                elinewidth=0.5,
                capsize=1.5,
            )
        ax.set_ylabel(label)
    axs[0, 0].set_yscale("log")
    axs[0, 1].axhline(1, color="0.5", lw=0.6)
    axs[0, 2].axhline(1, color="0.5", lw=0.6)
    axs[0, 2].set_ylim(0, STRETCH_BOUNDS[1] * 1.1)
    style = {"color": "k", "marker": "o", "ms": 3, "lw": 0.6, "capsize": 2}
    axs[1, 0].errorbar(
        t,
        by_slot([r.get("exponent_n", np.nan) for r in records]),
        by_slot([r.get("exponent_err", np.nan) for r in records]),
        **style,
    )
    axs[1, 0].axhline(2, color="0.5", ls="--", lw=0.8)
    axs[1, 0].axhline(1, color="0.5", ls=":", lw=0.8)
    axs[1, 0].set_ylabel(r"$n$ in $\Gamma \propto q^n$")
    axs[1, 1].errorbar(
        t,
        by_slot([r.get("D_um2_per_s", np.nan) for r in records]),
        by_slot([r.get("D_err_um2_per_s", np.nan) for r in records]),
        **style,
    )
    axs[1, 1].set_ylabel(r"$D = \Gamma/q^2$ ($\mu$m$^2$/s)")
    axs[1, 2].plot(t, by_slot([r["n_trains"] for r in records]), "ko-", ms=3, lw=0.6)
    axs[1, 2].set_ylabel("hit trains per window")
    concentration = axs[1, 2].twinx()
    concentration.plot(
        t,
        by_slot([r["ferritin_mg_per_ml"] for r in records]),
        "s--",
        color="C3",
        ms=3,
        lw=0.6,
    )
    concentration.set_ylabel("ferritin (mg/ml)", color="C3")
    for ax in axs.flat:
        ax.set_xlabel("$t/t^*$")
        ax.set_xlim(0, t_norm.vmax)
    fig.colorbar(q_mappable, ax=axs[0], label=r"$q$ (nm$^{-1}$)", shrink=0.9)
    fig.suptitle(
        f"{head}\nKWW fit per ring and {WINDOW_WIDTH:g}-wide $t/t^*$ window, "
        r"$\beta$ fixed at $\beta_\mathrm{ref}$; $\Gamma$ and $\alpha$ of resolved "
        r"rings only, $c_2(\tau_0)/\beta_\mathrm{ref}$ of every fitted ring; "
        "$n$ and $D$ over the resolved rings",
        fontsize=9,
    )
    save(fig, "evolution_params.png")
    return names


def run_one(droplet, db, refresh=False):
    droplet_id = droplet.id
    out = outdir(droplet)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"{droplet_id}: reading", flush=True)
    pooled, info = saved_windows(droplet, db, refresh)
    windows = fit_windows(pooled)
    names = make_figures(droplet, windows, info)
    # a previous run's TTC files of windows that no longer exist
    for stale in out.glob("ttc_t*.png"):
        if stale.name not in names:
            stale.unlink()
    record = {
        "id": droplet_id,
        "sample": droplet.sample,
        "runs": droplet.kept_runs,
        "mesh_scan": droplet.mesh,
        "made": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "window_width": WINDOW_WIDTH,
        "pooling": "every usable hit train in the window: g2 by "
        "analysis.cache.pool_g2, the two-time maps by summing each run's "
        "XpcsHits.pooled_sums",
        "model": "analysis.xpcs.kww.fit_rings, beta fixed: c2 = beta_ref "
        "exp(-2 (Gamma tau)^alpha) + b per ring, weighted by sigma, bounded; "
        "resolved = no parameter within AT_BOUND of a bound, 1/Gamma inside the "
        "lags, sigma_Gamma < Gamma",
        "bounds": {
            "gamma_per_s": list(GAMMA_BOUNDS),
            "alpha": list(STRETCH_BOUNDS),
            "b": list(BASE_BOUNDS),
        },
        **info,
        "windows": [w[0] for w in windows],
        "figures": names,
    }
    (out / "xpcs_fits.json").write_text(json.dumps(record, indent=1))
    seconds = time.perf_counter() - started
    print(f"{droplet_id}: {len(names)} figures in {seconds:.0f} s -> {out}", flush=True)
    write_run(
        droplet,
        {"status": "ok", "n_windows": len(windows), "seconds": round(seconds, 1)},
    )


def write_run(droplet, entry):
    "The droplet's ``run.json``, its line of ``index.json``."
    out = outdir(droplet)
    out.mkdir(parents=True, exist_ok=True)
    record = {"id": droplet.id, "folder": str(out.relative_to(REPO))} | entry
    (out / "run.json").write_text(json.dumps(record, indent=1))


def matched(sample, mesh, db, refresh=False):
    """`sample`'s droplets (mesh or static) side by side at matched t/t*, as the SAXS
    script's ``saxs_matched_tstar_*``: per MATCH_T every usable hit train within
    MATCH_TOLERANCE pooled (`saved_windows(matched=True)`), each ring fitted by
    `fit_rings` (β fixed), a line per droplet in its DROPLET_COLOURS colour. Three
    figures into ``figures/overview/xpcs_matched_tstar/``: (c2 − b)/β_ref against τ at
    MATCHED_RINGS, a panel per (ring, t/t*); Γ(q), a panel per t/t*; and against t/t*
    the slope n of Γ ∝ qⁿ, D = Γ/q², the mean α and c2(τ₀)/β_ref at the lowest ring.
    Returns the files written and the windows' records."""
    import numpy as np

    plt = plotting.pyplot()
    droplets = [
        d for d in DROPLETS if d.figures and (d.sample, d.mesh) == (sample, mesh)
    ]
    if not droplets:
        return [], []
    if len(droplets) > len(DROPLET_COLOURS):
        raise ValueError(
            f"{len(droplets)} droplets in {condition_title(sample, mesh)!r}, "
            f"only {len(DROPLET_COLOURS)} colours"
        )
    data = []
    record = []
    for droplet in droplets:
        windows, _ = matched_windows(droplet, db, refresh)
        data.append((droplet, windows))
        for target in MATCH_T:
            entry = {"sample": sample, "mesh_scan": mesh, "id": droplet.id}
            if target in windows:
                entry |= windows[target][0]
            else:
                entry |= {
                    "t_target": target,
                    "skipped": f"no hit train within {MATCH_TOLERANCE}",
                }
            record.append(entry)
        print(f"matched {sample}: {droplet.id} fitted", flush=True)
    colours = {d.id: c for (d, _), c in zip(data, DROPLET_COLOURS, strict=False)}
    labels = {d.id: f"{d.id}: {d.run_label}" for d, _ in data}
    stem = f"xpcs_matched_tstar_{sample}{'_mesh' if mesh else ''}"
    out = overview_dir("xpcs_matched_tstar")
    names = []
    suptitle = (
        f"{condition_title(sample, mesh)}: XPCS at matched $t/t^*$ (hit trains within "
        f"$\\pm${MATCH_TOLERANCE:g} pooled; ferritin range per panel)"
    )
    unresolved = (
        r"KWW fit, $\beta$ fixed at $\beta_\mathrm{ref}$; faint points, dashed fit, "
        "open markers: not resolved"
    )

    def title(target):
        conc = [w[target][0]["ferritin_mg_per_ml"] for _, w in data if target in w]
        text = f"$t/t^* = {target:g}$"
        if conc:
            low, high = (f"{f(conc):.0f}" for f in (min, max))
            text += f", {low} mg/ml" if low == high else f", {low}-{high} mg/ml"
        return text

    def legend(fig, x=0.91):
        fig.legend(
            [plt.Line2D([], [], color=colours[d], marker="o") for d in colours],
            [labels[d] for d in colours],
            loc="center left",
            bbox_to_anchor=(x, 0.5),
            frameon=False,
            fontsize=8,
        )

    def save(fig, name):
        names.append(plotting.save(fig, out / name))

    # (c2 - b)/β_ref at four rings, a panel per (ring, t/t*)
    q_nm = next(iter(next(w for _, w in data if w).values()))[2].q_nm
    fig, axs = plt.subplots(
        len(MATCHED_RINGS),
        len(MATCH_T),
        figsize=(2.6 * len(MATCH_T), 2.2 * len(MATCHED_RINGS) + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for j, target in enumerate(MATCH_T):
        for droplet, windows in data:
            if target not in windows:
                continue
            _, g2, fit, _ = windows[target]
            tau_fine = np.geomspace(g2.tau.values.min(), g2.tau.values.max(), 100)
            for i, r in enumerate(MATCHED_RINGS):
                if not np.isfinite(fit.gamma[r]):
                    continue
                ok = fit.resolved[r]
                colour = colours[droplet.id]
                axs[i, j].errorbar(
                    g2.tau,
                    (g2.c2.values[r] - fit.base[r]) / fit.beta[r],
                    g2.sigma.values[r] / fit.beta[r],
                    color=colour,
                    marker="o",
                    ms=1.5,
                    lw=0,
                    elinewidth=0.4,
                    alpha=1.0 if ok else 0.3,
                )
                axs[i, j].plot(
                    tau_fine,
                    np.exp(-2 * (fit.gamma[r] * tau_fine) ** fit.stretch[r]),
                    color=colour,
                    lw=0.8,
                    ls="-" if ok else "--",
                )
        axs[0, j].set_title(title(target), fontsize=7)
    for i, r in enumerate(MATCHED_RINGS):
        axs[i, 0].set_ylabel(
            rf"$(c_2-b)/\beta_\mathrm{{ref}}$, $q$ = {q_nm[r]:.3f} nm$^{{-1}}$",
            fontsize=7,
        )
        for ax in axs[i]:
            ax.axhline(0, color="0.6", lw=0.5)
            ax.set_ylim(-0.3, 1.3)
    for ax in axs[-1]:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau$ (s)")
    legend(fig)
    fig.suptitle(f"{suptitle}\n{unresolved}", fontsize=9)
    save(fig, f"{stem}_g2.png")

    # Γ(q), a panel per t/t*
    ncols = 3
    nrows = math.ceil(len(MATCH_T) / ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols + 0.4, 3.2 * nrows + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, target in zip(axs.flat, MATCH_T, strict=False):
        anchor = None
        for droplet, windows in data:
            if target not in windows:
                continue
            fit = windows[target][2]
            ok = fit.resolved
            colour = colours[droplet.id]
            for mask, face in ((ok, colour), (~ok, "none")):
                ax.errorbar(
                    fit.q_nm[mask],
                    fit.gamma[mask],
                    fit.err_gamma[mask],
                    color=colour,
                    mfc=face,
                    marker="o",
                    ms=3,
                    lw=0,
                    elinewidth=0.6,
                )
            if ok.any() and anchor is None:
                k = np.flatnonzero(ok)[ok.sum() // 2]
                anchor = (fit.q_nm[k], fit.gamma[k])
        if anchor is not None:
            qq = np.geomspace(0.22, 0.9, 20)
            for power, style in ((2, "--"), (1, ":")):
                ax.plot(
                    qq,
                    anchor[1] * (qq / anchor[0]) ** power,
                    color="0.5",
                    ls=style,
                    lw=0.8,
                )
        plotting.log_q_axis(ax, lim=(0.22, 0.9), ticks=(0.3, 0.5, 0.8))
        ax.set_yscale("log")
        ax.set_ylim(1e3, 3e7)
        ax.set_title(title(target), fontsize=8)
    for ax in axs.flat[len(MATCH_T) :]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel(r"$q$ (nm$^{-1}$)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\Gamma$ (s$^{-1}$)")
    legend(fig)
    fig.suptitle(
        f"{suptitle}\n{unresolved}; dashed $\\propto q^2$ (diffusion), "
        r"dotted $\propto q$ (shear)",
        fontsize=9,
    )
    save(fig, f"{stem}_gamma.png")

    # fit parameters against t/t*
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for droplet, windows in data:
        rows = [windows[t][0] for t in MATCH_T if t in windows]
        if not rows:
            continue
        t = np.array([r["t_over_tstar_mean"] for r in rows])
        style = {
            "color": colours[droplet.id],
            "marker": "o",
            "ms": 3,
            "lw": 0.8,
            "capsize": 2,
        }
        for ax, key, err in (
            (axs[0], "exponent_n", "exponent_err"),
            (axs[1], "D_um2_per_s", "D_err_um2_per_s"),
            (axs[2], "stretch_mean", "stretch_mean_err"),
        ):
            ax.errorbar(
                t,
                [r.get(key, np.nan) for r in rows],
                [r.get(err, np.nan) for r in rows],
                **style,
            )
        axs[3].errorbar(
            t,
            [r["visible"][0] for r in rows],
            [r["err_visible"][0] for r in rows],
            **style,
        )
    axs[0].axhline(2, color="0.5", ls="--", lw=0.8)
    axs[0].axhline(1, color="0.5", ls=":", lw=0.8)
    axs[0].set_ylabel(r"$n$ in $\Gamma \propto q^n$")
    axs[1].set_ylabel(r"$D = \Gamma/q^2$ ($\mu$m$^2$/s)")
    axs[1].set_yscale("log")
    axs[2].axhline(1, color="0.5", lw=0.6)
    axs[2].set_ylabel(r"mean KWW $\alpha$")
    axs[3].axhline(1, color="0.5", lw=0.6)
    axs[3].set_ylabel(
        rf"$c_2(\tau_0)/\beta_\mathrm{{ref}}$, $q$ = {q_nm[0]:.3f} nm$^{{-1}}$"
    )
    for ax in axs:
        ax.set_xlabel("$t/t^*$")
    legend(fig, x=0.99)
    fig.suptitle(
        f"{suptitle}\n$n$, $D$ and mean $\\alpha$ over the resolved rings", fontsize=9
    )
    fig.tight_layout()
    save(fig, f"{stem}_params.png")
    return names, record


def matched_all(db, refresh=False):
    "matched() for every condition, mesh and static apart; the fits in one JSON."
    written = []
    record = []
    for sample, mesh in groups():
        names, entries = matched(sample, mesh, db, refresh)
        written += names
        record += entries
    if written:
        name = "xpcs_matched_tstar_fits.json"
        (overview_dir("xpcs_matched_tstar") / name).write_text(
            json.dumps(
                {
                    "t_targets": list(MATCH_T),
                    "tolerance": MATCH_TOLERANCE,
                    "pooling": "every usable hit train within the tolerance: "
                    "analysis.cache.pool_g2 over the lags the window's runs share",
                    "model": "analysis.xpcs.kww.fit_rings, beta fixed at "
                    "REFERENCE_BETA",
                    "windows": record,
                },
                indent=1,
            )
        )
        written.append(name)
    return written


def mesh_vs_static(sample, db, refresh=False, beta_mode="fixed"):
    """`sample`'s mesh and static representatives (`analysis.droplets.REPRESENTATIVE`)
    on the same axes at matched t/t*, from their `matched_windows` fitted with
    `beta_mode` (fits drawn and Γ, τ filled only where resolved). Five figures into
    ``figures/overview/xpcs_mesh_vs_static/``: c2(τ) with the KWW fit at four rings, a
    panel per (ring, t/t*); Γ, τ = 1/Γ and the KWW exponent α against q, a panel per
    t/t*, Γ and τ with the Γ ∝ q² (diffusion) and Γ ∝ q (shear) slopes; and against
    t/t* the slope n of Γ ∝ qⁿ, D = Γ/q², c2 at the first lag over β at the lowest
    ring, τ at a middle one and the mean α, and with ``beta_mode="bounded"``
    1 − β/β_ref at the lowest ring; those figures' names end in ``_beta_bounded``.
    Returns the files written and the record."""
    import numpy as np

    plt = plotting.pyplot()
    mesh = [d.id for d in DROPLETS if d.figures and d.sample == sample and d.mesh]
    picks = {
        "static": REPRESENTATIVE.get((sample, False)),
        # a condition with one mesh droplet needs no entry
        "mesh": REPRESENTATIVE.get((sample, True), mesh[0] if len(mesh) == 1 else None),
    }
    if not all(picks.values()):
        return [], {}
    colours = KIND_COLOURS
    data = {}
    record = {"sample": sample, "droplets": {}}
    for kind, droplet_id in picks.items():
        windows, info = matched_windows(by_id(droplet_id), db, refresh, beta_mode)
        fits = {}
        for target in MATCH_T:
            if target in windows:
                entry, g2, fit, _ = windows[target]
                fits[target] = (g2, fit, {"id": droplet_id} | entry)
            else:
                entry = {"t_target": target, "id": droplet_id, "n_trains": 0}
                entry["skipped"] = f"no hit train within {MATCH_TOLERANCE}"
                fits[target] = (None, None, entry)
        data[kind] = fits
        record["droplets"][kind] = {
            "id": droplet_id,
            "hit_sources": info["hit_sources"],
            "windows": [e for _, _, e in fits.values()],
        }

    out = overview_dir("xpcs_mesh_vs_static")
    names = []
    suffix = "" if beta_mode == "fixed" else "_beta_bounded"
    labels = {k: f"{k}: {picks[k]}" for k in picks}

    def legend(fig, x=0.91):
        handles = [plt.Line2D([], [], color=colours[k], marker="o") for k in picks]
        fig.legend(
            handles,
            [labels[k] for k in picks],
            loc="center left",
            bbox_to_anchor=(x, 0.5),
            frameon=False,
            fontsize=8,
        )

    def title(target):
        parts = [f"$t/t^* = {target:g}$"]
        for kind in picks:
            entry = data[kind][target][2]
            if entry.get("n_trains"):
                c = entry["ferritin_mg_per_ml"]
                parts.append(f"{kind} {entry['n_trains']} tr, {c:.0f} mg/ml")
        return "\n".join(parts)

    suptitle = (
        CONDITIONS[sample]
        + r": XPCS, mesh (hit trains) vs static at matched $t/t^*$ "
        + rf"(hit trains within $\pm${MATCH_TOLERANCE:g} pooled)"
    )

    # c2(τ) and its fit at four rings
    rings = [0, 3, 7, 11]
    fig, axs = plotting.panel_grid(
        len(rings) * len(MATCH_T),
        len(MATCH_T),
        panel=(2.6, 2.2),
        pad=(0, 0.8),
        sharey="row",
    )
    for j, target in enumerate(MATCH_T):
        for kind in picks:
            g2, fit, _ = data[kind][target]
            if g2 is None:
                continue
            for i, r in enumerate(rings):
                ax = axs[i, j]
                ax.errorbar(
                    g2.tau,
                    g2.c2[r],
                    g2.sigma[r],
                    color=colours[kind],
                    marker="o",
                    ms=2,
                    lw=0,
                    elinewidth=0.5,
                )
                if fit is not None and fit.resolved[r]:
                    ax.plot(fit.tau, fit.curve[r], color=colours[kind], lw=1)
        axs[0, j].set_title(title(target), fontsize=7)
    q_rings = 10 * next(
        g2.roi_q.values
        for kind in picks
        for g2, _, _ in data[kind].values()
        if g2 is not None
    )
    for i, r in enumerate(rings):
        axs[i, 0].set_ylabel(f"$c_2$, $q$ = {q_rings[r]:.3f} nm$^{{-1}}$", fontsize=8)
        for ax in axs[i]:
            ax.axhline(0, color="0.6", lw=0.5)
    for ax in axs[-1]:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau$ (s)")
    legend(fig)
    fig.suptitle(suptitle, fontsize=10)
    names.append(
        plotting.save(fig, out / f"xpcs_mesh_vs_static_g2_{sample}{suffix}.png")
    )

    # Γ(q) and τ(q), a panel per t/t*
    for quantity in ("gamma", "tau", "stretch"):
        fig, axs = plotting.panel_grid(len(MATCH_T), 3, pad=(0.4, 0.8))
        for ax, target in zip(axs.flat, MATCH_T, strict=False):
            anchor = None
            for kind in picks:
                _, fit, entry = data[kind][target]
                if fit is None:
                    continue
                q = fit.q_nm
                ok = np.array(entry["resolved"])
                if quantity == "gamma":
                    y, err = fit.gamma, fit.err_gamma
                elif quantity == "stretch":
                    y, err = fit.stretch, fit.err_stretch
                else:
                    y = 1 / fit.gamma
                    err = fit.err_gamma / fit.gamma**2
                for mask, face in ((ok, colours[kind]), (~ok, "none")):
                    ax.errorbar(
                        q[mask],
                        y[mask],
                        err[mask],
                        color=colours[kind],
                        mfc=face,
                        marker="o",
                        ms=3,
                        lw=0,
                        elinewidth=0.6,
                    )
                if ok.any() and anchor is None:
                    k = np.flatnonzero(ok)[ok.sum() // 2]
                    anchor = (q[k], y[k])
                if "exponent_n" in entry and quantity != "stretch":
                    ax.text(
                        0.04,
                        0.94 - 0.08 * list(picks).index(kind),
                        f"$n$ = {entry['exponent_n']:.2f} "
                        f"$\\pm$ {entry['exponent_err']:.2f}",
                        transform=ax.transAxes,
                        color=colours[kind],
                        fontsize=7,
                        va="top",
                    )
            if quantity == "stretch":
                ax.axhline(1, color="0.5", ls="--", lw=0.8)
            elif anchor is not None:
                qq = np.geomspace(0.22, 0.9, 20)
                sign = 1 if quantity == "gamma" else -1
                for power, style in ((2, "--"), (1, ":")):
                    ax.plot(
                        qq,
                        anchor[1] * (qq / anchor[0]) ** (sign * power),
                        color="0.5",
                        ls=style,
                        lw=0.8,
                    )
            plotting.log_q_axis(ax, lim=(0.22, 0.9), ticks=(0.3, 0.5, 0.8))
            if quantity == "stretch":
                ax.set_ylim(0, 2.1)
            else:
                ax.set_yscale("log")
                ax.set_ylim(*((1e3, 1e7) if quantity == "gamma" else (1e-7, 1e-3)))
            ax.set_title(title(target), fontsize=7)
        plotting.label_grid(
            axs,
            {
                "gamma": r"$\Gamma$ (s$^{-1}$)",
                "tau": r"$\tau = 1/\Gamma$ (s)",
                "stretch": r"KWW exponent $\alpha$",
            }[quantity],
        )
        legend(fig)
        guide = (
            (
                r"$c_2 = \beta_\mathrm{ref}\,e^{-2(\Gamma\tau)^\alpha} + b$, "
                r"$\beta_\mathrm{ref}$ fixed (r450, r493-496); "
                if beta_mode == "fixed"
                else r"$c_2 = \beta\,e^{-2(\Gamma\tau)^\alpha} + b$, "
                r"$0 \leq \beta \leq \beta_\mathrm{ref}$'s p84 (r450, r493-496); "
            )
            + (
                r"dashed $\alpha = 1$ (simple exponential); "
                if quantity == "stretch"
                else r"dashed $\propto q^{2}$ (diffusion), dotted $\propto q$ (shear); "
            )
            + r"open: not resolved (a parameter at its bound, "
            r"$1/\Gamma$ outside the lags or $\sigma_\Gamma \geq \Gamma$); "
            + r"$n$ from $\Gamma \propto q^n$ over filled points"
        )
        fig.suptitle(suptitle + "\n" + guide, fontsize=9)
        name = f"xpcs_mesh_vs_static_{quantity}_{sample}{suffix}.png"
        names.append(plotting.save(fig, out / name))

    # fit parameters against t/t*
    mid = 5
    npanel = 5 if beta_mode == "fixed" else 6
    fig, axs = plt.subplots(1, npanel, figsize=(3.5 * npanel, 3.4))
    for kind in picks:
        rows = [
            (e["t_over_tstar_mean"], e)
            for _, fit, e in data[kind].values()
            if fit is not None
        ]
        if not rows:
            continue
        t = np.array([r[0] for r in rows])
        style = {"color": colours[kind], "marker": "o", "ms": 3, "capsize": 2}
        n = np.array([r[1].get("exponent_n", np.nan) for r in rows])
        n_err = np.array([r[1].get("exponent_err", np.nan) for r in rows])
        axs[0].errorbar(t, n, n_err, **style)
        axs[1].errorbar(
            t,
            [r[1].get("D_um2_per_s", np.nan) for r in rows],
            [r[1].get("D_err_um2_per_s", np.nan) for r in rows],
            **style,
        )
        axs[2].errorbar(
            t,
            [r[1]["visible"][0] for r in rows],
            [r[1]["err_visible"][0] for r in rows],
            **style,
        )
        tau_mid = np.array(
            [r[1]["tau_s"][mid] if r[1]["resolved"][mid] else np.nan for r in rows]
        )
        axs[3].plot(t, tau_mid, **{k: v for k, v in style.items() if k != "capsize"})
        axs[4].errorbar(
            t,
            [r[1].get("stretch_mean", np.nan) for r in rows],
            [r[1].get("stretch_mean_err", np.nan) for r in rows],
            **style,
        )
        if beta_mode == "bounded":
            axs[5].errorbar(
                t,
                [r[1]["lost"][0] for r in rows],
                [r[1]["err_beta"][0] / r[1]["beta_ref"][0] for r in rows],
                **style,
            )
    for ax in axs:
        ax.set_xlabel("$t/t^*$")
    axs[0].axhline(2, color="0.5", ls="--", lw=0.8)
    axs[0].axhline(1, color="0.5", ls=":", lw=0.8)
    axs[0].set_ylabel(r"$n$ in $\Gamma \propto q^n$")
    axs[1].set_ylabel(r"$D = \Gamma/q^2$ ($\mu$m$^2$/s)")
    axs[2].axhline(1, color="0.5", ls="--", lw=0.8)
    axs[2].set_ylabel(
        rf"$c_2(\tau_1)/\beta_\mathrm{{ref}}$, $q$ = {q_rings[0]:.3f} nm$^{{-1}}$"
    )
    axs[3].set_ylabel(rf"$\tau$, $q$ = {q_rings[mid]:.3f} nm$^{{-1}}$ (s)")
    axs[3].set_yscale("log")
    axs[4].axhline(1, color="0.5", ls="--", lw=0.8)
    axs[4].set_ylabel(r"KWW $\alpha$, mean over resolved rings")
    if beta_mode == "bounded":
        axs[5].axhline(0, color="0.5", ls="--", lw=0.8)
        axs[5].set_ylabel(
            rf"$1 - \beta/\beta_\mathrm{{ref}}$, $q$ = {q_rings[0]:.3f} nm$^{{-1}}$"
        )
    legend(fig, x=0.99)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    name = f"xpcs_mesh_vs_static_params_{sample}{suffix}.png"
    names.append(plotting.save(fig, out / name))
    return names, record


def mesh_vs_static_all(db, refresh=False, beta_mode="fixed"):
    "mesh_vs_static() for every condition with both kinds; the fits in one JSON."
    written = []
    record = []
    for sample in CONDITIONS:
        names, entry = mesh_vs_static(sample, db, refresh, beta_mode)
        written += names
        if entry:
            record.append(entry)
    if written:
        suffix = "" if beta_mode == "fixed" else "_beta_bounded"
        name = f"xpcs_mesh_vs_static_fits{suffix}.json"
        (overview_dir("xpcs_mesh_vs_static") / name).write_text(
            json.dumps(
                {
                    "t_targets": list(MATCH_T),
                    "tolerance": MATCH_TOLERANCE,
                    "pooling": "every usable hit train within the tolerance: "
                    "analysis.cache.pool_g2 over the lags the window's runs share",
                    "model": "c2 = beta exp(-2 (Gamma tau)^alpha) + b (KWW; 'stretch' "
                    "is alpha) per ring, "
                    + (
                        "beta fixed at REFERENCE_BETA"
                        if beta_mode == "fixed"
                        else "beta in [0, REFERENCE_BETA p84], lost = 1 - beta/beta_ref"
                    )
                    + ", weighted by sigma, bounded (analysis.xpcs.kww); resolved = no "
                    "parameter at a bound, 1/Gamma inside the lags, "
                    "sigma_Gamma < Gamma",
                    "beta_mode": beta_mode,
                    "conditions": record,
                },
                indent=1,
            )
        )
        written.append(name)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="droplet ids (default: all)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="pool the windows again instead of reading xpcs_windows.nc",
    )
    parser.add_argument(
        "--matched",
        action="store_true",
        help="only the comparison of each condition's droplets at matched t/t*",
    )
    parser.add_argument(
        "--mesh-vs-static",
        action="store_true",
        help="only each condition's mesh and static representatives at matched t/t*",
    )
    parser.add_argument(
        "--beta",
        choices=("fixed", "bounded"),
        default="fixed",
        help="--mesh-vs-static's β: fixed at REFERENCE_BETA, or free in [0, its p84]",
    )
    args = parser.parse_args(argv)
    unknown = set(args.ids) - {d.id for d in DROPLETS if d.figures}
    if unknown:
        parser.error(f"unknown droplet ids: {sorted(unknown)}")
    if not plotting.latex_ok():
        parser.error(
            "LaTeX without type1cm.sty on PATH; run `module load texlive/2022` first"
        )
    import damnit

    db = damnit.Damnit(PROPOSAL)
    if args.matched:
        for name in matched_all(db, args.refresh):
            out = overview_dir("xpcs_matched_tstar").relative_to(REPO)
            print(f"matched t/t*: {out / name}", flush=True)
        return 0
    if args.mesh_vs_static:
        for name in mesh_vs_static_all(db, args.refresh, args.beta):
            out = overview_dir("xpcs_mesh_vs_static").relative_to(REPO)
            print(f"mesh vs static: {out / name}", flush=True)
        return 0
    failed = []
    for droplet in DROPLETS:
        if not droplet.figures or (args.ids and droplet.id not in args.ids):
            continue
        try:
            run_one(droplet, db, args.refresh)
        except Exception:  # one droplet's failure must not stop the rest
            failed.append(droplet.id)
            error = traceback.format_exc()
            print(f"{droplet.id}: FAILED\n{error}", flush=True)
            write_run(droplet, {"status": "FAILED", "error": error.splitlines()[-1]})
    write_index("xpcs")
    if failed:
        print(f"failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
