"""Make the SAXS figures of every droplet after r240, one folder per droplet.

Each droplet gets ``figures/droplets/saxs/<id>_<sample>_r<first>-<last>/``:
``curves.nc``, its buffer-subtracted I(q) with what made them, and two figures of
them, ``saxs_subtracted_norm.png`` (normalised to the mean over q 0.80-0.95) and
``saxs_subtracted.png`` (per flux*path), coloured by t/t*. ``run.log`` holds the
droplet's summary, or the traceback when it failed, and ``index.json`` beside the
folders lists every droplet made.

A curve is the mean of N_AVG consecutive complete trains, CURVES_PER_RUN of them
spread over each run. Every train is scaled per flux*path (`analysis.cache.saxs_run`)
and has the buffer subtracted at its own t/t*, weighted by the volume fraction the
ferritin leaves to it. The buffer is the droplet r464 -> r465 -> r466 (PEG 6K 5 % +
150 mM NaCl, fresh in r464; r463 is left out, a bubble made it explode), binned by
t/t* (its figure goes to ``figures/droplets/saxs/s464_peg6k-buffer_r464-466/``). Its
D^2 law never breaks before the droplet goes unstable in r466, so its t* is the end
of its clean data. Each curve is plotted from its own q_min, the a_x rule below.

Then the overviews, each kind in its own ``figures/overview/<kind>/``, read from the
droplets' ``curves.nc``. ``saxs/``: one grid per condition, a panel per droplet, all
on the same axes and the same t/t* colour scale, once normalised to the mean over q
0.80-0.95 and once per flux*path. ``saxs_matched_tstar/``: per condition, its
droplets side by side at matched t/t* (``saxs_matched_tstar_<sample>_norm.png``, a
panel per t/t*, a line per droplet), with the blocks they used in one JSON.
``saxs_mesh_vs_static/``: each condition's mesh and static representatives on the
same axes and their ratio. The mesh droplets get figures of their own
(``saxs_<sample>_mesh_norm.png``, ``saxs_matched_tstar_<sample>_mesh_norm.png``),
never mixed with the static ones. ``volume/`` holds ``--volumes``' V/V0 figures.

The droplets are `analysis.droplets.DROPLETS` with ``figures`` set, the runs left out
there. A mesh droplet (SSHEX fly scans, attenuator 0.1006) averages only the trains
of its mesh runs that hit the droplet (`analysis.triage.classify.hits`, class "on").
Their figures carry caveats the static ones do not. A hit can cross the droplet
off-centre while the scale assumes the full chord. The q_min rule comes from static
droplets. Like s338-s400, the mesh droplets were measured before the stray-light
pattern changed between r433 and r443, and the buffer r464 after it.

The plot style renders text with LaTeX, and the system TeX on the Maxwell nodes lacks
``type1cm.sty``, so load a complete one first or uncached labels fail. Run with the
shared environment:

    module load texlive/2022
    PY=/gpfs/exfel/exp/MID/202601/p010400/usr/Software/MID-10400/.venv/bin/python
    $PY scripts/damnit_saxs_figures.py                  # every droplet, then overviews
    $PY scripts/damnit_saxs_figures.py s443 s409        # some droplets, then overviews
    $PY scripts/damnit_saxs_figures.py --overview-only  # overviews from curves.nc
    $PY scripts/damnit_saxs_figures.py --volumes ferritin50  # V/V0 against t/t*

The XPCS comparison of the same representatives is scripts/damnit_xpcs_figures.py
--mesh-vs-static.
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
import traceback

from _common import REPO, droplets_dir, overview_dir

from analysis import plotting
from analysis.droplets import (
    CONDITIONS,
    DROPLETS,
    FERRITIN_MG_PER_ML,
    MATCH_T,
    MATCH_TOLERANCE,
    NEAR_ZERO_BLOCKS,
    PROPOSAL,
    REPRESENTATIVE,
    condition_title,
    groups,
)
from analysis.plotting import DROPLET_COLOURS, KIND_COLOURS

BUFFER_RUNS = [464, 465, 466]
BUFFER_NAME = f"s{BUFFER_RUNS[0]}_peg6k-buffer_r{BUFFER_RUNS[0]}-{BUFFER_RUNS[-1]}"
BUFFER_BIN = 100  # consecutive complete trains per buffer bin
N_PULSES = 155  # pulses per train averaged in the cache (and in the XGM normalisation)
N_AVG = 10  # consecutive complete trains averaged per curve
CURVES_PER_RUN = 3
# Lowest trustworthy q per curve. Below it the three chi sectors of the unmasked wedge
# stop agreeing, because the droplet shadows the stray scatter near the beam; measured
# on r443-452 it scales with the droplet's horizontal semi-axis a_x. The floor covers
# the 4 of 15 plateau blocks that already disagreed at 0.105-0.114 nm^-1.
Q_MIN_PER_MM = 0.156  # nm^-1 per mm of a_x
Q_MIN_FLOOR = 0.11  # nm^-1
# Divisor per attenuator transmission that puts a sample on the buffer's scale. At
# 0.506 everything per logged flux (the stray-light streak and the droplet's own
# scattering) reads ~1.3x the transmission-1.0 runs (r467/477/487 against r443/453).
FLUX_CORR = {0.506: 1.3, 0.10059096681319211: 10}

# The V/V0 figure draws a running median over VOLUME_MEDIAN_TRAINS (5 s at 10 Hz) over
# the per-train volumes, whose fit jumps as a mesh scan moves the droplet. Its line
# breaks where consecutive trains are more than VOLUME_GAP_S apart, i.e. between runs.
VOLUME_MEDIAN_TRAINS = 51
VOLUME_GAP_S = 5.0
# the common q grid of the mesh/static ratio, inside the droplet figures' q range
RATIO_Q = [0.2 * (1.0 / 0.2) ** (i / 199) for i in range(200)]


def folder(droplet):
    return droplets_dir("saxs") / droplet.name


def code_version():
    "The commit this tree is on, and whether scripts/ or src/ differ from it."
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", "scripts", "src"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"git {head}" + (" with uncommitted changes" if dirty else "")


def buffer_library(db):
    """The buffer's I(q) per flux*path against t/t*, over (tau, q), each bin the mean
    of BUFFER_BIN consecutive complete trains, and its figure. Attributes:
    ``t_star_s``, ``t_star_source`` and ``transmission``, the buffer's one attenuator
    transmission."""
    import numpy as np
    import xarray as xr

    from analysis import cache
    from analysis.utils import style_plot

    tl = cache.droplet_timeline(BUFFER_RUNS, db, stop_at_min_volume=True, t_star="end")
    runs = xr.concat(
        [
            cache.saxs_run(r, db, v0_run=BUFFER_RUNS[0], ferritin_mg_per_ml_initial=0)
            for r in BUFFER_RUNS
        ],
        "trainId",
        combine_attrs="drop",
    )
    tau = (tl.elapsed_s / tl.attrs["t_star_s"]).where(tl.used_in_fit)
    # complete trains with a scale factor and a t/t* before the droplet was lost
    usable = (runs.n_pulses_with_frames == N_PULSES) & np.isfinite(runs.scale)
    tids = np.intersect1d(
        runs.trainId.values[usable.values], tau.trainId.values[np.isfinite(tau.values)]
    )
    b = runs.sel(trainId=tids)
    n_bins = tids.size // BUFFER_BIN
    kept = slice(0, n_bins * BUFFER_BIN)
    library = xr.DataArray(
        np.nanmean(
            (b.intensity * b.scale).values[kept].reshape(n_bins, BUFFER_BIN, -1), axis=1
        ),
        dims=("tau", "q"),
        coords={
            "tau": tau.sel(trainId=tids)
            .values[kept]
            .reshape(n_bins, BUFFER_BIN)
            .mean(1),
            "q": b.q.values,
        },
    ).sortby("tau")
    if not np.all(np.diff(library.tau) > 0):
        raise ValueError("buffer bins are not monotonic in t/t*")
    library.attrs = {
        "t_star_s": tl.attrs["t_star_s"],
        "t_star_source": tl.attrs["t_star_source"],
        "transmission": float(
            np.unique(np.round(b.total_transmission.values, 4)).item()
        ),
    }

    plt = plotting.pyplot()
    fig, ax = plt.subplots(figsize=(6, 5))
    norm = plt.Normalize(vmin=float(library.tau.min()), vmax=float(library.tau.max()))
    for t in library.tau.values[:: max(1, n_bins // 12)]:
        ax.loglog(library.q, library.sel(tau=t), color=plt.cm.plasma(norm(t)), lw=0.8)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax, label="buffer $t/t^*$"
    )
    ax.set_title(f"buffer r{BUFFER_RUNS[0]}-{BUFFER_RUNS[-1]}", fontsize=14)
    style_plot(ax, ylabel=r"$I(q)$ per flux$\cdot$path")
    fig.tight_layout()
    plotting.save(fig, droplets_dir("saxs") / BUFFER_NAME / "buffer_library.png", 300)
    return library


def flux_corr(transmission, buffer_transmission):
    "The divisor that puts a sample at attenuator `transmission` on the buffer's scale."
    import numpy as np

    if np.isclose(transmission, buffer_transmission, rtol=0.01):
        return 1.0
    for att, corr in FLUX_CORR.items():
        if np.isclose(transmission, att, rtol=0.01):
            return corr
    raise ValueError(
        f"no flux correction known for attenuator {transmission} "
        f"(buffer: {buffer_transmission})"
    )


def subtract(s, tau, buffer):
    """Buffer-subtracted I(q) per flux*path, the mean over the trains of `s`, each
    train's buffer taken at its own t/t* `tau`."""
    import xarray as xr

    corr = xr.DataArray(
        [
            flux_corr(t, buffer.attrs["transmission"])
            for t in s.total_transmission.values
        ],
        coords={"trainId": s.trainId},
    )
    bg = buffer.interp(tau=tau.clip(buffer.tau.min(), buffer.tau.max()))
    return (s.intensity * s.scale / corr - (1 - s.phi_ferritin) * bg).mean("trainId")


def droplet_curves(droplet, db, buffer):
    """`droplet`'s buffer-subtracted curves as ``curves.nc`` holds them, over
    (curve, q), with their t/t*, q_min, ferritin concentration and first train."""
    import numpy as np
    import xarray as xr

    from analysis import cache
    from analysis.triage.classify import hits

    runs = droplet.kept_runs
    tl = cache.droplet_timeline(runs, db)
    conc = droplet.ferritin_mg_per_ml(tl).assign_coords(trainId=tl.trainId)
    t_star = tl.attrs["t_star_s"]
    tau_s = tl.elapsed_s / t_star
    curves = []  # (t/t*, q_min, I(q), ferritin mg/ml, run, first trainId)
    for run_nr in runs:
        ds = cache.saxs_run(
            run_nr,
            db,
            v0_run=runs[0],
            ferritin_mg_per_ml_initial=FERRITIN_MG_PER_ML,
            peg_percent_wv_initial=droplet.peg_percent_wv,
        )
        # complete trains that have a scale factor and a t/t*, picked by label
        usable = (ds.n_pulses_with_frames == N_PULSES) & np.isfinite(ds.scale)
        tids = np.intersect1d(ds.trainId.values[usable.values], tau_s.trainId.values)
        if droplet.mesh:
            run_hits = hits(run_nr)
            if run_hits["kind"] == "mesh":  # a static run of the droplet keeps all
                tids = np.intersect1d(tids, run_hits["train_ids"])
        for i in np.linspace(0, len(tids) - N_AVG, CURVES_PER_RUN).round().astype(int):
            block = tids[i : i + N_AVG]
            tau = tau_s.sel(trainId=block)
            a_x = float(ds.chord_mm.sel(trainId=block).mean()) / 2
            curves.append(
                (
                    float(tau.mean()),
                    max(Q_MIN_PER_MM * a_x, Q_MIN_FLOOR),
                    subtract(ds.sel(trainId=block), tau, buffer),
                    float(conc.sel(trainId=block).mean()),
                    run_nr,
                    int(block[0]),
                )
            )
    t, q_min, iq, ferritin, run, first = zip(*curves, strict=True)
    return xr.Dataset(
        {
            "intensity": (("curve", "q"), np.array([c.values for c in iq])),
            "t_over_tstar": ("curve", list(t)),
            "q_min": ("curve", list(q_min)),
            "ferritin_mg_per_ml": ("curve", list(ferritin)),
            "run": ("curve", list(run)),
            "first_trainId": ("curve", np.array(first, np.uint64)),
        },
        coords={"q": iq[0].q.values},
        attrs={
            "description": "buffer-subtracted I(q) per flux*path, each the mean of "
            f"{N_AVG} consecutive complete trains; q_min is the lowest q plotted "
            "(a_x rule); made by scripts/damnit_saxs_figures.py"
            + (
                "; mesh runs: hit trains only (analysis.triage.classify.hits)"
                if droplet.mesh
                else ""
            ),
            "code_version": code_version(),
            "droplet_id": droplet.id,
            "sample": droplet.sample,
            "droplet_runs": list(runs),
            "ferritin_mg_per_ml_initial": FERRITIN_MG_PER_ML,
            "peg_percent_wv_initial": droplet.peg_percent_wv,
            "mesh_scan": int(droplet.mesh),
            "t_star_s": t_star,
            "buffer_runs": list(BUFFER_RUNS),
            "buffer_t_star_s": buffer.attrs["t_star_s"],
            "flux_corr": json.dumps({str(k): v for k, v in FLUX_CORR.items()}),
            "q_min_per_mm": Q_MIN_PER_MM,
            "q_min_floor": Q_MIN_FLOOR,
        },
    )


def plot_curves(ds, normalise):
    """The droplet's curves coloured by t/t*, each from its q_min: normalised to the
    mean over q 0.80-0.95, or per flux*path."""
    from analysis.utils import style_plot

    plt = plotting.pyplot()
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(
        vmin=float(ds.t_over_tstar.min()), vmax=float(ds.t_over_tstar.max())
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in range(ds.sizes["curve"]):
        if normalise:
            iq = plotting.normalised_curve(ds, k)
        else:
            iq = ds.intensity[k]
            iq = iq.where((iq > 0) & (ds.q >= ds.q_min[k]))
        ax.loglog(ds.q, iq, color=cmap(norm(float(ds.t_over_tstar[k]))), lw=0.8)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="$t/t^*$")
    runs = ds.attrs["droplet_runs"]
    ax.set_title(f"[{runs[0]} ... {runs[-1]}]", fontsize=14)
    style_plot(
        ax,
        ylabel=r"$I(q)\,/\,\langle I \rangle_{0.80-0.95}$"
        if normalise
        else r"$I(q)$ per flux$\cdot$path",
    )
    ax.set_xlim(0.2, 1.0)  # the last bins at the detector edge are noisy
    if normalise:
        ax.set_ylim(0.5, 25)
    fig.tight_layout()
    return fig


def run_one(droplet, db, buffer):
    out = folder(droplet)
    started = time.perf_counter()
    try:
        ds = droplet_curves(droplet, db, buffer)
        out.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(out / "curves.nc")
        for normalise, name in (
            (True, "saxs_subtracted_norm.png"),
            (False, "saxs_subtracted.png"),
        ):
            plotting.save(plot_curves(ds, normalise), out / name, 300)
        beyond = int((ds.t_over_tstar > float(buffer.tau.max())).sum())
        log = (
            f"buffer: t* = {buffer.attrs['t_star_s']:.0f} s "
            f"({buffer.attrs['t_star_source']}), {buffer.sizes['tau']} bins over t/t* "
            f"{float(buffer.tau.min()):.3f}-{float(buffer.tau.max()):.3f}, "
            f"attenuator {buffer.attrs['transmission']}\n"
            f"sample: t* = {ds.attrs['t_star_s']:.0f} s; t/t* up to "
            f"{float(ds.t_over_tstar.max()):.2f}\n"
            f"{ds.sizes['curve']} curves; {beyond} lie beyond the buffer's t/t* = "
            f"{float(buffer.tau.max()):.2f} and use its final state\n"
        )
        status = "ok"
    except Exception:  # one droplet's failure must not stop the rest
        log = traceback.format_exc()
        status = "FAILED"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.log").write_text(log)
    return {
        "id": droplet.id,
        "sample": droplet.sample,
        "folder": str(out.relative_to(REPO)),
        "runs": droplet.kept_runs,
        "excluded": {str(r): why for r, why in droplet.excluded.items()},
        "peg_percent_wv": droplet.peg_percent_wv,
        "mesh_scan": droplet.mesh,
        "made": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_version": code_version(),
        "status": status,
        "seconds": round(time.perf_counter() - started, 1),
    }


def load_curves(sample, mesh):
    "(droplet id, curves.nc) of every droplet of `sample` and `mesh` made so far."
    import xarray as xr

    return [
        (droplet.id, xr.load_dataset(folder(droplet) / "curves.nc"))
        for droplet in DROPLETS
        if droplet.figures
        and (droplet.sample, droplet.mesh) == (sample, mesh)
        and (folder(droplet) / "curves.nc").exists()
    ]


def overview(sample, mesh):
    """One figure per normalisation: a panel per droplet of `sample` (mesh or static),
    shared axes and t/t* colour scale. Returns the files written."""
    import matplotlib
    import numpy as np

    plt = plotting.pyplot()
    data = load_curves(sample, mesh)
    if not data:
        return []
    norm = matplotlib.colors.Normalize(
        0, max(float(ds.t_over_tstar.max()) for _, ds in data)
    )
    cmap = plt.get_cmap("viridis")
    ncols = min(4, len(data))
    written = []
    for normalise in (True, False):
        fig, axs = plotting.panel_grid(len(data), ncols, pad=(1.2, 0.6))
        shown = []
        for ax, (droplet_id, ds) in zip(axs.flat, data, strict=False):
            for k in range(ds.sizes["curve"]):
                if normalise:
                    iq = plotting.normalised_curve(ds, k)
                else:
                    iq = ds.intensity[k]
                    iq = iq.where((iq > 0) & (ds.q >= ds.q_min[k]))
                color = cmap(norm(float(ds.t_over_tstar[k])))
                ax.loglog(ds.q, iq, color=color, lw=0.6)
                shown.append(iq.sel(q=slice(0.2, 1.0)).values)
            runs = ds.attrs["droplet_runs"]
            ax.set_title(
                f"{droplet_id}: r{runs[0]}-{runs[-1]}, "
                f"$t^*$ = {ds.attrs['t_star_s']:.0f} s",
                fontsize=9,
            )
        plotting.log_q_axis(axs[0, 0])
        if normalise:
            axs[0, 0].set_ylim(0.5, 25)
        else:  # the bulk of the curves; a stray near-zero block may run off the bottom
            low, high = np.nanpercentile(np.concatenate(shown), [1, 100])
            axs[0, 0].set_ylim(low / 2, high * 2)
        plotting.label_grid(
            axs,
            r"$I(q)\,/\,\langle I \rangle_{0.80-0.95}$"
            if normalise
            else r"$I(q)$ per flux$\cdot$path",
        )
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axs,
            label="$t/t^*$",
            shrink=0.9,
        )
        fig.suptitle(condition_title(sample, mesh))
        name = (
            f"saxs_{sample}{'_mesh' if mesh else ''}{'_norm' if normalise else ''}.png"
        )
        written.append(plotting.save(fig, overview_dir("saxs") / name))
    return written


def write_blocks(kind, name, record, **extra):
    """The blocks a set of matched figures shows, as JSON beside them in the `kind`
    overview folder."""
    (overview_dir(kind) / name).write_text(
        json.dumps(
            {
                "t_targets": list(MATCH_T),
                "tolerance": MATCH_TOLERANCE,
                **extra,
                "blocks": record,
            },
            indent=1,
        )
    )
    return name


def match_block(ds, target):
    """The block of `ds` to show at t/t* = `target`, or None.

    It is the nearest block not in NEAR_ZERO_BLOCKS, if within MATCH_TOLERANCE.
    Returns (index or None, record)."""
    import numpy as np

    t = ds.t_over_tstar.values
    keys = [
        (int(r), int(tid))
        for r, tid in zip(ds.run.values, ds.first_trainId.values, strict=True)
    ]
    bad = np.array([key in NEAR_ZERO_BLOCKS for key in keys])
    k = int(np.argmin(np.where(bad, np.inf, np.abs(t - target))))
    entry = {"t_target": target, "id": ds.attrs["droplet_id"]}
    near = [
        f"r{keys[b][0]} {keys[b][1]}: {NEAR_ZERO_BLOCKS[keys[b]]}"
        for b in np.flatnonzero(bad)
        if abs(t[b] - target) <= MATCH_TOLERANCE
    ]
    if near:
        entry["passed_over"] = near
    if abs(t[k] - target) > MATCH_TOLERANCE:
        entry["skipped"] = (
            f"no block within {MATCH_TOLERANCE} (nearest t/t* {t[k]:.2f})"
        )
        return None, entry
    entry |= {
        "t_over_tstar": round(float(t[k]), 3),
        "run": int(ds.run[k]),
        "first_trainId": int(ds.first_trainId[k]),
        "ferritin_mg_per_ml": round(float(ds.ferritin_mg_per_ml[k]), 1),
        "q_min": round(float(ds.q_min[k]), 4),
    }
    return k, entry


def matched(sample, mesh):
    """`sample`'s droplets (mesh or static) side by side at matched t/t*: a panel per
    MATCH_T, a line per droplet in its own colour, normalised to the mean over q
    0.80-0.95. Returns the file written and, per panel and droplet, the block shown or
    why none."""
    data = load_curves(sample, mesh)
    if not data:
        return None, []
    if len(data) > len(DROPLET_COLOURS):
        raise ValueError(
            f"{len(data)} droplets in {condition_title(sample, mesh)!r}, "
            f"only {len(DROPLET_COLOURS)} colours"
        )
    fig, axs = plotting.panel_grid(len(MATCH_T), 3)
    record = []
    handles = {}
    for ax, target in zip(axs.flat, MATCH_T, strict=False):
        concentrations = []
        for (droplet_id, ds), colour in zip(data, DROPLET_COLOURS, strict=False):
            k, entry = match_block(ds, target)
            record.append({"sample": sample, "mesh_scan": mesh} | entry)
            if k is None:
                continue
            iq = plotting.normalised_curve(ds, k)
            (handles[droplet_id],) = ax.loglog(ds.q, iq, color=colour, lw=1.0)
            concentrations.append(entry["ferritin_mg_per_ml"])
        title = f"$t/t^* = {target:g}$"
        if concentrations:
            low, high = (f"{f(concentrations):.0f}" for f in (min, max))
            title += f", {low} mg/ml" if low == high else f", {low}-{high} mg/ml"
        ax.set_title(title, fontsize=9)
    plotting.log_q_axis(axs[0, 0])
    axs[0, 0].set_ylim(0.5, 25)  # as the normalised overviews
    plotting.label_grid(axs, r"$I(q)\,/\,\langle I \rangle_{0.80-0.95}$")
    shown = [(d, ds) for d, ds in data if d in handles]
    fig.legend(
        [handles[d] for d, _ in shown],
        [
            f"{d}: r{ds.attrs['droplet_runs'][0]}-{ds.attrs['droplet_runs'][-1]}"
            for d, ds in shown
        ],
        loc="center left",
        bbox_to_anchor=(0.91, 0.5),  # just right of the grid
        frameon=False,
    )
    fig.suptitle(
        f"{condition_title(sample, mesh)} at matched $t/t^*$ (nearest block within "
        f"$\\pm${MATCH_TOLERANCE:g}; ferritin range per panel)"
    )
    name = f"saxs_matched_tstar_{sample}{'_mesh' if mesh else ''}_norm.png"
    return plotting.save(fig, overview_dir("saxs_matched_tstar") / name), record


def matched_all():
    "matched() for every condition, mesh and static apart; their blocks in one JSON."
    written = []
    record = []
    for sample, mesh in groups():
        name, blocks = matched(sample, mesh)
        if name:
            written.append(name)
            record += blocks
    if written:
        written.append(
            write_blocks(
                "saxs_matched_tstar",
                "saxs_matched_tstar_blocks.json",
                record,
                normalisation="mean over q 0.80-0.95 nm^-1",
            )
        )
    return written


def compare_mesh_static(sample):
    """`sample`'s mesh and static droplets on the same axes at matched t/t*: a panel per
    MATCH_T, static droplets in blue, mesh droplets in vermillion, normalised to the
    mean over q 0.80-0.95. A second figure gives, per panel, the ratio of the mesh
    droplets' mean curve to the static ones' (geometric means over the droplets with a
    block there). Returns the files written and the blocks shown. Needs both kinds."""
    import numpy as np

    kinds = {
        kind: [
            (d, ds)
            for d, ds in load_curves(sample, mesh)
            if d == REPRESENTATIVE.get((sample, mesh), d)
        ]
        for kind, mesh in (("static", False), ("mesh", True))
    }
    if not all(kinds.values()):
        return [], []
    fig, axs = plotting.panel_grid(len(MATCH_T), 3)
    fig_r, axs_r = plotting.panel_grid(len(MATCH_T), 3)
    record = []
    handles = {}
    for ax, ax_r, target in zip(axs.flat, axs_r.flat, MATCH_T, strict=False):
        curves = {"static": [], "mesh": []}
        concentrations = {"static": [], "mesh": []}
        for kind, data in kinds.items():
            for _, ds in data:
                k, entry = match_block(ds, target)
                record.append({"sample": sample, "mesh_scan": kind == "mesh"} | entry)
                if k is None:
                    continue
                iq = plotting.normalised_curve(ds, k)
                (handles[kind],) = ax.loglog(ds.q, iq, color=KIND_COLOURS[kind], lw=1.2)
                curves[kind].append(iq.interp(q=RATIO_Q).values)
                concentrations[kind].append(entry["ferritin_mg_per_ml"])
        title = f"$t/t^* = {target:g}$"
        for kind, values in concentrations.items():
            if values:
                title += f", {kind} {min(values):.0f}-{max(values):.0f} mg/ml"
        ax.set_title(title, fontsize=8)
        if all(curves.values()):
            with np.errstate(all="ignore"):
                mean = {
                    kind: np.exp(np.nanmean(np.log(np.array(c)), axis=0))
                    for kind, c in curves.items()
                }
            ax_r.semilogx(RATIO_Q, mean["mesh"] / mean["static"], color="k", lw=1.0)
            ax_r.axhline(1, color="0.6", lw=0.5)
        ax_r.set_title(f"$t/t^* = {target:g}$", fontsize=9)
    for a in (axs, axs_r):
        plotting.log_q_axis(a[0, 0])
    axs[0, 0].set_ylim(0.5, 25)  # as the normalised overviews
    axs_r[0, 0].set_ylim(0.5, 2)
    axs_r[0, 0].set_yscale("log")
    plotting.label_grid(axs, r"$I(q)\,/\,\langle I \rangle_{0.80-0.95}$")
    plotting.label_grid(axs_r, "mesh / static")
    fig.legend(
        [handles[k] for k in kinds if k in handles],
        [f"{k}: " + ", ".join(d for d, _ in kinds[k]) for k in kinds if k in handles],
        loc="center left",
        bbox_to_anchor=(0.91, 0.5),
        frameon=False,
        fontsize=8,
    )
    suffix = (
        r" at matched $t/t^*$ (nearest block within "
        rf"$\pm${MATCH_TOLERANCE:g}; ferritin range per panel)"
    )
    fig.suptitle(CONDITIONS[sample] + ": mesh (hit trains) vs static" + suffix)
    fig_r.suptitle(CONDITIONS[sample] + r": mesh / static")
    out = overview_dir("saxs_mesh_vs_static")
    names = [
        plotting.save(fig, out / f"saxs_mesh_vs_static_{sample}_norm.png"),
        plotting.save(fig_r, out / f"saxs_mesh_vs_static_ratio_{sample}.png"),
    ]
    return names, record


def compare_all():
    "compare_mesh_static() for every condition with both kinds; blocks in one JSON."
    written = []
    record = []
    for sample in CONDITIONS:
        names, blocks = compare_mesh_static(sample)
        written += names
        record += blocks
    if written:
        written.append(
            write_blocks(
                "saxs_mesh_vs_static",
                "saxs_mesh_vs_static_blocks.json",
                record,
                normalisation="mean over q 0.80-0.95 nm^-1",
            )
        )
    return written


def run_fluxes(runs, db):
    """Per run, the median over its trains of the XGM pulse energy (DAMNIT
    ``xgm_intensity``, the mean over every pulse of a train) times the attenuators'
    ``total_transmission``, in uJ per pulse. Runs without either are left out."""
    import numpy as np

    fluxes = {}
    for run_nr in runs:
        try:
            xgm = db[run_nr]["xgm_intensity"].read()
            att = db[run_nr]["total_transmission"].read()
        except KeyError:
            continue
        flux = float(np.nanmedian((xgm * att).values))  # trainIds aligned by xarray
        if np.isfinite(flux):
            fluxes[run_nr] = flux
    return fluxes


def flux_label(fluxes):
    "The legend's flux: one value, or the range when the runs differ by over 10 %."
    low, high = min(fluxes.values()), max(fluxes.values())
    if high <= 1.1 * low:
        return f"{sorted(fluxes.values())[len(fluxes) // 2]:.3g}"
    return f"{low:.3g}-{high:.3g}"


def volume_timelines(sample):
    """V/V0 against t/t* of every droplet of `sample`, static and mesh droplets in
    panels of their own, from the timeline the curves use (`cache.droplet_timeline` of
    the droplet's kept runs; V0 its first train). Each train faint, with a running
    median over VOLUME_MEDIAN_TRAINS on top; the line breaks where trains are more
    than VOLUME_GAP_S apart (between runs). The legend gives each droplet's flux,
    XGM times total transmission (`run_fluxes`). Returns the files written."""
    import damnit
    import numpy as np

    from analysis import cache

    plt = plotting.pyplot()
    db = damnit.Damnit(PROPOSAL)
    droplets = sorted(
        (d for d in DROPLETS if d.sample == sample), key=lambda d: d.runs[0]
    )
    kinds = [m for m in (False, True) if any(d.mesh == m for d in droplets)]
    if not kinds:
        return []
    fig, axs = plotting.panel_grid(len(kinds), len(kinds), panel=(5.0, 4.0), pad=(0, 0))
    record = []
    for ax, kind in zip(axs[0], kinds, strict=True):
        group = [d for d in droplets if d.mesh == kind]
        # more droplets than the palette has colours: viridis in run order
        colours = (
            DROPLET_COLOURS
            if len(group) <= len(DROPLET_COLOURS)
            else plt.get_cmap("viridis")(np.linspace(0, 0.95, len(group)))
        )
        for droplet, colour in zip(group, colours, strict=False):
            runs = droplet.kept_runs
            tl = cache.droplet_timeline(runs, db)
            t_star = tl.attrs["t_star_s"]
            v0 = float(tl.volume_mm3[0])
            fluxes = run_fluxes(runs, db)
            v = tl.volume_mm3 / v0
            smooth = v.rolling(
                trainId=VOLUME_MEDIAN_TRAINS, center=True, min_periods=1
            ).median()
            # a NaN after every gap breaks the line there
            gap = np.flatnonzero(np.diff(tl.elapsed_s.values) > VOLUME_GAP_S) + 1
            tau = np.insert((tl.elapsed_s / t_star).values, gap, np.nan)
            ax.plot(
                tau, np.insert(v.values, gap, np.nan), color=colour, lw=0.4, alpha=0.25
            )
            ax.plot(
                tau,
                np.insert(smooth.values, gap, np.nan),
                color=colour,
                lw=0.9,
                label=f"{droplet.id} {droplet.run_label}, $t^*$ = {t_star:.0f} s, "
                f"$F$ = {flux_label(fluxes)} $\\mu$J",
            )
            record.append(
                {
                    "id": droplet.id,
                    "mesh_scan": kind,
                    "runs": runs,
                    "v0_mm3": round(v0, 4),
                    "t_star_s": round(t_star, 1),
                    "t_star_source": tl.attrs["t_star_source"],
                    "t_star_fit_err_s": round(tl.attrs["t_star_fit_err_s"], 1),
                    "v_over_v0_at_t_star": round(
                        float(
                            (tl.volume_mm3 / v0).where(
                                tl.elapsed_s <= t_star, drop=True
                            )[-1]
                        ),
                        3,
                    ),
                    "t_over_tstar_max": round(float(tl.elapsed_s.max() / t_star), 2),
                    "flux_uJ_per_run": {str(r): round(f, 4) for r, f in fluxes.items()},
                }
            )
        ax.axvline(1.0, color="0.5", lw=0.6, ls=":")
        ax.set_xlabel("$t/t^*$")
        ax.set_title("mesh scans" if kind else "static", fontsize=9)
        ax.legend(fontsize=6, frameon=False)
    axs[0, 0].set_xlim(0, None)
    axs[0, 0].set_ylim(0, 1.05)
    axs[0, 0].set_ylabel("$V/V_0$")
    fig.suptitle(CONDITIONS[sample])
    out = overview_dir("volume")
    names = [plotting.save(fig, out / f"volume_{sample}.png"), f"volume_{sample}.json"]
    (out / names[1]).write_text(json.dumps(record, indent=1))
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="droplet ids (default: all)")
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help="only remake the overviews from existing curves.nc",
    )
    parser.add_argument(
        "--volumes",
        metavar="SAMPLE",
        choices=tuple(CONDITIONS),
        help="only make V/V0 against t/t* for every droplet of SAMPLE",
    )
    args = parser.parse_args(argv)
    unknown = set(args.ids) - {d.id for d in DROPLETS if d.figures}
    if unknown:
        parser.error(f"unknown droplet ids: {sorted(unknown)}")
    if not plotting.latex_ok():
        parser.error(
            "LaTeX without type1cm.sty on PATH; run `module load texlive/2022` first"
        )

    def shown(kind, name):
        return overview_dir(kind).relative_to(REPO) / name

    if args.volumes:
        for name in volume_timelines(args.volumes):
            print(f"volumes: {shown('volume', name)}", flush=True)
        return 0

    results = []
    if not args.overview_only:
        import damnit

        db = damnit.Damnit(PROPOSAL)
        buffer = buffer_library(db)
        for droplet in DROPLETS:
            if not droplet.figures or (args.ids and droplet.id not in args.ids):
                continue
            results.append(run_one(droplet, db, buffer))
            r = results[-1]
            print(
                f"{r['id']}: {r['status']} in {r['seconds']:.0f} s -> {r['folder']}",
                flush=True,
            )
        index = droplets_dir("saxs") / "index.json"
        previous = json.loads(index.read_text()) if index.exists() else []
        merged = {r["id"]: r for r in previous} | {r["id"]: r for r in results}
        index.write_text(
            json.dumps(sorted(merged.values(), key=lambda r: r["id"]), indent=1)
        )
    for sample, mesh in groups():
        for name in overview(sample, mesh):
            print(f"overview: {shown('saxs', name)}", flush=True)
    for name in matched_all():
        print(f"matched t/t*: {shown('saxs_matched_tstar', name)}", flush=True)
    for name in compare_all():
        print(f"mesh vs static: {shown('saxs_mesh_vs_static', name)}", flush=True)
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
