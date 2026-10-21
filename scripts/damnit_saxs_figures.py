"""Make the SAXS figures of every droplet after r240 from notebooks/damnit_saxs.ipynb.

For each droplet the notebook is exported to a plain Python script with that droplet's
parameters filled in, saved as
``figures/droplets/<id>_<sample>_r<first>-<last>/damnit_saxs_<id>.py``, and run with
the shared environment's Python (the ``mid-10400`` kernel's). The script writes beside
itself the two subtracted-I(q) figures and ``curves.nc``, the plotted curves with what
made them; running it again remakes them. The notebook stays the source; the script
records the run. The buffer's figure goes to ``figures/buffer/``.

Then one overview per condition goes to ``figures/overview/``: a grid with one panel per
droplet, all on the same axes and the same t/t* colour scale, read from the droplets'
``curves.nc``. Normalised (to the mean over q 0.80-0.95) and per flux*path versions.
Beside them, per condition, its droplets side by side at matched t/t*
(``saxs_matched_tstar_<sample>_norm.png``, a panel per t/t*, a line per droplet), with
the blocks they used in one JSON. The mesh droplets get figures of their own
(``saxs_<sample>_mesh_norm.png``, ``saxs_matched_tstar_<sample>_mesh_norm.png``), never
mixed with the static ones.

The droplets are the run triage's series after r240 (inventory of 2026-10-21). Left
out: the silica references (r288, r382-385, r389), r463 (exploded), the buffer r464-466
itself (it is the background), and r289 (attenuator 0.041), r307 (0.0047) and r402-408
(0.6), for which the notebook has no flux correction.

The mesh droplets (s240, s258, s289, s308; SSHEX fly scans, attenuator 0.1006) are run
with the notebook's MESH_SCAN, so a mesh run's curves average only the trains that hit
the droplet (`analysis.triage.classify.hits`, class "on"). Their figures carry caveats
the static ones do not: a hit can cross the droplet off-centre while the scale assumes
the full chord; the q_min rule (0.156 a_x) was measured on static droplets; and like
s338-s400 they predate the r433 -> r443 stray-light change, while the buffer r464 is
after it.

The notebook's plot style renders text with LaTeX, and the system TeX on the Maxwell
nodes lacks ``type1cm.sty``: load a complete one first, or uncached labels fail. Run
this script with the shared environment too (the overviews need xarray and matplotlib):

    module load texlive/2022
    PY=/gpfs/exfel/exp/MID/202601/p010400/usr/Software/MID-10400/.venv/bin/python
    $PY scripts/damnit_saxs_figures.py                  # every droplet, then overviews
    $PY scripts/damnit_saxs_figures.py s443 s409        # some droplets, then overviews
    $PY scripts/damnit_saxs_figures.py --overview-only  # overviews from curves.nc
    $PY scripts/damnit_saxs_figures.py --xpcs           # XPCS mesh vs static only

The XPCS comparison (``xpcs_mesh_vs_static_*.png``) takes the same mesh and static
representatives and, per MATCH_T, pools every hit train within MATCH_TOLERANCE
(`analysis.cache.pool_g2`) and fits each ring (weighted, bounded): c2(τ),
Γ(q), τ(q) and the fit parameters against t/t*, the fits in one JSON.
"""

import argparse
import datetime
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "damnit_saxs.ipynb"
FIGURES = REPO / "figures"
# the interpreter of the shared mid-10400 kernel (scripts/shared_env.sh)
SHARED_PYTHON = Path(
    "/gpfs/exfel/exp/MID/202601/p010400/usr/Software/MID-10400/.venv/bin/python"
)

# id, sample, runs (what the droplet's V0 and t* come from), runs left out and why,
# PEG w/v % at the start
DROPLETS = [
    ("s240", "ferritin50-peg6k", range(240, 258), {}, 5.0),
    ("s258", "ferritin50-peg6k", range(258, 276), {}, 5.0),
    (
        "s289",
        "ferritin50-peg6k",
        range(289, 308),
        {
            289: "attenuator 0.041, no flux correction",
            307: "attenuator 0.0047, no flux correction",
        },
        5.0,
    ),
    ("s308", "ferritin50", range(308, 338), {}, 0.0),
    (
        "s338",
        "ferritin50",
        range(338, 370),
        {
            349: "no droplet fit",
            353: "dark",
            354: "dark",
            355: "dark",
            356: "left out by clean_range",
        },
        0.0,
    ),
    ("s370", "ferritin50", range(370, 382), {}, 0.0),
    ("s390", "ferritin50", range(390, 394), {}, 0.0),
    ("s394", "ferritin50", range(394, 401), {}, 0.0),
    ("s409", "ferritin50-peg1k", range(409, 420), {}, 5.0),
    ("s423", "ferritin50-peg6k", range(423, 433), {}, 5.0),
    ("s433", "ferritin50-peg6k", range(433, 443), {}, 5.0),
    (
        "s443",
        "ferritin50-peg6k",
        range(443, 453),
        {447: "left out by clean_range"},
        5.0,
    ),
    ("s453", "ferritin50-peg6k", range(453, 463), {}, 5.0),
    ("s467", "ferritin50-peg6k", range(467, 477), {}, 5.0),
    ("s477", "ferritin50-peg6k", range(477, 487), {}, 5.0),
    ("s487", "ferritin50-peg6k", range(487, 497), {}, 5.0),
]
# probed by SSHEX mesh scans (s308's last two runs, r336-337, are static; the notebook
# keeps every train of a static run)
MESH_DROPLETS = {"s240", "s258", "s289", "s308"}
# figure titles; they are set in LaTeX, where an unescaped % starts a comment
CONDITIONS = {
    "ferritin50": "ferritin 50 mg/ml, no PEG",
    "ferritin50-peg1k": r"ferritin 50 mg/ml + PEG 1K 5\,\% w/v",
    "ferritin50-peg6k": r"ferritin 50 mg/ml + PEG 6K 5\,\% w/v",
}
OUTPUTS = ("saxs_subtracted_norm.png", "saxs_subtracted.png", "curves.nc")
# the conditions compared at matched t/t*: a droplet shows its nearest block, and only
# within MATCH_TOLERANCE of the target (about half the 0.13-0.18 spacing of the blocks).
# Pure ferritin ends near t/t* = 2.1, hence 2.0 as the last.
MATCH_T = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
MATCH_TOLERANCE = 0.08
# the common q grid of the mesh/static ratio, inside the droplet figures' q range
# the mesh droplet shown per condition in the mesh/static comparison (the others stay in
# their own mesh figures); a condition not listed shows all its mesh droplets
MESH_REPRESENTATIVE = {"ferritin50-peg6k": "s258"}
# likewise the static droplet; static droplets vary too much to pool. s394 for no PEG is
# my pick (a fresh droplet at attenuator 1.0, no flux-correction issue), not the user's
STATIC_REPRESENTATIVE = {"ferritin50-peg6k": "s443", "ferritin50": "s394"}
# the bounds of Γ in the per-ring XPCS fit, s⁻¹: 1/Γ from 10 ms down to 10 ns, two
# decades past either end of the 0.44-135 µs lag window
XPCS_GAMMA_BOUNDS = (1e2, 1e8)
KNOWN_Q = [0.2 * (1.0 / 0.2) ** (i / 199) for i in range(200)]
# blocks the matching passes over, (run, first trainId) -> why. Found 2026-10-08 as the
# only blocks in any curves.nc whose mean I over q 0.3-0.4 is under a tenth of the
# median of up to two blocks on each side; the droplet figures still show them.
NEAR_ZERO_BLOCKS = {
    (357, 2637483940): "near zero: I(q 0.3-0.4) 0.8 % of its neighbours'",
    (477, 2637914817): "near zero: I(q 0.3-0.4) 1.3 % of its neighbours'",
}
# a condition's droplets, in DROPLETS order: Okabe & Ito's colour-blind-safe palette
# without its yellow, which is too faint on white
DROPLET_COLOURS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
)

HEADER = '''"""SAXS figures of droplet {id} ({sample}, runs {first}-{last}).

Exported from notebooks/damnit_saxs.ipynb ({version}) by scripts/damnit_saxs_figures.py
on {date}. The notebook is the source; this file records the run that made the figures
and curves.nc beside it, and running it again remakes them:

    module load texlive/2022
    {python} {name}
"""

import matplotlib

matplotlib.use("Agg")  # no display: the figures are saved, plt.show() does nothing'''


def kept_runs(droplet):
    _, _, runs, excluded, _ = droplet
    return [r for r in runs if r not in excluded]


def folder(droplet):
    droplet_id, sample = droplet[:2]
    runs = kept_runs(droplet)
    return FIGURES / "droplets" / f"{droplet_id}_{sample}_r{runs[0]}-{runs[-1]}"


def notebook_version():
    "The commit the notebook comes from, and whether it differs from it."
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", str(NOTEBOOK)],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"git {head}" + (" with uncommitted changes" if dirty else "")


def export(droplet, parameters, name):
    """The notebook's code cells as one script, the droplet's values after the
    parameters cell; IPython magics are dropped."""
    droplet_id, sample = droplet[:2]
    runs = parameters["DROPLET_RUNS"]
    parts = [
        HEADER.format(
            id=droplet_id,
            sample=sample,
            first=runs[0],
            last=runs[-1],
            version=notebook_version(),
            date=datetime.date.today().isoformat(),
            python=SHARED_PYTHON,
            name=name,
        )
    ]
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        code = "\n".join(line for line in lines if not line.lstrip().startswith("%"))
        if code.strip():
            parts.append(code.strip())
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            values = "\n".join(
                f"{key} = {value!r}" for key, value in parameters.items()
            )
            parts.append(
                f"# This droplet's parameters (they replace the ones above)\n{values}"
            )
    return "\n\n\n".join(parts) + "\n"


def run_one(droplet):
    droplet_id, sample, _, excluded, peg = droplet
    out = folder(droplet)
    out.mkdir(parents=True, exist_ok=True)
    parameters = {
        "DROPLET_ID": droplet_id,
        "SAMPLE": sample,
        "DROPLET_RUNS": kept_runs(droplet),
        "PLOT_RUNS": None,
        "FERRITIN_MG_PER_ML": 50.0,
        "PEG_PERCENT_WV": peg,
        "MESH_SCAN": droplet_id in MESH_DROPLETS,
        "FIGURES_ROOT": str(FIGURES),
    }
    script = out / f"damnit_saxs_{droplet_id}.py"
    script.write_text(export(droplet, parameters, script.name))
    started = time.perf_counter()
    begun = time.time()
    result = subprocess.run(
        [str(SHARED_PYTHON), script.name], cwd=out, capture_output=True, text=True
    )
    (out / "run.log").write_text(result.stdout + result.stderr)
    fresh = all(
        (out / f).exists() and (out / f).stat().st_mtime >= begun for f in OUTPUTS
    )
    return {
        "id": droplet_id,
        "sample": sample,
        "folder": str(out.relative_to(REPO)),
        "script": script.name,
        "runs": kept_runs(droplet),
        "excluded": {str(r): why for r, why in excluded.items()},
        "peg_percent_wv": peg,
        "mesh_scan": droplet_id in MESH_DROPLETS,
        "status": "ok" if result.returncode == 0 and fresh else "FAILED",
        "seconds": round(time.perf_counter() - started, 1),
    }


def groups():
    "The (sample, mesh) pairs that get overviews of their own, in CONDITIONS order."
    present = {(d[1], d[0] in MESH_DROPLETS) for d in DROPLETS}
    return [(s, m) for s in CONDITIONS for m in (False, True) if (s, m) in present]


def group_suffix(mesh):
    return "_mesh" if mesh else ""


def group_title(sample, mesh):
    return CONDITIONS[sample] + (", mesh scans (hit trains)" if mesh else "")


def load_curves(sample, mesh):
    "(droplet id, curves.nc) of every droplet of `sample` and `mesh` made so far."
    import xarray as xr

    return [
        (droplet[0], xr.load_dataset(folder(droplet) / "curves.nc"))
        for droplet in DROPLETS
        if droplet[1] == sample
        and (droplet[0] in MESH_DROPLETS) == mesh
        and (folder(droplet) / "curves.nc").exists()
    ]


def q_axis(ax):
    "The droplet figures' q range and ticks, on a log axis."
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    ax.set_xlim(0.2, 1.0)
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def overview(sample, mesh):
    """One figure per normalisation: a panel per droplet of `sample` (mesh or static),
    shared axes and t/t* colour scale. Returns the files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scienceplots  # noqa: F401  (registers the notebook's "science" style)

    plt.style.use("science")
    data = load_curves(sample, mesh)
    if not data:
        return []
    norm = matplotlib.colors.Normalize(
        0, max(float(ds.t_over_tstar.max()) for _, ds in data)
    )
    cmap = plt.get_cmap("viridis")
    ncols = min(4, len(data))
    nrows = math.ceil(len(data) / ncols)
    written = []
    for normalise in (True, False):
        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(3.6 * ncols + 1.2, 3.2 * nrows + 0.6),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        # the grid can have more panels than droplets
        shown = []
        for ax, (droplet_id, ds) in zip(axs.flat, data, strict=False):
            for k in range(ds.sizes["curve"]):
                iq = ds.intensity[k]
                if normalise:
                    iq = iq / iq.sel(q=slice(0.80, 0.95)).mean()
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
        for ax in axs.flat[len(data) :]:
            ax.set_visible(False)
        q_axis(axs[0, 0])
        if normalise:
            axs[0, 0].set_ylim(0.5, 25)
        else:  # the bulk of the curves; a stray near-zero block may run off the bottom
            low, high = np.nanpercentile(np.concatenate(shown), [1, 100])
            axs[0, 0].set_ylim(low / 2, high * 2)
        for ax in axs[-1]:
            ax.set_xlabel(r"$q$ (nm$^{-1}$)")
        for ax in axs[:, 0]:
            ax.set_ylabel(
                r"$I(q)\,/\,\langle I \rangle_{0.80-0.95}$"
                if normalise
                else r"$I(q)$ per flux$\cdot$path"
            )
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axs,
            label="$t/t^*$",
            shrink=0.9,
        )
        fig.suptitle(group_title(sample, mesh))
        name = f"saxs_{sample}{group_suffix(mesh)}{'_norm' if normalise else ''}.png"
        (FIGURES / "overview").mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "overview" / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(name)
    return written


def match_block(ds, target):
    """The block of `ds` to show at t/t* = `target`: the nearest one not in
    NEAR_ZERO_BLOCKS, if within MATCH_TOLERANCE. Returns (index or None, record)."""
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401  (registers the notebook's "science" style)

    plt.style.use("science")
    data = load_curves(sample, mesh)
    if not data:
        return None, []
    if len(data) > len(DROPLET_COLOURS):
        raise ValueError(
            f"{len(data)} droplets in {group_title(sample, mesh)!r}, "
            f"only {len(DROPLET_COLOURS)} colours"
        )
    ncols = 3
    nrows = math.ceil(len(MATCH_T) / ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.2 * nrows + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    record = []
    handles = {}
    for ax, target in zip(axs.flat, MATCH_T, strict=False):
        concentrations = []
        for (droplet_id, ds), colour in zip(data, DROPLET_COLOURS, strict=False):
            k, entry = match_block(ds, target)
            record.append({"sample": sample, "mesh_scan": mesh} | entry)
            if k is None:
                continue
            iq = ds.intensity[k]
            iq = iq / iq.sel(q=slice(0.80, 0.95)).mean()
            iq = iq.where((iq > 0) & (ds.q >= ds.q_min[k]))
            (handles[droplet_id],) = ax.loglog(ds.q, iq, color=colour, lw=1.0)
            concentrations.append(entry["ferritin_mg_per_ml"])
        title = f"$t/t^* = {target:g}$"
        if concentrations:
            low, high = (f"{f(concentrations):.0f}" for f in (min, max))
            title += f", {low} mg/ml" if low == high else f", {low}-{high} mg/ml"
        ax.set_title(title, fontsize=9)
    for ax in axs.flat[len(MATCH_T) :]:
        ax.set_visible(False)
    q_axis(axs[0, 0])
    axs[0, 0].set_ylim(0.5, 25)  # as the normalised overviews
    for ax in axs[-1]:
        ax.set_xlabel(r"$q$ (nm$^{-1}$)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$I(q)\,/\,\langle I \rangle_{0.80-0.95}$")
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
        f"{group_title(sample, mesh)} at matched $t/t^*$ (nearest block within "
        f"$\\pm${MATCH_TOLERANCE:g}; ferritin range per panel)"
    )
    out = FIGURES / "overview"
    out.mkdir(parents=True, exist_ok=True)
    name = f"saxs_matched_tstar_{sample}{group_suffix(mesh)}_norm.png"
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return name, record


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
        blocks = "saxs_matched_tstar_blocks.json"
        (FIGURES / "overview" / blocks).write_text(
            json.dumps(
                {
                    "t_targets": list(MATCH_T),
                    "tolerance": MATCH_TOLERANCE,
                    "normalisation": "mean over q 0.80-0.95 nm^-1",
                    "blocks": record,
                },
                indent=1,
            )
        )
        written.append(blocks)
    return written


def compare_mesh_static(sample):
    """`sample`'s mesh and static droplets on the same axes at matched t/t*: a panel per
    MATCH_T, static droplets in blue, mesh droplets in vermillion, normalised to the mean
    over q 0.80-0.95. A second figure gives, per panel, the ratio of the mesh droplets'
    mean curve to the static ones' (geometric means over the droplets with a block
    there). Returns the files written and the blocks shown. Needs both kinds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scienceplots  # noqa: F401  (registers the notebook's "science" style)

    plt.style.use("science")
    mesh = [
        (d, ds)
        for d, ds in load_curves(sample, True)
        if d == MESH_REPRESENTATIVE.get(sample, d)
    ]
    static = [
        (d, ds)
        for d, ds in load_curves(sample, False)
        if d == STATIC_REPRESENTATIVE.get(sample, d)
    ]
    kinds = {"static": static, "mesh": mesh}
    if not all(kinds.values()):
        return [], []
    colours = {"static": "#0072B2", "mesh": "#D55E00"}
    ncols = 3
    nrows = math.ceil(len(MATCH_T) / ncols)
    size = (3.6 * ncols, 3.2 * nrows + 0.6)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=size, sharex=True, sharey=True, squeeze=False
    )
    fig_r, axs_r = plt.subplots(
        nrows, ncols, figsize=size, sharex=True, sharey=True, squeeze=False
    )
    record = []
    handles = {}
    for ax, ax_r, target in zip(axs.flat, axs_r.flat, MATCH_T, strict=False):
        curves = {"static": [], "mesh": []}
        concentrations = {"static": [], "mesh": []}
        for kind, data in kinds.items():
            for droplet_id, ds in data:
                k, entry = match_block(ds, target)
                record.append({"sample": sample, "mesh_scan": kind == "mesh"} | entry)
                if k is None:
                    continue
                iq = ds.intensity[k]
                iq = iq / iq.sel(q=slice(0.80, 0.95)).mean()
                iq = iq.where((iq > 0) & (ds.q >= ds.q_min[k]))
                (handles[kind],) = ax.loglog(ds.q, iq, color=colours[kind], lw=1.2)
                curves[kind].append(iq.interp(q=KNOWN_Q).values)
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
            ax_r.semilogx(KNOWN_Q, mean["mesh"] / mean["static"], color="k", lw=1.0)
            ax_r.axhline(1, color="0.6", lw=0.5)
        ax_r.set_title(f"$t/t^* = {target:g}$", fontsize=9)
    for a in (axs, axs_r):
        for ax in a.flat[len(MATCH_T) :]:
            ax.set_visible(False)
        q_axis(a[0, 0])
        for ax in a[-1]:
            ax.set_xlabel(r"$q$ (nm$^{-1}$)")
    axs[0, 0].set_ylim(0.5, 25)  # as the normalised overviews
    axs_r[0, 0].set_ylim(0.5, 2)
    axs_r[0, 0].set_yscale("log")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$I(q)\,/\,\langle I \rangle_{0.80-0.95}$")
    for ax in axs_r[:, 0]:
        ax.set_ylabel("mesh / static")
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
    out = FIGURES / "overview"
    out.mkdir(parents=True, exist_ok=True)
    names = [
        f"saxs_mesh_vs_static_{sample}_norm.png",
        f"saxs_mesh_vs_static_ratio_{sample}.png",
    ]
    for f, name in zip((fig, fig_r), names, strict=True):
        f.savefig(out / name, dpi=200, bbox_inches="tight")
        plt.close(f)
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
        name = "saxs_mesh_vs_static_blocks.json"
        (FIGURES / "overview" / name).write_text(
            json.dumps(
                {
                    "t_targets": list(MATCH_T),
                    "tolerance": MATCH_TOLERANCE,
                    "normalisation": "mean over q 0.80-0.95 nm^-1",
                    "blocks": record,
                },
                indent=1,
            )
        )
        written.append(name)
    return written


def xpcs_windows(droplet, db):
    """`droplet`'s hit trains in each MATCH_T window, pooled: per target the pooled
    g2 (`analysis.cache.pool_g2`) of every usable hit train of its runs whose t/t* is
    within MATCH_TOLERANCE, with how many there were and their mean t/t* and ferritin
    concentration. t* and t/t* are the SAXS notebook's (`cache.droplet_timeline` of
    the droplet's runs). Returns {target: (pooled g2 or None, record)} and the runs'
    hit sources."""
    import numpy as np
    import xarray as xr

    from analysis import cache
    from analysis.common.status import FrameStatus
    from analysis.utils import droplet_composition

    droplet_id, sample, _, _, peg = droplet
    runs = kept_runs(droplet)
    tl = cache.droplet_timeline(runs, db)
    tau = tl.elapsed_s / tl.attrs["t_star_s"]
    conc = xr.DataArray(
        droplet_composition(
            tl.volume_mm3,
            tl.volume_mm3[0].values,
            ferritin_mg_per_ml_initial=50.0,
            peg_percent_wv_initial=peg,
        ).ferritin_mg_per_ml,
        dims="trainId",
        coords={"trainId": tl.trainId},
    )
    terms = (
        "status",
        "train_g2_num",
        "train_g2_den",
        "train_baseline_num",
        "train_baseline_den",
    )
    parts = []
    sources = {}
    for run_nr in runs:
        ds = cache.xpcs_run(run_nr, db, v0_run=runs[0])
        sources[run_nr] = ds.attrs.get("hits_source")
        ok = ds.trainId[ds.status == int(FrameStatus.OK)].values
        t = tau.reindex(trainId=ok).values
        near = np.isfinite(t) & np.any(
            np.abs(t[:, None] - np.array(MATCH_T)[None, :]) <= MATCH_TOLERANCE, axis=1
        )
        part = ds[list(terms)].sel(trainId=ok[near]).load()
        part.attrs = {"period_s": ds.attrs.get("period_s")}
        parts.append(part)
        ds.close()
    shapes = {(p.sizes["lag_frame"], p.attrs["period_s"]) for p in parts}
    if len(shapes) != 1:
        raise ValueError(f"{droplet_id}: runs differ in lags or pulse period {shapes}")
    pool = xr.concat(parts, "trainId", combine_attrs="override")
    t = tau.reindex(trainId=pool.trainId).values
    out = {}
    for target in MATCH_T:
        chosen = pool.trainId.values[np.abs(t - target) <= MATCH_TOLERANCE]
        entry = {"t_target": target, "id": droplet_id, "n_trains": int(chosen.size)}
        if chosen.size == 0:
            entry["skipped"] = f"no hit train within {MATCH_TOLERANCE}"
            out[target] = (None, entry)
            continue
        entry |= {
            "t_over_tstar_mean": round(float(tau.sel(trainId=chosen).mean()), 3),
            "ferritin_mg_per_ml": round(float(conc.sel(trainId=chosen).mean()), 1),
            "runs": sorted({int(r) for r in tl.run.sel(trainId=chosen).values}),
        }
        out[target] = (cache.pool_g2(pool, chosen), entry)
    return out, {str(r): s for r, s in sources.items()}


def fit_window(g2):
    """Per ring, a weighted fit of a pooled g2's baseline-subtracted ``c2`` to
    ``β exp(−2Γτ) + b`` (errors ``sigma``, β in [0, 1], Γ in XPCS_GAMMA_BOUNDS). Not
    `analysis.xpcs.figures.fit_g2`, the DAMNIT column's: that is one unweighted joint
    fit over all rings, so one ring that does not converge fails them all, and it
    fits flat noise as readily as a decay. A ring is ``resolved`` when its contrast is
    significant (β ≥ 3 σ_β), 1/Γ lies inside the lag window and σ_Γ < Γ. The slope n
    of Γ ∝ qⁿ and D = Γ/q² (weighted mean) use the resolved rings only. Returns (fit,
    record), the fit a namespace of per-ring arrays."""
    import types

    import numpy as np
    from scipy.optimize import curve_fit

    q_nm = 10 * g2.roi_q.values  # roi_q is in Å⁻¹
    tau = g2.tau.values
    n = q_nm.size
    fit = types.SimpleNamespace(
        q=g2.roi_q.values,
        tau=tau,
        curve=np.full((n, tau.size), np.nan),
        **{k: np.full(n, np.nan) for k in ("gamma", "err_gamma", "beta", "err_beta")},
    )

    def model(t, beta, gamma, base):
        return beta * np.exp(-2 * gamma * t) + base

    failed = {}
    for r in range(n):
        y, e = g2.c2.values[r], g2.sigma.values[r]
        ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
        if ok.sum() < 6:
            failed[r] = "fewer than 6 points"
            continue
        p0 = (max(float(y[ok][:3].mean()), 1e-3), 1 / float(np.median(tau)), 0.0)
        try:
            popt, pcov = curve_fit(
                model,
                tau[ok],
                y[ok],
                p0=p0,
                sigma=e[ok],
                absolute_sigma=True,
                bounds=((0, XPCS_GAMMA_BOUNDS[0], -1), (1, XPCS_GAMMA_BOUNDS[1], 1)),
                maxfev=20000,
            )
        except (RuntimeError, ValueError) as error:
            failed[r] = str(error)
            continue
        err = np.sqrt(np.diag(pcov))
        fit.beta[r], fit.gamma[r] = popt[:2]
        fit.err_beta[r], fit.err_gamma[r] = err[:2]
        fit.curve[r] = model(tau, *popt)
    relax = 1 / fit.gamma
    with np.errstate(invalid="ignore"):
        resolved = (
            (fit.beta >= 3 * fit.err_beta)
            & (relax >= tau.min())
            & (relax <= tau.max())
            & (fit.err_gamma < fit.gamma)
        )
    fit.resolved = resolved
    record = {
        "q_nm": np.round(q_nm, 4).tolist(),
        "gamma_per_s": fit.gamma.tolist(),
        "err_gamma_per_s": fit.err_gamma.tolist(),
        "tau_s": relax.tolist(),
        "beta": fit.beta.tolist(),
        "err_beta": fit.err_beta.tolist(),
        "resolved": resolved.tolist(),
        "n_resolved": int(resolved.sum()),
    }
    if failed:
        record["ring_fit_failed"] = {str(k): v for k, v in failed.items()}
    if resolved.any():
        # D = Γ/q², nm²/s → µm²/s, weighted by its inverse variance
        d = fit.gamma[resolved] / q_nm[resolved] ** 2 * 1e-6
        w = (fit.err_gamma[resolved] / q_nm[resolved] ** 2 * 1e-6) ** -2
        record |= {
            "D_um2_per_s": float(np.average(d, weights=w)),
            "D_err_um2_per_s": float(w.sum() ** -0.5),
        }
    if resolved.sum() >= 3:
        # Γ ∝ q^n over the resolved rings, weighted by the relative errors of Γ
        (slope, _), cov = np.polyfit(
            np.log(q_nm[resolved]),
            np.log(fit.gamma[resolved]),
            1,
            w=fit.gamma[resolved] / fit.err_gamma[resolved],
            cov="unscaled",
        )
        record |= {"exponent_n": float(slope), "exponent_err": float(cov[0, 0] ** 0.5)}
    return fit, record


def compare_xpcs_mesh_static(sample, db):
    """`sample`'s mesh and static representatives' XPCS at matched t/t*, every hit
    train within MATCH_TOLERANCE pooled (`xpcs_windows`) and fitted per ring
    (`fit_window`; fits drawn and Γ, τ filled only where resolved).
    Four figures: c2(τ) with the fit at four rings, a panel per (ring, t/t*); Γ against
    q and τ = 1/Γ against q, a panel per t/t*, with the Γ ∝ q² (diffusion) and Γ ∝ q
    (shear) slopes; and against t/t* the slope n of Γ ∝ qⁿ, D = Γ/q², β at the lowest
    ring and τ at a middle one. Returns the files written and the record."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scienceplots  # noqa: F401  (registers the notebook's "science" style)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    plt.style.use("science")
    mesh = [d[0] for d in DROPLETS if d[1] == sample and d[0] in MESH_DROPLETS]
    picks = {
        "static": STATIC_REPRESENTATIVE.get(sample),
        # a condition with one mesh droplet needs no entry
        "mesh": MESH_REPRESENTATIVE.get(sample, mesh[0] if len(mesh) == 1 else None),
    }
    if not all(picks.values()):
        return [], {}
    by_id = {d[0]: d for d in DROPLETS}
    colours = {"static": "#0072B2", "mesh": "#D55E00"}
    data = {}
    record = {"sample": sample, "droplets": {}}
    for kind, droplet_id in picks.items():
        windows, sources = xpcs_windows(by_id[droplet_id], db)
        fits = {}
        for target, (g2, entry) in windows.items():
            fit, fitted = (None, {}) if g2 is None else fit_window(g2)
            entry |= fitted
            fits[target] = (g2, fit, entry)
        data[kind] = fits
        record["droplets"][kind] = {
            "id": droplet_id,
            "hit_sources": sources,
            "windows": [e for _, _, e in fits.values()],
        }
        print(f"xpcs {sample} {kind} {droplet_id}: windows done", flush=True)

    out = FIGURES / "overview"
    out.mkdir(parents=True, exist_ok=True)
    names = []
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
    ncols = len(MATCH_T)
    fig, axs = plt.subplots(
        len(rings),
        ncols,
        figsize=(2.6 * ncols, 2.2 * len(rings) + 0.8),
        sharex=True,
        sharey="row",
        squeeze=False,
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
    names.append(f"xpcs_mesh_vs_static_g2_{sample}.png")
    fig.savefig(out / names[-1], dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Γ(q) and τ(q), a panel per t/t*
    nrows = math.ceil(len(MATCH_T) / 3)
    for quantity in ("gamma", "tau"):
        fig, axs = plt.subplots(
            nrows,
            3,
            figsize=(3.6 * 3 + 0.4, 3.2 * nrows + 0.8),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for ax, target in zip(axs.flat, MATCH_T, strict=False):
            anchor = None
            for kind in picks:
                _, fit, entry = data[kind][target]
                if fit is None:
                    continue
                q = 10 * fit.q
                ok = np.array(entry["resolved"])
                if quantity == "gamma":
                    y, err = fit.gamma, fit.err_gamma
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
                if "exponent_n" in entry:
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
            if anchor is not None:
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
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.22, 0.9)
            ax.set_ylim(*((1e3, 1e7) if quantity == "gamma" else (1e-7, 1e-3)))
            ax.xaxis.set_major_locator(FixedLocator([0.3, 0.5, 0.8]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(title(target), fontsize=7)
        for ax in axs.flat[len(MATCH_T) :]:
            ax.set_visible(False)
        for ax in axs[-1]:
            ax.set_xlabel(r"$q$ (nm$^{-1}$)")
        for ax in axs[:, 0]:
            ax.set_ylabel(
                r"$\Gamma$ (s$^{-1}$)"
                if quantity == "gamma"
                else r"$\tau = 1/\Gamma$ (s)"
            )
        legend(fig)
        guide = (
            r"dashed $\propto q^{2}$ (diffusion), dotted $\propto q$ (shear); "
            r"open: not resolved ($\beta < 3\sigma_\beta$, $1/\Gamma$ outside the lags "
            r"or $\sigma_\Gamma \geq \Gamma$); "
            r"$n$ from $\Gamma \propto q^n$ over filled points"
        )
        fig.suptitle(suptitle + "\n" + guide, fontsize=9)
        names.append(f"xpcs_mesh_vs_static_{quantity}_{sample}.png")
        fig.savefig(out / names[-1], dpi=200, bbox_inches="tight")
        plt.close(fig)

    # fit parameters against t/t*
    mid = 5
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
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
            [r[1]["beta"][0] for r in rows],
            [r[1]["err_beta"][0] for r in rows],
            **style,
        )
        tau_mid = np.array(
            [r[1]["tau_s"][mid] if r[1]["resolved"][mid] else np.nan for r in rows]
        )
        axs[3].plot(t, tau_mid, **{k: v for k, v in style.items() if k != "capsize"})
    for ax in axs:
        ax.set_xlabel("$t/t^*$")
    axs[0].axhline(2, color="0.5", ls="--", lw=0.8)
    axs[0].axhline(1, color="0.5", ls=":", lw=0.8)
    axs[0].set_ylabel(r"$n$ in $\Gamma \propto q^n$")
    axs[1].set_ylabel(r"$D = \Gamma/q^2$ ($\mu$m$^2$/s)")
    axs[2].set_ylabel(rf"$\beta$, $q$ = {q_rings[0]:.3f} nm$^{{-1}}$")
    axs[3].set_ylabel(rf"$\tau$, $q$ = {q_rings[mid]:.3f} nm$^{{-1}}$ (s)")
    axs[3].set_yscale("log")
    legend(fig, x=0.99)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    names.append(f"xpcs_mesh_vs_static_params_{sample}.png")
    fig.savefig(out / names[-1], dpi=200, bbox_inches="tight")
    plt.close(fig)
    return names, record


def compare_xpcs_all():
    "compare_xpcs_mesh_static() for every condition with both kinds; fits in one JSON."
    import damnit

    db = damnit.Damnit(10400)
    written = []
    record = []
    for sample in CONDITIONS:
        names, entry = compare_xpcs_mesh_static(sample, db)
        written += names
        if entry:
            record.append(entry)
    if written:
        name = "xpcs_mesh_vs_static_fits.json"
        (FIGURES / "overview" / name).write_text(
            json.dumps(
                {
                    "t_targets": list(MATCH_T),
                    "tolerance": MATCH_TOLERANCE,
                    "pooling": "every usable hit train within the tolerance, "
                    "analysis.cache.pool_g2",
                    "model": "c2 = beta exp(-2 Gamma tau) + b per ring, weighted by "
                    "sigma, bounded (fit_window); resolved = beta >= 3 sigma_beta, "
                    "1/Gamma inside the lags, sigma_Gamma < Gamma",
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
        "--overview-only",
        action="store_true",
        help="only remake the overviews from existing curves.nc",
    )
    parser.add_argument(
        "--xpcs",
        action="store_true",
        help="only make the XPCS mesh-vs-static comparison",
    )
    args = parser.parse_args(argv)
    unknown = set(args.ids) - {d[0] for d in DROPLETS}
    if unknown:
        parser.error(f"unknown droplet ids: {sorted(unknown)}")
    kpsewhich = shutil.which("kpsewhich")
    found = (
        kpsewhich
        and subprocess.run(
            [kpsewhich, "type1cm.sty"], capture_output=True
        ).stdout.strip()
    )
    if not found:
        parser.error(
            "LaTeX without type1cm.sty on PATH; run `module load texlive/2022` first"
        )
    if args.xpcs:
        for name in compare_xpcs_all():
            print(f"xpcs mesh vs static: figures/overview/{name}", flush=True)
        return 0

    results = []
    if not args.overview_only:
        for droplet in DROPLETS:
            if args.ids and droplet[0] not in args.ids:
                continue
            results.append(run_one(droplet))
            r = results[-1]
            print(
                f"{r['id']}: {r['status']} in {r['seconds']:.0f} s -> {r['folder']}",
                flush=True,
            )
        index = FIGURES / "droplets" / "index.json"
        previous = json.loads(index.read_text()) if index.exists() else []
        merged = {r["id"]: r for r in previous} | {r["id"]: r for r in results}
        index.write_text(
            json.dumps(sorted(merged.values(), key=lambda r: r["id"]), indent=1)
        )
    for sample, mesh in groups():
        for name in overview(sample, mesh):
            print(f"overview: figures/overview/{name}", flush=True)
    for name in matched_all():
        print(f"matched t/t*: figures/overview/{name}", flush=True)
    for name in compare_all():
        print(f"mesh vs static: figures/overview/{name}", flush=True)
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
