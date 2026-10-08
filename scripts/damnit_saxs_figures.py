"""Make the SAXS figures of every droplet after r338 from notebooks/damnit_saxs.ipynb.

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

The droplets are the run triage's series after r338 (inventory of 2026-10-01), static
droplets only. Left out: the silica references (r382-385, r389), r463 (exploded),
the buffer r464-466 itself (it is the background) and r402-408 (attenuator 0.6, for
which the notebook has no flux correction).

The notebook's plot style renders text with LaTeX, and the system TeX on the Maxwell
nodes lacks ``type1cm.sty``: load a complete one first, or uncached labels fail. Run
this script with the shared environment too (the overviews need xarray and matplotlib):

    module load texlive/2022
    PY=/gpfs/exfel/exp/MID/202601/p010400/usr/Software/MID-10400/.venv/bin/python
    $PY scripts/damnit_saxs_figures.py                  # every droplet, then overviews
    $PY scripts/damnit_saxs_figures.py s443 s409        # some droplets, then overviews
    $PY scripts/damnit_saxs_figures.py --overview-only  # overviews from curves.nc
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
# figure titles; they are set in LaTeX, where an unescaped % starts a comment
CONDITIONS = {
    "ferritin50": "ferritin 50 mg/ml, no PEG",
    "ferritin50-peg1k": r"ferritin 50 mg/ml + PEG 1K 5\,\% w/v",
    "ferritin50-peg6k": r"ferritin 50 mg/ml + PEG 6K 5\,\% w/v",
}
OUTPUTS = ("saxs_subtracted_norm.png", "saxs_subtracted.png", "curves.nc")

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
        "status": "ok" if result.returncode == 0 and fresh else "FAILED",
        "seconds": round(time.perf_counter() - started, 1),
    }


def overview(sample):
    """One figure per normalisation: a panel per droplet of `sample`, shared axes and
    t/t* colour scale. Returns the files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scienceplots  # noqa: F401  (registers the notebook's "science" style)
    import xarray as xr
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    plt.style.use("science")
    data = []
    for droplet in DROPLETS:
        path = folder(droplet) / "curves.nc"
        if droplet[1] == sample and path.exists():
            data.append((droplet[0], xr.load_dataset(path)))
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
        axs[0, 0].set_xlim(0.2, 1.0)  # as the droplet figures
        axs[0, 0].xaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1.0]))
        axs[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axs[0, 0].xaxis.set_minor_formatter(NullFormatter())
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
        fig.suptitle(CONDITIONS[sample])
        name = f"saxs_{sample}_norm.png" if normalise else f"saxs_{sample}.png"
        (FIGURES / "overview").mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "overview" / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
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
    for sample in CONDITIONS:
        for name in overview(sample):
            print(f"overview: figures/overview/{name}", flush=True)
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
