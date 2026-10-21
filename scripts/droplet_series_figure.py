"""V/V0 of one droplet beside its buffer-subtracted SAXS at points along it.

The MID version of the CoSAXS figure: (a) the droplet's V/V0 against time, with CAM5
snapshots at the chosen points, and (b) the I(q) at those points, normalised, offset
by a constant factor for clarity and coloured to match the markers.

Nothing here subtracts or scales. The curves are the droplet's ``curves.nc``, which
``notebooks/damnit_saxs.ipynb`` writes (via ``scripts/damnit_saxs_figures.py``): each
the buffer-subtracted I(q) per flux*path of N_AVG consecutive complete trains, with
the buffer matched at the same t/t*. Make or refresh it there first. This script
picks the curves nearest the requested t/t* (by default the six points of the
CoSAXS figure, COSAXS_TAU), normalises them the way the notebook's
NORMALISE does (mean over q 0.80-0.95) and cuts each below its own q_min.

Panel (a) is `analysis.cache.droplet_timeline` over the droplet's runs, as the
notebook calls it, so t* and V0 are the ones the curves were made with; the curves'
``t_star_s`` is checked against it. V0 is the first fitted volume of the droplet's
first run, t counts from that train. Each marker sits at its curve's first train,
and the inset is the CAM5 image of that train, cropped to the ROI ``droplet_fit``
fits in.

    PY=/gpfs/exfel/exp/MID/202601/p010400/usr/Software/MID-10400/.venv/bin/python
    $PY scripts/droplet_series_figure.py s443
    $PY scripts/droplet_series_figure.py s443 --tau 0.2 0.5 0.8 1.0 1.2 1.5

Writes ``droplet_series.{pdf,png}`` and ``droplet_series.json`` (what each marker
used) into the droplet's figure folder, beside ``curves.nc``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from damnit_saxs_figures import DROPLETS, folder, kept_runs

PROPOSAL = 10400
# The ROI `droplet_fit` fits in (src/amore/context.py), so the insets show what the
# volume was measured on.
CAMERA_ROI = np.s_[80:230, 200:400]
# the notebook's NORMALISE window and plotted q range (damnit_saxs.ipynb)
NORM_Q = (0.80, 0.95)
Q_RANGE = (0.2, 1.0)
# The six points of the CoSAXS figure (JPCL manuscript, Fig. 1a: 50 mg/ml ferritin,
# 3 w/v % PEG 6000), read off the plot at t = 276, 550, 711, 984, 1158 and 1401 s and
# divided by its t* = 1100 s ("a turning point at 1100 s", main.tex), t = 0 at
# injection.
COSAXS_TAU = (0.25, 0.50, 0.65, 0.89, 1.05, 1.27)


def pick_curves(curves, taus):
    """Index of the curve nearest each of `taus`, printing how far each one is."""
    tau = curves.t_over_tstar.values
    picked = [int(np.argmin(np.abs(tau - t))) for t in taus]
    for target, i in zip(taus, picked, strict=True):
        print(f"t/t* {target:.2f}: curve at {tau[i]:.3f} (run {int(curves.run[i])})")
    if len(set(picked)) < len(picked):
        print("note: some requested t/t* share a nearest curve")
    return picked


def normalised(curves, i):
    """Curve `i` over its mean in NORM_Q, NaN below its q_min and where not positive."""
    iq = curves.intensity.isel(curve=i)
    iq = iq / iq.sel(q=slice(*NORM_Q)).mean()
    return iq.where((iq.q >= float(curves.q_min[i])) & (iq > 0))


def camera_frame(run, train_id):
    import extra_data as ex
    from extra_data import by_id

    from analysis.camera import CAMERA_SOURCE, IMAGE_KEY

    cam = ex.open_run(PROPOSAL, run, data="raw")[CAMERA_SOURCE]
    image = cam.select_trains(by_id[[train_id]])[IMAGE_KEY].ndarray().squeeze()
    return image[CAMERA_ROI]


def draw(tl, curves, picked, images, offset, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0, 1, len(picked)))
    vv0 = tl.volume_mm3 / tl.volume_mm3[0]
    fig, (ax_v, ax_q) = plt.subplots(1, 2, figsize=(9, 4))

    ax_v.plot(tl.elapsed_s, vv0, color="0.3", lw=1.5)
    for k, i in enumerate(picked):
        tid = curves.first_trainId.values[i]
        ax_v.plot(
            tl.elapsed_s.sel(trainId=tid),
            vv0.sel(trainId=tid),
            "o",
            ms=9,
            color=colors[k],
            zorder=3,
        )
    ax_v.set_ylim(0, 1.05 * float(vv0.max()))
    # Insets on a fixed staircase down the upper right, as in the CoSAXS figure: the
    # curve falls left to right, so that region stays empty, and a fixed layout cannot
    # stack insets on top of each other.
    width, height = 0.24, 0.17
    for k, image in enumerate(images):
        tid = curves.first_trainId.values[picked[k]]
        x0, y0 = 0.3 + 0.14 * k, 0.8 - (height + 0.04) * k
        inset = ax_v.inset_axes([x0, y0, width, height])
        inset.imshow(image, cmap="gray")
        inset.set_xticks([])
        inset.set_yticks([])
        ax_v.annotate(
            "",
            xy=(x0, y0 + height / 2),
            xycoords="axes fraction",
            xytext=(float(tl.elapsed_s.sel(trainId=tid)), float(vv0.sel(trainId=tid))),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=1.2, shrinkA=6),
        )
    ax_v.set(xlabel=r"$t$ (s)", ylabel=r"$V/V(0)$")
    ax_v.grid(alpha=0.4)
    ax_v.text(0.95, 0.92, "(a)", transform=ax_v.transAxes, ha="right", fontsize=14)

    for k, i in enumerate(picked):
        iq = normalised(curves, i) * offset**k
        ax_q.loglog(
            iq.q,
            iq,
            color=colors[k],
            lw=1.5,
            label=f"$t/t^*={float(curves.t_over_tstar[i]):.2f}$",
        )
    ax_q.set(xlabel=r"$q$ (nm$^{-1}$)", ylabel=r"$I(q)$ norm.", xlim=Q_RANGE)
    ax_q.grid(which="major", alpha=0.4)
    ax_q.legend(fontsize=7, loc="lower left", reverse=True)
    ax_q.text(0.95, 0.92, "(b)", transform=ax_q.transAxes, ha="right", fontsize=14)

    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(destination.with_suffix(suffix), dpi=300)
    print(f"wrote {destination.with_suffix('.pdf')} and .png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("droplet", help="series id from damnit_saxs_figures, e.g. s443")
    parser.add_argument(
        "--tau",
        type=float,
        nargs="+",
        default=COSAXS_TAU,
        help="t/t* of the markers; default the CoSAXS figure's points",
    )
    parser.add_argument("--n-insets", type=int, default=4, help="CAM5 insets, first N")
    parser.add_argument("--offset", type=float, default=10.0, help="factor per curve")
    args = parser.parse_args(argv)

    import damnit
    import xarray as xr

    from analysis import cache

    (droplet,) = [d for d in DROPLETS if d[0] == args.droplet]
    out_dir = folder(droplet)
    curves_path = out_dir / "curves.nc"
    if not curves_path.exists():
        raise SystemExit(
            f"{curves_path} is missing: run scripts/damnit_saxs_figures.py "
            f"{args.droplet} first"
        )
    curves = xr.load_dataset(curves_path)
    runs = kept_runs(droplet)
    if list(curves.attrs["droplet_runs"]) != list(runs):
        raise SystemExit(
            f"{curves_path} was made from runs {curves.attrs['droplet_runs']}, "
            f"damnit_saxs_figures lists {runs}: remake it"
        )

    tl = cache.droplet_timeline(runs, damnit.Damnit(PROPOSAL))
    if not np.isclose(tl.attrs["t_star_s"], curves.attrs["t_star_s"], rtol=1e-6):
        raise SystemExit(
            f"t* of the timeline ({tl.attrs['t_star_s']:.1f} s) is not the "
            f"curves' ({curves.attrs['t_star_s']:.1f} s): remake curves.nc"
        )

    picked = pick_curves(curves, args.tau)
    images = [
        camera_frame(int(curves.run[i]), int(curves.first_trainId[i]))
        for i in picked[: args.n_insets]
    ]
    destination = Path(out_dir) / "droplet_series"
    draw(tl, curves, picked, images, args.offset, destination)
    report = [
        {
            "run": int(curves.run[i]),
            "first_trainId": int(curves.first_trainId[i]),
            "requested_t_over_tstar": target,
            "t_over_tstar": float(curves.t_over_tstar[i]),
            "q_min": float(curves.q_min[i]),
            "inset": k < args.n_insets,
        }
        for k, (target, i) in enumerate(zip(args.tau, picked, strict=True))
    ]
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "droplet": args.droplet,
                "curves": str(curves_path),
                "args": vars(args),
                "markers": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
