"""What the droplet figures share: the notebook's style, panel grids, colours, q axes.

The "science" style sets text in LaTeX, and the system TeX on the Maxwell nodes lacks
``type1cm.sty``, so run ``module load texlive/2022`` first (`latex_ok` checks).
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import xarray as xr

__all__ = [
    "DROPLET_COLOURS",
    "KIND_COLOURS",
    "NORM_Q",
    "label_grid",
    "latex_ok",
    "log_q_axis",
    "normalised_curve",
    "panel_grid",
    "pyplot",
    "save",
]

#: A group's droplets in order: Okabe & Ito's colour-blind-safe palette without its
#: yellow, which is too faint on white.
DROPLET_COLOURS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
)
KIND_COLOURS = {"static": DROPLET_COLOURS[0], "mesh": DROPLET_COLOURS[1]}

#: The SAXS notebook's normalisation window, nm⁻¹.
NORM_Q = (0.80, 0.95)


def latex_ok() -> bool:
    """Whether the TeX on PATH has ``type1cm.sty``, which the style's labels need."""
    kpsewhich = shutil.which("kpsewhich")
    return bool(
        kpsewhich
        and subprocess.run(
            [kpsewhich, "type1cm.sty"], capture_output=True
        ).stdout.strip()
    )


def pyplot():
    """``matplotlib.pyplot`` on the Agg backend with the notebook's "science" style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401  (registers the "science" style)

    plt.style.use("science")
    return plt


def panel_grid(n, ncols, *, panel=(3.6, 3.2), pad=(0.0, 0.6), **kwargs):
    """A figure of `n` panels in rows of `ncols`, `panel` inches each plus `pad`,
    shared axes unless `kwargs` say otherwise; the grid's unused panels hidden."""
    nrows = math.ceil(n / ncols)
    kwargs = {"sharex": True, "sharey": True} | kwargs
    fig, axs = pyplot().subplots(
        nrows,
        ncols,
        figsize=(panel[0] * ncols + pad[0], panel[1] * nrows + pad[1]),
        squeeze=False,
        **kwargs,
    )
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    return fig, axs


def label_grid(axs, ylabel, xlabel=r"$q$ (nm$^{-1}$)"):
    """Label a grid's bottom row with `xlabel` and its first column with `ylabel`."""
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)


def log_q_axis(ax, lim=(0.2, 1.0), ticks=(0.2, 0.3, 0.5, 1.0)):
    """A log q axis over `lim` (None: matplotlib's) labelled at `ticks` only, in
    plain numbers."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    ax.set_xscale("log")
    if lim is not None:
        ax.set_xlim(*lim)
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def normalised_curve(curves: xr.Dataset, k: int) -> xr.DataArray:
    """Curve `k` of a droplet's ``curves.nc`` over its mean in NORM_Q, NaN below its
    q_min and where not positive."""
    iq = curves.intensity.isel(curve=k)
    iq = iq / iq.sel(q=slice(*NORM_Q)).mean()
    return iq.where((iq > 0) & (curves.q >= curves.q_min[k]))


def save(fig, path: Path, dpi=200) -> str:
    """Save `fig` to `path`, creating its folder, close it and return the file name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    pyplot().close(fig)
    return path.name
