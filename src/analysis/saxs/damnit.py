"""The DAMNIT-facing surface of ``agipd_saxs``.

Everything the DAMNIT variable needs lives here rather than in
``src/amore/context.py``. DAMNIT ``exec``s that file into a dict, so a function defined
there cannot be pickled to the spawned workers this pass runs.

The context-file variable is therefore two lines: call :func:`agipd_saxs` and
hand DAMNIT the result. Nothing here imports DAMNIT, so the same functions
work from a notebook, from a Slurm script, or from the pyBeamtime reader.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.writer import per_pulse, pooled_per_train

__all__ = [
    "config_for",
    "agipd_saxs",
    "overview_figure",
    "per_pulse_from_file",
    "pooled_from_file",
]

log = logging.getLogger(__name__)


def config_for(proposal: int, run_no: int, **overrides: Any) -> AgipdSaxsConfig:
    """The run's configuration, with the beamtime defaults from ``config.py``.

    :param overrides: any :class:`AgipdSaxsConfig` field. The geometry file,
        the pixel mask and the photon energy all enter ``config_hash``, so
        overriding one means the run will not resume into an output file
        written without it — which is the intended behaviour, not a limitation.
    """
    return AgipdSaxsConfig(proposal=proposal, run=run_no, **overrides)


def agipd_saxs(proposal: int, run_no: int, **overrides: Any) -> Any:
    """Integrate a whole run and return ``I(q)`` per ``(trainId, pulseId)``.

    The per-frame sums stay under ``cfg.output_root``; what comes back is the
    intensity grid DAMNIT stores. DAMNIT's own ``run`` object is not used — it
    would be a proc-only collection, leaving every check in ``plan.run_checks``
    unavailable — so the pass opens proc and raw itself.

    :raises IncompleteRun: some frame did not reach ``OK``. Deliberately not
        caught: a partial I(q) that looks whole is worse than a failed variable.
    """
    from analysis.saxs.run import run_agipd_saxs

    cfg = config_for(proposal, run_no, **overrides)
    log.info("agipd_saxs on p%d r%d -> %s", proposal, run_no, cfg.output_file)
    return run_agipd_saxs(cfg, reduce="per_pulse")


def per_pulse_from_file(path: str | Path) -> Any:
    """Re-read a finished run's ``(trainId, pulseId, q)`` grid."""
    return per_pulse(Path(path))


def pooled_from_file(path: str | Path) -> Any:
    """Re-read a finished run's per-train pooled ``I(q)`` and ``σ(q)``."""
    return pooled_per_train(Path(path))


def overview_figure(grid: Any, title: str = "") -> Any:
    """Three-panel I(q) overview: mean curve, per-pulse map, per-train map.

    The two maps go through ``extra.utils.imshow2``, which is what the old
    ``agipd_iq_overview`` used: for a DataArray it hands off to xarray's own
    ``plot.imshow``, so the colour bar and the real train and pulse ids on the
    axes come from the array's coordinates rather than being drawn by hand.

    :param grid: the DataArray from :func:`agipd_saxs` or
        :func:`per_pulse_from_file`.
    """
    import matplotlib.pyplot as plt
    from extra.utils import imshow2
    from matplotlib.gridspec import GridSpec

    # Slots holding no frame are stored as zeros;
    # NaN is what makes xarray's mean skip them and what imshow2 leaves blank.
    # The copy is skipped when every slot holds a frame — the usual case, and
    # ~0.9 GB of it.
    present = grid["n_frames"] > 0
    intensity = grid if bool(present.all()) else grid.where(present)

    fig = plt.figure(figsize=(9, 7))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.plot(grid["q"].values, intensity.mean(("trainId", "pulseId")).values)
    ax1.set_yscale("log")
    ax1.set_xlabel("q [nm$^{-1}$]")
    ax1.set_ylabel("I(q) [photons / solid angle]")
    ax1.set_title(title or "Mean I(q) over all trains and pulses")
    ax1.grid(alpha=0.3)

    panels = (
        (
            ax2,
            intensity.mean("trainId"),
            "Pulse ID",
            "I(q) per-pulse (averaged over trains)",
        ),
        (
            ax3,
            intensity.mean("pulseId"),
            "Train",
            "I(q) per-train (averaged over pulses)",
        ),
    )
    for ax, data, ylabel, subtitle in panels:
        values = np.asarray(data.values)
        # LogNorm needs a positive value to anchor on, and imshow2 raises
        # looking for one when a run has nothing integrated at all.
        if np.isfinite(values).any() and (values[np.isfinite(values)] > 0).any():
            imshow2(data, ax=ax, lognorm=True)
        else:
            ax.text(
                0.5,
                0.5,
                "no integrated frames",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel("q [nm$^{-1}$]")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)

    fig.tight_layout()
    return fig
