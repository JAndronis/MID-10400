"""The DAMNIT-facing surface of ``agipd_saxs`` (context file §9, phase P5).

Everything the DAMNIT context file needs lives here rather than in the context
file itself. DAMNIT ``exec``s that file into a dict, so a function defined
there cannot be pickled to a spawned worker — and this spawns one worker per
physical core (CLAUDE.md, Project).

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

    The per-frame sums stay in the output file under ``cfg.output_root``; what
    comes back is the intensity grid, which is what DAMNIT stores and what the
    overview plots read.

    DAMNIT's own ``run`` object is deliberately not used. A ``data="proc"``
    variable is handed a proc-only collection, and proc holds corrected
    detector files alone — no timeserver, no XGM, no motors — so every run
    check in ``plan.run_checks`` would come back unavailable. Passing only the
    proposal and run number lets the pass open proc for the frames and raw for
    the checks.

    :raises IncompleteRun: some frame did not reach ``OK``. Deliberately not
        caught: a partial I(q) that looks like a whole one is worse than a
        failed variable (context file §10, P5).
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


def _masked(grid: Any) -> np.ma.MaskedArray:
    """The intensity grid with un-integrated slots masked out.

    Slots that hold no frame are stored as zeros (context file §3 rule 7), so
    a plain ``mean`` would average them in and pull every curve down. The mask
    is ``n_frames``, which is the companion array those zeros are read with.
    """
    intensity = np.asarray(grid["intensity"].values)
    present = np.asarray(grid["n_frames"].values, dtype=bool)
    return np.ma.masked_array(
        intensity, mask=~np.broadcast_to(present[..., None], intensity.shape)
    )


def overview_figure(grid: Any, title: str = "") -> Any:
    """Three-panel I(q) overview: mean curve, per-pulse map, per-train map.

    :param grid: the Dataset from :func:`agipd_saxs` or
        :func:`per_pulse_from_file`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec

    masked = _masked(grid)
    q = np.asarray(grid["q"].values)
    n_trains, n_pulses, _ = masked.shape

    fig = plt.figure(figsize=(9, 7))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.plot(q, masked.mean(axis=(0, 1)))
    ax1.set_yscale("log")
    ax1.set_xlabel("q [nm$^{-1}$]")
    ax1.set_ylabel("I(q) [photons / solid angle]")
    ax1.set_title(title or "Mean I(q) over all trains and pulses")
    ax1.grid(alpha=0.3)

    for ax, data, ylabel, n in (
        (ax2, masked.mean(axis=0), "Pulse", n_pulses),
        (ax3, masked.mean(axis=1), "Train", n_trains),
    ):
        # LogNorm cannot take a non-positive vmin, and a run with nothing
        # integrated has no positive value at all to take one from.
        finite = np.ma.compressed(data)
        finite = finite[finite > 0]
        norm = LogNorm(vmin=finite.min(), vmax=finite.max()) if finite.size else None
        ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            norm=norm,
            extent=[float(q.min()), float(q.max()), n, 0],
        )
        ax.set_xlabel("q [nm$^{-1}$]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"I(q) per {ylabel.lower()}")

    fig.tight_layout()
    return fig
