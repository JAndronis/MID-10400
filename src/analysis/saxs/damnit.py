"""The DAMNIT-facing surface of ``agipd_saxs``.

Everything the DAMNIT variable needs lives here rather than in
``src/amore/context.py``. DAMNIT ``exec``s that file into a dict, so a function defined
there cannot be pickled to the spawned workers this pass runs.

The context-file variable is therefore two lines: call :func:`agipd_saxs` and
hand DAMNIT the result. Nothing here imports DAMNIT, so the same functions
work from a notebook or from a Slurm script.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common.writer import pooled_per_train
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.writer import per_pulse

__all__ = [
    "config_for",
    "agipd_saxs",
    "agipd_saxs_from_file",
    "overview_figure",
    "per_pulse_from_file",
    "pooled_from_file",
    "preprocess",
    "product_status",
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


def preprocess(proposal: int, run_no: int, **overrides: Any) -> Any:
    """The ``agipd_preprocessing`` cell: the run's one read of AGIPD proc.

    Writes whichever of the frame table, the window sums and the photon
    statistics are not complete yet, from one read (``run_agipd_saxs``), and
    answers a complete run from its files without writing (CLAUDE.md pitfall 26).
    Every column that shows one of these products reads it from its file.

    Failures are a red cell rather than an exception, so the readers still run
    and each says what is wrong with its own file.

    :returns: a ``Cell``: per product, its status counts (JSON) as data, and a
        one-line summary.
    """
    from damnit.context import Cell

    from analysis.common.lock import OutputBusy
    from analysis.diagnostics.damnit import COLORS
    from analysis.saxs.run import run_agipd_saxs

    cfg = config_for(proposal, run_no, **overrides)
    log.info("agipd_preprocessing on p%d r%d -> %s", proposal, run_no, cfg.output_file)
    try:
        run_agipd_saxs(cfg, reduce="none")
    except OutputBusy as error:
        return Cell(str(error), summary_value="busy", background=COLORS["warning"])
    except Exception as error:  # noqa: BLE001 - shown in the table, logged in full
        log.exception("agipd_preprocessing r%d failed", run_no)
        status = product_status(cfg)
        return Cell(
            json.dumps(status | {"error": f"{type(error).__name__}: {error}"}),
            summary_value=f"failed: {type(error).__name__}",
            background=COLORS["error"],
        )
    status = product_status(cfg)
    return Cell(
        json.dumps(status),
        summary_value=_status_line(status),
        background=COLORS["good"] if _all_ok(status) else COLORS["error"],
    )


def product_status(cfg: AgipdSaxsConfig) -> dict[str, Any]:
    """Status counts of each product the pass writes for ``cfg``, or ``None``."""
    import h5py

    from analysis.common.writer import status_counts

    def counts(path: Path, column: str) -> dict[str, int] | None:
        if not path.exists():
            return None
        try:
            with h5py.File(path, "r") as handle:
                return status_counts(handle[column][:], nonzero_only=True)
        except (OSError, KeyError):
            return {"unreadable": 1}

    status = {"frames": counts(cfg.output_file, "frames/status")}
    if cfg.pixel_sum_trains is not None:
        status["window_sums"] = counts(cfg.pixel_sums_file, "trains/status")
    if cfg.photon_stats:
        status["photon_stats"] = counts(cfg.photon_stats_file, "frames/status")
    return status


def _all_ok(status: dict[str, Any]) -> bool:
    return all(c is not None and set(c) <= {"OK"} for c in status.values())


def _status_line(status: dict[str, Any]) -> str:
    names = {"frames": "frames", "window_sums": "sums", "photon_stats": "stats"}
    parts = []
    for key, counts in status.items():
        if counts is None:
            parts.append(f"{names[key]} missing")
        elif set(counts) <= {"OK"}:
            parts.append(f"{names[key]} {counts.get('OK', 0)} OK")
        else:
            bad = {k: v for k, v in counts.items() if k != "OK"}
            parts.append(f"{names[key]} {bad}")
    return " · ".join(parts)


def agipd_saxs_from_file(proposal: int, run_no: int, **overrides: Any) -> Any:
    """The ``agipd_saxs`` column: ``I(q)`` per (trainId, pulseId), from the file.

    Written by ``agipd_preprocessing``; this only reads it.

    :raises Skip: the run has no frame table.
    :raises IncompleteRun: some frame is not ``OK``, as the pass itself would.
    """
    import h5py
    from damnit.context import Skip

    from analysis.common.writer import IncompleteRun, status_counts

    path = config_for(proposal, run_no, **overrides).output_file
    if not path.exists():
        raise Skip(f"no frame table at {path}: see agipd_preprocessing")
    with h5py.File(path, "r") as handle:
        summary = status_counts(handle["frames/status"][:], nonzero_only=True)
    if set(summary) - {"OK"}:
        raise IncompleteRun(f"not every frame reached OK: {summary}")
    return per_pulse(path)


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
