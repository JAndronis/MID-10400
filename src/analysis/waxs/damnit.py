"""The DAMNIT surface for the JUNGFRAU WAXS pass.

Thin on purpose: DAMNIT ``exec``s its variable file into a dict, so a function
defined there cannot be pickled to the workers this pass spawns. The context
file holds only the decorated wrappers; everything they call lives here.

One variable per detector, because the pass is per detector, plus an
overview that draws both.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common.writer import pooled_per_train
from analysis.waxs.combine import (
    NoOverlap,
    combine_curves,
    mean_curve,
    require_same_units,
    scale_to_overlap,
    units_of,
)
from analysis.waxs.config import DETECTORS, JungfrauWaxsConfig, config_for
from analysis.waxs.writer import per_cell

__all__ = [
    "DAMNIT_OVERRIDES",
    "combined_curve",
    "config_for",
    "damnit_config",
    "jungfrau_waxs",
    "jungfrau_waxs_from_file",
    "overview_figure",
    "per_cell_from_file",
    "pooled_from_file",
    "preprocess",
]

log = logging.getLogger(__name__)

#: Where the DAMNIT variables depart from the config defaults. Kept here rather
#: than in the context file, so that a script judging DAMNIT's files
#: (``scripts/reprocess_safe.py``) builds the config DAMNIT builds.
#:
#: ``npt`` 200: 836 of the 839 files under ``scratch/jungfrau_waxs`` were written
#: at 200 on 2026-09-14, by a live context that passed it; kept on 2026-10-13
#: (open task 17d). The config default stays 500, which W4 was run at.
DAMNIT_OVERRIDES: dict[str, Any] = {"npt": 200}


def damnit_config(
    proposal: int, run_no: int, detector: str, **overrides: Any
) -> JungfrauWaxsConfig:
    """The config the DAMNIT variables run with.

    :param overrides: any config field, taking precedence over
        :data:`DAMNIT_OVERRIDES`.
    """
    return config_for(proposal, run_no, detector, **(DAMNIT_OVERRIDES | overrides))


def jungfrau_waxs(proposal: int, run_no: int, detector: str, **overrides: Any) -> Any:
    """Integrate one detector's run and return the ``(trainId, cellId, q)`` grid.

    A partial result is never returned: ``IncompleteRun`` propagates. The value
    check is the exception — it drops the offending pixel and keeps the frame,
    so an affected frame's ring bin reads low and a reader should filter on
    ``frames/n_extreme_pixels`` before treating that bin quantitatively.

    :returns: the grid, or ``None`` for a run with no lit memory cell — no beam
        is a result, not a failure, and the evidence is logged at WARNING.
    :raises IncompleteRun: a frame did not reach ``OK``. A frame with *every*
        pixel excluded is tolerated by default; pass
        ``allow_data_check_failures=False`` to refuse instead.
    """
    from analysis.waxs.run import run_jungfrau_waxs

    cfg = damnit_config(proposal, run_no, detector, **overrides)
    log.info(
        "jungfrau_waxs on p%d r%d %s -> %s", proposal, run_no, detector, cfg.output_file
    )
    return run_jungfrau_waxs(cfg, reduce="per_cell")


def preprocess(
    proposal: int,
    run_no: int,
    detectors: tuple[str, ...] = DETECTORS,
    **overrides: Any,
) -> Any:
    """The ``jungfrau_preprocessing`` cell: each detector's pass, one after the other.

    A detector whose file is complete is answered from it without writing
    (CLAUDE.md pitfall 26). A run with no lit cell writes no file and reads
    "dark". A failure on one detector does not stop the other; it turns the
    cell red, and the reader of that detector says what is wrong.

    :returns: a ``Cell``: per detector its outcome (JSON) and a one-line summary.
    """
    import json

    import h5py
    from damnit.context import Cell

    from analysis.common.writer import status_counts
    from analysis.diagnostics.damnit import COLORS
    from analysis.waxs.run import run_jungfrau_waxs

    outcome: dict[str, Any] = {}
    tolerated: dict[str, set[str]] = {}
    for detector in detectors:
        cfg = damnit_config(proposal, run_no, detector, **overrides)
        tolerated[detector] = _tolerated(cfg)
        try:
            run_jungfrau_waxs(cfg, reduce="none")
        except Exception as error:  # noqa: BLE001 - shown in the table, logged
            log.exception("jungfrau_preprocessing r%d %s failed", run_no, detector)
            outcome[detector] = {"error": f"{type(error).__name__}: {error}"}
            continue
        if not cfg.output_file.exists():
            outcome[detector] = "dark"
            continue
        with h5py.File(cfg.output_file, "r") as handle:
            outcome[detector] = status_counts(
                handle["frames/status"][:], nonzero_only=True
            )

    def line(name: str, value: Any) -> str:
        if value == "dark":
            return f"{name.upper()} dark"
        if "error" in value:
            return f"{name.upper()} failed"
        if set(value) <= {"OK"}:
            return f"{name.upper()} {value.get('OK', 0)} OK"
        return f"{name.upper()} {value}"

    good = all(
        v == "dark" or ("error" not in v and set(v) <= tolerated[k])
        for k, v in outcome.items()
    )
    return Cell(
        json.dumps(outcome),
        summary_value=" · ".join(line(k, v) for k, v in outcome.items()),
        background=COLORS["good"] if good else COLORS["error"],
    )


def jungfrau_waxs_from_file(
    proposal: int, run_no: int, detector: str, **overrides: Any
) -> Any:
    """A ``jungfrau_waxs_jf*`` column: the ``(trainId, cellId, q)`` grid, from the file.

    Written by ``jungfrau_preprocessing``; this only reads it.

    :raises Skip: no file — a dark run, or a failed pass (the preprocessing
        cell says which).
    :raises IncompleteRun: some frame is not ``OK``, other than the tolerated
        ``DATA_CHECK_FAILED`` (a frame with every pixel excluded), as the pass.
    """
    import h5py
    from damnit.context import Skip

    from analysis.common.writer import IncompleteRun, status_counts

    cfg = damnit_config(proposal, run_no, detector, **overrides)
    path = cfg.output_file
    if not path.exists():
        raise Skip(f"no {detector} file at {path}: see jungfrau_preprocessing")
    with h5py.File(path, "r") as handle:
        summary = status_counts(handle["frames/status"][:], nonzero_only=True)
    if set(summary) - _tolerated(cfg):
        raise IncompleteRun(f"not every frame reached OK: {summary}")
    return per_cell(path)


def _tolerated(cfg: JungfrauWaxsConfig) -> set[str]:
    """Status names a finished run may hold, as the pass's own gate allows."""
    tolerated = {"OK"}
    if cfg.allow_data_check_failures:
        tolerated.add("DATA_CHECK_FAILED")
    return tolerated


def per_cell_from_file(path: str | Path) -> Any:
    """Re-read a finished run's ``(trainId, cellId, q)`` grid."""
    return per_cell(Path(path))


def pooled_from_file(path: str | Path) -> Any:
    """Re-read a finished run's per-train pooled ``I(q)`` and ``σ(q)``."""
    return pooled_per_train(Path(path))


def combined_curve(jf1: Any, jf2: Any, *, method: str = "wls") -> Any:
    """One I(q) from both detectors, jf2 scaled onto jf1 over their overlap.

    Returns a DataArray rather than a Dataset so DAMNIT renders it in the table
    as its dtype and shape; ``sigma`` and ``source`` ride along as non-dimension
    coordinates, and the fitted scaling is in ``attrs``.

    ``source`` is 0 where only jf1 contributes, 1 where only the scaled jf2
    does, and 2 in the overlap, so a reader can always tell which detector a
    point came from.

    Returns ``None`` when either detector produced nothing — a run with no beam
    gives no I(q) on either, and there is nothing to combine. The per-detector
    variables say the same thing on their own, so this stays quiet rather than
    raising and taking a reprocess down with it.

    :raises NoOverlap: the two populate no shared q range. That is a geometry
        problem, not a plotting one, so it is not swallowed.
    :raises analysis.waxs.combine.UnitsMismatch: one grid counts keV and the
        other photons.
    """
    import xarray as xr

    missing = [
        name for name, grid in zip(DETECTORS, (jf1, jf2), strict=True) if grid is None
    ]
    if missing:
        log.info(
            "no combined curve: %s produced no frames to combine",
            " and ".join(missing),
        )
        return None

    units = require_same_units(units_of(jf1), units_of(jf2))
    q_ref, i_ref, s_ref = mean_curve(jf1)
    q_other, i_other, s_other = mean_curve(jf2)
    combined, scaling = combine_curves(
        q_ref,
        i_ref,
        q_other,
        i_other,
        sigma_ref=s_ref,
        sigma_other=s_other,
        method=method,
        units=units,
    )
    result = xr.DataArray(
        combined["intensity"].values,
        dims=("q",),
        coords={
            "q": combined["q"].values,
            "sigma": (("q",), combined["sigma"].values),
            "source": (("q",), combined["source"].values),
        },
        name="intensity",
    )
    result.attrs.update(combined.attrs)
    result.attrs["reference"] = "jf1"
    return result


def overview_figure(
    jf1: Any, jf2: Any, title: str = "", labels: tuple[str, str] = DETECTORS
) -> Any:
    """Both detectors as two traces, with their overlap visible.

    Deliberately **not** a concatenation. After masking the two populate
    11.5-23.7 and 9.8-18.5 nm⁻¹, so they overlap over roughly 11.5-18.5 — wide
    enough to be a genuine cross-check on the two PONIs, and the place any
    normalisation mismatch between the detectors would show. Pooling them onto a
    common grid needs both ``N`` arrays on the same absolute scale, which is
    untested and involves different solid-angle coverage and different masks; do
    that only once the two traces are seen to agree.

    :param jf1: the DataArray from :func:`jungfrau_waxs` for jf1, or ``None``.
    :param jf2: the same for jf2. Either may be missing, so the overview still
        draws when only one detector has been processed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    grids = [
        (label, g) for label, g in zip(labels, (jf1, jf2), strict=True) if g is not None
    ]
    # Two traces on one axis, one scaled onto the other: only in one unit.
    units = [units_of(grid) for _, grid in grids]
    if len(units) == 2:
        require_same_units(*units)

    fig = plt.figure(figsize=(9, 7))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    maps = [fig.add_subplot(gs[1, index]) for index in range(2)]

    curves = []
    for label, grid in grids:
        try:
            q, curve, sigma = mean_curve(grid)
        except ValueError:
            # A run where nothing reached OK. The map panel says so; the curve
            # panel simply has one fewer trace.
            continue
        curves.append((label, q, curve, sigma))

    # The second detector is drawn on the first's scale, fitted over the
    # overlap, so the two traces are directly comparable. The reference keeps
    # its own values; only the other moves, and by how much is on the figure.
    scaling = None
    if len(curves) == 2:
        (_, q_ref, i_ref, s_ref), (_, q_other, i_other, s_other) = curves
        try:
            scaling = scale_to_overlap(
                q_ref,
                i_ref,
                q_other,
                i_other,
                sigma_ref=s_ref,
                sigma_other=s_other,
            )
        except NoOverlap:
            scaling = None

    for index, (label, q, curve, _) in enumerate(curves):
        scaled = scaling is not None and index == 1
        factor = scaling.factor if scaled else 1.0
        # Always label the factor when one was applied, including 1: a reader
        # must be able to tell a scaled trace from an unscaled one without
        # inferring it from the value.
        suffix = f" x {factor:.4g}" if scaled else ""
        ax1.plot(q, curve * factor, label=f"{label}{suffix}", lw=1.2)

    if scaling is not None:
        ax1.axvspan(scaling.q_low, scaling.q_high, color="0.85", zorder=0)
        note = (
            f"overlap {scaling.q_low:.1f}–{scaling.q_high:.1f} nm$^{{-1}}$, "
            f"{scaling.n_bins} bins; scale {scaling.factor:.4g}"
        )
        if np.isfinite(scaling.reduced_chi2):
            note += f", $\\chi^2_\\nu$ {scaling.reduced_chi2:.2f}"
        ax1.annotate(
            note,
            xy=((scaling.q_low + scaling.q_high) / 2, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="0.3",
        )

    ax1.set_yscale("log")
    ax1.set_xlabel("q [nm$^{-1}$]")
    ax1.set_ylabel(f"I(q) [{units[0] if units else 'photons'} / solid angle]")
    ax1.set_title(title or "Mean I(q), both JUNGFRAUs (jf2 scaled onto jf1)")
    ax1.grid(alpha=0.3)
    if grids:
        ax1.legend(fontsize=8)
    for ax, label in zip(maps, labels, strict=True):
        grid = dict(grids).get(label)
        if grid is None:
            ax.text(
                0.5,
                0.5,
                f"{label}: not processed",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        present = grid["n_frames"] > 0
        intensity = grid if bool(present.all()) else grid.where(present)
        values = np.asarray(intensity.mean("cellId").values)
        positive = np.isfinite(values) & (values > 0)
        if positive.any():
            from matplotlib.colors import LogNorm

            q = np.asarray(grid["q"].values)
            trains = np.asarray(grid["trainId"].values)
            image = ax.imshow(
                values,
                aspect="auto",
                origin="lower",
                norm=LogNorm(vmin=values[positive].min(), vmax=values[positive].max()),
                extent=(q[0], q[-1], 0, max(trains.size - 1, 1)),
            )
            fig.colorbar(image, ax=ax)
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
        ax.set_ylabel("train index")
        ax.set_title(f"{label}: I(q) per train (averaged over cells)")

    fig.tight_layout()
    return fig
