"""The DAMNIT-facing surface of the hit-train XPCS pass.

Everything the ``xpcs_hits_*`` variables need lives here, not in
``src/amore/context.py``: DAMNIT ``exec``s that file into a dict, so a function
defined there cannot be pickled to the workers the pass spawns. A wrapper in the
context file is a docstring, one import and a call.

**This is the only module that imports** ``analysis.triage.classify`` (the
hit rule is another session's to change; the pass itself takes train ids).
``xpcs_hits_preprocessing`` is the second read of proc by design: the hit rule
needs the whole run's ``agipd_saxs``, which exists only after the first.

Nothing here imports DAMNIT at module level, so the functions run from a
notebook or a Slurm script. ``Cell`` and ``Skip`` are imported inside them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from analysis.diagnostics.damnit import COLORS, FAILED
from analysis.saxs.damnit import config_for as saxs_config_for
from analysis.xpcs import figures
from analysis.xpcs.config import XpcsConfig
from analysis.xpcs.hits import HitSet
from analysis.xpcs.result import Averaging, XpcsHits
from analysis.xpcs.run import run_xpcs_hits

__all__ = [
    "NO_HITS_COLOR",
    "beamcenter_figure",
    "config_for",
    "correction_figure",
    "coverage_cell",
    "frames_used",
    "g2_fit_figure",
    "g2_figure",
    "g2_from_file",
    "hit_set",
    "intensity_outliers_figure",
    "kbar_figure",
    "legacy_mean_dataset",
    "legacy_saxs",
    "legacy_ttcf",
    "outliers_figure",
    "overview_figure",
    "preprocess",
    "q_rings_figure",
    "result_file",
    "saxs_figure",
    "ttcf_from_file",
    "ttcfs_figure",
]

log = logging.getLogger(__name__)

#: A run with no droplet hit is a result, not a failure: neither green nor red.
NO_HITS_COLOR = "#b4b4b4"


def config_for(proposal: int, run_no: int, **overrides: Any) -> XpcsConfig:
    """The run's configuration: the SAXS pass's defaults and the XPCS ones.

    :param overrides: any :class:`XpcsConfig` field, e.g. ``output_root``.
    """
    return XpcsConfig(saxs=saxs_config_for(proposal, run_no), **overrides)


def result_file(proposal: int, run_no: int, **overrides: Any) -> Path:
    """The results file ``xpcs_hits_preprocessing`` writes for this run."""
    return config_for(proposal, run_no, **overrides).output_file


def hit_set(proposal: int, run_no: int, saxs: Any, xgm: Any, transmission: Any):
    """This run's droplet hits, from DAMNIT's values in memory.

    Inside the cluster job the run's ``agipd_saxs`` is not in the per-run file
    until the job ends, so the hits come from the dependencies, as in
    ``analysis.diagnostics.damnit.photon_statistics``.
    """
    from analysis.triage.classify import hits, signals_from

    found = hits(
        run_no, proposal=proposal, signals=signals_from(saxs, xgm, transmission)
    )
    source = f"analysis.triage.classify.hits kind={found['kind']}"
    if found["block"]:
        source += f" block={found['block']}"
    return HitSet.of(found["train_ids"], source=source)


def preprocess(
    proposal: int,
    run_no: int,
    saxs: Any,
    xgm: Any,
    transmission: Any,
    **overrides: Any,
) -> Any:
    """The ``xpcs_hits_preprocessing`` cell: the hit trains' one read of proc.

    Extracts the hit trains the photon store lacks and writes their sums; a
    complete run is answered from its files without writing (CLAUDE.md pitfall
    26). Every failure is a cell, so the readers still run and say what their
    file lacks.

    :returns: a ``Cell`` whose data is the coverage as JSON: green when every
        hit is usable, amber when some are final non-OK (no frames, a pulse
        pattern that differs), red when the pass failed or lost a hit, grey
        when the run has no hit.
    """
    from damnit.context import Cell

    from analysis.common.lock import OutputBusy

    try:
        hits = hit_set(proposal, run_no, saxs, xgm, transmission)
    except Exception as error:  # noqa: BLE001 - shown in the table, logged in full
        log.exception("xpcs_hits_preprocessing r%d: no hit set", run_no)
        return _failed(Cell, "hit rule", error)
    cfg = config_for(proposal, run_no, **overrides)
    log.info(
        "xpcs_hits_preprocessing on p%d r%d: %d hits -> %s",
        proposal,
        run_no,
        len(hits),
        cfg.output_file,
    )
    try:
        path = run_xpcs_hits(cfg, hits)
    except OutputBusy as error:
        return Cell(str(error), summary_value="busy", background=COLORS["warning"])
    except Exception as error:  # noqa: BLE001
        log.exception("xpcs_hits_preprocessing r%d failed", run_no)
        return _failed(Cell, "pass", error)
    return coverage_cell(path, hit_set_sha256=hits.sha256, source=hits.source)


def _failed(cell_class: Any, stage: str, error: Exception) -> Any:
    return cell_class(
        json.dumps({"error": f"{type(error).__name__}: {error}", "stage": stage}),
        summary_value=f"failed ({stage}): {type(error).__name__}",
        background=COLORS["error"],
    )


def coverage_cell(path: str | Path, **extra: Any) -> Any:
    """``used / hits`` of a results file, coloured by what became of each hit.

    - grey: no hit train;
    - red: a WORKER_ERROR or LABEL_MISMATCH, an unfinished train, or a file HDF5
      cannot read;
    - amber: any other non-OK status (NO_FRAMES, MISSING_MODULES,
      DATA_CHECK_FAILED);
    - green: every hit usable.

    The data is the coverage as JSON, with ``extra`` added to it.

    :raises Skip: no file, i.e. the pass has not written this run.
    """
    from damnit.context import Cell, Skip

    from analysis.common.status import UNFINISHED

    path = Path(path)
    if not path.exists():
        raise Skip(f"no XPCS results at {path}: see xpcs_hits_preprocessing")
    try:
        cov = XpcsHits(path).coverage()
    except OSError as error:
        return Cell(str(error), summary_value="unreadable", background=COLORS["error"])
    record = {
        "n_hits": cov.n_hits,
        "n_usable": cov.n_usable,
        "n_with_baseline": cov.n_with_baseline,
        "by_status": cov.by_status,
        "by_pairing": cov.by_pairing,
        "accounted": cov.accounted,
        "path": str(path),
        **extra,
    }
    if cov.n_hits == 0:
        return Cell(
            json.dumps(record), summary_value="no hits", background=NO_HITS_COLOR
        )
    pct = 100.0 * cov.n_usable / cov.n_hits
    summary = f"{cov.n_usable} / {cov.n_hits} hits ({pct:.1f}%)"
    summary += f" · {cov.n_with_baseline} with a baseline"
    lost = {
        name: n
        for name, n in cov.by_status.items()
        if name in {code.name for code in (*FAILED, *UNFINISHED)}
    }
    if lost or not cov.accounted:
        summary += f" · {lost}"
        colour = COLORS["error"]
    elif cov.n_usable < cov.n_hits:
        colour = COLORS["warning"]
    else:
        colour = COLORS["good"]
    return Cell(json.dumps(record), summary_value=summary, background=colour)


def frames_used(proposal: int, run_no: int, **overrides: Any) -> Any:
    """The ``xpcs_hits_frames_used`` cell: hits used, read from the results file.

    Like the other ``*_frames_used`` cells it has no ``var#`` on the pass, so
    asking for it cannot relaunch the cluster job (CLAUDE.md pitfall 26).
    """
    return coverage_cell(result_file(proposal, run_no, **overrides))


def _open(
    proposal: int, run_no: int, rep_rate_mhz: float | None, overrides: dict[str, Any]
) -> XpcsHits:
    """The run's results, refused unless every hit reached a final status.

    :raises Skip: no file, no hit, or no usable hit train.
    :raises IncompleteRun: a hit is still unfinished, as the pass itself would.
    """
    from damnit.context import Skip

    from analysis.common.status import UNFINISHED
    from analysis.common.writer import IncompleteRun

    path = result_file(proposal, run_no, **overrides)
    if not path.exists():
        raise Skip(f"no XPCS results at {path}: see xpcs_hits_preprocessing")
    hits = XpcsHits(path, rep_rate_mhz)
    if not hits.train_ids.size:
        raise Skip(f"r{run_no}: no droplet hit")
    unfinished = np.isin(hits.status, [int(code) for code in UNFINISHED])
    if unfinished.any():
        raise IncompleteRun(f"{int(unfinished.sum())} hit trains are unfinished")
    if not hits.usable.any():
        raise Skip(f"r{run_no}: no usable hit train ({hits.coverage()})")
    return hits


def g2_from_file(
    proposal: int, run_no: int, rep_rate_mhz: float | None = None, **overrides: Any
) -> Any:
    """The ``xpcs_hits_g2`` column: pooled g2, baseline and c2 with jackknife errors.

    :param rep_rate_mhz: DAMNIT's ``rep_rate``; a pulse pattern that implies
        another rate refuses to make a lag axis (``RepRateMismatch``).
    """
    hits = _open(proposal, run_no, rep_rate_mhz, overrides)
    out = hits.g2(Averaging.POOLED)
    out.attrs["coverage"] = str(hits.coverage())
    return out


def ttcf_from_file(proposal: int, run_no: int, **overrides: Any) -> Any:
    """The ``xpcs_hits_ttcf`` column: pooled two-time maps ``raw``, ``off``, ``c2``."""
    hits = _open(proposal, run_no, None, overrides)
    out = hits.ttcf(Averaging.POOLED)
    out.attrs["coverage"] = str(hits.coverage())
    return out


def overview_figure(g2: Any, ttcf: Any) -> Any:
    """c2(τ) per ring with its jackknife error, and the two-time map of one ring.

    The map is the ring with the most finite pixels in its baseline-subtracted
    TTCF. Rings with no photons are NaN and plot as gaps.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    tau_us = g2["tau"].values * 1e6
    for ring, roi in enumerate(g2["roi_q"].values):
        ax1.errorbar(
            tau_us,
            g2["c2"].values[ring],
            g2["sigma"].values[ring],
            fmt=".-",
            lw=0.7,
            ms=3,
            label=f"{roi:.4g}",
        )
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("τ [µs]")
    ax1.set_ylabel("c2 = g2 − baseline")
    ax1.legend(title="q [Å$^{-1}$]", fontsize=7)

    c2 = ttcf["c2"].values
    ring = int(np.argmax(np.isfinite(c2).sum(axis=(1, 2))))
    finite = c2[ring][np.isfinite(c2[ring])]
    image = ax2.imshow(
        c2[ring],
        origin="lower",
        vmin=0,
        vmax=float(np.percentile(finite, 99)) if finite.size else 1.0,
    )
    ax2.set_title(f"c2, q = {ttcf['roi_q'].values[ring]:.4g} Å$^{{-1}}$", fontsize=9)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("frame")
    fig.colorbar(image, ax=ax2)
    a = g2.attrs
    fig.suptitle(
        f"{a.get('n_trains', '?')} hit trains, {a.get('n_with_baseline', '?')} with a "
        "baseline",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# ── the old ``xpcs_*`` columns, from the hit store ───────────────────────────────
# One wrapper per old DAMNIT variable (``src/amore/context.py`` ``xpcs_*``), each
# a thin call into ``analysis.xpcs.figures``. They read the results file through
# ``_open``, so a run without a file or without a usable hit is a ``Skip`` and an
# unfinished one an ``IncompleteRun``, as for ``xpcs_hits_g2``.


def _detector_mean(
    proposal: int, run_no: int, hits: XpcsHits, max_trains: int | None, overrides: dict
) -> figures.DetectorMean:
    """The photon store's mean over some hit trains; ``Skip`` when it is gone."""
    from damnit.context import Skip

    cfg = config_for(proposal, run_no, **overrides)
    try:
        return figures.detector_mean(hits, cfg, max_trains=max_trains)
    except FileNotFoundError as error:
        raise Skip(f"{error}: the figure needs the photon store") from error


def q_rings_figure(
    proposal: int, run_no: int, max_trains: int | None = 200, **overrides: Any
) -> Any:
    """The old ``XPCS q-rings`` column: mean image and ring map (annulus only)."""
    hits = _open(proposal, run_no, None, overrides)
    det = _detector_mean(proposal, run_no, hits, max_trains, overrides)
    return figures.figure_q_rings(det)


def beamcenter_figure(
    proposal: int, run_no: int, max_trains: int | None = 200, **overrides: Any
) -> Any:
    """The old ``XPCS beamcenter`` column: the cone test at the configured centre."""
    hits = _open(proposal, run_no, None, overrides)
    det = _detector_mean(proposal, run_no, hits, max_trains, overrides)
    return figures.figure_beamcenter(det)


def outliers_figure(
    proposal: int, run_no: int, max_trains: int | None = 200, **overrides: Any
) -> Any:
    """The old ``XPCS outliers`` column: I(q) before and after, and the statistic."""
    hits = _open(proposal, run_no, None, overrides)
    det = _detector_mean(proposal, run_no, hits, max_trains, overrides)
    return figures.figure_outliers(det)


def intensity_outliers_figure(proposal: int, run_no: int, **overrides: Any) -> Any:
    """The old ``XPCS intensity outliers`` column: mean intensity per hit train."""
    return figures.figure_intensity_outliers(_open(proposal, run_no, None, overrides))


def kbar_figure(proposal: int, run_no: int, **overrides: Any) -> Any:
    """The old ``XPCS kbar`` column."""
    return figures.figure_kbar(_open(proposal, run_no, None, overrides))


def correction_figure(
    proposal: int, run_no: int, rep_rate_mhz: float | None = None, **overrides: Any
) -> Any:
    """The old ``XPCS correction`` column: raw, baseline and corrected, maps and g2."""
    return figures.figure_correction(_open(proposal, run_no, rep_rate_mhz, overrides))


def ttcfs_figure(proposal: int, run_no: int, **overrides: Any) -> Any:
    """The old ``XPCS TTCF's`` column: the corrected two-time map of every ring."""
    return figures.figure_ttcfs(_open(proposal, run_no, None, overrides))


def g2_figure(
    proposal: int, run_no: int, rep_rate_mhz: float | None = None, **overrides: Any
) -> Any:
    """The old ``XPCS g2`` column."""
    return figures.figure_g2(_open(proposal, run_no, rep_rate_mhz, overrides))


def g2_fit_figure(
    proposal: int, run_no: int, rep_rate_mhz: float | None = None, **overrides: Any
) -> Any:
    """The old ``XPCS g2 fit`` column; ``Skip`` when the fit fails, as before."""
    from damnit.context import Skip

    hits = _open(proposal, run_no, rep_rate_mhz, overrides)
    try:
        return figures.figure_g2_fit(hits)
    except ValueError as error:
        raise Skip(f"r{run_no}: {error}") from error


def saxs_figure(
    proposal: int, run_no: int, max_trains: int | None = 200, **overrides: Any
) -> Any:
    """The old ``XPCS SAXS overview`` column: q-binned from the store, else per ring."""
    from damnit.context import Skip

    hits = _open(proposal, run_no, None, overrides)
    try:
        det = _detector_mean(proposal, run_no, hits, max_trains, overrides)
    except Skip:
        det = None
    return figures.figure_saxs(hits, det)


def legacy_ttcf(proposal: int, run_no: int, **overrides: Any) -> Any:
    """The old ``XPCS TTCF`` column: ``TTCF_raw − TTCF_off`` of every hit train."""
    return figures.legacy_ttcf(_open(proposal, run_no, None, overrides))


def legacy_mean_dataset(
    proposal: int, run_no: int, rep_rate_mhz: float | None = None, **overrides: Any
) -> Any:
    """The old ``XPCS mean dataset`` column: ``ttcf_mean`` and ``g2``."""
    return figures.legacy_mean_dataset(_open(proposal, run_no, rep_rate_mhz, overrides))


def legacy_saxs(
    proposal: int, run_no: int, max_trains: int | None = 200, **overrides: Any
) -> Any:
    """The old ``XPCS SAXS`` column: I(q) per frame on the annulus, ``(frame, q)``."""
    hits = _open(proposal, run_no, None, overrides)
    return figures.saxs_curves(
        _detector_mean(proposal, run_no, hits, max_trains, overrides)
    )
