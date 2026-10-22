"""DAMNIT cells of the hit recognition: a summary cell and a geometry figure.

Inputs are the ones the hit pass gets (``agipd_saxs``, ``xgm_intensity``,
``total_transmission`` as DAMNIT values in memory), so nothing here reads the
run's own DAMNIT file mid-job. This module classifies itself rather than
through ``analysis.xpcs.damnit.hit_set``, which keeps only the train ids.
``Cell`` is imported inside ``hit_summary_cell``, so importing this module does
not import DAMNIT.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.diagnostics.damnit import COLORS

__all__ = ["HEXAPOD", "classify_in_memory", "hit_geometry_figure", "hit_summary_cell"]

NO_HITS_COLOR = "#b4b4b4"
HEXAPOD = "MID_EXP_SAM/HEXAPOD/SSHEX"


def classify_in_memory(
    proposal: int, run_no: int, saxs: Any, xgm: Any, transmission: Any, **kwargs: Any
) -> dict:
    """``classify.hits`` over DAMNIT's in-memory values (kwargs go to ``hits``)."""
    from analysis.triage.classify import hits, signals_from

    return hits(
        run_no,
        proposal=proposal,
        signals=signals_from(saxs, xgm, transmission),
        **kwargs,
    )


def summary_text(found: dict) -> tuple[str, int, int]:
    counts = {
        k: int((found["classes"].values == k).sum()) for k in ("on", "edge", "off")
    }
    total = int(found["classes"].size)
    on, edge, off = counts["on"], counts["edge"], counts["off"]
    return f"{on} hits / {edge} edge / {off} off of {total}", on, total


def hit_summary_cell(
    proposal: int, run_no: int, saxs: Any, xgm: Any, transmission: Any, **kwargs: Any
) -> Any:
    """``'<n_hits> hits / <n_edge> edge / <n_off> off of N'``.

    Amber when the block is ambiguous, otherwise green with hits and grey
    without (no hit is a result, not a failure). The data is the classification
    summary as JSON.
    """
    import json

    from damnit.context import Cell

    found = classify_in_memory(proposal, run_no, saxs, xgm, transmission, **kwargs)
    text, hits_n, _ = summary_text(found)
    summary = found["summary"]
    if summary.get("ambiguous"):
        colour = COLORS["warning"]
        text += " (ambiguous)"
    elif hits_n:
        colour = COLORS["good"]
    else:
        colour = NO_HITS_COLOR
    record = {"kind": found["kind"], "block": found["block"],
              "summary": summary}  # fmt: skip
    return Cell(
        json.dumps(record, default=_jsonable), summary_value=text, background=colour
    )


def hit_geometry_figure(
    proposal: int,
    run_no: int,
    saxs: Any,
    xgm: Any,
    transmission: Any,
    run: Any = None,
    droplet_fit: Any = None,
    **kwargs: Any,
):
    """The figure of ``triage.geometry.hit_geometry_figure`` for one run.

    A mesh run gets the scan map when ``run`` (DAMNIT's raw run) has the
    hexapod, and the droplet outline when ``droplet_fit`` carries a ``radius``
    and its ``pixel_size_um``.
    """
    from analysis.triage.geometry import hit_geometry_figure as figure

    found = classify_in_memory(proposal, run_no, saxs, xgm, transmission, **kwargs)
    positions = hexapod_positions(run, found["classes"].trainId.values)
    return figure(found, positions, semi_axes(droplet_fit))


def hexapod_positions(run: Any, train_ids):
    """``(X, Z)`` [mm] per train id, or None when the run has no hexapod source."""
    if run is None or HEXAPOD not in run.all_sources:
        return None
    out = []
    for key in ("X.actualPosition.value", "Z.actualPosition.value"):
        values = run[HEXAPOD, key].xarray().reindex(trainId=train_ids)
        out.append(values.values.astype(float))
    return tuple(out)


def semi_axes(droplet_fit: Any):
    """Median ``radius`` (horizontal, vertical) in mm, or None."""
    try:
        px = float(np.asarray(droplet_fit.attrs["pixel_size_um"]).ravel()[0])
        radius = np.asarray(droplet_fit["radius"].values, float)
        return tuple(np.nanmedian(radius, axis=0) * px * 1e-3)
    except (AttributeError, KeyError, TypeError, IndexError):
        return None


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
