"""Diagnostics of a run's stored products (context/diagnostics.md).

Plain functions over the pass output files and DAMNIT's per-run file. They read
small, final files only — the frame tables in scratch and the DAMNIT variables —
so they keep working after raw and proc go to tape, and they never alter a stored
number: every normalisation is applied to an in-memory copy and named in the
result's ``attrs``.

``frames``  what a frame table holds: status per code (and per cell), per-frame
            scalars on a (train, label) grid, the stored units
``curves``  per-train flux from DAMNIT, and the checks on per-train curves: SVD,
            χ² against the run median, scatter before and after the flux, and the
            JUNGFRAU cell–XGM correlation
"""

from __future__ import annotations

from analysis.diagnostics.curves import (
    band_mean,
    cell_xgm_correlation,
    chi2_vs_median,
    detrend,
    per_train_flux,
    svd,
    xgm_scatter,
)
from analysis.diagnostics.frames import (
    default_scalar_column,
    detector_kind,
    frame_accounting,
    frame_scalar,
    per_train,
    units,
)

__all__ = [
    "band_mean",
    "cell_xgm_correlation",
    "chi2_vs_median",
    "default_scalar_column",
    "detrend",
    "detector_kind",
    "frame_accounting",
    "frame_scalar",
    "per_train",
    "per_train_flux",
    "svd",
    "units",
    "xgm_scatter",
]
