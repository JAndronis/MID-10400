"""Diagnostics of a run's stored products.

Plain functions over the pass output files and DAMNIT's per-run file. They read
small, final files only (the frame tables in scratch and the DAMNIT variables), so
they keep working after raw and proc go to tape. They never alter a stored number:
every normalisation is applied to an in-memory copy and named in the result's
``attrs``.
"""

from __future__ import annotations

from analysis.diagnostics.curves import (
    band_mean,
    cell_xgm_correlation,
    chi2_vs_median,
    detrend,
    flux_from,
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
    "flux_from",
    "per_train_flux",
    "svd",
    "units",
    "xgm_scatter",
]
