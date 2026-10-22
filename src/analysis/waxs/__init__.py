"""JUNGFRAU WAXS loader and integrator behind the ``jungfrau_waxs`` DAMNIT variables.

The per-frame mask is carried as NaN in the data rather than passed as
``integrate1d(mask=)`` (:mod:`analysis.waxs.operator`). Frames are counted as
photons per ASIC before they are integrated, with a Poisson variance
(:mod:`analysis.waxs.photons`). :mod:`analysis.waxs.combine` puts the two
detectors on one scale by fitting a factor over their overlap, and refuses to
mix a keV file with a photon one.
"""

from __future__ import annotations

from analysis.common.masks import frame_bad
from analysis.waxs.cells import (
    CellAccumulator,
    CellClassification,
    UnexpectedLitCells,
)
from analysis.waxs.combine import (
    NoOverlap,
    OverlapScaling,
    UnitsMismatch,
    combine_curves,
    combine_files,
    mean_curve,
    scale_to_overlap,
)
from analysis.waxs.config import (
    DETECTORS,
    EXPECTED_BITS,
    EXPECTED_LIT_CELLS,
    METHOD,
    MODULE_SHAPE,
    NPIX,
    JungfrauWaxsConfig,
    config_for,
)
from analysis.waxs.integrate import (
    ErrorModel,
    FrameResult,
    extreme_pixels,
    integrate_frame,
)
from analysis.waxs.masks import build_static_bad, load_static_mask
from analysis.waxs.operator import (
    WavelengthMismatch,
    WaxsOperator,
    build_operator,
    operator_sha256,
)
from analysis.waxs.photons import PhotonCounting
from analysis.waxs.selftest import SelfTestFailed, SelfTestReport, run_selftest

__all__ = [
    "DETECTORS",
    "EXPECTED_BITS",
    "EXPECTED_LIT_CELLS",
    "METHOD",
    "MODULE_SHAPE",
    "NPIX",
    "CellAccumulator",
    "CellClassification",
    "ErrorModel",
    "FrameResult",
    "JungfrauWaxsConfig",
    "NoOverlap",
    "OverlapScaling",
    "PhotonCounting",
    "SelfTestFailed",
    "SelfTestReport",
    "UnexpectedLitCells",
    "UnitsMismatch",
    "WavelengthMismatch",
    "WaxsOperator",
    "build_operator",
    "build_static_bad",
    "combine_curves",
    "combine_files",
    "config_for",
    "frame_bad",
    "extreme_pixels",
    "integrate_frame",
    "load_static_mask",
    "mean_curve",
    "operator_sha256",
    "scale_to_overlap",
    "run_selftest",
]
