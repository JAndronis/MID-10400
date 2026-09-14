"""JUNGFRAU WAXS loader and integrator — the ``jungfrau_waxs`` DAMNIT variables.

See ``context/jungfrau-waxs-integrator.md`` for the design, which is written as
a delta against ``context/agipd-saxs-integrator.md``. Two of its decisions were
amended by measurement during implementation and the modules that carry them
say so: the per-frame mask is a NaN rather than an ``integrate1d(mask=)``
argument (:mod:`analysis.waxs.operator`), and the variance is stored unclamped
(:mod:`analysis.waxs.integrate`). :mod:`analysis.waxs.combine` puts the two
detectors on one scale by fitting a factor over their overlap.
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
    "SelfTestFailed",
    "SelfTestReport",
    "UnexpectedLitCells",
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
