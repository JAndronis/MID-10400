"""AGIPD SAXS loader and integrator — the ``agipd_saxs`` DAMNIT variable.

Supersedes ``amore/analysis_helpers.integrate_run`` in place: the DAMNIT
variable keeps its name, and what it computes moves here.

See ``context/agipd-saxs-integrator.md`` for the design. P1 provides the
operator, the sparse kernels, the pure per-frame integration and the
sparse-vs-pyFAI gate; P2 adds the static and per-cell base masks;
P3 the plan, worker, writer and orchestration. DAMNIT integration is P5.
"""

from __future__ import annotations

from analysis.saxs.config import (
    EXPECTED_BITS,
    METHOD,
    NPIX,
    SHAPE,
    AgipdSaxsConfig,
)
from analysis.saxs.masks import (
    BaseMaskAccumulator,
    BaseMasks,
    MaskSource,
    StaticMask,
    UnexpectedMaskBits,
    build_static_bad,
    evenly_spaced,
    frame_bad,
    load_masks,
    load_pixel_mask,
    save_masks,
)
from analysis.saxs.operator import (
    SparseOperator,
    build_operator,
    geometry_from_config,
    load_operator,
    save_operator,
)
from analysis.saxs.plan import Block, RunPlan, TrainRecord, build_plan
from analysis.saxs.run import run_agipd_saxs
from analysis.saxs.selftest import SelfTestFailed, SelfTestReport, run_selftest
from analysis.saxs.sparse import (
    FrameResult,
    denominator,
    frame_data_status,
    gather,
    integrate_frame,
)
from analysis.saxs.status import DataCheckFailed, FrameStatus
from analysis.saxs.worker import BlockResult, WorkerPaths
from analysis.saxs.writer import AgipdSaxsWriter, ConfigHashMismatch, IncompleteRun

__all__ = [
    "EXPECTED_BITS",
    "METHOD",
    "NPIX",
    "SHAPE",
    "BaseMaskAccumulator",
    "BaseMasks",
    "Block",
    "BlockResult",
    "ConfigHashMismatch",
    "DataCheckFailed",
    "AgipdSaxsConfig",
    "FrameResult",
    "AgipdSaxsWriter",
    "FrameStatus",
    "IncompleteRun",
    "MaskSource",
    "RunPlan",
    "SelfTestFailed",
    "SelfTestReport",
    "SparseOperator",
    "StaticMask",
    "TrainRecord",
    "UnexpectedMaskBits",
    "WorkerPaths",
    "build_operator",
    "build_plan",
    "build_static_bad",
    "denominator",
    "evenly_spaced",
    "frame_bad",
    "frame_data_status",
    "gather",
    "geometry_from_config",
    "integrate_frame",
    "load_masks",
    "load_operator",
    "load_pixel_mask",
    "run_agipd_saxs",
    "run_selftest",
    "save_masks",
    "save_operator",
]
