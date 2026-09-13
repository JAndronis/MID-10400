"""Detector-agnostic pieces shared by the AGIPD SAXS and JUNGFRAU WAXS passes.

Nothing here knows which detector it is serving. Anything that does — the
geometry, the integration kernel, the mask sources, how many rows a train owns
— belongs to ``analysis.saxs`` or ``analysis.waxs``.
"""

from __future__ import annotations

from analysis.common.cpu import (
    CPU_TOPOLOGY_ROOT,
    THREAD_ENV,
    default_pool,
    file_sha256,
    package_versions,
    phase,
    physical_cores,
    set_thread_env,
)
from analysis.common.masks import (
    MaskSource,
    StaticMask,
    UnexpectedMaskBits,
    bits_to_mask,
    describe_bits,
    frame_bad,
)
from analysis.common.plan import (
    Block,
    RunPlan,
    TrainRecord,
    build_blocks,
    evenly_spaced,
)
from analysis.common.status import DataCheckFailed, FrameStatus
from analysis.common.writer import (
    ConfigHashMismatch,
    FrameTableWriter,
    IncompleteRun,
    PassConfig,
    as_handle,
    config_payload,
    pooled_per_train,
    q_centers,
)

__all__ = [
    "CPU_TOPOLOGY_ROOT",
    "THREAD_ENV",
    "Block",
    "ConfigHashMismatch",
    "DataCheckFailed",
    "FrameStatus",
    "FrameTableWriter",
    "IncompleteRun",
    "MaskSource",
    "PassConfig",
    "RunPlan",
    "StaticMask",
    "TrainRecord",
    "UnexpectedMaskBits",
    "as_handle",
    "bits_to_mask",
    "build_blocks",
    "config_payload",
    "default_pool",
    "describe_bits",
    "evenly_spaced",
    "file_sha256",
    "frame_bad",
    "package_versions",
    "phase",
    "physical_cores",
    "pooled_per_train",
    "q_centers",
    "set_thread_env",
]
