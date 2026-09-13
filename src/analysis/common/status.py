"""Frame status codes and the data-check exception.

Detector-independent: the codes are the ledger values written to
``/frames/status`` by every pass (AGIPD SAXS context file §8, JUNGFRAU WAXS
context file §4). They live here rather than in a per-detector ``config`` so
that the kernels, the workers, the writers and the orchestration of both passes
share one enumeration.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["DataCheckFailed", "FrameStatus"]


class FrameStatus(IntEnum):
    """Per-frame ledger codes."""

    OK = 0
    MISSING_MODULES = 1
    NO_FRAMES = 2
    LABEL_MISMATCH = 3
    DATA_CHECK_FAILED = 4
    WORKER_ERROR = 5
    NOT_PROCESSED = 255


class DataCheckFailed(ValueError):
    """A frame's values are not what the pass is entitled to assume.

    What that means is the detector's business: AGIPD requires integer photon
    counts with no negatives, JUNGFRAU requires finite keV values inside a
    configured range (WAXS context file §3 D6). Both raise this rather than
    returning, because neither pass has a fallback path or a NaN sentinel, so a
    frame that fails the check must not hand back arrays a caller could mistake
    for an integration.
    """
