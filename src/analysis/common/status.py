"""Frame status codes and the data-check exception.

The codes are the ledger values written to ``/frames/status``. They are
detector-independent so that the kernels, workers, writers and orchestration of
both passes share one enumeration.
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
    counts with no negatives, JUNGFRAU finite keV values inside a configured
    range. Both raise rather than return, because neither pass has a NaN
    sentinel and a failed frame must not hand back arrays that look integrated.
    """
