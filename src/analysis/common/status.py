"""Frame status codes and the data-check exception.

The codes are the ledger values written to ``/frames/status``. They are
detector-independent so that the kernels, workers, writers and orchestration of
both passes share one enumeration.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["UNFINISHED", "DataCheckFailed", "FrameStatus"]


class FrameStatus(IntEnum):
    """Per-frame ledger codes."""

    OK = 0
    MISSING_MODULES = 1
    NO_FRAMES = 2
    LABEL_MISMATCH = 3
    DATA_CHECK_FAILED = 4
    WORKER_ERROR = 5
    NOT_PROCESSED = 255


#: Codes a rerun retries. ``WORKER_ERROR`` says nothing about the data (a node,
#: a file system or a pool can fail and succeed the next time).
#: ``LABEL_MISMATCH`` and ``DATA_CHECK_FAILED`` are properties of the data and
#: would come out the same, so they count as done.
UNFINISHED: frozenset[FrameStatus] = frozenset(
    {FrameStatus.NOT_PROCESSED, FrameStatus.WORKER_ERROR}
)


class DataCheckFailed(ValueError):
    """A frame's values are not what the pass is entitled to assume.

    What that means depends on the detector. AGIPD requires integer photon
    counts with no negatives, JUNGFRAU at least one pixel left after masking.
    Both raise rather than return, so a failed frame never hands back arrays
    that look integrated.
    """
