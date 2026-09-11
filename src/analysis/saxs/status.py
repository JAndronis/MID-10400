"""Frame status codes and the data-check exception.

The codes are the ledger values written to ``/frames/status`` (context file
§8). They are shared by the sparse kernels, the worker, the writer and the
orchestration, so they live in their own module rather than in ``config``,
which holds only the frozen config dataclass.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["DataCheckFailed", "FrameStatus"]


class FrameStatus(IntEnum):
    """Per-frame ledger codes (context file §8)."""

    OK = 0
    MISSING_MODULES = 1
    NO_FRAMES = 2
    LABEL_MISMATCH = 3
    DATA_CHECK_FAILED = 4
    WORKER_ERROR = 5
    NOT_PROCESSED = 255


class DataCheckFailed(ValueError):
    """``image.data`` is not integer photon counts, or carries negative counts.

    Raised rather than returned: there is no dense fallback and no NaN
    sentinel, so a frame that fails the check must not hand back arrays a
    caller could mistake for an integration. Callers that need the §8 code for
    the ledger use :func:`analysis.saxs.sparse.frame_data_status` instead.
    """
