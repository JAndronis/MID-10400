"""Frame status codes and the data-check exception.

These moved to :mod:`analysis.common.status` when the JUNGFRAU WAXS pass
started sharing them; the names stay importable from here because the whole
SAXS package and its tests refer to them by this path.
"""

from __future__ import annotations

from analysis.common.status import DataCheckFailed, FrameStatus

__all__ = ["DataCheckFailed", "FrameStatus"]
