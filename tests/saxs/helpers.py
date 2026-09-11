"""Small assertions shared by the P3 tests."""

from __future__ import annotations

from analysis.saxs.plan import RunPlan
from analysis.saxs.status import FrameStatus


def status_of(plan: RunPlan, train_id: int) -> FrameStatus:
    return plan.record(train_id).status
