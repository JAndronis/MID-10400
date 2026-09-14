"""Small assertions shared by the P3 tests."""

from __future__ import annotations

from analysis.common.plan import RunPlan
from analysis.common.status import FrameStatus


def status_of(plan: RunPlan, train_id: int) -> FrameStatus:
    return plan.record(train_id).status
