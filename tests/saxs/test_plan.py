"""Run plan over a mock run."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from analysis.common.status import FrameStatus
from analysis.saxs.plan import build_plan

pytest.importorskip("extra_data")

from mockrun import AWKWARD  # noqa: E402


def test_plan_over_a_clean_run(run_cfg, mock_run_factory):
    run, dc = mock_run_factory()
    plan = build_plan(run_cfg, dc=dc)

    assert [t.train_id for t in plan.trains] == list(run.train_ids)
    assert all(t.status is FrameStatus.OK for t in plan.trains)
    assert plan.n_frames == run.n_frames
    assert [t.first_row for t in plan.trains] == [
        run.first_row(t) for t in run.train_ids
    ]
    assert all(len(b.train_ids) <= run_cfg.trains_per_block for b in plan.blocks)
    assert [b.index for b in plan.blocks] == list(range(len(plan.blocks)))


def test_plan_over_a_dropped_a_short_and_a_zero_frame_train(
    run_cfg, mock_run_factory, status_of
):
    """10002 was never written, 10004 is short of modules, 10006 has no frames.

    The dropped train is absent from the plan; the other two are present with
    distinct statuses and own no rows, so the rows stay contiguous.
    """
    run, dc = mock_run_factory(**AWKWARD)
    plan = build_plan(run_cfg, dc=dc)

    assert [t.train_id for t in plan.trains] == list(run.train_ids)
    assert status_of(plan, 10004) is FrameStatus.MISSING_MODULES
    assert status_of(plan, 10006) is FrameStatus.NO_FRAMES
    assert plan.status_counts() == {"OK": 5, "MISSING_MODULES": 1, "NO_FRAMES": 1}
    assert plan.n_frames == run.n_frames
    for train in run.detector_trains:
        assert plan.record(train).first_row == run.first_row(train)
    for train in (10004, 10006):
        assert plan.record(train).n_frames == 0


def test_blocks_never_straddle_a_non_ok_train(run_cfg, mock_run_factory):
    _, dc = mock_run_factory(short_module_trains=(10002,))
    plan = build_plan(replace(run_cfg, trains_per_block=4), dc=dc)

    for block in plan.blocks:
        assert 10002 not in block.train_ids
    # the bad train splits the run into two groups rather than one block of four
    assert len(plan.blocks) == 2
    assert plan.blocks[0].train_ids == (10000, 10001)
    assert plan.blocks[1].train_ids == (10003, 10004, 10005)


def test_block_rows_are_contiguous_and_match_the_plan(run_cfg, mock_run_factory):
    _, dc = mock_run_factory()
    plan = build_plan(run_cfg, dc=dc)
    seen: list[int] = []
    for block in plan.blocks:
        rows = block.rows()
        assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))
        seen.extend(rows.tolist())
    assert seen == list(range(plan.n_frames))


def test_run_checks_are_recorded_even_when_unavailable(run_cfg, mock_pipeline):
    """A missing source makes a check unavailable, never the run fail.

    Given a collection, the checks are attempted against it (the mock has no
    control sources); given none, proc holds no control data to try.
    """
    from analysis.saxs.plan import run_checks

    names = {"xray_pulses", "quadrant_motors", "xgm_photon_energy"}
    tried = mock_pipeline.plan.checks
    assert set(tried) == names
    assert all(str(v).startswith("unavailable") for v in tried.values())
    assert not any(
        str(v).startswith("unavailable: no control data") for v in tried.values()
    )

    untried = run_checks(None, mock_pipeline.detector, mock_pipeline.plan.trains)
    assert set(untried) == names
    assert all(
        str(v).startswith("unavailable: no control data") for v in untried.values()
    )
