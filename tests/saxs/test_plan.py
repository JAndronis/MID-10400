"""Run plan over a mock run (context file §6.8; phase P3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from analysis.common.status import FrameStatus
from analysis.saxs.plan import build_plan

pytest.importorskip("extra_data")

from extra_data import RunDirectory  # noqa: E402


def test_plan_over_a_clean_run(run_cfg, mock_run_factory):
    run, dc = mock_run_factory()
    plan = build_plan(run_cfg, dc=dc)

    assert [t.train_id for t in plan.trains] == list(run.train_ids)
    assert all(t.status is FrameStatus.OK for t in plan.trains)
    assert plan.n_frames == run.n_frames
    assert [t.first_row for t in plan.trains] == [
        run.first_row(t) for t in run.train_ids
    ]


def test_dropped_train_leaves_a_gap_in_the_run(run_cfg, mock_run_factory):
    """A train the DAQ never wrote is simply absent from the run."""
    run, dc = mock_run_factory(train_ids=(10000, 10001, 10003, 10004))
    plan = build_plan(run_cfg, dc=dc)

    assert [t.train_id for t in plan.trains] == [10000, 10001, 10003, 10004]
    assert 10002 not in [t.train_id for t in plan.trains]
    # rows stay contiguous across the gap
    assert [t.first_row for t in plan.trains] == [0, 3, 6, 9]


def test_train_with_too_few_modules(run_cfg, mock_run_factory, status_of):
    run, dc = mock_run_factory(short_module_trains=(10002,))
    plan = build_plan(run_cfg, dc=dc)

    assert status_of(plan, 10002) is FrameStatus.MISSING_MODULES
    assert plan.record(10002).n_frames == 0
    # the short train owns no rows, so the next train starts where it would have
    assert plan.record(10002).first_row == plan.record(10003).first_row
    assert plan.n_frames == run.n_frames


def test_zero_frame_train(run_cfg, mock_run_factory, status_of):
    """No module wrote a frame — distinct from too few modules."""
    _, dc = mock_run_factory(zero_frame_trains=(10002,))
    plan = build_plan(run_cfg, dc=dc)
    assert status_of(plan, 10002) is FrameStatus.NO_FRAMES
    assert plan.record(10002).n_frames == 0


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


def test_blocks_are_capped_by_trains_per_block(run_cfg, mock_run_factory):
    _, dc = mock_run_factory()
    plan = build_plan(run_cfg, dc=dc)
    assert all(len(b.train_ids) <= run_cfg.trains_per_block for b in plan.blocks)
    assert [b.index for b in plan.blocks] == list(range(len(plan.blocks)))


def test_run_checks_are_recorded_even_when_unavailable(run_cfg, mock_run_factory):
    """A missing source makes a check unavailable, not the run fail."""
    _, dc = mock_run_factory()
    plan = build_plan(run_cfg, dc=dc)
    assert set(plan.checks) == {
        "xray_pulses",
        "quadrant_motors",
        "xgm_photon_energy",
    }
    # the mock run has no timeserver, XGM or motors
    assert all(
        isinstance(value, str) and value.startswith("unavailable")
        for value in plan.checks.values()
    )


def test_status_counts(run_cfg, mock_run_factory):
    _, dc = mock_run_factory(short_module_trains=(10002,), zero_frame_trains=(10004,))
    plan = build_plan(run_cfg, dc=dc)
    assert plan.status_counts() == {
        "OK": 4,
        "MISSING_MODULES": 1,
        "NO_FRAMES": 1,
    }


def test_plan_rejects_an_unknown_train(run_cfg, mock_run_factory):
    _, dc = mock_run_factory()
    plan = build_plan(run_cfg, dc=dc)
    with pytest.raises(KeyError):
        plan.record(99999)


def test_run_directory_is_readable_twice(run_cfg, mock_run_factory, tmp_path):
    """The mock run is a real directory; reopening it must give the same plan."""
    run, dc = mock_run_factory()
    first = build_plan(run_cfg, dc=dc)
    second = build_plan(run_cfg, dc=RunDirectory(str(run.path)))
    assert first.trains == second.trains


def test_run_checks_without_control_data_records_every_check(run_cfg, mock_pipeline):
    """Proc holds no control sources, so the checks need their own collection."""
    from analysis.saxs.plan import run_checks

    checks = run_checks(None, mock_pipeline.detector, mock_pipeline.plan.trains)
    assert set(checks) == {"xray_pulses", "quadrant_motors", "xgm_photon_energy"}
    assert all(
        str(v).startswith("unavailable: no control data") for v in checks.values()
    )


def test_build_plan_uses_the_given_collection_for_checks(run_cfg, mock_run_factory):
    """A caller-supplied collection is both the data and the control source."""
    from analysis.saxs.plan import build_plan

    _, dc = mock_run_factory()
    plan = build_plan(run_cfg, dc=dc)
    # The mock run has no control sources, but the checks were attempted
    # against dc rather than short-circuited as "no control data".
    assert set(plan.checks) == {"xray_pulses", "quadrant_motors", "xgm_photon_energy"}
    assert not any(
        str(v).startswith("unavailable: no control data") for v in plan.checks.values()
    )
