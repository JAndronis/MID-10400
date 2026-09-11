"""The P4 acceptance gates, exercised on the mock run (context file §10).

``scripts/p4_acceptance.py`` runs once, on a node, against a run that takes
minutes to integrate. These tests make sure the gates themselves are right
before that happens: that gate B's dense accumulation really does reproduce
what the writer stored, that gate C divides the worker timings by the right
frame count, and that gate D refuses a ledger with a non-``OK`` frame in it.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("extra_data")

import h5py  # noqa: E402
from test_run import InlinePool  # noqa: E402  (tests/saxs is on sys.path)

from analysis.saxs.status import FrameStatus  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "p4_acceptance.py"


@pytest.fixture(scope="module")
def p4():
    """Import the script by path: ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("p4_acceptance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["p4_acceptance"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def finished_run(mock_pipeline, geom, tmp_path):
    """A completed pass over the mock run, plus its output file."""
    from analysis.saxs.run import run_first_pass

    output = tmp_path / "out.h5"
    run_first_pass(
        mock_pipeline.cfg,
        dc=mock_pipeline.dc,
        geometry=geom,
        run_dir=mock_pipeline.run.path,
        output_path=output,
        work_dir=tmp_path,
        pool_factory=InlinePool,
    )
    return mock_pipeline, output


# ── gate B ────────────────────────────────────────────────────────────────────
def test_dense_reference_reproduces_the_stored_sums(p4, finished_run, geom):
    pipeline, output = finished_run
    result = p4.stage_reference(
        pipeline.cfg,
        output,
        n_trains=2,
        tolerance=1e-6,
        run_dir=pipeline.run.path,
        geometry=geom,
    )
    assert result["passed"], result
    assert result["empty_bins_agree"]
    assert result["max_rel_intensity"] < 1e-6
    assert len(result["per_train"]) == 2
    assert all(t["populated_bins"] > 0 for t in result["per_train"])


def test_dense_reference_refuses_a_different_operator(p4, finished_run, geom):
    """A reference built on another geometry is not a reference."""
    pipeline, output = finished_run
    result = p4.stage_reference(
        replace(pipeline.cfg, sdd_m=pipeline.cfg.sdd_m * 1.05),
        output,
        n_trains=1,
        tolerance=1e-6,
        run_dir=pipeline.run.path,
        geometry=geom,
    )
    assert result["passed"] is False
    assert "operator" in result["reason"]


def test_dense_reference_catches_a_corrupted_row(p4, finished_run, geom):
    """Scaling one stored frame must show up as a pooled disagreement."""
    pipeline, output = finished_run
    with h5py.File(output, "r+") as handle:
        handle["frames/signal"][0] = handle["frames/signal"][0] * 1.5

    result = p4.stage_reference(
        pipeline.cfg,
        output,
        n_trains=2,
        tolerance=1e-6,
        run_dir=pipeline.run.path,
        geometry=geom,
    )
    assert result["passed"] is False
    assert result["max_rel_intensity"] > 1e-6


# ── gate C ────────────────────────────────────────────────────────────────────
def test_timing_divides_by_the_ok_frames(p4, finished_run):
    pipeline, output = finished_run
    result = p4.stage_timing(output, measured_wall_s=None)

    assert result["ok_frames"] == pipeline.plan.n_frames
    assert set(result["stages"]) == set(p4.BUDGET_MS)
    for name, stage in result["stages"].items():
        assert stage["cpu_s"] > 0
        assert stage["ms_per_frame_per_core"] == pytest.approx(
            stage["cpu_s"] * 1e3 / result["ok_frames"]
        )
        assert stage["ratio"] == pytest.approx(
            stage["ms_per_frame_per_core"] / p4.BUDGET_MS[name]
        )
    assert result["wall_s"] > 0  # from provenance, not from the caller
    assert result["passed"] is True


def test_timing_uses_the_wall_time_from_provenance(p4, finished_run):
    _, output = finished_run
    with h5py.File(output, "r") as handle:
        stored = float(handle["provenance"].attrs["wall_s"])
    assert p4.stage_timing(output, measured_wall_s=1e9)["wall_s"] == stored


def test_timing_fails_when_the_wall_target_is_missed(p4, finished_run, monkeypatch):
    _, output = finished_run
    monkeypatch.setattr(p4, "WALL_TARGET_S", 1e-9)
    assert p4.stage_timing(output, None)["passed"] is False


# ── gate D ────────────────────────────────────────────────────────────────────
def test_ledger_passes_on_an_all_ok_run(p4, finished_run):
    pipeline, output = finished_run
    result = p4.stage_ledger(output)

    assert result["passed"] is True
    assert result["reconciles"]
    assert result["frame_status"] == {FrameStatus.OK.name: pipeline.plan.n_frames}
    assert result["non_ok"] == {}
    assert result["unexpected_bits"] == 0
    assert "xray_pulses" in result["run_checks"]


def test_ledger_names_the_trains_behind_a_non_ok_status(p4, finished_run):
    """A count nobody can trace back to a train is not an account of it."""
    pipeline, output = finished_run
    with h5py.File(output, "r+") as handle:
        first = int(handle["trains/first"][1])
        handle["frames/status"][first] = FrameStatus.DATA_CHECK_FAILED
        expected = int(handle["trains/trainId"][1])

    result = p4.stage_ledger(output)
    assert result["passed"] is False
    offender = result["non_ok"][FrameStatus.DATA_CHECK_FAILED.name]
    assert offender["n_frames"] == 1
    assert offender["trains"] == [expected]
    assert result["reconciles"]  # the frames still add up; only the status moved


def test_ledger_reports_a_frame_count_that_does_not_add_up(p4, finished_run):
    _, output = finished_run
    with h5py.File(output, "r+") as handle:
        handle["trains/count"][0] = handle["trains/count"][0] + 1

    result = p4.stage_ledger(output)
    assert result["reconciles"] is False
    assert result["passed"] is False


# ── the driver ────────────────────────────────────────────────────────────────
def test_thread_env_is_pinned_at_import(p4):
    import os

    assert all(os.environ[name] == "1" for name in p4._THREAD_ENV)


def test_budget_matches_the_context_file(p4):
    """The §2 budget is a measurement; a silent edit here would hide a miss."""
    assert p4.BUDGET_MS == {"read_data": 4.6, "read_mask": 8.2, "integrate": 6.2}
    assert sum(p4.BUDGET_MS.values()) == pytest.approx(19.0)


def test_pooled_from_file_matches_a_manual_pool(p4, finished_run):
    _, output = finished_run
    with h5py.File(output, "r") as handle:
        signal, normalization = p4._pooled_from_file(
            handle, int(handle["trains/trainId"][0])
        )
        n = int(handle["trains/count"][0])
        expected = handle["frames/signal"][:n].astype(np.float64).sum(axis=0)
    assert np.array_equal(signal, expected)
    assert (normalization > 0).any()
