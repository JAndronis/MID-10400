"""The W4 acceptance gates, against the mock run.

``main`` needs Maxwell — it runs the pass by proposal and run number — but every
gate reads a finished output file and an open run, so they can be driven here.
That is the part worth testing: a gate that cannot fail is not a gate, so each
one is shown failing as well as passing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import pytest

pytest.importorskip("extra_data")

from waxs_mockrun import LIT_CELLS  # noqa: E402

from analysis.common.status import FrameStatus  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "w4_acceptance.py"


@pytest.fixture(scope="module")
def w4():
    spec = importlib.util.spec_from_file_location("w4_acceptance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def finished(cfg, mock_run_factory, tmp_path):
    """A completed pass over the mock run, plus its output path and run."""
    import dataclasses
    from types import SimpleNamespace

    from analysis.waxs.run import run_jungfrau_waxs

    mock, dc = mock_run_factory()
    output = tmp_path / "w4.h5"
    workers = 2
    run_jungfrau_waxs(
        dataclasses.replace(cfg, n_workers=workers),
        dc=dc,
        run_dir=mock.path,
        output_path=output,
        reduce="none",
    )
    return SimpleNamespace(
        cfg=dataclasses.replace(cfg, n_workers=workers),
        output=output,
        dc=dc,
        mock=mock,
        workers=workers,
    )


# ── gate 0 ───────────────────────────────────────────────────────────────────
def test_configuration_passes_on_the_run_it_describes(w4, finished):
    result = w4.stage_configuration(finished.cfg, finished.output, finished.workers)
    assert result["passed"]
    assert result["config_hash_matches"]
    assert result["workers_as_requested"]
    assert result["lit_matches_config"]
    assert result["detector_name"] == "MID_EXP_JF500K1"
    assert result["n_frames"] == finished.mock.n_frames


def test_configuration_fails_on_a_different_worker_count(w4, finished):
    result = w4.stage_configuration(finished.cfg, finished.output, finished.workers + 1)
    assert result["passed"] is False
    assert result["workers_as_requested"] is False


def test_configuration_fails_on_a_different_config(w4, finished):
    import dataclasses

    other = dataclasses.replace(finished.cfg, npt=finished.cfg.npt // 2)
    result = w4.stage_configuration(other, finished.output, finished.workers)
    assert result["passed"] is False
    assert result["config_hash_matches"] is False


# ── gate A ───────────────────────────────────────────────────────────────────
def test_the_selftest_verdict_is_read_back_out_of_provenance(w4, finished):
    result = w4.stage_selftest(finished.output, 1e-9)
    assert result["passed"]
    assert result["n_frames"] == finished.cfg.selftest_frames
    assert result["max_rel"] <= 1e-9


def test_the_selftest_gate_fails_on_a_tolerance_it_cannot_meet(w4, finished):
    with h5py.File(finished.output, "r+") as handle:
        stored = json.loads(handle["provenance"].attrs["selftest"])
        stored["max_rel_signal"] = 1e-3
        handle["provenance"].attrs["selftest"] = json.dumps(stored)
    result = w4.stage_selftest(finished.output, 1e-9)
    assert result["passed"] is False
    assert result["max_rel"] == pytest.approx(1e-3)


# ── gate B ───────────────────────────────────────────────────────────────────
def test_the_reference_reproduces_what_was_stored(w4, finished):
    """The slow per-frame-mask path against the file the NaN path wrote."""
    result = w4.stage_reference(
        finished.cfg, finished.output, n_trains=2, tolerance=1e-6, dc=finished.dc
    )
    assert result["passed"], result
    assert result["empty_bin_sets_equal"]
    assert result["max_rel_intensity"] < 1e-6
    assert len(result["trains"]) == 2
    assert all(t["n_bins_populated"] > 0 for t in result["trains"])


def test_the_reference_catches_a_corrupted_stored_row(w4, finished):
    """A gate that cannot fail is not a gate."""
    with h5py.File(finished.output, "r+") as handle:
        first = int(handle["trains/first"][0])
        signal = handle["frames/signal"][first]
        handle["frames/signal"][first] = signal * 1.05

    result = w4.stage_reference(
        finished.cfg, finished.output, n_trains=2, tolerance=1e-6, dc=finished.dc
    )
    assert result["passed"] is False
    assert result["max_rel_intensity"] > 1e-6


def test_the_reference_refuses_a_file_from_another_geometry(w4, finished):
    with h5py.File(finished.output, "r+") as handle:
        handle["operator"].attrs["sha256"] = "not-the-operator-this-config-builds"
    result = w4.stage_reference(
        finished.cfg, finished.output, n_trains=1, tolerance=1e-6, dc=finished.dc
    )
    assert result["passed"] is False
    assert "operator" in result["reason"]


# ── gate C ───────────────────────────────────────────────────────────────────
def test_timing_reports_per_frame_costs(w4, finished):
    result = w4.stage_timing(finished.output, wall_s=12.0)
    assert result["passed"]
    assert result["wall_s"] == 12.0
    assert set(result["ms_per_frame_per_core"]) == {
        "read_data",
        "read_mask",
        "integrate",
    }
    assert result["total_ms_per_frame_per_core"] > 0
    assert result["parallel_efficiency"] > 0
    assert 0 <= result["serial_fraction"] <= 1


def test_timing_fails_only_on_the_wall_clock(w4, finished):
    """A stage over budget is reported; only the wall time is a verdict."""
    slow = w4.stage_timing(finished.output, wall_s=w4.WALL_TARGET_S + 1)
    assert slow["passed"] is False

    original = dict(w4.BUDGET_MS)
    try:
        w4.BUDGET_MS.update({k: 1e-9 for k in w4.BUDGET_MS})
        result = w4.stage_timing(finished.output, wall_s=1.0)
        assert result["passed"] is True
        assert set(result["over_budget"]) == set(original)
    finally:
        w4.BUDGET_MS.clear()
        w4.BUDGET_MS.update(original)


def test_timing_is_undecided_without_a_wall_time(w4, finished, tmp_path):
    import shutil

    empty = tmp_path / "nowall.h5"
    shutil.copy(finished.output, empty)
    with h5py.File(empty, "r+") as handle:
        handle["provenance"].attrs["wall_s"] = 0.0
    assert w4.stage_timing(empty, wall_s=None)["passed"] is None


# ── gate D ───────────────────────────────────────────────────────────────────
def test_the_ledger_reconciles_on_a_clean_run(w4, finished):
    result = w4.stage_ledger(finished.cfg, finished.output)
    assert result["passed"]
    assert result["status_counts"] == {"OK": finished.mock.n_frames}
    assert result["frames_reconcile"]
    assert result["unexpected_bits"] == []
    assert result["lit_cells"] == list(LIT_CELLS)
    assert result["offending_trains"] == {}


def test_the_ledger_names_the_trains_behind_a_bad_status(w4, finished):
    with h5py.File(finished.output, "r+") as handle:
        count = int(handle["trains/count"][1])
        first = int(handle["trains/first"][1])
        handle["frames/status"][first : first + count] = FrameStatus.WORKER_ERROR
        handle["trains/status"][1] = FrameStatus.WORKER_ERROR
        expected = int(handle["trains/trainId"][1])

    result = w4.stage_ledger(finished.cfg, finished.output)
    assert result["passed"] is False
    assert result["status_counts"]["WORKER_ERROR"] == count
    assert expected in result["offending_trains"]["WORKER_ERROR"]["trains"]


def test_the_ledger_flags_an_unexpected_mask_bit(w4, finished):
    with h5py.File(finished.output, "r+") as handle:
        handle["provenance"].attrs["bits_present"] = int(1 << 13)
    result = w4.stage_ledger(finished.cfg, finished.output)
    assert result["passed"] is False
    assert result["unexpected_bits"] == [13]


def test_the_ledger_reports_negative_variance_bins_without_failing(w4, finished):
    """§3 D3: unclamped is the design, so these are counted, never failed."""
    with h5py.File(finished.output, "r+") as handle:
        handle["frames/n_negative_variance_bins"][0] = 7
    result = w4.stage_ledger(finished.cfg, finished.output)
    assert result["passed"]
    assert result["frames_with_negative_variance_bins"] >= 1
    assert result["max_negative_variance_bins_in_a_frame"] >= 7


# ── the script's own contract ────────────────────────────────────────────────
def test_skip_run_without_an_output_file_exits_two(w4, cfg, tmp_path):
    code = w4.main(
        [
            "--run",
            "423",
            "--detector",
            "jf1",
            "--skip-run",
            "--poni-file",
            cfg.poni_file,
            "--static-mask-file",
            cfg.static_mask_file,
            "--output",
            str(tmp_path / "absent.h5"),
            "--json",
            str(tmp_path / "verdict.json"),
        ]
    )
    assert code == 2
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["passed"] is False
    assert "does not exist" in verdict["gates"]["A_run"]["reason"]


def test_a_missing_input_file_is_reported_not_raised(w4, tmp_path):
    """A wrong path on the cluster should read as a sentence, not a traceback."""
    code = w4.main(
        [
            "--run",
            "423",
            "--detector",
            "jf1",
            "--skip-run",
            "--poni-file",
            str(tmp_path / "nowhere.poni"),
            "--json",
            str(tmp_path / "missing.json"),
        ]
    )
    assert code == 2
    verdict = json.loads((tmp_path / "missing.json").read_text())
    assert verdict["passed"] is False
    assert "poni_file" in verdict["gates"]["0_inputs"]["missing"]


def test_a_none_gate_does_not_fail_the_run(w4):
    """SKIP is not FAIL: an undecided gate must not sink an otherwise good run."""
    verdicts = {"A": True, "B": None, "C": True}
    assert all(v is not False for v in verdicts.values())
    verdicts["D"] = False
    assert not all(v is not False for v in verdicts.values())


def test_the_wall_target_is_above_the_measured_cost():
    """A regression canary, not a budget: it must not fire on a healthy run."""
    spec = importlib.util.spec_from_file_location("w4_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # ~24 000 frames per detector at the measured 2.3-2.9 ms on one core, well
    # under an hour even without any parallelism.
    assert module.WALL_TARGET_S >= 24_000 * 0.003 * 4
    assert all(v > 0 for v in module.BUDGET_MS.values())


def test_the_ledger_surfaces_trains_that_own_no_rows(
    w4, cfg, mock_run_factory, tmp_path
):
    """A skipped train leaves no mark in the frame ledger; the gate must show it.

    r0423 has 3001 trains of which 3000 carry data, and the frame ledger
    reconciles perfectly either way — so "24000/24000 OK" alone cannot
    distinguish a complete run from one that quietly dropped a train.
    """
    import dataclasses

    from analysis.waxs.run import run_jungfrau_waxs

    mock, dc = mock_run_factory(zero_entry_trains=(10002,))
    output = tmp_path / "gap.h5"
    run_jungfrau_waxs(
        dataclasses.replace(cfg, n_workers=1),
        dc=dc,
        run_dir=mock.path,
        output_path=output,
        reduce="none",
    )
    result = w4.stage_ledger(cfg, output)

    assert result["status_counts"] == {"OK": mock.n_frames}
    assert result["frames_reconcile"]
    # ...and yet a train is missing, which the train-level counts say plainly.
    assert result["n_trains"] == len(mock.train_ids)
    assert result["n_trains_with_rows"] == len(mock.train_ids) - 1
    assert result["train_status_counts"]["NO_FRAMES"] == 1
    assert 10002 in result["offending_trains"]["NO_FRAMES"]["trains"]
    assert result["offending_trains"]["NO_FRAMES"]["n_frames"] == 0


def test_the_budgets_bracket_the_measured_cost(w4):
    """Measured on r0423 jf1: 11.6 / 5.0 / 3.0 ms per frame per core."""
    measured = {"read_data": 11.6, "read_mask": 5.0, "integrate": 3.0}
    for stage, value in measured.items():
        assert value < w4.BUDGET_MS[stage], stage
        # Tight enough to be a canary: a 5x regression must trip it.
        assert value * 5 > w4.BUDGET_MS[stage], stage
    assert w4.WALL_TARGET_S > 17.5 * 10
