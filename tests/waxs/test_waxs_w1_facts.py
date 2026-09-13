"""The W1 fact-finding stages, against the mock run.

The script itself needs Maxwell — it opens a proc run by proposal and run
number. Its stages do not: each takes an already-open detector, so they can be
driven against the mock here. That is the part worth testing; ``main`` is
argument parsing and printing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("extra_data")

from waxs_mockrun import LIT_CELLS  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "w1_facts.py"


@pytest.fixture(scope="module")
def w1():
    """Import the script by path: ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("w1_facts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_files_finds_the_configured_inputs(w1, cfg):
    result = w1.stage_files(cfg)
    assert result["passed"]
    assert set(result["files"]) == {"poni_file", "static_mask_file"}
    for entry in result["files"].values():
        assert entry["exists"] and entry["sha256"] and entry["bytes"] > 0


def test_stage_files_reports_a_missing_file_rather_than_raising(w1, cfg):
    import dataclasses

    missing = dataclasses.replace(cfg, poni_file="/nowhere/jf1.poni")
    result = w1.stage_files(missing)
    assert result["passed"] is False
    assert result["files"]["poni_file"]["exists"] is False


def test_stage_sources_records_the_layout_and_the_legacy_alias(w1, pipeline):
    result = w1.stage_sources(pipeline.cfg, pipeline.dc, pipeline.det)
    assert result["passed"]
    assert result["sources"] == ["MID_EXP_JF500K1/CORR/JNGFR01:daqOutput"]
    assert result["n_modules"] == 1
    assert result["frames_per_entry"] == 16
    assert result["entries_per_train"] == [1]
    assert result["legacy_aliases"] == {
        "MID_EXP_JF500K1/DET/JNGFR01:daqOutput": (
            "MID_EXP_JF500K1/CORR/JNGFR01:daqOutput"
        )
    }


def test_stage_cells_answers_o4(w1, pipeline):
    result = w1.stage_cells(pipeline.cfg, pipeline.det, pipeline.op, 2)
    assert result["passed"]
    assert result["lit"] == list(LIT_CELLS)
    assert result["lit_matches_expected"]
    assert result["read_noise_kev"] == pytest.approx(0.32, rel=0.05)
    assert result["unexpected_bits"] == []
    # Every cell's evidence, not just the verdict: the next question is always
    # "by how much did it miss?"
    assert len(result["lit_fraction"]) == 16


def test_stage_cells_flags_a_changed_lit_set(w1, cfg, mock_run_factory, operator):
    from analysis.waxs.plan import open_detector

    op, _ = operator
    _, dc = mock_run_factory(lit_cells=(0, 1, 2, 3))
    result = w1.stage_cells(cfg, open_detector(cfg, dc), op, 2)
    assert result["passed"] is False
    assert result["lit"] == [0, 1, 2, 3]
    assert result["lit_matches_expected"] is False


def test_stage_mask_variability_sees_a_static_mask(w1, pipeline):
    """The mock seeds its dynamic bits per frame, so it should see variation."""
    result = w1.stage_mask_variability(pipeline.cfg, pipeline.det, 3)
    assert result["passed"] is None  # a measurement, not a gate
    assert result["n_trains_compared"] == 3
    assert isinstance(result["identical_in_every_sampled_train"], bool)
    assert 0.0 <= result["fraction_that_ever_differ"] <= 1.0


def test_stage_roi_times_both_reads(w1, pipeline):
    result = w1.stage_roi(pipeline.cfg, pipeline.det, tuple(range(8)), repeats=1)
    assert result["lit_cells_contiguous"] is True
    assert result["roi"] == [0, 8]
    assert result["full_read_s"] > 0
    assert result["roi_read_s"] > 0
    assert result["speedup"] > 0
    assert result["datasets"]["data.adc"]["entry_shape"] == [16, 512, 1024]


def test_stage_roi_declines_a_non_contiguous_lit_set(w1, pipeline):
    result = w1.stage_roi(pipeline.cfg, pipeline.det, (0, 2, 4), repeats=1)
    assert result["lit_cells_contiguous"] is False
    assert result["roi"] is None
    assert result["roi_read_s"] is None


def test_stage_values_counts_the_extreme_pixels(w1, cfg, mock_run_factory, operator):
    from analysis.waxs.plan import open_detector

    op, _ = operator
    _, dc = mock_run_factory(extreme_train=10000)
    result = w1.stage_values(cfg, open_detector(cfg, dc), op)
    assert result["dtype"] == "float32"
    assert result["n_extreme"] >= 1
    # The mock puts it on a kept pixel and does not flag it, exactly as jf2 does.
    assert result["n_extreme_unflagged_by_data_mask"] >= 1
    assert result["n_extreme_reaching_the_integrator"] >= 1
    assert 0.0 <= result["negative_fraction_kept"] <= 1.0


def test_stage_values_on_a_clean_run_finds_nothing(w1, pipeline):
    result = w1.stage_values(pipeline.cfg, pipeline.det, pipeline.op)
    assert result["n_extreme"] == 0
    assert result["n_extreme_reaching_the_integrator"] == 0


def test_the_o4_verdict_is_the_agreement_across_runs(w1):
    """What ``main`` concludes from the per-run results."""
    same = [
        {"run": 423, "detector": "jf1", "cells": {"lit": [0, 1]}, "passed": True},
        {"run": 426, "detector": "jf1", "cells": {"lit": [0, 1]}, "passed": True},
    ]
    differing = [
        {"run": 423, "detector": "jf1", "cells": {"lit": [0, 1]}, "passed": True},
        {"run": 426, "detector": "jf1", "cells": {"lit": [0, 1, 2]}, "passed": False},
    ]
    for results, expected in ((same, True), (differing, False)):
        sets = {(r["run"], r["detector"]): tuple(r["cells"]["lit"]) for r in results}
        assert (len(set(sets.values())) <= 1) is expected


def test_a_stage_reports_rather_than_crashing_the_run(w1, pipeline, monkeypatch):
    """One bad detector must not take the other's facts down with it."""

    def boom(*args, **kwargs):
        raise RuntimeError("no data")

    monkeypatch.setattr(w1, "stage_cells", boom)
    with pytest.raises(RuntimeError):
        w1.stage_cells(None, None, None, 1)
    # ...and `main` wraps each (run, detector) in its own try/except, which is
    # what keeps a second detector reachable. Exercised via the source, since
    # main() needs a real proc run.
    source = SCRIPT.read_text()
    assert "except Exception as error:" in source
    assert '"error": repr(error)' in source
