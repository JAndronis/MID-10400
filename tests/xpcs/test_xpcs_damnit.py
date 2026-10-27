"""``analysis.xpcs.damnit``: hit resolution, the pass call, cells and readers.

The pass itself is exercised in ``test_xpcs_run.py``; here it is replaced by a
recorder, so what is checked is what DAMNIT would see and what the pass is asked.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from xpcs_fakes import (
    build_results,
    make_dense,
    make_flags,
    make_rings,
    make_train,
)

pytest.importorskip("damnit")

from damnit.context import Cell, Skip  # noqa: E402

from analysis.common.lock import OutputBusy  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.common.writer import IncompleteRun  # noqa: E402
from analysis.diagnostics.damnit import COLORS  # noqa: E402
from analysis.xpcs import damnit as xd  # noqa: E402
from analysis.xpcs.hits import HitSet  # noqa: E402
from analysis.xpcs.result import RepRateMismatch  # noqa: E402

IDS = (100, 101, 104, 140, 141, 145)
N_ENV, P, R = 300, 14, 3
SIGNALS = ("saxs", "xgm", "transmission")


def colour(cell) -> str:
    return "#" + bytes(cell.background).hex()


def results_file(path: Path, ids=IDS, failed=None):
    rng = np.random.default_rng(3)
    rings = make_rings(rng, N_ENV, R)
    photons = [
        make_train(t, make_dense(rng, P, N_ENV, 0.12), make_flags(rng, P, N_ENV))
        for t in ids
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    build_results(path, photons, rings, failed=failed)
    return path


@pytest.fixture
def classifier(monkeypatch):
    """The hit rule, replaced; records how it was called."""
    import analysis.triage.classify as classify

    calls = {}

    def fake_signals_from(*values):
        calls["signals_from"] = values
        return "SIGNALS"

    def fake_hits(run, **kwargs):
        calls["hits"] = (run, kwargs)
        return {"train_ids": np.array(calls["ids"], dtype=np.uint64),
                "kind": "static", "block": None}  # fmt: skip

    monkeypatch.setattr(classify, "signals_from", fake_signals_from)
    monkeypatch.setattr(classify, "hits", fake_hits)
    calls["ids"] = IDS
    return calls


@pytest.fixture
def the_pass(monkeypatch, tmp_path):
    """``run_xpcs_hits`` replaced by a recorder that returns a real results file."""
    calls = []

    def fake(cfg, hits):
        calls.append((cfg, hits))
        return results_file(
            tmp_path / "out" / "xpcs_hits.h5", ids=tuple(hits.train_ids)
        )

    monkeypatch.setattr(xd, "run_xpcs_hits", fake)
    return calls


def test_the_pass_is_run_on_the_hits_of_the_dependencies(classifier, the_pass):
    cell = xd.preprocess(10400, 423, *SIGNALS, output_root="/elsewhere", n_workers=3)
    assert classifier["signals_from"] == SIGNALS
    assert classifier["hits"] == (423, {"proposal": 10400, "signals": "SIGNALS"})
    ((cfg, hits),) = the_pass
    assert isinstance(hits, HitSet) and hits.train_ids.tolist() == list(IDS)
    assert "classify.hits" in hits.source and "static" in hits.source
    assert (cfg.proposal, cfg.run) == (10400, 423)
    assert cfg.saxs == xd.saxs_config_for(10400, 423)
    assert cfg.max_partner_gap == 1 and cfg.overwrite is False
    assert cfg.output_root == "/elsewhere" and cfg.n_workers == 3
    assert isinstance(cell, Cell) and colour(cell) == COLORS["good"]
    record = json.loads(cell.data)
    assert record["n_hits"] == len(IDS) == record["n_usable"]
    assert record["hit_set_sha256"] == hits.sha256


def test_a_run_without_hits_is_a_grey_result_not_a_failure(classifier, the_pass):
    classifier["ids"] = ()
    cell = xd.preprocess(10400, 423, *SIGNALS)
    ((_, hits),) = the_pass  # the pass runs; its zero-train file is the answer
    assert len(hits) == 0
    assert cell.summary_value == "no hits" and colour(cell) == xd.NO_HITS_COLOR
    assert json.loads(cell.data)["n_hits"] == 0


def test_a_failing_pass_is_a_red_cell_not_an_exception(classifier, monkeypatch):
    def fail(cfg, hits):
        raise RuntimeError("proc is gone")

    monkeypatch.setattr(xd, "run_xpcs_hits", fail)
    cell = xd.preprocess(10400, 423, *SIGNALS)
    assert colour(cell) == COLORS["error"]
    assert cell.summary_value.startswith("failed (pass)")
    assert "proc is gone" in json.loads(cell.data)["error"]


def test_a_failing_hit_rule_is_a_red_cell_and_the_pass_is_not_run(
    classifier, the_pass, monkeypatch
):
    import analysis.triage.classify as classify

    def fail(run, **kwargs):
        raise LookupError("r423 is not in DAMNIT's run table")

    monkeypatch.setattr(classify, "hits", fail)
    cell = xd.preprocess(10400, 423, *SIGNALS)
    assert cell.summary_value.startswith("failed (hit rule)")
    assert colour(cell) == COLORS["error"] and not the_pass


def test_a_busy_output_is_amber(classifier, monkeypatch):
    def busy(cfg, hits):
        raise OutputBusy("locked by another job")

    monkeypatch.setattr(xd, "run_xpcs_hits", busy)
    cell = xd.preprocess(10400, 423, *SIGNALS)
    assert cell.summary_value == "busy" and colour(cell) == COLORS["warning"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (FrameStatus.NO_FRAMES, "warning"),
        (FrameStatus.LABEL_MISMATCH, "error"),
    ],
)
def test_the_cell_is_coloured_by_what_became_of_the_hits(tmp_path, status, expected):
    path = results_file(tmp_path / "x.h5", failed={999: status})
    cell = xd.coverage_cell(path)
    assert colour(cell) == COLORS[expected]
    assert cell.summary_value.startswith(f"{len(IDS)} / {len(IDS) + 1} hits")


def test_a_cell_without_a_file_is_skipped(tmp_path):
    with pytest.raises(Skip, match="xpcs_hits_preprocessing"):
        xd.coverage_cell(tmp_path / "none.h5")
    with pytest.raises(Skip, match="xpcs_hits_preprocessing"):
        xd.frames_used(10400, 423, output_root=str(tmp_path))


@pytest.fixture
def written(tmp_path):
    results_file(tmp_path / "xpcs_hits" / "r0423" / "xpcs_hits.h5")
    return {"output_root": str(tmp_path)}


def test_frames_used_reads_the_file_the_config_names(written):
    cell = xd.frames_used(10400, 423, **written)
    assert cell.summary_value.startswith(f"{len(IDS)} / {len(IDS)} hits")
    assert colour(cell) == COLORS["good"]


def test_the_g2_reader_checks_the_repetition_rate_against_the_pattern(written):
    g2 = xd.g2_from_file(10400, 423, 2.2569, **written)
    assert {"g2", "sigma", "c2", "baseline"} <= set(g2.data_vars)
    assert g2.sizes["roi_q"] == R and g2.attrs["averaging"] == "pooled"
    with pytest.raises(RepRateMismatch):
        xd.g2_from_file(10400, 423, 1.13, **written)


def test_the_readers_skip_a_run_without_hits_or_without_a_file(tmp_path):
    with pytest.raises(Skip, match="xpcs_hits_preprocessing"):
        xd.g2_from_file(10400, 423, output_root=str(tmp_path))
    results_file(tmp_path / "xpcs_hits" / "r0423" / "xpcs_hits.h5", ids=())
    with pytest.raises(Skip, match="no droplet hit"):
        xd.ttcf_from_file(10400, 423, output_root=str(tmp_path))


def test_the_readers_refuse_unfinished_hits(tmp_path):
    results_file(
        tmp_path / "xpcs_hits" / "r0423" / "xpcs_hits.h5",
        failed={999: FrameStatus.NOT_PROCESSED},
    )
    with pytest.raises(IncompleteRun):
        xd.g2_from_file(10400, 423, output_root=str(tmp_path))


def test_only_the_damnit_module_imports_the_classifier():
    """The hit rule changes on its own schedule, so exactly one module may import it."""
    package = Path(xd.__file__).parent
    importers = []
    for path in sorted(package.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [node.module]
                if isinstance(node, ast.ImportFrom) and node.module
                else [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            if any(n.startswith("analysis.triage.classify") for n in names):
                importers.append(path.name)
    assert importers == ["damnit.py"]
