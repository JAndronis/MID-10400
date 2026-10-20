"""Per-run XPCS SNR and score from a hit-train results file: β from a report, the
verdict, k̄ per ring, the hit coverage."""

from __future__ import annotations

import json

import numpy as np
import pytest
from triage_fakes import FIRST_TRAIN, write_hit_results

from analysis.common.status import FrameStatus
from analysis.diagnostics.xpcs import (
    KBAR_MIN,
    beta_by_roi,
    kbar_rois,
    load_beta,
    xpcs_score,
    xpcs_snr,
)
from analysis.triage.xpcs import (
    ES_REFERENCE_BETA,
    OFF_SUBTRACTION,
    REFERENCE_BETA_RUNS,
    measure_beta,
    measure_run,
)
from analysis.xpcs.result import read

RUN = 423
FAILED = {FIRST_TRAIN + 7: int(FrameStatus.NO_FRAMES)}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """An XPCS output root holding one run's results file, one hit of it failed."""
    base = tmp_path_factory.mktemp("xpcs_root")
    write_hit_results(
        base / "xpcs_hits" / f"r{RUN:04d}" / "xpcs_hits.h5", failed=FAILED
    )
    return base


def report_for(m) -> dict:
    """The ``beta`` block as scripts/run_triage.py writes it."""
    beta = measure_beta([m])
    return {
        "beta": {
            str(float(m.roi_q_A[r])): {
                "value": b.value, "n_eligible": b.n_eligible, "p16": b.p16,
                "p84": b.p84, "runs": list(b.runs),
            }
            for r, b in beta.items()
        }
    }  # fmt: skip


def test_snr_with_a_beta_from_a_report(root, tmp_path):
    out = xpcs_snr(RUN, {}, output_root=root)
    m = out["measurement"]
    report = tmp_path / "report.json"
    given = {
        "beta": {
            str(float(q)): {"value": 0.03, "n_eligible": 4, "p16": 0.02, "p84": 0.04,
                            "runs": [450]}
            for q in m.roi_q_A
        }
    }  # fmt: skip
    report.write_text(json.dumps(given))
    out = xpcs_snr(RUN, load_beta(report), output_root=root)
    rois = out["rois"]
    expected = (
        0.03 * m.kbar[0] * np.sqrt(m.npix[0] * (m.n_frames - 1) * m.phi[0] * m.n_usable)
    )
    assert rois.snr_run[0] == pytest.approx(expected / OFF_SUBTRACTION)
    assert out["summary"]["beta_known"] is True
    assert out["summary"]["beta_source"] == "given"
    assert out["summary"]["extractable"] == (out["summary"]["best_snr"] >= 3)
    assert list(out["slices"].columns[:4]) == [
        "slice",
        "start_s",
        "n_usable",
        "best_snr",
    ]
    assert out["slices"].n_usable.sum() == m.n_usable


def test_the_default_beta_is_the_reference_table(root):
    out = xpcs_snr(RUN, output_root=root)
    rois = out["rois"]
    for q, (value, _, _) in ES_REFERENCE_BETA.items():  # the fake has ES rings
        row = rois[np.isclose(rois.roi_q_A, q, atol=1e-6)]
        assert row.beta.item() == value
        assert row.beta_n_eligible.item() == len(REFERENCE_BETA_RUNS)
    assert out["summary"]["beta_source"] == "REFERENCE_BETA"
    assert out["summary"]["beta_known"] is True


def test_an_unknown_beta_gives_no_verdict(root):
    out = xpcs_snr(RUN, {}, output_root=root)
    assert out["summary"]["extractable"] is None
    assert out["summary"]["beta_known"] is False
    assert out["rois"].snr_run.isna().all()


def test_beta_matches_rois_by_centre_and_reads_json_nulls():
    report = {
        "beta": {
            "0.02875": {"value": 0.03, "n_eligible": 4, "p16": 0.02, "p84": 0.04,
                        "runs": [400, 450]},
            "0.04": {"value": None, "n_eligible": 0, "p16": None, "p84": None,
                     "runs": []},
        }
    }  # fmt: skip
    beta = load_beta(report)
    by_index = beta_by_roi(np.array([0.04, 0.02875, 0.0805]), beta)
    assert np.isnan(by_index[0].value)
    assert by_index[1].value == 0.03 and by_index[1].runs == (400, 450)
    assert np.isnan(by_index[2].value) and by_index[2].n_eligible == 0


def test_a_run_without_a_results_file_is_refused(tmp_path):
    with pytest.raises(LookupError, match="no XPCS results file"):
        xpcs_snr(RUN, {}, output_root=tmp_path)
    with pytest.raises(LookupError, match="no XPCS results file"):
        xpcs_score(RUN, output_root=tmp_path)


# ── k̄ per ring and the per-run score ─────────────────────────────────────────


def test_kbar_rois_counts_the_rings_above_the_threshold(root):
    table = kbar_rois(RUN, output_root=root)
    np.testing.assert_allclose(table.kbar[:4], [0.4, 0.2, 0.02, 0.01], rtol=0.2)
    assert list(table.above) == [True] * 4 + [False] * 3
    assert table.attrs["n_above"] == 4
    assert table.attrs["n_plain_above"] == 3  # 0.0605 is a Bragg ROI
    assert table.attrs["kbar_min"] == KBAR_MIN == 3e-3
    assert table.attrs["n_trains"] == 59  # the failed hit has no sums
    assert "xpcs_hits.h5" in table.attrs["source"]


def test_kbar_rois_over_given_hits(root):
    some = np.array([FIRST_TRAIN + i for i in (1, 2, 3)], dtype=np.uint64)
    assert kbar_rois(RUN, some, output_root=root).attrs["n_trains"] == 3


def test_the_score_is_taken_over_the_hits_in_the_file(root):
    score = xpcs_score(RUN, output_root=root)
    assert score["n_hits"] == 60  # the failed hit is counted, and accounted for
    assert score["hits"]["n_usable"] == 59
    assert score["hits"]["by_status"]["NO_FRAMES"] == 1
    assert score["n_rois_kbar_ok"] == 4
    assert score["n_plain_rois_kbar_ok"] == 3
    assert (score["n_rois"], score["n_plain_rois"]) == (7, 4), "the file's ES rings"
    expected = measure_run(RUN, read(10400, RUN, output_root=root))
    assert score["snr"]["n_usable"] == expected.n_usable > 0


def test_a_run_with_no_usable_hit_has_no_score(tmp_path):
    write_hit_results(
        tmp_path / "xpcs_hits" / f"r{RUN:04d}" / "xpcs_hits.h5",
        n_trains=2,
        failed={
            FIRST_TRAIN: int(FrameStatus.NO_FRAMES),
            FIRST_TRAIN + 1: int(FrameStatus.NO_FRAMES),
        },
    )
    score = xpcs_score(RUN, output_root=tmp_path)
    assert score["n_hits"] == 2
    assert score["kbar"] is None and score["snr"] is None
    assert score["n_rois_kbar_ok"] is None
