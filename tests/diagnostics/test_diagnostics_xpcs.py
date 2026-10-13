"""Per-run XPCS SNR: the triage's reduction, β from a report, the verdict."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from triage_fakes import REP_RATE_HZ, write_xpcs_run

from analysis.diagnostics.xpcs import beta_by_roi, load_beta, rep_rate_hz, xpcs_snr
from analysis.triage.xpcs import OFF_SUBTRACTION, measure_beta, measure_run

RUN = 423


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    """DAMNIT and SAXS roots laid out as on GPFS, holding one synthetic run."""
    base = tmp_path_factory.mktemp("roots")
    damnit = base / "amore" / "extracted_data" / f"p10400_r{RUN}.h5"
    scratch = base / "agipd_saxs" / f"r{RUN:04d}" / "agipd_saxs.h5"
    damnit.parent.mkdir(parents=True)
    scratch.parent.mkdir(parents=True)
    truth = write_xpcs_run(damnit, scratch, np.random.default_rng(5))
    with h5py.File(damnit, "a") as f:
        f.create_group("rep_rate")["data"] = REP_RATE_HZ / 1e6
    return base / "amore", base / "agipd_saxs", truth


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


def test_rep_rate_is_read_in_hz(roots):
    damnit_root, _, _ = roots
    path = damnit_root / "extracted_data" / f"p10400_r{RUN}.h5"
    assert rep_rate_hz(path) == pytest.approx(REP_RATE_HZ)


def test_snr_with_a_measured_beta(roots, tmp_path):
    damnit_root, saxs_root, truth = roots
    path = damnit_root / "extracted_data" / f"p10400_r{RUN}.h5"
    m = measure_run(RUN, path, saxs_root / f"r{RUN:04d}" / "agipd_saxs.h5", REP_RATE_HZ)
    report = tmp_path / "report.json"
    report.write_text(json.dumps(report_for(m)))
    out = xpcs_snr(RUN, load_beta(report), damnit_root=damnit_root, saxs_root=saxs_root)

    rois = out["rois"]
    b0 = rois.beta[0]
    assert b0 == pytest.approx(truth["beta"], rel=0.1)
    expected = b0 * m.kbar[0] * np.sqrt(m.npix[0] * (m.n_frames - 1) * m.phi[0] * 200)
    assert rois.snr_run[0] == pytest.approx(expected / OFF_SUBTRACTION)
    assert out["summary"]["beta_known"] is True
    assert out["summary"]["extractable"] == (out["summary"]["best_snr"] >= 3)
    assert list(out["slices"].columns[:4]) == [
        "slice",
        "start_s",
        "n_usable",
        "best_snr",
    ]
    assert out["slices"].n_usable.sum() == 200


def test_an_unknown_beta_gives_no_verdict(roots):
    damnit_root, saxs_root, _ = roots
    out = xpcs_snr(RUN, {}, damnit_root=damnit_root, saxs_root=saxs_root)
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


def test_a_run_without_ttcf_is_refused(roots, tmp_path):
    damnit_root, saxs_root, _ = roots
    src = damnit_root / "extracted_data" / f"p10400_r{RUN}.h5"
    other = tmp_path / "amore" / "extracted_data"
    other.mkdir(parents=True)
    with h5py.File(src, "r") as a, h5py.File(other / f"p10400_r{RUN}.h5", "w") as b:
        for name in a:
            if name != "xpcs_ttcf":
                a.copy(name, b)
    with pytest.raises(LookupError, match="no xpcs_ttcf"):
        xpcs_snr(RUN, {}, damnit_root=tmp_path / "amore", saxs_root=saxs_root)
