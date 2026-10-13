"""Per-run XPCS SNR: the triage's reduction, β from a report, the verdict."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from triage_fakes import REP_RATE_HZ, write_xpcs_run

from analysis.diagnostics.xpcs import (
    KBAR_MIN,
    beta_by_roi,
    kbar_rois,
    load_beta,
    rep_rate_hz,
    xpcs_score,
    xpcs_snr,
)
from analysis.triage.xpcs import (
    OFF_SUBTRACTION,
    REFERENCE_BETA,
    REFERENCE_BETA_RUNS,
    measure_beta,
    measure_run,
)

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


def test_the_default_beta_is_the_reference_table(roots):
    damnit_root, saxs_root, _ = roots
    out = xpcs_snr(RUN, damnit_root=damnit_root, saxs_root=saxs_root)
    rois = out["rois"]
    for q, (value, _, _) in REFERENCE_BETA.items():
        row = rois[np.isclose(rois.roi_q_A, q, atol=1e-6)]
        assert row.beta.item() == value
        assert row.beta_n_eligible.item() == len(REFERENCE_BETA_RUNS)
    assert out["summary"]["beta_source"] == "REFERENCE_BETA"
    assert out["summary"]["beta_known"] is True


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


# ── k̄ per ROI and the per-run score ──────────────────────────────────────────


def write_kbar_nc(path, train_ids, kbar_per_roi, n_frames=10):
    """extra-speckle's kbar.nc: k̄ per (trainId, frame, roi_q)."""
    import xarray as xr

    values = np.broadcast_to(
        np.asarray(kbar_per_roi, float)[None, None, :],
        (len(train_ids), n_frames, len(kbar_per_roi)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.DataArray(
        values.copy(),
        dims=("trainId", "frame", "roi_q"),
        coords={
            "trainId": np.asarray(train_ids, dtype=np.uint64),
            "frame": np.arange(n_frames),
            "roi_q": [0.02875, 0.04, 0.0505, 0.0605, 0.067, 0.072, 0.0805],
        },
    ).to_netcdf(path, engine="h5netcdf")
    return path


def test_kbar_rois_counts_the_rois_above_the_threshold(tmp_path):
    tids = np.arange(100, 110)
    kbar = [0.02, 0.01, 0.004, 0.003, 0.0029, 0.001, 0.0005]
    write_kbar_nc(tmp_path / "r0423" / "kbar.nc", tids, kbar)
    table = kbar_rois(423, xpcs_root=tmp_path)
    np.testing.assert_allclose(table.kbar, kbar)
    assert list(table.above) == [True, True, True, True, False, False, False]
    assert table.attrs["n_above"] == 4
    assert table.attrs["n_plain_above"] == 3  # 0.0605 is a Bragg ROI
    assert table.attrs["kbar_min"] == KBAR_MIN == 3e-3
    assert table.attrs["n_trains"] == 10
    assert kbar_rois(423, tids[:3], xpcs_root=tmp_path).attrs["n_trains"] == 3


def test_kbar_rois_fall_back_to_our_geometry(roots, tmp_path):
    damnit_root, saxs_root, truth = roots
    table = kbar_rois(RUN, xpcs_root=tmp_path / "none", damnit_root=damnit_root,
                      saxs_root=saxs_root)  # fmt: skip
    assert "our geometry" in table.attrs["source"]
    np.testing.assert_allclose(table.kbar, truth["kbar"], rtol=1e-6)


def test_the_score_is_taken_over_the_hits(roots, tmp_path):
    import xarray as xr
    from triage_fakes import run_row, write_sqlite

    damnit_root, saxs_root, truth = roots
    path = damnit_root / "extracted_data" / f"p10400_r{RUN}.h5"
    with h5py.File(path, "r") as f:
        tids = f["agipd_saxs/trainId"][:]
    for group, name in (
        ("xgm_intensity", "Energy"),
        ("total_transmission", "__xarray_dataarray_variable__"),
    ):
        with h5py.File(path, "a") as f:
            if group in f:
                del f[group]
        xr.DataArray(np.ones(tids.size), dims="trainId", coords={"trainId": tids},
                     name=name).to_dataset().to_netcdf(
            path, group=group, mode="a", engine="h5netcdf")  # fmt: skip
    write_sqlite(damnit_root / "runs.sqlite", [run_row(RUN)])
    score = xpcs_score(RUN, damnit_root=damnit_root, saxs_root=saxs_root,
                       xpcs_root=tmp_path / "none")  # fmt: skip
    assert score["kind"] == "static"
    assert score["n_hits"] == tids.size  # a steady synthetic run never leaves
    assert score["snr"]["n_usable"] == tids.size
    expected = np.broadcast_to(np.asarray(truth["kbar"]), (7,)) >= KBAR_MIN
    assert score["n_rois_kbar_ok"] == int(expected.sum())
