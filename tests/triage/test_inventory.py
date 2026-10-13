"""The inventory: run classes, read-only sqlite, product status per DAMNIT file."""

from __future__ import annotations

import os

import numpy as np
import pytest
from triage_fakes import run_row, write_damnit_file, write_sqlite

from analysis.triage.inventory import (
    build_inventory,
    classify,
    damnit_file,
    file_facts,
    jf_block,
    parse_run_length,
    read_run_table,
    stray_light_era,
)


@pytest.mark.parametrize(
    ("run", "run_type", "scan_type", "sample", "expected"),
    [
        (35, "Calibration - Dark HG", None, "none", "calib_dark"),
        (506, "Dark Burst Mode", None, "none", "calib_dark"),
        (58, "Test DAQ", None, "none", "test"),
        (24, "XPCS", "dmesh fsssx", "ferritin_peg_commissioning", "commissioning"),
        (
            30,
            "XPCS",
            "dmesh SSHEX_X",
            "ferritin50_peg6k",
            "commissioning",
        ),  # r <= 42 first
        (120, "XPCS", "dmesh SSHEX_X (-0.5 -> 0.5)", "ferritin50_peg6k", "mesh_sshex"),
        (318, "XPCS", "dmesh SSHEX_X", "ferritin50", "mesh_sshex"),
        (55, "XPCS", None, "air", "reference"),
        (383, "XPCS", None, "silica_np", "reference"),
        (423, "XPCS", None, "ferritin50_peg6k", "static_droplet"),
    ],
)
def test_classify_follows_the_rule_order(run, run_type, scan_type, sample, expected):
    assert classify(run, run_type, scan_type, sample) == expected


def test_stray_light_era_leaves_the_change_window_undetermined():
    assert stray_light_era(433) == "le_r433"
    assert stray_light_era(440) == "undetermined"
    assert stray_light_era(443) == "ge_r443"


def test_jf_block_names_the_readout_pattern():
    assert jf_block(np.arange(16)) == "all16"
    assert jf_block([15, 0, 1, 2, 3, 4, 5, 6]) == "cells0,1,2,3,4,5,6,15"
    assert jf_block(None) is None


def test_parse_run_length():
    assert parse_run_length("0:04:59") == 299.0
    assert parse_run_length("1:00:00") == 3600.0
    assert np.isnan(parse_run_length(None))
    assert np.isnan(parse_run_length(float("nan")))  # pandas' missing value


def test_read_run_table_works_on_a_read_only_database(tmp_path):
    db = write_sqlite(tmp_path / "runs.sqlite", [run_row(424), run_row(423)])
    os.chmod(db, 0o444)
    try:
        df = read_run_table(db)
    finally:
        os.chmod(db, 0o644)
    assert list(df.run) == [423, 424]
    assert "agipd_iq_overview" not in df.columns
    assert not (tmp_path / "runs.sqlite-journal").exists()


def test_file_facts_tells_present_from_empty_from_errored(tmp_path):
    path = write_damnit_file(
        tmp_path / "p10400_r120.h5",
        volume=np.linspace(2.0, 1.5, 100),
        xgm=np.r_[np.full(50, 300.0), np.full(50, 340.0)],
        jf_cells=list(range(16)),
        errors={"agipd_saxs": "This data has no source named 'AGIPD'.\nSee data."},
        empty=("xpcs_mean_dataset",),
    )
    facts = file_facts(path)
    assert facts["var_droplet_fit"] == "present"
    assert facts["var_xpcs_mean_dataset"] == "empty"
    assert facts["var_agipd_saxs"] == "errored"
    assert facts["errors"]["agipd_saxs"] == "This data has no source named 'AGIPD'."
    assert facts["var_xpcs_ttcf"] == "absent"
    assert facts["jf_block"] == "all16"
    assert facts["xgm_mean_uJ"] == pytest.approx(320.0)
    assert facts["xgm_cv"] == pytest.approx(20.0 / 320.0, rel=1e-5)


def test_an_all_nan_droplet_fit_is_empty(tmp_path):
    path = write_damnit_file(tmp_path / "f.h5", volume=np.full(100, np.nan))
    assert file_facts(path)["var_droplet_fit"] == "empty"


def test_a_missing_file_is_recorded_not_raised(tmp_path):
    facts = file_facts(tmp_path / "nope.h5")
    assert facts["var_agipd_saxs"] == "no_file" and facts["errors"] == {}


def test_build_inventory_applies_overrides_and_reads_each_file(tmp_path, damnit_root):
    rows = [
        run_row(289, sample="Silica NP", scan_type="dmesh SSHEX_X", n_trains=840),
        run_row(463),
        run_row(464),
        run_row(35, run_type="Calibration - Dark HG", sample="No Sample"),
    ]
    for r in (289, 463, 464):
        write_damnit_file(damnit_file(r, damnit_root), volume=np.full(100, 1.0))
    inv = build_inventory(
        read_run_table(write_sqlite(tmp_path / "db", rows)), damnit_root
    )
    inv = inv.set_index("run")
    assert inv.loc[289, "sample"] == "ferritin50_peg6k"
    assert inv.loc[289, "label"] == "Silica NP"
    assert inv.loc[289, "cls"] == "mesh_sshex"
    assert inv.loc[464, "background_role"] == "saxs_buffer_peg6k"
    assert inv.loc[463, "exclude"].startswith("explodes")
    assert inv.loc[35, "cls"] == "calib_dark"
    assert inv.loc[35, "var_droplet_fit"] == "no_file"
    assert inv.loc[464, "run_length_s"] == 300.0
    assert inv.loc[464, "start_utc"].tzinfo is not None
