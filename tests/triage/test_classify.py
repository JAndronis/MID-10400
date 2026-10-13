"""Droplet hits (`analysis.triage.classify`) against synthetic runs with known classes.

The synthetic mesh block mimics the measured one (2026-10-14): a narrow off peak per
run whose level drifts from run to run, hits ~1.3 dex above it, a thin continuum of
edge trains between, hit fractions from 10 to 60 %, one run where the beam stayed on
the droplet and one whose single level matches nothing.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest
import xarray as xr
from triage_fakes import FIRST_TRAIN, run_row, write_sqlite

from analysis.triage.classify import (
    block_cut,
    classify_block,
    hits,
    otsu_split,
    run_blocks,
    run_levels,
    static_hits,
    train_signals,
    waxs_check,
)
from analysis.triage.inventory import read_run_table

ON_DEX, OFF_SD, ON_SD = 1.3, 0.02, 0.1


def mesh_run(rng, off, hit_fraction, n=800, edge_fraction=0.02):
    """log10(s/F) of one mesh run, and its true classes."""
    n_on = int(hit_fraction * n)
    n_edge = int(edge_fraction * n)
    n_off = n - n_on - n_edge
    x = np.concatenate([
        off + OFF_SD * rng.standard_normal(n_off),
        off + ON_DEX + ON_SD * rng.standard_normal(n_on),
        off + rng.uniform(0.2, 0.5, n_edge),
    ])  # fmt: skip
    truth = np.array(["off"] * n_off + ["on"] * n_on + ["edge"] * n_edge)
    order = rng.permutation(n)
    return x[order], truth[order]


@pytest.fixture(scope="module")
def block():
    rng = np.random.default_rng(7)
    xs, truth = {}, {}
    for i, r in enumerate(range(100, 106)):
        xs[r], truth[r] = mesh_run(rng, -4.3 + 0.06 * i, 0.1 + 0.1 * i)
    # beam parked on the droplet: one level, the neighbours' hit level
    xs[106] = -4.0 + ON_DEX + ON_SD * rng.standard_normal(700)
    truth[106] = np.array(["on"] * 700)
    # one level, 0.6 dex below the neighbours' off level: neither class
    xs[107] = -4.6 + OFF_SD * rng.standard_normal(700)
    truth[107] = None
    return xs, truth, classify_block(xs)


def test_otsu_splits_between_two_modes():
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0, 0.02, 900), rng.normal(1.3, 0.1, 100)]
    assert 0.1 < otsu_split(x) < 1.1
    with pytest.raises(ValueError):
        otsu_split([np.nan, 1.0])


def test_run_levels_take_the_off_level_from_the_low_class():
    rng = np.random.default_rng(2)
    x, _ = mesh_run(rng, -4.3, 0.6)  # most trains hit: the run's median is no use
    v = run_levels(x)
    assert v["off"] == pytest.approx(-4.3, abs=0.01)
    assert v["sigma"] == pytest.approx(OFF_SD, rel=0.2)
    assert v["high"] == pytest.approx(-4.3 + ON_DEX, abs=0.05)


def test_block_classes_match_the_truth(block):
    xs, truth, out = block
    assert not out["block"]["ambiguous"]
    assert 5 * OFF_SD < out["block"]["t_on"] < ON_DEX
    for r in range(100, 106):
        cls, true = out["classes"][r], truth[r]
        assert (cls[true == "on"] == "on").mean() > 0.99
        assert (cls[true == "off"] == "off").mean() > 0.99
        assert not (cls[true == "off"] == "on").any()
        assert out["runs"][r]["two_classes"]


def test_a_run_without_off_trains_is_all_hits(block):
    _, _, out = block
    v = out["runs"][106]
    assert not v["two_classes"] and not v["ambiguous"]
    assert v["reference"] == 105
    assert v["n_on"] == 700


def test_a_level_that_matches_nothing_is_ambiguous(block):
    _, _, out = block
    v = out["runs"][107]
    assert v["ambiguous"] and v["n_on"] == 0
    assert (out["classes"][107] == "none").all()


def test_a_block_without_a_second_mode_is_ambiguous():
    rng = np.random.default_rng(3)
    xs = {r: -2.85 + 0.024 * rng.standard_normal(800) for r in (68, 69)}
    out = classify_block(xs)
    assert out["block"]["ambiguous"]
    assert all(v["n_on"] == 0 and v["ambiguous"] for v in out["runs"].values())


def test_block_cut_needs_a_mode_beyond_the_off_band():
    rel = np.r_[np.zeros(100), np.full(5, 0.11)]
    assert block_cut(rel, off_cut=0.1)["ambiguous"]


def test_static_hits_find_long_and_short_absences():
    rng = np.random.default_rng(4)
    n = 3000
    x = np.linspace(-3.0, -2.6, n) + 0.02 * rng.standard_normal(n)
    out = np.zeros(n, dtype=bool)
    # 60 and 150 trains: longer than half of §4.3's 51-train window, which misses them
    for start, length in ((500, 20), (1500, 60), (2200, 150)):
        out[start : start + length] = True
    x[out] -= 1.3
    x[10] = np.nan
    on = static_hits(x)
    assert not on[out].any()
    assert on[~out & np.isfinite(x)].mean() > 0.999
    assert not on[10]


def test_waxs_check_measures_the_shift_of_the_hits():
    rng = np.random.default_rng(5)
    cls = np.array(["off"] * 700 + ["on"] * 100)
    w = np.r_[rng.normal(0, 0.07, 700), rng.normal(0.33, 0.07, 100)]
    v = waxs_check(w, cls)
    assert v["waxs_shift_dex"] == pytest.approx(0.33, abs=0.03)
    assert v["waxs_z"] > 20
    assert np.isnan(waxs_check(w, np.array(["off"] * 800))["waxs_z"])


def test_run_blocks_key_mesh_runs_by_attenuator_and_sample(tmp_path):
    rows = [
        run_row(120, scan_type="dmesh SSHEX", total_transmission=0.0123,
                pulses=350.0),
        run_row(318, scan_type="dmesh SSHEX", total_transmission=0.1006,
                sample="Ferritin_50"),
        run_row(423),
    ]  # fmt: skip
    table = run_blocks(read_run_table(write_sqlite(tmp_path / "runs.sqlite", rows)))
    assert table.loc[120, "cls"] == "mesh_sshex"
    assert table.loc[120, "block"] == "0.012|ferritin50_peg6k"
    assert table.loc[318, "block"].startswith("0.1|")
    assert not isinstance(table.loc[423, "block"], str)
    assert table.loc[423, "cls"] == "static_droplet"


# ── DAMNIT files ──────────────────────────────────────────────────────────────


def write_damnit(path, saxs, *, n_frames=None, xgm=None, trans=None, waxs=None):
    """agipd_saxs (train, pulse, q), xgm_intensity, total_transmission, jf1."""
    n, p, nq = saxs.shape
    tids = FIRST_TRAIN + np.arange(n, dtype=np.uint64)
    q = np.linspace(0.08, 1.06, nq)
    with h5py.File(path, "w") as f:
        g = f.create_group("agipd_saxs")
        g["trainId"], g["q"] = tids, q
        g["pulseId"] = np.arange(p, dtype=np.uint64)
        g["intensity"] = saxs.astype(np.float32)
        g["n_frames"] = (np.ones((n, p)) if n_frames is None else n_frames).astype(
            np.uint8
        )
        if waxs is not None:
            j = f.create_group("jungfrau_waxs_jf1")
            j["trainId"], j["q"] = tids, np.linspace(11.5, 23.7, waxs.shape[2])
            j["cellId"] = np.arange(waxs.shape[1], dtype=np.uint16)
            j["intensity"] = waxs.astype(np.float32)
            j["n_frames"] = np.ones(waxs.shape[:2], np.uint8)
    for group, name, values in (
        ("xgm_intensity", "Energy", np.full(n, 500.0) if xgm is None else xgm),
        ("total_transmission", "__xarray_dataarray_variable__",
         np.full(n, 0.1) if trans is None else trans),
    ):  # fmt: skip
        xr.DataArray(
            np.asarray(values, float),
            dims="trainId",
            coords={"trainId": tids},
            name=name,
        ).to_dataset().to_netcdf(path, group=group, mode="a", engine="h5netcdf")
    return path, q


def test_train_signals_average_the_band_over_present_frames(tmp_path):
    root = tmp_path / "amore"
    (root / "extracted_data").mkdir(parents=True)
    saxs = np.zeros((4, 3, 50))
    saxs[:, 0], saxs[:, 1], saxs[:, 2] = 1.0, 3.0, 100.0
    n_frames = np.ones((4, 3))
    n_frames[:, 2] = 0  # an empty slot is stored as a value and must not count
    n_frames[3] = 0  # a train with no frame
    path, _ = write_damnit(
        root / "extracted_data" / "p10400_r5.h5", saxs, n_frames=n_frames,
        waxs=np.full((4, 2, 20), 7.0),
    )  # fmt: skip
    ds = train_signals(5, damnit_root=root, cache_dir=tmp_path / "cache")
    np.testing.assert_allclose(ds.saxs_mid.values[:3], 2.0)
    assert np.isnan(ds.saxs_mid.values[3])
    np.testing.assert_allclose(ds.flux.values, 50.0)
    np.testing.assert_allclose(ds.waxs_ring.values, 7.0)
    # a second call is served from the cache, a changed source is not
    again = train_signals(5, damnit_root=root, cache_dir=tmp_path / "cache")
    xr.testing.assert_identical(ds.saxs_mid, again.saxs_mid)
    saxs[:, 0] = 5.0
    write_damnit(path, saxs, n_frames=n_frames)
    os.utime(path, ns=(1, 1))
    fresh = train_signals(5, damnit_root=root, cache_dir=tmp_path / "cache")
    np.testing.assert_allclose(fresh.saxs_mid.values[:3], 4.0)
    assert np.isnan(fresh.waxs_ring.values).all()  # no jf1 in the new file


def test_hits_end_to_end_on_a_small_block(tmp_path):
    rng = np.random.default_rng(6)
    root = tmp_path / "amore"
    (root / "extracted_data").mkdir(parents=True)
    rows = []
    for r in (120, 121, 122):
        x, _ = mesh_run(rng, -4.3, 0.2, n=400)
        level = 10.0**x  # s/F with F = 1 (xgm 10 × transmission 0.1)
        saxs = np.broadcast_to(level[:, None, None], (400, 2, 30)).copy()
        write_damnit(root / "extracted_data" / f"p10400_r{r}.h5", saxs,
                     xgm=np.full(400, 10.0))  # fmt: skip
        rows.append(run_row(r, scan_type="dmesh SSHEX", total_transmission=0.0123,
                            pulses=350.0))  # fmt: skip
    rows.append(run_row(423))
    level = 10.0 ** np.r_[np.full(300, -3.0), np.full(20, -4.3), np.full(300, -3.0)]
    write_damnit(root / "extracted_data" / "p10400_r423.h5",
                 np.broadcast_to(level[:, None, None], (620, 2, 30)).copy(),
                 xgm=np.full(620, 10.0))  # fmt: skip
    write_sqlite(root / "runs.sqlite", rows)

    h = hits(121, damnit_root=root, cache_dir=tmp_path / "cache")
    assert h["kind"] == "mesh" and h["block"] == "0.012|ferritin50_peg6k"
    assert 0.17 < h["train_ids"].size / 400 < 0.23
    assert set(np.unique(h["classes"].values)) <= {"on", "edge", "off"}
    s = hits(423, damnit_root=root, cache_dir=tmp_path / "cache")
    assert s["kind"] == "static"
    assert s["summary"]["n_off"] == 20
    assert s["train_ids"].size == 600
    with pytest.raises(LookupError):
        hits(999, damnit_root=root, cache_dir=tmp_path / "cache")
