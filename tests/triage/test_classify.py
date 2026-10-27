"""Droplet hits (`analysis.triage.classify`) against synthetic runs with known classes.

The synthetic mesh block mimics a measured one: a narrow off peak per run whose
level drifts from run to run, hits ~1.3 dex above it, a thin continuum of edge
trains between, partial overlaps carrying 15 to 60 % of a full hit's droplet
signal, hit fractions from 10 to 60 %, one run where the beam stayed on the droplet
and one whose single level matches nothing. The hits' spread of 0.05 dex is close
to r240's measured plateau width (robust σ 0.055 dex).
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import xarray as xr
from triage_fakes import (
    OFF_SD,
    ON_DEX,
    ON_SD,
    in_memory,
    mesh_run,
    partial,
    run_row,
    write_damnit,
    write_sqlite,
)

from analysis.triage.classify import (
    ON_FRACTION,
    block_cut,
    classify_block,
    hits,
    otsu_split,
    run_blocks,
    run_levels,
    signals_from,
    static_hits,
    train_signals,
    waxs_check,
)
from analysis.triage.inventory import read_run_table


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
        assert (cls[true == "on"] == "on").mean() > 0.98
        assert (cls[true == "off"] == "off").mean() > 0.99
        assert not (cls[true == "off"] == "on").any()
        assert (cls[true == "partial"] == "edge").all()
        assert out["runs"][r]["two_classes"]


def test_a_run_without_off_trains_is_all_hits(block):
    _, _, out = block
    v = out["runs"][106]
    assert not v["two_classes"] and not v["ambiguous"]
    assert v["reference"] == 105
    assert v["n_off"] == 0 and v["n_on"] > 0.98 * 700


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
    # a second mode only just above the off band is no mode either
    assert block_cut(np.r_[np.zeros(100), np.full(5, 0.11)], off_cut=0.1)["ambiguous"]


def test_the_overlap_fraction_is_measured_against_the_runs_full_hit(block):
    _, truth, out = block
    for r in range(100, 106):
        v, f, true = out["runs"][r], out["overlap"][r], truth[r]
        assert v["plateau_source"] == "run"
        assert v["plateau"] == pytest.approx(ON_DEX, abs=0.04)
        assert np.median(f[true == "on"]) == pytest.approx(1.0, abs=0.1)
        assert np.median(f[true == "off"]) == pytest.approx(0.0, abs=0.01)
        assert (f[true == "partial"] < ON_FRACTION).all()


def test_partial_overlaps_above_t_on_are_edge_not_hits():
    rng = np.random.default_rng(8)
    xs = {r: mesh_run(rng, -4.3, 0.3, partial_fraction=0.0)[0] for r in (200, 201)}
    clipped = partial(np.array([0.3, 0.5, 0.6, 0.9, 1.0]))
    xs[202] = np.r_[mesh_run(rng, -4.3, 0.3, partial_fraction=0.0)[0], -4.3 + clipped]
    out = classify_block(xs)
    assert (clipped > out["block"]["t_on"]).all()
    assert list(out["classes"][202][-5:]) == ["edge", "edge", "edge", "on", "on"]


def test_a_run_with_few_hits_takes_its_blocks_full_hit_level():
    rng = np.random.default_rng(9)
    xs = {r: mesh_run(rng, -4.3, 0.3)[0] for r in (300, 301, 302)}
    xs[303] = np.r_[-4.3 + OFF_SD * rng.standard_normal(790),
                    -4.3 + ON_DEX + ON_SD * rng.standard_normal(10)]  # fmt: skip
    out = classify_block(xs)
    v = out["runs"][303]
    assert v["plateau_source"] == "block"
    assert v["plateau"] == out["block"]["on_mode"]
    assert v["n_on"] >= 9


def test_static_hits_find_long_and_short_absences():
    rng = np.random.default_rng(4)
    n = 3000
    x = np.linspace(-3.0, -2.6, n) + 0.02 * rng.standard_normal(n)
    out = np.zeros(n, dtype=bool)
    # 200 trains: more than half the 301-train window, which a running median misses;
    # 260: more than 0.75 · 301, found only once the absence's edges, flagged on a
    # first pass, leave the reference
    for start, length in ((300, 20), (700, 60), (1300, 200), (2200, 260)):
        out[start : start + length] = True
    x[out] -= 1.3
    x[10] = np.nan
    on = static_hits(x)
    assert not on[out].any()
    assert on[~out & np.isfinite(x)].mean() > 0.999
    assert not on[10]


def test_waxs_z_is_the_shift_over_the_standard_error_of_a_median():
    off = np.tile([-0.1, 0.1], 200)  # median 0, MAD 0.1
    cls = np.array(["off"] * 400 + ["on"] * 100)
    v = waxs_check(np.r_[off, np.full(100, 0.3)], cls)
    assert v["waxs_shift_dex"] == pytest.approx(0.3)
    # the median of n normal values scatters by √(π/2) · σ / √n
    se = np.sqrt(np.pi / 2) * 1.4826 * 0.1 / np.sqrt(100)
    assert v["waxs_z"] == pytest.approx(0.3 / se, rel=1e-4)


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


def test_hits_end_to_end_on_a_small_block(mesh_store, tmp_path):
    h = hits(121, damnit_root=mesh_store, cache_dir=tmp_path / "cache")
    assert h["kind"] == "mesh" and h["block"] == "0.012|ferritin50_peg6k"
    assert 0.17 < h["train_ids"].size / 400 < 0.23
    assert set(np.unique(h["classes"].values)) <= {"on", "edge", "off"}
    assert (h["overlap"].sel(trainId=h["train_ids"]) >= ON_FRACTION).all()
    s = hits(423, damnit_root=mesh_store, cache_dir=tmp_path / "cache")
    assert s["kind"] == "static" and s["overlap"] is None
    assert s["summary"]["n_off"] == 20
    assert s["train_ids"].size == 600
    with pytest.raises(LookupError):
        hits(999, damnit_root=mesh_store, cache_dir=tmp_path / "cache")


def test_signals_from_memory_equal_those_from_the_file(tmp_path):
    root = tmp_path / "amore"
    (root / "extracted_data").mkdir(parents=True)
    rng = np.random.default_rng(8)
    n_frames = (rng.random((6, 3)) > 0.2).astype(float)
    path, _ = write_damnit(
        root / "extracted_data" / "p10400_r5.h5", rng.random((6, 3, 50)),
        n_frames=n_frames, xgm=rng.random(6) * 500, trans=rng.random(6) * 0.1,
        waxs=rng.random((6, 2, 20)),
    )  # fmt: skip
    want = train_signals(5, damnit_root=root, cache_dir=None)
    got = signals_from(*in_memory(path))
    for name in want.data_vars:
        np.testing.assert_array_equal(got[name].values, want[name].values, name)


def test_hits_take_this_run_from_memory_and_skip_members_not_yet_stored(
    mesh_store, tmp_path
):
    root = shutil.copytree(mesh_store, tmp_path / "amore")
    stored = hits(121, damnit_root=root, cache_dir=None)

    memory = in_memory(root / "extracted_data" / "p10400_r121.h5")
    # in the DAMNIT job, r121 itself is not in its file yet
    (root / "extracted_data" / "p10400_r121.h5").unlink()
    got = hits(121, damnit_root=root, cache_dir=None, signals=signals_from(*memory))
    np.testing.assert_array_equal(got["train_ids"], stored["train_ids"])
    assert got["summary"]["block_missing"] == []
    (root / "extracted_data" / "p10400_r122.h5").unlink()
    fewer = hits(121, damnit_root=root, cache_dir=None, signals=signals_from(*memory))
    assert fewer["summary"]["block_missing"] == [122]
    assert fewer["kind"] == "mesh" and fewer["train_ids"].size > 0


def test_a_neighbour_being_written_is_waited_for_not_left_out(mesh_store, monkeypatch):
    """A block member's file that is unreadable at first (its own job is writing
    it) is read again, and the block is classified with every member."""
    from analysis.triage import classify

    want = hits(121, damnit_root=mesh_store, cache_dir=None)
    monkeypatch.setattr(classify, "MEMBER_WAIT_S", 0.0)
    real, calls = classify.train_signals, {"n": 0}

    def flaky(run, **kwargs):
        if run == 122 and calls["n"] < 2:
            calls["n"] += 1
            raise OSError("bad object header version number")
        return real(run, **kwargs)

    monkeypatch.setattr(classify, "train_signals", flaky)
    got = hits(121, damnit_root=mesh_store, cache_dir=None)
    assert got["summary"]["block_missing"] == []
    np.testing.assert_array_equal(got["train_ids"], want["train_ids"])


def test_a_neighbour_that_stays_unreadable_fails_the_run(mesh_store, monkeypatch):
    """Not "missing": leaving a run out would change the cut of the whole block."""
    from analysis.triage import classify

    monkeypatch.setattr(classify, "MEMBER_WAIT_S", 0.0)
    real = classify.train_signals

    def broken(run, **kwargs):
        if run == 122:
            raise ValueError("agipd.saxs has no q bin in (0.3, 0.5)")
        return real(run, **kwargs)

    monkeypatch.setattr(classify, "train_signals", broken)
    with pytest.raises(ValueError, match="no q bin"):
        hits(121, damnit_root=mesh_store, cache_dir=None)
