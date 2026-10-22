"""Droplet series: which run boundaries start a new droplet, and per-run metrics.

One scripted history covers every boundary case the series rule distinguishes:
evaporation across runs, a pause, a run without a fit, a small fit glitch upwards,
a top-up injection, a fresh injection, a calibration run in between, a noisy small
droplet, a change of sample and a jump inside a run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from triage_fakes import write_damnit_file

from analysis.triage.series import (
    WINDOW,
    Droplet,
    in_run_jump,
    load_droplet,
    run_noise,
    segment,
    window_medians,
)
from analysis.triage.stats import mad_dof, z_for_tests

N = 1000  # trains per run: ~50 windows, as in a real static run


def droplet(run, lv_start, lv_end, noise, rng, *, jump_at=None, jump=0.0):
    lv = np.linspace(lv_start, lv_end, N) + rng.normal(0.0, noise, N)
    if jump_at is not None:
        lv[jump_at:] += jump
    centre = np.c_[100 + rng.normal(0, 1, N), 80 + rng.normal(0, 2, N)]
    return Droplet(run, np.arange(N, dtype=np.uint64), 10**lv, centre, N)


@pytest.fixture(scope="module")
def history():
    rng = np.random.default_rng(7)
    lg = np.log10
    spec = [  # run, class, sample, droplet or None
        (100, "static_droplet", "ferritin50_peg6k", (lg(2.0), lg(1.5), 0.004)),
        (101, "static_droplet", "ferritin50_peg6k", (lg(1.5), lg(1.2), 0.004)),
        (102, "static_droplet", "ferritin50_peg6k", None),  # no fit
        (103, "static_droplet", "ferritin50_peg6k", (lg(1.15), lg(0.9), 0.004)),
        (104, "static_droplet", "ferritin50_peg6k", (lg(0.9) + 0.05, lg(0.8), 0.004)),
        (105, "static_droplet", "ferritin50_peg6k", (lg(0.8) + 0.2, lg(0.9), 0.004)),
        (106, "calib_dark", "none", None),
        (107, "static_droplet", "ferritin50_peg6k", (lg(0.9) + 0.6, lg(3.0), 0.004)),
        (108, "static_droplet", "ferritin50_peg6k", (lg(3.0) + 0.01, lg(2.8), 0.2)),
        (109, "static_droplet", "ferritin50", (lg(2.8), lg(2.6), 0.004)),
        (110, "mesh_sshex", "ferritin50", (lg(2.6) + 0.7, lg(10.0), 0.004)),
        (111, "mesh_sshex", "ferritin50", (lg(10.0), lg(9.0), 0.004)),
        (112, "mesh_sshex", "ferritin50", None),
        (113, "reference", "air", None),  # not a droplet sample
    ]
    inv = pd.DataFrame([{"run": r, "cls": c, "sample": s} for r, c, s, _ in spec])
    drops = {}
    for r, _, _, d in spec:
        if d is None:
            continue
        steps = {  # on a window edge: an injection-sized jump, and a glitch
            111: {"jump_at": 25 * WINDOW, "jump": 0.3},
            107: {"jump_at": 25 * WINDOW, "jump": 0.03},
        }
        kw = steps.get(r, {})
        drops[r] = droplet(r, *d, rng, **kw)
    return segment(inv, drops)


def test_boundary_kinds(history):
    kinds = dict(zip(history.boundaries.after, history.boundaries.kind, strict=True))
    assert kinds == {
        101: "continuing",
        103: "continuing",  # across the unfitted r102
        104: "increase",  # +0.05: significant, but a glitch
        105: "fresh",  # +0.2: a top-up
        107: "fresh",  # across the calibration run r106
        108: "continuing",  # +0.01 against 0.2 noise
        109: "sample_change",
        110: "fresh",
        111: "continuing",
    }


def test_a_label_change_without_a_volume_step_is_suspect(history):
    b = history.boundaries.set_index("after")
    assert bool(b.loc[109, "label_suspect"])  # same droplet, new label
    assert not b.loc[[105, 107, 110], "label_suspect"].any()


def test_a_checked_label_change_carries_its_verdict():
    rng = np.random.default_rng(3)
    inv = pd.DataFrame(
        {"run": [1, 2], "cls": ["reference", "static_droplet"],
         "sample": ["silica_np", "ferritin50"]}
    )  # fmt: skip
    drops = {1: droplet(1, 0.5, 0.35, 0.004, rng), 2: droplet(2, 0.34, 0.3, 0.004, rng)}
    unchecked = segment(inv, drops, checked={}).boundaries.iloc[0]
    assert unchecked.label_suspect and unchecked.label_checked is None
    verdict = "labels correct: a new droplet of similar volume"
    b = segment(inv, drops, checked={(1, 2): verdict}).boundaries.iloc[0]
    assert b.label_suspect and b.label_checked == verdict and b.kind == "sample_change"


def test_a_boundary_too_noisy_to_see_an_injection_is_underpowered(history):
    b = history.boundaries.set_index("after")
    assert bool(b.loc[108, "underpowered"])  # r108 scatters 0.2 dex per train
    # only a "continuing" verdict can lack power; r109's noisy side does not count
    assert not b.loc[[101, 103, 104, 105, 109, 110, 111], "underpowered"].any()


def test_the_fresh_cut_sits_in_the_empty_band_of_significant_increases(history):
    t = history.thresholds
    assert 0.05 < t["d_fresh"] < 0.2
    assert t["n_boundaries"] == 9 and t["n_significant_increases"] == 4


def test_series_membership(history):
    s = history.series.set_index("series_id")
    assert s.loc["s100", "runs"] == [100, 101, 102, 103, 104]
    assert s.loc["s100", "no_fit_runs"] == [102]
    assert s.loc["s100", "n_increase"] == 1
    assert s.loc["s105", "runs"] == [105]
    assert s.loc["s107", "runs"] == [107, 108]
    assert s.loc["s109", "start_kind"] == "sample_change"
    assert s.loc["s110", "runs"] == [110, 111, 112]
    expected = {100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112}
    assert set(history.runs.run) == expected


def test_v_over_v0_is_relative_to_the_series_start(history):
    r = history.runs.set_index("run")
    assert r.loc[100, "v_over_v0_start"] == pytest.approx(1.0)
    assert r.loc[103, "v_over_v0_end"] == pytest.approx(0.45, rel=0.03)
    assert r.loc[101, "v_over_v0_start"] == pytest.approx(0.75, rel=0.03)


def test_a_jump_inside_a_run_is_sized_against_d_fresh(history):
    r = history.runs.set_index("run")
    assert bool(r.loc[111, "volume_jump"]) and not r.loc[111, "volume_glitch"]
    assert r.loc[111, "max_jump_dlog"] == pytest.approx(0.3, abs=0.02)
    assert bool(r.loc[107, "volume_glitch"]) and not r.loc[107, "volume_jump"]
    clean = r.loc[[100, 101, 103, 104, 108, 109, 110], "max_jump_z"]
    assert r.loc[111, "max_jump_z"] > 10 * clean.max()


def test_the_jump_cut_holds_its_designed_false_count_on_clean_runs():
    """A triage of 10 clean runs should raise 0.1 false jump flags; this is a rate.

    One draw of a few runs cannot show it (seed 7 of the history above raises two,
    a 1-in-60 case), so it is measured over 1000 clean runs.
    """
    rng = np.random.default_rng(11)
    n_diffs = N // WINDOW - 1
    cut = z_for_tests(10 * n_diffs, two_sided=True, dof=mad_dof(n_diffs))
    false = 0
    for _ in range(1000):
        lv = np.linspace(0.3, 0.2, N) + rng.normal(0.0, 0.004, N)
        false += in_run_jump(lv, run_noise(lv))[1] > cut
    assert false / 1000 * 10 <= 0.15  # designed for 0.1; this seed gives 0.085


def test_centre_jitter_is_in_micrometres(history):
    r = history.runs.set_index("run")
    assert r.loc[100, "center_jitter_x_um"] == pytest.approx(13.9, rel=0.15)
    assert r.loc[100, "center_jitter_y_um"] == pytest.approx(27.8, rel=0.15)


def test_an_unfitted_run_before_a_fresh_start_is_ambiguous():
    rng = np.random.default_rng(1)
    inv = pd.DataFrame(
        {"run": [1, 2, 3], "cls": ["static_droplet"] * 3, "sample": ["ferritin50"] * 3}
    )
    drops = {
        1: droplet(1, 0.3, 0.2, 0.004, rng),
        3: droplet(3, 0.9, 0.8, 0.004, rng),
    }
    runs = segment(inv, drops).runs.set_index("run")
    assert runs.loc[2, "member"] == "ambiguous" and pd.isna(runs.loc[2, "series_id"])


def test_window_medians_and_noise_on_known_data():
    x = np.arange(65, dtype=float)
    np.testing.assert_array_equal(window_medians(x, 20), [9.5, 29.5, 49.5])
    assert np.isnan(run_noise(np.zeros(3 * WINDOW)))  # two differences only
    jump, z, n = in_run_jump(np.r_[np.zeros(100), np.ones(100)], sigma=0.1)
    assert (jump, n) == (1.0, 9) and z == pytest.approx(10.0)


def test_load_droplet_drops_non_finite_fits_and_orders_by_train(tmp_path):
    volume = np.r_[np.nan, np.linspace(2.0, 1.0, 99)]
    centre = np.c_[np.arange(100.0), 2 * np.arange(100.0)]
    path = write_damnit_file(tmp_path / "f.h5", volume=volume, center=centre)
    d = load_droplet(path, 5)
    assert d.n_rows == 100 and d.volume_mm3.size == 99
    assert d.coverage == pytest.approx(0.99)
    np.testing.assert_allclose(d.center_px[0], [1.0, 2.0])
    assert load_droplet(tmp_path / "missing.h5", 5) is None
