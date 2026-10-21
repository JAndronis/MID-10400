"""The on-disk cache in `analysis.cache`.

A fake DAMNIT database whose runs point at small netCDF files (``agipd_saxs`` and
``droplet_fit`` groups), fake raw runs that hand out labelled train timestamps, and a
stand-in for the XGM. The cache must store what the sources hold, averaged over pulses
and nothing else, label everything by trainId, and rebuild exactly when its inputs or
sources change.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("extra_data")  # analysis.utils imports it at module level
pytest.importorskip("h5netcdf")  # the cache and the fake DAMNIT files are netCDF

sys.path.insert(0, str(Path(__file__).parent / "xpcs"))  # xpcs_fakes lives there

from xpcs_fakes import (  # noqa: E402
    build_results,
    make_dense,
    make_flags,
    make_rings,
    make_train,
)

from analysis import cache, utils  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.common.writer import IncompleteRun  # noqa: E402
from analysis.utils import deff_sq_to_volume  # noqa: E402
from analysis.xpcs.result import Averaging, XpcsHits  # noqa: E402

FIRST_TRAIN = 2637862276


def train_ids(n: int, offset: int = 0) -> np.ndarray:
    return FIRST_TRAIN + offset + np.arange(n, dtype=np.uint64)


class FakeVariable:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value


class FakeRun:
    """What `damnit.Damnit()[run]` offers: a `.file` and variables by name."""

    def __init__(self, file, total_transmission):
        self.file = str(file)
        self.variables = {"total_transmission": FakeVariable(total_transmission)}

    def __getitem__(self, name):
        return self.variables[name]


class FakeRaw:
    """A raw run: labelled train timestamps (tz-aware UTC, as EXtra-data) and an XGM."""

    def __init__(self, times: pd.Series, xgm: xr.DataArray | None = None):
        self.times, self.xgm = times, xgm

    def train_timestamps(self, labelled=False):
        assert labelled, "timestamps must be taken by label"
        return self.times


def timestamps(tids, start="2026-05-12T00:32:00", step_s=0.1) -> pd.Series:
    """UTC timestamps for `tids`, `step_s` apart, handed out in scrambled order."""
    times = pd.Series(
        pd.date_range(
            start, periods=len(tids), freq=pd.Timedelta(seconds=step_s), tz="UTC"
        ),
        index=np.asarray(tids, dtype=np.uint64),
    )
    return times.iloc[np.random.default_rng(3).permutation(len(times))]


def damnit_run(path, tids, volume, *, transmission=None, grid=None):
    """One fake DAMNIT run file: droplet_fit (and agipd_saxs, if `grid` is given)."""
    radius_x = 60.0 - 0.1 * np.arange(len(tids))
    fit = xr.Dataset(
        {
            "volume": ("trainId", np.asarray(volume, dtype=np.float32)),
            "radius": (
                ("trainId", "dim"),
                np.c_[radius_x, 0.65 * radius_x].astype(np.float32),
            ),
            "center": (("trainId", "dim"), np.full((len(tids), 2), 100.0, np.float32)),
        },
        coords={"trainId": tids, "dim": ["x", "y"]},
        attrs={"pixel_size_um": 13.9},
    )
    fit.to_netcdf(path, group="droplet_fit", engine="h5netcdf")
    if grid is not None:
        grid.to_netcdf(path, group="agipd_saxs", mode="a", engine="h5netcdf")
    if transmission is None:
        transmission = np.ones(len(tids))
    return FakeRun(
        path, xr.DataArray(transmission, dims="trainId", coords={"trainId": tids})
    )


@pytest.fixture
def xgm_by_raw(monkeypatch):
    """`train_pulse_energy(raw, n)` returns the raw run's own fake XGM series."""
    calls = []

    def fake(raw, n_pulses):
        calls.append(n_pulses)
        return raw.xgm

    monkeypatch.setattr(utils, "train_pulse_energy", fake)
    return calls


# ── saxs_run ──────────────────────────────────────────────────────────────────
PULSE_IDS = np.array([8, 2, 6, 4])  # stored out of order: "first n" means by pulseId


@pytest.fixture
def saxs(tmp_path, xgm_by_raw):
    """Run 1: six trains, four pulses, five q bins.

    Train 2 is missing its pulseId-4 frame (stored as I = 0, as DAMNIT does), train 5
    has no frames at all, the XGM has no value for train 3, and the timestamps come
    scrambled with one train the grid does not have.
    """
    tids = train_ids(6)
    rng = np.random.default_rng(1)
    intensity = rng.uniform(1.0, 2.0, (6, 4, 5)).astype(np.float32)
    n_frames = np.ones((6, 4), np.uint8)
    n_frames[2, list(PULSE_IDS).index(4)] = 0
    n_frames[5] = 0
    intensity[n_frames == 0] = 0.0
    grid = xr.Dataset(
        {"intensity": (("trainId", "pulseId", "q"), intensity)},
        coords={
            "trainId": tids,
            "pulseId": PULSE_IDS.astype(np.uint64),
            "q": np.linspace(0.08, 1.0, 5),
            "n_frames": (("trainId", "pulseId"), n_frames),
        },
    )
    db = {1: damnit_run(tmp_path / "r1.h5", tids, 2.2 - 0.01 * np.arange(6), grid=grid)}
    xgm = xr.DataArray(300.0 + np.arange(6.0), dims="trainId", coords={"trainId": tids})
    raw = FakeRaw(
        timestamps(np.r_[tids, train_ids(1, 6)]), xgm.drop_sel(trainId=tids[3])
    )
    return {"db": db, "raw": raw, "tids": tids, "grid": grid, "dir": tmp_path / "cache"}


def expected_mean(grid, pulse_ids):
    sel = grid.sel(pulseId=pulse_ids)
    with np.errstate(invalid="ignore"):
        return sel.intensity.where(sel.n_frames > 0).mean("pulseId").values


def test_intensity_is_the_mean_over_the_frames_a_train_has(saxs):
    ds = cache.saxs_run(
        1, saxs["db"], raw=saxs["raw"], n_pulses=4, cache_dir=saxs["dir"]
    )

    want = expected_mean(saxs["grid"], [2, 4, 6, 8])
    np.testing.assert_allclose(ds.intensity.values, want, rtol=1e-6)
    assert np.all(np.isnan(ds.intensity.sel(trainId=saxs["tids"][5])))
    np.testing.assert_array_equal(ds.n_pulses_with_frames, [4, 4, 3, 4, 4, 0])
    # the zero stored for the missing frame is not averaged in
    assert ds.intensity.sel(trainId=saxs["tids"][2]).min() >= 1.0


def test_only_the_first_n_pulse_ids_are_averaged(saxs):
    ds = cache.saxs_run(
        1, saxs["db"], raw=saxs["raw"], n_pulses=3, cache_dir=saxs["dir"]
    )

    np.testing.assert_allclose(
        ds.intensity, expected_mean(saxs["grid"], [2, 4, 6]), rtol=1e-6
    )
    with pytest.raises(ValueError, match="fewer than n_pulses=5"):
        cache.saxs_run(
            1, saxs["db"], raw=saxs["raw"], n_pulses=5, cache_dir=saxs["dir"]
        )


def test_the_scaling_is_stored_by_train_id_and_never_applied(saxs):
    ds = cache.saxs_run(
        1, saxs["db"], raw=saxs["raw"], n_pulses=4, cache_dir=saxs["dir"]
    )
    scaling = utils.droplet_scaling(1, saxs["db"], raw=saxs["raw"], n_pulses=4)

    tids = saxs["tids"]
    kept = tids[[0, 1, 2, 4, 5]]  # train 3 has no XGM
    np.testing.assert_allclose(
        ds.scale.sel(trainId=kept), scaling.scale.sel(trainId=kept)
    )
    np.testing.assert_allclose(ds.phi_ferritin.sel(trainId=kept), scaling.phi_ferritin)
    assert np.isnan(ds.scale.sel(trainId=tids[3]))
    # train 3 keeps its intensity, unscaled like every other train
    np.testing.assert_allclose(
        ds.intensity.sel(trainId=tids[3]),
        expected_mean(saxs["grid"], [2, 4, 6, 8])[3],
        rtol=1e-6,
    )
    assert ds.attrs["v0_mm3"] == pytest.approx(scaling.attrs["v0_mm3"])


def test_timestamps_follow_train_ids(saxs):
    ds = cache.saxs_run(
        1, saxs["db"], raw=saxs["raw"], n_pulses=4, cache_dir=saxs["dir"]
    )

    by_label = saxs["raw"].times.dt.tz_localize(None).loc[saxs["tids"]].to_numpy()
    np.testing.assert_array_equal(ds.time.values, by_label.astype("datetime64[ns]"))


@pytest.fixture
def count_builds(monkeypatch):
    """Count the builds; each calls droplet_scaling exactly once."""
    calls = []
    real = utils.droplet_scaling

    def counting(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(utils, "droplet_scaling", counting)
    return calls


def test_a_repeat_call_is_served_from_the_file(saxs, count_builds):
    first = cache.saxs_run(
        1, saxs["db"], raw=saxs["raw"], cache_dir=saxs["dir"], n_pulses=4
    )
    second = cache.saxs_run(
        1, saxs["db"], raw=saxs["raw"], cache_dir=saxs["dir"], n_pulses=4
    )

    assert len(count_builds) == 1
    xr.testing.assert_allclose(first.intensity, second.intensity)
    assert second.attrs["created_utc"] == first.attrs["created_utc"]
    assert len(list(saxs["dir"].glob("saxs_r0001_*.nc"))) == 1


def test_the_same_value_as_int_or_float_is_the_same_file(saxs, count_builds):
    kw = {"raw": saxs["raw"], "cache_dir": saxs["dir"]}
    cache.saxs_run(1, saxs["db"], n_pulses=4, ferritin_mg_per_ml_initial=0, **kw)
    cache.saxs_run(
        np.int64(1), saxs["db"], n_pulses=4.0, ferritin_mg_per_ml_initial=0.0, **kw
    )

    assert len(count_builds) == 1
    assert len(list(saxs["dir"].glob("saxs_r0001_*.nc"))) == 1


def test_new_inputs_a_changed_source_or_refresh_rebuild(saxs, count_builds):
    kw = {"raw": saxs["raw"], "cache_dir": saxs["dir"], "n_pulses": 4}
    cache.saxs_run(1, saxs["db"], **kw)
    cache.saxs_run(1, saxs["db"], ferritin_mg_per_ml_initial=0.0, **kw)
    assert len(count_builds) == 2
    assert len(list(saxs["dir"].glob("saxs_r0001_*.nc"))) == 2  # one file per input set

    source = saxs["db"][1].file
    stat = os.stat(source)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    cache.saxs_run(1, saxs["db"], **kw)
    assert len(count_builds) == 3

    cache.saxs_run(1, saxs["db"], refresh=True, **kw)
    assert len(count_builds) == 4


def test_a_failed_write_leaves_no_file_behind(saxs, monkeypatch):
    def broken(self, path, **kwargs):
        open(path, "wb").write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", broken)
    with pytest.raises(OSError, match="disk full"):
        cache.saxs_run(
            1, saxs["db"], raw=saxs["raw"], n_pulses=4, cache_dir=saxs["dir"]
        )
    assert list(saxs["dir"].iterdir()) == []


# ── waxs_run ──────────────────────────────────────────────────────────────────
JF2_SCALE = 0.8  # jf2 reads this much of jf1; the fitted factor must undo it


def truth(q):
    return 100.0 / q**2 + 0.5


def jf_grid(q, lo, hi, tids, scale, units="photons"):
    """A per-cell grid as DAMNIT stores it: zeros outside [lo, hi] and for empty slots.

    Train 2 lacks its last cell, train 5 has no frame at all.
    """
    rng = np.random.default_rng(int(lo))
    wave = 1 + 0.1 * np.sin(np.arange(len(tids)))[:, None, None]  # per-train drift
    noise = 1 + 0.01 * rng.standard_normal((len(tids), 3, q.size))
    values = scale * truth(q) * wave * noise
    values[:, :, (q < lo) | (q > hi)] = 0.0
    n_frames = np.ones((len(tids), 3), np.uint8)
    n_frames[2, 2] = 0
    n_frames[5] = 0
    values[n_frames == 0] = 0.0
    return xr.DataArray(
        values.astype(np.float32),
        dims=("trainId", "cellId", "q"),
        coords={
            "trainId": tids,
            "cellId": [0, 1, 15],
            "q": q,
            "n_frames": (("trainId", "cellId"), n_frames),
        },
        name="intensity",
        attrs={"units": units},
    )


@pytest.fixture
def waxs(tmp_path, xgm_by_raw):
    tids = train_ids(6)
    q1, q2 = np.linspace(11.0, 24.0, 66), np.linspace(9.5, 19.0, 50)
    grids = {
        "jf1": jf_grid(q1, 11.5, 23.7, tids, 1.0),
        "jf2": jf_grid(q2, 9.8, 18.5, tids, JF2_SCALE),
    }
    run = damnit_run(tmp_path / "w1.h5", tids, 2.2 - 0.01 * np.arange(6))
    for det, grid in grids.items():
        grid.to_netcdf(
            run.file, group=f"jungfrau_waxs_{det}", mode="a", engine="h5netcdf"
        )
    xgm = xr.DataArray(300.0 + np.arange(6.0), dims="trainId", coords={"trainId": tids})
    raw = FakeRaw(timestamps(np.r_[tids, train_ids(1, 6)]), xgm)
    return {
        "db": {1: run},
        "raw": raw,
        "tids": tids,
        "grids": grids,
        "dir": tmp_path / "c",
    }


def cell_mean(grid):
    with np.errstate(invalid="ignore"):
        return grid.where(grid.n_frames > 0).mean("cellId").values


def test_combined_puts_jf2_on_jf1_over_a_unified_q_axis(waxs):
    ds = cache.waxs_run(1, waxs["db"], raw=waxs["raw"], cache_dir=waxs["dir"])

    q = ds.q.values
    assert q.min() >= 9.8 and q.max() <= 23.7
    assert np.all(np.diff(q) > 0)
    assert ds.attrs["scaling_factor"] == pytest.approx(1 / JF2_SCALE, rel=0.02)
    # jf1 only, both, jf2 only
    src = dict(zip(q, ds.source.values, strict=True))
    assert src[q[0]] == 1 and src[q[-1]] == 0 and (ds.source == 2).any()
    # every populated train follows the truth up to its own drift
    good = ds.intensity.sel(trainId=waxs["tids"][[0, 1, 3, 4]])
    drift = 1 + 0.1 * np.sin(np.arange(6))[[0, 1, 3, 4]]
    np.testing.assert_allclose(good.values, drift[:, None] * truth(q)[None], rtol=0.05)
    assert ds.attrs["units"] == "photons"


def test_combined_keeps_missing_frames_out_and_empty_trains_nan(waxs):
    ds = cache.waxs_run(1, waxs["db"], raw=waxs["raw"], cache_dir=waxs["dir"])

    assert ds.intensity.sel(trainId=waxs["tids"][5]).isnull().all()
    np.testing.assert_array_equal(ds.n_cells_with_frames_jf1, [3, 3, 2, 3, 3, 0])
    assert ds.intensity.sel(trainId=waxs["tids"][2]).min() > 0  # zeros not averaged in


def test_split_returns_two_datasets_on_their_own_q_axes(waxs):
    jf1, jf2 = cache.waxs_run(
        1, waxs["db"], raw=waxs["raw"], combine=False, cache_dir=waxs["dir"]
    )

    assert jf1.attrs["detector"] == "jf1" and jf2.attrs["detector"] == "jf2"
    np.testing.assert_allclose(jf1.q, waxs["grids"]["jf1"].q)
    np.testing.assert_allclose(jf2.q, waxs["grids"]["jf2"].q)
    assert jf1.q.size == 66 and jf2.q.size == 50
    assert jf1.detector == "jf1" and jf2.detector == "jf2"
    assert "q1" not in jf1.dims and "q2" not in jf2.dims
    np.testing.assert_allclose(
        jf1.intensity.values, cell_mean(waxs["grids"]["jf1"]), rtol=1e-6
    )
    np.testing.assert_allclose(
        jf2.intensity.values, cell_mean(waxs["grids"]["jf2"]), rtol=1e-6
    )  # stored unscaled
    np.testing.assert_array_equal(jf2.n_cells_with_frames, [3, 3, 2, 3, 3, 0])
    assert "intensity_jf1" not in jf2 and "scale" in jf2 and "time" in jf1


def test_waxs_scaling_and_times_are_stored_by_train_id(waxs):
    ds = cache.waxs_run(1, waxs["db"], raw=waxs["raw"], cache_dir=waxs["dir"])
    scaling = utils.droplet_scaling(1, waxs["db"], raw=waxs["raw"], n_pulses=155)

    np.testing.assert_allclose(ds.scale, scaling.scale.reindex(trainId=ds.trainId))
    by_label = waxs["raw"].times.dt.tz_localize(None).loc[waxs["tids"]].to_numpy()
    np.testing.assert_array_equal(ds.time.values, by_label.astype("datetime64[ns]"))


def test_waxs_is_cached_per_mode_and_rebuilt_on_refresh(waxs, count_builds):
    kw = {"raw": waxs["raw"], "cache_dir": waxs["dir"]}
    first = cache.waxs_run(1, waxs["db"], **kw)
    second = cache.waxs_run(1, waxs["db"], **kw)
    assert len(count_builds) == 1
    assert second.attrs["created_utc"] == first.attrs["created_utc"]

    cache.waxs_run(1, waxs["db"], combine=False, **kw)
    cache.waxs_run(1, waxs["db"], combine=False, **kw)
    assert len(count_builds) == 2
    assert len(list(waxs["dir"].glob("waxs_r0001_*.nc"))) == 2

    cache.waxs_run(1, waxs["db"], refresh=True, **kw)
    assert len(count_builds) == 3


def test_mixed_units_are_refused_when_combining_but_not_when_split(waxs):
    kev = jf_grid(waxs["grids"]["jf2"].q.values, 9.8, 18.5, waxs["tids"], 1.0, "keV")
    kev.to_netcdf(
        waxs["db"][1].file, group="jungfrau_waxs_jf2", mode="a", engine="h5netcdf"
    )
    kw = {"raw": waxs["raw"], "cache_dir": waxs["dir"]}

    from analysis.waxs.combine import UnitsMismatch

    with pytest.raises(UnitsMismatch):
        cache.waxs_run(1, waxs["db"], **kw)
    jf1, jf2 = cache.waxs_run(1, waxs["db"], combine=False, **kw)
    assert (jf1.attrs["units"], jf2.attrs["units"]) == ("photons", "keV")


def test_an_unknown_method_is_refused(waxs):
    with pytest.raises(ValueError, match="method must be"):
        cache.waxs_run(1, waxs["db"], method="mean", cache_dir=waxs["dir"])


# ── droplet_timeline ──────────────────────────────────────────────────────────
def piecewise_volume(t, t_star=400.0, d0_sq=2.8, k1=1.8e-3, k2=4e-5):
    d2 = np.where(t <= t_star, d0_sq - k1 * t, d0_sq - k1 * t_star - k2 * (t - t_star))
    return deff_sq_to_volume(d2)


def timeline_runs(tmp_path, volume, step_s=2.0, n_runs=2):
    """`volume` spread over `n_runs` consecutive fake runs, trains `step_s` apart."""
    tids = train_ids(len(volume))
    all_times = timestamps(tids, step_s=step_s).sort_index()
    db, raws = {}, {}
    for run_nr, idx in enumerate(
        np.array_split(np.arange(len(volume)), n_runs), start=1
    ):
        db[run_nr] = damnit_run(tmp_path / f"t{run_nr}.h5", tids[idx], volume[idx])
        extra = pd.Series(
            all_times.iloc[idx[-1]] + pd.Timedelta(hours=1),
            index=[tids[idx[-1]] + 10**6],
        )
        raws[run_nr] = FakeRaw(pd.concat([all_times.iloc[idx], extra]))
    return db, raws, tids


def test_the_fit_finds_the_break_of_a_piecewise_d2_law(tmp_path):
    t = np.arange(0.0, 1200.0, 2.0)
    volume = piecewise_volume(t) * (
        1 + 1e-4 * np.random.default_rng(0).standard_normal(t.size)
    )
    db, raws, tids = timeline_runs(tmp_path, volume)

    tl = cache.droplet_timeline([1, 2], db, raws=raws, cache_dir=tmp_path / "c")

    assert tl.attrs["t_star_source"] == "fit"
    assert tl.attrs["t_star_s"] == pytest.approx(400.0, abs=2.0)
    assert tl.attrs["t_star_s"] == tl.attrs["t_star_fit_s"]
    np.testing.assert_array_equal(tl.trainId, tids)  # the extra timestamps fall out
    np.testing.assert_allclose(tl.elapsed_s, t)
    np.testing.assert_array_equal(tl.run, np.repeat([1, 2], 300))
    assert bool(tl.used_in_fit.all())


def test_end_stops_at_the_smallest_volume_and_keeps_every_train(tmp_path):
    t = np.arange(0.0, 900.0, 2.0)
    d2 = 1.6 - 1.9e-3 * t  # a D^2 law that never breaks ...
    d2[t > 718] = np.where(
        np.arange((t > 718).sum()) % 2, 0.42, 0.78
    )  # ... then garbage
    volume = deff_sq_to_volume(d2)
    db, raws, _ = timeline_runs(tmp_path, volume, n_runs=3)

    tl = cache.droplet_timeline(
        [1, 2, 3],
        db,
        raws=raws,
        stop_at_min_volume=True,
        t_star="end",
        cache_dir=tmp_path / "c",
    )

    last = int(np.argmin(volume))
    assert tl.attrs["t_star_source"] == "end"
    assert tl.attrs["t_star_s"] == pytest.approx(t[last])
    np.testing.assert_array_equal(tl.used_in_fit, np.arange(t.size) <= last)
    np.testing.assert_allclose(tl.volume_mm3, volume, rtol=1e-6)  # stored untouched
    assert np.isfinite(tl.attrs["t_star_fit_s"])


def test_a_number_overrides_t_star_and_a_bad_word_is_refused(tmp_path):
    t = np.arange(0.0, 1200.0, 2.0)
    db, raws, _ = timeline_runs(tmp_path, piecewise_volume(t))

    tl = cache.droplet_timeline(
        [1, 2], db, raws=raws, t_star=123.0, cache_dir=tmp_path / "c"
    )
    assert tl.attrs["t_star_s"] == 123.0 and tl.attrs["t_star_source"] == "override"

    with pytest.raises(ValueError, match="t_star must be"):
        cache.droplet_timeline(
            [1, 2], db, raws=raws, t_star="middle", cache_dir=tmp_path / "c"
        )


def test_the_timeline_is_cached_and_reloaded(tmp_path, monkeypatch):
    t = np.arange(0.0, 1200.0, 2.0)
    db, raws, _ = timeline_runs(tmp_path, piecewise_volume(t))
    fits = []
    real = utils.fit_D2_law
    monkeypatch.setattr(
        utils, "fit_D2_law", lambda *a, **k: fits.append(1) or real(*a, **k)
    )

    first = cache.droplet_timeline([1, 2], db, raws=raws, cache_dir=tmp_path / "c")
    second = cache.droplet_timeline([1, 2], db, raws=raws, cache_dir=tmp_path / "c")

    assert len(fits) == 1
    assert second.attrs["t_star_s"] == first.attrs["t_star_s"]
    np.testing.assert_array_equal(second.time, first.time)


# ── detector_sums ─────────────────────────────────────────────────────────────
mockrun = pytest.importorskip("saxs.mockrun")  # tests/saxs, as a namespace package


@pytest.fixture(scope="module")
def proc_run(tmp_path_factory):
    """A 16-module proc-like run: 7 trains of 4 frames; module 0 lacks train 10005."""
    return mockrun.write_mock_run(
        tmp_path_factory.mktemp("proc"),
        train_ids=tuple(range(10000, 10007)),
        frames_per_train=4,
        short_module_trains=(10005,),
    )


def reference_sums(run, train_ids):
    """Counts and valid frames per pixel, read straight from the files with h5py."""
    import h5py

    counts = np.zeros((16, 512, 128), np.int64)
    valid = np.zeros_like(counts)
    for m in range(16):
        path = run.path / f"CORR-R0423-AGIPD{m:02d}-S00000.h5"
        with h5py.File(path, "r") as f:
            image = f[f"INSTRUMENT/{mockrun.source(m)}/image"]
            rows = np.isin(image["trainId"][:].ravel(), train_ids)
            ok = image["mask"][rows] == 0
            counts[m] = np.where(ok, image["data"][rows], 0).sum(0)
            valid[m] = ok.sum(0)
    return counts, valid


def test_sums_equal_an_independent_read_of_the_files(proc_run, tmp_path):
    trains = [10003, 10000, 10002]
    ds = cache.detector_sums(
        1, trains, run_dir=proc_run.path, n_workers=2, cache_dir=tmp_path
    )

    counts, valid = reference_sums(proc_run, trains)
    np.testing.assert_array_equal(ds.counts, counts)
    np.testing.assert_array_equal(ds.valid_frames, valid)
    assert ds.attrs["n_frames"] == 12
    np.testing.assert_array_equal(ds.train_id, sorted(trains))
    # the frame mask removed pixels from some frames, and photons were counted
    assert int(ds.valid_frames.min()) < 12 and int(ds.counts.sum()) > 0


def test_a_train_without_every_module_is_refused(proc_run, tmp_path):
    with pytest.raises(ValueError, match="no frames from all 16 modules"):
        cache.detector_sums(
            1, [10004, 10005], run_dir=proc_run.path, n_workers=1, cache_dir=tmp_path
        )


def test_sums_are_cached_and_rebuilt_on_new_trains_a_changed_run_or_refresh(
    proc_run, tmp_path, monkeypatch
):
    reads = []
    real = cache._read_sums
    monkeypatch.setattr(
        cache, "_read_sums", lambda *a, **k: reads.append(1) or real(*a, **k)
    )
    kw = {"run_dir": proc_run.path, "n_workers": 1, "cache_dir": tmp_path}

    first = cache.detector_sums(1, [10000, 10001], label="pair", **kw)
    again = cache.detector_sums(1, [10001, 10000], label="pair", **kw)
    assert len(reads) == 1
    np.testing.assert_array_equal(again.counts, first.counts)

    other = cache.detector_sums(1, [10002, 10003], label="pair", **kw)  # same size
    assert len(reads) == 2
    np.testing.assert_array_equal(
        other.counts, reference_sums(proc_run, [10002, 10003])[0]
    )

    cache.detector_sums(1, [10000, 10001], label="pair", refresh=True, **kw)
    assert len(reads) == 3

    some_file = next(proc_run.path.glob("*.h5"))
    stat = os.stat(some_file)
    os.utime(some_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    cache.detector_sums(1, [10000, 10001], label="pair", **kw)
    assert len(reads) == 4
    assert len(list(tmp_path.glob("sums_r0001_pair_*.nc"))) == 2


# ── xpcs_run ──────────────────────────────────────────────────────────────────
XPCS_OFFSETS = (0, 1, 2, 4, 5, 6)  # hit trains that gave photons
XPCS_FAILED = 3  # a hit train with no sums
XPCS_FRAMES, XPCS_ENV, XPCS_RINGS = 14, 300, 3


@pytest.fixture
def xpcs(tmp_path, xgm_by_raw):
    """Run 1: seven hit trains in a real results file, one of them failed.

    The DAMNIT side has droplet entries for eight trains, so every hit has a train
    timestamp and a scaling; the hit trains are not the run's first seven.
    """
    rng = np.random.default_rng(5)
    rings = make_rings(rng, XPCS_ENV, XPCS_RINGS)
    photons = [
        make_train(
            int(train_ids(1, k)[0]),
            make_dense(rng, XPCS_FRAMES, XPCS_ENV, 0.12),
            make_flags(rng, XPCS_FRAMES, XPCS_ENV),
        )
        for k in XPCS_OFFSETS
    ]
    failed_id = int(train_ids(1, XPCS_FAILED)[0])
    path = tmp_path / "xpcs_hits.h5"
    build_results(path, photons, rings, failed={failed_id: FrameStatus.MISSING_MODULES})
    tids = train_ids(8)
    db = {1: damnit_run(tmp_path / "x1.h5", tids, 2.2 - 0.01 * np.arange(8))}
    xgm = xr.DataArray(300.0 + np.arange(8.0), dims="trainId", coords={"trainId": tids})
    raw = FakeRaw(timestamps(tids), xgm)
    return {
        "db": db,
        "raw": raw,
        "path": path,
        "failed": failed_id,
        "hit_ids": np.array(sorted([*(t.train_id for t in photons), failed_id])),
        "kw": {"results_file": path, "raw": raw, "cache_dir": tmp_path / "c"},
        "dir": tmp_path / "c",
    }


@pytest.mark.parametrize("averaging", ["pooled", "es"])
def test_g2_and_the_two_time_maps_are_the_readers_own(xpcs, averaging):
    ds = cache.xpcs_run(1, xpcs["db"], averaging=averaging, **xpcs["kw"])

    hits = XpcsHits(xpcs["path"])
    mode = Averaging(averaging)
    g2, ttcf = hits.g2(mode), hits.ttcf(mode)
    np.testing.assert_allclose(ds.tau, g2.tau)
    for name in g2.data_vars:
        kept = ds[name].sel(roi_q=g2.roi_q)  # es may leave rings out
        np.testing.assert_allclose(kept.values, g2[name].values, equal_nan=True)
    for name in ttcf.data_vars:
        np.testing.assert_allclose(
            ds[f"ttcf_{name}"].values, ttcf[name].values, equal_nan=True
        )
    assert ("sigma" in ds) == (averaging == "pooled")
    assert ("ttcf_raw_paired" in ds) == (averaging == "pooled")
    assert ds.attrs["averaging"] == averaging
    assert ds.attrs["n_trains"] == ttcf.attrs["n_trains"] == len(XPCS_OFFSETS)
    assert ds.attrs["coverage"] == str(hits.coverage())


def test_the_pooled_lag_axis_is_stored(xpcs):
    ds = cache.xpcs_run(1, xpcs["db"], **xpcs["kw"])

    want = XpcsHits(xpcs["path"]).g2()
    np.testing.assert_array_equal(ds.lag, want.lag)
    assert ds.attrs["period_s"] == pytest.approx(want.attrs["period_s"])
    assert ds.attrs["n_with_baseline"] == want.attrs["n_with_baseline"]
    np.testing.assert_allclose(
        ds.ring_bounds_nm.values, XpcsHits(xpcs["path"]).bounds_nm
    )


def test_every_hit_train_is_listed_by_train_id(xpcs):
    ds = cache.xpcs_run(1, xpcs["db"], **xpcs["kw"])
    hits = XpcsHits(xpcs["path"])

    np.testing.assert_array_equal(ds.trainId, xpcs["hit_ids"])
    failed = ds.sel(trainId=xpcs["failed"])
    assert failed.status == FrameStatus.MISSING_MODULES
    assert failed.kbar.isnull().all()
    assert not failed.has_off
    usable = hits.train_ids[hits.usable]
    np.testing.assert_allclose(
        ds.kbar.sel(trainId=usable).values, hits.kbar().values, equal_nan=True
    )
    np.testing.assert_array_equal(ds.status, hits.status)
    np.testing.assert_array_equal(ds.pair_status, hits.pair_status)
    np.testing.assert_array_equal(ds.partner, hits.partner_ids)
    np.testing.assert_array_equal(ds.has_off, hits.has_off)
    assert ds.status.dtype == np.uint8 and ds.has_off.dtype == bool


def test_xpcs_scaling_and_times_sit_on_the_hit_train(xpcs):
    ds = cache.xpcs_run(1, xpcs["db"], **xpcs["kw"])
    scaling = utils.droplet_scaling(1, xpcs["db"], raw=xpcs["raw"], n_pulses=155)

    np.testing.assert_allclose(ds.scale, scaling.scale.sel(trainId=xpcs["hit_ids"]))
    by_label = xpcs["raw"].times.dt.tz_localize(None).loc[xpcs["hit_ids"]]
    np.testing.assert_array_equal(
        ds.time.values, by_label.to_numpy().astype("datetime64[ns]")
    )
    # the failed hit has no sums but keeps its place and its droplet
    assert np.isfinite(ds.scale.sel(trainId=xpcs["failed"]))
    assert np.isfinite(ds.droplet_volume.sel(trainId=xpcs["failed"]))


def test_an_unfinished_results_file_is_refused(xpcs):
    with h5py.File(xpcs["path"], "r+") as f:
        f["hits/status"][2] = int(FrameStatus.NOT_PROCESSED)

    with pytest.raises(IncompleteRun, match="1 hit trains are unfinished"):
        cache.xpcs_run(1, xpcs["db"], **xpcs["kw"])
    assert not xpcs["dir"].exists()


def test_a_missing_results_file_and_a_bad_averaging_are_refused(xpcs, tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.h5"):
        cache.xpcs_run(
            1, xpcs["db"], **{**xpcs["kw"], "results_file": tmp_path / "nowhere.h5"}
        )
    with pytest.raises(ValueError, match="'pooled', 'es'"):
        cache.xpcs_run(1, xpcs["db"], averaging="mean", **xpcs["kw"])


def test_xpcs_is_cached_per_averaging_and_rebuilt_on_new_results(xpcs, count_builds):
    first = cache.xpcs_run(1, xpcs["db"], **xpcs["kw"])
    second = cache.xpcs_run(1, xpcs["db"], **xpcs["kw"])
    assert len(count_builds) == 1
    assert second.attrs["created_utc"] == first.attrs["created_utc"]

    cache.xpcs_run(1, xpcs["db"], averaging="es", **xpcs["kw"])
    assert len(count_builds) == 2
    assert len(list(xpcs["dir"].glob("xpcs_r0001_*.nc"))) == 2

    stat = os.stat(xpcs["path"])
    os.utime(xpcs["path"], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    cache.xpcs_run(1, xpcs["db"], **xpcs["kw"])
    assert len(count_builds) == 3

    cache.xpcs_run(1, xpcs["db"], refresh=True, **xpcs["kw"])
    assert len(count_builds) == 4
