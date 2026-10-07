"""The on-disk cache in `analysis.cache`.

A fake DAMNIT database whose runs point at small netCDF files (``agipd_saxs`` and
``droplet_fit`` groups), fake raw runs that hand out labelled train timestamps, and a
stand-in for the XGM. The cache must store what the sources hold, averaged over pulses
and nothing else, label everything by trainId, and rebuild exactly when its inputs or
sources change.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("extra_data")  # analysis.utils imports it at module level
pytest.importorskip("h5netcdf")  # the cache and the fake DAMNIT files are netCDF

from analysis import cache, utils  # noqa: E402
from analysis.utils import deff_sq_to_volume  # noqa: E402

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
