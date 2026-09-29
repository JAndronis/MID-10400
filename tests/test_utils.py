"""Droplet attenuation and the per-train SAXS scale factors in `analysis.utils`.

Everything is synthetic: a fake DAMNIT database whose runs point at small netCDF
files with a `droplet_fit` group, and a stand-in for the XGM. The physics is held
to limits that do not depend on the implementation (pure water, the maximum of
t exp(-mu t), conserved dry volume), and `droplet_scaling` to a known
cross-section it must recover after a buffer subtraction, with the intensity grid
deliberately out of order and carrying a train the scaling does not know.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("extra_data")  # analysis.utils imports it at module level
pytest.importorskip("h5netcdf")  # writes the fake droplet_fit files

from analysis import utils  # noqa: E402
from analysis.utils import (  # noqa: E402
    DROPLET_PX_MM,
    M_NACL,
    MU_RHO_9P04_KEV,
    RHO_WATER_25C,
    droplet_attenuation,
    droplet_composition,
    droplet_scaling,
    train_pulse_energy,
)

WATER_ONLY = dict(
    ferritin_mg_per_ml_initial=0.0, peg_percent_wv_initial=0.0, nacl_mM_initial=0.0
)
FIRST_TRAIN = 2637862276


def train_ids(n: int) -> np.ndarray:
    return FIRST_TRAIN + np.arange(n, dtype=np.uint64)


# ── droplet_attenuation ───────────────────────────────────────────────────────
def test_pure_water_is_the_tabulated_water_attenuation():
    chord = np.array([0.5, 1.0, 2.0])
    att = droplet_attenuation(np.ones(3), 1.0, chord, **WATER_ONLY)

    mu = RHO_WATER_25C * MU_RHO_9P04_KEV["H2O"] / 10.0  # 1/mm
    np.testing.assert_allclose(att.mu_per_mm, mu, rtol=1e-12)
    np.testing.assert_allclose(att.droplet_transmission, np.exp(-mu * chord))
    np.testing.assert_allclose(att.path_mm, chord * np.exp(-mu * chord))
    np.testing.assert_array_equal(att.phi_ferritin, 0.0)
    np.testing.assert_array_equal(att.v_over_v0, 1.0)


def test_the_path_factor_peaks_at_one_attenuation_length():
    """t exp(-mu t) is largest at t = 1/mu: a thicker droplet scatters less."""
    chord = np.linspace(0.01, 5.0, 5000)
    att = droplet_attenuation(np.ones_like(chord), 1.0, chord)

    mu = att.mu_per_mm[0]
    assert chord[np.argmax(att.path_mm)] == pytest.approx(1.0 / mu, abs=2e-3)


def test_evaporation_concentrates_the_absorbers_and_conserves_the_ferritin():
    v_over_v0 = np.linspace(1.0, 0.4, 7)
    att = droplet_attenuation(v_over_v0, 1.0, np.ones(7))

    np.testing.assert_allclose(att.v_over_v0, v_over_v0)
    assert np.all(np.diff(att.mu_per_mm) > 0)
    # the ferritin's own volume does not evaporate: phi * V is constant
    np.testing.assert_allclose(att.phi_ferritin * v_over_v0, att.phi_ferritin[0])


def test_the_iron_core_sets_the_ferritin_attenuation():
    mu = [
        droplet_attenuation(1.0, 1.0, 1.0, n_fe=n).mu_per_mm for n in (1e3, 18e2, 35e2)
    ]
    assert mu[0] < mu[1] < mu[2]

    buffer = dict(ferritin_mg_per_ml_initial=0.0)
    mu_buffer = [
        droplet_attenuation(1.0, 1.0, 1.0, n_fe=n, **buffer).mu_per_mm
        for n in (1e3, 35e2)
    ]
    assert mu_buffer[0] == mu_buffer[1]


def test_salt_is_converted_from_millimolar_and_concentrates():
    def mu(nacl_mM, volume):
        return droplet_attenuation(
            volume, 1.0, 1.0, **{**WATER_ONLY, "nacl_mM_initial": nacl_mM}
        ).mu_per_mm

    per_150_mM = 150e-3 * M_NACL * 1e-3 * MU_RHO_9P04_KEV["NaCl"] / 10.0  # 1/mm
    assert mu(150.0, 1.0) - mu(0.0, 1.0) == pytest.approx(per_150_mM, rel=1e-12)
    assert mu(150.0, 0.5) - mu(0.0, 0.5) == pytest.approx(2 * per_150_mM, rel=1e-12)


def test_a_volume_below_the_dry_volume_is_nan_everywhere():
    dry = droplet_composition(1.0, 1.0).volume_dry
    att = droplet_attenuation(0.5 * dry, 1.0, 1.0)

    assert all(np.isnan(value) for value in att)


# ── train_pulse_energy ────────────────────────────────────────────────────────
@pytest.fixture
def fake_xgm(monkeypatch):
    """Make `XGM(raw).pulse_energy()` return whatever the test sets."""
    components = pytest.importorskip("extra.components")
    returned = {}

    class FakeXGM:
        def __init__(self, raw):
            returned["raw"] = raw

        def pulse_energy(self):
            return returned["pulse_energy"]

    monkeypatch.setattr(components, "XGM", FakeXGM)
    return returned


def pulse_energies(per_train: np.ndarray, n_pulses: int) -> xr.DataArray:
    """The per-pulse profile the XGM reports at 2.26 MHz: first 20 pulses 20 % high."""
    profile = np.where(np.arange(n_pulses) < 20, 1.2, 1.0)
    return xr.DataArray(
        per_train[:, None] * profile[None, :],
        dims=("trainId", "pulseIndex"),
        coords={"trainId": train_ids(len(per_train))},
    )


def test_train_pulse_energy_averages_only_the_first_n_pulses(fake_xgm):
    fake_xgm["pulse_energy"] = pulse_energies(np.array([300.0, 350.0, 400.0]), 350)

    got = train_pulse_energy("raw", n_pulses=155)

    expected = np.array([300.0, 350.0, 400.0]) * (20 * 1.2 + 135 * 1.0) / 155
    np.testing.assert_allclose(got.values, expected)
    np.testing.assert_array_equal(got.trainId, train_ids(3))
    assert fake_xgm["raw"] == "raw"


def test_train_pulse_energy_refuses_more_pulses_than_the_train_has(fake_xgm):
    fake_xgm["pulse_energy"] = pulse_energies(np.array([300.0]), 155)

    with pytest.raises(ValueError, match="155 pulses per train"):
        train_pulse_energy("raw", n_pulses=350)


def test_a_train_missing_a_pulse_is_nan_not_a_shorter_average(fake_xgm):
    energies = pulse_energies(np.array([300.0, 350.0]), 155)
    energies[1, 50] = np.nan
    fake_xgm["pulse_energy"] = energies

    got = train_pulse_energy("raw", n_pulses=155)

    assert np.isfinite(got[0]) and np.isnan(got[1])


# ── droplet_scaling ───────────────────────────────────────────────────────────
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


def droplet_run(tmp_path, run_nr, tids, volume, radius_x_px, transmission):
    """One fake DAMNIT run: a netCDF droplet_fit group and a transmission per train."""
    fit = xr.Dataset(
        {
            "volume": ("trainId", np.asarray(volume, dtype=np.float32)),
            "radius": (
                ("trainId", "dim"),
                np.c_[radius_x_px, 0.65 * np.asarray(radius_x_px)].astype(np.float32),
            ),
        },
        coords={"trainId": tids, "dim": ["x", "y"]},
    )
    path = tmp_path / f"p10400_r{run_nr}.h5"
    fit.to_netcdf(path, group="droplet_fit", engine="h5netcdf")
    return FakeRun(
        path, xr.DataArray(transmission, dims="trainId", coords={"trainId": tids})
    )


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """A ferritin droplet (run 1, 8 trains) and a buffer droplet (run 2, 5 trains).

    Run 1 has one train without XGM (index 2), one with a NaN transmission (4) and
    one without a droplet fit (6), so exactly five trains survive, and every input
    varies per train so a value attached to the wrong train would show.
    """
    tids = train_ids(8)
    volume = 2.2 - 0.02 * np.arange(8)
    volume[6] = np.nan
    sample = droplet_run(
        tmp_path,
        1,
        tids,
        volume=volume,
        radius_x_px=66.0 - 0.3 * np.arange(8),
        transmission=np.array([0.1, 0.1, 0.1, 0.1, np.nan, 0.1, 0.1, 0.1]),
    )

    buffer_tids = train_ids(5) + 1000
    buffer = droplet_run(
        tmp_path,
        2,
        buffer_tids,
        volume=1.07 - 0.001 * np.arange(5),
        radius_x_px=53.0 - 0.02 * np.arange(5),
        transmission=np.ones(5),
    )
    xgm = {
        "raw1": xr.DataArray(
            300.0 + np.arange(8.0), dims="trainId", coords={"trainId": tids}
        ).drop_sel(trainId=tids[2]),
        "raw2": xr.DataArray(
            360.0 + np.arange(5.0), dims="trainId", coords={"trainId": buffer_tids}
        ),
    }
    calls = []

    def fake_train_pulse_energy(raw, n_pulses):
        calls.append((raw, n_pulses))
        return xgm[raw]

    monkeypatch.setattr(utils, "train_pulse_energy", fake_train_pulse_energy)
    return {"db": {1: sample, 2: buffer}, "tids": tids, "calls": calls}


def test_every_input_is_aligned_by_train_id(runs):
    ds = droplet_scaling(1, runs["db"], raw="raw1", peg_percent_wv_initial=0.0)

    tids = runs["tids"]
    kept = [0, 1, 3, 5, 7]
    np.testing.assert_array_equal(ds.trainId, tids[kept])

    fit = xr.open_dataset(runs["db"][1].file, group="droplet_fit")
    chord = 2 * fit.radius.sel(dim="x").values[kept] * DROPLET_PX_MM
    att = droplet_attenuation(
        fit.volume.values[kept], float(fit.volume[0]), chord, peg_percent_wv_initial=0.0
    )
    flux = (300.0 + np.array(kept)) * 0.1
    np.testing.assert_allclose(ds.xgm_uJ, 300.0 + np.array(kept))
    np.testing.assert_allclose(ds.chord_mm, chord, rtol=1e-6)
    np.testing.assert_allclose(ds.path_mm, att.path_mm, rtol=1e-6)
    np.testing.assert_allclose(ds.scale, 1.0 / (flux * att.path_mm), rtol=1e-6)


def test_trains_select_by_position_after_alignment(runs):
    first_two = droplet_scaling(1, runs["db"], raw="raw1", trains=slice(0, 2))
    np.testing.assert_array_equal(first_two.trainId, runs["tids"][[0, 1]])

    third = droplet_scaling(1, runs["db"], raw="raw1", trains=2)
    assert third.scale.dims == ("trainId",)
    np.testing.assert_array_equal(third.trainId, runs["tids"][[3]])


def test_v0_comes_from_the_run_the_droplet_started_in(runs):
    fit = xr.open_dataset(runs["db"][2].file, group="droplet_fit")
    ds = droplet_scaling(1, runs["db"], raw="raw1", v0_run=2)

    assert ds.attrs["v0_mm3"] == pytest.approx(float(fit.volume[0]))
    np.testing.assert_allclose(ds.v_over_v0, ds.volume_mm3 / float(fit.volume[0]))
    assert ds.attrs["v0_run"] == 2


def test_n_pulses_reaches_the_xgm(runs):
    droplet_scaling(2, runs["db"], raw="raw2", n_pulses=100)
    assert runs["calls"] == [("raw2", 100)]


def test_raw_data_is_opened_when_no_run_is_given(runs, monkeypatch):
    opened = []
    monkeypatch.setattr(
        utils.ex, "open_run", lambda *a, **kw: opened.append((a, kw)) or "raw2"
    )
    droplet_scaling(2, runs["db"], proposal=10400)
    assert opened == [((10400, 2), {"data": "raw"})]


def test_the_subtraction_recovers_the_ferritin_cross_section(runs):
    """Build I(q) from known cross-sections and check the documented recipe undoes it.

    The sample grid is shuffled and carries a train the scaling dropped, holding
    garbage: xarray must align by label, and the garbage train must fall out.
    """
    q = np.linspace(0.08, 1.05, 50)
    sigma_ferritin = 1e-3 * np.exp(-((q * 5.0) ** 2) / 3)
    sigma_buffer = 1e-5 * (1 + 0.1 / q)

    s = droplet_scaling(1, runs["db"], raw="raw1", peg_percent_wv_initial=0.0)
    b = droplet_scaling(2, runs["db"], raw="raw2", ferritin_mg_per_ml_initial=0.0)

    measured_s = (s.flux * s.path_mm) * (
        xr.DataArray(sigma_ferritin, dims="q")
        + (1 - s.phi_ferritin) * xr.DataArray(sigma_buffer, dims="q")
    )
    garbage = xr.full_like(measured_s.isel(trainId=[0]), 1e9).assign_coords(
        trainId=[runs["tids"][2]]
    )
    grid_s = xr.concat([measured_s, garbage], "trainId").isel(
        trainId=[5, 2, 0, 4, 1, 3]
    )
    grid_b = (b.flux * b.path_mm) * xr.DataArray(sigma_buffer, dims="q")

    diff = grid_s * s.scale - (1 - s.phi_ferritin) * (grid_b * b.scale).mean("trainId")

    assert set(diff.trainId.values) == set(s.trainId.values)
    np.testing.assert_allclose(
        diff.transpose("trainId", "q").values,
        np.broadcast_to(sigma_ferritin, (s.sizes["trainId"], q.size)),
        rtol=1e-10,
    )
