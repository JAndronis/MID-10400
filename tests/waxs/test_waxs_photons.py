"""Per-ASIC photon counting (``analysis.waxs.photons``), on synthetic frames.

The estimator is p010646's; these pin what it must do here: recover a zero level
per ASIC from the frame itself, count the photons that were put in, double them
at half the energy, give no level rather than a guess where too few pixels
remain, and be reproducible from the float32 values a file stores.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from analysis.waxs.config import MODULE_SHAPE
from analysis.waxs.photons import (
    ASIC_ID,
    ASIC_SIZE,
    N_ASICS,
    PhotonCounting,
    asic_view,
    count_photons,
    zero_levels,
)

E = 9.04
NOISE = 0.30  # the pure pixel noise W7 measured on both detectors
OFFSETS = np.array([0.13, -0.04, 0.17, 0.06, -0.12, 0.10, 0.00, 0.21])


@pytest.fixture
def counting():
    return PhotonCounting(energy_kev=E)


def frame(seed=0, rate=0.45, offsets=OFFSETS, noise=NOISE):
    """``(keV frame, true counts)``, one zero level per ASIC."""
    rng = np.random.default_rng(seed)
    truth = rng.poisson(rate, MODULE_SHAPE).reshape(-1)
    values = E * truth + rng.normal(0.0, noise, truth.size) + offsets[ASIC_ID]
    return values.astype(np.float32), truth


def everything():
    return np.ones(ASIC_ID.size, dtype=bool)


# ── the tiling ────────────────────────────────────────────────────────────────
def test_the_asic_map_and_the_asic_view_agree():
    """Two descriptions of one tiling: a pixel's ASIC id is the view it lands in."""
    view = asic_view(ASIC_ID.reshape(MODULE_SHAPE), 0)
    assert view.shape == (N_ASICS, ASIC_SIZE * ASIC_SIZE)
    for asic in range(N_ASICS):
        assert (view[asic] == asic).all()


def test_the_edge_is_left_out_of_every_asic():
    view = asic_view(np.zeros(MODULE_SHAPE), 3)
    assert view.shape == (N_ASICS, (ASIC_SIZE - 6) ** 2)


# ── the zero level ────────────────────────────────────────────────────────────
def test_a_level_per_asic_is_recovered_under_photons(counting):
    values, _ = frame()
    levels, n_pixels = zero_levels(values, everything(), counting)
    assert np.abs(levels - OFFSETS).max() < 0.005
    assert (n_pixels == (ASIC_SIZE - 6) ** 2).all()


def test_the_level_follows_a_common_mode_jump(counting):
    """jf1's 0.15 keV train-to-train jump: the frame's own level moves with it."""
    first, _ = frame(seed=1)
    second, _ = frame(seed=2, offsets=OFFSETS + 0.15)
    a, _ = zero_levels(first, everything(), counting)
    b, _ = zero_levels(second, everything(), counting)
    assert (b - a) == pytest.approx(np.full(N_ASICS, 0.15), abs=0.005)


def test_too_few_pixels_give_no_level_not_a_guess(counting):
    values, _ = frame()
    usable = everything()
    keep = np.flatnonzero(ASIC_ID == 4)[:500]
    usable[ASIC_ID == 4] = False
    usable[keep] = True
    levels, n_pixels = zero_levels(values, usable, counting)
    assert n_pixels[4] < counting.min_pixels
    assert np.isnan(levels[4])
    assert np.isfinite(np.delete(levels, 4)).all()


def test_an_asic_with_nothing_below_the_cut_has_no_level(counting):
    values, _ = frame()
    values[ASIC_ID == 3] = 50.0  # saturated: no zero-photon pixel at all
    levels, _ = zero_levels(values, everything(), counting)
    assert np.isnan(levels[3])


def test_the_estimator_converges_by_three_iterations(counting):
    """W7: iterations 3 and 10 differ by <= 0.0009 keV on real frames."""
    values, _ = frame(seed=4)
    three, _ = zero_levels(values, everything(), counting)
    ten, _ = zero_levels(
        values, everything(), dataclasses.replace(counting, level_iterations=10)
    )
    assert np.abs(ten - three).max() < 1e-3


def test_the_estimator_is_the_template_one(counting):
    """p010646's ``baseline``, verbatim, gives the same levels."""
    values, _ = frame(seed=5)
    a = asic_view(values.reshape(MODULE_SHAPE).astype(np.float64), 3)
    est = np.nanmedian(np.where(a < 2.0, a, np.nan), -1)
    for _ in range(3):
        w = np.abs(a - est[..., None]) < 1.0
        est = np.nansum(np.where(w, a, 0), -1) / w.sum(-1)
    mine, _ = zero_levels(values, everything(), counting)
    assert mine == pytest.approx(est, abs=1e-12)


# ── counting ──────────────────────────────────────────────────────────────────
def test_the_photons_put_in_are_the_photons_counted(counting):
    values, truth = frame()
    levels, _ = zero_levels(values, everything(), counting)
    photons = count_photons(values, levels.astype(np.float32), counting.energy_f4)
    assert photons.dtype == np.float32
    assert np.array_equal(photons, truth)


def test_half_the_energy_counts_every_photon_twice(counting):
    """The E/2 test (W7 measured 2.000–2.008 on real frames)."""
    values, truth = frame(seed=6, rate=1.5)
    levels = zero_levels(values, everything(), counting)[0].astype(np.float32)
    full = count_photons(values, levels, counting.energy_f4)
    half = count_photons(values, levels, np.float32(E / 2))
    assert np.array_equal(half, 2 * full)
    assert np.array_equal(full, truth)


def test_a_level_beyond_the_cut_is_refused_not_miscounted(counting):
    """An ASIC whose zero level sits far off (a broken pedestal, say) gets no level.

    The first estimate takes only values below ``level_cut_kev``, so a level of
    ``E/2 - 0.1`` keV leaves nothing to estimate from. Counting against zero
    instead would turn most of the ASIC's noise into photons, so refusing is
    the safe outcome: the pixels are left out and ``n_level_missing`` says so.
    """
    offsets = OFFSETS.copy()
    offsets[5] = E / 2 - 0.1
    values, truth = frame(seed=7, offsets=offsets)
    naive = count_photons(values, np.zeros(N_ASICS, np.float32), counting.energy_f4)
    assert (naive != truth)[ASIC_ID == 5].mean() > 0.3
    levels, _ = zero_levels(values, everything(), counting)
    assert np.isnan(levels[5])
    assert np.abs(np.delete(levels, 5) - np.delete(OFFSETS, 5)).max() < 0.005


def test_negative_values_count_zero_and_nan_stays_nan(counting):
    values = np.full(ASIC_ID.size, -3.0, dtype=np.float32)
    values[0] = np.nan
    photons = count_photons(values, np.zeros(N_ASICS, np.float32), counting.energy_f4)
    assert np.isnan(photons[0])
    assert (photons[1:] == 0).all()


def test_a_missing_level_gives_nan_counts_on_its_asic(counting):
    values, _ = frame()
    levels = OFFSETS.astype(np.float32)
    levels[2] = np.nan
    photons = count_photons(values, levels, counting.energy_f4)
    assert np.isnan(photons[ASIC_ID == 2]).all()
    assert np.isfinite(photons[ASIC_ID != 2]).all()


def test_counts_are_reproducible_from_the_stored_float32_values(counting):
    """A reader recounts a frame from proc, ``asic_level`` and the stored E."""
    values, _ = frame(seed=8)
    levels = zero_levels(values, everything(), counting)[0].astype(np.float32)
    stored_levels = np.frombuffer(levels.tobytes(), dtype=np.float32)
    stored_energy = np.float32(float(counting.energy_f4))
    first = count_photons(values, levels, counting.energy_f4)
    again = count_photons(values, stored_levels, stored_energy)
    assert np.array_equal(first, again)


# ── settings ──────────────────────────────────────────────────────────────────
def test_the_settings_come_from_the_config():
    from analysis.waxs.config import JungfrauWaxsConfig

    cfg = JungfrauWaxsConfig(
        proposal=10400,
        run=423,
        detector="jf2",
        poni_file=None,
        static_mask_file=None,
        level_min_pixels=2000,
        level_pixels="kept",
    )
    counting = PhotonCounting.from_config(cfg)
    assert counting.energy_kev == cfg.photon_energy_kev
    assert counting.min_pixels == 2000
    assert counting.level_pixels == "kept"
    assert "rint" in counting.method and "kept" in counting.method
    assert counting.summary()["method"] == counting.method


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("energy_kev", 0.0, "energy_kev"),
        ("level_window_kev", 0.0, "level_window_kev"),
        ("level_iterations", -1, "level_iterations"),
        ("edge_px", 128, "edge_px"),
        ("min_pixels", 0, "min_pixels"),
        ("level_pixels", "shadow", "level_pixels"),
    ],
)
def test_nonsense_settings_are_refused(field, value, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(PhotonCounting(energy_kev=E), **{field: value})
