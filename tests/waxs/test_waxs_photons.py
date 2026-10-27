"""Per-ASIC photon counting (``analysis.waxs.photons``), on synthetic frames.

These tests pin what the estimator must do: recover a zero level per ASIC from
the frame itself, count the photons that were put in, and give no level rather
than a guess where too few pixels remain.
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
NOISE = 0.30  # keV, the per-pixel noise of both detectors
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
    assert asic_view(np.zeros(MODULE_SHAPE), 3).shape == (N_ASICS, (ASIC_SIZE - 6) ** 2)


# ── the zero level ────────────────────────────────────────────────────────────
def test_a_level_per_asic_is_recovered_under_photons(counting):
    values, _ = frame()
    levels, n_pixels = zero_levels(values, everything(), counting)
    assert np.abs(levels - OFFSETS).max() < 0.005
    assert (n_pixels == (ASIC_SIZE - 6) ** 2).all()


def test_too_few_pixels_give_no_level_not_a_guess(counting):
    values, _ = frame()
    usable = everything()
    # 500 pixels inside the ASIC's edge, so they all reach the estimator.
    interior = asic_view(np.arange(ASIC_ID.size).reshape(MODULE_SHAPE), 3)[4]
    usable[ASIC_ID == 4] = False
    usable[interior[:500]] = True
    levels, n_pixels = zero_levels(values, usable, counting)
    assert n_pixels[4] == 500 < counting.min_pixels
    assert np.isnan(levels[4])
    assert np.isfinite(np.delete(levels, 4)).all()


# ── counting ──────────────────────────────────────────────────────────────────
def test_the_photons_put_in_are_the_photons_counted(counting):
    values, truth = frame()
    levels, _ = zero_levels(values, everything(), counting)
    photons = count_photons(values, levels.astype(np.float32), counting.energy_f4)
    assert photons.dtype == np.float32
    assert np.array_equal(photons, truth)


def test_a_level_beyond_the_cut_is_refused_not_miscounted(counting):
    """An ASIC whose zero level sits far off (a broken pedestal, say) gets no level.

    The first estimate takes only values below ``level_cut_kev``, so a level of
    ``E/2 - 0.1`` keV leaves nothing to estimate from, as does a saturated ASIC.
    Counting against zero instead would turn most of the ASIC's noise into
    photons, so refusing is the safe outcome: the pixels are left out and
    ``n_level_missing`` says so.
    """
    offsets = OFFSETS.copy()
    offsets[5] = E / 2 - 0.1
    values, truth = frame(seed=7, offsets=offsets)
    values[ASIC_ID == 3] = 50.0
    naive = count_photons(values, np.zeros(N_ASICS, np.float32), counting.energy_f4)
    assert (naive != truth)[ASIC_ID == 5].mean() > 0.3
    levels, _ = zero_levels(values, everything(), counting)
    assert np.isnan(levels[[3, 5]]).all()
    others = np.delete(np.arange(N_ASICS), [3, 5])
    assert np.abs(levels[others] - OFFSETS[others]).max() < 0.005


def test_negative_values_count_zero_and_nan_stays_nan(counting):
    """A NaN pixel or a NaN level gives NaN counts, which the caller excludes."""
    # -6 keV rounds to -1 photon before the clip.
    values = np.full(ASIC_ID.size, -6.0, dtype=np.float32)
    values[0] = np.nan
    photons = count_photons(values, np.zeros(N_ASICS, np.float32), counting.energy_f4)
    assert np.isnan(photons[0])
    assert (photons[1:] == 0).all()

    values, _ = frame()
    levels = OFFSETS.astype(np.float32)
    levels[2] = np.nan
    photons = count_photons(values, levels, counting.energy_f4)
    assert np.isnan(photons[ASIC_ID == 2]).all()
    assert np.isfinite(photons[ASIC_ID != 2]).all()


def test_an_unknown_choice_of_level_pixels_is_refused():
    with pytest.raises(ValueError, match="level_pixels"):
        dataclasses.replace(PhotonCounting(energy_kev=E), level_pixels="shadow")
