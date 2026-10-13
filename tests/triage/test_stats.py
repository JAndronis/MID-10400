"""Threshold statistics: robust spread, multiple-testing cut, empty band."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from analysis.triage.stats import (
    EmptyBand,
    empty_band,
    mad_dof,
    robust_sd,
    welch_dof,
    z_for_tests,
)


def test_robust_sd_recovers_a_normal_sigma_and_ignores_outliers():
    x = np.random.default_rng(0).normal(0.0, 2.0, 20_000)
    x[:200] = 1e6  # 1 % wild values
    assert robust_sd(x) == pytest.approx(2.0, rel=0.05)


def test_robust_sd_needs_three_finite_values():
    assert np.isnan(robust_sd([1.0, np.nan, 2.0]))


def test_z_for_tests_fixes_the_expected_false_count():
    assert z_for_tests(1000) == pytest.approx(norm.isf(0.1 / 1000))
    assert z_for_tests(1000, expected_false=1.0) == pytest.approx(norm.isf(1e-3))
    with pytest.raises(ValueError):
        z_for_tests(0)


def test_empty_band_is_the_widest_gap_between_measured_values():
    band = empty_band([0.0, 1.0, 2.0, 10.0, 11.0, np.nan])
    assert band == EmptyBand(2.0, 10.0)
    assert band.centre == 6.0 and band.width == 8.0


def test_empty_band_respects_its_search_window():
    assert empty_band([0.0, 0.5, 2.0, 10.0, 11.0], hi=5.0) == EmptyBand(0.5, 2.0)
    assert empty_band([3.0]) is None


def test_z_for_tests_two_sided_and_studentised():
    assert z_for_tests(100, two_sided=True) == pytest.approx(norm.isf(0.1 / 200))
    assert z_for_tests(100, dof=20.0) > z_for_tests(100)  # a noisy sigma costs margin
    assert z_for_tests(100, dof=float("inf")) == z_for_tests(100)


def test_welch_dof_combines_two_estimated_variances():
    assert welch_dof(1.0, 10.0, 1.0, 10.0) == pytest.approx(20.0)
    assert welch_dof(1.0, 10.0, 0.0, 10.0) == pytest.approx(10.0)
    assert welch_dof(1.0, float("inf"), 1.0, float("inf")) == float("inf")
    assert mad_dof(101) == pytest.approx(37.0)
