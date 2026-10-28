"""The pure estimators: diagonal sums of a TTCF stack and the block jackknife."""

from __future__ import annotations

import numpy as np

from analysis.xpcs.estimators import block_jackknife, diagonal_sums, safe_divide


def test_diagonal_sums_skip_non_finite_entries():
    t = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    t[0, 0, 0, 1] = np.nan
    sums, counts = diagonal_sums(t)
    assert counts[0, 0].tolist() == [2, 2, 1]  # lag 1 lost (0, 1)
    assert sums[0, 0].tolist() == [6.0 + 11.0, 2.0 + 7.0, 3.0]


def test_jackknife_finds_iid_noise():
    rng = np.random.default_rng(0)
    sums = rng.normal(0.0, 1.0, (500, 1, 3))
    counts = np.ones_like(sums, dtype=np.int32)
    g2, sigma, _ = block_jackknife(safe_divide, sums, counts, n_blocks=50)
    np.testing.assert_allclose(sigma, 1 / np.sqrt(500), rtol=0.25)
