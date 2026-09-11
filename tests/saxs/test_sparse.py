"""Gather kernel and per-frame data checks (context file §6.2, §6.4)."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.saxs.config import METHOD, NPIX
from analysis.saxs.operator import SparseOperator
from analysis.saxs.sparse import (
    denominator,
    frame_data_status,
    gather,
    integrate_frame,
)
from analysis.saxs.status import DataCheckFailed, FrameStatus

pytest.importorskip("pyFAI")


def dense_column_sum(op, cols, weights, *, squared=False):
    """Reference gather, written straight from the CSC definition."""
    out = np.zeros(op.npt, dtype=np.float64)
    for col, weight in zip(cols, weights, strict=True):
        for entry in range(op.indptr[col], op.indptr[col + 1]):
            coefficient = float(op.coef[entry])
            if squared:
                coefficient *= coefficient
            out[op.bins[entry]] += coefficient * weight
    return out


def test_gather_matches_the_csc_definition(op):
    """Independent of pyFAI: the kernel is the CSC matrix-vector product."""
    rng = np.random.default_rng(7)
    cols = np.sort(rng.choice(NPIX, 400, replace=False))
    weights = rng.random(cols.size)
    for squared in (False, True):
        got = gather(op, cols, weights, squared=squared)
        want = dense_column_sum(op, cols, weights, squared=squared)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_gather_on_no_columns_returns_zeros(op):
    out = gather(op, np.array([], dtype=np.int64), np.array([]))
    assert out.shape == (op.npt,)
    assert out.dtype == np.float64
    assert not out.any()


def tiny_operator() -> SparseOperator:
    """A hand-built 6-pixel, 4-bin CSC operator whose column 2 is empty.

    Pixels outside the radial range contribute no CSC entries. The synthetic
    AGIPD geometry populates every column, so that case is built by hand here
    rather than left untested.
    """
    coef = np.array([1.0, 0.5, 0.25, 2.0, 1.5, -1e-9], dtype=np.float32)
    bins = np.array([0, 1, 1, 2, 3, 3], dtype=np.int32)
    #            pixel: 0     1     2(empty) 3     4     5
    indptr = np.array([0, 2, 3, 3, 4, 5, 6], dtype=np.int32)
    omega = np.linspace(0.999, 1.0, 6)
    return SparseOperator(
        coef=coef,
        bins=bins,
        indptr=indptr,
        omega=omega,
        q=np.arange(4, dtype=np.float64),
        npt=4,
        nnz=6,
        shape=(6, 1),
        method=METHOD,
        sdd_m=7.531,
        wavelength_m=1.3715e-10,
        sha256="tiny",
    )


def test_gather_on_columns_with_no_entries_returns_zeros():
    """A column with no CSC entries contributes nothing and must not crash."""
    op = tiny_operator()
    out = gather(op, np.array([2]), np.array([5.0]))
    assert out.shape == (op.npt,)
    assert not out.any()


def test_gather_on_a_hand_built_operator():
    """The kernel against arithmetic done by hand."""
    op = tiny_operator()
    out = gather(op, np.array([0, 1, 3]), np.array([2.0, 4.0, 1.0]))
    # pixel 0 -> bins 0 (c=1.0) and 1 (c=0.5); pixel 1 -> bin 1 (c=0.25);
    # pixel 3 -> bin 2 (c=2.0)
    np.testing.assert_allclose(out, [2.0, 2.0 * 0.5 + 4.0 * 0.25, 2.0, 0.0])
    squared = gather(op, np.array([0]), np.array([2.0]), squared=True)
    np.testing.assert_allclose(squared, [2.0, 2.0 * 0.25, 0.0, 0.0])


def test_integrate_frame_on_a_hand_built_operator():
    """End-to-end §6.4 arithmetic small enough to check by hand.

    Pixel 1 is bad in the cell but good in this frame (its Ω is added back);
    pixel 4 is good in the cell but bad in this frame (its Ω is subtracted).
    """
    op = tiny_operator()
    x = np.array([3, 2, 0, 0, 7, 0], dtype=np.int16)
    base_bad = np.array([False, True, False, False, False, False])
    bad = np.array([False, False, False, False, True, False])

    result = integrate_frame(op, x, bad, base_bad, denominator(op, base_bad))

    # signal: pixel 0 (x=3) into bins 0,1; pixel 1 (x=2) into bin 1. Pixel 4 is
    # flagged by this frame, so its 7 photons are dropped.
    np.testing.assert_allclose(result.signal, [3.0, 3.0 * 0.5 + 2.0 * 0.25, 0.0, 0.0])
    np.testing.assert_allclose(
        result.variance, [3.0, 3.0 * 0.25 + 2.0 * 0.0625, 0.0, 0.0]
    )
    # The corrected denominator must equal one built from ``bad`` directly.
    # atol is not zero because bin 3 is fed only by the deliberately tiny
    # negative coefficient, so the two summation orders differ there at 1e-16 —
    # the same reason the self-test compares only bins with N > 0.
    np.testing.assert_allclose(
        result.normalization, denominator(op, bad), rtol=1e-9, atol=1e-15
    )
    assert result.photons_valid == pytest.approx(5.0)
    assert result.n_bad_pixels == 1
    assert result.n_frame_specific == 2
    assert result.max_count == 7


def test_gather_rejects_mismatched_weights(op):
    with pytest.raises(ValueError, match="same shape"):
        gather(op, np.arange(5), np.ones(4))


def test_gather_is_additive_over_columns(op):
    """Splitting the column set must not change the sum."""
    rng = np.random.default_rng(11)
    cols = np.sort(rng.choice(NPIX, 600, replace=False))
    weights = rng.random(cols.size)
    whole = gather(op, cols, weights)
    halves = gather(op, cols[:300], weights[:300]) + gather(
        op, cols[300:], weights[300:]
    )
    np.testing.assert_allclose(whole, halves, rtol=1e-12, atol=0.0)


def test_denominator_excludes_bad_pixels(op):
    rng = np.random.default_rng(3)
    bad = rng.random(NPIX) < 0.05
    good = np.flatnonzero(~bad)
    np.testing.assert_allclose(
        denominator(op, bad), gather(op, good, op.omega[good]), rtol=0, atol=0
    )


# ── data checks (context file §6.4, §8) ───────────────────────────────────────
def test_frame_data_status_accepts_integer_counts():
    assert frame_data_status(np.zeros(16, dtype=np.int16)) is FrameStatus.OK
    counts = np.zeros(16, dtype=np.int16)
    counts[3] = 19  # CLAUDE.md: max < 20 per pixel per pulse
    assert frame_data_status(counts) is FrameStatus.OK


def test_frame_data_status_rejects_negative_counts():
    counts = np.zeros(16, dtype=np.int16)
    counts[5] = -1
    assert frame_data_status(counts) is FrameStatus.DATA_CHECK_FAILED


def test_frame_data_status_rejects_non_integer_dtype():
    assert (
        frame_data_status(np.zeros(16, dtype=np.float32))
        is FrameStatus.DATA_CHECK_FAILED
    )


def test_integrate_frame_raises_on_negative_counts(op):
    """P1 acceptance: never a silent result."""
    x = np.zeros(NPIX, dtype=np.int16)
    x[10] = -3
    bad = np.zeros(NPIX, dtype=bool)
    with pytest.raises(DataCheckFailed, match="negative"):
        integrate_frame(op, x, bad, bad, denominator(op, bad))


def test_integrate_frame_raises_on_non_integer_dtype(op):
    """P1 acceptance: no silent dense fallback for float data."""
    x = np.zeros(NPIX, dtype=np.float32)
    bad = np.zeros(NPIX, dtype=bool)
    with pytest.raises(DataCheckFailed, match="integer"):
        integrate_frame(op, x, bad, bad, denominator(op, bad))


@pytest.mark.parametrize("name", ["x", "bad", "base_bad"])
def test_integrate_frame_rejects_wrong_pixel_count(op, name):
    arrays = {
        "x": np.zeros(NPIX, dtype=np.int16),
        "bad": np.zeros(NPIX, dtype=bool),
        "base_bad": np.zeros(NPIX, dtype=bool),
    }
    arrays[name] = arrays[name][:-1]
    with pytest.raises(ValueError, match=name):
        integrate_frame(
            op,
            arrays["x"],
            arrays["bad"],
            arrays["base_bad"],
            denominator(op, np.zeros(NPIX, dtype=bool)),
        )


def test_integrate_frame_reports_scalars(op):
    x = np.zeros(NPIX, dtype=np.int16)
    x[[1, 2, 3, 4]] = np.array([1, 2, 5, 3], dtype=np.int16)
    bad = np.zeros(NPIX, dtype=bool)
    bad[4] = True  # a flagged photon: excluded from photons_valid
    base_bad = np.zeros(NPIX, dtype=bool)
    result = integrate_frame(op, x, bad, base_bad, denominator(op, base_bad))
    assert result.photons_valid == pytest.approx(1 + 2 + 5)
    assert result.n_bad_pixels == 1
    assert result.n_frame_specific == 1
    assert result.max_count == 5  # max over all nonzero pixels, flagged or not
    assert result.signal.dtype == np.float64
