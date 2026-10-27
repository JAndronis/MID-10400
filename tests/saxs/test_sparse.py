"""Gather kernel, base-mask correction and per-frame data checks."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.common.status import DataCheckFailed, FrameStatus
from analysis.saxs.config import METHOD, NPIX
from analysis.saxs.operator import SparseOperator
from analysis.saxs.sparse import (
    denominator,
    frame_data_status,
    gather,
    integrate_frame,
)

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
        beam_center=None,
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
    """The base-mask correction end to end, small enough to check by hand.

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
    # negative coefficient, so the two summation orders differ there at 1e-16.
    # The self-test compares only bins with N > 0 for the same reason.
    np.testing.assert_allclose(
        result.normalization, denominator(op, bad), rtol=1e-9, atol=1e-15
    )
    assert result.photons_valid == pytest.approx(5.0)
    assert result.n_bad_pixels == 1
    assert result.n_frame_specific == 2
    assert result.max_count == 7


@pytest.mark.parametrize(
    ("dtype", "value", "match"),
    [(np.int16, -3, "negative"), (np.float32, 1, "integer")],
)
def test_data_that_are_not_photon_counts_fail_the_data_check(op, dtype, value, match):
    """Negative or non-integer data are DATA_CHECK_FAILED, never a silent result.

    Counts up to 19 per pixel, as corrected AGIPD data hold, pass.
    """
    counts = np.zeros(NPIX, dtype=np.int16)
    counts[3] = 19
    assert frame_data_status(counts) is FrameStatus.OK
    x = np.zeros(NPIX, dtype=dtype)
    x[10] = value
    assert frame_data_status(x) is FrameStatus.DATA_CHECK_FAILED
    bad = np.zeros(NPIX, dtype=bool)
    with pytest.raises(DataCheckFailed, match=match):
        integrate_frame(op, x, bad, bad, denominator(op, bad))


def test_integrate_frame_rejects_a_mask_of_another_size(op):
    bad = np.zeros(NPIX, dtype=bool)
    with pytest.raises(ValueError, match="base_bad"):
        integrate_frame(
            op, np.zeros(NPIX, np.int16), bad, bad[:-1], denominator(op, bad)
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
