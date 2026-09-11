"""The P1 gate: sparse path versus the pyFAI engine (context file §6.6).

Every test here is an acceptance criterion of phase P1.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from analysis.saxs.config import NPIX
from analysis.saxs.selftest import (
    SelfTestFailed,
    compare_frame,
    reference_frame,
    run_selftest,
)
from analysis.saxs.sparse import denominator, integrate_frame

pytest.importorskip("pyFAI")

TOLERANCE = 1e-6


@pytest.mark.parametrize("occupancy", [0.005, 0.02, 0.05])
def test_matches_pyfai_at_each_occupancy(op, engine, occupancy, make_frame, make_masks):
    """P1 acceptance: S, N, V within 1e-6 at 0.5 %, 2 % and 5 % occupancy."""
    rng = np.random.default_rng(int(occupancy * 1e4))
    x = make_frame(rng, occupancy)
    bad, base_bad = make_masks(rng)
    report = run_selftest(
        engine,
        op,
        [(x, bad, base_bad, denominator(op, base_bad))],
        tolerance=TOLERANCE,
    )
    assert report.passed
    assert report.empty_bins_agree
    assert (
        max(
            report.max_rel_signal,
            report.max_rel_normalization,
            report.max_rel_variance,
        )
        < TOLERANCE
    )


def test_matches_pyfai_with_disagreement_in_both_directions(
    op, engine, make_frame, make_masks
):
    """P1 acceptance: frame flags differing from the cell in either sign.

    ``make_masks`` flips half the pixels from bad-in-cell to good-in-frame and
    half the other way, so both branches of the §6.4 normalization correction
    are exercised in one frame.
    """
    rng = np.random.default_rng(99)
    x = make_frame(rng, 0.02)
    bad, base_bad = make_masks(rng, disagree=4000)
    assert (bad & ~base_bad).sum() > 0
    assert (~bad & base_bad).sum() > 0

    base_denominator = denominator(op, base_bad)
    result = integrate_frame(op, x, bad, base_bad, base_denominator)
    assert result.n_frame_specific == 4000

    # The corrected denominator must equal the one built from scratch.
    np.testing.assert_allclose(
        result.normalization, denominator(op, bad), rtol=1e-9, atol=0.0
    )

    report = run_selftest(
        engine, op, [(x, bad, base_bad, base_denominator)], tolerance=TOLERANCE
    )
    assert report.passed


def test_matches_pyfai_with_a_random_static_mask(op, engine, make_frame):
    """A denser random mask, with no per-cell structure at all."""
    rng = np.random.default_rng(5)
    x = make_frame(rng, 0.02)
    bad = rng.random(NPIX) < 0.12
    base_bad = np.zeros(NPIX, dtype=bool)  # cell knows nothing; all frame-specific
    report = run_selftest(
        engine,
        op,
        [(x, bad, base_bad, denominator(op, base_bad))],
        tolerance=TOLERANCE,
    )
    assert report.passed


def test_matches_pyfai_on_an_empty_frame(op, engine, make_frame, make_masks):
    """Zero photons: signal and variance are zero, normalization is not."""
    rng = np.random.default_rng(13)
    x = np.zeros(NPIX, dtype=np.int16)
    bad, base_bad = make_masks(rng)
    result = integrate_frame(op, x, bad, base_bad, denominator(op, base_bad))
    assert not result.signal.any()
    assert not result.variance.any()
    assert result.max_count == 0
    assert run_selftest(
        engine,
        op,
        [(x, bad, base_bad, denominator(op, base_bad))],
        tolerance=TOLERANCE,
    ).passed


def test_empty_bins_are_identical_in_both_paths(op, engine, make_frame, make_masks):
    """P1 acceptance: the same bins are empty in the sparse and dense paths.

    The synthetic geometry populates all 500 bins, so bins are emptied on
    purpose: every pixel contributing to a chosen slice of q is flagged bad.
    Emptiness is tested as ``N > 0`` rather than ``nnz > 0`` because full-split
    coefficients can be marginally negative from rounding.
    """
    rng = np.random.default_rng(21)
    x = make_frame(rng, 0.02)
    target = np.arange(op.npt // 4, op.npt // 4 + 20)

    contributes = np.isin(op.bins, target)
    entry_to_pixel = np.repeat(np.arange(NPIX), np.diff(op.indptr.astype(np.int64)))
    bad = np.zeros(NPIX, dtype=bool)
    bad[np.unique(entry_to_pixel[contributes])] = True
    base_bad = np.zeros(NPIX, dtype=bool)

    result = integrate_frame(op, x, bad, base_bad, denominator(op, base_bad))
    ref = reference_frame(engine, op, x, bad)

    assert not (result.normalization[target] > 0).any()
    assert not (ref.normalization[target] > 0).any()
    assert np.array_equal(result.normalization > 0, ref.normalization > 0)

    _, _, _, agree = compare_frame(
        engine, op, x, bad, base_bad, denominator(op, base_bad)
    )
    assert agree


def test_selftest_raises_when_the_operator_is_wrong(op, engine, make_frame, make_masks):
    """A perturbed operator must fail the gate, not slip through."""
    rng = np.random.default_rng(31)
    x = make_frame(rng, 0.02)
    bad, base_bad = make_masks(rng)

    perturbed_coef = np.array(op.coef, copy=True)
    perturbed_coef[: perturbed_coef.size // 10] *= np.float32(1.001)
    perturbed_coef.flags.writeable = False
    broken = dataclasses.replace(op, coef=perturbed_coef)

    with pytest.raises(SelfTestFailed) as excinfo:
        run_selftest(
            engine,
            broken,
            [(x, bad, base_bad, denominator(broken, base_bad))],
            tolerance=TOLERANCE,
        )
    assert not excinfo.value.report.passed
    assert excinfo.value.report.as_provenance()["n_frames"] == 1


def test_selftest_needs_at_least_one_frame(op, engine):
    with pytest.raises(ValueError, match="at least one frame"):
        run_selftest(engine, op, [])


def test_selftest_aggregates_over_several_frames(op, engine, make_frame, make_masks):
    """The report carries the worst case over every frame compared."""
    rng = np.random.default_rng(41)
    frames = []
    for occupancy in (0.005, 0.02, 0.05):
        x = make_frame(rng, occupancy)
        bad, base_bad = make_masks(rng, disagree=1000)
        frames.append((x, bad, base_bad, denominator(op, base_bad)))
    report = run_selftest(engine, op, frames, tolerance=TOLERANCE)
    assert report.n_frames == 3
    assert report.passed
