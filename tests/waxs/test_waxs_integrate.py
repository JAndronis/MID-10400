"""The per-frame path and the NaN-equivalence gate.

The gate is the correctness argument for the whole design. The production path
builds pyFAI's sparse matrix once with the static mask and carries every other
exclusion as NaN, and it has to give the same answer as the obvious slow path
that passes them to ``integrate1d(mask=)`` each frame. Both paths integrate the
same counted frame.
"""

from __future__ import annotations

import numpy as np
import pytest
from waxs_mockrun import MODULE_SHAPE, PHOTON_KEV, READ_NOISE_KEV  # noqa: E402

from analysis.common.status import DataCheckFailed  # noqa: E402
from analysis.waxs.integrate import (  # noqa: E402
    extreme_pixels,
    frame_maxima,
    integrate_frame,
    integrate_prepared,
    prepare_frame,
)
from analysis.waxs.photons import ASIC_ID  # noqa: E402
from analysis.waxs.selftest import (  # noqa: E402
    SelfTestFailed,
    reference_frame,
    reference_integrator,
    run_selftest,
)

DYNAMIC_BIT = np.uint32(1 << 21)
OFFSETS = np.array([0.12, -0.08, 0.05, 0.15, -0.03, 0.09, -0.11, 0.02])


def make_frame(op, seed=0, rate=0.6, dynamic=500, offsets=OFFSETS):
    """A dense keV frame with a zero level per ASIC, plus ``(mask, dynamic)``."""
    rng = np.random.default_rng(seed)
    truth = rng.poisson(rate, MODULE_SHAPE)
    values = (
        PHOTON_KEV * truth
        + rng.normal(0.0, READ_NOISE_KEV, MODULE_SHAPE)
        + offsets[ASIC_ID].reshape(MODULE_SHAPE)
    ).astype(np.float32)
    mask = np.zeros(MODULE_SHAPE, dtype=np.uint32)
    mask.reshape(-1)[rng.choice(values.size, dynamic, replace=False)] |= DYNAMIC_BIT
    return values, mask, mask.reshape(-1) != 0


def union(op, dynamic):
    return dynamic | op.static_bad


# ── counting inside the frame path ───────────────────────────────────────────
def test_each_asic_level_is_recovered_from_the_frame_itself(operator, counting):
    op, _ = operator
    values, _, dynamic = make_frame(op)
    prepared = prepare_frame(op, counting, values, dynamic, max_abs_kev=1e3)
    assert np.abs(prepared.asic_level - OFFSETS).max() < 0.005
    assert prepared.n_level_missing == 0


def test_the_counts_are_the_photons_that_were_put_in(operator, counting):
    op, ai = operator
    rng = np.random.default_rng(0)
    truth = rng.poisson(0.6, MODULE_SHAPE).reshape(-1)
    values, _, dynamic = make_frame(op)
    prepared = prepare_frame(op, counting, values, dynamic, max_abs_kev=1e3)
    kept = ~prepared.excluded
    assert np.array_equal(prepared.photons[kept], truth[kept])
    assert np.isnan(prepared.photons[prepared.excluded]).all()

    result = integrate_frame(ai, op, counting, values, dynamic, max_abs_kev=1e3)
    assert result.n_photons == int(truth[kept].sum())
    assert result.energy_valid == pytest.approx(
        float(values.reshape(-1)[kept].sum()), rel=1e-6
    )
    assert result.count_energy_kev == pytest.approx(PHOTON_KEV, rel=1e-7)
    assert np.array_equal(result.asic_level, prepared.asic_level)


def test_the_zero_level_comes_from_static_masked_pixels_too(operator, counting):
    """``level_pixels="all_good"``: shadowed pixels are good level pixels."""
    import dataclasses

    op, _ = operator
    values, _, dynamic = make_frame(op)
    flat = values.reshape(-1)
    # Shift only what the .edf masks: an all_good level moves, a kept one not.
    shadow = flat.copy()
    shadow[op.static_bad] += 0.2
    every = prepare_frame(op, counting, shadow, dynamic, max_abs_kev=1e3)
    kept_only = prepare_frame(
        op,
        dataclasses.replace(counting, level_pixels="kept"),
        shadow,
        dynamic,
        max_abs_kev=1e3,
    )
    assert np.abs(kept_only.asic_level - OFFSETS).max() < 0.005
    assert (every.asic_level - kept_only.asic_level).min() > 0.05


def test_an_asic_without_a_level_is_not_counted(operator, counting):
    """Fewer than ``min_pixels`` usable pixels: no level, so no counts there.

    An ASIC the masks remove entirely has no level either, but nothing on it is
    lost, so it is not counted as missing.
    """
    op, ai = operator
    values, _, dynamic = make_frame(op)
    dead = ASIC_ID == 1
    # Leave the ASIC's edge pixels good: integrated, but never level pixels.
    rows, cols = np.indices(MODULE_SHAPE)
    edge = (((rows % 256) < 3) | ((cols % 256) < 3)).reshape(-1)
    dynamic = dynamic | (dead & ~edge) | (ASIC_ID == 6)
    prepared = prepare_frame(op, counting, values, dynamic, max_abs_kev=1e3)

    assert np.isnan(prepared.asic_level[[1, 6]]).all()
    assert np.isfinite(np.delete(prepared.asic_level, [1, 6])).all()
    assert prepared.n_level_missing == 1
    assert prepared.excluded[dead].all()
    result = integrate_frame(ai, op, counting, values, dynamic, max_abs_kev=1e3)
    assert result.n_level_missing == 1
    assert result.n_bad_pixels == int(prepared.excluded.sum())


def _edge_of_asic_2():
    rows, cols = np.indices(MODULE_SHAPE)
    edge = (((rows % 256) < 3) | ((cols % 256) < 3)).reshape(-1)
    return ~((ASIC_ID == 2) & edge)


@pytest.mark.parametrize(
    ("excluded", "match"),
    [
        (lambda op: np.ones_like(op.static_bad), "nothing left to integrate"),
        # Only ASIC 2's edge is left, and an edge never yields a zero level.
        (lambda op: _edge_of_asic_2(), "no zero level"),
    ],
)
def test_a_frame_with_nothing_left_to_integrate_fails(
    operator, counting, excluded, match
):
    """The one case that must not become a row of zeros placed as measured."""
    op, ai = operator
    values, _, _ = make_frame(op)
    with pytest.raises(DataCheckFailed, match=match):
        integrate_frame(ai, op, counting, values, excluded(op), max_abs_kev=1e3)


# ── the NaN-equivalence gate ─────────────────────────────────────────────────
def test_the_nan_path_equals_the_per_frame_mask_reference(operator, counting):
    op, ai = operator
    values, _, dynamic = make_frame(op)
    prepared = prepare_frame(op, counting, values, dynamic, max_abs_kev=1e3)
    got = integrate_prepared(ai, op, prepared)
    want = reference_frame(
        reference_integrator(op), op, prepared.photons, prepared.excluded
    )
    occupied = want[1] > 0
    assert occupied.any()
    for name, mine, theirs in zip(
        ("signal", "normalization", "variance"), got, want, strict=True
    ):
        relative = np.abs(mine[occupied] - theirs[occupied]) / np.maximum(
            np.abs(theirs[occupied]), 1e-12
        )
        assert relative.max() < 1e-9, name


def test_the_production_path_never_rebuilds_the_matrix(operator, counting):
    """A per-frame mask would rebuild pyFAI's sparse matrix, hence the NaN path."""
    op, ai = operator
    for seed in range(4):
        values, _, dynamic = make_frame(op, seed=seed)
        integrate_frame(ai, op, counting, values, dynamic, max_abs_kev=1e3)
    assert len(ai.engines) == 1


def test_the_dynamic_mask_leaves_the_denominator_exact(operator, counting):
    """A masked pixel must leave both the numerator and the denominator."""
    op, ai = operator
    values, _, dynamic = make_frame(op, dynamic=0)
    clean = integrate_frame(ai, op, counting, values, dynamic, max_abs_kev=1e3)

    extra = dynamic.copy()
    keep = np.flatnonzero(~union(op, dynamic))
    extra[keep[:2000]] = True
    masked = integrate_frame(ai, op, counting, values, extra, max_abs_kev=1e3)

    assert masked.n_bad_pixels == clean.n_bad_pixels + 2000
    assert (masked.normalization <= clean.normalization + 1e-9).all()
    assert (masked.normalization < clean.normalization).any()


def test_the_selftest_catches_a_broken_nan_path(operator, counting, monkeypatch):
    """If pyFAI ever stops dropping non-finite pixels, this is what notices."""
    import analysis.waxs.selftest as selftest_module

    op, ai = operator
    values, _, dynamic = make_frame(op)

    def wrong(*args, **kwargs):
        signal, normalization, variance = original(*args, **kwargs)
        return signal * 1.05, normalization, variance

    original = selftest_module.reference_frame
    monkeypatch.setattr(selftest_module, "reference_frame", wrong)
    with pytest.raises(SelfTestFailed, match="disagrees"):
        run_selftest(ai, op, counting, [(values, dynamic)], max_abs_kev=1e3)


# ── the data check ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("value", "under_static", "n_extreme"),
    [(np.nan, False, 1), (1.8e5, False, 1), (1.8e5, True, 0)],
)
def test_a_wild_kept_pixel_is_excluded_and_the_frame_kept(
    operator, counting, value, under_static, n_extreme
):
    """A non-finite or extreme pixel costs that pixel, never the frame.

    Failing the frame would lose every good pixel in it, and such pixels (salt
    Bragg spots from an evaporating droplet) track crystallisation. The pixel is
    excluded before counting, so its value never reaches a level or a count. A
    pixel the static mask already removes is not this check's to count.
    """
    op, ai = operator
    values, _, dynamic = make_frame(op)
    bad = union(op, dynamic)
    candidates = op.static_bad if under_static else ~bad
    pixel = np.flatnonzero(candidates)[0]
    values.reshape(-1)[pixel] = value

    assert extreme_pixels(values, bad, 1e3)[pixel] == bool(n_extreme)
    result = integrate_frame(ai, op, counting, values, dynamic, max_abs_kev=1e3)
    assert result.n_extreme_pixels == n_extreme
    assert result.n_bad_pixels == int(bad.sum()) + n_extreme
    assert np.isfinite(result.signal).all()
    assert (result.normalization > 0).any()
    assert np.abs(result.asic_level - OFFSETS).max() < 0.005


def test_excluding_an_extreme_pixel_equals_masking_it_by_hand(operator, counting):
    """The exclusion is the same act the two masks already perform.

    The pixel joins the excluded set and takes the NaN path, and the level
    estimator ignores a value that far above its cut, so the sums must be
    bit-identical to passing the pixel as masked in the first place.
    """
    op, ai = operator
    values, _, dynamic = make_frame(op)
    pixel = np.flatnonzero(~union(op, dynamic))[0]
    values.reshape(-1)[pixel] = 1.8e5

    automatic = integrate_frame(ai, op, counting, values, dynamic, max_abs_kev=1e3)
    by_hand_mask = dynamic.copy()
    by_hand_mask[pixel] = True
    clean = values.copy()
    clean.reshape(-1)[pixel] = 0.0
    by_hand = integrate_frame(ai, op, counting, clean, by_hand_mask, max_abs_kev=1e3)

    for name in ("signal", "normalization", "variance", "asic_level"):
        assert np.array_equal(getattr(automatic, name), getattr(by_hand, name)), name


def test_the_callers_mask_is_not_modified(operator, counting):
    """``dynamic`` belongs to the worker's loop; extending it in place would leak
    one frame's exclusions into the next."""
    op, ai = operator
    values, _, dynamic = make_frame(op)
    values.reshape(-1)[np.flatnonzero(~union(op, dynamic))[0]] = 1.8e5
    before = dynamic.copy()
    integrate_frame(ai, op, counting, values, dynamic, max_abs_kev=1e3)
    assert np.array_equal(dynamic, before)


def test_the_two_maxima_tell_a_masked_wild_pixel_from_a_live_one(operator, counting):
    """``max_kev`` covers what reaches the sums, ``max_kev_static`` what the
    static mask keeps.

    An ordinary ``max_kev`` beside a huge ``max_kev_static`` means the wild value
    never reached a stored number. A value ``data.mask`` already drops is not
    an extreme pixel, since the check did not remove it. A NaN maximum records
    the reason itself.
    """
    op, ai = operator
    values, _, dynamic = make_frame(op)
    bad = union(op, dynamic)
    flat = values.reshape(-1)

    clean = integrate_frame(ai, op, counting, values, dynamic, max_abs_kev=1e3)
    assert clean.max_kev == pytest.approx(flat[~bad].max())
    assert clean.max_kev_static == pytest.approx(flat[~op.static_bad].max())

    flagged = np.flatnonzero((~op.static_bad) & bad)[0]
    flat[flagged] = 1.8e5
    reaching, checked = frame_maxima(values, bad, op.static_bad)
    assert checked == pytest.approx(1.8e5) and reaching < 1e3
    masked = integrate_frame(ai, op, counting, values, dynamic, max_abs_kev=1e3)
    assert masked.n_extreme_pixels == 0
    assert masked.max_kev_static == pytest.approx(1.8e5)

    live = np.flatnonzero(~bad)[0]
    flat[live] = 1.8e5
    assert frame_maxima(values, bad, op.static_bad) == pytest.approx((1.8e5, 1.8e5))
    flat[live] = np.nan
    assert np.isnan(frame_maxima(values, bad, op.static_bad)).all()
