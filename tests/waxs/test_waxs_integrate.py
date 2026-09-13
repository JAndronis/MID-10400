"""The error model, the per-frame integration and the NaN-equivalence gate.

The gate is the correctness argument for the whole design: the production path
builds pyFAI's sparse matrix once with the static mask and carries the dynamic
mask as NaN, and it has to give the same answer as the obvious slow path that
passes the union to ``integrate1d(mask=)`` each frame.
"""

from __future__ import annotations

import numpy as np
import pytest
from waxs_mockrun import MODULE_SHAPE, PHOTON_KEV, READ_NOISE_KEV  # noqa: E402

from analysis.common.masks import frame_bad  # noqa: E402
from analysis.common.status import DataCheckFailed, FrameStatus  # noqa: E402
from analysis.waxs.integrate import (  # noqa: E402
    ErrorModel,
    frame_data_status,
    integrate_frame,
)
from analysis.waxs.selftest import (  # noqa: E402
    SelfTestFailed,
    reference_frame,
    reference_integrator,
    run_selftest,
)

DYNAMIC_BIT = np.uint32(1 << 21)


def make_frame(op, seed=0, rate=0.6, dynamic=500):
    """A dense keV frame plus its ``(mask, bad)`` pair."""
    rng = np.random.default_rng(seed)
    values = (
        PHOTON_KEV * rng.poisson(rate, MODULE_SHAPE)
        + rng.normal(0.0, READ_NOISE_KEV, MODULE_SHAPE)
    ).astype(np.float32)
    mask = np.zeros(MODULE_SHAPE, dtype=np.uint32)
    mask.reshape(-1)[rng.choice(values.size, dynamic, replace=False)] |= DYNAMIC_BIT
    return values, mask, frame_bad(mask, 0xFFFFFFFF, op.static_bad)


# ── the error model ──────────────────────────────────────────────────────────
def test_the_variance_is_not_clamped(model):
    """§3 D3: unclamped is the unbiased estimator, and the negatives cancel."""
    x = np.array([-1.7, -0.1, 0.0, 5.0], dtype=np.float32)
    got = model.variance(x)
    assert got[0] < 0  # the whole point: a negative pixel gives negative variance
    assert got == pytest.approx(model.read_noise_kev**2 + model.photon_energy_kev * x)


def test_a_synthetic_poisson_frame_recovers_its_input_variance():
    rng = np.random.default_rng(0)
    rate = 3.0
    counts = rng.poisson(rate, MODULE_SHAPE)
    frame = (counts * PHOTON_KEV + rng.normal(0, READ_NOISE_KEV, MODULE_SHAPE)).astype(
        np.float32
    )
    model = ErrorModel(READ_NOISE_KEV, PHOTON_KEV)
    # Var(x) = E^2 Var(N) + sigma^2, and Var(N) = rate for a Poisson count.
    expected = PHOTON_KEV**2 * rate + READ_NOISE_KEV**2
    assert float(model.variance(frame).mean()) == pytest.approx(expected, rel=0.01)


def test_the_model_refuses_nonsense():
    with pytest.raises(ValueError, match="read_noise_kev"):
        ErrorModel(0.0, PHOTON_KEV)
    with pytest.raises(ValueError, match="photon_energy_kev"):
        ErrorModel(0.32, -1.0)


def test_the_model_hash_moves_with_its_contents():
    first = ErrorModel(0.32, PHOTON_KEV)
    assert first.sha256 == ErrorModel(0.32, PHOTON_KEV).sha256
    assert first.sha256 != ErrorModel(0.33, PHOTON_KEV).sha256


# ── the NaN-equivalence gate ─────────────────────────────────────────────────
def test_the_nan_path_equals_the_per_frame_mask_reference(operator, model):
    op, ai = operator
    values, _, bad = make_frame(op)
    got = integrate_frame(ai, op, model, values, bad, max_abs_kev=1e3)
    want_s, want_n, want_v = reference_frame(
        reference_integrator(op), op, model, values, bad
    )
    occupied = want_n > 0
    assert occupied.any()
    for name, mine, theirs in (
        ("signal", got.signal, want_s),
        ("normalization", got.normalization, want_n),
        ("variance", got.variance, want_v),
    ):
        relative = np.abs(mine[occupied] - theirs[occupied]) / np.maximum(
            np.abs(theirs[occupied]), 1e-12
        )
        assert relative.max() < 1e-9, name


def test_the_production_path_never_rebuilds_the_matrix(operator, model):
    """The whole reason for the NaN: a changing mask evicts pyFAI's cache."""
    op, ai = operator
    for seed in range(4):
        values, _, bad = make_frame(op, seed=seed)
        integrate_frame(ai, op, model, values, bad, max_abs_kev=1e3)
    assert len(ai.engines) == 1


def test_the_dynamic_mask_leaves_the_denominator_exact(operator, model):
    """A masked pixel must leave both the numerator and the denominator."""
    op, ai = operator
    values, _, bad = make_frame(op, dynamic=0)
    clean = integrate_frame(ai, op, model, values, bad, max_abs_kev=1e3)

    extra = bad.copy()
    keep = np.flatnonzero(~bad)
    extra[keep[:2000]] = True
    masked = integrate_frame(ai, op, model, values, extra, max_abs_kev=1e3)

    assert masked.n_bad_pixels == clean.n_bad_pixels + 2000
    assert (masked.normalization <= clean.normalization + 1e-9).all()
    assert (masked.normalization < clean.normalization).any()


def test_the_selftest_reports_the_worst_frame(operator, model):
    op, ai = operator
    frames = [(f[0], f[2]) for f in (make_frame(op, seed=s) for s in range(3))]
    report = run_selftest(ai, op, model, frames, max_abs_kev=1e3)
    assert report.n_frames == 3
    assert (
        max(
            report.max_rel_signal, report.max_rel_normalization, report.max_rel_variance
        )
        <= report.tolerance
    )


def test_the_selftest_refuses_an_empty_frame_list(operator, model):
    op, ai = operator
    with pytest.raises(SelfTestFailed, match="no frames"):
        run_selftest(ai, op, model, [], max_abs_kev=1e3)


def test_the_selftest_catches_a_broken_nan_path(operator, model, monkeypatch):
    """If pyFAI ever stops dropping non-finite pixels, this is what notices."""
    import analysis.waxs.selftest as selftest_module

    op, ai = operator
    values, _, bad = make_frame(op)

    def wrong(*args, **kwargs):
        signal, normalization, variance = original(*args, **kwargs)
        return signal * 1.05, normalization, variance

    original = selftest_module.reference_frame
    monkeypatch.setattr(selftest_module, "reference_frame", wrong)
    with pytest.raises(SelfTestFailed, match="disagrees"):
        run_selftest(ai, op, model, [(values, bad)], max_abs_kev=1e3)


# ── the data check ───────────────────────────────────────────────────────────
def test_a_non_finite_kept_pixel_fails_the_check(operator, model):
    """The dynamic mask is a NaN, so a NaN in the data is indistinguishable."""
    op, ai = operator
    values, _, bad = make_frame(op)
    values.reshape(-1)[np.flatnonzero(~op.static_bad)[0]] = np.nan
    assert (
        frame_data_status(values, op.static_bad, 1e3) is FrameStatus.DATA_CHECK_FAILED
    )
    with pytest.raises(DataCheckFailed, match="non-finite"):
        integrate_frame(ai, op, model, values, bad, max_abs_kev=1e3)


def test_an_extreme_kept_pixel_fails_the_check(operator, model):
    """§3 D6: jf2 carries unflagged pixels at 1.8e5 keV in every cell."""
    op, ai = operator
    values, _, bad = make_frame(op)
    values.reshape(-1)[np.flatnonzero(~op.static_bad)[0]] = 1.8e5
    assert (
        frame_data_status(values, op.static_bad, 1e3) is FrameStatus.DATA_CHECK_FAILED
    )
    with pytest.raises(DataCheckFailed, match="max"):
        integrate_frame(ai, op, model, values, bad, max_abs_kev=1e3)


def test_an_extreme_pixel_under_the_static_mask_is_not_this_frames_problem(
    operator, model
):
    op, ai = operator
    values, _, bad = make_frame(op)
    values.reshape(-1)[np.flatnonzero(op.static_bad)[0]] = 1.8e5
    assert frame_data_status(values, op.static_bad, 1e3) is FrameStatus.OK
    integrate_frame(ai, op, model, values, bad, max_abs_kev=1e3)


def test_negative_variance_bins_are_counted_not_hidden(operator):
    """The ledger is where the unclamped variance stays visible."""
    op, ai = operator
    # An almost-empty frame: the readout noise dominates and a good fraction of
    # pixels sit below -sigma^2/E.
    rng = np.random.default_rng(3)
    values = rng.normal(-0.5, 0.32, MODULE_SHAPE).astype(np.float32)
    model = ErrorModel(0.32, PHOTON_KEV)
    result = integrate_frame(
        ai, op, model, values, op.static_bad.copy(), max_abs_kev=1e3
    )
    assert result.n_negative_variance_bins > 0
    assert (result.variance < 0).any()


def test_shape_disagreements_are_refused(operator, model):
    op, ai = operator
    values, _, bad = make_frame(op)
    with pytest.raises(ValueError, match="pixels"):
        integrate_frame(ai, op, model, values.reshape(-1)[:100], bad, max_abs_kev=1e3)
    with pytest.raises(ValueError, match="bad has shape"):
        integrate_frame(ai, op, model, values, bad[:100], max_abs_kev=1e3)
