"""The exact integer sums of one train, against brute force and extra-speckle."""

from __future__ import annotations

import numpy as np
import pytest
from xpcs_fakes import (
    es_chain,
    make_dense,
    make_flags,
    make_rings,
    make_train,
)

from analysis.xpcs.config import Kernel, MaskPolicy
from analysis.xpcs.kernel import train_sums

N_ENV, P, R = 300, 7, 3


@pytest.fixture
def world():
    rng = np.random.default_rng(11)
    rings = make_rings(rng, N_ENV, R)
    dense = [make_dense(rng, P, N_ENV) for _ in range(3)]
    flags = [make_flags(rng, P, N_ENV) for _ in range(3)]
    trains = [
        make_train(100 + i, d, f, sequence=0)
        for i, (d, f) in enumerate(zip(dense, flags, strict=True))
    ]
    return rings, dense, flags, trains


def any_flagged(flags):
    out = np.zeros(N_ENV, bool)
    for f in flags:
        out[f] = True
    return out


def reference(cur, cur_flags, prev, prev_flags, rings, policy=MaskPolicy.ANY_FRAME):
    """Dense numpy, written from the formulas in the kernel's docstring."""
    label = np.asarray(rings.label)
    good = label >= 0
    if policy is MaskPolicy.ANY_FRAME:
        good = good & ~any_flagged(cur_flags)
    out = {k: [] for k in ("on", "s_on", "cnt", "off", "s_cur", "s_prev", "cnt_pair")}
    for r in range(R):
        m = good & (label == r)
        x = cur[:, m].astype(np.int64)
        out["on"].append(x @ x.T)
        out["s_on"].append(x.sum(1))
        out["cnt"].append(m.sum())
        if prev is None:
            continue
        pair = good & (label == r)
        if policy is MaskPolicy.ANY_FRAME:
            pair = pair & ~any_flagged(prev_flags)
        xc, yp = cur[:, pair].astype(np.int64), prev[:, pair].astype(np.int64)
        out["off"].append(xc @ yp.T)
        out["s_cur"].append(xc.sum(1))
        out["s_prev"].append(yp.sum(1))
        out["cnt_pair"].append(pair.sum())
    return {k: np.array(v) for k, v in out.items()}


@pytest.mark.parametrize("policy", list(MaskPolicy))
def test_clean_sums_equal_the_dense_reference(world, policy):
    rings, dense, flags, trains = world
    sums = train_sums(trains[1], trains[0], rings, Kernel.CLEAN, policy)
    want = reference(dense[1], flags[1], dense[0], flags[0], rings, policy)
    assert sums.has_off and sums.partner_id == 100
    np.testing.assert_array_equal(sums.on, want["on"])
    np.testing.assert_array_equal(sums.s_on, want["s_on"])
    np.testing.assert_array_equal(sums.cnt, want["cnt"])
    np.testing.assert_array_equal(sums.off, want["off"])
    np.testing.assert_array_equal(sums.s_off_cur, want["s_cur"])
    np.testing.assert_array_equal(sums.s_off_prev, want["s_prev"])
    np.testing.assert_array_equal(sums.cnt_pair, want["cnt_pair"])
    assert sums.on.dtype == np.int32


def test_clean_without_a_partner_has_no_baseline_and_the_same_raw_sums(world):
    rings, _, _, trains = world
    alone = train_sums(trains[1], None, rings, Kernel.CLEAN)
    paired = train_sums(trains[1], trains[0], rings, Kernel.CLEAN)
    assert not alone.has_off and alone.partner_id == 0
    assert not alone.off.any() and not alone.cnt_pair.any()
    np.testing.assert_array_equal(alone.on, paired.on)
    np.testing.assert_array_equal(alone.cnt, paired.cnt)


def test_the_partners_mask_reaches_the_baseline_and_not_the_raw_sums(world):
    """ES lets the pair mask leak into the raw sums; the clean kernel does not."""
    rings, dense, flags, trains = world
    heavy = make_train(100, dense[0], [np.arange(0, N_ENV, 2)] * P)
    leaky = train_sums(trains[1], heavy, rings, Kernel.ES)
    clean = train_sums(trains[1], heavy, rings, Kernel.CLEAN)
    own = train_sums(trains[1], None, rings, Kernel.CLEAN)
    np.testing.assert_array_equal(clean.on, own.on)
    assert not np.array_equal(leaky.on, own.on)
    assert (leaky.cnt <= own.cnt).all() and (leaky.cnt < own.cnt).any()
    assert (clean.cnt_pair < clean.cnt).any()


def test_a_ring_that_keeps_no_pixel_gives_zero_sums(world):
    rings, dense, _, _ = world
    everything = [np.arange(N_ENV)] * P
    sums = train_sums(make_train(1, dense[0], everything), None, rings, Kernel.CLEAN)
    assert not sums.on.any() and not sums.cnt.any()


def test_a_partner_with_another_frame_count_is_refused(world):
    rings, dense, flags, trains = world
    short = make_train(5, dense[0][:4], flags[0][:4])
    with pytest.raises(ValueError, match="frames"):
        train_sums(trains[1], short, rings, Kernel.CLEAN)


def test_a_sum_int32_cannot_hold_is_an_error_not_a_wraparound():
    n = 40_000
    rings = make_rings(np.random.default_rng(0), n, 1, outside=0.0)
    full = np.full((2, n), 255, dtype=np.int32)
    with pytest.raises(OverflowError, match="int32"):
        train_sums(
            make_train(1, full, [np.zeros(0, int)] * 2), None, rings, Kernel.CLEAN
        )


@pytest.mark.parametrize("sparse", [False, True])
def test_the_es_kernel_reproduces_extra_speckles_sums_exactly(world, sparse):
    rings, dense, flags, trains = world
    chain = es_chain(dense, flags, rings, sparse=sparse)
    previous = None
    for train, expected in zip(trains, chain, strict=True):
        sums = train_sums(train, previous, rings, Kernel.ES)
        np.testing.assert_array_equal(sums.on, expected["ttc_on"].astype(np.int32))
        np.testing.assert_array_equal(sums.off, expected["ttc_off"].astype(np.int32))
        np.testing.assert_array_equal(sums.s_on, expected["s_on"].astype(np.int32))
        np.testing.assert_array_equal(
            sums.s_off_prev, expected["s_off"].astype(np.int32)
        )
        np.testing.assert_array_equal(sums.cnt, expected["s_cnt"].astype(np.int32))
        previous = train
