"""Pairing hits: the previous usable hit, no later one, every train accounted for."""

from __future__ import annotations

import numpy as np

from analysis.xpcs.config import Kernel
from analysis.xpcs.kernel import PairStatus as S
from analysis.xpcs.kernel import partners


def pair(ids, ok=None, sequence=None, kernel=Kernel.CLEAN, gap=10):
    ids = np.asarray(ids, dtype=np.uint64)
    ok = np.ones(ids.size, bool) if ok is None else np.asarray(ok, bool)
    sequence = np.zeros(ids.size, int) if sequence is None else np.asarray(sequence)
    result = partners(ids, ok, sequence, kernel, gap)
    return result.partner.tolist(), [S(s) for s in result.status]


def test_each_hit_pairs_with_the_previous_hit_inside_the_gap():
    partner, status = pair([10, 12, 30, 31])
    assert partner == [0, 10, 0, 30]
    assert status == [S.FIRST_HIT, S.PAIRED, S.GAP_TOO_LARGE, S.PAIRED]


def test_the_gap_is_inclusive():
    assert pair([10, 20], gap=10)[0] == [0, 10]
    assert pair([10, 21], gap=10)[0] == [0, 0]


def test_a_train_that_gave_no_photons_neither_has_a_partner_nor_is_one():
    partner, status = pair([10, 11, 12], ok=[True, False, True])
    assert partner == [0, 0, 10]
    assert status == [S.FIRST_HIT, S.NOT_OK, S.PAIRED]


def test_losing_the_only_nearby_partner_is_told_apart_from_isolation():
    _, status = pair([10, 30, 32], ok=[True, False, True])
    assert status == [S.FIRST_HIT, S.NOT_OK, S.PARTNER_FAILED]
    _, status = pair([30, 32], ok=[False, True])
    assert status == [S.NOT_OK, S.PARTNER_FAILED]
    _, status = pair([5, 40], ok=[False, True])
    assert status == [S.NOT_OK, S.FIRST_HIT]


def test_a_failed_hit_behind_a_usable_partner_changes_nothing():
    partner, status = pair([10, 12, 14], ok=[True, True, False])
    assert (partner, status) == ([0, 10, 0], [S.FIRST_HIT, S.PAIRED, S.NOT_OK])
    partner, status = pair([10, 12, 14, 15], ok=[True, False, True, True])
    assert partner == [0, 0, 10, 14]


def test_a_later_hit_never_changes_an_earlier_trains_partner():
    rng = np.random.default_rng(0)
    ids = np.sort(rng.choice(200, 40, replace=False)) + 1000
    ok = rng.random(40) > 0.2
    whole = partners(ids.astype(np.uint64), ok, np.zeros(40, int), Kernel.CLEAN, 10)
    for keep in (5, 17, 33):
        prefix = partners(
            ids[:keep].astype(np.uint64),
            ok[:keep],
            np.zeros(keep, int),
            Kernel.CLEAN,
            10,
        )
        np.testing.assert_array_equal(prefix.partner, whole.partner[:keep])
        np.testing.assert_array_equal(prefix.status, whole.status[:keep])


def test_every_hit_gets_a_status_and_only_paired_ones_a_partner():
    rng = np.random.default_rng(1)
    ids = np.sort(rng.choice(500, 80, replace=False)).astype(np.uint64) + 7
    ok = rng.random(80) > 0.3
    result = partners(ids, ok, np.zeros(80, int), Kernel.CLEAN, 10)
    paired = result.status == S.PAIRED
    assert ((result.partner != 0) == paired).all()
    assert (result.status[~ok] == S.NOT_OK).all()
    assert (result.status[ok] != S.NOT_OK).all()
    assert np.isin(result.partner[paired], ids[ok]).all()


def test_extra_speckles_chain_restarts_at_each_file_and_skips_unusable_trains():
    partner, status = pair(
        [10, 5000, 6000, 6001], sequence=[0, 0, 1, 1], kernel=Kernel.ES, gap=1
    )
    assert partner == [0, 10, 0, 6000]
    assert status == [S.FIRST_HIT, S.PAIRED, S.FIRST_HIT, S.PAIRED]
    partner, _ = pair([1, 2, 3], ok=[True, False, True], kernel=Kernel.ES)
    assert partner == [0, 0, 1], "an unusable train within a file is skipped"
