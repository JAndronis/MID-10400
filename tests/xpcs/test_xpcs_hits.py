"""The hit set: validated once, where the train ids enter the pass."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.xpcs.hits import HitSet


def test_ids_come_back_as_read_only_uint64():
    hits = HitSet.of([5, 9, 12], source="classifier v3")
    assert hits.train_ids.dtype == np.uint64
    assert hits.train_ids.tolist() == [5, 9, 12]
    assert not hits.train_ids.flags.writeable
    assert len(hits) == 3
    assert HitSet.of(np.array([3.0, 4.0])).train_ids.tolist() == [3, 4]
    for empty in ([], np.array([], dtype=float)):  # no hits is a result
        assert HitSet.of(empty).train_ids.dtype == np.uint64


@pytest.mark.parametrize(
    "ids, message",
    [
        ([3, 2], "sorted"),
        ([2, 2, 3], "duplicates"),
        ([-1, 4], "negative"),
        ([1.5, 2.0], "integers"),
        ([[1, 2], [3, 4]], "1-D"),
    ],
)
def test_ids_that_are_not_a_sorted_unique_integer_list_are_refused(ids, message):
    with pytest.raises(ValueError, match=message):
        HitSet.of(ids)


def test_the_digest_is_the_ids_alone():
    a = HitSet.of([1, 2, 3], source="one classifier")
    b = HitSet.of(np.array([1, 2, 3], dtype=np.int32), source="another")
    assert a.sha256 == b.sha256
    assert a.sha256 != HitSet.of([1, 2, 4]).sha256
    assert a.sha256 != HitSet.of([1, 2]).sha256
