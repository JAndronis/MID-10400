"""Train sampling for the base masks, the self-test and the readout noise."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.common.plan import evenly_spaced


def test_samples_are_spread_to_the_nearest_train_endpoints_included():
    # positions 0, 10/3, 20/3 and 10 round to 0, 3, 7 and 10
    assert evenly_spaced(np.arange(100, 111), 4).tolist() == [100, 103, 107, 110]


def test_a_run_shorter_than_the_sample_gives_every_train():
    assert evenly_spaced(np.array([5, 9, 12]), 8).tolist() == [5, 9, 12]


@pytest.mark.parametrize(
    ("train_ids", "n"), [(np.array([]), 8), (np.arange(10), 0), (np.ones((2, 2)), 2)]
)
def test_nothing_to_sample_from_or_nothing_asked_is_refused(train_ids, n):
    with pytest.raises(ValueError):
        evenly_spaced(train_ids, n)
