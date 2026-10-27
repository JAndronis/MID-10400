"""The per-frame bad-pixel mask and the ``BadPixels`` bit vocabulary."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.common.masks import bits_to_mask, describe_bits, frame_bad

OFFSET = 1 << 0  # OFFSET_OUT_OF_THRESHOLD
RANGE = 1 << 12  # VALUE_OUT_OF_RANGE


def test_a_frame_s_bad_pixels_are_its_selected_bits_or_the_static_mask():
    static = np.zeros(8, dtype=bool)
    static[7] = True
    mask = np.zeros((2, 4), dtype=np.uint32)
    mask.reshape(-1)[[1, 2, 3]] = [OFFSET, RANGE, OFFSET | RANGE]

    assert np.flatnonzero(frame_bad(mask, 0xFFFFFFFF, static)).tolist() == [1, 2, 3, 7]
    assert np.flatnonzero(frame_bad(mask, OFFSET, static)).tolist() == [1, 3, 7]
    assert np.flatnonzero(frame_bad(mask, RANGE, static)).tolist() == [2, 3, 7]


def test_a_frame_of_another_size_than_the_static_mask_is_refused():
    with pytest.raises(ValueError):
        frame_bad(np.zeros(8, np.uint32), 0xFFFFFFFF, np.zeros(7, dtype=bool))


def test_bit_positions_become_a_uint32_mask():
    assert bits_to_mask([0, 12]) == 4097
    assert bits_to_mask([31]) == 2**31
    assert bits_to_mask([]) == 0
    for outside in (32, -1):
        with pytest.raises(ValueError):
            bits_to_mask([outside])


def test_set_bits_are_named_by_position():
    pytest.importorskip("extra.calibration")
    assert describe_bits(OFFSET | (1 << 22)) == (
        "0 OFFSET_OUT_OF_THRESHOLD, 22 NON_STANDARD_SIZE"
    )
    assert describe_bits(0) == "none"
