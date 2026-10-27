"""The photon encoding: what a train becomes, and what can be read back."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.common.status import DataCheckFailed, FrameStatus
from analysis.xpcs.extract import encode_train
from analysis.xpcs.store import FlagBase, delta_decode, delta_encode

N_ENV = 400
P = 6


def brute_force_train(rng, n_env=N_ENV, n_frames=P, density=0.05):
    """A train built from dense arrays, so the encoding has an independent answer."""
    envelope = np.sort(rng.choice(16 * 512 * 128, n_env, replace=False)).astype(
        np.int32
    )
    data = np.zeros((16 * 512 * 128, n_frames), dtype=np.int16)
    mask = np.zeros((16 * 512 * 128, n_frames), dtype=np.uint32)
    hit = rng.random((n_env, n_frames)) < density
    data[envelope] = np.where(hit, rng.integers(1, 4, hit.shape), 0)
    flagged = rng.random((n_env, n_frames)) < 0.1
    mask[envelope] = np.where(flagged, rng.choice([1, 1 << 12], flagged.shape), 0)
    mask[envelope[0], 0] = 1 << 5  # a bit outside the mask below
    return (
        envelope,
        data.reshape(16, 512, 128, n_frames).transpose(0, 3, 1, 2),
        mask.reshape(16, 512, 128, n_frames).transpose(0, 3, 1, 2),
    )


@pytest.fixture
def train():
    rng = np.random.default_rng(3)
    envelope, data, mask = brute_force_train(rng)
    cells = np.arange(P, dtype=np.uint16)
    return envelope, data, mask, cells


def encode(train, base=None, bits=(1 << 12) | 1):
    envelope, data, mask, cells = train
    return encode_train(
        data,
        mask,
        envelope,
        bits,
        base or FlagBase.empty(),
        train_id=77,
        sequence=2,
        pulse_ids=2 * np.arange(P) + 200,
        cell_ids=cells,
    )


def test_photons_round_trip_to_the_dense_frames(train):
    envelope, data, _, _ = train
    photons = encode(train)
    dense = photons.frames(envelope.size).toarray()
    expected = data.transpose(1, 0, 2, 3).reshape(P, -1)[:, envelope]
    np.testing.assert_array_equal(dense, expected)
    assert photons.status is FrameStatus.OK
    assert photons.sequence == 2
    assert photons.pulse_ids.tolist() == (2 * np.arange(P) + 200).tolist()


def test_only_the_configured_mask_bits_flag_a_pixel(train):
    envelope, _, mask, _ = train
    photons = encode(train)
    flat = mask.transpose(1, 0, 2, 3).reshape(P, -1)[:, envelope]
    for frame in range(P):
        expected = np.flatnonzero((flat[frame] & ((1 << 12) | 1)) != 0)
        np.testing.assert_array_equal(photons.flagged_frame(frame), expected)
    unflagged_bit = np.flatnonzero(flat[0] == (1 << 5))
    assert (
        unflagged_bit.size
        and not np.isin(unflagged_bit, photons.flagged_frame(0)).any()
    )


def test_flagged_any_is_the_union_of_the_frames(train):
    envelope, _, mask, _ = train
    photons = encode(train)
    flat = mask.transpose(1, 0, 2, 3).reshape(P, -1)[:, envelope]
    expected = ((flat & ((1 << 12) | 1)) != 0).any(axis=0)
    np.testing.assert_array_equal(photons.flagged_any(envelope.size), expected)


def test_a_flag_base_changes_the_bytes_and_never_the_flags(train):
    envelope, _, _, cells = train
    plain = encode(train)
    base = FlagBase.from_samples([(cells, [plain.flagged_frame(f) for f in range(P)])])
    based = encode(train, base)
    assert based.flag_pixel.size < plain.flag_pixel.size
    for frame in range(P):
        np.testing.assert_array_equal(
            based.flagged_frame(frame), plain.flagged_frame(frame)
        )
    np.testing.assert_array_equal(
        based.flagged_any(envelope.size), plain.flagged_any(envelope.size)
    )


def test_a_cell_the_base_has_never_seen_lists_all_its_flags(train):
    envelope, _, _, cells = train
    plain = encode(train)
    base = FlagBase.from_samples(
        [(cells[:3], [plain.flagged_frame(f) for f in range(3)])]
    )
    assert base.of_cell(5).size == 0
    based = encode(train, base)
    np.testing.assert_array_equal(based.flagged_frame(5), plain.flagged_frame(5))


def test_the_base_is_what_more_than_half_the_samples_flag():
    cells = np.array([4, 4, 9], dtype=np.uint16)
    a = (cells, [np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([8])])
    b = (cells, [np.array([2, 3, 5]), np.array([1, 2, 3]), np.array([8])])
    c = (cells, [np.array([2, 3]), np.array([2]), np.array([9])])
    base = FlagBase.from_samples([a, b, c])
    # cell 4 appears in 6 frames: pixel 2 in 6, 3 in 5, 1 in 3, 5 in 1
    assert base.of_cell(4).tolist() == [2, 3]
    assert base.of_cell(9).tolist() == [8]
    assert base.cells.tolist() == [4, 9]


@pytest.mark.parametrize("bad", [-1, 256])
def test_counts_outside_a_byte_fail_the_data_check(train, bad):
    envelope, data, mask, cells = train
    data = data.astype(np.int32)
    flat = data.transpose(0, 2, 3, 1).reshape(-1, P)
    flat[envelope[1], 1] = bad
    data = flat.reshape(16, 512, 128, P).transpose(0, 3, 1, 2)
    with pytest.raises(DataCheckFailed):
        encode((envelope, data, mask, cells))


def test_float_data_fail_the_data_check(train):
    envelope, data, mask, cells = train
    with pytest.raises(DataCheckFailed, match="not photons"):
        encode((envelope, data.astype(np.float32), mask, cells))


def test_delta_encoding_restarts_each_frame_and_inverts(train):
    pixel = np.array([5, 9, 300, 7, 8, 0, 2], dtype=np.uint32)
    ptr = np.array([0, 3, 3, 5, 7, 7], dtype=np.int64)  # frames 1 and 4 are empty
    step = delta_encode(pixel, ptr)
    assert step.tolist() == [5, 4, 291, 7, 1, 0, 2]
    np.testing.assert_array_equal(delta_decode(step, ptr), pixel)
    empty = np.zeros(0, dtype=np.uint32)
    assert delta_decode(delta_encode(empty, np.zeros(4, np.int64)), ptr[:1]).size == 0
    photons = encode(train)
    step = delta_encode(photons.pixel, photons.frame_ptr)
    np.testing.assert_array_equal(delta_decode(step, photons.frame_ptr), photons.pixel)
