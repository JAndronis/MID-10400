"""Static and per-cell base masks of the AGIPD SAXS pass."""

from __future__ import annotations

import numpy as np
import pytest
from extra_geom import agipd_asic_seams

from analysis.common.masks import (
    UnexpectedMaskBits,
)
from analysis.saxs.config import NPIX, AgipdSaxsConfig
from analysis.saxs.masks import (
    BaseMaskAccumulator,
    build_static_bad,
    load_masks,
    load_pixel_mask,
    save_masks,
)

pytest.importorskip("pyFAI")

STATIC_BIT = 1 << 0  # OFFSET_OUT_OF_THRESHOLD
DYNAMIC_BIT = 1 << 12  # VALUE_OUT_OF_RANGE
UNEXPECTED_BIT = 1 << 22  # NON_STANDARD_SIZE, never set in corrected AGIPD files


def write_mask(tmp_path, array, name="mask.npy"):
    path = tmp_path / name
    np.save(path, array)
    return path


def mask_cfg(tmp_path, **overrides) -> AgipdSaxsConfig:
    base = {
        "proposal": 10400,
        "run": 423,
        "npt": 500,
        "geometry_file": None,
        "pixel_mask_file": None,
        "use_asic_seams": False,
    }
    return AgipdSaxsConfig(**{**base, **overrides})


# ── static mask: shape and convention asserts ─────────────────────────────────
@pytest.mark.parametrize("shape", [(16, 512, 128), (512, 128), (8192, 128)])
def test_load_pixel_mask_accepts_every_documented_shape(tmp_path, shape):
    array = np.zeros(shape, dtype=np.uint8)
    array.reshape(-1)[:100] = 1
    bad = load_pixel_mask(write_mask(tmp_path, array))
    assert bad.shape == (NPIX,)
    assert bad.dtype == np.bool_


def test_single_module_mask_is_broadcast_over_all_sixteen(tmp_path):
    array = np.zeros((512, 128), dtype=np.uint8)
    array[0, 0] = 1
    bad = load_pixel_mask(write_mask(tmp_path, array))
    # one pixel per module, at the same position in each
    assert bad.sum() == 16
    assert np.array_equal(np.flatnonzero(bad), np.arange(16) * 512 * 128)


def test_load_pixel_mask_rejects_a_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        load_pixel_mask(write_mask(tmp_path, np.zeros((16, 256, 128), dtype=np.uint8)))


def test_load_pixel_mask_rejects_a_float_dtype(tmp_path):
    """A float mask makes 'non-zero = excluded' ambiguous."""
    with pytest.raises(TypeError, match="dtype"):
        load_pixel_mask(write_mask(tmp_path, np.zeros((512, 128), dtype=np.float32)))


def test_load_pixel_mask_rejects_an_all_excluded_mask(tmp_path):
    with pytest.raises(ValueError, match="every pixel"):
        load_pixel_mask(write_mask(tmp_path, np.ones((512, 128), dtype=np.uint8)))


@pytest.mark.parametrize("flag", [255, -1])
def test_non_zero_is_excluded_whatever_the_value(tmp_path, flag):
    """The convention is non-zero, not equal-to-one."""
    array = np.zeros((512, 128), dtype=np.int16)
    array[3, 4] = flag
    bad = load_pixel_mask(write_mask(tmp_path, array))
    assert bad.sum() == 16


def test_boolean_masks_are_accepted(tmp_path):
    array = np.zeros((16, 512, 128), dtype=bool)
    array[2, 5, 6] = True
    assert load_pixel_mask(write_mask(tmp_path, array)).sum() == 1


# ── static mask: composition ──────────────────────────────────────────────────
def test_asic_seams_are_repeated_over_sixteen_modules():
    """NON_STANDARD_SIZE is never set in corrected AGIPD files, so the double-width
    ASIC-edge pixels must come from the geometry package instead."""
    static = build_static_bad(mask_cfg_seams_only())
    seams = agipd_asic_seams()
    assert np.array_equal(
        static.bad.reshape(16, 512, 128), np.broadcast_to(seams, (16, 512, 128))
    )
    assert static.bad.sum() == 16 * seams.sum()
    rows = np.flatnonzero(seams.any(axis=1))
    assert np.array_equal(
        rows, np.union1d(np.arange(64, 512, 64), np.arange(63, 511, 64))
    )


def mask_cfg_seams_only() -> AgipdSaxsConfig:
    return AgipdSaxsConfig(
        proposal=10400,
        run=423,
        npt=500,
        geometry_file=None,
        pixel_mask_file=None,
        use_asic_seams=True,
    )


def test_static_mask_is_the_union_of_seams_and_the_pixel_mask(tmp_path):
    """The pixel mask overlaps the seams; the static mask is a union, not a sum."""
    seams = np.broadcast_to(agipd_asic_seams(), (16, 512, 128))
    pixel = np.zeros((16, 512, 128), dtype=np.uint8)
    pixel[0, 0, :10] = 1  # not a seam row
    pixel[0, 64, :10] = 1  # a seam row, so deliberately overlapping

    static = build_static_bad(
        mask_cfg(
            tmp_path,
            use_asic_seams=True,
            pixel_mask_file=str(write_mask(tmp_path, pixel, "pixel.npy")),
        )
    )
    by_name = {source.name: source for source in static.sources}
    assert by_name["asic_seams"].n_excluded == int(seams.sum())
    assert by_name["pixel_mask"].n_excluded == 20  # own count, before the OR
    assert by_name["pixel_mask"].sha256 is not None
    # the ten seam-row pixels are counted once, not twice
    assert static.n_excluded == int(seams.sum()) + 10


def test_static_hash_is_stable_and_content_sensitive(tmp_path):
    pixel = np.zeros((16, 512, 128), dtype=np.uint8)
    pixel[0, 0, :10] = 1
    cfg = mask_cfg(tmp_path, pixel_mask_file=str(write_mask(tmp_path, pixel, "a.npy")))
    assert build_static_bad(cfg).sha256 == build_static_bad(cfg).sha256

    pixel[0, 1, 0] = 1
    other = mask_cfg(
        tmp_path, pixel_mask_file=str(write_mask(tmp_path, pixel, "b.npy"))
    )
    assert build_static_bad(other).sha256 != build_static_bad(cfg).sha256


def test_static_mask_is_read_only():
    assert not build_static_bad(mask_cfg_seams_only()).bad.flags.writeable


# ── base masks: helpers ───────────────────────────────────────────────────────
PX_PER_MODULE = 512 * 128


def blank_train(n_frames: int = 1) -> np.ndarray:
    return np.zeros((16, n_frames, 512, 128), dtype=np.uint32)


def flag(mask: np.ndarray, frame: int, pixels, bit: int) -> None:
    """Set ``bit`` at flat pixel indices ``pixels`` of one frame."""
    flat = mask.reshape(16, mask.shape[1], -1)
    for pixel in np.atleast_1d(pixels).tolist():
        flat[pixel // PX_PER_MODULE, frame, pixel % PX_PER_MODULE] |= np.uint32(bit)


def accumulate(op, static_mask, trains, *, mask_bits=0xFFFFFFFF, expected=(0, 12)):
    acc = BaseMaskAccumulator(
        op, static_mask, mask_bits=mask_bits, expected_bits=expected
    )
    for mask, cells in trains:
        acc.update(mask, cells)
    return acc


# ── base masks: the majority vote ─────────────────────────────────────────────
@pytest.mark.parametrize(
    ("votes", "expected_bad"),
    [(5, True), (4, False)],
)
def test_majority_vote_over_eight_samples(op, static_mask, votes, expected_bad):
    """Strict majority: 5 of 8 is bad, 4 of 8 is not."""
    pixel = int(np.flatnonzero(~static_mask.bad)[0])
    trains = []
    for sample in range(8):
        mask = blank_train()
        if sample < votes:
            flag(mask, 0, pixel, STATIC_BIT)
        trains.append((mask, np.array([0], dtype=np.uint16)))

    base = accumulate(op, static_mask, trains).finalise()
    assert base.n_samples.tolist() == [8]
    assert bool(base.base_bad[0, pixel]) is expected_bad


@pytest.mark.parametrize(("n", "votes", "expected_bad"), [(5, 3, True), (5, 2, False)])
def test_majority_vote_with_an_odd_sample_count(
    op, static_mask, n, votes, expected_bad
):
    pixel = int(np.flatnonzero(~static_mask.bad)[0])
    trains = []
    for sample in range(n):
        mask = blank_train()
        if sample < votes:
            flag(mask, 0, pixel, STATIC_BIT)
        trains.append((mask, np.array([0], dtype=np.uint16)))
    base = accumulate(op, static_mask, trains).finalise()
    assert bool(base.base_bad[0, pixel]) is expected_bad


def test_base_mask_is_a_superset_of_the_static_mask(op, static_mask):
    """Static pixels vote on every sample, so they always survive the vote."""
    base = accumulate(
        op, static_mask, [(blank_train(2), np.array([0, 1], dtype=np.uint16))] * 3
    ).finalise()
    for index in range(base.cells.size):
        assert not (static_mask.bad & ~base.base_bad[index]).any()


# ── base masks: the unseen-cell fallback ──────────────────────────────────────
def test_unseen_cell_falls_back_to_the_static_mask(op, static_mask):
    # give cell 0 flags of its own, so its base mask is a strict superset of
    # the static mask and the fallback is distinguishable from it
    mask = blank_train()
    flag(mask, 0, np.flatnonzero(~static_mask.bad)[:500], STATIC_BIT)
    base = accumulate(
        op, static_mask, [(mask, np.array([0], dtype=np.uint16))]
    ).finalise()

    assert base.has_cell(0)
    assert not base.has_cell(154)

    seen_bad, _ = base.for_cell(0)
    unseen_bad, unseen_denominator = base.for_cell(154)
    assert np.array_equal(unseen_bad, base.static_bad)
    assert np.array_equal(unseen_denominator, base.static_denominator)
    assert not np.array_equal(unseen_bad, seen_bad)
    assert int(seen_bad.sum()) == int(unseen_bad.sum()) + 500


# ── base masks: exactness for frame/cell disagreement, both directions ────────


# ── base masks: bit bookkeeping ───────────────────────────────────────────────
def test_bits_present_accumulates_across_trains(op, static_mask):
    first, second = blank_train(), blank_train()
    flag(first, 0, 0, STATIC_BIT)
    flag(second, 0, 1, DYNAMIC_BIT)
    cells = np.array([0], dtype=np.uint16)
    base = accumulate(op, static_mask, [(first, cells), (second, cells)]).finalise()
    assert base.bits_present == STATIC_BIT | DYNAMIC_BIT
    assert base.unexpected_bits == 0


def test_an_unexpected_bit_warns_and_is_recorded(op, static_mask):
    mask = blank_train()
    flag(mask, 0, 0, STATIC_BIT)
    flag(mask, 0, 1, UNEXPECTED_BIT)
    acc = accumulate(op, static_mask, [(mask, np.array([0], dtype=np.uint16))])

    with pytest.warns(UnexpectedMaskBits, match="NON_STANDARD_SIZE"):
        base = acc.finalise()
    assert base.unexpected_bits == UNEXPECTED_BIT
    assert base.bits_present == STATIC_BIT | UNEXPECTED_BIT


def test_expected_bits_do_not_warn(op, static_mask, recwarn):
    mask = blank_train()
    flag(mask, 0, 0, DYNAMIC_BIT)
    accumulate(op, static_mask, [(mask, np.array([0], dtype=np.uint16))]).finalise()
    assert not [w for w in recwarn if issubclass(w.category, UnexpectedMaskBits)]


# ── base masks: input validation ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mask", "cells", "error", "match"),
    [
        (np.zeros((16, 512, 128), np.uint32), [0], ValueError, "shape"),
        (np.zeros((16, 1, 512, 128), np.uint16), [0], TypeError, "uint32"),
        (np.zeros((16, 2, 512, 128), np.uint32), [0], ValueError, "cell_ids"),
        (None, None, ValueError, "no frames"),
    ],
)
def test_the_accumulator_refuses_what_it_cannot_vote_on(
    op, static_mask, mask, cells, error, match
):
    """A wrong train layout or dtype at ``update``; no sample at ``finalise``."""
    acc = BaseMaskAccumulator(op, static_mask, mask_bits=0xFFFFFFFF, expected_bits=[0])
    with pytest.raises(error, match=match):
        if mask is None:
            acc.finalise()
        else:
            acc.update(mask, np.array(cells))


# ── persistence ───────────────────────────────────────────────────────────────
def test_save_load_roundtrip(op, static_mask, tmp_path):
    mask = blank_train(2)
    flag(mask, 0, [0, 1, 2], STATIC_BIT)
    base = accumulate(
        op, static_mask, [(mask, np.array([0, 1], dtype=np.uint16))] * 2
    ).finalise()

    loaded = load_masks(save_masks(base, tmp_path / "masks.npz"))
    assert loaded.sha256 == base.sha256
    assert loaded.static_sha256 == base.static_sha256
    assert loaded.operator_sha256 == op.sha256
    assert loaded.bits_present == base.bits_present
    assert loaded.unexpected_bits == base.unexpected_bits
    for name in ("cells", "base_bad", "denominators", "n_samples", "static_bad"):
        assert np.array_equal(getattr(loaded, name), getattr(base, name))
    np.testing.assert_allclose(
        loaded.static_denominator, base.static_denominator, rtol=0, atol=0
    )


def test_load_rejects_corrupt_masks(op, static_mask, tmp_path):
    base = accumulate(
        op, static_mask, [(blank_train(), np.array([0], dtype=np.uint16))]
    ).finalise()
    path = save_masks(base, tmp_path / "masks.npz")
    with np.load(path) as handle:
        arrays = {name: handle[name] for name in handle.files}
    arrays["base_bad_packed"] = arrays["base_bad_packed"].copy()
    arrays["base_bad_packed"][0, 0] ^= np.uint8(0xFF)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="corrupt"):
        load_masks(path)
