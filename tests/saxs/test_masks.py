"""Static and per-cell base masks (context file §6.3; phase P2 acceptance)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from extra_geom import agipd_asic_seams

from analysis.saxs.config import NPIX, FirstPassConfig
from analysis.saxs.masks import (
    BaseMaskAccumulator,
    UnexpectedMaskBits,
    bits_to_mask,
    build_static_bad,
    describe_bits,
    evenly_spaced,
    frame_bad,
    load_masks,
    load_pixel_mask,
    save_masks,
)
from analysis.saxs.sparse import denominator, integrate_frame

pytest.importorskip("pyFAI")

STATIC_BIT = 1 << 0  # OFFSET_OUT_OF_THRESHOLD
DYNAMIC_BIT = 1 << 12  # VALUE_OUT_OF_RANGE
UNEXPECTED_BIT = 1 << 22  # NON_STANDARD_SIZE — never set in these files


def write_mask(tmp_path, array, name="mask.npy"):
    path = tmp_path / name
    np.save(path, array)
    return path


def mask_cfg(tmp_path, **overrides) -> FirstPassConfig:
    base = {
        "proposal": 10400,
        "run": 423,
        "npt": 500,
        "geometry_file": None,
        "custom_mask_file": None,
        "lobe_mask_file": None,
        "use_asic_seams": False,
    }
    return FirstPassConfig(**{**base, **overrides})


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


@pytest.mark.parametrize("shape", [(512, 129), (16, 256, 128), (1024,), (8192, 64)])
def test_load_pixel_mask_rejects_a_wrong_shape(tmp_path, shape):
    with pytest.raises(ValueError, match="shape"):
        load_pixel_mask(write_mask(tmp_path, np.zeros(shape, dtype=np.uint8)))


def test_load_pixel_mask_rejects_a_float_dtype(tmp_path):
    """A float mask makes 'non-zero = excluded' ambiguous."""
    with pytest.raises(TypeError, match="dtype"):
        load_pixel_mask(write_mask(tmp_path, np.zeros((512, 128), dtype=np.float32)))


def test_load_pixel_mask_rejects_an_all_excluded_mask(tmp_path):
    with pytest.raises(ValueError, match="every pixel"):
        load_pixel_mask(write_mask(tmp_path, np.ones((512, 128), dtype=np.uint8)))


@pytest.mark.parametrize("flag", [1, 255, -1, 7])
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
    """NON_STANDARD_SIZE is never set in these files, so the double-width
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


def mask_cfg_seams_only() -> FirstPassConfig:
    return FirstPassConfig(
        proposal=10400,
        run=423,
        npt=500,
        geometry_file=None,
        custom_mask_file=None,
        lobe_mask_file=None,
        use_asic_seams=True,
    )


def test_static_mask_is_the_union_of_its_sources(tmp_path):
    custom = np.zeros((16, 512, 128), dtype=np.uint8)
    custom[0, 0, :10] = 1
    lobe = np.zeros((16, 512, 128), dtype=np.uint8)
    lobe[0, 0, 5:15] = 1  # deliberately overlapping the custom mask

    static = build_static_bad(
        mask_cfg(
            tmp_path,
            custom_mask_file=str(write_mask(tmp_path, custom, "custom.npy")),
            lobe_mask_file=str(write_mask(tmp_path, lobe, "lobe.npy")),
        )
    )
    assert static.n_excluded == 15  # union, not sum
    by_name = {source.name: source for source in static.sources}
    assert by_name["custom_mask"].n_excluded == 10
    assert by_name["lobe_mask"].n_excluded == 10  # own count, before the OR
    assert by_name["custom_mask"].sha256 != by_name["lobe_mask"].sha256
    assert all(source.sha256 is not None for source in static.sources)


def test_static_mask_records_the_seam_source_without_a_path():
    static = build_static_bad(mask_cfg_seams_only())
    (source,) = static.sources
    assert source.name == "asic_seams"
    assert source.path is None and source.sha256 is None


def test_static_hash_is_stable_and_content_sensitive(tmp_path):
    custom = np.zeros((16, 512, 128), dtype=np.uint8)
    custom[0, 0, :10] = 1
    cfg = mask_cfg(
        tmp_path, custom_mask_file=str(write_mask(tmp_path, custom, "a.npy"))
    )
    assert build_static_bad(cfg).sha256 == build_static_bad(cfg).sha256

    custom[0, 1, 0] = 1
    other = mask_cfg(
        tmp_path, custom_mask_file=str(write_mask(tmp_path, custom, "b.npy"))
    )
    assert build_static_bad(other).sha256 != build_static_bad(cfg).sha256


def test_static_mask_is_read_only():
    assert not build_static_bad(mask_cfg_seams_only()).bad.flags.writeable


# ── frame_bad ─────────────────────────────────────────────────────────────────
def test_frame_bad_ors_in_the_static_mask():
    static = np.zeros(NPIX, dtype=bool)
    static[7] = True
    mask = np.zeros((16, 512, 128), dtype=np.uint32)
    mask.reshape(-1)[9] = STATIC_BIT
    bad = frame_bad(mask, 0xFFFFFFFF, static)
    assert np.array_equal(np.flatnonzero(bad), [7, 9])


def test_frame_bad_honours_a_narrowed_mask_bits():
    static = np.zeros(NPIX, dtype=bool)
    mask = np.zeros((16, 512, 128), dtype=np.uint32)
    mask.reshape(-1)[0] = STATIC_BIT
    mask.reshape(-1)[1] = DYNAMIC_BIT
    assert frame_bad(mask, 0xFFFFFFFF, static).sum() == 2
    assert np.array_equal(np.flatnonzero(frame_bad(mask, STATIC_BIT, static)), [0])
    assert np.array_equal(np.flatnonzero(frame_bad(mask, DYNAMIC_BIT, static)), [1])


def test_frame_bad_rejects_a_size_mismatch():
    with pytest.raises(ValueError, match="pixels"):
        frame_bad(
            np.zeros((16, 512, 128), dtype=np.uint32),
            0xFFFFFFFF,
            np.zeros(NPIX - 1, dtype=bool),
        )


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
    [(8, True), (5, True), (4, False), (3, False), (0, False)],
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


def test_per_cell_structure_is_preserved(op, static_mask, make_mask_train):
    """Each cell keeps its own base mask, not a run-wide average."""
    rng = np.random.default_rng(0)
    cells = np.array([0, 1, 2], dtype=np.uint16)
    trains = [(make_mask_train(rng, cells), cells) for _ in range(4)]
    base = accumulate(op, static_mask, trains).finalise()

    assert base.cells.tolist() == [0, 1, 2]
    assert base.n_samples.tolist() == [4, 4, 4]
    assert not np.array_equal(base.base_bad[0], base.base_bad[1])
    # the per-frame dynamic bit appears in only one of four samples, so the
    # majority vote must drop it
    assert (
        base.base_bad.sum(axis=1).max()
        < static_mask.bad.sum() + 16 * 0.02 * (PX_PER_MODULE) + 40
    )


def test_denominators_match_a_denominator_built_from_scratch(op, static_mask):
    base = accumulate(
        op, static_mask, [(blank_train(2), np.array([0, 1], dtype=np.uint16))] * 3
    ).finalise()
    for index in range(base.cells.size):
        np.testing.assert_allclose(
            base.denominators[index],
            denominator(op, base.base_bad[index]),
            rtol=0,
            atol=0,
        )
    np.testing.assert_allclose(
        base.static_denominator, denominator(op, static_mask.bad), rtol=0, atol=0
    )


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

    seen_bad, seen_denominator = base.for_cell(0)
    # views onto the stored arrays, not copies
    assert np.shares_memory(seen_bad, base.base_bad)
    assert np.shares_memory(seen_denominator, base.denominators)

    unseen_bad, unseen_denominator = base.for_cell(154)
    assert np.shares_memory(unseen_bad, base.static_bad)
    assert np.shares_memory(unseen_denominator, base.static_denominator)
    assert np.array_equal(unseen_bad, base.static_bad)
    assert not np.array_equal(unseen_bad, seen_bad)
    assert int(seen_bad.sum()) == int(unseen_bad.sum()) + 500


def test_the_fallback_still_integrates_exactly(op, static_mask, make_frame):
    """An unseen cell costs more per-frame corrections, never accuracy."""
    rng = np.random.default_rng(17)
    base = accumulate(
        op, static_mask, [(blank_train(), np.array([0], dtype=np.uint16))]
    ).finalise()

    x = make_frame(rng, 0.02)
    bad = static_mask.bad | (rng.random(NPIX) < 0.001)
    fallback_bad, fallback_denominator = base.for_cell(154)
    result = integrate_frame(op, x, bad, fallback_bad, fallback_denominator)
    np.testing.assert_allclose(
        result.normalization, denominator(op, bad), rtol=1e-9, atol=1e-15
    )


# ── base masks: exactness for frame/cell disagreement, both directions ────────
def test_frame_cell_disagreement_is_exact_in_both_directions(
    op, static_mask, make_frame
):
    """The P2-to-P1 seam, with a base mask from a real majority vote."""
    rng = np.random.default_rng(23)
    trains = []
    for _ in range(4):
        mask = blank_train()
        flag(mask, 0, rng.choice(NPIX, 5000, replace=False), STATIC_BIT)
        trains.append((mask, np.array([0], dtype=np.uint16)))
    # one pixel flagged in every sample, so the vote certainly calls it bad
    always_bad = int(np.flatnonzero(~static_mask.bad)[0])
    for mask, _ in trains:
        flag(mask, 0, always_bad, STATIC_BIT)

    base = accumulate(op, static_mask, trains).finalise()
    base_bad, base_denominator = base.for_cell(0)

    bad = np.array(base_bad, copy=True)
    bad[always_bad] = False  # cell says bad, frame says good  → Ω added back
    newly_bad = np.flatnonzero(~base_bad)[:2000]
    bad[newly_bad] = True  # cell says good, frame says bad  → Ω subtracted
    assert (bad & ~base_bad).sum() == 2000
    assert (~bad & base_bad).sum() == 1

    x = make_frame(rng, 0.02)
    result = integrate_frame(op, x, bad, base_bad, base_denominator)
    assert result.n_frame_specific == 2001
    np.testing.assert_allclose(
        result.normalization, denominator(op, bad), rtol=1e-9, atol=1e-15
    )


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


def test_bits_to_mask_and_describe_bits():
    assert bits_to_mask([0, 12]) == (1 << 0) | (1 << 12)
    assert bits_to_mask([]) == 0
    with pytest.raises(ValueError, match="out of range"):
        bits_to_mask([32])
    described = describe_bits((1 << 0) | (1 << 22))
    assert "OFFSET_OUT_OF_THRESHOLD" in described
    assert "NON_STANDARD_SIZE" in described
    assert describe_bits(0) == "none"


# ── base masks: input validation ──────────────────────────────────────────────
def test_update_rejects_a_wrong_mask_shape(op, static_mask):
    acc = BaseMaskAccumulator(op, static_mask, mask_bits=0xFFFFFFFF, expected_bits=[0])
    with pytest.raises(ValueError, match="shape"):
        acc.update(np.zeros((16, 512, 128), dtype=np.uint32), np.array([0]))


def test_update_rejects_a_non_uint32_mask(op, static_mask):
    acc = BaseMaskAccumulator(op, static_mask, mask_bits=0xFFFFFFFF, expected_bits=[0])
    with pytest.raises(TypeError, match="uint32"):
        acc.update(np.zeros((16, 1, 512, 128), dtype=np.uint16), np.array([0]))


def test_update_rejects_a_cell_id_count_mismatch(op, static_mask):
    acc = BaseMaskAccumulator(op, static_mask, mask_bits=0xFFFFFFFF, expected_bits=[0])
    with pytest.raises(ValueError, match="cell_ids"):
        acc.update(blank_train(2), np.array([0]))


def test_finalise_without_samples_raises(op, static_mask):
    acc = BaseMaskAccumulator(op, static_mask, mask_bits=0xFFFFFFFF, expected_bits=[0])
    with pytest.raises(ValueError, match="no frames"):
        acc.finalise()


# ── train sampling ────────────────────────────────────────────────────────────
def test_evenly_spaced_spans_the_run():
    train_ids = np.arange(1000, 4000)
    picked = evenly_spaced(train_ids, 8)
    assert picked.size == 8
    assert picked[0] == 1000
    assert picked[-1] == 3999
    assert np.all(np.diff(picked) > 0)


def test_evenly_spaced_is_capped_by_the_run_length():
    assert evenly_spaced(np.arange(3), 8).tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("train_ids", "n", "match"),
    [(np.array([]), 8, "no trains"), (np.arange(10), 0, "positive")],
)
def test_evenly_spaced_rejects_bad_input(train_ids, n, match):
    with pytest.raises(ValueError, match=match):
        evenly_spaced(train_ids, n)


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


def test_base_masks_hash_tracks_the_operator(op, static_mask):
    """Masks built against a different operator must not compare equal."""
    base = accumulate(
        op, static_mask, [(blank_train(), np.array([0], dtype=np.uint16))]
    ).finalise()
    other = dataclasses.replace(base, operator_sha256="0" * 64)
    assert other.operator_sha256 != base.operator_sha256
