"""The envelope, and the rings inside it, on EXtra-geom's idealised detector."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.saxs.config import NPIX
from analysis.xpcs.rings import build_envelope, build_rings, pixel_map


@pytest.fixture(scope="module")
def pixels(cfg, geom):
    return pixel_map(cfg, geom)


@pytest.fixture(scope="module")
def static_bad():
    bad = np.zeros(NPIX, dtype=bool)
    bad[::97] = True
    return bad


def test_rings_are_the_half_open_intervals_between_the_edges(cfg, pixels):
    q = pixels.q_nm
    for ring, (lo, hi) in enumerate(
        zip(cfg.ring_edges_nm, cfg.ring_edges_nm[1:], strict=False)
    ):
        assert (pixels.label == ring).sum() == ((q >= lo) & (q < hi)).sum()
    assert pixels.n_rings == len(cfg.ring_edges_nm) - 1
    outside = (q < cfg.ring_edges_nm[0]) | (q >= cfg.ring_edges_nm[-1])
    assert (pixels.label[outside] == -1).all()


def test_the_envelope_holds_every_ring_pixel_and_a_margin_of_the_annulus(
    cfg, with_config, geom, pixels
):
    envelope = build_envelope(pixels, cfg)
    assert envelope.pixels.dtype == np.int32
    assert (np.diff(envelope.pixels) > 0).all()
    assert not envelope.pixels.flags.writeable
    assert np.isin(np.flatnonzero(pixels.label >= 0), envelope.pixels).all()
    assert envelope.size > (pixels.label >= 0).sum()

    tighter = build_envelope(pixels, with_config(envelope_margin_mm=0.0))
    assert np.isin(tighter.pixels, envelope.pixels).all()
    assert tighter.size < envelope.size


def test_the_margin_covers_a_beam_centre_shift_of_that_size(cfg, pixels):
    """A pixel within the margin of a ring edge stays in the store."""
    envelope = build_envelope(pixels, cfg)
    sin_theta = pixels.q_nm * cfg.saxs.wavelength_m * 1e9 / (4 * np.pi)
    radius = cfg.saxs.sdd_m * 1e3 * np.tan(2 * np.arcsin(sin_theta))
    in_ring_radius = radius[pixels.label >= 0]
    near = (radius >= in_ring_radius.min() - 4.0) & (
        radius <= in_ring_radius.max() + 4.0
    )
    assert np.isin(np.flatnonzero(near), envelope.pixels).all()


def test_the_envelope_is_identified_by_its_pixels(cfg, with_config, pixels):
    assert build_envelope(pixels, cfg).sha256 == build_envelope(pixels, cfg).sha256
    other = build_envelope(pixels, with_config(envelope_margin_mm=2.0))
    assert other.sha256 != build_envelope(pixels, cfg).sha256


def test_static_bad_pixels_stay_in_the_envelope_and_leave_the_rings(
    cfg, pixels, static_bad
):
    envelope = build_envelope(pixels, cfg)
    rings = build_rings(pixels, envelope, static_bad, cfg)
    assert rings.label.size == envelope.size
    flagged = static_bad[envelope.pixels]
    assert flagged.any()
    assert (rings.label[flagged] == -1).all()
    unflagged = ~flagged
    np.testing.assert_array_equal(
        rings.label[unflagged], pixels.label[envelope.pixels][unflagged]
    )
    assert rings.centres_nm().tolist() == pytest.approx([0.25, 0.375])


def test_the_static_mask_changes_the_rings_but_not_the_envelope(
    cfg, pixels, static_bad
):
    envelope = build_envelope(pixels, cfg)
    clean = build_rings(pixels, envelope, np.zeros(NPIX, dtype=bool), cfg)
    masked = build_rings(pixels, envelope, static_bad, cfg)
    assert clean.sha256 != masked.sha256


def test_a_ring_without_a_good_pixel_is_an_error_not_a_silent_drop(
    cfg, pixels, static_bad
):
    envelope = build_envelope(pixels, cfg)
    everything = np.ones(NPIX, dtype=bool)
    with pytest.raises(ValueError, match="hold no good pixel"):
        build_rings(pixels, envelope, everything, cfg)


def test_rings_can_come_from_a_masks_file(cfg, with_config, tmp_path):
    masks = np.zeros((2, 16, 512, 128), dtype=bool)
    masks[0, 0, :10] = True
    masks[1, 3, 100:120] = True
    path = tmp_path / "qmsk.npy"
    np.save(path, masks)
    from_file = with_config(ring_masks_file=str(path))

    pixels = pixel_map(from_file)
    envelope = build_envelope(pixels, from_file)
    assert envelope.size == masks.sum()
    rings = build_rings(pixels, envelope, np.zeros(NPIX, dtype=bool), from_file)
    assert rings.n_rings == 2
    assert np.bincount(rings.label).tolist() == [
        int(masks[0].sum()),
        int(masks[1].sum()),
    ]
    assert rings.edges_nm == ()


def test_overlapping_ring_masks_are_refused(with_config, tmp_path):
    masks = np.zeros((2, 16, 512, 128), dtype=bool)
    masks[:, 0, :4] = True
    path = tmp_path / "qmsk.npy"
    np.save(path, masks)
    with pytest.raises(ValueError, match="two rings"):
        pixel_map(with_config(ring_masks_file=str(path)))


def test_ring_masks_of_the_wrong_shape_are_refused(with_config, tmp_path):
    path = tmp_path / "qmsk.npy"
    np.save(path, np.zeros((2, 8, 512, 128), dtype=bool))
    with pytest.raises(ValueError, match="expected"):
        pixel_map(with_config(ring_masks_file=str(path)))
