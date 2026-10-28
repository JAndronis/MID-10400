"""The envelope, and the rings inside it, on EXtra-geom's idealised detector."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.saxs.config import NPIX
from analysis.xpcs.config import (
    BRAGG_WINDOWS_NM,
    ES_RINGS_NM,
    RINGS_NM,
    roi_edges_nm,
    roi_kinds,
)
from analysis.xpcs.rings import build_envelope, build_rings, pixel_map, static_bad_for

#: The fcc 111 and 200 windows [nm⁻¹], as the user set them.
USER_BRAGG_WINDOWS_NM = ((0.57, 0.65), (0.68, 0.74))


def test_the_bragg_windows_are_the_users():
    assert BRAGG_WINDOWS_NM == USER_BRAGG_WINDOWS_NM


def test_no_ring_touches_a_bragg_window():
    """Bragg speckle gives a contrast near 1, so no ring may reach into a window."""
    for lo, hi in RINGS_NM:
        for w_lo, w_hi in USER_BRAGG_WINDOWS_NM:
            assert hi <= w_lo or lo >= w_hi, f"ring {lo}-{hi} in {w_lo}-{w_hi}"


def test_every_ring_is_plain_to_the_triage():
    assert roi_kinds(np.asarray(RINGS_NM)) == ["plain"] * len(RINGS_NM)


def test_the_rings_keep_extra_speckles_envelope():
    """The same outer floats keep the envelope, and so every photon store, valid."""
    assert RINGS_NM[0][0] == ES_RINGS_NM[0][0]
    assert RINGS_NM[-1][1] == ES_RINGS_NM[-1][1]
    assert {round(hi - lo, 9) for lo, hi in RINGS_NM} == {0.045}


def test_reference_beta_holds_one_value_per_ring_and_none_besides():
    from analysis.xpcs.snr import ES_REFERENCE_BETA, REFERENCE_BETA, ROI_MATCH_A

    centres_A = [(lo + hi) / 20 for lo, hi in RINGS_NM]
    own = [q for q in REFERENCE_BETA if q not in ES_REFERENCE_BETA]
    assert len(own) == len(centres_A)
    for q in centres_A:
        assert sum(abs(key - q) < ROI_MATCH_A for key in own) == 1, q


@pytest.fixture(scope="module")
def pixels(cfg, geom):
    return pixel_map(cfg, geom)


@pytest.fixture(scope="module")
def static_bad():
    bad = np.zeros(NPIX, dtype=bool)
    bad[::97] = True
    return bad


def test_rings_are_their_half_open_intervals(cfg, pixels):
    q = pixels.q_nm
    for ring, (lo, hi) in enumerate(cfg.rings_nm):
        assert (pixels.label == ring).sum() == ((q >= lo) & (q < hi)).sum()
    assert pixels.n_rings == len(cfg.rings_nm)
    outside = (q < cfg.rings_nm[0][0]) | (q >= cfg.rings_nm[-1][1])
    assert (pixels.label[outside] == -1).all()


def test_a_gap_between_rings_is_in_no_ring_but_in_the_envelope(
    cfg, with_config, geom, pixels
):
    gapped = with_config(rings_nm=((0.2, 0.25), (0.3, 0.45)))
    gap_pixels = pixel_map(gapped, geom)
    q = gap_pixels.q_nm
    gap = (q >= 0.25) & (q < 0.3)
    assert gap.any()
    assert (gap_pixels.label[gap] == -1).all()
    assert (gap_pixels.label[(q >= 0.3) & (q < 0.45)] == 1).all()
    assert (
        build_envelope(gap_pixels, gapped).sha256 == build_envelope(pixels, cfg).sha256
    ), "the envelope depends on the outer bounds only"


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
    clean = build_rings(pixels, envelope, np.zeros(NPIX, dtype=bool), cfg)
    assert clean.sha256 != rings.sha256


def test_other_ring_bounds_on_the_same_envelope_change_the_rings_sha(
    cfg, with_config, geom, pixels, static_bad
):
    """A results file merged on another ring set must not read as complete."""
    envelope = build_envelope(pixels, cfg)
    gapped = with_config(rings_nm=((0.2, 0.25), (0.3, 0.45)))
    gapped_pixels = pixel_map(gapped, geom)
    assert build_envelope(gapped_pixels, gapped).sha256 == envelope.sha256
    other = build_rings(gapped_pixels, envelope, static_bad, gapped)
    assert other.sha256 != build_rings(pixels, envelope, static_bad, cfg).sha256


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
    assert rings.bounds_nm == ()


def test_overlapping_or_misshapen_ring_masks_are_refused(with_config, tmp_path):
    masks = np.zeros((2, 16, 512, 128), dtype=bool)
    masks[:, 0, :4] = True
    path = tmp_path / "qmsk.npy"
    np.save(path, masks)
    with pytest.raises(ValueError, match="two rings"):
        pixel_map(with_config(ring_masks_file=str(path)))
    np.save(path, np.zeros((2, 8, 512, 128), dtype=bool))
    with pytest.raises(ValueError, match="expected"):
        pixel_map(with_config(ring_masks_file=str(path)))


def test_the_static_mask_is_the_saxs_passs_unless_the_rings_are_a_files(
    cfg, with_config, tmp_path
):
    assert static_bad_for(cfg).any()  # the ASIC seams, at least
    path = tmp_path / "qmsk.npy"
    np.save(path, np.zeros((1, 16, 512, 128), dtype=bool))
    assert not static_bad_for(with_config(ring_masks_file=str(path))).any()


def test_rois_split_into_plain_and_bragg():
    edges = roi_edges_nm()
    np.testing.assert_allclose(edges[0], [0.225, 0.35])
    assert roi_kinds(edges) == [
        "plain",
        "plain",
        "plain",
        "bragg",
        "bragg",
        "bragg",
        "plain",
    ]
