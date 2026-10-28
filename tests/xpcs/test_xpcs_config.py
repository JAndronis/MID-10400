"""Validation, and what each of the two hashes covers."""

from __future__ import annotations

import pytest

from analysis.xpcs.config import (
    XPCS_OPERATIONAL_FIELDS,
    Kernel,
    MaskPolicy,
    XpcsConfig,
)

ENVELOPE, RINGS = "e" * 64, "r" * 64


def test_an_unpickled_config_is_validated(cfg):
    state = dict(cfg.__getstate__(), max_partner_gap=0)
    with pytest.raises(ValueError, match="max_partner_gap"):
        XpcsConfig.__new__(XpcsConfig).__setstate__(state)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"rings_nm": ((0.3, 0.2),)}, "rings_nm"),
        ({"rings_nm": ((0.2, 0.3), (0.25, 0.4))}, "rings_nm"),
        ({"max_partner_gap": 0}, "max_partner_gap"),
        ({"envelope_margin_mm": -1.0}, "envelope_margin_mm"),
        ({"trains_per_block": 0}, "trains_per_block"),
        ({"n_workers": 0}, "n_workers"),
        ({"kernel": "clean"}, "Kernel"),
        ({"mask_policy": "any_frame"}, "MaskPolicy"),
    ],
)
def test_invalid_fields_are_refused(with_config, change, message):
    with pytest.raises((ValueError, TypeError), match=message):
        with_config(**change)


def test_operational_fields_move_neither_hash(cfg, with_config):
    changes = {
        "n_workers": 7,
        "trains_per_block": 3,
        "flag_base_trains": 2,
        "output_root": "/elsewhere",
        "allow_incomplete": True,
        "overwrite": True,
    }
    assert set(changes) == XPCS_OPERATIONAL_FIELDS
    for field, value in changes.items():
        other = with_config(**{field: value})
        assert other.extract_hash(ENVELOPE) == cfg.extract_hash(ENVELOPE), field
        assert other.merge_hash(ENVELOPE, RINGS) == cfg.merge_hash(ENVELOPE, RINGS)


def test_how_sums_are_made_moves_the_merge_hash_and_never_the_extract_hash(
    cfg, with_config
):
    """A threshold or partner change must not invalidate the photon store."""
    for change in (
        {"kernel": Kernel.ES},
        {"mask_policy": MaskPolicy.STATIC_ONLY},
        {"max_partner_gap": 25},
    ):
        other = with_config(**change)
        assert other.extract_hash(ENVELOPE) == cfg.extract_hash(ENVELOPE)
        assert other.merge_hash(ENVELOPE, RINGS) != cfg.merge_hash(ENVELOPE, RINGS)


def test_ring_bounds_and_the_static_mask_reach_the_merge_hash_only_through_the_rings(
    cfg, with_config
):
    other = with_config(rings_nm=((0.2, 0.25), (0.3, 0.45)))
    assert other.extract_hash(ENVELOPE) == cfg.extract_hash(ENVELOPE)
    assert other.merge_hash(ENVELOPE, RINGS) == cfg.merge_hash(ENVELOPE, RINGS)
    assert cfg.merge_hash(ENVELOPE, "x" * 64) != cfg.merge_hash(ENVELOPE, RINGS)


def test_the_store_is_keyed_on_the_envelope_the_bits_and_the_run(cfg, with_config):
    from dataclasses import replace

    base = cfg.extract_hash(ENVELOPE)
    assert cfg.extract_hash("f" * 64) != base
    assert cfg.merge_hash("f" * 64, RINGS) != cfg.merge_hash(ENVELOPE, RINGS)
    for changed in (
        replace(cfg.saxs, mask_bits=0xFF),
        replace(cfg.saxs, run=424),
    ):
        assert with_config(saxs=changed).extract_hash(ENVELOPE) != base


def test_a_missing_module_is_not_tolerated(cfg):
    from dataclasses import replace

    with pytest.raises(ValueError, match="all 16 modules"):
        XpcsConfig(saxs=replace(cfg.saxs, min_modules=15, pixel_sum_trains=None))


def test_ring_masks_files_cannot_carry_the_outlier_mask(with_config):
    with pytest.raises(ValueError, match="ring_masks_file"):
        with_config(outlier_thresholds=(0.75, 1.75), ring_masks_file="x.npy")


def test_the_merge_hash_names_the_outlier_settings(with_config):
    hashes = {
        with_config(outlier_thresholds=t).merge_hash("e", "r")
        for t in (None, (0.75, 1.75), (0.5, 2.0))
    }
    assert len(hashes) == 3
