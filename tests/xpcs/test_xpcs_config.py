"""Pickling, validation and what each of the two hashes covers."""

from __future__ import annotations

import pickle

import pytest

from analysis.common.config import ConfigStateMismatch
from analysis.xpcs.config import (
    RING_EDGES_NM,
    XPCS_OPERATIONAL_FIELDS,
    Kernel,
    MaskPolicy,
    XpcsConfig,
)

ENVELOPE, RINGS = "e" * 64, "r" * 64


def test_the_ring_edges_are_the_triage_rois():
    from analysis.triage.xpcs import roi_edges_nm

    pairs = roi_edges_nm()
    assert list(RING_EDGES_NM[:-1]) == list(pairs[:, 0])
    assert list(RING_EDGES_NM[1:]) == list(pairs[:, 1])


def test_the_default_partner_is_the_previous_train_only(saxs_cfg):
    """Benchmark B4: the baseline drifts with the gap, so only gap 1 is unbiased."""
    assert XpcsConfig(saxs=saxs_cfg).max_partner_gap == 1


def test_the_state_is_pickled_by_name_and_a_positional_one_is_refused(cfg):
    state = cfg.__getstate__()
    assert isinstance(state, dict)
    assert pickle.loads(pickle.dumps(cfg)) == cfg
    with pytest.raises(ConfigStateMismatch, match="by position"):
        XpcsConfig.__new__(XpcsConfig).__setstate__(list(state.values()))


def test_a_state_with_an_unknown_field_is_refused(cfg):
    with pytest.raises(ConfigStateMismatch, match="unknown here"):
        XpcsConfig.__new__(XpcsConfig).__setstate__(dict(cfg.__getstate__(), extra=1))


def test_an_unpickled_config_is_validated(cfg):
    state = dict(cfg.__getstate__(), max_partner_gap=0)
    with pytest.raises(ValueError, match="max_partner_gap"):
        XpcsConfig.__new__(XpcsConfig).__setstate__(state)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"ring_edges_nm": (0.3,)}, "ring_edges_nm"),
        ({"ring_edges_nm": (0.3, 0.2)}, "ring_edges_nm"),
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


@pytest.mark.parametrize(
    "change",
    [
        {"n_workers": 7},
        {"trains_per_block": 3},
        {"flag_base_trains": 2},
        {"output_root": "/elsewhere"},
        {"allow_incomplete": True},
        {"overwrite": True},
    ],
)
def test_operational_fields_move_neither_hash(cfg, with_config, change):
    other = with_config(**change)
    assert other.extract_hash(ENVELOPE) == cfg.extract_hash(ENVELOPE)
    assert other.merge_hash(ENVELOPE, RINGS) == cfg.merge_hash(ENVELOPE, RINGS)
    assert set(change) <= XPCS_OPERATIONAL_FIELDS


@pytest.mark.parametrize(
    "change",
    [
        {"kernel": Kernel.ES},
        {"mask_policy": MaskPolicy.STATIC_ONLY},
        {"max_partner_gap": 25},
    ],
)
def test_how_sums_are_made_moves_the_merge_hash_and_never_the_extract_hash(
    cfg, with_config, change
):
    """A threshold or partner change must not invalidate the photon store."""
    other = with_config(**change)
    assert other.extract_hash(ENVELOPE) == cfg.extract_hash(ENVELOPE)
    assert other.merge_hash(ENVELOPE, RINGS) != cfg.merge_hash(ENVELOPE, RINGS)


def test_ring_edges_and_the_static_mask_reach_the_merge_hash_only_through_the_rings(
    cfg, with_config
):
    other = with_config(ring_edges_nm=(0.2, 0.25, 0.45))
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


def test_outputs_sit_in_per_run_directories_under_the_root(cfg):
    assert cfg.store_file.parts[-3:] == (
        "agipd_ring_photons",
        "r0423",
        "agipd_ring_photons.h5",
    )
    assert cfg.output_file.name == "xpcs_hits.h5"
    assert cfg.output_file.parts[-2] == "r0423"
    assert cfg.store_file.is_relative_to(cfg.output_root)


def test_the_payload_is_json_ready(cfg):
    import json

    payload = cfg.payload()
    json.dumps(payload)
    assert payload["kernel"] == "clean"
    assert payload["saxs"]["run"] == 423
