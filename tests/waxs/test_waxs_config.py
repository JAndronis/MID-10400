"""Config validation and the hash."""

from __future__ import annotations

import dataclasses

import pytest

from analysis.waxs.config import (
    EXPECTED_LIT_CELLS,
    JungfrauWaxsConfig,
    config_for,
)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        # A method string resolves silently to another engine, so only the
        # tuple pyFAI is asked for is accepted.
        ("method", ("bbox", "csr", "cython"), "method must be"),
        ("expected_lit_cells", (), "at least one cell"),
        # No storageCellStart produces this, so pinning it means a typo.
        ("expected_lit_cells", (0, 2, 4), "storage-cell sequence"),
    ],
)
def test_nonsense_is_refused(field, value, match):
    with pytest.raises(ValueError, match=match):
        JungfrauWaxsConfig(proposal=10400, run=423, detector="jf1", **{field: value})


def test_the_hash_covers_every_field_and_the_input_files(cfg):
    baseline = cfg.config_hash()
    assert cfg.config_hash() == baseline  # stable
    for field, value in (
        ("npt", cfg.npt // 2),
        ("photon_energy_kev", 9.0),
        ("lit_threshold_kev", 5.0),
        ("lit_fraction_min", 1e-3),
        ("lit_gap_ratio", 50.0),
        ("max_abs_kev", 1e4),
        ("read_noise_kev", 0.32),
        ("read_noise_fallback_kev", 0.40),
        # Which Karabo source a run was integrated from identifies the result.
        ("detector_name", "MID_EXP_JF500K1_OLD"),
    ):
        assert dataclasses.replace(cfg, **{field: value}).config_hash() != baseline, (
            field
        )


def test_the_hash_ignores_the_pinned_lit_cells(cfg):
    """It gates before the file is opened, so it cannot change a stored number.

    ``lit_fraction_min`` and ``lit_gap_ratio`` above are the opposite case and
    stay in: they decide which cells are integrated, and so which rows exist.
    """
    pinned = dataclasses.replace(cfg, expected_lit_cells=EXPECTED_LIT_CELLS)
    assert pinned.config_hash() == cfg.config_hash()


def test_the_hash_moves_when_an_input_file_is_rewritten_in_place(cfg, tmp_path):
    """A mask replaced at the same path is a different input."""
    import numpy as np
    from waxs_mockrun import MODULE_SHAPE, write_static_mask

    mask = np.zeros(MODULE_SHAPE, dtype=np.uint8)
    path = write_static_mask(tmp_path / "jf1_mask.edf", mask)
    moved = dataclasses.replace(cfg, static_mask_file=str(path))
    before = moved.config_hash()
    mask[:100] = 1
    write_static_mask(path, mask)
    assert moved.config_hash() != before


@pytest.mark.parametrize(
    ("build", "match"),
    [
        # It would keep jf1's source name, PONI and mask.
        (
            lambda: dataclasses.replace(config_for(10400, 423, "jf1"), detector="jf2"),
            "belongs to 'jf1'",
        ),
        (
            lambda: JungfrauWaxsConfig(
                proposal=10400, run=423, detector="jf2", poni_file="/x/jf1.poni"
            ),
            "names another detector",
        ),
        # Only an unambiguous mix-up is refused, not any unusual path.
        (
            lambda: JungfrauWaxsConfig(
                proposal=10400, run=423, detector="jf2", poni_file="/x/refined.poni"
            ),
            None,
        ),
    ],
)
def test_one_detector_is_never_given_the_others_inputs(build, match):
    if match is None:
        assert build().detector == "jf2"
    else:
        with pytest.raises(ValueError, match=match):
            build()
