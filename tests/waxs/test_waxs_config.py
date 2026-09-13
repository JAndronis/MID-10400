"""Config validation, defaults and the hash."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from analysis.waxs.config import (
    DETECTORS,
    EXPECTED_BITS,
    EXPECTED_LIT_CELLS,
    JungfrauWaxsConfig,
    config_for,
    default_poni_file,
    default_static_mask_file,
)


def test_the_output_file_is_per_detector(config_for_detector):
    """§3 D2: two detectors, two files, combined only at the plot."""
    jf1 = config_for_detector("jf1")
    jf2 = config_for_detector("jf2")
    assert jf1.output_file.name == "jungfrau_waxs_jf1.h5"
    assert jf2.output_file.name == "jungfrau_waxs_jf2.h5"
    assert jf1.output_file.parent == jf2.output_file.parent
    assert jf1.output_file.parent == Path(jf1.output_root) / "r0423"


def test_config_for_fills_the_per_detector_paths():
    for detector in DETECTORS:
        cfg = config_for(10400, 423, detector)
        assert cfg.poni_file == default_poni_file(detector)
        assert cfg.static_mask_file == default_static_mask_file(detector)
        assert detector in cfg.poni_file and detector in cfg.static_mask_file


def test_the_defaults_are_the_measured_ones():
    """These came from r0423 and belong in one place (CLAUDE.md rule 2)."""
    assert EXPECTED_BITS == frozenset({0, 1, 21, 22})
    assert EXPECTED_LIT_CELLS == (0, 1, 2, 3, 4, 5, 6, 15)
    cfg = JungfrauWaxsConfig(proposal=10400, run=423, detector="jf1")
    assert cfg.photon_energy_kev == 9.04
    assert cfg.method == ("full", "csc", "cython")
    assert cfg.read_noise_kev is None  # measured per run, not hardcoded


def test_an_unknown_detector_is_refused():
    with pytest.raises(ValueError, match="detector must be one of"):
        JungfrauWaxsConfig(proposal=10400, run=423, detector="jf3")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("npt", 0, "npt must be positive"),
        ("photon_energy_kev", 0.0, "photon_energy_kev"),
        ("method", ("bbox", "csr", "cython"), "method must be"),
        ("mask_bits", -1, "uint32"),
        ("expected_lit_cells", (), "at least one cell"),
        ("lit_fraction_min", 0.0, r"\(0, 1\)"),
        ("lit_fraction_min", 1.0, r"\(0, 1\)"),
        ("read_noise_kev", 0.0, "read_noise_kev"),
        ("max_abs_kev", 0.0, "max_abs_kev"),
        ("cell_sample_trains", 0, "cell_sample_trains"),
        ("trains_per_block", 0, "trains_per_block"),
        ("n_workers", 0, "n_workers"),
        ("selftest_frames", 0, "selftest_frames"),
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
        ("expected_lit_cells", (0, 1)),
        ("lit_threshold_kev", 5.0),
        ("max_abs_kev", 1e4),
        ("read_noise_kev", 0.32),
    ):
        assert dataclasses.replace(cfg, **{field: value}).config_hash() != baseline


def test_the_hash_moves_when_an_input_file_changes(cfg, tmp_path):
    import numpy as np
    from waxs_mockrun import MODULE_SHAPE, write_static_mask

    baseline = cfg.config_hash()
    other = np.zeros(MODULE_SHAPE, dtype=np.uint8)
    other[:100] = 1
    path = write_static_mask(tmp_path / "other.edf", other)
    assert (
        dataclasses.replace(cfg, static_mask_file=str(path)).config_hash() != baseline
    )


def test_workers_default_to_physical_cores(cfg):
    """CLAUDE.md pitfall 10: never default to logical CPUs."""
    assert dataclasses.replace(cfg, n_workers=7).workers == 7
    assert dataclasses.replace(cfg, n_workers=None).workers >= 1


# ── the source names, from lsxfel on r0423 (§6 O1) ───────────────────────────
def test_the_detector_names_are_the_ones_lsxfel_reports():
    from analysis.waxs.config import DETECTOR_MODNOS, DETECTOR_NAMES

    assert DETECTOR_NAMES == {"jf1": "MID_EXP_JF500K1", "jf2": "MID_EXP_JF500K2"}
    assert DETECTOR_MODNOS == {"jf1": 1, "jf2": 2}
    for detector in DETECTORS:
        cfg = JungfrauWaxsConfig(proposal=10400, run=423, detector=detector)
        assert cfg.detector_name == DETECTOR_NAMES[detector]
        assert cfg.first_modno == DETECTOR_MODNOS[detector]


def test_the_filled_source_name_enters_the_hash():
    """Which Karabo source a run was integrated from identifies the result."""
    base = JungfrauWaxsConfig(proposal=10400, run=423, detector="jf1")
    renamed = dataclasses.replace(base, detector_name="MID_EXP_JF500K1_OLD")
    assert renamed.config_hash() != base.config_hash()


def test_swapping_only_the_detector_is_refused():
    """It would keep the other detector's name, PONI and mask (pitfall 14)."""
    jf1 = config_for(10400, 423, "jf1")
    with pytest.raises(ValueError, match="belongs to 'jf1'"):
        dataclasses.replace(jf1, detector="jf2")


def test_a_poni_naming_the_other_detector_is_refused():
    with pytest.raises(ValueError, match="names another detector"):
        JungfrauWaxsConfig(
            proposal=10400,
            run=423,
            detector="jf2",
            poni_file="/somewhere/jf1.poni",
        )


def test_an_unrelated_path_is_still_allowed(tmp_path):
    """Only an unambiguous mix-up is refused, not any unusual path."""
    cfg = JungfrauWaxsConfig(
        proposal=10400,
        run=423,
        detector="jf2",
        poni_file=str(tmp_path / "refined-2026-09-13.poni"),
    )
    assert cfg.detector == "jf2"
