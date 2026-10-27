"""What ``config_hash`` covers, for both passes.

The hash gates resume and gates reopening a file. Covering a field that cannot
change a stored number makes two identical results look incompatible; failing to
cover one that can makes two *different* results look interchangeable. Both
directions are pinned here, for both configs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from analysis.common.config import OPERATIONAL_FIELDS, result_fields

pytest.importorskip("pyFAI")

from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.waxs.config import (  # noqa: E402
    EXPECTED_LIT_CELLS,
    WAXS_OPERATIONAL_FIELDS,
    JungfrauWaxsConfig,
)

#: Fields that must not move a hash. Every pass shares this set.
OPERATIONAL = (
    ("n_workers", 8),
    ("trains_per_block", 16),
    ("selftest_frames", 4),
    ("output_root", "/tmp/somewhere-else"),
    ("allow_incomplete", True),
    ("overwrite", True),
)


@pytest.fixture(scope="module")
def geometry(tmp_path_factory):
    """A synthetic PONI and ``.edf``, so the hash has real files to digest."""
    import numpy as np
    from pyFAI.detectors import Jungfrau
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.units import hc

    fabio = pytest.importorskip("fabio")
    directory = tmp_path_factory.mktemp("config-hash")

    poni = directory / "jf1.poni"
    AzimuthalIntegrator(
        detector=Jungfrau(),
        dist=0.232,
        poni1=256 * 75e-6,
        poni2=512 * 75e-6,
        wavelength=hc / 9.04 * 1e-10,
    ).save(str(poni))

    mask = np.zeros((512, 1024), dtype=np.uint8)
    mask[:100] = 1
    edf = directory / "jf1_mask.edf"
    fabio.edfimage.EdfImage(data=mask).write(str(edf))
    return poni, edf


@pytest.fixture
def agipd():
    return AgipdSaxsConfig(
        proposal=10400, run=423, geometry_file=None, pixel_mask_file=None
    )


@pytest.fixture
def waxs(geometry):
    poni, edf = geometry
    return JungfrauWaxsConfig(
        proposal=10400,
        run=423,
        detector="jf1",
        poni_file=str(poni),
        static_mask_file=str(edf),
    )


def test_pinning_the_lit_cells_does_not_change_a_waxs_hash(waxs):
    """``expected_lit_cells`` only *gates*, so it stays out of the hash.

    It refuses a run whose measured pattern is not the pinned one, and it does
    that before the output file is opened. So it can never change a value in
    ``/frames``, and pinning it for a reprocess must not invalidate the files
    written without it.
    """
    pinned = dataclasses.replace(waxs, expected_lit_cells=EXPECTED_LIT_CELLS)
    assert pinned.expected_lit_cells != waxs.expected_lit_cells
    assert pinned.config_hash() == waxs.config_hash()


def test_tolerating_data_check_failures_does_not_change_a_waxs_hash(waxs):
    """It decides whether the pass raises *after* the file is finished.

    Every row is identical either way, so a run that tolerated failing frames
    and a strict rerun of the same run must be able to share a file.
    """
    strict = dataclasses.replace(waxs, allow_data_check_failures=False)
    assert strict.allow_data_check_failures != waxs.allow_data_check_failures
    assert strict.config_hash() == waxs.config_hash()


def test_the_lit_thresholds_do_move_a_waxs_hash(waxs):
    """The opposite case, and the reason the two are not one field.

    ``lit_fraction_min`` and ``lit_gap_ratio`` decide which cells come out lit,
    and so which rows exist at all. They stay in the hash.
    """
    for field, value in (("lit_fraction_min", 1e-3), ("lit_gap_ratio", 50.0)):
        assert (
            dataclasses.replace(waxs, **{field: value}).config_hash()
            != waxs.config_hash()
        ), field


# ── both directions, both configs ────────────────────────────────────────────
def test_operational_fields_leave_the_hash_alone(agipd, waxs):
    """A worker count, a block size or ``overwrite`` must not refuse a stored file.

    ``open_or_create`` compares the stored hash with the current config's, so a
    hash covering ``overwrite`` would refuse every file a forced run wrote.
    """
    for cfg in (agipd, waxs):
        for field, value in OPERATIONAL:
            moved = dataclasses.replace(cfg, **{field: value}).config_hash()
            assert moved == cfg.config_hash(), (type(cfg).__name__, field)


def test_result_affecting_fields_move_the_hash(agipd, waxs):
    changes = {
        waxs: [
            ("npt", 250),
            ("photon_energy_kev", 9.0),
            ("min_modules", 2),
            ("mask_bits", 0xFF),
            ("detector_name", "SOMETHING_ELSE"),
        ],
        agipd: [
            ("npt", 250),
            ("sdd_m", 7.5),
            ("beam_center_px", 600.0),
            ("photon_energy_kev", 9.0),
            ("use_asic_seams", False),
            # window sums need all 16 modules, and are off here
            ("min_modules", 8),
        ],
    }
    for cfg, fields in changes.items():
        base = dataclasses.replace(cfg, pixel_sum_trains=None) if cfg is agipd else cfg
        for field, value in fields:
            moved = dataclasses.replace(base, **{field: value}).config_hash()
            assert moved != base.config_hash(), (type(cfg).__name__, field)


def test_the_window_sum_fields_leave_the_agipd_hash_alone(agipd):
    """The window sums are a separate file with a hash of their own.

    So neither field can change a number in ``agipd_saxs.h5``, and adding them
    must not refuse a file written before they existed.
    """
    for field, value in (
        ("pixel_sum_trains", 50),
        ("pixel_sum_trains", None),
        ("pixel_sums_root", "/tmp/x"),
    ):
        moved = dataclasses.replace(agipd, **{field: value}).config_hash()
        assert moved == agipd.config_hash(), field


def test_the_train_sampling_fields_are_not_operational(agipd, waxs):
    """They look like tuning knobs and are not.

    ``base_mask_trains`` and ``cell_sample_trains`` choose which trains are
    sampled, and so change the masks and the measured readout noise that every
    stored row depends on. Excluding either would let two genuinely different
    results share a hash.
    """
    assert "base_mask_trains" not in OPERATIONAL_FIELDS
    assert "cell_sample_trains" not in OPERATIONAL_FIELDS
    assert (
        dataclasses.replace(agipd, base_mask_trains=4).config_hash()
        != agipd.config_hash()
    )
    assert (
        dataclasses.replace(waxs, cell_sample_trains=4).config_hash()
        != waxs.config_hash()
    )


def test_result_fields_renders_sets_and_tuples_stably(waxs):
    pinned = dataclasses.replace(waxs, expected_lit_cells=EXPECTED_LIT_CELLS)
    payload = result_fields(pinned)
    assert payload["expected_bits"] == sorted(pinned.expected_bits)
    assert payload["method"] == list(pinned.method)
    assert payload["expected_lit_cells"] == list(pinned.expected_lit_cells)
    # An unpinned set is None, not an empty tuple: "measure it" and "measure it
    # and expect nothing" are different instructions and must not render alike.
    assert result_fields(waxs)["expected_lit_cells"] is None


def test_rewriting_an_input_file_in_place_moves_the_hash(geometry, tmp_path):
    """A new PONI or mask at the same path is a different result.

    The hash must follow what the file holds, not where it lives, or a stored
    file integrated with the old mask would be resumed under the new one.
    """
    import shutil

    import fabio

    poni, edf = (Path(shutil.copy(path, tmp_path)) for path in geometry)
    cfg = JungfrauWaxsConfig(
        proposal=10400,
        run=423,
        detector="jf1",
        poni_file=str(poni),
        static_mask_file=str(edf),
    )
    before = cfg.config_hash()

    data = fabio.open(str(edf)).data.copy()
    data[200] = 1
    fabio.edfimage.EdfImage(data=data).write(str(edf))
    after_mask = cfg.config_hash()
    assert after_mask != before

    poni.write_text(poni.read_text().replace("Distance: 0.232", "Distance: 0.233"))
    assert cfg.config_hash() != after_mask


def test_the_stored_file_says_which_fields_the_hash_covers(waxs, tmp_path):
    """So a file can be reasoned about without the matching source version."""
    import json

    import h5py

    from analysis.common.plan import RunPlan, TrainRecord
    from analysis.common.status import FrameStatus
    from analysis.waxs.writer import JungfrauWaxsWriter

    plan = RunPlan(
        trains=(TrainRecord(10000, 4, 0, FrameStatus.OK),),
        blocks=(),
        n_frames=4,
        detector_name="MID_EXP_JF500K1",
    )
    path = tmp_path / "provenance.h5"
    JungfrauWaxsWriter.open_or_create(waxs, plan, path).close()

    with h5py.File(path) as handle:
        recorded = json.loads(handle["provenance"].attrs["config_operational_fields"])
        stored = json.loads(handle["provenance"].attrs["config"])
    # The set the *config* excludes, not the shared constant: the JUNGFRAU pass
    # excludes more, and a provenance record that claimed otherwise would be
    # worse than none.
    assert set(recorded) == set(waxs.operational_fields)
    assert set(recorded) == set(OPERATIONAL_FIELDS) | {
        "expected_lit_cells",
        "allow_data_check_failures",
    }
    assert set(recorded) == set(WAXS_OPERATIONAL_FIELDS)
    # ...and the operational values themselves are still there, in full.
    assert stored["n_workers"] == waxs.n_workers
    assert stored["output_root"] == waxs.output_root
