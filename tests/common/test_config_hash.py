"""What ``config_hash`` covers, for both passes.

The hash gates resume and gates reopening a file. Covering a field that cannot
change a stored number makes two identical results look incompatible; failing to
cover one that can makes two *different* results look interchangeable. Both
directions are pinned here, for both configs, because the first was found only
when the DAMNIT variables refused a file the acceptance script had just written.
"""

from __future__ import annotations

import dataclasses

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


# ── the case that actually broke ─────────────────────────────────────────────
def test_a_worker_count_does_not_change_a_waxs_hash(waxs):
    """``w4_acceptance`` passes n_workers=36; the DAMNIT variable does not.

    Those two produced different hashes, so the variable refused the file the
    acceptance run had just written and validated.
    """
    explicit = dataclasses.replace(waxs, n_workers=36)
    default = dataclasses.replace(waxs, n_workers=None)
    assert explicit.config_hash() == default.config_hash()


def test_pinning_the_lit_cells_does_not_change_a_waxs_hash(waxs):
    """``expected_lit_cells`` only *gates*, so by pitfall 12 it stays out.

    It refuses a run whose measured pattern is not the pinned one, and it does
    that before the output file is opened — so it can never change a value in
    ``/frames``, and pinning it for a reprocess must not invalidate the files
    written without it.
    """
    pinned = dataclasses.replace(waxs, expected_lit_cells=EXPECTED_LIT_CELLS)
    assert pinned.expected_lit_cells != waxs.expected_lit_cells
    assert pinned.config_hash() == waxs.config_hash()


def test_tolerating_data_check_failures_does_not_change_a_waxs_hash(waxs):
    """It decides whether the pass raises *after* the file is finished.

    Every row is identical either way, so a run that tolerated 473 failing
    frames and a strict rerun of the same run must be able to share a file.
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


def test_overwrite_does_not_change_a_hash(waxs):
    """The perverse one: a file written with overwrite=True could never match.

    ``open_or_create`` compares the stored hash against the *current* config's,
    so a run that had to force its way over an old file produced a file the next
    ordinary run refused, every time.
    """
    forced = dataclasses.replace(waxs, overwrite=True)
    assert forced.config_hash() == waxs.config_hash()


# ── both directions, both configs ────────────────────────────────────────────
@pytest.mark.parametrize(("field", "value"), OPERATIONAL)
@pytest.mark.parametrize("which", ["agipd", "waxs"])
def test_operational_fields_leave_the_hash_alone(which, field, value, request):
    cfg = request.getfixturevalue(which)
    assert dataclasses.replace(cfg, **{field: value}).config_hash() == cfg.config_hash()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("npt", 250),
        ("photon_energy_kev", 9.0),
        ("min_modules", 2),
        ("mask_bits", 0xFF),
        ("detector_name", "SOMETHING_ELSE"),
    ],
)
def test_result_affecting_fields_move_the_waxs_hash(waxs, field, value):
    assert (
        dataclasses.replace(waxs, **{field: value}).config_hash() != waxs.config_hash()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("npt", 250),
        ("sdd_m", 7.5),
        ("beam_center_px", 600.0),
        ("photon_energy_kev", 9.0),
        ("use_asic_seams", False),
        ("min_modules", 8),
    ],
)
def test_result_affecting_fields_move_the_agipd_hash(agipd, field, value):
    assert (
        dataclasses.replace(agipd, **{field: value}).config_hash()
        != agipd.config_hash()
    )


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


def test_every_operational_field_exists_on_both_configs(agipd, waxs):
    """A typo in the set would silently widen the hash instead of narrowing it."""
    for cfg in (agipd, waxs):
        names = {f.name for f in dataclasses.fields(cfg)}
        assert OPERATIONAL_FIELDS <= names, sorted(OPERATIONAL_FIELDS - names)


def test_result_fields_excludes_exactly_the_operational_set(waxs):
    names = {f.name for f in dataclasses.fields(waxs)}
    # The default is the shared six...
    assert set(result_fields(waxs)) == names - OPERATIONAL_FIELDS
    # ...and what the hash actually uses is the config's own set, which for
    # this pass is one wider.
    assert set(result_fields(waxs, waxs.operational_fields)) == (
        names - WAXS_OPERATIONAL_FIELDS
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
    # excludes one more, and a provenance record that claimed otherwise would
    # be worse than none.
    assert set(recorded) == set(waxs.operational_fields)
    assert set(recorded) == set(OPERATIONAL_FIELDS) | {
        "expected_lit_cells",
        "allow_data_check_failures",
    }
    assert set(recorded) == set(WAXS_OPERATIONAL_FIELDS)
    # ...and the operational values themselves are still there, in full.
    assert stored["n_workers"] == waxs.n_workers
    assert stored["output_root"] == waxs.output_root
