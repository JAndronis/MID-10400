"""The row model over a mock JUNGFRAU run (WAXS context file §7, W3)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("extra_data")

from waxs_mockrun import LIT_CELLS  # noqa: E402

from analysis.common.status import FrameStatus  # noqa: E402
from analysis.waxs.plan import (  # noqa: E402
    InconsistentEntries,
    build_plan,
    open_detector,
)


def status_of(plan, train_id):
    return plan.record(train_id).status


def test_rows_are_the_lit_cells_not_the_entry_count(pipeline):
    """The trap: JUNGFRAU frame_counts counts entries (1/train), not frames."""
    plan, run = pipeline.plan, pipeline.run
    assert set(pipeline.classification.lit) == set(run.lit_cells)
    assert all(t.n_frames == len(run.lit_cells) for t in plan.trains if t.n_frames)
    assert plan.n_frames == run.n_frames
    # ...and that is emphatically not what the reader reports per train.
    assert set(int(c) for c in pipeline.det.frame_counts) == {1}


def test_every_train_of_the_run_is_in_the_ledger(pipeline):
    plan, run = pipeline.plan, pipeline.run
    assert [t.train_id for t in plan.trains] == list(run.train_ids)
    assert all(status_of(plan, t) is FrameStatus.OK for t in run.detector_trains)


def test_rows_are_contiguous_and_match_the_run(pipeline):
    plan, run = pipeline.plan, pipeline.run
    for train_id in run.detector_trains:
        assert plan.record(train_id).first_row == run.first_row(train_id)


def test_blocks_cover_every_row_exactly_once(pipeline):
    rows = np.concatenate([b.rows() for b in pipeline.plan.blocks])
    assert np.array_equal(np.sort(rows), np.arange(pipeline.plan.n_frames))
    assert all(
        len(b.train_ids) <= pipeline.cfg.trains_per_block for b in pipeline.plan.blocks
    )


def test_a_train_with_no_entry_is_no_frames(cfg, mock_run_factory, operator):
    run, dc = mock_run_factory(zero_entry_trains=(10002,))
    op, _ = operator
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)

    assert status_of(plan, 10002) is FrameStatus.NO_FRAMES
    assert plan.record(10002).n_frames == 0
    assert plan.n_frames == (len(run.train_ids) - 1) * len(LIT_CELLS)


def test_a_dropped_train_does_not_shift_rows(cfg, mock_run_factory, operator):
    """CLAUDE.md pitfall 4: the run has a gap, the rows must not slide."""
    run, dc = mock_run_factory(train_ids=(10000, 10001, 10003, 10004))
    op, _ = operator
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)

    assert [t.train_id for t in plan.trains] == [10000, 10001, 10003, 10004]
    assert [t.first_row for t in plan.trains] == [0, 8, 16, 24]


def test_a_block_never_straddles_a_gap(cfg, mock_run_factory, operator):
    run, dc = mock_run_factory(zero_entry_trains=(10001,))
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)

    for block in plan.blocks:
        rows = block.rows()
        assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))
    assert 10001 not in {t for b in plan.blocks for t in b.train_ids}


def test_empty_lit_cells_is_refused(cfg, mock_run_factory):
    _, dc = mock_run_factory()
    with pytest.raises(ValueError, match="nothing to integrate"):
        build_plan(cfg, (), dc=dc, control_dc=dc)


def test_run_checks_record_the_entry_shape(pipeline):
    checks = pipeline.plan.checks
    assert checks["entries_per_train"] == [1]
    assert checks["frames_per_entry"] == 16
    assert checks["lit_cells"] == list(pipeline.classification.lit)


def test_multi_entry_trains_are_refused(cfg, mock_run_factory):
    """EXtra-data cannot shape them, so the plan must refuse rather than guess.

    ``frame_counts`` is an instance attribute set in ``MultimodDetectorBase``'s
    constructor, so the second entry is injected there rather than mocked.
    """
    _, dc = mock_run_factory()
    det = open_detector(cfg, dc)
    det.frame_counts = det.frame_counts.copy()
    det.frame_counts.iloc[1] = 2
    with pytest.raises(InconsistentEntries, match="one JUNGFRAU entry"):
        build_plan(cfg, (0, 1), dc=dc, control_dc=dc, det=det)


# ── source names, from lsxfel on the r0423 proc files (§6 O1) ────────────────
def test_the_corrected_source_is_the_one_selected(pipeline):
    """Corrected data uses /CORR/ since 2026/1; /DET/ survives as a soft link."""
    assert pipeline.det.detector_name == "MID_EXP_JF500K1"
    assert list(pipeline.det.source_to_modno) == [
        "MID_EXP_JF500K1/CORR/JNGFR01:daqOutput"
    ]


def test_the_legacy_det_alias_is_not_a_second_module(pipeline):
    """The alias *is* in instrument_sources, and must not be read as a module.

    ``DataCollection.instrument_sources`` includes legacy names — only
    ``detector_sources`` subtracts them — so ``_source_matches`` really does see
    it. What keeps it out is that ``_source_corr_pat`` matches ``/CORR/`` alone.
    If a future EXtra-data widened that pattern, ``modno_to_source`` would
    collapse two sources onto one module number and the constructor's own
    assertion would fire; this pins the behaviour before that.
    """
    legacy = "MID_EXP_JF500K1/DET/JNGFR01:daqOutput"
    assert legacy in pipeline.dc.instrument_sources
    assert pipeline.dc.legacy_sources[legacy] == (
        "MID_EXP_JF500K1/CORR/JNGFR01:daqOutput"
    )
    assert len(pipeline.det.source_to_modno) == 1
    assert pipeline.det.n_modules == 1


def test_each_detector_is_addressed_by_its_own_name_and_module(
    config_for_detector, mock_run_factory
):
    from analysis.waxs.config import DETECTOR_MODNOS, DETECTOR_NAMES

    for detector in ("jf1", "jf2"):
        name, modno = DETECTOR_NAMES[detector], DETECTOR_MODNOS[detector]
        _, dc = mock_run_factory(detector_name=name, module=modno)
        det = open_detector(config_for_detector(detector), dc)
        assert det.detector_name == name
        assert list(det.source_to_modno) == [f"{name}/CORR/JNGFR{modno:02d}:daqOutput"]
        # first_modno normalises each single-module detector to modno 1.
        assert det.source_to_modno[f"{name}/CORR/JNGFR{modno:02d}:daqOutput"] == 1
        assert det.n_modules == 1


def test_auto_detection_is_ambiguous_with_both_detectors_present(
    config_for_detector, tmp_path
):
    """Which is why cfg.detector_name is filled in rather than left to it."""
    from extra_data import RunDirectory
    from extra_data.components import JUNGFRAU
    from waxs_mockrun import write_mock_run

    from analysis.waxs.config import DETECTOR_MODNOS, DETECTOR_NAMES

    root = tmp_path / "both"
    for detector in ("jf1", "jf2"):
        write_mock_run(
            root,
            detector_name=DETECTOR_NAMES[detector],
            module=DETECTOR_MODNOS[detector],
            train_ids=(10000, 10001),
        )
    dc = RunDirectory(str(root))

    with pytest.raises(ValueError, match="Multiple detectors found"):
        JUNGFRAU(dc)

    # ...and naming them works, which is what the config fills in by default.
    for detector in ("jf1", "jf2"):
        det = open_detector(config_for_detector(detector), dc)
        assert det.detector_name == DETECTOR_NAMES[detector]
