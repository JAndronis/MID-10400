"""The row model over a mock JUNGFRAU run."""

from __future__ import annotations

import pytest

pytest.importorskip("extra_data")

from waxs_mockrun import LIT_CELLS  # noqa: E402

from analysis.common.status import FrameStatus  # noqa: E402
from analysis.waxs.plan import (  # noqa: E402
    InconsistentEntries,
    build_plan,
    open_detector,
)


def test_rows_are_the_lit_cells_not_the_entry_count(pipeline):
    """The trap: JUNGFRAU frame_counts counts entries (1/train), not frames."""
    plan, run = pipeline.plan, pipeline.run
    assert set(pipeline.classification.lit) == set(run.lit_cells)
    assert all(t.n_frames == len(run.lit_cells) for t in plan.trains if t.n_frames)
    assert plan.n_frames == run.n_frames
    # ...and that is emphatically not what the reader reports per train.
    assert set(int(c) for c in pipeline.det.frame_counts) == {1}


def test_a_train_with_no_entry_is_no_frames(cfg, mock_run_factory, status_of):
    run, dc = mock_run_factory(zero_entry_trains=(10002,), fill=False)
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)

    assert status_of(plan, 10002) is FrameStatus.NO_FRAMES
    assert plan.record(10002).n_frames == 0
    assert plan.n_frames == (len(run.train_ids) - 1) * len(LIT_CELLS)


def test_a_dropped_train_does_not_shift_rows(cfg, mock_run_factory):
    """The run has a gap, and the rows after it must not slide onto other trains."""
    run, dc = mock_run_factory(train_ids=(10000, 10001, 10003, 10004), fill=False)
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)

    assert [t.train_id for t in plan.trains] == [10000, 10001, 10003, 10004]
    assert [t.first_row for t in plan.trains] == [0, 8, 16, 24]


def test_an_empty_lit_set_plans_no_rows_rather_than_refusing(cfg, mock_run_factory):
    """A run that saw no beam is a valid run with no rows, not a broken one.

    ``run_jungfrau_waxs`` returns before it ever gets here on such a run, but
    the plan itself must not refuse one: whether an empty set is worth
    integrating is the pass's call, not the row model's.
    """
    _, dc = mock_run_factory()
    plan = build_plan(cfg, (), dc=dc, control_dc=dc)

    assert plan.n_frames == 0
    assert all(record.n_frames == 0 for record in plan.trains)
    assert plan.checks["lit_cells"] == []


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


# ── source names, as lsxfel reports them for proc files ──────────────────────
def test_each_detector_reads_its_corrected_source_and_no_legacy_alias(
    pipeline, config_for_detector, mock_run_factory
):
    """Corrected data use /CORR/; the /DET/ name remains only as a soft link.

    ``DataCollection.instrument_sources`` includes legacy names (only
    ``detector_sources`` subtracts them), so ``_source_matches`` does see the
    alias. What keeps it out is that ``_source_corr_pat`` matches ``/CORR/``
    alone. If a future EXtra-data widened that pattern, ``modno_to_source``
    would collapse two sources onto one module number; this pins the
    behaviour before that.
    """
    from analysis.waxs.config import DETECTOR_MODNOS, DETECTOR_NAMES

    legacy = "MID_EXP_JF500K1/DET/JNGFR01:daqOutput"
    assert legacy in pipeline.dc.instrument_sources
    assert pipeline.dc.legacy_sources[legacy] == (
        "MID_EXP_JF500K1/CORR/JNGFR01:daqOutput"
    )
    for detector in ("jf1", "jf2"):
        name, modno = DETECTOR_NAMES[detector], DETECTOR_MODNOS[detector]
        _, dc = mock_run_factory(detector_name=name, module=modno, fill=False)
        det = open_detector(config_for_detector(detector), dc)
        corrected = f"{name}/CORR/JNGFR{modno:02d}:daqOutput"
        assert det.detector_name == name
        # first_modno normalises each single-module detector to modno 1.
        assert dict(det.source_to_modno) == {corrected: 1}
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
        # fill=False: this asserts on source names and nothing reads a frame,
        # so writing 135 MB of adc twice would buy nothing.
        write_mock_run(
            root,
            detector_name=DETECTOR_NAMES[detector],
            module=DETECTOR_MODNOS[detector],
            train_ids=(10000, 10001),
            fill=False,
        )
    dc = RunDirectory(str(root))

    with pytest.raises(ValueError, match="Multiple detectors found"):
        JUNGFRAU(dc)

    # ...and naming them works, which is what the config fills in by default.
    for detector in ("jf1", "jf2"):
        det = open_detector(config_for_detector(detector), dc)
        assert det.detector_name == DETECTOR_NAMES[detector]
