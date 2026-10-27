"""``FrameTableWriter``: rows by label, the ledger, resume and pooling."""

from __future__ import annotations

import json
from dataclasses import replace

import h5py
import numpy as np
import pytest
from tiny_table import (
    OK,
    PLAN,
    ROW_PULSE,
    ROW_TRAIN,
    TRAINS,
    TinyConfig,
    TinyWriter,
    WiderWriter,
    result_for,
    signal_of,
)

from analysis.common.status import FrameStatus
from analysis.common.writer import (
    ConfigHashMismatch,
    SchemaMismatch,
    pooled_per_train,
)


@pytest.fixture
def cfg(tmp_path):
    return TinyConfig(output_dir=str(tmp_path))


def open_writer(cfg, writer=TinyWriter):
    return writer.open_or_create(cfg, PLAN)


def status_column(cfg) -> np.ndarray:
    with h5py.File(cfg.output_file) as f:
        return f["frames/status"][:]


def test_the_blocks_never_straddle_a_gap_and_cover_every_row_once():
    assert [b.train_ids for b in PLAN.blocks] == [(10000,), (10002, 10003), (10004,)]
    rows = np.concatenate([b.rows() for b in PLAN.blocks])
    np.testing.assert_array_equal(rows, np.arange(len(ROW_TRAIN)))


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 0, 1]])
def test_rows_are_written_by_label_in_any_order(cfg, order):
    with open_writer(cfg) as out:
        for index in order:
            out.write_block(PLAN.blocks[index], result_for(PLAN.blocks[index]))

    with h5py.File(cfg.output_file) as f:
        np.testing.assert_array_equal(f["frames/trainId"][:], ROW_TRAIN)
        np.testing.assert_array_equal(f["frames/pulse"][:], ROW_PULSE)
        np.testing.assert_array_equal(
            f["frames/signal"][:], [signal_of(r) for r in range(len(ROW_TRAIN))]
        )
        assert (f["frames/status"][:] == OK).all()


def test_a_frame_whose_label_disagrees_with_the_plan_is_never_stored_as_ok(cfg):
    block = PLAN.blocks[1]  # rows 3..7: trains 10002 and 10003
    result = result_for(block)
    result.train_id[1] = 10003  # row 4 claims the next train
    result.status[2] = FrameStatus.DATA_CHECK_FAILED
    result.train_id[2] = 99  # a failed frame carries no trustworthy label

    with open_writer(cfg) as out:
        out.write_block(block, result)

    with h5py.File(cfg.output_file) as f:
        status = f["frames/status"][3:8]
        train = f["frames/trainId"][3:8]
    assert status.tolist() == [
        OK,
        FrameStatus.LABEL_MISMATCH,
        FrameStatus.DATA_CHECK_FAILED,
        OK,
        OK,
    ]
    np.testing.assert_array_equal(train, ROW_TRAIN[3:8])  # the plan's label


def test_a_result_of_the_wrong_length_is_refused_and_nothing_is_written(cfg):
    block = PLAN.blocks[1]
    short = result_for(block)
    short.n_frames -= 1
    with open_writer(cfg) as out, pytest.raises(ValueError):
        out.write_block(block, short)
    assert (status_column(cfg) == FrameStatus.NOT_PROCESSED).all()


def test_block_complete_retries_worker_errors_and_keeps_data_verdicts(cfg):
    first, middle, last = PLAN.blocks
    with open_writer(cfg) as out:
        assert not any(out.block_complete(b) for b in PLAN.blocks)
        out.write_block(first, result_for(first))
        out.mark(middle, FrameStatus.WORKER_ERROR, "RuntimeError('node')")
        out.write_block(last, result_for(last, status=[FrameStatus.DATA_CHECK_FAILED]))

        assert out.block_complete(first)
        assert not out.block_complete(middle)
        assert out.block_complete(last)


def test_reopening_resumes_rather_than_restarting(cfg):
    first = PLAN.blocks[0]
    with open_writer(cfg) as out:
        out.write_block(first, result_for(first))

    with open_writer(cfg) as out:
        assert [b.index for b in PLAN.blocks if not out.block_complete(b)] == [1, 2]
    with h5py.File(cfg.output_file) as f:
        np.testing.assert_array_equal(f["frames/signal"][0], signal_of(0))


def test_a_file_written_with_another_config_is_refused(cfg):
    open_writer(cfg).close()
    with pytest.raises(ConfigHashMismatch):
        open_writer(replace(cfg, tag="b"))


def test_an_operational_field_does_not_refuse_the_file(cfg, tmp_path):
    open_writer(cfg).close()
    same_file = replace(cfg, output_dir=str(tmp_path / ".." / tmp_path.name))
    open_writer(same_file).close()


def test_overwrite_replaces_a_file_written_with_another_config(cfg):
    first = PLAN.blocks[0]
    with open_writer(cfg) as out:
        out.write_block(first, result_for(first))

    forced = replace(cfg, tag="b", overwrite=True)
    TinyWriter.check_resumable(forced, cfg.output_file)  # the pre-setup check
    with open_writer(forced) as out:
        assert not out.block_complete(first)


def test_a_file_missing_a_column_is_refused(cfg):
    open_writer(cfg).close()
    with pytest.raises(SchemaMismatch):
        open_writer(cfg, WiderWriter)


def test_mark_records_the_status_and_the_reason(cfg):
    middle = PLAN.blocks[1]
    with open_writer(cfg) as out:
        out.mark(middle, FrameStatus.WORKER_ERROR, "RuntimeError('node')")

    status = status_column(cfg)
    assert (status[3:8] == FrameStatus.WORKER_ERROR).all()
    assert (np.delete(status, range(3, 8)) == FrameStatus.NOT_PROCESSED).all()
    with h5py.File(cfg.output_file) as f:
        errors = json.loads(f["provenance"].attrs["block_errors"])
    assert errors == {"1": "RuntimeError('node')"}


def test_a_block_that_succeeds_on_retry_keeps_its_old_error_as_resolved(cfg):
    middle = PLAN.blocks[1]
    with open_writer(cfg) as out:
        out.mark(middle, FrameStatus.WORKER_ERROR, "RuntimeError('node')")
        out.write_block(middle, result_for(middle))

    with h5py.File(cfg.output_file) as f:
        attrs = f["provenance"].attrs
        assert "block_errors" not in attrs
        assert json.loads(attrs["block_errors_resolved"]) == {
            "1": "RuntimeError('node')"
        }


def test_mark_remaining_stamps_only_the_unprocessed_frames(cfg):
    first, middle, _ = PLAN.blocks
    with open_writer(cfg) as out:
        out.write_block(first, result_for(first))
        out.mark(middle, FrameStatus.WORKER_ERROR, "boom")
        out.mark_remaining(FrameStatus.MISSING_MODULES)

    assert status_column(cfg).tolist() == (
        [OK] * 3 + [FrameStatus.WORKER_ERROR] * 5 + [FrameStatus.MISSING_MODULES]
    )


def test_pooling_sums_the_ok_frames_of_each_train(cfg):
    """I = sum S / sum N and sigma = sqrt(sum V) / sum N, over OK frames only."""
    first, middle, last = PLAN.blocks
    with open_writer(cfg) as out:
        out.write_block(
            first, result_for(first, status=[OK, FrameStatus.LABEL_MISMATCH, OK])
        )
        out.write_block(middle, result_for(middle))
        variance = result_for(last)
        variance.variance[0, 2] = -1.0  # an unbiased estimate can dip below zero
        out.write_block(last, variance)

    pooled = pooled_per_train(cfg.output_file)

    np.testing.assert_array_equal(pooled.trainId, [t for t, _ in TRAINS])
    assert pooled.n_frames.values.tolist() == [2, 0, 2, 3, 1]
    signal = np.stack([signal_of(r) for r in range(len(ROW_TRAIN))])
    # train 10000: rows 0 and 2; N is 2 per frame, V is 4 per frame
    np.testing.assert_allclose(pooled.intensity[0], (signal[0] + signal[2]) / 4)
    np.testing.assert_allclose(pooled.sigma[0], np.sqrt(8.0) / 4)
    np.testing.assert_array_equal(pooled.intensity[1], 0.0)  # no frames, no NaN
    np.testing.assert_allclose(pooled.intensity[3], signal[5:8].sum(0) / 6)
    np.testing.assert_allclose(pooled.sigma[3], np.sqrt(12.0) / 6)
    np.testing.assert_allclose(pooled.sigma[4], [1.0, 1.0, 0.0])
    assert pooled.attrs["negative_variance_bins"] == 1
