"""The worker, the output schema, the ledger and the reducers (W3)."""

from __future__ import annotations

import dataclasses
import json

import h5py
import numpy as np
import pytest

pytest.importorskip("extra_data")

from waxs_mockrun import LIT_CELLS  # noqa: E402

from analysis.common.status import FrameStatus  # noqa: E402
from analysis.waxs import worker  # noqa: E402
from analysis.waxs.writer import (  # noqa: E402
    ConfigHashMismatch,
    JungfrauWaxsWriter,
    per_cell,
    pooled_per_train,
)


def process_all(pipeline):
    return {b.index: worker.process_block(b) for b in pipeline.plan.blocks}


def written(pipeline, tmp_path, name="out.h5"):
    """Run every block through the worker and the writer, return the path."""
    path = tmp_path / name
    with JungfrauWaxsWriter.open_or_create(pipeline.cfg, pipeline.plan, path) as out:
        out.store_operator(pipeline.op)
        out.store_cells(
            pipeline.classification, pipeline.model, np.array([10000], dtype=np.uint64)
        )
        for block in pipeline.plan.blocks:
            out.write_block(block, worker.process_block(block))
    return path


# ── the worker ───────────────────────────────────────────────────────────────
def test_every_lit_frame_is_integrated(worker_ready):
    for result in process_all(worker_ready).values():
        assert (result.status == FrameStatus.OK).all()
        assert (result.normalization.sum(axis=1) > 0).all()
        assert result.energy_valid.min() > 0


def test_identity_comes_from_the_reader(worker_ready):
    """Train from train_id_coordinates, cell from data.memoryCell (§5 R5)."""
    lit = worker_ready.classification.lit
    for block in worker_ready.plan.blocks:
        result = worker.process_block(block)
        expected_cells = np.tile(lit, len(block.train_ids))
        assert np.array_equal(result.cell_id, expected_cells)
        expected_trains = np.repeat(block.train_ids, len(lit))
        assert np.array_equal(result.train_id, expected_trains)


def test_a_repeated_memory_cell_is_a_label_mismatch(
    cfg, mock_run_factory, operator, model
):
    """Rows cannot be filled by label, and position is what pitfall 4 forbids."""
    from analysis.waxs.plan import build_plan, open_detector

    run, dc = mock_run_factory(shuffled_cells_train=10001)
    op, ai = operator
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)
    worker.init_from_detector(cfg, op, ai, model, LIT_CELLS, det)
    try:
        statuses = {}
        for block in plan.blocks:
            result = worker.process_block(block)
            for train_id, status in zip(
                np.repeat(block.train_ids, 8), result.status, strict=True
            ):
                statuses.setdefault(int(train_id), set()).add(int(status))
        assert statuses[10001] == {FrameStatus.LABEL_MISMATCH}
        assert statuses[10000] == {FrameStatus.OK}
    finally:
        worker._STATE = None


def test_a_frame_failing_the_data_check_is_recorded_not_raised(
    cfg, mock_run_factory, operator, model
):
    from analysis.waxs.plan import build_plan, open_detector

    run, dc = mock_run_factory(extreme_train=10002)
    op, ai = operator
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)
    worker.init_from_detector(cfg, op, ai, model, LIT_CELLS, det)
    try:
        failed = 0
        for block in plan.blocks:
            result = worker.process_block(block)
            failed += int((result.status == FrameStatus.DATA_CHECK_FAILED).sum())
        # The extreme pixel sits on a kept pixel of one cell of one train.
        assert failed == 1
    finally:
        worker._STATE = None


def test_the_worker_refuses_to_run_uninitialised(pipeline):
    worker._STATE = None
    with pytest.raises(RuntimeError, match="worker.init has not run"):
        worker.process_block(pipeline.plan.blocks[0])


# ── the output file ──────────────────────────────────────────────────────────
def test_the_schema_is_the_jungfrau_one(worker_ready, tmp_path):
    path = written(worker_ready, tmp_path)
    with h5py.File(path) as handle:
        frames = handle["frames"]
        assert set(frames) == {
            "trainId",
            "cellId",
            "status",
            "energy_valid",
            "n_bad_pixels",
            "n_negative_variance_bins",
            "max_kev",
            "signal",
            "normalization",
            "variance",
        }
        # No pulse id: the reader has none and one must not be invented.
        assert "reader_pulseId" not in frames
        assert frames["signal"].shape == (
            worker_ready.plan.n_frames,
            worker_ready.cfg.npt,
        )
        assert handle["q/centers"].attrs["unit"] == "q_nm^-1"
        assert handle["operator"].attrs["poni"].startswith("#")
        assert handle["cells"].attrs["error_model_source"] == "measured"
        assert list(handle["cells"]["lit"][:]) == list(worker_ready.classification.lit)


def test_there_are_no_nan_sentinels_anywhere(worker_ready, tmp_path):
    path = written(worker_ready, tmp_path)
    with h5py.File(path) as handle:
        for name in ("signal", "normalization", "variance", "energy_valid", "max_kev"):
            values = handle[f"frames/{name}"][:]
            assert np.isfinite(values).all(), name


def test_rows_are_written_by_label(worker_ready, tmp_path):
    path = written(worker_ready, tmp_path)
    run = worker_ready.run
    with h5py.File(path) as handle:
        train_ids = handle["frames/trainId"][:]
        cell_ids = handle["frames/cellId"][:]
    for train_id in run.detector_trains:
        first = run.first_row(train_id)
        rows = slice(first, first + run.rows_per_train)
        assert (train_ids[rows] == train_id).all()
        assert list(cell_ids[rows]) == list(worker_ready.classification.lit)


def test_a_result_of_the_wrong_length_is_a_programming_fault(worker_ready, tmp_path):
    block = worker_ready.plan.blocks[0]
    result = worker.process_block(block)
    result.train_id = result.train_id[:-1]
    with h5py.File(tmp_path / "short.h5", "w"):
        pass
    with JungfrauWaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, tmp_path / "short2.h5"
    ) as out:
        with pytest.raises(ValueError, match="expects"):
            out.write_block(block, result)


# ── resume ───────────────────────────────────────────────────────────────────
def test_a_fresh_file_has_no_complete_block(worker_ready, tmp_path):
    with JungfrauWaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, tmp_path / "fresh.h5"
    ) as out:
        assert not any(out.block_complete(b) for b in worker_ready.plan.blocks)


def test_a_written_block_is_complete_and_reopening_keeps_it(worker_ready, tmp_path):
    path = tmp_path / "resume.h5"
    first = worker_ready.plan.blocks[0]
    with JungfrauWaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        out.write_block(first, worker.process_block(first))
    with JungfrauWaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        assert out.block_complete(first)
        assert not out.block_complete(worker_ready.plan.blocks[-1])


def test_a_different_config_hash_is_refused(worker_ready, tmp_path):
    path = tmp_path / "hash.h5"
    JungfrauWaxsWriter.open_or_create(worker_ready.cfg, worker_ready.plan, path).close()
    other = dataclasses.replace(worker_ready.cfg, npt=worker_ready.cfg.npt // 2)
    with pytest.raises(ConfigHashMismatch, match="overwrite=True"):
        JungfrauWaxsWriter.open_or_create(other, worker_ready.plan, path)
    JungfrauWaxsWriter.open_or_create(
        dataclasses.replace(other, overwrite=True), worker_ready.plan, path
    ).close()


def test_mark_and_mark_remaining_stamp_the_ledger(worker_ready, tmp_path):
    path = tmp_path / "marks.h5"
    with JungfrauWaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        out.mark(worker_ready.plan.blocks[0], FrameStatus.WORKER_ERROR, "boom")
        out.mark_remaining(FrameStatus.NOT_PROCESSED)
        assert (
            out.status_summary()["WORKER_ERROR"] == worker_ready.plan.blocks[0].n_frames
        )
        assert out.any_not_ok()
    with h5py.File(path) as handle:
        errors = json.loads(handle["provenance"].attrs["block_errors"])
        assert errors["0"] == "boom"


# ── the reducers ─────────────────────────────────────────────────────────────
def test_per_cell_places_frames_by_label(worker_ready, tmp_path):
    grid = per_cell(written(worker_ready, tmp_path))
    run = worker_ready.run

    assert grid.dims == ("trainId", "cellId", "q")
    assert list(grid.trainId.values) == list(run.train_ids)
    assert list(grid.cellId.values) == list(worker_ready.classification.lit)
    assert grid.shape[2] == worker_ready.cfg.npt
    assert (grid["n_frames"] == 1).all()
    assert json.loads(grid.attrs["unplaced"]) == {}
    assert np.isfinite(grid.values).all()
    assert (grid.values >= 0).all()


def test_per_cell_leaves_a_missing_train_as_zeros(
    cfg, mock_run_factory, operator, model, tmp_path
):
    from analysis.waxs.plan import build_plan, open_detector

    run, dc = mock_run_factory(zero_entry_trains=(10002,))
    op, ai = operator
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)
    worker.init_from_detector(cfg, op, ai, model, LIT_CELLS, det)
    try:
        path = tmp_path / "gap.h5"
        with JungfrauWaxsWriter.open_or_create(cfg, plan, path) as out:
            for block in plan.blocks:
                out.write_block(block, worker.process_block(block))
    finally:
        worker._STATE = None

    grid = per_cell(path)
    empty = grid.sel(trainId=10002)
    assert (empty["n_frames"] == 0).all()
    assert (empty.values == 0).all()  # zeros, never NaN


def test_pooled_per_train_agrees_with_the_stored_sums(worker_ready, tmp_path):
    path = written(worker_ready, tmp_path)
    pooled = pooled_per_train(path)
    with h5py.File(path) as handle:
        signal = handle["frames/signal"][:]
        normalization = handle["frames/normalization"][:]
        first = int(handle["trains/first"][0])
        count = int(handle["trains/count"][0])

    rows = slice(first, first + count)
    expected = signal[rows].sum(axis=0) / normalization[rows].sum(axis=0)
    got = pooled["intensity"].values[0]
    assert got == pytest.approx(expected, rel=1e-5)
    assert int(pooled["n_frames"].values[0]) == count


def test_pooling_really_does_cancel_per_frame_negatives(worker_ready, tmp_path):
    """§3 D3's actual claim, forced rather than waited for.

    "Store the variance unclamped, pooling fixes it" is a claim about
    *cancellation*, and a test that only ever saw positive frames would pass
    whether or not it were true. So half a train's frames are driven negative
    in one bin and the rest positive by more — how noise around a small true
    variance behaves — and the pooled sum has to survive it.
    """
    path = written(worker_ready, tmp_path)
    with h5py.File(path, "r+") as handle:
        variance = handle["frames/variance"][:]
        count = int(handle["trains/count"][0])
        first = int(handle["trains/first"][0])
        variance[first : first + count // 2, 0] = -1.0
        variance[first + count // 2 : first + count, 0] = 3.0
        handle["frames/variance"][:] = variance

    per_train = pooled_per_train(path)
    assert per_train["sigma"].values[0, 0] > 0
    assert int(per_train.attrs["negative_variance_bins"]) == 0

    # ...and when they do not cancel, the reducer says so instead of rooting a
    # negative number.
    with h5py.File(path, "r+") as handle:
        variance = handle["frames/variance"][:]
        variance[first : first + count, 0] = -1.0
        handle["frames/variance"][:] = variance

    per_train = pooled_per_train(path)
    assert per_train["sigma"].values[0, 0] == 0.0
    assert int(per_train.attrs["negative_variance_bins"]) >= 1
