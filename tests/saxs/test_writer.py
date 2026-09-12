"""Worker results, output schema, rows by label and resume (§6.5, §7; P3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("extra_data")

import h5py  # noqa: E402

from analysis.saxs import worker  # noqa: E402
from analysis.saxs.status import FrameStatus  # noqa: E402
from analysis.saxs.writer import (  # noqa: E402
    AgipdSaxsWriter,
    ConfigHashMismatch,
)


def process_all(pipeline):
    return {b.index: worker.process_block(b) for b in pipeline.plan.blocks}


# ── worker ────────────────────────────────────────────────────────────────────
def test_worker_integrates_every_frame(worker_ready):
    for block in worker_ready.plan.blocks:
        result = worker.process_block(block)
        assert result.n_frames == block.n_frames
        assert (result.status == FrameStatus.OK).all()
        assert result.normalization.sum() > 0
        assert np.isfinite(result.signal).all()


def test_worker_labels_come_from_the_reader(worker_ready):
    """Identity is read, never inferred from position (CLAUDE.md pitfall 4)."""
    for block in worker_ready.plan.blocks:
        result = worker.process_block(block)
        expected = np.concatenate(
            [
                np.full(count, train_id, dtype=np.uint64)
                for train_id, count in zip(
                    block.train_ids, block.expected_frames, strict=True
                )
            ]
        )
        assert np.array_equal(result.train_id, expected)
        # cellIds cycle within a train rather than counting up across it
        for train_id in block.train_ids:
            first, count = block.frames_for(train_id)
            offset = int(np.flatnonzero(result.train_id == train_id)[0])
            assert np.array_equal(
                result.cell_id[offset : offset + count], np.arange(count)
            )


def test_worker_requires_init():
    worker._STATE = None
    from analysis.saxs.plan import Block

    with pytest.raises(RuntimeError, match="worker.init"):
        worker.process_block(Block(0, (1,), (1,), (0,)))


def test_worker_records_bits_and_unseen_cells(worker_ready):
    result = worker.process_block(worker_ready.plan.blocks[0])
    assert result.bits_present != 0
    assert result.unseen_cells == 0  # every cell was sampled for the base masks
    assert set(result.timings) == {"read_data", "read_mask", "integrate"}


# ── writer ────────────────────────────────────────────────────────────────────
def test_output_schema(worker_ready, tmp_path):
    path = tmp_path / "out.h5"
    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        for block in worker_ready.plan.blocks:
            out.write_block(block, worker.process_block(block))

    n, npt = worker_ready.plan.n_frames, worker_ready.cfg.npt
    with h5py.File(path) as f:
        for name in (
            "trainId",
            "reader_pulseId",
            "cellId",
            "status",
            "photons_valid",
            "n_bad_pixels",
            "n_frame_specific",
            "max_count",
        ):
            assert f[f"frames/{name}"].shape == (n,)
        for name in ("signal", "normalization", "variance"):
            assert f[f"frames/{name}"].shape == (n, npt)
            assert f[f"frames/{name}"].dtype == np.float32
        assert f["trains/trainId"].shape == (len(worker_ready.plan.trains),)
        assert f["provenance"].attrs["config_hash"] == worker_ready.cfg.config_hash()
        assert f["provenance"].attrs["detector_name"] == "MID_DET_AGIPD1M-1"


def test_no_nan_anywhere(worker_ready, tmp_path):
    """§3 rule 7: status codes, never NaN sentinels."""
    path = tmp_path / "out.h5"
    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        for block in worker_ready.plan.blocks:
            out.write_block(block, worker.process_block(block))

    with h5py.File(path) as f:

        def check(name, obj):
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.floating):
                assert np.isfinite(obj[:]).all(), name

        f.visititems(check)


def test_rows_are_written_by_label_in_any_order(worker_ready, tmp_path):
    """Completing blocks out of order must not move a single row."""
    ordered = tmp_path / "ordered.h5"
    shuffled = tmp_path / "shuffled.h5"
    results = process_all(worker_ready)

    for path, order in (
        (ordered, list(results)),
        (shuffled, list(reversed(list(results)))),
    ):
        with AgipdSaxsWriter.open_or_create(
            worker_ready.cfg, worker_ready.plan, path
        ) as out:
            for index in order:
                block = worker_ready.plan.blocks[index]
                out.write_block(block, results[index])

    with h5py.File(ordered) as a, h5py.File(shuffled) as b:
        for name in ("trainId", "cellId", "status", "signal"):
            assert np.array_equal(a[f"frames/{name}"][:], b[f"frames/{name}"][:])
        # every row carries the trainId the plan assigned it
        expected = np.concatenate(
            [
                np.full(t.n_frames, t.train_id, dtype=np.uint64)
                for t in worker_ready.plan.trains
                if t.n_frames
            ]
        )
        assert np.array_equal(a["frames/trainId"][:], expected)


def test_mislabelled_frames_become_label_mismatch(worker_ready, tmp_path):
    """A result whose labels disagree with the plan is never stored as OK."""
    block = worker_ready.plan.blocks[0]
    result = worker.process_block(block)
    result.train_id[0] = 999999

    path = tmp_path / "out.h5"
    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        out.write_block(block, result)
        with h5py.File(path) as _:
            pass
    with h5py.File(path) as f:
        assert f["frames/status"][0] == FrameStatus.LABEL_MISMATCH
        # the row still carries the trainId the plan assigned, not the bad one
        assert f["frames/trainId"][0] == block.train_ids[0]


def test_write_block_rejects_a_wrong_length_result(worker_ready, tmp_path):
    block = worker_ready.plan.blocks[0]
    result = worker.process_block(block)
    result.status = result.status[:-1]
    result.train_id = result.train_id[:-1]
    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, tmp_path / "out.h5"
    ) as out:
        with pytest.raises(ValueError, match="frames"):
            out.write_block(block, result)


# ── resume and config hash ────────────────────────────────────────────────────
def test_block_complete_tracks_progress(worker_ready, tmp_path):
    path = tmp_path / "out.h5"
    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        blocks = worker_ready.plan.blocks
        assert not any(out.block_complete(b) for b in blocks)
        out.write_block(blocks[0], worker.process_block(blocks[0]))
        assert out.block_complete(blocks[0])
        assert not out.block_complete(blocks[1])


def test_reopening_resumes_rather_than_restarting(worker_ready, tmp_path):
    path = tmp_path / "out.h5"
    blocks = worker_ready.plan.blocks
    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        out.write_block(blocks[0], worker.process_block(blocks[0]))

    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        remaining = [b for b in blocks if not out.block_complete(b)]
        assert [b.index for b in remaining] == [b.index for b in blocks[1:]]


def test_config_hash_mismatch_is_refused(worker_ready, tmp_path):
    path = tmp_path / "out.h5"
    with AgipdSaxsWriter.open_or_create(worker_ready.cfg, worker_ready.plan, path):
        pass
    changed = replace(worker_ready.cfg, sdd_m=7.6)
    with pytest.raises(ConfigHashMismatch, match="config hash"):
        AgipdSaxsWriter.open_or_create(changed, worker_ready.plan, path)


def test_overwrite_replaces_a_mismatched_file(worker_ready, tmp_path):
    path = tmp_path / "out.h5"
    with AgipdSaxsWriter.open_or_create(worker_ready.cfg, worker_ready.plan, path):
        pass
    changed = replace(worker_ready.cfg, sdd_m=7.6, overwrite=True)
    with AgipdSaxsWriter.open_or_create(changed, worker_ready.plan, path) as out:
        assert not out.block_complete(worker_ready.plan.blocks[0])


# ── marking ───────────────────────────────────────────────────────────────────
def test_mark_records_a_status_and_a_message(worker_ready, tmp_path):
    path = tmp_path / "out.h5"
    block = worker_ready.plan.blocks[0]
    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        out.mark(block, FrameStatus.WORKER_ERROR, "RuntimeError('boom')")
        assert out.status_summary()["WORKER_ERROR"] == block.n_frames
    with h5py.File(path) as f:
        assert "boom" in f["provenance"].attrs["block_errors"]


def test_mark_remaining_stamps_only_unprocessed_frames(worker_ready, tmp_path):
    path = tmp_path / "out.h5"
    blocks = worker_ready.plan.blocks
    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        out.write_block(blocks[0], worker.process_block(blocks[0]))
        out.mark_remaining(FrameStatus.NOT_PROCESSED)
        summary = out.status_summary()
        assert summary["OK"] == blocks[0].n_frames
        assert summary["NOT_PROCESSED"] == sum(b.n_frames for b in blocks[1:])


def test_pooled_per_train(worker_ready, tmp_path):
    path = tmp_path / "out.h5"
    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        out.store_operator(worker_ready.op)
        for block in worker_ready.plan.blocks:
            out.write_block(block, worker.process_block(block))
        pooled = out.pooled_per_train()

    assert pooled.intensity.shape == (
        len(worker_ready.plan.trains),
        worker_ready.cfg.npt,
    )
    assert np.isfinite(pooled.intensity.values).all()
    assert (pooled.intensity.values >= 0).all()
    assert pooled.n_frames.values.sum() == worker_ready.plan.n_frames
    assert np.array_equal(pooled.q.values, worker_ready.op.q)
