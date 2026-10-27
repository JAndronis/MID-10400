"""The AGIPD frame table: labels from the reader, its schema, and what it stores.

The detector-agnostic writer (rows by label, resume, refusal, marking) is
tested once, in ``tests/common/test_writer.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("extra_data")

import h5py  # noqa: E402

from analysis.common.status import FrameStatus  # noqa: E402
from analysis.saxs import worker  # noqa: E402
from analysis.saxs.writer import AgipdSaxsWriter  # noqa: E402


def test_worker_labels_come_from_the_reader(worker_ready):
    """Identity is read, never inferred from position, which a dropped train shifts."""
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
            _, count = block.frames_for(train_id)
            offset = int(np.flatnonzero(result.train_id == train_id)[0])
            assert np.array_equal(
                result.cell_id[offset : offset + count], np.arange(count)
            )


def test_the_frame_table_has_its_columns_and_no_nan(worker_ready, tmp_path):
    """One row per frame and float32 sums per q bin; a failure is a code, never NaN."""
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
        assert (f["frames/status"][:] == FrameStatus.OK).all()
        assert f["trains/trainId"].shape == (len(worker_ready.plan.trains),)
        assert f["provenance"].attrs["config_hash"] == worker_ready.cfg.config_hash()
        assert f["provenance"].attrs["detector_name"] == "MID_DET_AGIPD1M-1"

        def check(name, obj):
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.floating):
                assert np.isfinite(obj[:]).all(), name

        f.visititems(check)


def test_the_operator_and_masks_are_stored_as_built(worker_ready, tmp_path):
    """A frame table carries the operator and base masks that integrated it."""
    op, masks = worker_ready.op, worker_ready.masks
    sampled = np.array([10000, 10003], dtype=np.uint64)
    path = tmp_path / "out.h5"
    with AgipdSaxsWriter.open_or_create(
        worker_ready.cfg, worker_ready.plan, path
    ) as out:
        out.store_operator(op)
        out.store_masks(masks, sampled)

    with h5py.File(path) as f:
        for name in ("coef", "bins", "indptr", "omega", "q"):
            np.testing.assert_array_equal(f[f"operator/{name}"][()], getattr(op, name))
        assert f["operator"].attrs["sha256"] == op.sha256
        np.testing.assert_array_equal(f["q/centers"][()], op.q)
        group = f["masks"]
        np.testing.assert_array_equal(group["cells"][()], masks.cells)
        np.testing.assert_array_equal(
            np.unpackbits(group["base_bad_packed"][()], axis=-1)[
                :, : masks.base_bad.shape[-1]
            ].astype(bool),
            masks.base_bad,
        )
        np.testing.assert_array_equal(group["D"][()], masks.denominators)
        np.testing.assert_array_equal(group["sampled_trains"][()], sampled)
        assert group.attrs["sha256"] == masks.sha256
        assert group.attrs["operator_sha256"] == op.sha256
