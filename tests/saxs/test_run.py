"""Orchestration of the SAXS pass: failure paths, resume and a real spawned pool."""

from __future__ import annotations

import os
from concurrent.futures import BrokenExecutor, Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("extra_data")

import h5py  # noqa: E402

from analysis.common.cpu import THREAD_ENV  # noqa: E402
from analysis.common.plan import Block  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.common.writer import IncompleteRun  # noqa: E402
from analysis.saxs import worker  # noqa: E402
from analysis.saxs.run import run_agipd_saxs  # noqa: E402


class InlinePool:
    """Runs blocks in this process, so failure paths need no real children.

    The initializer runs once, exactly as a spawned worker would run it, and
    ``submit`` returns an already-resolved future. Injecting this is what lets
    ``BrokenExecutor`` and ``WORKER_ERROR`` be exercised without killing
    processes or putting test hooks into ``worker``.
    """

    def __init__(self, n_workers, initializer=None, initargs=(), **kwargs):
        self.n_workers = n_workers
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        future: Future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as error:  # noqa: BLE001 - mirrors a real pool
            future.set_exception(error)
        return future


class BrokenPool(InlinePool):
    """A pool that dies part-way through, as a killed worker would."""

    def submit(self, fn, *args):
        future: Future = Future()
        future.set_exception(BrokenExecutor("worker died"))
        return future


def run_on_mock(pipeline, tmp_path, **overrides):
    factory = overrides.pop("pool_factory", None) or InlinePool
    cfg = replace(pipeline.cfg, **overrides) if overrides else pipeline.cfg
    return run_agipd_saxs(
        cfg,
        dc=pipeline.dc,
        geometry=None if cfg.geometry_file else pipeline.geometry,
        run_dir=pipeline.run.path,
        output_path=tmp_path / "out.h5",
        work_dir=tmp_path,
        pool_factory=factory,
    )


@pytest.fixture
def pipeline(mock_pipeline, geom):
    mock_pipeline.geometry = geom
    return mock_pipeline


# ── the happy path ────────────────────────────────────────────────────────────
def test_end_to_end_inline(pipeline, tmp_path):
    pooled = run_on_mock(pipeline, tmp_path)
    assert pooled.n_frames.values.sum() == pipeline.plan.n_frames
    assert np.isfinite(pooled.intensity.values).all()

    with h5py.File(tmp_path / "out.h5") as f:
        assert (f["frames/status"][:] == FrameStatus.OK).all()
        assert f["provenance"].attrs["n_workers"] == pipeline.cfg.workers
        assert "operator_sha256" in f["provenance"].attrs
        # Acceptance reads the wall time from the file alone.
        assert f["provenance"].attrs["wall_s"] > 0
        assert f["provenance"].attrs["started_at"] > 0
        assert f["masks"].attrs["operator_sha256"] == f["operator"].attrs["sha256"]
        assert f["q/centers"].attrs["unit"] == "nm^-1"


def test_end_to_end_with_a_real_spawned_pool(pipeline, tmp_path):
    """The only test that actually spawns workers."""
    pooled = run_agipd_saxs(
        pipeline.cfg,
        dc=pipeline.dc,
        geometry=pipeline.geometry,
        run_dir=pipeline.run.path,
        output_path=tmp_path / "out.h5",
        work_dir=tmp_path,
    )
    assert pooled.n_frames.values.sum() == pipeline.plan.n_frames

    inline = run_agipd_saxs(
        pipeline.cfg,
        dc=pipeline.dc,
        geometry=pipeline.geometry,
        run_dir=pipeline.run.path,
        output_path=tmp_path / "inline.h5",
        work_dir=tmp_path,
        pool_factory=InlinePool,
    )
    np.testing.assert_allclose(
        pooled.intensity.values, inline.intensity.values, rtol=0, atol=0
    )


def test_thread_env_is_set_before_any_pool(pipeline, tmp_path, monkeypatch):
    """Spawned children inherit the thread variables, so the parent must set them."""
    for name in THREAD_ENV:
        monkeypatch.delenv(name, raising=False)
    run_on_mock(pipeline, tmp_path)
    for name in THREAD_ENV:
        assert os.environ[name] == "1"


# ── failure paths ─────────────────────────────────────────────────────────────
def test_worker_exception_becomes_worker_error(pipeline, tmp_path, monkeypatch):
    """A block the worker cannot read is recorded, and the run continues."""
    real = worker.process_block

    def explode(block: Block):
        if block.index == 0:
            raise RuntimeError("synthetic worker failure")
        return real(block)

    monkeypatch.setattr(worker, "process_block", explode)
    with pytest.raises(IncompleteRun):
        run_on_mock(pipeline, tmp_path)

    failed = pipeline.plan.blocks[0]
    with h5py.File(tmp_path / "out.h5") as f:
        status = f["frames/status"][:]
        rows = failed.rows()
        assert (status[rows] == FrameStatus.WORKER_ERROR).all()
        # every other block still finished
        assert (np.delete(status, rows) == FrameStatus.OK).all()
        assert "synthetic worker failure" in f["provenance"].attrs["block_errors"]


def test_killed_worker_marks_not_processed_and_raises(pipeline, tmp_path):
    with pytest.raises(BrokenExecutor):
        run_agipd_saxs(
            pipeline.cfg,
            dc=pipeline.dc,
            geometry=pipeline.geometry,
            run_dir=pipeline.run.path,
            output_path=tmp_path / "out.h5",
            work_dir=tmp_path,
            pool_factory=BrokenPool,
        )
    with h5py.File(tmp_path / "out.h5") as f:
        assert (f["frames/status"][:] == FrameStatus.NOT_PROCESSED).all()


def test_incomplete_run_raises_unless_allowed(pipeline, tmp_path, monkeypatch):
    real = worker.process_block
    monkeypatch.setattr(
        worker,
        "process_block",
        lambda b: (
            (_ for _ in ()).throw(RuntimeError("nope")) if b.index == 0 else real(b)
        ),
    )
    with pytest.raises(IncompleteRun, match="OK"):
        run_on_mock(pipeline, tmp_path)

    pooled = run_on_mock(
        pipeline, tmp_path / "allowed", allow_incomplete=True, overwrite=True
    )
    assert pooled is not None


# ── resume ────────────────────────────────────────────────────────────────────
def test_resume_processes_only_the_missing_blocks(pipeline, tmp_path, monkeypatch):
    """The frame table's resume, with window sums off.

    With them on, the rerun would also rebuild the window that lost the failed
    block's trains, reading its other blocks for their sums. ``test_pixel_sums.py``
    covers that case.
    """
    pipeline = SimpleNamespace(**vars(pipeline))
    pipeline.cfg = replace(pipeline.cfg, pixel_sum_trains=None)
    real = worker.process_block
    monkeypatch.setattr(
        worker,
        "process_block",
        lambda b: (
            (_ for _ in ()).throw(RuntimeError("first attempt"))
            if b.index == 0
            else real(b)
        ),
    )
    with pytest.raises(IncompleteRun):
        run_on_mock(pipeline, tmp_path)

    # reset the failed block the way a rerun after a crash would find it
    failed = pipeline.plan.blocks[0]
    with h5py.File(tmp_path / "out.h5", "r+") as f:
        rows = failed.rows()
        f["frames/status"][int(rows[0]) : int(rows[-1]) + 1] = FrameStatus.NOT_PROCESSED

    monkeypatch.setattr(worker, "process_block", real)
    seen: list[int] = []

    def recording(block):
        seen.append(block.index)
        return real(block)

    monkeypatch.setattr(worker, "process_block", recording)
    run_on_mock(pipeline, tmp_path)

    assert seen == [failed.index]
    with h5py.File(tmp_path / "out.h5") as f:
        assert (f["frames/status"][:] == FrameStatus.OK).all()


def test_rerun_of_a_complete_run_does_no_work(pipeline, tmp_path, monkeypatch):
    run_on_mock(pipeline, tmp_path)

    def forbidden(block):
        raise AssertionError("a complete run must not reprocess anything")

    monkeypatch.setattr(worker, "process_block", forbidden)
    pooled = run_on_mock(pipeline, tmp_path)
    assert pooled.n_frames.values.sum() == pipeline.plan.n_frames


def test_a_train_whose_frames_carry_another_train_id_is_label_mismatch(
    pipeline, mock_run_factory, tmp_path
):
    """Module 0's rows of train 10003 say 11003; the INDEX still files them under 10003.

    The frames' identity is then unknown, so none of that train's frames may be
    integrated, summed or histogrammed, while every other train is.
    """
    from analysis.saxs.photon_stats import FILE_NAME as PHOTON_FILE
    from analysis.saxs.pixel_sums import FILE_NAME as SUMS_FILE

    run, dc = mock_run_factory(mislabel_train=10003)
    cfg = replace(pipeline.cfg, pixel_sum_trains=3)
    kwargs = {
        "dc": dc,
        "geometry": pipeline.geometry,
        "run_dir": run.path,
        "output_path": tmp_path / "out.h5",
        "work_dir": tmp_path,
        "pool_factory": InlinePool,
    }
    with pytest.raises(IncompleteRun, match="LABEL_MISMATCH"):
        run_agipd_saxs(cfg, **kwargs)

    first, n = run.first_row(10003), run.frames_by_train[10003]
    for name in ("out.h5", PHOTON_FILE):
        with h5py.File(tmp_path / name) as f:
            status = f["frames/status"][:]
        mislabelled = np.zeros(status.size, dtype=bool)
        mislabelled[first : first + n] = True
        assert (status[mislabelled] == FrameStatus.LABEL_MISMATCH).all(), name
        assert (status[~mislabelled] == FrameStatus.OK).all(), name
    with h5py.File(tmp_path / SUMS_FILE) as f:
        ids, codes = f["trains/trainId"][:], f["trains/status"][:]
    trains = dict(zip(ids.tolist(), codes, strict=True))
    assert trains.pop(10003) == FrameStatus.LABEL_MISMATCH
    assert all(code == FrameStatus.OK for code in trains.values())
