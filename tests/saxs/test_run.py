"""Orchestration: failure paths, resume and a real spawned pool (§6.7; P3)."""

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
        # P4 checks the wall time against the §2 budget from the file alone.
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
    """§3 rule 3: spawned children inherit it, so it must be set in the parent."""
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
    block's trains, reading its other blocks for their sums — covered in
    ``test_pixel_sums.py``.
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

    # the failed block is WORKER_ERROR, not NOT_PROCESSED, so it is complete;
    # reset it the way a rerun after a crash would find it
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
