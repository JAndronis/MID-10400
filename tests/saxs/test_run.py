"""Orchestration of the SAXS pass: failure paths, resume and a real spawned pool."""

from __future__ import annotations

import os
from concurrent.futures import BrokenExecutor, Future
from dataclasses import replace

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


def test_a_real_spawned_pool_writes_what_an_inline_one_writes(pipeline, tmp_path):
    """The only test that spawns workers: everything they receive must pickle.

    Every dataset of the frame table, the window sums and the photon
    statistics must equal an in-process run's exactly.
    """
    from analysis.saxs.photon_stats import FILE_NAME as PHOTON_FILE
    from analysis.saxs.pixel_sums import FILE_NAME as SUMS_FILE

    cfg = replace(pipeline.cfg, pixel_sum_trains=3)
    for name, pool in (("spawned", None), ("inline", InlinePool)):
        run_agipd_saxs(
            cfg,
            dc=pipeline.dc,
            geometry=pipeline.geometry,
            run_dir=pipeline.run.path,
            output_path=tmp_path / name / "out.h5",
            work_dir=tmp_path / name,
            pool_factory=pool,
            reduce="none",
        )
    for name in ("out.h5", SUMS_FILE, PHOTON_FILE):
        spawned = datasets(tmp_path / "spawned" / name)
        inline = datasets(tmp_path / "inline" / name)
        assert spawned.keys() == inline.keys(), name
        for key, values in inline.items():
            np.testing.assert_array_equal(spawned[key], values, f"{name}:{key}")


def datasets(path):
    """Every dataset of an HDF5 file, by name."""
    found = {}
    with h5py.File(path, "r") as handle:
        handle.visititems(
            lambda name, obj: (
                found.__setitem__(name, obj[()])
                if isinstance(obj, h5py.Dataset)
                else None
            )
        )
    return found


def test_thread_env_is_set_before_any_pool(pipeline, tmp_path, monkeypatch):
    """Spawned children inherit the thread variables, so the parent must set them."""
    for name in THREAD_ENV:
        monkeypatch.delenv(name, raising=False)
    run_on_mock(pipeline, tmp_path)
    for name in THREAD_ENV:
        assert os.environ[name] == "1"


# ── failure paths ─────────────────────────────────────────────────────────────
def test_worker_exception_becomes_worker_error(pipeline, tmp_path, monkeypatch):
    """A block the worker cannot read is recorded, the run continues, and the
    pass raises unless an incomplete run is allowed."""
    real = worker.process_block

    def explode(block: Block):
        if block.index == 0:
            raise RuntimeError("synthetic worker failure")
        return real(block)

    monkeypatch.setattr(worker, "process_block", explode)
    with pytest.raises(IncompleteRun, match="WORKER_ERROR"):
        run_on_mock(pipeline, tmp_path)
    allowed = run_on_mock(pipeline, tmp_path / "allowed", allow_incomplete=True)
    assert allowed.n_frames.values.sum() == pipeline.plan.n_frames - len(
        pipeline.plan.blocks[0].rows()
    )

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


# ── resume ────────────────────────────────────────────────────────────────────


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
