"""Per-pixel window sums (context file §15; phase S1).

The reference throughout is :func:`analysis.cache.detector_sums`, an independent
read of the same proc files: every window the pass writes must equal it over the
window's member trains, bit for bit. The awkward run carries a dropped train, a
train short of modules and a zero-frame train, so windows are defined by train
id and not by position (CLAUDE.md pitfall 4).
"""

from __future__ import annotations

import pickle
from concurrent.futures import BrokenExecutor, Future, ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("extra_data")
pytest.importorskip("h5netcdf")  # analysis.cache writes its reference as netCDF

import h5py  # noqa: E402
from test_run import BrokenPool, InlinePool  # noqa: E402

from analysis import cache  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.common.writer import ConfigHashMismatch, IncompleteRun  # noqa: E402
from analysis.saxs import worker  # noqa: E402
from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.saxs.masks import load_masks  # noqa: E402
from analysis.saxs.operator import load_operator  # noqa: E402
from analysis.saxs.pixel_sums import (  # noqa: E402
    FILE_NAME,
    WindowSpec,
    accumulate_train,
    detector_sums,
    pixel_sums_hash,
    train_data_ok,
    window_sums,
    window_table,
)
from analysis.saxs.run import run_agipd_saxs, run_pixel_sums  # noqa: E402
from analysis.saxs.sparse import gather  # noqa: E402

#: A dropped train (10002), one short of modules (10004), one with no frames
#: (10006): three windows of three train ids from 10000 hold 2, 2 and 1 trains.
AWKWARD = {
    "train_ids": (10000, 10001, 10003, 10004, 10005, 10006, 10007),
    "short_module_trains": (10004,),
    "zero_frame_trains": (10006,),
}


@pytest.fixture
def reference(monkeypatch, tmp_path):
    """``cache.detector_sums`` over some trains, in threads rather than processes."""
    monkeypatch.setattr(
        "analysis.common.cpu.default_pool", lambda n, **kw: ThreadPoolExecutor(n)
    )

    def read(run, train_ids):
        return cache.detector_sums(
            1,
            [int(t) for t in train_ids],
            run_dir=run.path,
            n_workers=1,
            cache_dir=tmp_path / "reference",
            refresh=True,
        )

    return read


def run_pass(run_cfg, geom, factory_result, out_dir, *, pool=InlinePool, **changes):
    """The full pass on a mock run, both files under ``out_dir``."""
    run, dc = factory_result
    cfg = replace(run_cfg, **changes)
    run_agipd_saxs(
        cfg,
        dc=dc,
        geometry=geom,
        run_dir=run.path,
        output_path=out_dir / "out.h5",
        work_dir=out_dir,
        pool_factory=pool,
        reduce="none",
    )
    return out_dir / FILE_NAME


def assert_same_sums(ours, ref):
    """Same variables, dims, dtypes, coordinates and values as the cache's."""
    for name in ("counts", "valid_frames", "train_id"):
        assert ours[name].dims == ref[name].dims, name
        assert ours[name].dtype == ref[name].dtype, name
        np.testing.assert_array_equal(ours[name].values, ref[name].values, name)
    np.testing.assert_array_equal(ours.module.values, ref.module.values)
    assert ours.attrs["n_frames"] == ref.attrs["n_frames"]


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


def frame_table_digest(path, h5_digest):
    """The frame table's digest, leaving out the recorded config itself."""
    skip = h5_digest.__defaults__[0] | {"config", "config_operational_fields"}
    return h5_digest(path, skip)


# ── equivalence with the cache ────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("trains_per_block", "window"), [(2, 3), (3, 3), (2, 1), (4, 10)]
)
def test_every_window_equals_detector_sums_of_its_trains(
    run_cfg, geom, mock_run_factory, reference, tmp_path, trains_per_block, window
):
    """Including blocks that straddle two windows and a partial last window."""
    run, dc = mock_run_factory(**AWKWARD)
    path = run_pass(
        run_cfg,
        geom,
        (run, dc),
        tmp_path,
        trains_per_block=trains_per_block,
        pixel_sum_trains=window,
    )

    table = window_table(path)
    assert table.written.values.all()
    summed = 0
    for index in table.window.values:
        if table.n_trains.values[index] == 0:
            continue
        ours = window_sums(path, [index])
        assert_same_sums(ours, reference(run, ours.train_id.values))
        summed += ours.train_id.size
    assert summed == 5  # every train with rows, each in exactly one window


def test_windows_are_train_id_ranges_and_record_why_a_train_is_missing(
    run_cfg, geom, mock_run_factory, tmp_path
):
    run_dc = mock_run_factory(**AWKWARD)
    path = run_pass(run_cfg, geom, run_dc, tmp_path, pixel_sum_trains=3)

    table = window_table(path)
    np.testing.assert_array_equal(table.start_trainId, [10000, 10003, 10006])
    np.testing.assert_array_equal(table.n_trains, [2, 2, 1])
    np.testing.assert_array_equal(table.n_frames, [6, 6, 3])
    assert table.attrs["window_origin_trainId"] == 10000
    with h5py.File(path, "r") as handle:
        train_ids = handle["trains/trainId"][:]
        status = dict(zip(train_ids.tolist(), handle["trains/status"][:], strict=True))
        window = handle["trains/window"][:]
    assert 10002 not in train_ids  # dropped by the DAQ: not in the run at all
    assert status[10004] == FrameStatus.MISSING_MODULES
    assert status[10006] == FrameStatus.NO_FRAMES
    assert all(status[t] == FrameStatus.OK for t in (10000, 10001, 10003, 10005, 10007))
    np.testing.assert_array_equal(window, [0, 0, 1, 1, 1, 2, 2])


def test_a_real_spawned_pool_writes_the_same_windows(
    mock_pipeline, geom, mock_run_factory, tmp_path
):
    """The partials survive pickling between processes."""
    run_dc = mock_run_factory()
    cfg = mock_pipeline.cfg
    spawned = run_pass(
        cfg, geom, run_dc, tmp_path / "spawned", pool=None, pixel_sum_trains=3
    )
    inline = run_pass(cfg, geom, run_dc, tmp_path / "inline", pixel_sum_trains=3)
    for name, values in datasets(inline).items():
        np.testing.assert_array_equal(datasets(spawned)[name], values, name)


# ── the frame table ───────────────────────────────────────────────────────────
def test_the_frame_table_does_not_depend_on_the_window_sums(
    run_cfg, geom, mock_run_factory, tmp_path, h5_digest
):
    run_dc = mock_run_factory()
    run_pass(run_cfg, geom, run_dc, tmp_path / "off", pixel_sum_trains=None)
    run_pass(run_cfg, geom, run_dc, tmp_path / "on", pixel_sum_trains=3)

    assert not (tmp_path / "off" / FILE_NAME).exists()
    assert frame_table_digest(tmp_path / "off" / "out.h5", h5_digest) == (
        frame_table_digest(tmp_path / "on" / "out.h5", h5_digest)
    )


def test_window_pooled_frame_table_equals_the_gather_over_the_window_sums(
    run_cfg, geom, mock_run_factory, tmp_path
):
    """§15: Σ S = Σ c·counts, Σ N = Σ c·Ω·valid, Σ V = Σ c²·counts off the static mask.

    This is what makes the window sums the geometry-independent record: the
    1D result of a window can be rebuilt from them under any operator.
    """
    run_dc = mock_run_factory(**AWKWARD)
    path = run_pass(run_cfg, geom, run_dc, tmp_path, pixel_sum_trains=3)
    op = load_operator(tmp_path / "operator.npz")
    static = load_masks(tmp_path / "masks.npz").static_bad

    with h5py.File(tmp_path / "out.h5", "r") as frames:
        row_train = frames["frames/trainId"][:]
        ok = frames["frames/status"][:] == FrameStatus.OK
        stored = {
            name: frames[f"frames/{name}"][:].astype(np.float64)
            for name in ("signal", "normalization", "variance")
        }

    for index in window_table(path).window.values:
        sums = window_sums(path, [index])
        rows = ok & np.isin(row_train, sums.train_id.values)
        counts = sums.counts.values.reshape(-1).astype(np.float64)
        valid = sums.valid_frames.values.reshape(-1).astype(np.float64)
        hit = np.flatnonzero((counts > 0) & ~static)
        seen = np.flatnonzero((valid > 0) & ~static)
        expected = {
            "signal": gather(op, hit, counts[hit]),
            "variance": gather(op, hit, counts[hit], squared=True),
            "normalization": gather(op, seen, op.omega[seen] * valid[seen]),
        }
        for name, value in expected.items():
            np.testing.assert_allclose(
                stored[name][rows].sum(axis=0), value, rtol=1e-6, atol=0, err_msg=name
            )


# ── trains left out ───────────────────────────────────────────────────────────
def test_a_train_with_negative_counts_is_left_out_whole(
    run_cfg, mock_run_factory, reference, tmp_path
):
    """Module 3 carries a -1 in the first frame of train 10000.

    Through the sums-only mode: the full pass refuses this run earlier, at the
    self-test, which samples that frame. Both modes leave a train out with the
    same ``BlockSums.add_train``.
    """
    run, dc = mock_run_factory(negative_counts_module=3)
    cfg = replace(run_cfg, pixel_sum_trains=3)
    path = tmp_path / FILE_NAME
    kwargs = {"dc": dc, "run_dir": run.path, "pool_factory": InlinePool}
    with pytest.raises(IncompleteRun, match="DATA_CHECK_FAILED"):
        run_pixel_sums(cfg, pixel_sums_path=path, **kwargs)
    run_pixel_sums(
        replace(cfg, allow_incomplete=True),
        pixel_sums_path=tmp_path / "allowed" / FILE_NAME,
        **kwargs,
    )

    with h5py.File(path, "r") as handle:
        status = dict(
            zip(
                handle["trains/trainId"][:].tolist(),
                handle["trains/status"][:],
                strict=True,
            )
        )
    assert status[10000] == FrameStatus.DATA_CHECK_FAILED
    first = window_sums(path, [0])
    np.testing.assert_array_equal(first.train_id, [10001, 10002])
    assert_same_sums(first, reference(run, [10001, 10002]))


def test_a_failed_block_is_left_out_and_the_run_is_incomplete(
    run_cfg, geom, mock_run_factory, reference, tmp_path, monkeypatch
):
    run, dc = mock_run_factory()
    real = worker.process_block

    def explode(block):
        if block.index == 1:  # trains 10002 and 10003, one in each window
            raise RuntimeError("synthetic worker failure")
        return real(block)

    monkeypatch.setattr(worker, "process_block", explode)
    with pytest.raises(IncompleteRun, match="window_sums"):
        run_pass(run_cfg, geom, (run, dc), tmp_path, pixel_sum_trains=3)

    path = tmp_path / FILE_NAME
    assert window_table(path).written.values.all()
    with h5py.File(path, "r") as handle:
        status = handle["trains/status"][:]
        assert "synthetic worker failure" in handle["provenance"].attrs["block_errors"]
    np.testing.assert_array_equal(
        status, [0, 0, FrameStatus.WORKER_ERROR, FrameStatus.WORKER_ERROR, 0, 0]
    )
    for index, members in ((0, [10000, 10001]), (1, [10004, 10005])):
        sums = window_sums(path, [index])
        np.testing.assert_array_equal(sums.train_id, members)
        assert_same_sums(sums, reference(run, members))


@pytest.mark.parametrize(("window", "reread"), [(3, [0, 1, 2]), (1, [1])])
def test_a_window_that_lost_trains_to_a_worker_error_is_rebuilt(
    run_cfg, geom, mock_run_factory, reference, tmp_path, monkeypatch, window, reread
):
    """A worker error says nothing about the data, so a rerun retries it.

    Block 1 (trains 10002, 10003) fails once. With 3-train windows it holds a
    train of each window, so the rerun reads every block; with 1-train windows
    only its own. The frame table keeps its WORKER_ERROR either way — that one
    stays until its rows are reset — so nothing is integrated and the run is
    still incomplete, but the window sums come out whole.
    """
    run, dc = mock_run_factory()
    real_process, real_integrate = worker.process_block, worker.integrate_frame

    def explode(block):
        if block.index == 1:
            raise RuntimeError("transient")
        return real_process(block)

    monkeypatch.setattr(worker, "process_block", explode)
    with pytest.raises(IncompleteRun):
        run_pass(run_cfg, geom, (run, dc), tmp_path, pixel_sum_trains=window)

    seen, integrated = [], []
    monkeypatch.setattr(
        worker, "process_block", lambda b: seen.append(b.index) or real_process(b)
    )
    monkeypatch.setattr(
        worker,
        "integrate_frame",
        lambda *a: integrated.append(1) or real_integrate(*a),
    )
    with pytest.raises(IncompleteRun, match="'frames'.*WORKER_ERROR"):
        run_pass(run_cfg, geom, (run, dc), tmp_path, pixel_sum_trains=window)

    assert sorted(seen) == reread
    assert not integrated
    path = tmp_path / FILE_NAME
    with h5py.File(path, "r") as handle:
        assert (handle["trains/status"][:] == FrameStatus.OK).all()
    clean = run_pass(
        run_cfg, geom, (run, dc), tmp_path / "clean", pixel_sum_trains=window
    )
    for name, values in datasets(clean).items():
        np.testing.assert_array_equal(datasets(path)[name], values, name)
    sums = window_sums(path, window_table(path).window.values)
    assert_same_sums(sums, reference(run, range(10000, 10006)))


def test_resetting_a_failed_block_completes_frames_and_windows(
    run_cfg, geom, mock_run_factory, tmp_path, monkeypatch
):
    """The frame table's retry — reset the rows, rerun — repairs both files."""
    run_dc = mock_run_factory()
    real = worker.process_block
    monkeypatch.setattr(
        worker,
        "process_block",
        lambda b: (_ for _ in ()).throw(RuntimeError("x")) if b.index == 0 else real(b),
    )
    with pytest.raises(IncompleteRun):
        run_pass(run_cfg, geom, run_dc, tmp_path, pixel_sum_trains=3)
    with h5py.File(tmp_path / "out.h5", "r+") as frames:
        frames["frames/status"][0:6] = FrameStatus.NOT_PROCESSED

    monkeypatch.setattr(worker, "process_block", real)
    run_pass(run_cfg, geom, run_dc, tmp_path, pixel_sum_trains=3)  # no raise
    with h5py.File(tmp_path / FILE_NAME, "r") as handle:
        assert (handle["trains/status"][:] == FrameStatus.OK).all()


# ── resume ────────────────────────────────────────────────────────────────────
class DiesAfterOneBlock(InlinePool):
    """Completes the first block, then dies as a killed worker would."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted = 0

    def submit(self, fn, *args):
        self.submitted += 1
        if self.submitted == 1:
            return super().submit(fn, *args)
        future: Future = Future()
        future.set_exception(BrokenExecutor("worker died"))
        return future


def test_a_killed_pool_leaves_windows_unwritten_and_a_rerun_completes_them(
    run_cfg, geom, mock_run_factory, reference, tmp_path, monkeypatch
):
    run, dc = mock_run_factory()
    with pytest.raises(BrokenExecutor):
        run_pass(
            run_cfg,
            geom,
            (run, dc),
            tmp_path,
            pool=DiesAfterOneBlock,
            pixel_sum_trains=3,
        )
    path = tmp_path / FILE_NAME
    # window 0 still waits for block 1, which never came back
    assert not window_table(path).written.values.any()
    # whether block 0's result was written before the pool broke depends on the
    # order as_completed yields finished futures in, which is a set's
    with h5py.File(tmp_path / "out.h5", "r") as frames:
        unprocessed = int(
            (frames["frames/status"][:] == FrameStatus.NOT_PROCESSED).sum()
        )
    assert unprocessed in (12, 18)

    seen, integrated = [], []
    real_process, real_integrate = worker.process_block, worker.integrate_frame
    monkeypatch.setattr(
        worker, "process_block", lambda b: seen.append(b.index) or real_process(b)
    )
    monkeypatch.setattr(
        worker,
        "integrate_frame",
        lambda *a: integrated.append(1) or real_integrate(*a),
    )
    run_pass(run_cfg, geom, (run, dc), tmp_path, pixel_sum_trains=3)

    assert sorted(seen) == [0, 1, 2]  # every block, for the windows
    assert len(integrated) == unprocessed  # but only unwritten frames integrated
    with h5py.File(tmp_path / "out.h5", "r") as frames:
        assert (frames["frames/status"][:] == FrameStatus.OK).all()
    clean = run_pass(run_cfg, geom, (run, dc), tmp_path / "clean", pixel_sum_trains=3)
    for name, values in datasets(clean).items():
        np.testing.assert_array_equal(datasets(path)[name], values, name)
    for index in (0, 1):
        sums = window_sums(path, [index])
        assert_same_sums(sums, reference(run, sums.train_id.values))


def test_a_complete_frame_table_gets_its_windows_without_reintegration(
    run_cfg, geom, mock_run_factory, reference, tmp_path, monkeypatch, h5_digest
):
    """The case of every run integrated before the window sums existed."""
    run, dc = mock_run_factory()
    run_pass(run_cfg, geom, (run, dc), tmp_path, pixel_sum_trains=None)
    before = frame_table_digest(tmp_path / "out.h5", h5_digest)

    def forbidden(*args):
        raise AssertionError("a frame the table already holds must not be integrated")

    monkeypatch.setattr(worker, "integrate_frame", forbidden)
    path = run_pass(run_cfg, geom, (run, dc), tmp_path, pixel_sum_trains=3)

    assert frame_table_digest(tmp_path / "out.h5", h5_digest) == before
    for index in (0, 1):
        sums = window_sums(path, [index])
        assert_same_sums(sums, reference(run, sums.train_id.values))


def test_reintegrating_a_block_leaves_its_written_windows_alone(
    run_cfg, geom, mock_run_factory, tmp_path
):
    run_dc = mock_run_factory()
    path = run_pass(run_cfg, geom, run_dc, tmp_path, pixel_sum_trains=3)
    before = datasets(path)
    with h5py.File(tmp_path / "out.h5", "r+") as frames:
        frames["frames/status"][0:6] = FrameStatus.NOT_PROCESSED  # block 0

    run_pass(run_cfg, geom, run_dc, tmp_path, pixel_sum_trains=3)
    for name, values in before.items():
        np.testing.assert_array_equal(datasets(path)[name], values, name)


def test_a_rerun_with_everything_written_does_no_work(
    run_cfg, geom, mock_run_factory, tmp_path, monkeypatch
):
    run_dc = mock_run_factory()
    run_pass(run_cfg, geom, run_dc, tmp_path, pixel_sum_trains=3)

    def forbidden(block):
        raise AssertionError("nothing is left to read")

    monkeypatch.setattr(worker, "process_block", forbidden)
    run_pass(run_cfg, geom, run_dc, tmp_path, pixel_sum_trains=3)


def test_a_killed_pool_before_any_block_writes_no_window(
    run_cfg, geom, mock_run_factory, tmp_path
):
    run_dc = mock_run_factory()
    with pytest.raises(BrokenExecutor):
        run_pass(run_cfg, geom, run_dc, tmp_path, pool=BrokenPool, pixel_sum_trains=3)
    assert not window_table(tmp_path / FILE_NAME).written.values.any()


# ── the sums-only mode ────────────────────────────────────────────────────────
def test_the_sums_only_mode_writes_the_same_windows(
    run_cfg, geom, mock_run_factory, tmp_path, monkeypatch
):
    run, dc = mock_run_factory(**AWKWARD)
    full = run_pass(run_cfg, geom, (run, dc), tmp_path / "full", pixel_sum_trains=3)

    def forbidden(*args):
        raise AssertionError("the sums-only mode integrates nothing")

    monkeypatch.setattr(worker, "integrate_frame", forbidden)
    alone = tmp_path / "alone" / FILE_NAME
    cfg = replace(run_cfg, pixel_sum_trains=3)
    written = run_pixel_sums(
        cfg, dc=dc, run_dir=run.path, pixel_sums_path=alone, pool_factory=InlinePool
    )

    assert written == alone
    assert sorted(p.name for p in alone.parent.iterdir()) == [FILE_NAME]
    for name, values in datasets(full).items():
        np.testing.assert_array_equal(datasets(alone)[name], values, name)


def test_the_sums_only_mode_resumes(run_cfg, mock_run_factory, tmp_path, monkeypatch):
    run, dc = mock_run_factory()
    cfg = replace(run_cfg, pixel_sum_trains=3)
    kwargs = {
        "dc": dc,
        "run_dir": run.path,
        "pixel_sums_path": tmp_path / FILE_NAME,
    }
    with pytest.raises(BrokenExecutor):
        run_pixel_sums(cfg, pool_factory=DiesAfterOneBlock, **kwargs)

    seen = []
    real = worker.process_block
    monkeypatch.setattr(
        worker, "process_block", lambda b: seen.append(b.index) or real(b)
    )
    run_pixel_sums(cfg, pool_factory=InlinePool, **kwargs)
    assert sorted(seen) == [0, 1, 2]
    assert window_table(tmp_path / FILE_NAME).written.values.all()


def test_the_sums_only_mode_needs_windows(run_cfg):
    with pytest.raises(ValueError, match="no windows"):
        run_pixel_sums(replace(run_cfg, pixel_sum_trains=None))


# ── hashes and refusal ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("changes", "moves"),
    [
        ({"pixel_sum_trains": 50}, True),
        ({"run": 424}, True),
        ({"proposal": 1}, True),
        ({"npt": 2000}, False),
        ({"sdd_m": 7.5}, False),
        ({"beam_center_px": 600.0, "beam_center_py": 650.0}, False),
        ({"photon_energy_kev": 9.0}, False),
        ({"mask_bits": 0xFF}, False),
        ({"use_asic_seams": False}, False),
        ({"pixel_mask_file": "/elsewhere.npy"}, False),
        ({"output_root": "/tmp/a"}, False),
        ({"pixel_sums_root": "/tmp/b"}, False),
        ({"trains_per_block": 16}, False),
        ({"overwrite": True}, False),
    ],
)
def test_the_window_sums_hash_covers_only_what_changes_a_count(run_cfg, changes, moves):
    changed = replace(run_cfg, **changes)
    before = pixel_sums_hash(run_cfg, "MID_DET_AGIPD1M-1")
    assert (pixel_sums_hash(changed, "MID_DET_AGIPD1M-1") != before) is moves


def test_the_hash_takes_the_detector_name_as_resolved(run_cfg):
    """``None`` and the name it detects write the same file."""
    named = replace(run_cfg, detector_name="MID_DET_AGIPD1M-1")
    assert pixel_sums_hash(named, "MID_DET_AGIPD1M-1") == pixel_sums_hash(
        run_cfg, "MID_DET_AGIPD1M-1"
    )
    assert pixel_sums_hash(run_cfg, "X") != pixel_sums_hash(run_cfg, "Y")


def test_window_sums_are_never_replaced_even_with_overwrite(
    run_cfg, geom, mock_run_factory, tmp_path
):
    run_dc = mock_run_factory()
    path = run_pass(run_cfg, geom, run_dc, tmp_path, pixel_sum_trains=3)
    frames_before = datasets(tmp_path / "out.h5")

    with pytest.raises(ConfigHashMismatch, match="never replaces window sums"):
        run_pass(run_cfg, geom, run_dc, tmp_path, pixel_sum_trains=2, overwrite=True)
    assert window_table(path).attrs["window_trains"] == 3
    # refused before overwrite could recreate the frame table
    for name, values in frames_before.items():
        np.testing.assert_array_equal(datasets(tmp_path / "out.h5")[name], values)


# ── config ────────────────────────────────────────────────────────────────────
def test_window_sums_need_every_module(run_cfg):
    with pytest.raises(ValueError, match="all 16 modules"):
        replace(run_cfg, min_modules=8)
    assert replace(run_cfg, min_modules=8, pixel_sum_trains=None).min_modules == 8


@pytest.mark.parametrize("value", [0, -3])
def test_a_window_holds_at_least_one_train(run_cfg, value):
    with pytest.raises(ValueError, match="pixel_sum_trains"):
        replace(run_cfg, pixel_sum_trains=value)


def test_the_default_root_is_not_scratch():
    cfg = AgipdSaxsConfig(proposal=10400, run=423)
    assert cfg.pixel_sum_trains == 10
    assert "/scratch/" not in str(cfg.pixel_sums_file)
    assert str(cfg.pixel_sums_file).endswith(
        "usr/cached_files/agipd_pixel_sums/r0423/agipd_pixel_sums.h5"
    )


# ── the reader ────────────────────────────────────────────────────────────────
@pytest.fixture
def written(run_cfg, geom, mock_run_factory, tmp_path):
    run, dc = mock_run_factory(**AWKWARD)
    return run, run_pass(run_cfg, geom, (run, dc), tmp_path, pixel_sum_trains=3)


def test_the_reader_sums_several_windows_like_the_cache(written, reference):
    run, path = written
    ours = detector_sums(423, [10000, 10001, 10003, 10005], path=path)
    assert_same_sums(ours, reference(run, [10000, 10001, 10003, 10005]))
    np.testing.assert_array_equal(ours.attrs["windows"], [0, 1])
    by_window = detector_sums(423, windows=[1, 0], path=path)
    np.testing.assert_array_equal(by_window.counts, ours.counts)


def test_the_reader_finds_the_file_under_its_root(written, tmp_path):
    _, path = written
    root = tmp_path / "root"
    (root / "r0007").mkdir(parents=True)
    (root / "r0007" / FILE_NAME).write_bytes(path.read_bytes())
    sums = detector_sums(7, windows=[2], root=root)
    np.testing.assert_array_equal(sums.train_id, [10007])


@pytest.mark.parametrize(
    ("train_ids", "message"),
    [
        ([10000], "not a union of whole windows.*0 \\[10000, 10003\\)"),
        ([10002], "not in this run"),
        ([10004, 10003, 10005], "not summed.*MISSING_MODULES"),
    ],
)
def test_the_reader_never_sums_other_trains_than_asked(written, train_ids, message):
    _, path = written
    with pytest.raises(ValueError, match=message):
        detector_sums(423, train_ids, path=path)


def test_the_reader_wants_one_selection(written):
    _, path = written
    with pytest.raises(ValueError, match="exactly one"):
        detector_sums(423, path=path)
    with pytest.raises(ValueError, match="exactly one"):
        detector_sums(423, [10000], windows=[0], path=path)
    with pytest.raises(ValueError, match="do not exist"):
        detector_sums(423, windows=[9], path=path)


def test_the_reader_refuses_an_unwritten_window(
    run_cfg, geom, mock_run_factory, tmp_path
):
    run_dc = mock_run_factory()
    with pytest.raises(BrokenExecutor):
        run_pass(
            run_cfg, geom, run_dc, tmp_path, pool=DiesAfterOneBlock, pixel_sum_trains=3
        )
    with pytest.raises(ValueError, match="not written yet"):
        detector_sums(423, windows=[0], path=tmp_path / FILE_NAME)


# ── the kernel and the window spec ────────────────────────────────────────────
def test_the_kernel_is_the_caches_expression():
    """``analysis.cache._sum_job`` per module: where(mask == 0, data, 0) summed."""
    rng = np.random.default_rng(7)
    data = rng.integers(0, 4, (16, 5, 512, 128)).astype(np.int16)
    mask = np.where(rng.random(data.shape) < 0.05, 1 << 12, 0).astype(np.uint32)
    mask[3, :, :8] |= 1  # masked pixels that carry counts must not be summed

    counts = np.zeros((16, 512, 128), np.int64)
    valid = np.zeros_like(counts)
    accumulate_train(counts, valid, data, mask)
    accumulate_train(counts, valid, data, mask)

    ok = mask == 0
    np.testing.assert_array_equal(
        counts, 2 * np.where(ok, data, 0).sum(axis=1, dtype=np.int64)
    )
    np.testing.assert_array_equal(valid, 2 * ok.sum(axis=1, dtype=np.int64))


def test_a_train_is_summed_only_as_integer_counts_none_negative():
    assert train_data_ok(np.zeros((16, 2, 4, 4), np.int16))
    assert train_data_ok(np.zeros((16, 0, 4, 4), np.int16))
    assert not train_data_ok(np.zeros((16, 2, 4, 4), np.float32))
    negative = np.zeros((16, 2, 4, 4), np.int16)
    negative[5, 1, 2, 3] = -1
    assert not train_data_ok(negative)


def test_the_window_spec_is_anchored_at_the_first_train(mock_pipeline):
    spec = WindowSpec.for_plan(mock_pipeline.plan, 4)
    assert (spec.origin, spec.length, spec.count) == (10000, 4, 2)
    assert [spec.index_of(t) for t in (10000, 10003, 10004, 10005)] == [0, 0, 1, 1]
    np.testing.assert_array_equal(spec.starts(), [10000, 10004])
    with pytest.raises(ValueError, match="outside every window"):
        spec.index_of(10008)
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert isinstance(spec.__getstate__(), dict)  # by name: CLAUDE.md pitfall 20
