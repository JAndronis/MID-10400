"""End-to-end orchestration for one detector.

One test runs a genuine spawned pool, because that is the only way to find out
whether the initializer's arguments actually pickle. The rest inject a pool
(:mod:`waxs_harness`).
"""

from __future__ import annotations

import dataclasses
import json
import os

import h5py
import numpy as np
import pytest

pytest.importorskip("extra_data")

from waxs_harness import BrokenPool, FailingPool, InlinePool, run  # noqa: E402
from waxs_mockrun import CELL_ORDER, N_MODULE_PX, lit_rate  # noqa: E402

from analysis.common.status import FrameStatus  # noqa: E402
from analysis.common.writer import IncompleteRun  # noqa: E402
from analysis.waxs.cells import UnexpectedLitCells  # noqa: E402
from analysis.waxs.config import EXPECTED_LIT_CELLS  # noqa: E402
from analysis.waxs.run import run_jungfrau_waxs  # noqa: E402
from analysis.waxs.selftest import SelfTestFailed  # noqa: E402


def test_a_whole_run_integrates_and_returns_the_grid(finished):
    done = finished()
    grid, mock, cfg = done.grid, done.mock, done.cfg

    assert grid.dims == ("trainId", "cellId", "q")
    assert grid.shape == (len(mock.train_ids), mock.rows_per_train, cfg.npt)
    assert (grid["n_frames"] == 1).all()
    with h5py.File(done.path) as handle:
        assert (handle["frames/status"][:] == FrameStatus.OK).all()


def test_each_row_holds_the_frame_its_memory_cell_labels(finished):
    """The cell id comes from ``data.memoryCell``, never from array position.

    The mock stores cell 15 at position 0, as the real detectors do, and gives
    every lit cell its own photon rate. A frame read from the position equal to
    its cell id lands on the wrong cell and carries that cell's neighbour's rate.
    """
    assert CELL_ORDER[0] == 15
    done = finished()
    mock = done.mock
    with h5py.File(done.path) as handle:
        frames = handle["frames"]
        cells = frames["cellId"][:]
        photons = frames["n_photons"][:].astype(np.float64)
        kept = N_MODULE_PX - frames["n_bad_pixels"][:].astype(np.float64)
    rate = photons / kept
    for cell in mock.lit_cells:
        mine = cells == cell
        assert mine.sum() == len(mock.detector_trains), cell
        assert rate[mine] == pytest.approx(lit_rate(cell), rel=0.01), cell


def test_a_fully_masked_frame_is_refused_counted_and_tolerated(cfg, finished, tmp_path):
    """A frame with no pixel left is ``DATA_CHECK_FAILED``, never a row of zeros.

    It is a property of the data, so by default the run finishes and records it.
    ``allow_data_check_failures=False`` makes the same frame stop the run.
    """
    done = finished(train_ids=(10000, 10001), dead_frame_train=10001)
    mock, dc, path, grid = done.mock, done.dc, done.path, done.grid

    dead_cell = mock.lit_cells[0]
    with h5py.File(path) as handle:
        frames = handle["frames"]
        status = frames["status"][:]
        refused = status == FrameStatus.DATA_CHECK_FAILED
        assert refused.sum() == 1
        assert (status[~refused] == FrameStatus.OK).all()
        assert frames["trainId"][:][refused][0] == 10001
        assert frames["cellId"][:][refused][0] == dead_cell
        assert frames["normalization"][:][refused].sum() == 0
        assert frames["n_bad_pixels"][:][refused][0] == N_MODULE_PX
        recorded = json.loads(handle["provenance"].attrs["data_check"])
    assert recorded["n_frames_refused"] == 1
    assert grid.sel(trainId=10001, cellId=dead_cell)["n_frames"] == 0
    assert json.loads(grid.attrs["unplaced"]) == {"DATA_CHECK_FAILED": 1}

    strict = dataclasses.replace(cfg, allow_data_check_failures=False)
    with pytest.raises(IncompleteRun, match="DATA_CHECK_FAILED"):
        run(strict, mock, dc, output_path=tmp_path / "strict.h5")


def test_provenance_records_what_was_decided(finished):
    done = finished()
    cfg, mock = done.cfg, done.mock

    with h5py.File(done.path) as handle:
        attrs = handle["provenance"].attrs
        assert attrs["detector"] == "jf1"
        error_model = json.loads(attrs["error_model"])
        assert error_model["source"] == "measured"
        assert error_model["read_noise_kev"] == pytest.approx(0.32, rel=0.05)
        assert error_model["variance"] == "poisson"
        assert "diagnostic" in error_model["note"]
        assert attrs["units"] == "photons"
        counting = json.loads(attrs["photon_counting"])
        assert counting["energy_kev"] == cfg.count_energy_kev
        assert counting["level_pixels"] == cfg.level_pixels
        assert "rint" in counting["method"]
        levels = json.loads(attrs["photon_levels"])
        assert levels["n_frames_with_a_missing_level"] == 0
        assert len(levels["median_kev"]) == 8
        cells = json.loads(attrs["cells"])
        assert cells["lit"] == list(mock.lit_cells)
        selftest = json.loads(attrs["selftest"])
        assert selftest["n_frames"] == cfg.selftest_frames
        assert (
            max(
                selftest["max_rel_signal"],
                selftest["max_rel_normalization"],
                selftest["max_rel_variance"],
            )
            <= selftest["tolerance"]
        )
        assert float(attrs["wall_s"]) > 0
        assert json.loads(attrs["package_versions"])["pyFAI"]


def test_a_real_spawned_pool_works(cfg, mock_run_factory, tmp_path):
    """The only test that finds out whether the initargs actually pickle."""
    mock, dc = mock_run_factory()
    grid = run_jungfrau_waxs(
        dataclasses.replace(cfg, n_workers=2),
        dc=dc,
        run_dir=mock.path,
        output_path=tmp_path / "spawn.h5",
    )
    assert (grid["n_frames"] == 1).all()


def test_an_extreme_pixel_no_longer_takes_the_run_down(
    cfg, mock_run_factory, tmp_path, caplog
):
    """An extreme pixel costs that pixel, and the run completes.

    Such pixels (salt Bragg spots from an evaporating droplet) become more
    common through a run, so failing their frames would lose the frames the run
    exists to record. The pixel goes, the frame stays, and the run records what
    it cost.
    """
    import logging

    mock, dc = mock_run_factory(extreme_train=10002)
    with caplog.at_level(logging.WARNING, logger="analysis.waxs.run"):
        grid = run(cfg, mock, dc, output_path=tmp_path / "extreme.h5")

    assert grid is not None
    with h5py.File(tmp_path / "extreme.h5") as handle:
        assert (handle["frames/status"][:] == FrameStatus.OK).all()
        assert (handle["frames/n_extreme_pixels"][:] > 0).sum() == 1
        recorded = json.loads(handle["provenance"].attrs["data_check"])
    assert recorded["n_pixels_excluded"] == 1
    assert recorded["n_frames_with_excluded_pixels"] == 1
    assert recorded["n_frames_refused"] == 0

    # Every frame is placed, and the cost is on the record three times over.
    assert (grid["n_frames"] == 1).all()
    assert "the value check excluded" in caplog.text
    assert json.loads(grid.attrs["data_check"])["n_pixels_excluded"] == 1
    assert json.loads(grid.attrs["unplaced"]) == {}


def test_a_label_mismatch_still_takes_the_run_down(cfg, mock_run_factory, tmp_path):
    """Only the value check is tolerated: a train that cannot be addressed by
    label is not a property of the data the pass can record and move past."""
    mock, dc = mock_run_factory(train_ids=(10000, 10001), shuffled_cells_train=10001)
    with pytest.raises(IncompleteRun, match="LABEL_MISMATCH"):
        run(cfg, mock, dc, output_path=tmp_path / "label.h5")
    # A frame that could not be placed is not one the value check refused.
    with h5py.File(tmp_path / "label.h5") as handle:
        assert json.loads(handle["provenance"].attrs["data_check"]) == {}


def test_a_train_whose_own_rows_name_another_train_is_a_label_mismatch(
    cfg, mock_run_factory, tmp_path
):
    """INDEX files the entry under one train, its ``data.trainId`` says another.

    The reader's train coordinate comes from INDEX, so only the entry's own
    ``trainId`` row can tell the frames belong elsewhere.
    """
    mock, dc = mock_run_factory(train_ids=(10000, 10001), mislabel_train=10001)
    path = tmp_path / "mislabel.h5"
    with pytest.raises(IncompleteRun, match="LABEL_MISMATCH"):
        run(cfg, mock, dc, output_path=path)
    with h5py.File(path) as handle:
        status = handle["frames/status"][:]
        trains = handle["trains/trainId"][:]
        first, count = handle["trains/first"][:], handle["trains/count"][:]
    for train_id, start, n in zip(trains, first, count, strict=True):
        want = FrameStatus.LABEL_MISMATCH if train_id == 10001 else FrameStatus.OK
        assert (status[start : start + n] == want).all(), train_id


def test_the_cell_classification_reads_no_train_whose_own_rows_name_another(
    cfg, mock_run_factory, tmp_path
):
    """A misfiled entry is left out of the cell sample, not counted as the train."""
    mock, dc = mock_run_factory(train_ids=(10000, 10001), mislabel_train=10001)
    path = tmp_path / "mislabel.h5"
    with pytest.raises(IncompleteRun):
        run(cfg, mock, dc, output_path=path)
    with h5py.File(path) as handle:
        assert handle["cells/sampled_trains"][:].tolist() == [10000]


def test_the_selftest_gates_on_no_train_whose_own_rows_name_another(
    cfg, mock_run_factory, tmp_path, monkeypatch
):
    """The self-test's frames all come from a train whose labels hold."""
    from extra_data import RunDirectory, by_id

    from analysis.waxs import run as run_module
    from analysis.waxs.plan import open_detector

    mock, dc = mock_run_factory(train_ids=(10000, 10001), mislabel_train=10000)
    fed = []
    real = run_module.run_selftest

    def spy(ai, op, counting, frames, **kwargs):
        fed.extend(frames)
        return real(ai, op, counting, frames, **kwargs)

    monkeypatch.setattr(run_module, "run_selftest", spy)
    with pytest.raises(IncompleteRun):
        run(cfg, mock, dc, output_path=tmp_path / "mislabel.h5")

    det = open_detector(cfg, RunDirectory(str(mock.path)))
    good = det.select_trains(by_id[[10001]])["data.adc"].ndarray()[0, 0]
    assert fed
    for data, _ in fed:
        assert any(np.array_equal(data, frame) for frame in good)


def test_a_broken_pool_marks_everything_unprocessed_and_raises(
    cfg, mock_run_factory, tmp_path
):
    from concurrent.futures import BrokenExecutor

    mock, dc = mock_run_factory()
    with pytest.raises(BrokenExecutor):
        run(cfg, mock, dc, output_path=tmp_path / "broken.h5", pool_factory=BrokenPool)

    with h5py.File(tmp_path / "broken.h5") as handle:
        assert (handle["frames/status"][:] == FrameStatus.NOT_PROCESSED).all()


def test_resume_processes_only_the_missing_blocks(
    cfg, mock_run_factory, tmp_path, monkeypatch
):
    """A rerun integrates only the blocks the ledger lacks, in a pinned env.

    The first attempt writes block 0 and loses every later block to a worker
    error. A spawned child reads the thread environment at start, so the pool
    must also find it set before it is built.
    """
    from analysis.common.cpu import THREAD_ENV

    mock, dc = mock_run_factory()
    path = tmp_path / "resume.h5"

    class FirstOnly(InlinePool):
        def submit(self, fn, *args):
            if args[0].index == 0:
                return super().submit(fn, *args)
            return FailingPool.submit(self, fn, *args)

    with pytest.raises(IncompleteRun, match="WORKER_ERROR"):
        run(cfg, mock, dc, output_path=path, pool_factory=FirstOnly)

    for name in THREAD_ENV:
        monkeypatch.delenv(name, raising=False)
    submitted, env = [], {}

    class Counting(InlinePool):
        def __init__(self, *args, **kwargs):
            env.update({name: os.environ.get(name) for name in THREAD_ENV})
            super().__init__(*args, **kwargs)

        def submit(self, fn, *args):
            submitted.append(args[0].index)
            return super().submit(fn, *args)

    grid = run(cfg, mock, dc, output_path=path, pool_factory=Counting)
    n_blocks = -(-len(mock.detector_trains) // cfg.trains_per_block)
    assert sorted(submitted) == list(range(1, n_blocks))
    assert env == dict.fromkeys(THREAD_ENV, "1")
    assert (grid["n_frames"] == 1).all()

    submitted.clear()
    run(cfg, mock, dc, output_path=path, pool_factory=Counting)
    assert submitted == []  # a complete run does no work


def test_a_pinned_lit_set_that_the_data_contradicts_stops_the_run(
    cfg, finished, tmp_path
):
    """Nothing is written when the pinned lit-cell pattern is wrong.

    The pass measures the set per run by default, because the proposal used
    several readout patterns. ``(0, 1, 2, 3)`` is a pattern a JUNGFRAU can
    read, so only the pin catches it.
    """
    done = finished(lit_cells=(0, 1, 2, 3))
    pinned = dataclasses.replace(cfg, expected_lit_cells=EXPECTED_LIT_CELLS)
    with pytest.raises(UnexpectedLitCells, match="are lit"):
        run(pinned, done.mock, done.dc, output_path=tmp_path / "lit.h5")
    assert not (tmp_path / "lit.h5").exists()

    # ...and unpinned the same run is integrated without complaint, because a
    # run that really does read four cells is not a broken run.
    assert done.grid.shape[1] == 4


def test_a_run_with_no_lit_cell_returns_nothing_and_says_why(
    cfg, mock_run_factory, tmp_path, caplog
):
    """A run that saw no beam is a result, not a failure.

    Raising here would make a bulk DAMNIT reprocess fall over on dark runs, so
    the pass logs the evidence and returns ``None``.
    """
    import logging

    mock, dc = mock_run_factory(train_ids=(10000, 10001), lit_cells=())
    with caplog.at_level(logging.WARNING, logger="analysis.waxs.run"):
        assert run(cfg, mock, dc, output_path=tmp_path / "dark.h5") is None

    assert not (tmp_path / "dark.h5").exists()
    # The evidence, not just the verdict: which cell came closest, and to what.
    assert "no memory cell is lit" in caplog.text
    assert "lit fraction" in caplog.text


def test_a_run_with_no_dark_cell_falls_back_to_the_configured_noise(
    cfg, mock_run_factory, tmp_path
):
    """A run that reads all 16 cells leaves no dark cell to measure noise on.

    Raising there would take a whole DAMNIT reprocess down on ordinary runs, so
    sigma_read falls back to the per-detector median and ``ErrorModel.source``
    records that it did.
    """
    mock, dc = mock_run_factory(train_ids=(10000, 10001), lit_cells=tuple(range(16)))
    everything = dataclasses.replace(cfg, expected_lit_cells=tuple(range(16)))

    grid = run(everything, mock, dc, output_path=tmp_path / "fallback.h5")
    assert grid.shape[1] == 16
    with h5py.File(tmp_path / "fallback.h5") as handle:
        model = json.loads(handle["provenance"].attrs["error_model"])
    assert model["source"] == "fallback"
    assert model["read_noise_kev"] == pytest.approx(cfg.read_noise_fallback_kev)


def test_allow_incomplete_returns_instead_of_raising(cfg, mock_run_factory, tmp_path):
    mock, dc = mock_run_factory()
    grid = run(
        dataclasses.replace(cfg, allow_incomplete=True),
        mock,
        dc,
        output_path=tmp_path / "partial.h5",
        pool_factory=FailingPool,
    )
    assert (grid["n_frames"] == 0).all()
    assert (grid.values == 0).all()


def test_the_selftest_gate_runs_before_the_pool(
    cfg, mock_run_factory, tmp_path, monkeypatch
):
    import analysis.waxs.run as run_module

    def boom(*args, **kwargs):
        raise SelfTestFailed("deliberate")

    monkeypatch.setattr(run_module, "run_selftest", boom)
    mock, dc = mock_run_factory()
    pooled = []

    class Watching(InlinePool):
        def __init__(self, *args, **kwargs):
            pooled.append(1)
            super().__init__(*args, **kwargs)

    with pytest.raises(SelfTestFailed):
        run(cfg, mock, dc, output_path=tmp_path / "gate.h5", pool_factory=Watching)
    assert pooled == []
    assert not (tmp_path / "gate.h5").exists()


def test_both_detectors_run_independently_into_their_own_files(finished):
    """Two detectors, two passes, two files, no crosstalk.

    The jf2 path differs in more than a name. Its source is
    ``MID_EXP_JF500K2/CORR/JNGFR02:daqOutput``, so it exercises ``first_modno``
    and its own PONI and mask.
    """
    outputs = {}
    for detector in ("jf1", "jf2"):
        done = finished(detector)
        assert (done.grid["n_frames"] == 1).all()
        assert done.path.exists()
        outputs[detector] = (done.cfg, done.grid)

    jf1_cfg, jf1_grid = outputs["jf1"]
    jf2_cfg, jf2_grid = outputs["jf2"]
    assert jf1_cfg.output_file != jf2_cfg.output_file
    assert jf1_cfg.config_hash() != jf2_cfg.config_hash()

    with h5py.File(jf2_cfg.output_file) as handle:
        attrs = handle["provenance"].attrs
        assert attrs["detector"] == "jf2"
        assert handle["provenance"].attrs["detector_name"] == "MID_EXP_JF500K2"

    # Different geometries, so the two q axes must not be identical.
    assert not np.array_equal(jf1_grid["q"].values, jf2_grid["q"].values)
