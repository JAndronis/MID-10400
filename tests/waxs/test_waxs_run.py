"""End-to-end orchestration for one detector (W3).

The pool is injected rather than real wherever the test is about a failure
path: killing processes to exercise ``BrokenExecutor`` would put test hooks in
``worker``. One test does run a genuine spawned pool, because that is the only
way to find out whether the initializer's arguments actually pickle.
"""

from __future__ import annotations

import dataclasses
import json
import os

import h5py
import numpy as np
import pytest

pytest.importorskip("extra_data")

from analysis.common.status import FrameStatus  # noqa: E402
from analysis.common.writer import IncompleteRun  # noqa: E402
from analysis.waxs.cells import UnexpectedLitCells  # noqa: E402
from analysis.waxs.config import EXPECTED_LIT_CELLS  # noqa: E402
from analysis.waxs.run import run_jungfrau_waxs  # noqa: E402
from analysis.waxs.selftest import SelfTestFailed  # noqa: E402


class InlinePool:
    """Runs every block in this process, with the real initializer."""

    def __init__(self, n_workers, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        from analysis.waxs import worker

        worker._STATE = None

    def submit(self, fn, *args):
        from concurrent.futures import Future

        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as error:  # noqa: BLE001 - handed to the caller
            future.set_exception(error)
        return future


class FailingPool(InlinePool):
    """Every block raises, so the ledger takes ``WORKER_ERROR``."""

    def submit(self, fn, *args):
        from concurrent.futures import Future

        future = Future()
        future.set_exception(RuntimeError("block exploded"))
        return future


class BrokenPool(InlinePool):
    """The executor dies, as a killed worker makes it."""

    def submit(self, fn, *args):
        from concurrent.futures import BrokenExecutor, Future

        future = Future()
        future.set_exception(BrokenExecutor("pool died"))
        return future


def run(cfg, mock, dc, **kwargs):
    return run_jungfrau_waxs(
        cfg,
        dc=dc,
        run_dir=mock.path,
        output_path=kwargs.pop("output_path", None),
        pool_factory=kwargs.pop("pool_factory", InlinePool),
        **kwargs,
    )


def test_a_whole_run_integrates_and_returns_the_grid(cfg, mock_run_factory, tmp_path):
    mock, dc = mock_run_factory()
    grid = run(cfg, mock, dc, output_path=tmp_path / "run.h5")

    assert grid.dims == ("trainId", "cellId", "q")
    assert grid.shape == (len(mock.train_ids), mock.rows_per_train, cfg.npt)
    assert (grid["n_frames"] == 1).all()
    with h5py.File(tmp_path / "run.h5") as handle:
        assert (handle["frames/status"][:] == FrameStatus.OK).all()


def test_provenance_records_what_was_decided(cfg, mock_run_factory, tmp_path):
    import json

    mock, dc = mock_run_factory()
    run(cfg, mock, dc, output_path=tmp_path / "prov.h5")

    with h5py.File(tmp_path / "prov.h5") as handle:
        attrs = handle["provenance"].attrs
        assert attrs["detector"] == "jf1"
        error_model = json.loads(attrs["error_model"])
        assert error_model["source"] == "measured"
        assert error_model["read_noise_kev"] == pytest.approx(0.32, rel=0.05)
        assert "unclamped" in error_model["note"]
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


def test_the_thread_env_is_pinned_before_any_pool(
    cfg, mock_run_factory, tmp_path, monkeypatch
):
    """AGIPD §3 rule 3: a spawned child inherits it at process start."""
    from analysis.common.cpu import THREAD_ENV

    for name in THREAD_ENV:
        monkeypatch.delenv(name, raising=False)
    seen = {}

    class Recording(InlinePool):
        def __init__(self, *args, **kwargs):
            seen.update({name: os.environ.get(name) for name in THREAD_ENV})
            super().__init__(*args, **kwargs)

    mock, dc = mock_run_factory()
    run(cfg, mock, dc, output_path=tmp_path / "env.h5", pool_factory=Recording)
    assert seen == dict.fromkeys(THREAD_ENV, "1")


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
    """r0480: 473 of 23 992 frames failed the value check and the run raised.

    They were NaCl Bragg spots from the evaporating droplet, so the loss rose
    through the run — exactly the frames the run exists to record. Now the pixel
    goes and the frame stays, and the run says what it cost.
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
    mock, dc = mock_run_factory(shuffled_cells_train=10001)
    with pytest.raises(IncompleteRun, match="LABEL_MISMATCH"):
        run(cfg, mock, dc, output_path=tmp_path / "label.h5")


def test_which_statuses_block_the_run(cfg):
    """The split the gate rests on, without needing a run to produce each one.

    ``DATA_CHECK_FAILED`` now means a frame with nothing left to integrate — the
    value check drops pixels, not frames — and it is still a property of the
    data rather than a failure of the pass, so it stays tolerated by default.
    """
    from analysis.waxs.run import _blocking

    summary = {
        "OK": 100,
        "DATA_CHECK_FAILED": 3,
        "LABEL_MISMATCH": 2,
        "WORKER_ERROR": 1,
        "NOT_PROCESSED": 4,
    }
    assert _blocking(cfg, summary) == {
        "LABEL_MISMATCH": 2,
        "WORKER_ERROR": 1,
        "NOT_PROCESSED": 4,
    }
    strict = dataclasses.replace(cfg, allow_data_check_failures=False)
    assert _blocking(strict, summary)["DATA_CHECK_FAILED"] == 3
    assert _blocking(cfg, {"OK": 100}) == {}


def test_a_worker_exception_lands_in_the_ledger(cfg, mock_run_factory, tmp_path):
    mock, dc = mock_run_factory()
    with pytest.raises(IncompleteRun, match="WORKER_ERROR"):
        run(cfg, mock, dc, output_path=tmp_path / "err.h5", pool_factory=FailingPool)

    with h5py.File(tmp_path / "err.h5") as handle:
        assert (handle["frames/status"][:] == FrameStatus.WORKER_ERROR).all()
        assert "block exploded" in handle["provenance"].attrs["block_errors"]


def test_a_broken_pool_marks_everything_unprocessed_and_raises(
    cfg, mock_run_factory, tmp_path
):
    from concurrent.futures import BrokenExecutor

    mock, dc = mock_run_factory()
    with pytest.raises(BrokenExecutor):
        run(cfg, mock, dc, output_path=tmp_path / "broken.h5", pool_factory=BrokenPool)

    with h5py.File(tmp_path / "broken.h5") as handle:
        assert (handle["frames/status"][:] == FrameStatus.NOT_PROCESSED).all()


def test_resume_processes_only_the_missing_blocks(cfg, mock_run_factory, tmp_path):
    from concurrent.futures import BrokenExecutor

    mock, dc = mock_run_factory()
    path = tmp_path / "resume.h5"
    with pytest.raises(BrokenExecutor):
        run(cfg, mock, dc, output_path=path, pool_factory=BrokenPool)

    submitted = []

    class Counting(InlinePool):
        def submit(self, fn, *args):
            submitted.append(args[0].index)
            return super().submit(fn, *args)

    grid = run(cfg, mock, dc, output_path=path, pool_factory=Counting)
    assert len(submitted) == len(set(submitted))
    assert (grid["n_frames"] == 1).all()

    submitted.clear()
    run(cfg, mock, dc, output_path=path, pool_factory=Counting)
    assert submitted == []  # a complete run does no work


def test_a_pinned_lit_set_that_the_data_contradicts_stops_the_run(
    cfg, mock_run_factory, tmp_path
):
    """§3 D4, end to end: nothing is written when the pinned pattern is wrong.

    Pinned, because the pass measures the set per run by default — the
    proposal used four readout patterns. ``(0, 1, 2, 3)`` is a shape a
    JUNGFRAU can read, so only the pin catches it.
    """
    mock, dc = mock_run_factory(lit_cells=(0, 1, 2, 3))
    pinned = dataclasses.replace(cfg, expected_lit_cells=EXPECTED_LIT_CELLS)
    with pytest.raises(UnexpectedLitCells, match="are lit"):
        run(pinned, mock, dc, output_path=tmp_path / "lit.h5")
    assert not (tmp_path / "lit.h5").exists()

    # ...and unpinned the same run is integrated without complaint, because a
    # run that really does read four cells is not a broken run.
    grid = run(cfg, mock, dc, output_path=tmp_path / "unpinned.h5")
    assert grid.shape[1] == 4


def test_a_run_with_no_lit_cell_returns_nothing_and_says_why(
    cfg, mock_run_factory, tmp_path, caplog
):
    """43 of the proposal's runs saw no beam. That is a result, not a failure.

    Raising here would make a bulk DAMNIT reprocess fall over on runs that are
    simply dark, so the pass logs the evidence and returns ``None``.
    """
    import logging

    mock, dc = mock_run_factory(lit_cells=())
    with caplog.at_level(logging.WARNING, logger="analysis.waxs.run"):
        assert run(cfg, mock, dc, output_path=tmp_path / "dark.h5") is None

    assert not (tmp_path / "dark.h5").exists()
    # The evidence, not just the verdict: which cell came closest, and to what.
    assert "no memory cell is lit" in caplog.text
    assert "lit fraction" in caplog.text


def test_a_run_with_no_dark_cell_falls_back_to_the_configured_noise(
    cfg, mock_run_factory, tmp_path
):
    """213 of the proposal's runs read all 16 cells, so none is left to measure.

    Raising there would take a whole DAMNIT reprocess down on runs that are
    perfectly ordinary, so sigma_read falls back to the per-detector median and
    ``ErrorModel.source`` records that it did.
    """
    import json

    mock, dc = mock_run_factory(lit_cells=tuple(range(16)))
    everything = dataclasses.replace(cfg, expected_lit_cells=tuple(range(16)))

    grid = run(everything, mock, dc, output_path=tmp_path / "fallback.h5")
    assert grid.shape[1] == 16
    with h5py.File(tmp_path / "fallback.h5") as handle:
        model = json.loads(handle["provenance"].attrs["error_model"])
    assert model["source"] == "fallback"
    assert model["read_noise_kev"] == pytest.approx(cfg.read_noise_fallback_kev)

    # ...and the fallback cannot be cleared: ``None`` means "use the
    # per-detector default" and __post_init__ resolves it, so a constructed
    # config never holds None. Pinned because reading it the other way is what
    # made an earlier version of this test assert an unreachable branch.
    stripped = dataclasses.replace(everything, read_noise_fallback_kev=None)
    assert stripped.read_noise_fallback_kev == cfg.read_noise_fallback_kev

    # An explicit value still wins over both.
    configured = run(
        dataclasses.replace(everything, read_noise_kev=0.32),
        mock,
        dc,
        output_path=tmp_path / "noise2.h5",
    )
    assert configured.shape[1] == 16


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


def test_an_unknown_reducer_is_refused_before_any_work(cfg, mock_run_factory, tmp_path):
    mock, dc = mock_run_factory()
    with pytest.raises(ValueError, match="reduce must be one of"):
        run(cfg, mock, dc, output_path=tmp_path / "bad.h5", reduce="nonsense")
    assert not (tmp_path / "bad.h5").exists()


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


def test_the_pooled_reducer_is_reachable(cfg, mock_run_factory, tmp_path):
    mock, dc = mock_run_factory()
    pooled = run(cfg, mock, dc, output_path=tmp_path / "pooled.h5", reduce="pooled")
    assert set(pooled.data_vars) == {"intensity", "sigma", "n_frames"}
    assert np.isfinite(pooled["intensity"].values).all()


def test_both_detectors_run_independently_into_their_own_files(
    config_for_detector, mock_run_factory, tmp_path
):
    """§3 D2, end to end: two detectors, two passes, two files, no crosstalk.

    The jf2 path differs in more than a name — its source is
    ``MID_EXP_JF500K2/CORR/JNGFR02:daqOutput``, so it exercises ``first_modno``
    and its own PONI and mask.
    """
    from analysis.waxs.config import DETECTOR_MODNOS, DETECTOR_NAMES
    from analysis.waxs.run import run_jungfrau_waxs

    outputs = {}
    for detector in ("jf1", "jf2"):
        cfg = config_for_detector(detector, output_root=str(tmp_path / detector))
        mock, dc = mock_run_factory(
            detector_name=DETECTOR_NAMES[detector],
            module=DETECTOR_MODNOS[detector],
        )
        grid = run_jungfrau_waxs(
            cfg,
            dc=dc,
            run_dir=mock.path,
            pool_factory=InlinePool,
            reduce="per_cell",
        )
        assert (grid["n_frames"] == 1).all()
        assert cfg.output_file.exists()
        outputs[detector] = (cfg, grid)

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
