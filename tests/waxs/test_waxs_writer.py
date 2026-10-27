"""The worker, the output schema, the ledger and the reducers."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip("extra_data")

from waxs_mockrun import LIT_CELLS  # noqa: E402

from analysis.common.status import FrameStatus  # noqa: E402
from analysis.common.writer import (  # noqa: E402
    SchemaMismatch,
    pooled_per_train,
)
from analysis.waxs import worker  # noqa: E402
from analysis.waxs.writer import JungfrauWaxsWriter, per_cell  # noqa: E402


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


def test_a_repeated_memory_cell_is_a_label_mismatch(
    cfg, mock_run_factory, operator, model
):
    """A repeated cell id leaves rows unplaceable, and position is no substitute."""
    from analysis.waxs.plan import build_plan, open_detector

    run, dc = mock_run_factory(train_ids=(10000, 10001), shuffled_cells_train=10001)
    op, ai = operator
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)
    worker.init_from_detector(cfg, op, ai, LIT_CELLS, det)
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


def _written_with_an_extreme_pixel(cfg, mock_run_factory, operator, model, tmp_path):
    """A finished file whose one extreme pixel was excluded, plus its summary."""
    from analysis.waxs.plan import build_plan, open_detector

    _, dc = mock_run_factory(extreme_train=10002)
    op, ai = operator
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)
    worker.init_from_detector(cfg, op, ai, LIT_CELLS, det)
    path = tmp_path / "extreme.h5"
    try:
        with JungfrauWaxsWriter.open_or_create(cfg, plan, path) as out:
            out.store_operator(op)
            out.store_cells(
                SimpleNamespace(
                    cells=range(16),
                    lit_fraction=[0.0] * 16,
                    n_samples=[1] * 16,
                    lit=LIT_CELLS,
                    dark=(),
                ),
                model,
                np.array([10000], dtype=np.uint64),
            )
            for block in plan.blocks:
                out.write_block(block, worker.process_block(block))
            summary = out.data_check_summary()
    finally:
        worker._STATE = None
    return path, summary


def test_the_file_says_which_frame_lost_pixels_and_what_they_held(
    cfg, mock_run_factory, operator, model, tmp_path
):
    """The row is integrated, and still carries the evidence for the exclusion.

    ``max_kev`` is what survived into the sums and ``max_kev_static`` what was
    taken out, so the pair reads as "this bin is missing a 1.8e5 keV pixel"
    without anyone going back to proc. The run's summary counts the cost.
    """
    path, summary = _written_with_an_extreme_pixel(
        cfg, mock_run_factory, operator, model, tmp_path
    )
    with h5py.File(path) as handle:
        frames = handle["frames"]
        assert (frames["status"][:] == FrameStatus.OK).all()
        affected = frames["n_extreme_pixels"][:] > 0
        assert affected.sum() == 1
        assert frames["trainId"][:][affected][0] == 10002
        assert frames["max_kev_static"][:][affected][0] == pytest.approx(1.8e5)
        assert frames["max_kev"][:][affected][0] < 1e3
        # Integrated, not zeroed: losing a pixel does not lose the frame.
        assert frames["normalization"][:][affected].sum() > 0
    assert summary["n_frames_with_excluded_pixels"] == 1
    assert summary["n_pixels_excluded"] == 1
    assert summary["worst_frame_n_pixels"] == 1
    assert summary["worst_excluded_kev"] == pytest.approx(1.8e5)
    assert summary["n_frames_refused"] == 0


def test_a_file_missing_a_column_is_refused_rather_than_resumed(worker_ready, tmp_path):
    """A column added to the pass does not move ``config_hash``.

    So without the schema check an older file passes the hash gate, is opened
    for resume, and fails on the first write with the run part-processed.
    """
    path = written(worker_ready, tmp_path)
    with h5py.File(path, "r+") as handle:
        del handle["frames/n_extreme_pixels"]

    with pytest.raises(SchemaMismatch, match="n_extreme_pixels"):
        JungfrauWaxsWriter.open_or_create(worker_ready.cfg, worker_ready.plan, path)

    # overwrite=True is the documented way out, and it must still work.
    replaced = dataclasses.replace(worker_ready.cfg, overwrite=True)
    with JungfrauWaxsWriter.open_or_create(replaced, worker_ready.plan, path) as out:
        assert "n_extreme_pixels" in out._f["frames"]


# ── the output file ──────────────────────────────────────────────────────────


def test_there_are_no_nan_sentinels_anywhere(worker_ready, tmp_path):
    """``asic_level`` may hold NaN for an ASIC with no level, and only then."""
    path = written(worker_ready, tmp_path)
    with h5py.File(path) as handle:
        for name in (
            "signal",
            "normalization",
            "variance",
            "energy_valid",
            "max_kev",
            "count_energy_kev",
            "asic_level",
        ):
            values = handle[f"frames/{name}"][:]
            assert np.isfinite(values).all(), name
        assert (handle["frames/n_level_missing"][:] == 0).all()


def test_the_file_counts_photons_and_says_so(worker_ready, tmp_path):
    from waxs_mockrun import PHOTON_KEV

    from analysis.waxs.writer import PHOTONS, file_units

    path = written(worker_ready, tmp_path)
    assert file_units(path) == PHOTONS
    with h5py.File(path) as handle:
        frames = handle["frames"]
        ok = frames["status"][:] == FrameStatus.OK
        photons = frames["n_photons"][:][ok]
        energy = frames["energy_valid"][:][ok]
        assert frames["count_energy_kev"][:][ok] == pytest.approx(PHOTON_KEV)
        # Raw keV over photons: E plus the zero level spread over every pixel.
        assert (energy / photons).mean() == pytest.approx(PHOTON_KEV, rel=0.05)
        assert (frames["variance"][:][ok] >= 0).all()
    assert per_cell(path).attrs["units"] == PHOTONS
    assert pooled_per_train(path).attrs["units"] == PHOTONS


def test_each_frames_zero_level_is_the_one_put_in(worker_ready, tmp_path):
    """End to end: the mock's per-train common mode and lit shift come back.

    The level is the frame's own, not a dark cell's, because lit cells sit
    higher than dark ones in the same train.
    """
    from waxs_mockrun import injected_level

    path = written(worker_ready, tmp_path)
    run = worker_ready.run
    with h5py.File(path) as handle:
        train_ids = handle["frames/trainId"][:]
        levels = handle["frames/asic_level"][:]
        status = handle["frames/status"][:]
    assert (status == FrameStatus.OK).all()
    for row, train_id in enumerate(train_ids.tolist()):
        want = injected_level(run, train_id, lit=True)
        assert np.abs(levels[row] - want).max() < 0.01, (row, train_id)


def test_a_kev_file_is_never_resumed_as_photons(worker_ready, tmp_path):
    """A file without ``units`` predates photon counting and counts keV."""
    path = written(worker_ready, tmp_path)
    with h5py.File(path, "r+") as handle:
        del handle["provenance"].attrs["units"]
    with pytest.raises(SchemaMismatch, match="no units attribute"):
        JungfrauWaxsWriter.open_or_create(worker_ready.cfg, worker_ready.plan, path)


# ── the reducers ─────────────────────────────────────────────────────────────
def test_per_cell_places_frames_by_label(worker_ready, tmp_path):
    """The grid is the same whatever order the file's rows are in."""
    import shutil

    path = written(worker_ready, tmp_path)
    shuffled = tmp_path / "shuffled.h5"
    shutil.copy2(path, shuffled)
    with h5py.File(shuffled, "r+") as handle:
        frames = handle["frames"]
        order = np.random.default_rng(0).permutation(frames["status"].shape[0])
        for name in frames:
            frames[name][...] = frames[name][...][order]

    grid = per_cell(path)
    assert list(grid.trainId.values) == list(worker_ready.run.train_ids)
    assert list(grid.cellId.values) == sorted(worker_ready.classification.lit)
    assert (grid["n_frames"] == 1).all()
    assert json.loads(grid.attrs["unplaced"]) == {}
    assert np.array_equal(per_cell(shuffled).values, grid.values)


def test_per_cell_leaves_a_missing_train_as_zeros(
    cfg, mock_run_factory, operator, model, tmp_path
):
    from analysis.waxs.plan import build_plan, open_detector

    run, dc = mock_run_factory(zero_entry_trains=(10002,))
    op, ai = operator
    det = open_detector(cfg, dc)
    plan = build_plan(cfg, LIT_CELLS, dc=dc, control_dc=dc, det=det)
    worker.init_from_detector(cfg, op, ai, LIT_CELLS, det)
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
