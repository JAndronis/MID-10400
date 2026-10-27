"""The DAMNIT-facing functions and the per-pulse reducer.

The DAMNIT variable runs once per run on a cluster node, and everyone
downstream reads what it returns, so the grid's labelling is tested against a
manual computation rather than against itself. A frame must land on the
(train, pulse) slot its *stored* labels name, not the one its position implies,
because a dropped train shifts every later frame's position.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

pytest.importorskip("extra_data")
matplotlib.use("Agg")

import h5py  # noqa: E402
from mockrun import AWKWARD  # noqa: E402
from test_run import InlinePool  # noqa: E402  (tests/saxs is on sys.path)

from analysis.common.status import FrameStatus  # noqa: E402
from analysis.saxs import damnit  # noqa: E402
from analysis.saxs.writer import per_pulse  # noqa: E402


@pytest.fixture(scope="module")
def finished(run_cfg, geom, mock_run_factory, tmp_path_factory):
    """A completed pass over the awkward mock run, and its output file.

    The run has a dropped train, a train short of modules and a zero-frame
    train, so a frame's position and its labels part ways after each of them.
    """
    from analysis.saxs.run import run_agipd_saxs

    run, dc = mock_run_factory(**AWKWARD)
    tmp_path = tmp_path_factory.mktemp("finished")
    output = tmp_path / "out.h5"
    run_agipd_saxs(
        run_cfg,
        dc=dc,
        geometry=geom,
        run_dir=run.path,
        output_path=output,
        work_dir=tmp_path,
        pool_factory=InlinePool,
        reduce="none",
    )
    return run, output


@pytest.fixture
def editable(finished, tmp_path):
    """A copy of the finished frame table, for the tests that alter it."""
    run, output = finished
    copy = tmp_path / "out.h5"
    shutil.copy2(output, copy)
    return run, copy


# ── the per-pulse grid ────────────────────────────────────────────────────────


def test_every_frame_lands_on_the_slot_its_labels_name(finished):
    """The grid is checked against the rows, not against a reshape of them."""
    _, output = finished
    grid = per_pulse(output)
    trains = list(grid["trainId"].values)
    pulses = list(grid["pulseId"].values)

    with h5py.File(output) as handle:
        frames = handle["frames"]
        for row in range(frames["status"].size):
            if frames["status"][row] != FrameStatus.OK:
                continue
            signal = frames["signal"][row].astype(np.float64)
            normalization = frames["normalization"][row].astype(np.float64)
            expected = np.zeros_like(signal)
            valid = normalization > 0
            expected[valid] = signal[valid] / normalization[valid]

            t = trains.index(frames["trainId"][row])
            p = pulses.index(frames["reader_pulseId"][row])
            assert np.allclose(grid.values[t, p], expected, rtol=1e-6, atol=0)


def test_a_slot_with_no_frame_is_zero_not_nan(editable):
    """An empty slot holds zeros and a status count, never a NaN sentinel."""
    _, output = editable
    with h5py.File(output, "r+") as handle:
        handle["frames/status"][0] = FrameStatus.DATA_CHECK_FAILED
        train = int(handle["frames/trainId"][0])
        pulse = int(handle["frames/reader_pulseId"][0])

    grid = per_pulse(output)
    assert np.isfinite(grid.values).all()
    t = list(grid["trainId"].values).index(train)
    p = list(grid["pulseId"].values).index(pulse)
    assert grid["n_frames"].values[t, p] == 0
    assert not grid.values[t, p].any()
    assert json.loads(grid.attrs["unplaced"]) == {FrameStatus.DATA_CHECK_FAILED.name: 1}


def test_unwritten_rows_are_counted_never_guessed_onto_a_slot(editable):
    """An unprocessed row carries trainId 0, so it must not be placed anywhere."""
    _, output = editable
    with h5py.File(output, "r+") as handle:
        n = int(handle["trains/count"][0])
        handle["frames/status"][:n] = FrameStatus.NOT_PROCESSED
        handle["frames/trainId"][:n] = 0
        handle["frames/reader_pulseId"][:n] = 0

    grid = per_pulse(output)
    assert json.loads(grid.attrs["unplaced"])[FrameStatus.NOT_PROCESSED.name] == n
    assert 0 not in set(grid["trainId"].values.tolist())


def test_per_pulse_refuses_duplicate_train_pulse_slots(editable):
    """Two frames in one slot would silently overwrite; that must not pass."""
    _, output = editable
    with h5py.File(output, "r+") as handle:
        handle["frames/reader_pulseId"][1] = handle["frames/reader_pulseId"][0]
        handle["frames/trainId"][1] = handle["frames/trainId"][0]

    with pytest.raises(ValueError, match="shared a .train, pulse. slot"):
        per_pulse(output)


def test_per_pulse_refuses_an_unknown_train(editable):
    _, output = editable
    with h5py.File(output, "r+") as handle:
        handle["frames/trainId"][0] = 999_999_999

    with pytest.raises(ValueError, match="not in the train table"):
        per_pulse(output)


def test_chunking_does_not_change_the_grid(finished):
    """The chunk size is a memory knob, not a correctness one."""
    _, output = finished
    whole = per_pulse(output, chunk_rows=10**9)
    split = per_pulse(output, chunk_rows=3)
    assert np.array_equal(whole.values, split.values)
    assert np.array_equal(whole["n_frames"].values, split["n_frames"].values)


# ── pooled: sigma ─────────────────────────────────────────────────────────────


# ── the wrapper ───────────────────────────────────────────────────────────────


# ── the overview figure ───────────────────────────────────────────────────────
def _synthetic_grid(n_trains: int = 4, n_pulses: int = 6, npt: int = 5):
    """A full (trainId, pulseId, q) grid with known values in every slot."""
    import xarray as xr

    rng = np.random.default_rng(0)
    intensity = (rng.random((n_trains, n_pulses, npt)) + 1.0).astype(np.float32)
    return xr.DataArray(
        intensity,
        dims=("trainId", "pulseId", "q"),
        coords={
            "trainId": np.arange(n_trains, dtype=np.uint64),
            "pulseId": np.arange(n_pulses, dtype=np.uint64),
            "q": np.linspace(0.1, 1.0, npt),
            "n_frames": (
                ("trainId", "pulseId"),
                np.ones((n_trains, n_pulses), dtype=np.uint8),
            ),
        },
        name="intensity",
    )


def _axes(fig):
    """Split ``fig.axes`` into ``(plot axes, colour-bar axes)``."""
    bars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    return [ax for ax in fig.axes if ax not in bars], bars


def test_overview_averages_only_the_slots_that_hold_a_frame(finished):
    """Averaging the empty slots' zeros in would drag every curve towards zero.

    Built from a synthetic grid rather than the mock run, so the expected curve
    is exact instead of being whatever the mock happens to contain.
    """
    grid = _synthetic_grid(n_trains=4, n_pulses=6, npt=5)
    full = grid.values.copy()

    # Half the pulses never held a frame: zeros, with n_frames saying so.
    grid["n_frames"].values[:, 1::2] = 0
    grid.values[:, 1::2] = 0.0

    curve = damnit.overview_figure(grid).axes[0].lines[0].get_ydata()
    kept = full[:, 0::2].reshape(-1, full.shape[-1]).mean(axis=0)
    assert np.allclose(curve, kept, rtol=1e-6)

    # Averaging the zeros in would have halved it; prove the test can tell.
    assert not np.allclose(curve, kept / 2, rtol=1e-3)


def test_overview_survives_a_run_with_nothing_integrated(finished):
    """An empty run has no positive value for LogNorm to anchor on."""
    _, output = finished
    grid = damnit.per_pulse_from_file(output)
    grid["n_frames"].values[:] = 0
    grid.values[:] = 0.0

    plots, bars = _axes(damnit.overview_figure(grid))
    assert len(plots) == 3
    # Nothing to scale a colour bar against, so the maps say so instead.
    assert bars == []
    assert all(
        "no integrated frames" in [text.get_text() for text in ax.texts]
        for ax in plots[1:]
    )


# ── DAMNIT storage ────────────────────────────────────────────────────────────
def test_the_grid_survives_damnits_writer(finished, tmp_path):
    """DAMNIT stores a DataArray as netCDF; uint64 trainIds must come back."""
    damnit_pkg = pytest.importorskip("damnit")
    support = Path(damnit_pkg.__file__).parent / "ctxsupport"
    sys.path.insert(0, str(support))
    try:
        from damnit_ctx import Cell
        from damnit_writing import save_dataarray_netcdf
    finally:
        sys.path.remove(str(support))

    import xarray as xr

    _, output = finished
    grid = damnit.per_pulse_from_file(output)
    stored = tmp_path / "damnit.h5"
    with h5py.File(stored, "w") as handle:
        save_dataarray_netcdf(handle, "agipd_saxs", grid)

    back = xr.open_dataarray(stored, group="agipd_saxs", engine="h5netcdf")
    assert back["trainId"].dtype == np.uint64
    assert back.dims == ("trainId", "pulseId", "q")
    assert np.array_equal(back.values, grid.values)
    # n_frames is a coordinate, so it survives the round trip too.
    assert np.array_equal(back["n_frames"].values, grid["n_frames"].values)

    # The table cell shows dtype and shape, not the dataset size.
    summary = Cell(grid).get_summary("agipd_saxs")
    assert summary == f"{grid.dtype}: {grid.shape}"


# ── agipd_preprocessing and the agipd_saxs reader ─────────────────────────────
def _roots(tmp_path):
    return {
        "output_root": str(tmp_path / "saxs"),
        "pixel_sums_root": str(tmp_path / "sums"),
        "photon_stats_root": str(tmp_path / "stats"),
    }


@pytest.fixture(scope="module")
def preprocessed(run_cfg, geom, mock_run_factory, tmp_path_factory):
    """The awkward mock run written at its config's own paths, as ``preprocess``
    writes it, and the overrides that rebuild that config through ``config_for``.

    Its short and zero-frame trains are final in the window sums, so they must
    not turn the run red.
    """
    from dataclasses import fields, replace

    from analysis.saxs.run import run_agipd_saxs

    run, dc = mock_run_factory(**AWKWARD)
    cfg = replace(run_cfg, **_roots(tmp_path_factory.mktemp("roots")))
    run_agipd_saxs(
        cfg,
        dc=dc,
        geometry=geom,
        run_dir=run.path,
        pool_factory=InlinePool,
        reduce="none",
    )
    overrides = {
        f.name: getattr(cfg, f.name)
        for f in fields(cfg)
        if f.name not in ("proposal", "run")
    }
    assert damnit.config_for(cfg.proposal, cfg.run, **overrides) == cfg
    return cfg, overrides


def _colour(cell) -> str:
    return "#" + bytes(cell.background).hex()


def test_the_reader_returns_what_the_pass_wrote(preprocessed):
    import xarray as xr

    cfg, overrides = preprocessed
    grid = damnit.agipd_saxs_from_file(cfg.proposal, cfg.run, **overrides)
    xr.testing.assert_identical(grid, per_pulse(cfg.output_file))
    status = damnit.product_status(cfg)
    assert set(status["frames"]) == set(status["photon_stats"]) == {"OK"}
    assert status["window_sums"] == {"OK": 5, "MISSING_MODULES": 1, "NO_FRAMES": 1}


def test_preprocess_is_a_status_cell_and_a_failure_is_red(preprocessed, monkeypatch):
    pytest.importorskip("damnit")
    from analysis.diagnostics.damnit import COLORS

    cfg, overrides = preprocessed
    monkeypatch.setattr("analysis.saxs.run.run_agipd_saxs", lambda c, **kw: None)
    cell = damnit.preprocess(cfg.proposal, cfg.run, **overrides)
    assert (
        cell.summary_value.startswith("frames ") and " · stats " in cell.summary_value
    )
    assert _colour(cell) == COLORS["good"]

    def boom(c, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("analysis.saxs.run.run_agipd_saxs", boom)
    cell = damnit.preprocess(cfg.proposal, cfg.run, **overrides)
    assert cell.summary_value == "failed: RuntimeError"
    assert _colour(cell) == COLORS["error"]
    assert "disk full" in json.loads(cell.data)["error"]


def test_the_reader_skips_a_run_without_a_frame_table(tmp_path):
    damnit_ctx = pytest.importorskip("damnit.context")

    with pytest.raises(damnit_ctx.Skip):
        damnit.agipd_saxs_from_file(10400, 423, **_roots(tmp_path))
