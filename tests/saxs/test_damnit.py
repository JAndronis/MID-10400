"""The DAMNIT-facing surface and the per-pulse reducer (P5).

The variable this backs runs once per run on a cluster node, and what it
returns is what everyone downstream sees, so the grid's labelling is tested
against a manual computation rather than against itself: a frame must land on
the (train, pulse) slot its *stored* labels name, not the one its position
implies (CLAUDE.md pitfall 4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

pytest.importorskip("extra_data")
matplotlib.use("Agg")

import h5py  # noqa: E402
from test_run import InlinePool  # noqa: E402  (tests/saxs is on sys.path)

from analysis.saxs import damnit  # noqa: E402
from analysis.saxs.status import FrameStatus  # noqa: E402
from analysis.saxs.writer import per_pulse, pooled_per_train  # noqa: E402


@pytest.fixture
def finished(mock_pipeline, geom, tmp_path):
    """A completed pass over the mock run, plus its output file."""
    from analysis.saxs.run import run_agipd_saxs

    output = tmp_path / "out.h5"
    run_agipd_saxs(
        mock_pipeline.cfg,
        dc=mock_pipeline.dc,
        geometry=geom,
        run_dir=mock_pipeline.run.path,
        output_path=output,
        work_dir=tmp_path,
        pool_factory=InlinePool,
        reduce="none",
    )
    return mock_pipeline, output


# ── the per-pulse grid ────────────────────────────────────────────────────────
def test_per_pulse_has_the_dims_p5_asks_for(finished):
    pipeline, output = finished
    grid = per_pulse(output)

    assert grid["intensity"].dims == ("trainId", "pulseId", "q")
    assert grid["n_frames"].dims == ("trainId", "pulseId")
    with h5py.File(output) as handle:
        assert np.array_equal(grid["trainId"].values, handle["trains/trainId"][:])
        assert np.array_equal(grid["q"].values, handle["q/centers"][:])
    assert int(grid["n_frames"].values.sum()) == pipeline.plan.n_frames
    assert grid.attrs["n_placed"] == pipeline.plan.n_frames


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
            assert np.allclose(
                grid["intensity"].values[t, p], expected, rtol=1e-6, atol=0
            )


def test_a_slot_with_no_frame_is_zero_not_nan(finished):
    """Context file §3 rule 7: a status code and zeros, never a NaN sentinel."""
    _, output = finished
    with h5py.File(output, "r+") as handle:
        handle["frames/status"][0] = FrameStatus.DATA_CHECK_FAILED
        train = int(handle["frames/trainId"][0])
        pulse = int(handle["frames/reader_pulseId"][0])

    grid = per_pulse(output)
    assert np.isfinite(grid["intensity"].values).all()
    t = list(grid["trainId"].values).index(train)
    p = list(grid["pulseId"].values).index(pulse)
    assert grid["n_frames"].values[t, p] == 0
    assert not grid["intensity"].values[t, p].any()
    assert json.loads(grid.attrs["unplaced"]) == {FrameStatus.DATA_CHECK_FAILED.name: 1}


def test_unwritten_rows_are_counted_never_guessed_onto_a_slot(finished):
    """An unprocessed row carries trainId 0 — it must not be placed anywhere."""
    _, output = finished
    with h5py.File(output, "r+") as handle:
        n = int(handle["trains/count"][0])
        handle["frames/status"][:n] = FrameStatus.NOT_PROCESSED
        handle["frames/trainId"][:n] = 0
        handle["frames/reader_pulseId"][:n] = 0

    grid = per_pulse(output)
    assert json.loads(grid.attrs["unplaced"])[FrameStatus.NOT_PROCESSED.name] == n
    assert 0 not in set(grid["trainId"].values.tolist())


def test_per_pulse_refuses_duplicate_train_pulse_slots(finished):
    """Two frames in one slot would silently overwrite; that must not pass."""
    _, output = finished
    with h5py.File(output, "r+") as handle:
        handle["frames/reader_pulseId"][1] = handle["frames/reader_pulseId"][0]
        handle["frames/trainId"][1] = handle["frames/trainId"][0]

    with pytest.raises(ValueError, match="shared a .train, pulse. slot"):
        per_pulse(output)


def test_per_pulse_refuses_an_unknown_train(finished):
    _, output = finished
    with h5py.File(output, "r+") as handle:
        handle["frames/trainId"][0] = 999_999_999

    with pytest.raises(ValueError, match="not in the train table"):
        per_pulse(output)


def test_per_pulse_dtype_is_settable(finished):
    _, output = finished
    assert per_pulse(output)["intensity"].dtype == np.float32
    assert per_pulse(output, dtype=np.float64)["intensity"].dtype == np.float64


def test_chunking_does_not_change_the_grid(finished):
    """The chunk size is a memory knob, not a correctness one."""
    _, output = finished
    whole = per_pulse(output, chunk_rows=10**9)
    split = per_pulse(output, chunk_rows=3)
    assert np.array_equal(whole["intensity"].values, split["intensity"].values)
    assert np.array_equal(whole["n_frames"].values, split["n_frames"].values)


# ── pooled: sigma ─────────────────────────────────────────────────────────────
def test_pooled_carries_sigma(finished):
    """σ(q) = sqrt(ΣV)/ΣN — the stored variance was unreachable before."""
    _, output = finished
    pooled = pooled_per_train(output)
    assert pooled["sigma"].dims == ("trainId", "q")

    with h5py.File(output) as handle:
        first = int(handle["trains/first"][0])
        n = int(handle["trains/count"][0])
        rows = slice(first, first + n)
        ok = handle["frames/status"][rows] == FrameStatus.OK
        variance = handle["frames/variance"][rows][ok].sum(axis=0)
        normalization = handle["frames/normalization"][rows][ok].sum(axis=0)

    valid = normalization > 0
    expected = np.sqrt(variance[valid]) / normalization[valid]
    assert np.allclose(pooled["sigma"].values[0][valid], expected, rtol=1e-6)
    assert (pooled["sigma"].values >= 0).all()
    assert np.isfinite(pooled["sigma"].values).all()


def test_pooled_reads_from_the_file_alone(finished):
    """Post hoc use (§9) has no plan object, only the output file."""
    _, output = finished
    assert pooled_per_train(output)["intensity"].shape[0] > 0


# ── the wrapper ───────────────────────────────────────────────────────────────
def test_config_for_uses_the_beamtime_defaults():
    from analysis.saxs.config import (
        DEFAULT_GEOMETRY_FILE,
        DEFAULT_PIXEL_MASK_FILE,
    )

    cfg = damnit.config_for(10400, 423)
    assert (cfg.proposal, cfg.run) == (10400, 423)
    assert cfg.geometry_file == DEFAULT_GEOMETRY_FILE
    assert cfg.pixel_mask_file == DEFAULT_PIXEL_MASK_FILE
    assert cfg.npt == 500


def test_config_for_applies_overrides():
    cfg = damnit.config_for(10400, 423, npt=1000, allow_incomplete=True)
    assert (cfg.npt, cfg.allow_incomplete) == (1000, True)


def test_agipd_saxs_does_not_swallow_an_incomplete_run(monkeypatch, tmp_path):
    """P5: fail loudly. A partial I(q) must never look like a whole one."""
    from analysis.saxs.writer import IncompleteRun

    def boom(cfg, **kwargs):
        raise IncompleteRun("not every frame reached OK: {'WORKER_ERROR': 3}")

    monkeypatch.setattr("analysis.saxs.run.run_agipd_saxs", boom)
    with pytest.raises(IncompleteRun):
        damnit.agipd_saxs(10400, 423)


def test_agipd_saxs_asks_for_the_per_pulse_grid(monkeypatch):
    """The stored variable is (trainId, pulseId, q), not the pooled view."""
    seen = {}

    def capture(cfg, **kwargs):
        seen.update(kwargs)
        seen["cfg"] = cfg
        return "grid"

    monkeypatch.setattr("analysis.saxs.run.run_agipd_saxs", capture)
    assert damnit.agipd_saxs(10400, 423) == "grid"
    assert seen["reduce"] == "per_pulse"
    # DAMNIT's own proc-only run object must not be passed in: the run checks
    # need raw, and run_agipd_saxs opens both itself.
    assert "dc" not in seen
    assert seen["cfg"].run == 423


def test_readers_round_trip_a_finished_run(finished):
    pipeline, output = finished
    grid = damnit.per_pulse_from_file(output)
    pooled = damnit.pooled_from_file(output)
    assert int(grid["n_frames"].values.sum()) == pipeline.plan.n_frames
    assert int(pooled["n_frames"].values.sum()) == pipeline.plan.n_frames


# ── the overview figure ───────────────────────────────────────────────────────
def _synthetic_grid(n_trains: int = 4, n_pulses: int = 6, npt: int = 5):
    """A full (trainId, pulseId, q) grid with known values in every slot."""
    import xarray as xr

    rng = np.random.default_rng(0)
    intensity = (rng.random((n_trains, n_pulses, npt)) + 1.0).astype(np.float32)
    return xr.Dataset(
        {
            "intensity": (("trainId", "pulseId", "q"), intensity),
            "n_frames": (
                ("trainId", "pulseId"),
                np.ones((n_trains, n_pulses), dtype=np.uint8),
            ),
        },
        coords={
            "trainId": np.arange(n_trains, dtype=np.uint64),
            "pulseId": np.arange(n_pulses, dtype=np.uint64),
            "q": np.linspace(0.1, 1.0, npt),
        },
    )


def test_overview_figure_builds_from_the_grid_alone(finished):
    _, output = finished
    fig = damnit.overview_figure(damnit.per_pulse_from_file(output))
    assert len(fig.axes) == 3
    assert fig.axes[0].get_yscale() == "log"
    assert "nm" in fig.axes[0].get_xlabel()


def test_overview_averages_only_the_slots_that_hold_a_frame(finished):
    """Averaging the empty slots' zeros in would drag every curve towards zero.

    Built from a synthetic grid rather than the mock run, so the expected curve
    is exact instead of being whatever the mock happens to contain.
    """
    grid = _synthetic_grid(n_trains=4, n_pulses=6, npt=5)
    full = grid["intensity"].values.copy()

    # Half the pulses never held a frame: zeros, with n_frames saying so.
    grid["n_frames"].values[:, 1::2] = 0
    grid["intensity"].values[:, 1::2] = 0.0

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
    grid["intensity"].values[:] = 0.0
    assert len(damnit.overview_figure(grid).axes) == 3


# ── DAMNIT storage ────────────────────────────────────────────────────────────
def test_the_grid_survives_damnits_writer(finished, tmp_path):
    """DAMNIT stores a Dataset as netCDF; uint64 trainIds must come back."""
    damnit_pkg = pytest.importorskip("damnit")
    support = Path(damnit_pkg.__file__).parent / "ctxsupport"
    sys.path.insert(0, str(support))
    try:
        from damnit_writing import save_dataset_netcdf
    finally:
        sys.path.remove(str(support))

    import xarray as xr

    _, output = finished
    grid = damnit.per_pulse_from_file(output)
    stored = tmp_path / "damnit.h5"
    with h5py.File(stored, "w") as handle:
        save_dataset_netcdf(handle, "agipd_saxs", grid)

    back = xr.open_dataset(stored, group="agipd_saxs", engine="h5netcdf")
    assert back["trainId"].dtype == np.uint64
    assert back["intensity"].dims == ("trainId", "pulseId", "q")
    assert np.array_equal(back["intensity"].values, grid["intensity"].values)
