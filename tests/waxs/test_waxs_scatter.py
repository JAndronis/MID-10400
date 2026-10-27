"""``frame_scatter``: the stored σ against the scatter between a train's frames."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from analysis.common.status import FrameStatus
from analysis.waxs.scatter import frame_scatter

NPT = 200
FRAMES = 8
TRAINS = 300


def write_file(path, *, extra_scatter=0.0, flux_jitter=0.2, seed=0):
    """A file shaped like the pass's, with Poisson frames of known variance.

    Each frame's counts per bin are Poisson around ``flux x shape``, the stored
    variance is the counts (the photon pass's model), and ``extra_scatter`` adds
    multiplicative shape noise the model does not know about.
    """
    rng = np.random.default_rng(seed)
    q = np.linspace(10, 20, NPT)
    shape = 200.0 * np.exp(-((q - 14) ** 2) / 8) + 50.0
    normalization = np.full(NPT, 100.0)
    n = TRAINS * FRAMES
    signal = np.zeros((n, NPT))
    for row in range(n):
        flux = 1.0 + flux_jitter * rng.standard_normal()
        mean = flux * shape * (1.0 + extra_scatter * rng.standard_normal(NPT))
        signal[row] = rng.poisson(np.clip(mean, 0, None))
    with h5py.File(path, "w") as handle:
        frames = handle.create_group("frames")
        frames["signal"] = signal.astype(np.float32)
        frames["variance"] = signal.astype(np.float32)
        frames["normalization"] = np.tile(normalization, (n, 1)).astype(np.float32)
        frames["status"] = np.full(n, FrameStatus.OK, dtype=np.uint8)
        handle["q/centers"] = q
        trains = handle.create_group("trains")
        trains["first"] = np.arange(0, n, FRAMES, dtype=np.uint64)
        trains["count"] = np.full(TRAINS, FRAMES, dtype=np.uint32)
    return path


def test_a_pure_counting_process_gives_a_ratio_of_one(tmp_path):
    """Flux jitter of 20 % is taken out by the per-frame scaling."""
    path = write_file(tmp_path / "poisson.h5")
    report = frame_scatter(path)
    assert report["n_trains"] == TRAINS
    assert report["sigma_ratio_median"] == pytest.approx(1.0, abs=0.03)
    for band in report["sigma_ratio_by_q_band"]:
        assert band["sigma_ratio"] == pytest.approx(1.0, abs=0.05)

    restricted = frame_scatter(path, q_range=(12.0, 16.0))
    assert restricted["n_bins"] == pytest.approx(NPT * 4 / 10, abs=2)
    assert restricted["q_range"] == [12.0, 16.0]


def test_noise_the_model_leaves_out_raises_the_ratio(tmp_path):
    report = frame_scatter(write_file(tmp_path / "extra.h5", extra_scatter=0.1))
    assert report["sigma_ratio_median"] > 1.5


def test_the_train_spread_is_reported(tmp_path):
    report = frame_scatter(write_file(tmp_path / "jitter.h5", flux_jitter=0.2))
    # Eight frames per train average the 20 % jitter down by sqrt(8).
    assert report["relative_std_between_trains"] == pytest.approx(
        0.2 / np.sqrt(FRAMES), rel=0.2
    )


def test_a_train_with_a_failed_frame_is_left_out(tmp_path):
    path = write_file(tmp_path / "failed.h5")
    with h5py.File(path, "r+") as handle:
        status = handle["frames/status"][:]
        status[3] = FrameStatus.WORKER_ERROR
        handle["frames/status"][:] = status
    assert frame_scatter(path)["n_trains"] == TRAINS - 1
