"""The photon-statistics pass (context/xpcs-quality.md, Q1) on the mock AGIPD run.

Every frame's histogram must equal a dense count over the same pixels, rows must sit
where the frame table puts them, and the file must resume, refuse another config and
record a frame it could not read.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("extra_data")

import h5py  # noqa: E402
from test_run import InlinePool  # noqa: E402

from analysis.common.masks import frame_bad  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.common.writer import ConfigHashMismatch, IncompleteRun  # noqa: E402
from analysis.saxs.config import NPIX  # noqa: E402
from analysis.saxs.photon_stats import (  # noqa: E402
    RING_ARRAYS,
    PhotonStatsConfig,
    default_rings,
    frame_stats,
    ring_labels,
    run_photon_stats,
)
from analysis.saxs.plan import build_plan  # noqa: E402


def dense(counts, bad, labels, n_rings, k_max):
    """The same statistics by brute force, ring by ring."""
    out = {name: [] for name in ("n_valid", "hist", "s1", "s2", "max_k")}
    for r in range(n_rings):
        v = counts[(labels == r) & ~bad].astype(np.int64)
        out["n_valid"].append(v.size)
        out["hist"].append(np.bincount(np.minimum(v, k_max + 1), minlength=k_max + 2))
        out["s1"].append(v.sum())
        out["s2"].append((v * (v - 1)).sum())
        out["max_k"].append(v.max(initial=0))
    return {k: np.array(v) for k, v in out.items()}


def test_frame_stats_equal_a_dense_count():
    rng = np.random.default_rng(1)
    n_rings, k_max = 5, 3
    counts = np.zeros(NPIX, np.int16)
    hits = rng.choice(NPIX, 20_000, replace=False)
    counts[hits] = rng.integers(1, 9, hits.size)  # above k_max too: the overflow bin
    labels = rng.integers(-1, n_rings, NPIX).astype(np.int16)
    bad = rng.random(NPIX) < 0.05
    in_ring = labels >= 0
    ring_n = np.bincount(labels[in_ring], minlength=n_rings)
    got = frame_stats(counts, bad, labels, ring_n, in_ring, k_max)
    want = dense(counts, bad, labels, n_rings, k_max)
    for name, value in zip(("n_valid", "hist", "s1", "s2", "max_k"), got, strict=True):
        np.testing.assert_array_equal(value, want[name], name)
    assert (want["hist"][:, k_max + 1] > 0).all()  # the overflow bin was exercised


def test_default_rings_tile_the_plain_rois():
    rings = default_rings()
    assert len(rings) == 17
    assert rings[0][0] == pytest.approx(0.225) and rings[-1][1] == pytest.approx(0.86)
    for lo, hi in rings:
        assert not (lo < 0.65 and hi > 0.57) and not (lo < 0.74 and hi > 0.68)


@pytest.fixture(scope="module")
def mock(mock_run_factory):
    return mock_run_factory()


def pass_on(mock, run_cfg, geom, path, *, pool=InlinePool, **changes):
    run, dc = mock
    cfg = PhotonStatsConfig(saxs=run_cfg, **changes)
    run_photon_stats(cfg, dc=dc, geometry=geom, static_bad=np.zeros(NPIX, bool),
                     run_dir=run.path, output_path=path, pool_factory=pool)  # fmt: skip
    return cfg, path


@pytest.fixture(scope="module")
def written(mock, run_cfg, geom, tmp_path_factory):
    return pass_on(mock, run_cfg, geom, tmp_path_factory.mktemp("ps") / "ps.h5")


def test_every_frame_equals_a_dense_count_of_the_proc_frame(mock, written, geom):
    from extra_data import by_id
    from extra_data.components import AGIPD1M

    run, dc = mock
    cfg, path = written
    labels, _ = ring_labels(cfg, geom)
    det = AGIPD1M(dc, min_modules=cfg.saxs.min_modules)
    with h5py.File(path, "r") as f:
        frames = {name: f["frames"][name][:] for name in f["frames"]}
        np.testing.assert_array_equal(f["rings/labels"][:], labels)
    assert (frames["status"] == FrameStatus.OK).all()
    checked = 0
    for tid in run.train_ids:
        sel = det.select_trains(by_id[[tid]])
        data = sel["image.data"].ndarray()
        mask = sel["image.mask"].ndarray()
        cells = sel["image.data"].cell_id_coordinates()
        for f in range(data.shape[1]):
            row = np.flatnonzero(
                (frames["trainId"] == tid) & (frames["cellId"] == cells[f])
            )
            assert row.size == 1
            bad = frame_bad(mask[:, f], cfg.saxs.mask_bits, np.zeros(NPIX, bool))
            want = dense(data[:, f].reshape(-1), bad, labels, cfg.npt, cfg.k_max)
            for name in RING_ARRAYS:
                np.testing.assert_array_equal(frames[name][row[0]], want[name], name)
            checked += 1
    assert checked == frames["status"].size
    assert frames["hist"][:, :, 1:].sum() > 0  # the mock's photons reach the rings


def test_rows_are_laid_out_as_the_frame_table(mock, written, run_cfg):
    _, dc = mock
    _, path = written
    plan = build_plan(run_cfg, dc=dc)
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["trains/trainId"][:], plan.train_ids)
        np.testing.assert_array_equal(
            f["trains/count"][:], [t.n_frames for t in plan.trains]
        )
        assert f["rings/edges_nm"].shape == (17, 2)
        assert f["rings"].attrs["k_max"] == 15


def test_a_rerun_resumes_and_another_config_is_refused(mock, written, run_cfg, geom):
    cfg, path = written
    with h5py.File(path, "r") as f:
        before = f["frames/hist"][:]
    pass_on(mock, run_cfg, geom, path)  # every block complete: nothing to do
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["frames/hist"][:], before)
    with pytest.raises(ConfigHashMismatch):
        pass_on(mock, run_cfg, geom, path, k_max=7)


def test_the_hash_ignores_what_changes_no_count(run_cfg):
    cfg = PhotonStatsConfig(saxs=run_cfg)
    same = PhotonStatsConfig(
        saxs=replace(run_cfg, npt=2000, n_workers=7, trains_per_block=9),
        output_root="/elsewhere", overwrite=True,
    )  # fmt: skip
    assert cfg.config_hash() == same.config_hash()
    moved = PhotonStatsConfig(saxs=replace(run_cfg, sdd_m=7.0))
    assert cfg.config_hash() != moved.config_hash()


def test_a_frame_that_is_not_counts_is_recorded_not_histogrammed(
    mock_run_factory, run_cfg, geom, tmp_path
):
    bad = mock_run_factory(negative_counts_module=3)
    with pytest.raises(IncompleteRun):
        pass_on(bad, run_cfg, geom, tmp_path / "ps.h5")
    with h5py.File(tmp_path / "ps.h5", "r") as f:
        status = f["frames/status"][:]
        assert (status == FrameStatus.DATA_CHECK_FAILED).any()
        assert (f["frames/hist"][:][status != FrameStatus.OK] == 0).all()
    pass_on(bad, replace(run_cfg, allow_incomplete=True), geom, tmp_path / "again.h5")
