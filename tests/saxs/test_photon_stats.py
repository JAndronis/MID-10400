"""The photon-statistics pass on the mock AGIPD run.

Every frame's histogram must equal a dense count over the same pixels, rows must sit
where the frame table puts them, and a frame that is not photon counts is recorded,
not histogrammed.
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
from analysis.common.writer import IncompleteRun  # noqa: E402
from analysis.saxs.config import NPIX  # noqa: E402
from analysis.saxs.photon_stats import (  # noqa: E402
    RING_ARRAYS,
    PhotonStatsConfig,
    default_rings,
    frame_stats,
    ring_labels,
    run_photon_stats,
)
from analysis.xpcs.config import touches_bragg  # noqa: E402


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


@pytest.fixture(scope="module")
def mock(mock_run_factory):
    return mock_run_factory()


def pass_on(mock, run_cfg, geom, path, *, pool=InlinePool, **changes):
    run, dc = mock
    cfg = PhotonStatsConfig(saxs=run_cfg, **changes)
    run_photon_stats(cfg, dc=dc, geometry=geom, static_bad=np.zeros(NPIX, bool),
                     run_dir=run.path, output_path=path, pool_factory=pool)  # fmt: skip
    return cfg, path


def test_the_default_rings_leave_out_the_bragg_bands():
    for lo, hi in default_rings():
        assert not touches_bragg(lo, hi), (lo, hi)


def test_every_frame_equals_a_dense_count_of_the_proc_frame(fused, mock_pipeline, geom):
    """The statistics the pass wrote, against its proc frames counted by brute force."""
    from extra_data import by_id
    from extra_data.components import AGIPD1M

    from analysis.saxs.masks import build_static_bad
    from analysis.saxs.photon_stats import FILE_NAME

    run, dc = mock_pipeline.run, mock_pipeline.dc
    cfg, path = PhotonStatsConfig.for_saxs(mock_pipeline.cfg), fused / FILE_NAME
    labels, _ = ring_labels(cfg, geom)
    static = build_static_bad(mock_pipeline.cfg).bad
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
            bad = frame_bad(mask[:, f], cfg.saxs.mask_bits, static)
            want = dense(data[:, f].reshape(-1), bad, labels, cfg.npt, cfg.k_max)
            for name in RING_ARRAYS:
                np.testing.assert_array_equal(frames[name][row[0]], want[name], name)
            checked += 1
    assert checked == frames["status"].size
    assert frames["hist"][:, :, 1:].sum() > 0  # the mock's photons reach the rings


def test_rows_are_laid_out_as_the_frame_table(fused):
    from analysis.saxs.photon_stats import FILE_NAME

    names = ("trainId", "cellId", "reader_pulseId")
    with h5py.File(fused / "out.h5", "r") as table, h5py.File(fused / FILE_NAME) as f:
        for name in ("trainId", "first", "count"):
            np.testing.assert_array_equal(
                f[f"trains/{name}"][:], table[f"trains/{name}"][:], name
            )
        for name in names:
            np.testing.assert_array_equal(
                f[f"frames/{name}"][:], table[f"frames/{name}"][:], name
            )


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
    """Through the standalone pass: the fused one refuses this run earlier, at
    its self-test, which samples the negative frame."""
    bad = mock_run_factory(negative_counts_module=3)
    with pytest.raises(IncompleteRun):
        pass_on(bad, run_cfg, geom, tmp_path / "ps.h5")
    with h5py.File(tmp_path / "ps.h5", "r") as f:
        status = f["frames/status"][:]
        assert (status == FrameStatus.DATA_CHECK_FAILED).any()
        assert (f["frames/hist"][:][status != FrameStatus.OK] == 0).all()
    pass_on(bad, replace(run_cfg, allow_incomplete=True), geom, tmp_path / "again.h5")


# ── written by the agipd_saxs pass from its own read ─────────────────────────


@pytest.fixture(scope="module")
def fused(mock_pipeline, geom, tmp_path_factory):
    """A whole mock run through ``run_agipd_saxs``: frame table, sums, statistics."""
    from test_run import run_on_mock

    mock_pipeline.geometry = geom
    directory = tmp_path_factory.mktemp("fused")
    run_on_mock(mock_pipeline, directory)
    return directory


def contents(path):
    """Every per-frame column, the ring map and the hash of a statistics file."""
    with h5py.File(path, "r") as f:
        out = {f"frames/{name}": f["frames"][name][:] for name in f["frames"]}
        out["rings/labels"] = f["rings/labels"][:]
        out["rings/edges_nm"] = f["rings/edges_nm"][:]
        out["config_hash"] = f["provenance"].attrs["config_hash"]
    return out


def _stat(path):
    return path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size


def test_statistics_alone_leave_the_frame_table_and_sums_untouched(
    fused, mock_pipeline, tmp_path
):
    """A finished run gains its statistics; its other files keep inode and mtime,
    and no operator or masks archive is written for an integration."""
    import shutil

    from test_run import run_on_mock

    from analysis.saxs.photon_stats import FILE_NAME
    from analysis.saxs.pixel_sums import FILE_NAME as SUMS

    for name in ("out.h5", SUMS):
        shutil.copy2(fused / name, tmp_path / name)
    before = {name: _stat(tmp_path / name) for name in ("out.h5", SUMS)}

    run_on_mock(mock_pipeline, tmp_path)
    assert {name: _stat(tmp_path / name) for name in before} == before
    assert not list(tmp_path.glob("*.npz")) and not list(tmp_path.glob("*.lock"))
    got, want = contents(tmp_path / FILE_NAME), contents(fused / FILE_NAME)
    for name in want:
        np.testing.assert_array_equal(got[name], want[name], name)


def test_a_run_without_its_statistics_is_not_complete(fused, mock_pipeline, tmp_path):
    import shutil

    from analysis.saxs.complete import is_complete
    from analysis.saxs.photon_stats import FILE_NAME
    from analysis.saxs.pixel_sums import FILE_NAME as SUMS

    for name in ("out.h5", SUMS):
        shutil.copy2(fused / name, tmp_path / name)
    kwargs = {"pixel_sums_path": tmp_path / SUMS, "geometry": mock_pipeline.geometry}
    verdict = is_complete(
        tmp_path / "out.h5",
        mock_pipeline.cfg,
        photon_stats_path=tmp_path / FILE_NAME,
        **kwargs,
    )
    assert verdict.reasons == ("photon statistics: no file",)
    off = replace(mock_pipeline.cfg, photon_stats=False)
    assert is_complete(tmp_path / "out.h5", off, **kwargs)


def test_the_statistics_can_be_switched_off(mock_pipeline, geom, tmp_path):
    from test_run import run_on_mock

    from analysis.saxs.photon_stats import FILE_NAME

    mock_pipeline.geometry = geom
    run_on_mock(mock_pipeline, tmp_path, photon_stats=False)
    assert (tmp_path / "out.h5").exists() and not (tmp_path / FILE_NAME).exists()
