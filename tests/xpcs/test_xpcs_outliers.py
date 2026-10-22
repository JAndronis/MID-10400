"""extra-speckle's pixel-outlier mask, from the hit trains of the store."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from xpcs_fakes import InlinePool, make_train

from analysis.xpcs.config import XpcsConfig
from analysis.xpcs.outliers import BLOCK_LIMIT, accumulate, outlier_mask, q_interval
from analysis.xpcs.result import XpcsHits
from analysis.xpcs.store import FlagBase

GOOD = (10000, 10001, 10003, 10005, 10006, 10007, 10009)


def reference(dense, flags, candidate, q, q_range, thresholds, q_bins):
    """mask_outlier written out from extra-speckle's source, on dense frames."""
    n_env = dense[0].shape[1]
    x = np.stack(dense).astype(float)  # (T, P, n_env)
    good = np.ones(x.shape, bool)
    for t, train in enumerate(flags):
        for f, pixels in enumerate(train):
            good[t, f, pixels] = False
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (x * good).sum(0) / good.sum(0)
        s = np.sqrt(np.nanmean(mean**2, axis=0))
    s[~candidate] = np.nan
    lo, hi = q_range
    centres = lo + (np.arange(q_bins) + 0.5) * (hi - lo) / q_bins
    ratio = np.full(n_env, np.mean(thresholds))
    for i in range(q_bins - 1):
        inside = (q > centres[i]) & (q <= centres[i + 1])
        if inside.any():
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio[inside] = s[inside] / np.nanmedian(s[inside])
    return (ratio < thresholds[0]) | (ratio > thresholds[1])


def synthetic(rng, n_trains=20, n_frames=6, n_env=400):
    dense = [rng.poisson(2.0, (n_frames, n_env)) for _ in range(n_trains)]
    for d in dense:
        d[:3, 7] += 6  # hot in some frames only, like a cell-block pixel
        d[:, 11] = 0  # dead
    flags = [
        [np.flatnonzero(rng.random(n_env) < 0.05) for _ in range(n_frames)]
        for _ in range(n_trains)
    ]
    flags[0][0] = np.array([7])  # a flag on the hot pixel is honoured
    return dense, flags


def test_the_mask_is_extra_speckles_statistic_with_per_frame_flags():
    rng = np.random.default_rng(1)
    dense, flags = synthetic(rng)
    n_env = dense[0].shape[1]
    trains = [
        make_train(i, d, f) for i, (d, f) in enumerate(zip(dense, flags, strict=True))
    ]
    candidate = rng.random(n_env) > 0.1
    q = np.linspace(0.3, 0.6, n_env)
    sums = accumulate(trains, FlagBase.empty(), n_env, dense[0].shape[0])
    got = outlier_mask(
        sums, FlagBase.empty(), candidate, q, (0.2, 0.7), (0.75, 1.75), 20
    )
    want = reference(dense, flags, candidate, q, (0.2, 0.7), (0.75, 1.75), 20)
    np.testing.assert_array_equal(got.mask, want)
    assert got.mask[7] and (got.mask[11] or not candidate[11])
    assert got.n_trains == len(trains)


def test_a_flag_base_gives_the_same_mask_as_listing_every_flag():
    rng = np.random.default_rng(2)
    dense, flags = synthetic(rng, n_trains=6)
    n_env = dense[0].shape[1]
    listed = [
        make_train(i, d, f) for i, (d, f) in enumerate(zip(dense, flags, strict=True))
    ]
    base = FlagBase.from_samples(
        (t.cell_ids, f) for t, f in zip(listed, flags, strict=True)
    )
    based = [
        make_train(i, d, [base.difference(c, np.sort(p)) for c, p in enumerate(f)])
        for i, (d, f) in enumerate(zip(dense, flags, strict=True))
    ]
    based = [
        type(t)(**{**{k: getattr(t, k) for k in t.__slots__}, "base": base})
        for t in based
    ]
    args = (
        np.ones(n_env, bool),
        np.linspace(0.3, 0.6, n_env),
        (0.2, 0.7),
        (0.75, 1.75),
        20,
    )
    plain = outlier_mask(
        accumulate(listed, FlagBase.empty(), n_env, 6), FlagBase.empty(), *args
    )
    coded = outlier_mask(accumulate(based, base, n_env, 6), base, *args)
    np.testing.assert_array_equal(plain.ratio, coded.ratio)


def test_a_block_is_limited_so_its_sums_fit_uint16():
    big = np.full((1, 3), 255)
    trains = [make_train(i, big, [np.zeros(0, int)]) for i in range(BLOCK_LIMIT)]
    sums = accumulate(trains, FlagBase.empty(), 3, 1)
    assert int(sums.photons.max()) == BLOCK_LIMIT * 255
    with pytest.raises(ValueError, match="more than"):
        accumulate(trains + trains[:1], FlagBase.empty(), 3, 1)


def test_q_intervals_lie_between_consecutive_bin_centres():
    index = q_interval(np.array([0.0, 0.05, 0.15, 0.95, 1.0]), (0.0, 1.0), 10)
    assert index.tolist() == [-1, -1, 0, 8, -1]


def test_ring_masks_files_cannot_carry_the_outlier_mask(with_config):
    with pytest.raises(ValueError, match="ring_masks_file"):
        with_config(outlier_thresholds=(0.75, 1.75), ring_masks_file="x.npy")
    assert XpcsConfig.__dataclass_fields__["outlier_thresholds"].default == (0.75, 1.75)


def test_the_merge_hash_names_the_outlier_settings(with_config):
    hashes = {
        with_config(outlier_thresholds=t).merge_hash("e", "r")
        for t in (None, (0.75, 1.75), (0.5, 2.0))
    }
    assert len(hashes) == 3


def test_the_pass_removes_the_outliers_from_every_ring_and_records_them(pipeline):
    path = pipeline.run(GOOD, outlier_thresholds=(0.9, 1.1))
    hits = XpcsHits(path)
    applied, before = hits.outlier()
    with h5py.File(path, "r") as f:
        label = f["rings/label"][()]
        cnt = f["sums/cnt"][()]
        assert f["provenance"].attrs["outlier_mask_sha256"] == applied.sha256
    assert applied.mask.any() and applied.n_trains == len(GOOD)
    np.testing.assert_array_equal(label, np.where(applied.mask, -1, before))
    off = pipeline.run(GOOD, outlier_thresholds=None)
    with h5py.File(off, "r") as f:
        assert "outlier" not in f
        cnt_off = f["sums/cnt"][()]
    removed = np.bincount(before[applied.mask & (before >= 0)], minlength=cnt.shape[1])
    # a removed pixel some train already lost to ANY_FRAME costs that train nothing
    assert ((cnt_off - cnt) <= removed).all()
    assert ((cnt_off - cnt) > 0).all()


def test_the_mask_does_not_depend_on_workers_or_blocks(pipeline):
    one = XpcsHits(pipeline.run(GOOD, outlier_thresholds=(0.9, 1.1), n_workers=1))
    first = one.outlier()[0].mask.copy()
    three = XpcsHits(
        pipeline.run(GOOD, outlier_thresholds=(0.9, 1.1), n_workers=3, overwrite=True)
    )
    np.testing.assert_array_equal(first, three.outlier()[0].mask)


def test_a_complete_run_is_answered_without_the_statistic(pipeline, monkeypatch):
    path = pipeline.run(GOOD, outlier_thresholds=(0.9, 1.1))
    before = path.stat().st_mtime_ns
    from analysis.xpcs import outliers

    def refuse(*args, **kwargs):
        raise AssertionError("the statistic ran again")

    monkeypatch.setattr(outliers, "process_block", refuse)
    pipeline.run(GOOD, outlier_thresholds=(0.9, 1.1))
    assert path.stat().st_mtime_ns == before


def test_a_failed_block_stops_the_merge(pipeline, monkeypatch):
    from analysis.xpcs import outliers
    from analysis.xpcs.outliers import OutlierFailed

    def boom(block):
        raise OSError("disk")

    monkeypatch.setattr(outliers, "process_block", boom)
    with pytest.raises(OutlierFailed):
        pipeline.run(GOOD, outlier_thresholds=(0.9, 1.1), pool=InlinePool)
