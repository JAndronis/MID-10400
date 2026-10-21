"""The reader: pooled and extra-speckle averaging, g2 and its error, coverage."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from xpcs_fakes import (
    InlinePool,
    build_results,
    es_chain,
    load_extra_speckle_module,
    make_dense,
    make_flags,
    make_rings,
    make_train,
)

from analysis.common.status import FrameStatus
from analysis.xpcs.config import Kernel
from analysis.xpcs.kernel import PairStatus, train_sums
from analysis.xpcs.result import (
    BASE_CLOCK_HZ,
    CHUNK,
    Averaging,
    NonUniformPattern,
    RepRateMismatch,
    XpcsHits,
    _lag_bins,
    read,
)

N_ENV, P, R = 300, 14, 3
IDS = (100, 101, 104, 140, 141, 145)


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(5)
    rings = make_rings(rng, N_ENV, R)
    dense = [make_dense(rng, P, N_ENV, 0.12) for _ in IDS]
    flags = [make_flags(rng, P, N_ENV) for _ in IDS]
    photons = [make_train(t, d, f) for t, d, f in zip(IDS, dense, flags, strict=True)]
    return rings, dense, flags, photons


@pytest.fixture
def results(world, tmp_path):
    rings, _, _, photons = world
    return build_results(tmp_path / "xpcs_hits.h5", photons, rings)


def sums_of(world, kernel=Kernel.CLEAN, gap=10):
    """The TrainSums the results file holds, made again in memory."""
    from analysis.xpcs.kernel import partners

    rings, _, _, photons = world
    ids = np.array([t.train_id for t in photons], dtype=np.uint64)
    pairing = partners(
        ids, np.ones(len(ids), bool), np.zeros(len(ids), int), kernel, gap
    )
    by_id = {t.train_id: t for t in photons}
    return [
        train_sums(t, by_id.get(int(p)), rings, kernel)
        for t, p in zip(photons, pairing.partner, strict=True)
    ]


def reference_lag_terms(sums, lags):
    """Plain loops: the numerators and denominators of g2 and of the baseline."""
    shape = (len(sums), R, len(lags))
    on, den_on, off, den_off = (np.zeros(shape) for _ in range(4))
    for i, s in enumerate(sums):
        for r in range(R):
            cnt = float(s.cnt[r])
            for j, k in enumerate(lags):
                for a in range(P - k):
                    on[i, r, j] += s.on[r, a, a + k]
                    if cnt:
                        den_on[i, r, j] += s.s_on[r, a] * s.s_on[r, a + k] / cnt
                    off[i, r, j] += s.off[r, a, a + k] + s.off[r, a + k, a]
                    pair = float(s.cnt_pair[r])
                    if pair:
                        den_off[i, r, j] += (
                            s.s_off_cur[r, a] * s.s_off_prev[r, a + k]
                            + s.s_off_cur[r, a + k] * s.s_off_prev[r, a]
                        ) / pair
    return on, den_on, off, den_off


def ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def test_the_pooled_g2_is_a_ratio_of_summed_lag_sums(world, results):
    sums = sums_of(world)
    lags = results.lags()
    assert lags.tolist() == list(range(1, P))
    on, den_on, off, den_off = reference_lag_terms(sums, lags)
    paired = np.array([s.has_off for s in sums])
    g2 = results.g2(binned=False)
    np.testing.assert_allclose(
        g2.g2.values, ratio(on.sum(0), den_on.sum(0)), rtol=1e-12
    )
    np.testing.assert_allclose(
        g2.baseline.values,
        ratio(off[paired].sum(0), den_off[paired].sum(0)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        g2.c2.values,
        ratio(on[paired].sum(0), den_on[paired].sum(0))
        - ratio(off[paired].sum(0), den_off[paired].sum(0)),
        rtol=1e-12,
    )
    assert g2.attrs["n_trains"] == len(IDS)
    assert g2.attrs["n_with_baseline"] == int(paired.sum())
    assert g2.roi_q.values.tolist() == pytest.approx(results.roi_q.tolist())


def test_the_lag_axis_is_the_pulse_period_times_the_lag(results):
    g2 = results.g2(binned=False)
    period = 2 / BASE_CLOCK_HZ
    assert period == pytest.approx(443.08e-9, rel=1e-4)
    np.testing.assert_allclose(g2.tau.values, np.arange(1, P) * period)
    assert g2.lag.values.tolist() == list(range(1, P))


def test_a_subset_of_trains_pools_as_if_only_they_had_been_summed(world, results):
    subset = np.array([100, 101, 140, 145], dtype=np.uint64)
    sums = [s for s in sums_of(world) if s.train_id in subset.tolist()]
    on, den_on, _, _ = reference_lag_terms(sums, results.lags())
    g2 = results.g2(trains=subset, binned=False)
    np.testing.assert_allclose(
        g2.g2.values, ratio(on.sum(0), den_on.sum(0)), rtol=1e-12
    )
    assert g2.attrs["n_trains"] == 4
    raw = results.ttcf(trains=subset)["raw"].values
    num = sum(s.on.astype(float) for s in sums)
    den = sum(
        s.s_on[:, :, None].astype(float) * s.s_on[:, None, :] / s.cnt[:, None, None]
        for s in sums
    )
    np.testing.assert_allclose(raw, ratio(num, den), rtol=1e-12)


def test_asking_for_a_train_outside_the_hit_set_is_an_error(results):
    with pytest.raises(ValueError, match="not in the hit set"):
        results.g2(trains=np.array([100, 999], dtype=np.uint64))


def test_identical_trains_pool_to_the_per_train_ttcf(tmp_path):
    rng = np.random.default_rng(9)
    rings = make_rings(rng, N_ENV, R)
    dense, flags = make_dense(rng, P, N_ENV, 0.2), make_flags(rng, P, N_ENV)
    same = [make_train(t, dense, flags) for t in (10, 11, 12, 13)]
    hits = build_results(tmp_path / "same.h5", same, rings)
    pooled = hits.ttcf()["raw"].values
    single = hits.per_train(trains=np.array([10], dtype=np.uint64))["TTCF_raw"].values[
        0
    ]
    np.testing.assert_allclose(pooled, single, rtol=1e-5)


def test_a_frame_without_photons_is_nan_when_pooled_and_zero_in_extra_speckles_rule(
    tmp_path,
):
    rng = np.random.default_rng(10)
    rings = make_rings(rng, N_ENV, R)
    dense = make_dense(rng, P, N_ENV, 0.2)
    dense[3] = 0
    photons = [make_train(t, dense, make_flags(rng, P, N_ENV)) for t in (1, 2)]
    hits = build_results(tmp_path / "dark.h5", photons, rings)
    pooled = hits.ttcf()["raw"].values
    assert np.isnan(pooled[:, 3, :]).all() and np.isnan(pooled[:, :, 3]).all()
    assert np.isfinite(np.delete(np.delete(pooled, 3, 1), 3, 2)).all()
    per_train = hits.per_train()["TTCF_raw"].values
    assert (per_train[:, :, 3, :] == 0).all()


def test_a_train_without_a_partner_has_no_baseline_but_does_have_a_raw(results):
    per_train = results.per_train()
    off = per_train["TTCF_off"].values
    paired = results.has_off
    assert not paired[0] and paired[1]
    assert np.isnan(off[0]).all() and np.isfinite(off[1]).all()
    assert np.isfinite(per_train["TTCF_raw"].values).all()


def test_the_kbar_is_photons_per_pixel(world, results):
    rings, dense, flags, _ = world
    kbar = results.kbar()
    assert kbar.dims == ("trainId", "frame", "roi_q")
    assert kbar.shape == (len(IDS), P, R)
    sums = sums_of(world)[2]
    np.testing.assert_allclose(
        kbar.values[2], (sums.s_on / sums.cnt[:, None]).T, rtol=1e-12
    )


def es_reference(dense, flags, rings):
    """Per-train ``normalize_ttcf``, ``kbar_filter`` and the mean over trains."""
    kernel = load_extra_speckle_module("xpcs/kernel.py")
    raws, offs = [], []
    for out in es_chain(dense, flags, rings):
        norm = kernel.normalize_ttcf(SimpleNamespace(**out), off_correlation=True)
        raw, off = norm["ttcf_on"].copy(), norm["ttcf_off"].copy()
        # extra_speckle/xpcs/tools.py, kbar_filter
        lit = norm["s_on"] > 3e-3
        keep = np.einsum("aq,bq->qba", lit.T, lit.T)
        raw[~keep] = np.nan
        off[~keep] = np.nan
        raws.append(raw)
        offs.append(off)
    return np.nanmean(raws, axis=0), np.nanmean(offs, axis=0)


def test_the_es_view_is_extra_speckles_normalisation_filter_and_mean(world, tmp_path):
    rings, dense, flags, photons = world
    hits = build_results(tmp_path / "es.h5", photons, rings, kernel=Kernel.ES)
    mean_raw, mean_off = es_reference(dense, flags, rings)
    assert np.isfinite(mean_raw).all() and np.isfinite(mean_off).all()
    view = hits.ttcf(Averaging.ES)
    np.testing.assert_allclose(view["raw"].values, mean_raw, rtol=2e-5)
    np.testing.assert_allclose(view["off"].values, mean_off, rtol=2e-5)
    np.testing.assert_allclose(
        view["c2"].values, mean_raw - mean_off, rtol=2e-5, atol=1e-6
    )
    # every ES chain gives its first train a baseline of 1
    assert np.isfinite(hits.per_train()["TTCF_off"].values).all()
    assert view.attrs["kbar_min"] == 3e-3


def test_the_es_view_drops_the_entries_of_dim_frames_as_extra_speckle_does(
    world, tmp_path
):
    rings, dense, flags, _ = world
    dense = [d.copy() for d in dense]
    dense[2][4] = 0  # a dark frame in one train
    dense[4][9] = 0
    photons = [make_train(t, d, f) for t, d, f in zip(IDS, dense, flags, strict=True)]
    hits = build_results(tmp_path / "dim.h5", photons, rings, kernel=Kernel.ES)
    mean_raw, mean_off = es_reference(dense, flags, rings)
    view = hits.ttcf(Averaging.ES)
    np.testing.assert_allclose(view["raw"].values, mean_raw, rtol=2e-5, equal_nan=True)
    np.testing.assert_allclose(view["off"].values, mean_off, rtol=2e-5, equal_nan=True)
    unfiltered = hits.ttcf(Averaging.ES, kbar_min=None)["raw"].values
    # the dark frames contribute a 0 entry when unfiltered, so the mean moves
    assert not np.allclose(unfiltered, view["raw"].values)


def test_the_es_g2_is_the_diagonal_mean_of_the_baseline_subtracted_map(world, tmp_path):
    rings, _, _, photons = world
    hits = build_results(tmp_path / "es.h5", photons, rings, kernel=Kernel.ES)
    c2 = hits.ttcf(Averaging.ES)["c2"].values
    g2 = hits.g2(Averaging.ES, binned=False)
    for lag in (1, 5, P - 1):
        np.testing.assert_allclose(
            g2.g2.values[:, lag - 1], np.nanmean(np.diagonal(c2, lag, -2, -1), -1)
        )


def test_the_es_g2_drops_a_ring_whose_map_is_more_than_2p5_percent_nan(world, tmp_path):
    """extra-speckle's ``reduce_ttcf(mask_noisy=True)``, in the ES view only."""
    rings, dense, flags, _ = world
    label = np.asarray(rings.label)
    dense = [d.copy() for d in dense]
    for d in dense:
        d[:3, label == 0] = 0  # ring 0 is dark in three frames of every train
    dense[2][4, label == 1] = 0  # ring 1 is dark in one frame of one train only
    photons = [make_train(t, d, f) for t, d, f in zip(IDS, dense, flags, strict=True)]
    hits = build_results(tmp_path / "noisy.h5", photons, rings, kernel=Kernel.ES)
    c2 = hits.ttcf(Averaging.ES)["c2"].values
    nan_share = np.isnan(c2).mean(axis=(1, 2))
    assert nan_share[0] > 0.025 and nan_share[1] < 0.025  # the rule has both sides

    g2 = hits.g2(Averaging.ES, binned=False)
    kept = [r for r in range(R) if r != 0]
    np.testing.assert_array_equal(g2.roi_q.values, hits.roi_q[kept])
    unfiltered = hits.g2(Averaging.ES, binned=False, nan_thresh=None)
    assert unfiltered.sizes["roi_q"] == R
    np.testing.assert_array_equal(g2.g2.values, unfiltered.g2.values[kept])
    assert hits.g2(Averaging.POOLED, binned=False).sizes["roi_q"] == R


def test_the_nan_threshold_is_a_share_of_the_whole_map_not_a_count(world, tmp_path):
    rings, dense, flags, _ = world
    label = np.asarray(rings.label)
    dense = [d.copy() for d in dense]
    for d in dense:
        d[0, label == 0] = 0  # one dark frame: its row and column, 2P-1 of P*P entries
    photons = [make_train(t, d, f) for t, d, f in zip(IDS, dense, flags, strict=True)]
    hits = build_results(tmp_path / "edge.h5", photons, rings, kernel=Kernel.ES)
    share = np.isnan(hits.ttcf(Averaging.ES)["c2"].values[0]).mean()
    assert share == (2 * P - 1) / P**2
    assert (
        hits.g2(Averaging.ES, binned=False, nan_thresh=share * 1.01).sizes["roi_q"] == R
    )
    below = hits.g2(Averaging.ES, binned=False, nan_thresh=share * 0.99)
    assert below.sizes["roi_q"] == R - 1


def test_without_the_kbar_filter_no_entry_is_dropped(world, tmp_path):
    rings, _, _, photons = world
    hits = build_results(tmp_path / "es.h5", photons, rings, kernel=Kernel.ES)
    assert hits.ttcf(Averaging.ES, kbar_min=None).attrs["kbar_min"] == -1.0
    dense_filter = hits.ttcf(Averaging.ES)["raw"].values
    unfiltered = hits.ttcf(Averaging.ES, kbar_min=None)["raw"].values
    assert np.isfinite(unfiltered).sum() >= np.isfinite(dense_filter).sum()


def test_the_jackknife_error_is_the_triage_estimator_on_the_same_pooled_sums(
    world, results
):
    from analysis.triage.xpcs import jackknife_g2

    sums = sums_of(world)
    lags = results.lags()
    on, den_on, off, den_off = reference_lag_terms(sums, lags)
    g2 = results.g2(binned=False, n_blocks=3)
    value, sigma, _ = jackknife_g2(on, den_on, 3)
    np.testing.assert_allclose(g2.g2.values, value, rtol=1e-12)
    np.testing.assert_allclose(g2.sigma_g2.values, sigma, rtol=1e-9)
    assert (g2.sigma_g2.values > 0).all()


def test_the_error_of_c2_comes_from_leaving_out_blocks_of_both_terms(world, results):
    sums = sums_of(world)
    paired = np.array([s.has_off for s in sums])
    on, den_on, off, den_off = (
        a[paired] for a in reference_lag_terms(sums, results.lags())
    )
    blocks = np.array_split(np.arange(paired.sum()), 2)
    loo = np.stack(
        [
            ratio(on.sum(0) - on[b].sum(0), den_on.sum(0) - den_on[b].sum(0))
            - ratio(off.sum(0) - off[b].sum(0), den_off.sum(0) - den_off[b].sum(0))
            for b in blocks
        ]
    )
    expected = np.sqrt(0.5 * ((loo - loo.mean(0)) ** 2).sum(0))
    np.testing.assert_allclose(
        results.g2(binned=False, n_blocks=2).sigma.values, expected, rtol=1e-9
    )


def test_one_train_has_no_error_estimate(tmp_path):
    rng = np.random.default_rng(1)
    rings = make_rings(rng, N_ENV, R)
    one = [make_train(5, make_dense(rng, P, N_ENV, 0.2), make_flags(rng, P, N_ENV))]
    g2 = build_results(tmp_path / "one.h5", one, rings).g2(binned=False)
    assert np.isnan(g2.sigma_g2.values).all()
    assert np.isfinite(g2.g2.values).any()


@pytest.fixture
def long_world(tmp_path):
    rng = np.random.default_rng(21)
    rings = make_rings(rng, 200, 2)
    photons = [
        make_train(t, make_dense(rng, 80, 200, 0.2), make_flags(rng, 80, 200))
        for t in range(500, 506)
    ]
    return build_results(tmp_path / "long.h5", photons, rings), photons, rings


def test_binning_keeps_the_first_lags_and_pools_the_rest_into_log_bins(long_world):
    hits, photons, rings = long_world
    unbinned, binned = hits.g2(binned=False), hits.g2()
    # 80 frames: lags 1 … 69 after extra-speckle's 10 dropped
    assert unbinned.tau.size == 69 and binned.tau.size <= 10 + 20
    assert binned.attrs["binned"] == 1 and unbinned.attrs["binned"] == 0
    np.testing.assert_allclose(binned.g2.values[:, :10], unbinned.g2.values[:, :10])
    assert (np.diff(binned.tau.values) > 0).all()


def test_a_binned_value_is_the_ratio_of_its_lags_summed_numerators_and_denominators(
    long_world,
):
    hits, photons, rings = long_world
    ids = np.array([t.train_id for t in photons], dtype=np.uint64)
    from analysis.xpcs.kernel import partners

    pairing = partners(
        ids, np.ones(len(ids), bool), np.zeros(len(ids), int), Kernel.CLEAN, 10
    )
    by_id = {t.train_id: t for t in photons}
    sums = [
        train_sums(t, by_id.get(int(p)), rings, Kernel.CLEAN)
        for t, p in zip(photons, pairing.partner, strict=True)
    ]
    lags = hits.lags()
    n_rings = 2
    on = np.zeros((len(sums), n_rings, lags.size))
    den = np.zeros_like(on)
    for i, s in enumerate(sums):
        for r in range(n_rings):
            for j, k in enumerate(lags):
                on[i, r, j] = np.trace(s.on[r].astype(float), offset=k)
                a = s.s_on[r].astype(float)
                den[i, r, j] = (a[: 80 - k] * a[k:]).sum() / s.cnt[r]
    binned = hits.g2()
    groups = _lag_bins(lags)
    assert len(groups) == binned.tau.size
    for b, group in enumerate(groups):
        np.testing.assert_allclose(
            binned.g2.values[:, b],
            ratio(on[..., group].sum((0, 2)), den[..., group].sum((0, 2))),
            rtol=1e-10,
        )
    assert sorted(i for g in groups for i in g.tolist()) == list(range(lags.size))


def test_every_hit_is_accounted_for_with_the_reason_it_was_not_used(world, tmp_path):
    rings, _, _, photons = world
    failed = {7: FrameStatus.MISSING_MODULES, 300: FrameStatus.LABEL_MISMATCH}
    hits = build_results(tmp_path / "cov.h5", photons, rings, failed=failed)
    coverage = hits.coverage()
    assert coverage.n_hits == len(IDS) + 2 and coverage.accounted
    assert coverage.by_status == {
        "OK": len(IDS),
        "MISSING_MODULES": 1,
        "LABEL_MISMATCH": 1,
    }
    assert coverage.by_pairing["NOT_OK"] == 2
    assert coverage.by_pairing["FIRST_HIT"] >= 1
    assert coverage.by_pairing["GAP_TOO_LARGE"] >= 1
    assert coverage.by_pairing["PAIRED"] >= 1
    assert coverage.n_usable == len(IDS)
    assert coverage.n_with_baseline == coverage.by_pairing["PAIRED"]
    assert "hits" in str(coverage)
    assert PairStatus.PAIRED.name in coverage.by_pairing


def test_the_period_follows_the_pulse_step(world, tmp_path):
    rings, _, _, photons = world
    slow = build_results(tmp_path / "slow.h5", photons, rings, pattern_step=4)
    assert slow.period_s == pytest.approx(4 / BASE_CLOCK_HZ)


def test_a_repetition_rate_that_disagrees_with_the_pattern_is_refused(results):
    results.rep_rate_mhz = 2.2569  # DAMNIT's figure for a step of 2
    assert results.period_s == pytest.approx(2 / BASE_CLOCK_HZ)
    results.rep_rate_mhz = 1.13
    with pytest.raises(RepRateMismatch, match="2.2569|MHz"):
        results.g2()
    results.rep_rate_mhz = None


def test_pulses_that_are_not_evenly_spaced_make_no_lag_axis(results):
    with h5py.File(results.path, "r+") as f:
        pulses = f["pattern/pulse_ids"][()]
        pulses[5] += 1
        f["pattern/pulse_ids"][...] = pulses
    with pytest.raises(NonUniformPattern):
        XpcsHits(results.path).g2()


def test_diagonals_have_the_shape_the_triage_reads(world, results):
    from analysis.triage.xpcs import TtcfDiagonals, jackknife_g2

    diagonals = results.diagonals()
    paired = results.has_off
    assert isinstance(diagonals, TtcfDiagonals)
    assert diagonals.train_ids.dtype == np.uint64
    assert diagonals.train_ids.tolist() == results.train_ids[paired].tolist()
    assert np.all(np.diff(diagonals.train_ids.astype(np.int64)) > 0)
    assert diagonals.roi_q_A.tolist() == pytest.approx(results.roi_q.tolist())
    assert diagonals.sums.shape == (paired.sum(), R, P - 1)
    assert diagonals.counts.dtype == np.int32 and diagonals.n_frames == P
    # one entry, computed by hand: train 101, ring 1, lag 3
    s = sums_of(world)[1]
    kbar = s.s_on[1] / s.cnt[1]
    norm = np.outer(kbar, kbar)
    raw = np.where(norm > 0, s.on[1] / (norm * s.cnt[1]), np.nan)
    cur, prev = s.s_off_cur[1] / s.cnt_pair[1], s.s_off_prev[1] / s.cnt_pair[1]
    nrm = np.outer(cur, prev)
    off = np.where(nrm > 0, s.off[1] / (nrm * s.cnt_pair[1]), np.nan)
    c2 = raw - 0.5 * (off + off.T)
    diag = np.diagonal(c2, 3)
    row = int(np.flatnonzero(diagonals.train_ids == 101)[0])
    assert diagonals.counts[row, 1, 2] == np.isfinite(diag).sum()
    assert diagonals.sums[row, 1, 2] == pytest.approx(np.nansum(diag))
    g2, sigma, _ = jackknife_g2(diagonals.sums[:, :, :5], diagonals.counts[:, :, :5], 2)
    assert g2.shape == (R, 5) and np.isfinite(g2).any()


def test_an_empty_hit_set_reads_as_a_result_with_nothing_in_it(tmp_path):
    rings = make_rings(np.random.default_rng(0), N_ENV, R)
    hits = build_results(tmp_path / "empty.h5", [], rings)
    assert hits.coverage().n_hits == 0 and hits.coverage().accounted
    assert hits.g2().tau.size == 0
    assert hits.ttcf()["raw"].shape == (R, 0, 0)
    assert hits.diagonals().train_ids.size == 0
    assert hits.kbar().shape == (0, 0, R)


def test_a_run_whose_every_hit_failed_pools_to_nothing_not_to_one(tmp_path):
    rings = make_rings(np.random.default_rng(0), N_ENV, R)
    photons = [make_train(1, make_dense(np.random.default_rng(1), P, N_ENV), [])]
    hits = build_results(
        tmp_path / "bad.h5", photons[:0] or [photons[0]], rings,
        failed={1: FrameStatus.NO_FRAMES},
    )  # fmt: skip
    assert hits.coverage().n_usable == 0
    g2 = hits.g2(binned=False)
    assert np.isnan(g2.g2.values).all()


def test_read_finds_a_runs_file_under_the_output_root(results, tmp_path):
    target = tmp_path / "root" / "xpcs_hits" / "r0423"
    target.mkdir(parents=True)
    (target / "xpcs_hits.h5").write_bytes(results.path.read_bytes())
    again = read(10400, 423, output_root=tmp_path / "root", rep_rate_mhz=2.26)
    assert again.train_ids.tolist() == results.train_ids.tolist()
    assert again.rep_rate_mhz == 2.26


def assert_same_datasets(a, b, **tolerance):
    assert set(a.data_vars) == set(b.data_vars) and a.attrs == b.attrs
    for name in a.data_vars:
        if tolerance:
            np.testing.assert_allclose(a[name].values, b[name].values, **tolerance)
        else:
            np.testing.assert_array_equal(a[name].values, b[name].values)
        for coord in a[name].coords:
            np.testing.assert_array_equal(a[coord].values, b[coord].values)


@pytest.mark.parametrize("binned", [True, False])
def test_g2_and_ttcf_from_precomputed_sums_are_those_read_on_their_own(
    long_world, binned
):
    hits, _, _ = long_world
    sums = hits.pooled_sums()
    assert_same_datasets(hits.g2(binned=binned, sums=sums), hits.g2(binned=binned))
    assert_same_datasets(hits.ttcf(sums=sums), hits.ttcf())


def test_precomputed_sums_are_refused_with_trains_or_for_extra_speckles_average(
    results,
):
    sums = results.pooled_sums()
    some = results.train_ids[:2]
    with pytest.raises(ValueError, match="not both"):
        results.g2(trains=some, sums=sums)
    with pytest.raises(ValueError, match="not both"):
        results.ttcf(trains=some, sums=sums)
    with pytest.raises(ValueError, match="pooled"):
        results.g2(Averaging.ES, sums=sums)
    with pytest.raises(ValueError, match="pooled"):
        results.ttcf(Averaging.ES, sums=sums)


@pytest.fixture
def many_rows(tmp_path):
    """More than two read chunks of rows, so a pool gets several slabs."""
    rng = np.random.default_rng(8)
    rings = make_rings(rng, 120, 2)
    photons = [
        make_train(t, make_dense(rng, 12, 120, 0.2), make_flags(rng, 12, 120))
        for t in range(1000, 1000 + 2 * CHUNK + 5)
    ]
    return build_results(tmp_path / "many.h5", photons, rings)


def test_sums_read_by_a_pool_of_slabs_pool_to_the_serial_ones(many_rows):
    pools = []

    def factory(n_workers):
        pools.append(n_workers)
        return InlinePool(n_workers)

    serial = many_rows.pooled_sums()
    parallel = many_rows.pooled_sums(n_workers=3, pool_factory=factory)
    assert pools == [3]
    for name in ("rows", "has_off", "on", "den_on", "off", "den_off"):
        np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))
    for name in ("raw_num", "pair_num", "off_num"):  # integer counts: exact
        np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))
    for name in ("raw_den", "pair_den", "off_den"):
        np.testing.assert_allclose(
            getattr(parallel, name), getattr(serial, name), rtol=1e-12
        )
    assert_same_datasets(many_rows.g2(sums=parallel), many_rows.g2(sums=serial))
    assert_same_datasets(
        many_rows.ttcf(sums=parallel), many_rows.ttcf(sums=serial), rtol=1e-12
    )
