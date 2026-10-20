"""The old ``xpcs_offline`` figures and columns, reproduced from the hit store.

What is pinned is what a reader of the figure or the column sees: the numbers
(against the sums and the stored photons, independently recomputed), the
extra-speckle conventions (titles, axis labels, log scales, one curve per ring)
and the failure semantics (a missing store is a ``Skip``, a failed fit is
"no figure").
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure
from xpcs_fakes import (
    any_flagged,
    build_results,
    make_dense,
    make_flags,
    make_rings,
    make_train,
)

pytest.importorskip("damnit")

from damnit.context import Skip  # noqa: E402

from analysis.common.writer import IncompleteRun  # noqa: E402
from analysis.xpcs import damnit as xd  # noqa: E402
from analysis.xpcs import figures as F  # noqa: E402
from analysis.xpcs.result import Averaging, XpcsHits  # noqa: E402
from analysis.xpcs.store import RingPhotonStore  # noqa: E402

IDS = (100, 101, 104, 140, 141, 145)
N_ENV, P, R = 300, 14, 3
GOOD = (10000, 10001, 10003, 10005, 10006, 10007, 10009)


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(11)
    rings = make_rings(rng, N_ENV, R)
    dense = [make_dense(rng, P, N_ENV, 0.12) for _ in IDS]
    flags = [make_flags(rng, P, N_ENV) for _ in IDS]
    photons = [make_train(t, d, f) for t, d, f in zip(IDS, dense, flags, strict=True)]
    return rings, dense, flags, photons


@pytest.fixture
def hits(world, tmp_path) -> XpcsHits:
    rings, _, _, photons = world
    path = tmp_path / "xpcs_hits" / "r0423" / "xpcs_hits.h5"
    path.parent.mkdir(parents=True)
    return build_results(path, photons, rings)


def texts(fig: Figure, kind: str = "title") -> list[str]:
    return [getattr(ax, f"get_{kind}")() for ax in fig.axes]


def image_axes(fig: Figure):
    return [ax for ax in fig.axes if ax.images]


# ── data products ───────────────────────────────────────────────────────────────
def test_the_legacy_ttcf_is_raw_minus_off_per_train(hits):
    out = F.legacy_ttcf(hits)
    assert out.dims == ("trainId", "roi_q", "frame_1", "frame_2")
    assert out.dtype == np.float32 and out.shape == (len(IDS), R, P, P)
    assert out["trainId"].values.tolist() == list(IDS)
    own = hits.per_train()
    expected = own["TTCF_raw"].values - own["TTCF_off"].values
    np.testing.assert_array_equal(out.values, expected.astype(np.float32))
    assert not hits.has_off[0] and not hits.has_off[3], "first train, then a gap"
    has_baseline = np.isfinite(out.values).any(axis=(1, 2, 3))
    assert has_baseline.tolist() == hits.has_off.tolist()


def test_the_legacy_ttcf_can_be_taken_a_part_at_a_time(hits):
    part = F.legacy_ttcf(hits, trains=np.array([104, 140]), chunk=1)
    whole = F.legacy_ttcf(hits)
    assert part["trainId"].values.tolist() == [104, 140]
    np.testing.assert_array_equal(part.values, whole.sel(trainId=[104, 140]).values)


def test_the_legacy_mean_dataset_holds_the_corrected_map_and_g2(hits):
    out = F.legacy_mean_dataset(hits)
    maps, g2 = hits.ttcf(Averaging.POOLED), hits.g2(Averaging.POOLED)
    np.testing.assert_allclose(
        out["ttcf_mean"].values, maps["c2"].values.astype(np.float32)
    )
    np.testing.assert_allclose(out["g2"].values, g2["c2"].values.astype(np.float32))
    assert out["g2"].dims == ("roi_q", "tau")
    assert out["g2_sigma"].shape == out["g2"].shape
    es = F.legacy_mean_dataset(hits, Averaging.ES)
    np.testing.assert_allclose(
        es["g2"].values,
        hits.g2(Averaging.ES)["g2"].values.astype(np.float32),
        equal_nan=True,
    )
    assert "g2_sigma" not in es


def test_the_mean_intensity_is_photons_per_good_ring_pixel_and_frame(world, hits):
    rings, dense, flags, _ = world
    label = np.asarray(rings.label)
    expected = []
    for d, f in zip(dense, flags, strict=True):
        good = (label >= 0) & ~any_flagged(f, N_ENV)
        expected.append(d[:, good].sum() / (good.sum() * P))
    out = F.mean_intensity(hits)
    assert out["trainId"].values.tolist() == list(IDS)
    np.testing.assert_allclose(out.values, expected, rtol=1e-12)


def test_the_ring_saxs_pools_the_trains_per_frame(world, hits):
    rings, dense, flags, _ = world
    label = np.asarray(rings.label)
    out = F.saxs_rings(hits)
    assert out.dims == ("frame", "q")
    for r in range(R):
        num = np.zeros(P)
        den = 0
        for d, f in zip(dense, flags, strict=True):
            good = (label == r) & ~any_flagged(f, N_ENV)
            num += d[:, good].sum(1)
            den += good.sum()
        np.testing.assert_allclose(out.values[:, r], num / den, rtol=1e-12)
    np.testing.assert_allclose(out["q"].values, hits.roi_q)


# ── the fit ──────────────────────────────────────────────────────────────────────
def exponentials(q, diffusion=1e8, beta=0.04, baseline=0.002, noise=0.0):
    tau = np.geomspace(4e-7, 6e-5, 30)
    gamma = diffusion * q**2
    values = beta * np.exp(-2 * np.outer(gamma, tau)) + baseline
    values = values + noise * np.random.default_rng(8).standard_normal(values.shape)
    return xr.DataArray(values, dims=("roi_q", "tau"), coords={"roi_q": q, "tau": tau})


def test_the_fit_recovers_an_exponential_and_its_dispersion_relation():
    q = np.array([0.03, 0.04, 0.05, 0.06])
    fit = F.fit_g2(exponentials(q, noise=1e-5))
    np.testing.assert_allclose(fit.gamma, 1e8 * q**2, rtol=3e-3)
    np.testing.assert_allclose(fit.beta, 0.04, rtol=3e-3)
    np.testing.assert_allclose(fit.baseline, 0.002, atol=2e-5)
    np.testing.assert_allclose(fit.dr_a, 1e8, rtol=3e-3)
    kb_t, eta = 1.381e-23 * 298.15, 1e-23
    np.testing.assert_allclose(
        fit.rh_A, kb_t / (6 * np.pi * eta * fit.dr_a) * 1e10, rtol=1e-12
    )
    np.testing.assert_allclose(fit.curve, exponentials(q).values, atol=1e-4)


def test_a_ring_without_data_is_left_out_of_the_fit_and_nan():
    q = np.array([0.03, 0.04, 0.05, 0.06])
    g2 = exponentials(q, noise=1e-5)
    g2.values[2] = np.nan
    fit = F.fit_g2(g2)
    assert np.isnan(fit.gamma[2]) and np.isnan(fit.curve[2]).all()
    np.testing.assert_allclose(fit.gamma[[0, 1, 3]], 1e8 * q[[0, 1, 3]] ** 2, rtol=3e-3)


def test_a_fit_with_nothing_to_fit_is_a_value_error():
    g2 = exponentials(np.array([0.03, 0.04]))
    g2.values[:] = np.nan
    with pytest.raises(ValueError, match="enough finite"):
        F.fit_g2(g2)


# ── figures from the results file ────────────────────────────────────────────────
@pytest.mark.parametrize("averaging", [Averaging.POOLED, Averaging.ES])
def test_the_correction_figure_has_extra_speckles_layout(hits, averaging):
    fig = F.figure_correction(hits, averaging)
    assert len(image_axes(fig)) == 3
    titles = texts(fig)
    assert titles[:3] == [
        f"(raw), {np.around(hits.roi_q[R // 2], 3)} $\\AA^{{-1}}$",
        f"(off), {np.around(hits.roi_q[R // 2], 3)} $\\AA^{{-1}}$",
        f"(raw-off), {np.around(hits.roi_q[R // 2], 3)} $\\AA^{{-1}}$",
    ]
    curves = [ax for ax in fig.axes if ax.get_legend() is not None]
    assert [ax.get_title() for ax in curves] == [
        "(raw)",
        "baseline estimation",
        "corrected",
    ]
    for ax in curves:
        assert ax.get_xscale() == "log" and ax.get_xlabel() == r"$\tau,  s$"
        assert len(ax.get_legend().get_texts()) == R


def test_the_pooled_correction_curves_are_the_unbinned_g2(hits):
    fig = F.figure_correction(hits)
    g2 = hits.g2(binned=False)
    curves = [ax for ax in fig.axes if ax.get_legend() is not None]
    shown = [np.array([line.get_ydata() for line in ax.lines]) for ax in curves]
    for ax_values, name in zip(shown, ("g2", "baseline", "c2"), strict=True):
        np.testing.assert_allclose(ax_values, g2[name].values, equal_nan=True)


def test_the_ttcf_figure_has_a_panel_per_ring_with_zero_floor(hits):
    fig = F.figure_ttcfs(hits)
    panels = image_axes(fig)
    assert len(panels) == R
    assert all(p.images[0].get_clim()[0] == 0 for p in panels)
    assert all(p.get_title().startswith("TTCF run 423, ") for p in panels)
    np.testing.assert_allclose(
        panels[1].images[0].get_array().filled(np.nan),
        hits.ttcf()["c2"].values[1],
        equal_nan=True,
    )


def test_the_g2_figure_has_a_curve_per_ring_on_a_log_time_axis(hits):
    fig = F.figure_g2(hits)
    (ax,) = fig.axes
    assert ax.get_title() == "run 423"
    assert ax.get_xscale() == "log"
    assert ax.get_xlabel() == r"$\tau,  s$"
    assert len(ax.get_legend().get_texts()) == R
    g2 = hits.g2()
    np.testing.assert_allclose(ax.lines[0].get_xdata(), g2["tau"].values)
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), g2["c2"].values[0], equal_nan=True
    )


def test_the_g2_figure_can_carry_the_jackknife_errors(hits):
    fig = F.figure_g2(hits, errors=True)
    assert fig.axes[0].containers, "error bars are drawn"


def test_the_fit_figure_has_data_beta_and_dispersion_panels(hits):
    q = np.array([0.03, 0.04, 0.05])
    g2 = xr.Dataset({"c2": exponentials(q), "sigma": exponentials(q) * 0.01})
    fig = F.figure_g2_fit(hits, g2=g2)
    data, beta, dispersion = fig.axes
    assert data.get_title() == "(fitted) run 423"
    assert beta.get_ylabel() == r"$\beta $"
    assert dispersion.get_xlabel() == r"$q^2, \mathrm{\AA}^{-2}$"
    assert "R_H" in dispersion.get_legend().get_texts()[0].get_text()


def test_a_failed_fit_is_a_value_error_not_a_figure(hits):
    flat = exponentials(np.array([0.03, 0.04]))
    g2 = xr.Dataset({"c2": flat, "sigma": flat})
    g2["c2"].values[:] = np.nan
    with pytest.raises(ValueError):
        F.figure_g2_fit(hits, g2=g2)


def test_the_kbar_figure_draws_the_threshold_and_a_histogram_per_ring(hits):
    fig = F.figure_kbar(hits, threshold=0.05)
    trace, histogram = fig.axes
    assert trace.get_yscale() == "log" and histogram.get_xscale() == "log"
    assert len(histogram.get_legend().get_texts()) == R
    lines = [seg for c in trace.collections for seg in c.get_segments()]
    assert any(np.allclose(seg[:, 1], 0.05) for seg in lines)
    kbar = hits.kbar()
    plotted = np.concatenate([line.get_ydata() for line in trace.lines])
    np.testing.assert_allclose(
        np.sort(plotted[np.isfinite(plotted)]),
        np.sort(kbar.values[np.isfinite(kbar.values)]),
    )


def test_the_intensity_figure_is_one_dot_per_hit_train(hits):
    fig = F.figure_intensity_outliers(hits)
    (ax,) = fig.axes
    assert ax.get_title() == "Run 423"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), F.mean_intensity(hits).values)


def test_with_the_xgm_a_train_off_the_median_is_drawn_as_removed(hits):
    intensity = F.mean_intensity(hits)
    xgm = xr.DataArray(
        np.full(intensity.size, 2.0),
        dims="trainId",
        coords={"trainId": intensity.trainId},
    )
    xgm.values[3] = 20.0
    fig = F.figure_intensity_outliers(hits, xgm)
    (ax,) = fig.axes
    good, removed = ax.collections
    shown = np.asarray(removed.get_offsets())
    assert np.isfinite(shown[:, 1]).sum() == 1 and np.isfinite(shown[3, 1])
    assert np.isnan(np.asarray(good.get_offsets())[3, 1])


def test_the_ring_saxs_figure_says_it_is_ring_resolved(hits):
    fig = F.figure_saxs(hits)
    (curves, _colourbar) = fig.axes
    assert curves.get_title() == "I(q) per pulseId (rings)"
    assert curves.get_xscale() == "log" and curves.get_yscale() == "log"
    assert len(curves.lines) == P + 1, "a curve per frame and the threshold"


# ── what needs the photon store ──────────────────────────────────────────────────
def synthetic_detector(rng, n=400, frames=3):
    """A detector mean with pixels on a q ramp and a known hot pixel."""
    q = np.linspace(0.02, 0.09, n)
    total = rng.poisson(400.0, (frames, n)).astype(np.float32)
    return F.DetectorMean(
        total=total,
        n_trains=4,
        train_ids=np.arange(4),
        pixel=np.arange(n),
        ring=(np.arange(n) * 3 // n).astype(np.int8),
        q_A=q,
        chi_deg=np.linspace(-179, 179, n),
        ring_bounds_A=np.array([[0.02, 0.04], [0.04, 0.06], [0.06, 0.09]]),
        geometry=None,
        beam_center=None,
        run="7",
    )


def test_the_q_curve_is_the_mean_of_the_pixels_in_each_bin():
    det = synthetic_detector(np.random.default_rng(0))
    out = F.q_curves(det, dq_A=0.01)
    edges = np.arange(det.q_A.min(), det.q_A.max() + 0.01, 0.01)
    index = np.digitize(det.q_A, edges) - 1
    per_frame = det.per_frame()
    for b in range(out.sizes["q"]):
        inside = index == b
        np.testing.assert_allclose(
            out.values[:, b], per_frame[:, inside].mean(1), rtol=1e-6
        )
    removed = np.zeros(det.pixel.size, bool)
    removed[index == 2] = True
    assert np.isnan(F.q_curves(det, dq_A=0.01, pixels=~removed).values[:, 2]).all()


def test_the_outlier_statistic_flags_a_hot_and_a_dead_pixel_only():
    det = synthetic_detector(np.random.default_rng(1), n=2000)
    det.total[:, 100] *= 30
    det.total[:, 1500] = 0
    ratio = F.outlier_ratio(det, dq_A=0.005)
    flagged = np.flatnonzero((ratio < 0.75) | (ratio > 1.75))
    assert {100, 1500} <= set(flagged.tolist())
    assert flagged.size <= 4, "Poisson noise on 4 trains x 3 frames stays inside"


def test_the_outlier_figure_leaves_out_the_pixels_flagged_in_every_train():
    """The pass never correlates such a pixel: neither panel shows it."""
    from dataclasses import replace

    det = synthetic_detector(np.random.default_rng(2), n=2000)
    det.total[:, 700] *= 50
    always = np.zeros(det.pixel.size, bool)
    always[700] = True
    fig = F.figure_outliers(replace(det, always_flagged=always), dq_A=0.005)
    bin_of_700 = int(
        np.argmin(np.abs(F.q_curves(det, dq_A=0.005)["q"].values - det.q_A[700]))
    )
    for panel in fig.axes[:2]:
        peak = max(line.get_ydata()[bin_of_700] for line in panel.lines)
        assert peak < 2 * np.median(det.per_frame()), "the hot pixel is not drawn"
    assert "1 flagged in every train not shown" in fig._suptitle.get_text()


def test_an_empty_q_bin_is_drawn_although_its_ratio_is_nan():
    """No photons in a bin is 0/0, not a flagged pixel (r1, no beam)."""
    from dataclasses import replace

    det = synthetic_detector(np.random.default_rng(3), n=400)
    det.total[:] = 0
    det.total[:, ::7] = 1
    applied = replace(
        det,
        always_flagged=np.zeros(det.pixel.size, bool),
        outlier=np.zeros(det.pixel.size, bool),
        outlier_ratio=np.full(det.pixel.size, np.nan, np.float32),
        outlier_thresholds=(0.75, 1.75),
        outlier_trains=4,
    )
    fig = F.figure_outliers(applied, dq_A=0.005)
    assert "0 flagged in every train not shown" in fig._suptitle.get_text()


def test_the_cone_curves_split_the_detector_into_equal_azimuthal_sectors():
    det = synthetic_detector(np.random.default_rng(2), n=3200)
    cones = F.cone_curves(det, (0.02, 0.09), dq_A=0.02)
    assert cones.dims == ("q_slice", "q") and cones.sizes["q_slice"] == F.N_CONES
    mean = det.mean_image
    chi = det.chi_deg % 360.0
    first = (chi >= 0) & (chi < 45) & (det.q_A < 0.04)
    np.testing.assert_allclose(cones.values[0, 0], mean[first].mean(), rtol=1e-6)


@pytest.fixture
def stored(pipeline):
    """The mock run's results and the config whose store holds its photons."""
    path = pipeline.run(GOOD)
    return XpcsHits(path), pipeline.cfg(), pipeline


def test_the_detector_mean_is_the_photons_of_the_ring_pixels_of_the_store(stored):
    hits, cfg, pipe = stored
    det = F.detector_mean(hits, cfg, geometry=pipe.geom)
    assert det.n_trains == len(GOOD) and det.n_frames == hits.n_frames
    import h5py

    with h5py.File(hits.path, "r") as f:
        label = f["rings/label"][()]
    expected = 0
    with RingPhotonStore.open(cfg.store_file) as store:
        for train in GOOD:
            got = store.get(train)
            expected += int(got.count[label[got.pixel] >= 0].sum())
    assert det.total.sum() == expected
    assert (det.ring >= 0).all() and det.pixel.size == int((label >= 0).sum())


def test_the_detector_mean_knows_the_pixels_flagged_in_every_train(stored):
    hits, cfg, pipe = stored
    det = F.detector_mean(hits, cfg, geometry=pipe.geom)
    import h5py

    with h5py.File(hits.path, "r") as f:
        good = np.flatnonzero(f["rings/label"][()] >= 0)
    with RingPhotonStore.open(cfg.store_file) as store:
        n_env = store.envelope().size
        every = np.logical_and.reduce(
            [store.get(t).flagged_any(n_env)[good] for t in det.train_ids]
        )
    np.testing.assert_array_equal(det.always_flagged, every)


def test_the_detector_mean_takes_evenly_spread_trains(stored):
    hits, cfg, pipe = stored
    det = F.detector_mean(hits, cfg, max_trains=3, geometry=pipe.geom)
    assert det.n_trains == 3
    assert det.train_ids.tolist() == [GOOD[0], GOOD[3], GOOD[-1]]


def test_a_store_of_another_envelope_is_refused(stored):
    hits, cfg, pipe = stored
    hits.attrs["envelope_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another envelope"):
        F.detector_mean(hits, cfg, geometry=pipe.geom)


def test_no_store_is_a_file_not_found(stored, tmp_path):
    from dataclasses import replace

    hits, cfg, pipe = stored
    with pytest.raises(FileNotFoundError, match="photon store"):
        F.detector_mean(hits, replace(cfg, output_root=str(tmp_path / "empty")))


def test_the_detector_figures_draw_from_the_stored_photons(stored):
    hits, cfg, pipe = stored
    det = F.detector_mean(hits, cfg, geometry=pipe.geom)
    rings = F.figure_q_rings(det, remove_outliers=False)
    left, right = image_axes(rings)
    assert left.get_title() == "Mean detector image"
    assert right.get_title() == f"Rois in {R - 1} bins"
    assert left.xaxis_inverted(), "front view"
    shown = left.images[0].get_array()
    assert np.isfinite(shown).sum() == det.pixel.size

    outliers = F.figure_outliers(det)
    assert texts(outliers)[:2] == ["before outlier removal", "after outlier removal"]
    assert outliers.axes[2].get_xscale() == "log"

    centre = F.figure_beamcenter(det)
    assert centre.axes[0].get_title().startswith("At beam center")
    assert len(centre.axes[0].lines) == F.N_CONES

    saxs = F.figure_saxs(hits, det)
    assert saxs.axes[0].get_title() == "I(q) per pulseId (q bins)"


# ── the DAMNIT wrappers ──────────────────────────────────────────────────────────
@pytest.fixture
def written(hits, tmp_path):
    return {"output_root": str(tmp_path)}


@pytest.mark.parametrize(
    "name",
    [
        "kbar_figure",
        "correction_figure",
        "ttcfs_figure",
        "g2_figure",
        "intensity_outliers_figure",
        "saxs_figure",
    ],
)
def test_a_results_figure_wrapper_returns_a_figure(written, name):
    assert isinstance(getattr(xd, name)(10400, 423, **written), Figure)


def test_the_g2_wrapper_checks_the_repetition_rate(written):
    from analysis.xpcs.result import RepRateMismatch

    assert isinstance(xd.g2_figure(10400, 423, 2.2569, **written), Figure)
    with pytest.raises(RepRateMismatch):
        xd.g2_figure(10400, 423, 1.13, **written)


def test_the_saxs_wrapper_falls_back_to_the_rings_without_a_store(written):
    fig = xd.saxs_figure(10400, 423, **written)
    assert fig.axes[0].get_title().endswith("(rings)")


@pytest.mark.parametrize(
    "name", ["q_rings_figure", "beamcenter_figure", "outliers_figure"]
)
def test_the_detector_wrappers_skip_without_the_photon_store(written, name):
    with pytest.raises(Skip, match="photon store"):
        getattr(xd, name)(10400, 423, **written)


def test_the_detector_wrappers_draw_from_the_store(stored, tmp_path, monkeypatch):
    hits, cfg, pipe = stored
    monkeypatch.setattr(
        "analysis.saxs.operator.geometry_from_config", lambda _cfg: pipe.geom
    )
    out = {"output_root": cfg.output_root}
    assert isinstance(xd.q_rings_figure(10400, 423, 3, **out), Figure)
    assert isinstance(xd.beamcenter_figure(10400, 423, **out), Figure)
    assert isinstance(xd.outliers_figure(10400, 423, **out), Figure)
    saxs = xd.legacy_saxs(10400, 423, **out)
    assert saxs.dims == ("frame", "q") and saxs.attrs["resolution"] == "q bins"
    assert xd.saxs_figure(10400, 423, **out).axes[0].get_title().endswith("(q bins)")


def test_a_failed_fit_makes_the_wrapper_skip(written, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise ValueError("the g2 fit did not converge")

    monkeypatch.setattr(F, "fit_g2", refuse)
    with pytest.raises(Skip, match="did not converge"):
        xd.g2_fit_figure(10400, 423, **written)


def test_the_fit_wrapper_returns_a_figure_when_the_fit_works(written, monkeypatch):
    q = np.array([0.03, 0.04, 0.05])
    good = xr.Dataset({"c2": exponentials(q), "sigma": exponentials(q)})
    monkeypatch.setattr(XpcsHits, "g2", lambda self, *a, **k: good)
    assert isinstance(xd.g2_fit_figure(10400, 423, **written), Figure)


def test_the_legacy_wrappers_return_the_old_columns(written):
    ttcf = xd.legacy_ttcf(10400, 423, **written)
    assert ttcf.dims == ("trainId", "roi_q", "frame_1", "frame_2")
    mean = xd.legacy_mean_dataset(10400, 423, 2.2569, **written)
    assert set(mean.data_vars) >= {"ttcf_mean", "g2"}


def test_the_wrappers_skip_a_missing_file_and_refuse_an_unfinished_one(tmp_path):
    empty = {"output_root": str(tmp_path)}
    for name in ("kbar_figure", "correction_figure", "legacy_ttcf"):
        with pytest.raises(Skip, match="xpcs_hits_preprocessing"):
            getattr(xd, name)(10400, 423, **empty)
    from analysis.common.status import FrameStatus

    rng = np.random.default_rng(3)
    rings = make_rings(rng, N_ENV, R)
    photons = [
        make_train(t, make_dense(rng, P, N_ENV, 0.12), make_flags(rng, P, N_ENV))
        for t in IDS
    ]
    path = tmp_path / "xpcs_hits" / "r0423" / "xpcs_hits.h5"
    path.parent.mkdir(parents=True)
    build_results(path, photons, rings, failed={999: FrameStatus.NOT_PROCESSED})
    with pytest.raises(IncompleteRun):
        xd.kbar_figure(10400, 423, **empty)
