"""extra-speckle's ``xpcs_offline`` figures and columns, reproduced from the hit store.

Each figure test reads the plotted arrays back and compares them with values
computed independently, from the stored sums, the dense photons or a detector
mean with a known answer. Titles and axis labels are not pinned.
"""

from __future__ import annotations

import re
from dataclasses import replace

import h5py
import numpy as np
import pytest
import xarray as xr
from xpcs_fakes import (
    GOOD,
    any_flagged,
    build_results,
    make_dense,
    make_flags,
    make_rings,
    make_train,
)

pytest.importorskip("damnit")

from damnit.context import Skip  # noqa: E402

from analysis.xpcs import damnit as xd  # noqa: E402
from analysis.xpcs import detector as D  # noqa: E402
from analysis.xpcs import figures as F  # noqa: E402
from analysis.xpcs import g2fit as G  # noqa: E402
from analysis.xpcs import legacy as L  # noqa: E402
from analysis.xpcs.reader import Averaging, RepRateMismatch, XpcsHits  # noqa: E402
from analysis.xpcs.store import RingPhotonStore  # noqa: E402

IDS = (100, 101, 104, 140, 141, 145)
N_ENV, P, R = 300, 14, 3


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


@pytest.fixture
def written(hits, tmp_path):
    return {"output_root": str(tmp_path)}


def image_axes(fig):
    return [ax for ax in fig.axes if ax.images]


def curves(ax, n: int | None = None) -> np.ndarray:
    """The y data of the first ``n`` lines of ``ax`` (all by default)."""
    return np.array([line.get_ydata() for line in ax.lines[:n]])


# ── data products ───────────────────────────────────────────────────────────────
def test_the_legacy_ttcf_is_raw_minus_off_per_train(hits):
    out = L.legacy_ttcf(hits)
    assert out.dims == ("trainId", "roi_q", "frame_1", "frame_2")
    assert out["trainId"].values.tolist() == list(IDS)
    own = hits.per_train()
    expected = own["TTCF_raw"].values - own["TTCF_off"].values
    np.testing.assert_array_equal(out.values, expected.astype(np.float32))
    has_baseline = np.isfinite(out.values).any(axis=(1, 2, 3))
    assert has_baseline.tolist() == hits.has_off.tolist()
    part = L.legacy_ttcf(hits, trains=np.array([104, 140]), chunk=1)
    np.testing.assert_array_equal(part.values, out.sel(trainId=[104, 140]).values)


def test_the_legacy_mean_dataset_holds_the_corrected_map_and_g2(hits):
    out = L.legacy_mean_dataset(hits)
    maps, g2 = hits.ttcf(Averaging.POOLED), hits.g2(Averaging.POOLED)
    np.testing.assert_allclose(
        out["ttcf_mean"].values, maps["c2"].values.astype(np.float32)
    )
    np.testing.assert_allclose(out["g2"].values, g2["c2"].values.astype(np.float32))
    np.testing.assert_allclose(
        out["g2_sigma"].values, g2["sigma"].values.astype(np.float32)
    )
    es = L.legacy_mean_dataset(hits, Averaging.ES)
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
    out = L.mean_intensity(hits)
    assert out["trainId"].values.tolist() == list(IDS)
    np.testing.assert_allclose(out.values, expected, rtol=1e-12)


def test_the_ring_saxs_pools_the_trains_per_frame(world, hits):
    rings, dense, flags, _ = world
    label = np.asarray(rings.label)
    out = L.saxs_rings(hits)
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
    gamma = np.asarray(diffusion) * q**2
    values = beta * np.exp(-2 * np.outer(gamma, tau)) + baseline
    values = values + noise * np.random.default_rng(8).standard_normal(values.shape)
    return xr.DataArray(values, dims=("roi_q", "tau"), coords={"roi_q": q, "tau": tau})


def test_the_fit_recovers_an_exponential_and_its_dispersion_relation():
    q = np.array([0.03, 0.04, 0.05, 0.06])
    fit = G.fit_g2(exponentials(q, noise=1e-5))
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
    fit = G.fit_g2(g2)
    assert np.isnan(fit.gamma[2]) and np.isnan(fit.curve[2]).all()
    np.testing.assert_allclose(fit.gamma[[0, 1, 3]], 1e8 * q[[0, 1, 3]] ** 2, rtol=3e-3)
    g2.values[:] = np.nan
    with pytest.raises(ValueError, match="enough finite"):
        G.fit_g2(g2)


def test_the_dispersion_weights_each_ring_by_its_error_as_extra_speckle_does():
    """``fit_dr``'s ``weightedaverage`` weights by the errors, not their inverses."""
    q = np.array([0.03, 0.04, 0.05, 0.06])
    diffusion = np.array([1e8, 3e8, 6e8, 2e8])
    fit = G.fit_g2(exponentials(q, diffusion, noise=1e-4))
    weights = fit.err_gamma / q**2
    np.testing.assert_allclose(
        fit.dr_a, np.average(diffusion, weights=weights), rtol=1e-2
    )
    assert abs(fit.dr_a - diffusion.mean()) > 0.3 * diffusion.mean()
    np.testing.assert_allclose(fit.dr_a_err, np.std(fit.gamma / q**2), rtol=1e-12)


def test_a_failed_fit_is_a_value_error_and_the_wrapper_skips(
    hits, written, monkeypatch
):
    flat = exponentials(np.array([0.03, 0.04]))
    g2 = xr.Dataset({"c2": flat, "sigma": flat})
    g2["c2"].values[:] = np.nan
    with pytest.raises(ValueError):
        F.figure_g2_fit(hits, g2=g2)

    def refuse(*_args, **_kwargs):
        raise ValueError("the g2 fit did not converge")

    monkeypatch.setattr(F, "fit_g2", refuse)
    with pytest.raises(Skip, match="did not converge"):
        xd.g2_fit_figure(10400, 423, **written)


# ── figures from the results file ────────────────────────────────────────────────
def test_the_pooled_correction_curves_are_the_unbinned_g2(hits):
    fig = F.figure_correction(hits)
    g2 = hits.g2(binned=False)
    panels = [ax for ax in fig.axes if ax.get_legend() is not None]
    for ax, name in zip(panels, ("g2", "baseline", "c2"), strict=True):
        np.testing.assert_allclose(curves(ax), g2[name].values, equal_nan=True)


def test_the_g2_errors_are_the_jackknife_sigma(hits):
    (ax,) = F.figure_g2(hits, errors=True).axes
    g2 = hits.g2()
    for ring, container in enumerate(ax.containers):
        (bars,) = container.lines[2]
        half = np.array([abs(s[1, 1] - s[0, 1]) / 2 for s in bars.get_segments()])
        sigma = g2["sigma"].values[ring]
        np.testing.assert_allclose(half, sigma[np.isfinite(sigma)], rtol=1e-6)


def test_each_results_wrapper_draws_the_view_it_names(written):
    hits = xd._open(10400, 423, None, written)
    kbar = hits.kbar().values
    trace = xd.kbar_figure(10400, 423, **written).axes[0]
    plotted = np.array([y for line in trace.lines for y in line.get_ydata()])
    np.testing.assert_allclose(
        np.sort(plotted[np.isfinite(plotted)]), np.sort(kbar[np.isfinite(kbar)])
    )
    threshold = [seg for c in trace.collections for seg in c.get_segments()]
    assert any(np.allclose(seg[:, 1], 3e-3) for seg in threshold)

    (g2_axes,) = xd.g2_figure(10400, 423, 2.2569, **written).axes
    g2 = hits.g2()
    np.testing.assert_allclose(g2_axes.lines[0].get_xdata(), g2["tau"].values)
    np.testing.assert_allclose(curves(g2_axes), g2["c2"].values, equal_nan=True)
    with pytest.raises(RepRateMismatch):
        xd.g2_figure(10400, 423, 1.13, **written)

    panels = image_axes(xd.ttcfs_figure(10400, 423, **written))
    assert len(panels) == R and all(p.images[0].get_clim()[0] == 0 for p in panels)
    np.testing.assert_allclose(
        panels[1].images[0].get_array().filled(np.nan),
        hits.ttcf()["c2"].values[1],
        equal_nan=True,
    )


def test_the_intensity_figure_marks_a_train_off_the_xgm_median_as_removed(hits):
    intensity = L.mean_intensity(hits)
    (ax,) = F.figure_intensity_outliers(hits).axes
    np.testing.assert_allclose(ax.lines[0].get_ydata(), intensity.values)
    xgm = xr.DataArray(
        np.full(intensity.size, 2.0),
        dims="trainId",
        coords={"trainId": intensity.trainId},
    )
    xgm.values[3] = 20.0
    (ax,) = F.figure_intensity_outliers(hits, xgm).axes
    good, removed = ax.collections
    shown = np.asarray(removed.get_offsets())
    assert np.isfinite(shown[:, 1]).sum() == 1 and np.isfinite(shown[3, 1])
    assert np.isnan(np.asarray(good.get_offsets())[3, 1])


def test_without_a_store_the_saxs_column_shows_the_ring_means(hits, written):
    (ax, _colourbar) = xd.saxs_figure(10400, 423, **written).axes
    rings = L.saxs_rings(hits)
    np.testing.assert_allclose(curves(ax, P), rings.values)
    for name in ("q_rings_figure", "beamcenter_figure", "outliers_figure"):
        with pytest.raises(Skip, match="photon store"):
            getattr(xd, name)(10400, 423, **written)


# ── detector means with a known answer ─────────────────────────────────────────
def synthetic_detector(rng, n=400, frames=3):
    """A detector mean with pixels on a q ramp."""
    q = np.linspace(0.02, 0.09, n)
    total = rng.poisson(400.0, (frames, n)).astype(np.float32)
    return D.DetectorMean(
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
    out = D.q_curves(det, dq_A=0.01)
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
    assert np.isnan(D.q_curves(det, dq_A=0.01, pixels=~removed).values[:, 2]).all()


def test_the_outlier_figure_leaves_out_the_pixels_flagged_in_every_train():
    """The pass never correlates such a pixel: neither curve panel includes it."""
    det = synthetic_detector(np.random.default_rng(2), n=2000)
    det.total[:, 700] *= 50
    always = np.zeros(det.pixel.size, bool)
    always[700] = True
    fig = F.figure_outliers(replace(det, always_flagged=always), dq_A=0.005)
    without = D.q_curves(det, dq_A=0.005, pixels=~always).values
    np.testing.assert_allclose(curves(fig.axes[0], 3), without, equal_nan=True)
    assert "1 flagged in every train not shown" in fig.get_suptitle()


def test_an_empty_q_bin_is_drawn_although_its_ratio_is_nan():
    """No photons in a bin, as in a run without beam, is 0/0 and not a flagged pixel."""
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
    np.testing.assert_allclose(
        curves(fig.axes[1], 3), D.q_curves(det, dq_A=0.005).values, equal_nan=True
    )


def test_the_cone_curves_split_the_detector_into_equal_azimuthal_sectors():
    det = synthetic_detector(np.random.default_rng(2), n=3200)
    cones = D.cone_curves(det, (0.02, 0.09), dq_A=0.02)
    assert cones.sizes["q_slice"] == D.N_CONES
    mean = det.mean_image
    chi = det.chi_deg % 360.0
    first = (chi >= 0) & (chi < 45) & (det.q_A < 0.04)
    np.testing.assert_allclose(cones.values[0, 0], mean[first].mean(), rtol=1e-6)


def dipole_detector(epsilon, n_chi=720, n_q=40):
    """Two rings of pixels on a regular (q, χ) grid, the mean ∝ 1 + ε cos χ."""
    chi = np.tile((np.arange(n_chi) + 0.5) * 360 / n_chi - 180, n_q)
    q = np.repeat(np.linspace(0.021, 0.059, n_q), n_chi)
    mean = 400 * (1 + epsilon * np.cos(np.radians(chi)))
    return D.DetectorMean(
        total=np.tile(mean.astype(np.float32), (3, 1)) * 4,
        n_trains=4,
        train_ids=np.arange(4),
        pixel=np.arange(chi.size),
        ring=(q >= 0.04).astype(np.int8),
        q_A=q,
        chi_deg=chi,
        ring_bounds_A=np.array([[0.02, 0.04], [0.04, 0.06]]),
        geometry=None,
        beam_center=None,
        run="7",
    )


@pytest.mark.parametrize("epsilon", [0.0, 0.05, 0.1])
def test_the_beamcenter_figure_shows_each_cones_deviation_and_the_merit(epsilon):
    fig = F.figure_beamcenter(dipole_detector(epsilon))
    lo = np.radians(45 * np.arange(D.N_CONES))
    cos_in_cone = (np.sin(lo + np.radians(45)) - np.sin(lo)) / np.radians(45)
    for line in fig.axes[1].lines[:2]:
        np.testing.assert_allclose(line.get_ydata(), epsilon * cos_in_cone, atol=1e-6)
    merit = float(re.search(r"cone merit (\S+) ", fig.get_suptitle()).group(1))
    expected = np.var(np.log1p(epsilon * cos_in_cone))
    assert merit == pytest.approx(expected, rel=5e-3, abs=1e-12)


def test_the_ring_image_without_an_applied_mask_blanks_the_recomputed_outliers(geom):
    n = 4000
    total = np.full((3, n), 40.0, np.float32)
    total[:, 100] *= 30  # hot
    total[:, 1500] = 0  # dead
    det = D.DetectorMean(
        total=total,
        n_trains=4,
        train_ids=np.arange(4),
        pixel=np.arange(n) * 97,
        ring=(np.arange(n) * 2 // n).astype(np.int8),
        q_A=np.linspace(0.02, 0.06, n),
        chi_deg=np.linspace(-179, 179, n),
        ring_bounds_A=np.array([[0.02, 0.04], [0.04, 0.06]]),
        geometry=geom,
        beam_center=None,
        run="7",
    )
    shown = image_axes(F.figure_q_rings(det))[0].images[0].get_array().filled(np.nan)
    finite = shown[np.isfinite(shown)]
    assert finite.size == n - 2
    np.testing.assert_allclose(np.sort(finite), np.full(n - 2, 10.0), rtol=1e-6)


# ── what needs the photon store ──────────────────────────────────────────────────
@pytest.fixture
def stored(pipeline):
    """The mock run's results and the config whose store holds its photons."""
    return XpcsHits(pipeline.finished()), pipeline.cfg(), pipeline


def test_the_detector_mean_is_the_photons_of_the_ring_pixels_of_the_store(stored):
    hits, cfg, pipe = stored
    det = D.detector_mean(hits, cfg, geometry=pipe.geom)
    assert det.n_trains == len(GOOD) and det.n_frames == hits.n_frames
    with h5py.File(hits.path, "r") as f:
        label = f["rings/label"][()]
    good = np.flatnonzero(label >= 0)
    expected = 0
    with RingPhotonStore.open(cfg.store_file) as store:
        n_env = store.envelope().size
        for train in GOOD:
            got = store.get(train)
            expected += int(got.count[label[got.pixel] >= 0].sum())
        every = np.logical_and.reduce(
            [store.get(t).flagged_any(n_env)[good] for t in GOOD]
        )
    assert det.total.sum() == expected
    assert (det.ring >= 0).all() and det.pixel.size == good.size
    np.testing.assert_array_equal(det.always_flagged, every)

    three = D.detector_mean(hits, cfg, max_trains=3, geometry=pipe.geom)
    assert three.train_ids.tolist() == [GOOD[0], GOOD[3], GOOD[-1]]

    image = image_axes(F.figure_q_rings(det, remove_outliers=False))[0]
    assert np.isfinite(image.images[0].get_array()).sum() == det.pixel.size


def test_a_missing_store_or_one_of_another_envelope_is_refused(stored, tmp_path):
    hits, cfg, pipe = stored
    with pytest.raises(FileNotFoundError, match="photon store"):
        D.detector_mean(hits, replace(cfg, output_root=str(tmp_path / "empty")))
    hits.attrs["envelope_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="another envelope"):
        D.detector_mean(hits, cfg, geometry=pipe.geom)


def test_the_detector_columns_draw_from_the_store(stored, monkeypatch):
    hits, cfg, pipe = stored
    monkeypatch.setattr(
        "analysis.saxs.operator.geometry_from_config", lambda _cfg: pipe.geom
    )
    out = {"output_root": cfg.output_root}
    saxs = xd.legacy_saxs(10400, 423, **out)
    assert saxs.attrs["n_trains"] == len(GOOD)
    (ax, _colourbar) = xd.saxs_figure(10400, 423, **out).axes
    np.testing.assert_allclose(curves(ax, hits.n_frames), saxs.values, equal_nan=True)
    xd.q_rings_figure(10400, 423, 3, **out)
    xd.beamcenter_figure(10400, 423, **out)
    xd.outliers_figure(10400, 423, **out)
