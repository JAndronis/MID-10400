"""Per-train flux, SVD, χ² against the median, XGM scatter and cell correlation.

Each synthetic run has a known truth: a flux that varies train to train, a shape
that does or does not change, and Gaussian noise of exactly the stored σ.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from diagnostics_fakes import FIRST_TRAIN, write_damnit_flux, write_frame_table

from analysis.diagnostics.curves import (
    band_mean,
    cell_xgm_correlation,
    chi2_vs_median,
    detrend,
    per_train_flux,
    svd,
    xgm_scatter,
)
from analysis.diagnostics.frames import frame_scalar, per_train

N_TRAINS, NPT = 400, 60
Q = np.linspace(0.1, 1.0, NPT)


def flux_array(n=N_TRAINS, seed=1):
    rng = np.random.default_rng(seed)
    values = 1000.0 * (1 + 0.1 * rng.standard_normal(n))
    return xr.DataArray(
        values, dims="trainId",
        coords={"trainId": FIRST_TRAIN + np.arange(n, dtype=np.uint64)}, name="flux",
    )  # fmt: skip


def run_file(tmp_path, flux, *, shape_change=0.0, noise=0.01, outlier=None, seed=2):
    """One frame per train: I_t(q) = F_t·(S0 + b_t·S1) + noise, with σ stored."""
    rng = np.random.default_rng(seed)
    s0 = 1.0 / (1 + (Q / 0.3) ** 2)
    s1 = np.sin(np.pi * Q) * s0
    b = shape_change * np.linspace(-1, 1, flux.size)
    truth = flux.values[:, None] * (s0[None] + b[:, None] * s1[None])
    sigma = noise * truth
    intensity = truth + sigma * rng.standard_normal(truth.shape)
    if outlier is not None:
        intensity[outlier] *= 1 + 0.2 * Q
    path = write_frame_table(
        tmp_path / "run.h5", "agipd", intensity[:, None, :].astype(np.float32),
        variance=(sigma**2)[:, None, :].astype(np.float32), q=Q,
    )  # fmt: skip
    return path, b


def test_per_train_flux_joins_on_train_id(tmp_path):
    xgm = {FIRST_TRAIN + i: 500.0 + i for i in range(5)}
    trans = {FIRST_TRAIN + i: 0.5 for i in range(1, 7)}  # starts one train later
    path = write_damnit_flux(tmp_path / "p10400_r1.h5", xgm, trans)
    ds = per_train_flux(path)
    assert ds.sizes["trainId"] == 7
    assert np.isnan(ds.flux.sel(trainId=FIRST_TRAIN))  # no transmission
    assert np.isnan(ds.flux.sel(trainId=FIRST_TRAIN + 6))  # no XGM
    assert ds.flux.sel(trainId=FIRST_TRAIN + 2) == pytest.approx(0.5 * 502.0)


def test_band_mean_skips_trains_without_frames(tmp_path):
    flux = flux_array(n=4)
    path, _ = run_file(tmp_path, flux)
    ds = per_train(path)
    ds["n_frames"][1] = 0
    band = band_mean(ds, (0.2, 0.4))
    keep = (Q >= 0.2) & (Q < 0.4)
    assert band[0] == pytest.approx(float(ds.intensity[0, keep].mean()))
    assert np.isnan(band[1])
    with pytest.raises(ValueError, match="no q bin"):
        band_mean(ds, (5, 6))


def test_svd_finds_one_shape_once_the_flux_is_divided_out(tmp_path):
    flux = flux_array()
    path, _ = run_file(tmp_path, flux)
    result = svd(per_train(path), k=4, flux=flux)
    assert result.attrs["n_above_noise"] == 1
    assert result.attrs["normalised_by"].startswith("per-train XGM")
    weight = result.weight.sel(component=0)
    assert float(weight.std() / weight.mean()) < 0.01  # the flux is gone
    # without the flux, the level still lives in component 0
    raw = svd(per_train(path), k=4)
    assert (
        float(raw.weight.sel(component=0).std() / raw.weight.sel(component=0).mean())
        > 0.05
    )


def test_svd_finds_a_changing_shape(tmp_path):
    flux = flux_array()
    path, b = run_file(tmp_path, flux, shape_change=0.05)
    result = svd(per_train(path), k=4, flux=flux)
    assert result.attrs["n_above_noise"] == 2
    r = np.corrcoef(result.weight.sel(component=1), b)[0, 1]
    assert abs(r) > 0.95
    assert result.explained.sum() <= 1 + 1e-12


def test_svd_refuses_an_empty_matrix(tmp_path):
    flux = flux_array(n=4)
    path, _ = run_file(tmp_path, flux)
    with pytest.raises(ValueError, match="nothing to decompose"):
        svd(per_train(path), q_range=(5.0, 6.0))


def test_chi2_is_one_for_pure_noise_and_flags_an_outlier(tmp_path):
    flux = flux_array()
    path, _ = run_file(tmp_path, flux, outlier=7)
    ds = per_train(path)
    chi2 = chi2_vs_median(ds, flux=flux)
    others = chi2.chi2.drop_sel(trainId=FIRST_TRAIN + 7)
    # mean of 399 reduced χ² of 60 bins each: 1 ± sqrt(2/(60·399)) = 1 ± 0.009
    assert float(others.mean()) == pytest.approx(1.0, abs=0.04)
    assert float(chi2.chi2.sel(trainId=FIRST_TRAIN + 7)) > 50
    assert (chi2.n_bins == NPT).all()
    # without dividing by a 10 % flux, the flux scatter dominates χ²
    assert float(chi2_vs_median(ds).chi2.median()) > 20


def test_chi2_needs_sigma(tmp_path):
    flux = flux_array(n=4)
    path, _ = run_file(tmp_path, flux)
    with pytest.raises(ValueError, match="sigma"):
        chi2_vs_median(per_train(path).intensity)


def test_xgm_scatter_before_and_after():
    flux = flux_array()
    rng = np.random.default_rng(3)
    follows = flux * 2.0 * (1 + 0.01 * rng.standard_normal(flux.size))
    result = xgm_scatter(follows, flux)
    assert result.n_trains == N_TRAINS
    assert result.rel_std_normalised == pytest.approx(0.01, rel=0.15)
    assert result.ratio < 0.2
    assert result.r_signal_flux > 0.99
    unrelated = xgm_scatter(
        xr.full_like(flux, 1.0) * (1 + 0.05 * rng.standard_normal(flux.size)), flux
    )
    assert unrelated.ratio > 1.5  # dividing by an unrelated flux adds its scatter


def test_xgm_scatter_drops_trains_without_flux():
    flux = flux_array(n=10)
    flux[:4] = np.nan
    assert xgm_scatter(flux * 3, flux).n_trains == 6


def test_cell_correlation_separates_lit_from_dark(tmp_path):
    xgm = flux_array()
    rng = np.random.default_rng(4)
    cells = (0, 1, 7, 15)
    lit = xgm.values[:, None] * (1 + 0.02 * rng.standard_normal((N_TRAINS, 2)))
    dark = 5 + rng.standard_normal((N_TRAINS, 1))
    scalar = np.concatenate([lit[:, :1], lit[:, 1:], dark, lit[:, :1] * 0.95], axis=1)
    path = write_frame_table(
        tmp_path / "jf.h5", "jungfrau", np.ones((N_TRAINS, 4, 3), np.float32),
        labels=cells, scalar=scalar,
    )  # fmt: skip
    result = cell_xgm_correlation(frame_scalar(path), xgm)
    assert list(result.cellId.values) == list(cells)
    assert (result.r.sel(cellId=[0, 1, 15]) > 0.95).all()
    assert abs(float(result.r.sel(cellId=7))) < 0.2
    assert (result.n == N_TRAINS).all()
    assert (result.r_low <= result.r).all() and (result.r <= result.r_high).all()


def test_detrend_leaves_the_fluctuation():
    rng = np.random.default_rng(5)
    trend = np.linspace(2.0, 1.0, 3000)  # an evaporating droplet: −50 % over a run
    fluct = 0.03 * rng.standard_normal(3000)
    out = detrend(trend * (1 + fluct))
    assert np.nanstd(out) == pytest.approx(0.03, rel=0.1)
    assert np.isnan(detrend(np.array([-1.0, -1.0, -1.0]))).all()


def drifting(n=3000, seed=6):
    """A signal that follows the XGM train to train, under opposite slow drifts."""
    rng = np.random.default_rng(seed)
    tids = FIRST_TRAIN + np.arange(n, dtype=np.uint64)
    jitter = 0.05 * rng.standard_normal(n)
    xgm = np.linspace(900, 1100, n) * (1 + jitter)
    signal = np.linspace(2.0, 1.0, n) * (1 + jitter + 0.01 * rng.standard_normal(n))
    as_da = lambda v, name: xr.DataArray(  # noqa: E731
        v, dims="trainId", coords={"trainId": tids}, name=name
    )
    return as_da(signal, "signal"), as_da(xgm, "xgm")


def test_detrending_reveals_what_the_drifts_hide():
    signal, xgm = drifting()
    grid = signal.expand_dims(cellId=[0]).transpose("trainId", "cellId")
    plain = cell_xgm_correlation(grid, xgm, detrend_window=None)
    assert float(plain.r[0]) < 0  # two opposite trends decide r
    detrended = cell_xgm_correlation(grid, xgm)
    assert float(detrended.r[0]) > 0.95
    assert detrended.attrs["detrend_window"] == 51


def test_xgm_scatter_reports_detrended_numbers():
    signal, xgm = drifting()
    result = xgm_scatter(signal, xgm)
    assert result.ratio > 0.9  # the drift dominates the plain spread
    assert result.ratio_detrended < 0.3
    assert result.rel_std_normalised_detrended == pytest.approx(0.01, rel=0.2)
    assert "ratio_detrended" not in xgm_scatter(signal, xgm, detrend_window=None)
