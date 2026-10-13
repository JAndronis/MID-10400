"""Checks on per-train curves and per-frame totals against the per-train flux.

The flux is per train, XGM × ``total_transmission`` (memory, saxs-background-
subtraction): at 2.26 MHz the per-pulse XGM reads the first ~20 pulses of a train
15–20 % high while the detector's own signal per pulse is flat, so dividing frame by
frame writes that ramp into I(q). Every function here divides an in-memory copy and
says so in ``attrs["normalised_by"]``; nothing is written back.

The XGM is DAMNIT's ``xgm_intensity``, the mean over *all* pulses of a train. That is
a sound per-train scale inside one run, whose pulse pattern is constant; across runs
with different pulse counts it is not (context/run-triage.md D4), and nothing here
compares runs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from analysis.triage.stats import robust_sd

__all__ = [
    "band_mean",
    "cell_xgm_correlation",
    "chi2_vs_median",
    "detrend",
    "per_train_flux",
    "svd",
    "xgm_scatter",
]

#: DAMNIT variables the flux is built from (src/amore/context.py).
XGM_VARIABLE, TRANSMISSION_VARIABLE = "xgm_intensity", "total_transmission"
FLUX_DESCRIPTION = "per-train XGM (mean over all pulses) × total_transmission"
#: Running-median window for `detrend`, in trains: the stability window of
#: context/run-triage.md §4.3 (5 s at 10 Hz).
DETREND_TRAINS = 51


def _damnit_array(path: Path, group: str):
    """One DAMNIT variable as a DataArray over trainId, loaded and closed."""
    import xarray as xr

    with xr.open_dataset(path, group=group) as ds:
        if len(ds.data_vars) != 1 or "trainId" not in ds.dims:
            raise ValueError(f"{path}:{group} is not one per-train array: {ds}")
        return ds[next(iter(ds.data_vars))].load().rename(group)


def per_train_flux(damnit_file: str | Path):
    """XGM [µJ], ``total_transmission`` and their product, per train, from DAMNIT.

    Outer-joined on trainId: a train missing either input has NaN flux, which every
    function here then drops, rather than silently pairing values by position.
    """
    import xarray as xr

    path = Path(damnit_file)
    xgm, trans = xr.align(
        _damnit_array(path, XGM_VARIABLE),
        _damnit_array(path, TRANSMISSION_VARIABLE),
        join="outer",
    )
    ds = xr.Dataset(
        {
            "xgm_uJ": xgm.astype(np.float64),
            "total_transmission": trans.astype(np.float64),
        }
    )
    ds["flux"] = ds.xgm_uJ * ds.total_transmission
    ds.flux.attrs["description"] = FLUX_DESCRIPTION
    ds.attrs["source"] = str(path)
    return ds


def _q_mask(q, q_range):
    if q_range is None:
        return np.ones(q.size, dtype=bool)
    lo, hi = q_range
    return (q >= lo) & (q < hi)


def _curves_and_sigma(curves):
    """``(intensity, sigma or None)`` with absent trains and bins as NaN."""
    import xarray as xr

    if isinstance(curves, xr.Dataset):
        present = curves.n_frames > 0
        intensity = curves.intensity.where(present)
        sigma = None
        if "sigma" in curves:
            sigma = curves.sigma.where(present & (curves.sigma > 0))
        return intensity, sigma
    return curves, None


def _normalise(intensity, sigma, flux):
    """Divide by the flux of the same train; trains without a flux are dropped."""
    import xarray as xr

    if flux is None:
        return intensity, sigma, "none"
    flux = flux.where(np.isfinite(flux) & (flux > 0))
    if sigma is None:
        intensity, flux = xr.align(intensity, flux, join="inner")
    else:
        intensity, sigma, flux = xr.align(intensity, sigma, flux, join="inner")
        sigma = sigma / flux
    return intensity / flux, sigma, FLUX_DESCRIPTION


def band_mean(per_train, q_range: tuple[float, float]):
    """Mean intensity over the bins with ``q_range[0] <= q < q_range[1]``, per train.

    An unweighted mean over bins (pooled per-train curves do not carry the pixel
    count per bin), NaN for a train with no frame.
    """
    keep = _q_mask(per_train.q.values, q_range)
    if not keep.any():
        raise ValueError(f"no q bin in {q_range}")
    present = per_train.n_frames > 0
    return per_train.intensity.isel(q=keep).mean("q").where(present)


def svd(curves, k: int = 5, *, flux=None, q_range=None):
    """Singular value decomposition of per-train curves: how many shapes a run has.

    Each curve is divided by its train's flux (when given), and every q bin by the
    run-median curve, so low q does not outweigh high q by its intensity alone. The
    matrix is **not** centred: component 0 is the run's mean shape and its weight the
    per-train level; components 1… are the changes of shape.

    :param curves: `analysis.diagnostics.frames.per_train`'s Dataset (``intensity``,
        ``sigma``, ``n_frames``) or a ``(trainId, q)`` DataArray.
    :param flux: per-train flux (`per_train_flux`), or None to decompose raw curves.
    :param q_range: ``(lo, hi)`` in the file's q unit, or None for every bin.
    :returns: a Dataset with ``singular_value``, ``explained`` (s²/Σs²) per
        component, ``weight`` (trainId, component) = U·s, ``shape`` (q, component)
        and ``median_curve`` (q). Each component's sign makes its shape sum
        positive. ``attrs["noise_floor"]``, when σ is known, is the largest
        singular value pure noise would give: σ_rms·(√n_trains + √n_q), the edge of
        the Marchenko–Pastur law for i.i.d. noise of the matrix's rms relative σ
        (Marchenko & Pastur 1967; Gavish & Donoho 2014, IEEE TIT 60:5040). With
        heteroscedastic σ it is a guide, not a test.
    """
    import xarray as xr

    intensity, sigma = _curves_and_sigma(curves)
    intensity, sigma, normalised_by = _normalise(intensity, sigma, flux)
    keep_q = _q_mask(intensity.q.values, q_range)
    x = intensity.isel(q=keep_q).transpose("trainId", "q").values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a bin no train populates
        median = np.nanmedian(x, axis=0) if x.size else np.zeros(keep_q.sum())
    populated = np.isfinite(median) & (median > 0)
    x = x[:, populated]
    complete = np.isfinite(x).all(axis=1)
    if complete.sum() < 2 or populated.sum() < 2:
        raise ValueError(
            f"{complete.sum()} complete trains over {populated.sum()} bins: "
            "nothing to decompose"
        )
    rel = x[complete] / median[populated]
    u, s, vt = np.linalg.svd(rel, full_matrices=False)
    k = min(k, s.size)
    sign = np.where(vt[:k].sum(axis=1) < 0, -1.0, 1.0)
    u, vt = u[:, :k] * sign, vt[:k] * sign[:, None]

    q = intensity.q.values[keep_q][populated]
    trains = intensity.trainId.values[complete]
    attrs = {
        "normalised_by": normalised_by,
        "relative_to": "run-median curve",
        "centred": 0,
        "n_trains": int(complete.sum()),
        "n_trains_dropped": int((~complete).sum()),
        "n_q": int(populated.sum()),
    }
    if sigma is not None:
        sig = sigma.isel(q=keep_q).transpose("trainId", "q").values
        sig = sig[:, populated][complete] / median[populated]
        sigma_rms = float(np.sqrt(np.nanmean(sig**2)))
        floor = sigma_rms * (np.sqrt(rel.shape[0]) + np.sqrt(rel.shape[1]))
        attrs.update(
            sigma_rms_relative=sigma_rms,
            noise_floor=float(floor),
            n_above_noise=int((s > floor).sum()),
        )
    component = np.arange(k)
    return xr.Dataset(
        {
            "singular_value": ("component", s[:k]),
            "explained": ("component", (s[:k] ** 2) / (s**2).sum()),
            "weight": (("trainId", "component"), u * s[:k]),
            "shape": (("q", "component"), vt.T),
            "median_curve": ("q", median[populated]),
        },
        coords={"trainId": trains, "q": q, "component": component},
        attrs=attrs,
    )


def chi2_vs_median(per_train, *, flux=None, q_range=None):
    """Reduced χ² of each train's curve against the run-median curve, with stored σ.

    χ²_t = mean over q of ((I_t − M) / σ_t)², over the bins where σ_t > 0. The
    median's own error is ignored: it is ≈ √(π/2)·σ/√N, so χ² reads high by a factor
    of about 1 + π/(2N) — 5e-4 for 3000 trains.

    :param per_train: `analysis.diagnostics.frames.per_train`'s Dataset.
    :param flux: per-train flux, or None to compare raw curves (then the flux's own
        scatter shows up in χ²).
    :returns: a Dataset with ``chi2`` and ``n_bins`` per train and ``median_curve``.
    """
    import xarray as xr

    intensity, sigma = _curves_and_sigma(per_train)
    if sigma is None:
        raise ValueError(
            "chi2_vs_median needs the stored sigma: pass per_train's Dataset"
        )
    intensity, sigma, normalised_by = _normalise(intensity, sigma, flux)
    keep = _q_mask(intensity.q.values, q_range)
    intensity, sigma = intensity.isel(q=keep), sigma.isel(q=keep)
    median = intensity.median("trainId", skipna=True)
    z = (intensity - median) / sigma
    n_bins = z.notnull().sum("q")
    chi2 = (z**2).sum("q", skipna=True) / n_bins.where(n_bins > 0)
    return xr.Dataset(
        {"chi2": chi2, "n_bins": n_bins, "median_curve": median},
        attrs={"normalised_by": normalised_by, "statistic": "reduced chi2 vs median"},
    )


def detrend(values: np.ndarray, window: int = DETREND_TRAINS) -> np.ndarray:
    """``x / running_median(x, window) − 1``: the train-to-train fluctuation.

    The levitated droplet evaporates through a run, so a per-train signal drifts by
    tens of per cent while the XGM drifts on its own. Over a whole run two slow
    trends then decide r (r0423: −0.20 between every lit JF cell and the XGM, +0.20
    to +0.27 after this). The window counts trains in order, edges held at the
    nearest value; NaN where the running median is not positive.
    """
    from scipy.ndimage import median_filter

    x = np.asarray(values, dtype=np.float64)
    base = median_filter(x, size=min(window, x.size), mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(base > 0, x / base - 1.0, np.nan)


def _relative_spread(values: np.ndarray) -> tuple[float, float]:
    """``(std / mean, 1.4826·MAD / median)`` of finite values."""
    mean = values.mean()
    std = values.std(ddof=1) if values.size > 1 else np.nan
    return float(std / mean), float(robust_sd(values) / np.median(values))


def _fluctuation_spread(values: np.ndarray) -> tuple[float, float]:
    """`_relative_spread` of a series already detrended to fluctuations about 0."""
    std = values.std(ddof=1) if values.size > 1 else np.nan
    return float(std), float(robust_sd(values))


def xgm_scatter(signal, flux, *, detrend_window: int | None = DETREND_TRAINS):
    """Relative scatter of a per-train signal before and after dividing by the flux.

    A ratio well below 1 means the flux explains the signal's train-to-train
    variation; near 1, it does not (droplet transmission, for instance, varies
    independently of the XGM — CLAUDE.md "XGM"); above 1, dividing adds the flux's
    own scatter.

    :param signal: a per-train scalar, e.g. `band_mean`.
    :param flux: the per-train flux, e.g. ``per_train_flux(...).flux``.
    :param detrend_window: also compute every number on `detrend`ed series (suffix
        ``_detrended``), which is what the per-train normalisation acts on; None
        skips it.
    :returns: a Series: ``n_trains``, the plain (std/mean) and robust (MAD/median)
        relative spreads of the raw signal, of the normalised signal and of the
        flux, their ratios, and Pearson r between signal and flux.
    """
    import pandas as pd
    import xarray as xr

    signal, flux = xr.align(signal, flux, join="inner")
    s, f = signal.values.astype(np.float64), flux.values.astype(np.float64)
    ok = np.isfinite(s) & np.isfinite(f) & (f > 0)
    s, f = s[ok], f[ok]
    if s.size < 3:
        raise ValueError(f"{s.size} trains with both a signal and a flux")
    out = {"n_trains": int(s.size)}
    variants = [("", s, f, s / f, _relative_spread)]
    if detrend_window:
        variants.append(
            ("_detrended", detrend(s, detrend_window), detrend(f, detrend_window),
             detrend(s / f, detrend_window), _fluctuation_spread)
        )  # fmt: skip
    for suffix, sig, fl, norm, spread in variants:
        keep = np.isfinite(sig) & np.isfinite(fl) & np.isfinite(norm)
        raw, raw_robust = spread(sig[keep])
        normed, normed_robust = spread(norm[keep])
        flux_rel, flux_robust = spread(fl[keep])
        out.update(
            {
                f"rel_std_raw{suffix}": raw,
                f"rel_std_normalised{suffix}": normed,
                f"rel_std_flux{suffix}": flux_rel,
                f"ratio{suffix}": normed / raw,
                f"robust_rel_std_raw{suffix}": raw_robust,
                f"robust_rel_std_normalised{suffix}": normed_robust,
                f"robust_rel_std_flux{suffix}": flux_robust,
                f"robust_ratio{suffix}": normed_robust / raw_robust,
                f"r_signal_flux{suffix}": float(np.corrcoef(sig[keep], fl[keep])[0, 1]),
            }
        )
    return pd.Series(out)


def cell_xgm_correlation(
    grid, reference, *, detrend_window: int | None = DETREND_TRAINS
):
    """Pearson r of each cell's (or pulse's) per-train value against a reference.

    The lit-cell cross-check of transfer_context/06: on p010646, lit JUNGFRAU cells
    gave r ≈ 0.98 against the XGM and dark ones ≤ 0.2. The stored WAXS files hold
    lit cells only, so from them this can confirm that every stored cell follows the
    beam; a dark-cell control needs the cell from proc.

    The reference is usually the XGM, but any per-train series works, e.g. the
    AGIPD train total (``frame_scalar(agipd).sum("pulseId")``): on r0423 JF cells
    follow it at r ≈ 0.8 while the train-mean XGM reaches only ≈ 0.25 (detrended).

    :param grid: ``(trainId, label)`` values, e.g. `frame_scalar` of a JUNGFRAU file.
    :param reference: a per-train series: ``per_train_flux(...).xgm_uJ``, the flux,
        or another detector's train total.
    :param detrend_window: correlate `detrend`ed fluctuations (default), or the
        values as they are with None.
    :returns: a Dataset over the label: ``r``, ``n`` trains, and ``r_low``/``r_high``,
        the ±1σ interval from Fisher's z (σ_z = 1/√(n − 3)).
    """
    import xarray as xr

    grid, reference = xr.align(grid, reference, join="inner")
    (label_dim,) = [d for d in grid.dims if d != "trainId"]
    x = reference.values.astype(np.float64)
    values = grid.transpose("trainId", label_dim).values.astype(np.float64)
    r = np.full(values.shape[1], np.nan)
    n = np.zeros(values.shape[1], dtype=np.int64)
    for c in range(values.shape[1]):
        ok = np.isfinite(values[:, c]) & np.isfinite(x)
        a, b = values[ok, c], x[ok]
        if detrend_window and ok.sum():
            a, b = detrend(a, detrend_window), detrend(b, detrend_window)
            keep = np.isfinite(a) & np.isfinite(b)
            a, b = a[keep], b[keep]
        n[c] = a.size
        if n[c] >= 3 and a.std() > 0 and b.std() > 0:
            r[c] = np.corrcoef(a, b)[0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        z, dz = np.arctanh(np.clip(r, -1 + 1e-15, 1 - 1e-15)), 1 / np.sqrt(n - 3)
    return xr.Dataset(
        {
            "r": (label_dim, r),
            "n": (label_dim, n),
            "r_low": (label_dim, np.tanh(z - dz)),
            "r_high": (label_dim, np.tanh(z + dz)),
        },
        coords={label_dim: grid[label_dim].values},
        attrs={
            "column": str(grid.name),
            "against": str(reference.name),
            "detrend_window": detrend_window or 0,
        },
    )
