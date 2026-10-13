"""Fakes for the triage: DAMNIT's sqlite and per-run HDF5 files, in their real layout.

The layout is DAMNIT's as found on p10400 (2026-10-01): one group per variable, the
main dataset named as `analysis.triage.inventory.TRACKED` says, ``droplet_fit`` with
``trainId``/``volume``/``center``/``dim``, scalar byte strings under ``.errors``, and
a 0-d value for a variable that returned nothing.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import h5py
import numpy as np

from analysis.triage.inventory import RUN_COLUMNS, TRACKED

FIRST_TRAIN = 2637862276


def write_damnit_file(
    path: Path,
    *,
    n_trains: int = 100,
    volume=None,
    center=None,
    xgm=None,
    jf_cells=None,
    errors: dict[str, str] | None = None,
    empty: tuple[str, ...] = (),
) -> Path:
    tids = FIRST_TRAIN + np.arange(n_trains, dtype=np.uint64)
    with h5py.File(path, "w") as f:
        if volume is not None:
            g = f.create_group("droplet_fit")
            g["trainId"] = tids
            g["volume"] = np.asarray(volume, dtype=np.float32)
            g["dim"] = np.array([b"x", b"y"])
            g["center"] = (
                np.zeros((n_trains, 2), np.float32)
                if center is None
                else np.asarray(center, np.float32)
            )
            g["radius"] = np.full((n_trains, 2), 50.0, np.float32)
        if xgm is not None:
            g = f.create_group("xgm_intensity")
            g["Energy"] = np.asarray(xgm, dtype=np.float32)
            g["trainId"] = tids
        if jf_cells is not None:
            g = f.create_group("jungfrau_waxs_jf1")
            g["cellId"] = np.asarray(jf_cells, dtype=np.uint16)
            g["intensity"] = np.zeros((n_trains, len(jf_cells), 4), np.float32)
        for name in empty:
            f.create_group(name)[TRACKED[name]] = np.float64(np.nan)
        if errors:
            e = f.create_group(".errors")
            for name, text in errors.items():
                e[name] = text.encode()
    return path


def write_sqlite(path: Path, rows: list[dict]) -> Path:
    """A ``runs`` table with `RUN_COLUMNS` plus a preview column the triage ignores."""
    cols = list(RUN_COLUMNS) + ["agipd_iq_overview"]
    with closing(sqlite3.connect(path)) as con:
        con.execute(f"CREATE TABLE runs ({', '.join(cols)})")
        for row in rows:
            values = [row.get(c) for c in cols]
            con.execute(
                f"INSERT INTO runs VALUES ({', '.join('?' * len(cols))})", values
            )
        con.commit()
    return path


def run_row(run: int, **kw) -> dict:
    base = {
        "run": run,
        "start_time": 1778531000.0 + 600 * run,
        "run_length": "0:05:00",
        "run_type": "XPCS",
        "sample": "Ferritin_50_PEG6K_5p",
        "scan_type": None,
        "n_trains": 3000,
        "pulses": 155.0,
        "rep_rate": 2.256944,
        "raw_size": 4.7,
        "proc_size": 0.29,
        "total_transmission": 1.0,
        "xgm_intensity": 330.0,
        "agipd_iq_overview": b"\x89PNG",
    }
    base.update(kw)
    return base


REP_RATE_HZ = 2.2569444e6  # DAMNIT's rep_rate on p10400


def write_xpcs_run(
    damnit_path: Path,
    scratch_path: Path,
    rng: np.random.Generator,
    *,
    n_trains: int = 200,
    n_frames: int = 40,
    beta: float = 0.03,
    decay_lags: float = 8.0,
    kbar: float = 0.01,
    npix_per_bin: float = 40_000.0,
    n_bins: int = 400,
    nan_frames: dict[int, int] | None = None,
    write_mean_g2: bool = True,
) -> dict:
    """A DAMNIT file (agipd_saxs, xpcs_ttcf, xpcs_mean_dataset) and a scratch file.

    The TTCF of every train and ROI is β·exp(−2Γ|i−j|τ₁) plus Gaussian noise whose
    size is the pipeline's analytical noise per pair, √2 / (k̄ √N_pix), so a correct
    noise check returns ρ ≈ 1. ``nan_frames`` maps ROI index -> number of leading
    frames set to NaN in every train, as extra-speckle's kbar filter would.
    """
    from analysis.triage.xpcs import ROI_EDGES_A, roi_bin_masks, roi_edges_nm

    q = np.linspace(0.0813, 1.0570, n_bins)
    edges = roi_edges_nm()
    npix_roi = roi_bin_masks(q, edges).sum(axis=1) * npix_per_bin
    n_rois = len(ROI_EDGES_A) - 1
    tau1 = 1.0 / REP_RATE_HZ
    rate = 1.0 / (2.0 * decay_lags * tau1)
    lag = np.abs(np.subtract.outer(np.arange(n_frames), np.arange(n_frames)))
    truth = beta * np.exp(-2.0 * rate * lag * tau1)
    sigma_pair = np.sqrt(2.0) / (kbar * np.sqrt(npix_roi))
    tids = FIRST_TRAIN + np.arange(n_trains, dtype=np.uint64)

    ttcf = np.empty((n_trains, n_rois, n_frames, n_frames), np.float32)
    for r in range(n_rois):
        noise = rng.normal(0.0, sigma_pair[r], (n_trains, n_frames, n_frames))
        noise = np.triu(noise, 1)
        ttcf[:, r] = truth + noise + noise.transpose(0, 2, 1)
        for_nan = (nan_frames or {}).get(r, 0)
        if for_nan:
            ttcf[:, r, :for_nan, :] = np.nan
            ttcf[:, r, :, :for_nan] = np.nan

    with h5py.File(damnit_path, "w") as f:
        g = f.create_group("agipd_saxs")
        g["q"] = q
        g["trainId"] = tids
        g["n_frames"] = np.ones((n_trains, n_frames), np.uint8)
        g["intensity"] = np.full((n_trains, n_frames, n_bins), kbar, np.float32)
        g = f.create_group("xpcs_ttcf")
        g.create_dataset(
            "__xarray_dataarray_variable__",
            data=ttcf,
            chunks=(64, 1, n_frames, n_frames),
        )
        g["trainId"] = tids
        g["roi_q"] = 0.5 * (np.asarray(ROI_EDGES_A[:-1]) + np.asarray(ROI_EDGES_A[1:]))
        if write_mean_g2:
            g = f.create_group("xpcs_mean_dataset")
            g["g2"] = truth[0, 1:31][None, :].repeat(n_rois, axis=0).astype(np.float32)
            g["roi_q"] = f["xpcs_ttcf/roi_q"][:]
    with h5py.File(scratch_path, "w") as f:
        f["q/centers"] = q
        f["frames/status"] = np.zeros(50, np.uint8)
        f["frames/normalization"] = np.full((50, n_bins), npix_per_bin, np.float32)
    return {
        "beta": beta,
        "rate": rate,
        "tau1": tau1,
        "npix_roi": npix_roi,
        "kbar": kbar,
        "sigma_pair": sigma_pair,
        "train_ids": tids,
    }
