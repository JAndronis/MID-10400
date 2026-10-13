"""Per-run XPCS SNR, the keep criterion (context/run-triage.md §4.6), for one run.

A thin layer over `analysis.triage.xpcs`: the same reduction (`measure_run`), the same
analytical SNR per 30 s slice (`summarise`),

    SNR = β · k̄ · sqrt(N_pix · (F − 1) · φ · n) / √2,

and β read from a triage XPCS report rather than re-measured, since β is a property of
the optics and is measured over many runs (`scripts/run_triage.py --phase xpcs`).

An unknown β gives SNR NaN and ``extractable`` None — never False.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from analysis.saxs.config import DEFAULT_OUTPUT_ROOT as SAXS_ROOT
from analysis.triage.inventory import DAMNIT_ROOT, damnit_file
from analysis.triage.xpcs import Beta, measure_run, summarise

__all__ = ["beta_by_roi", "load_beta", "rep_rate_hz", "xpcs_snr"]

#: Two ROI centres within this [Å⁻¹] are the same ROI (DAMNIT stores float centres).
ROI_MATCH_A = 1e-6


def load_beta(report: str | Path | dict) -> dict[float, Beta]:
    """β per ROI centre [Å⁻¹] from a triage XPCS report (its ``beta`` block)."""
    if not isinstance(report, dict):
        report = json.loads(Path(report).read_text())
    out = {}
    for q, b in report["beta"].items():
        value = b["value"]
        out[float(q)] = Beta(
            float("nan") if value is None else float(value),
            int(b["n_eligible"]),
            float("nan") if b["p16"] is None else float(b["p16"]),
            float("nan") if b["p84"] is None else float(b["p84"]),
            tuple(b["runs"]),
        )
    return out


def beta_by_roi(roi_q_A: np.ndarray, beta: dict[float, Beta]) -> dict[int, Beta]:
    """β keyed by ROI index, matched on the centre; a ROI without one gets NaN."""
    nan = float("nan")
    out = {}
    for r, q in enumerate(np.asarray(roi_q_A, dtype=np.float64)):
        match = [b for key, b in beta.items() if abs(key - q) < ROI_MATCH_A]
        out[r] = match[0] if match else Beta(nan, 0, nan, nan, ())
    return out


def rep_rate_hz(path: str | Path) -> float:
    """DAMNIT's ``rep_rate`` [MHz] of a run, in Hz."""
    import h5py

    with h5py.File(path, "r") as f:
        if "rep_rate/data" not in f:
            raise KeyError(f"{path}: no rep_rate variable")
        return float(f["rep_rate/data"][()]) * 1e6


def xpcs_snr(
    run: int,
    beta: dict[float, Beta],
    *,
    proposal: int = 10400,
    damnit_root: str | Path = DAMNIT_ROOT,
    saxs_root: str | Path = SAXS_ROOT,
    rep_rate: float | None = None,
    select: np.ndarray | None = None,
) -> dict[str, Any]:
    """One run's SNR per plain ROI and 30 s slice, and its keep verdict.

    :param beta: `load_beta` of a triage XPCS report.
    :param rep_rate: pulse rate [Hz]; default DAMNIT's ``rep_rate`` of the run.
    :param select: train ids to restrict to (e.g. a mesh run's `on` trains).
    :returns: ``summary`` (the `summarise` dict without its slices), ``slices`` (a
        DataFrame: slice, start in s, usable trains, best plain SNR, SNR per ROI),
        ``rois`` (a DataFrame per ROI: kind, k̄, N_pix, φ, ρ, the fit, the β used and
        the run SNR) and ``measurement`` (the `RunMeasurement`).
    :raises LookupError: the run has no XPCS TTCF in DAMNIT.
    """
    import pandas as pd

    from analysis.triage.xpcs import SLICE_TRAINS, snr

    path = damnit_file(run, Path(damnit_root), proposal)
    scratch = Path(saxs_root) / f"r{run:04d}" / "agipd_saxs.h5"
    rate = rep_rate if rep_rate is not None else rep_rate_hz(path)
    m = measure_run(run, path, scratch, rate, select=select)
    if m is None:
        raise LookupError(f"r{run}: no xpcs_ttcf matrix in {path}")
    b = beta_by_roi(m.roi_q_A, beta)
    summary = summarise(m, b)
    rows = summary.pop("slices")

    names = [f"snr_{q:.4f}A" for q in m.roi_q_A]
    slices = pd.DataFrame(
        [
            {
                "slice": r["slice"],
                "start_s": r["slice"] * SLICE_TRAINS / 10.0,
                "n_usable": r["n_usable"],
                "best_snr": r["best"],
                **dict(zip(names, np.asarray(r["snr"], dtype=float), strict=True)),
            }
            for r in rows
        ]
    )
    beta_values = np.array([b[r].value for r in range(len(m.kinds))])
    run_snr = snr(beta_values, m.kbar, m.npix, m.n_frames, m.phi, m.n_usable)
    rois = pd.DataFrame(
        {
            "roi_q_A": m.roi_q_A,
            "q_lo_nm": m.edges_nm[:, 0],
            "q_hi_nm": m.edges_nm[:, 1],
            "kind": m.kinds,
            "kbar": m.kbar,
            "n_pix": m.npix,
            "phi": m.phi,
            "rho": m.rho,
            "A": [f.amplitude for f in m.fits],
            "sigma_A": [f.sigma_amplitude for f in m.fits],
            "gamma_per_s": [f.rate for f in m.fits],
            "beta": beta_values,
            "beta_n_eligible": [b[r].n_eligible for r in range(len(m.kinds))],
            "snr_run": run_snr,
        }
    )
    summary["beta_known"] = bool(
        np.isfinite(beta_values[[k == "plain" for k in m.kinds]]).any()
    )
    return {"summary": summary, "slices": slices, "rois": rois, "measurement": m}
