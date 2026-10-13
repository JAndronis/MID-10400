"""Per-run XPCS SNR, the keep criterion (context/run-triage.md §4.6), for one run.

A thin layer over `analysis.triage.xpcs`: the same reduction (`measure_run`), the same
analytical SNR per 30 s slice (`summarise`),

    SNR = β · k̄ · sqrt(N_pix · (F − 1) · φ · n) / √2,

with β the fixed `REFERENCE_BETA` table, the same for every run (user, 2026-10-13):
4.8 % at 0.029 Å⁻¹ to 1.9 % at 0.0805, from r450 and r493–496. A β per run, or one
that moves with the set of runs it came from, would make the verdict depend on that
set. A report's ``beta`` block can still be passed in to compare.

An unknown β gives SNR NaN and ``extractable`` None — never False.

`xpcs_score` is the per-run diagnostic score: the SNR and the k̄ per XPCS ROI, both
over the run's droplet hits only (`analysis.triage.classify.hits`; user, 2026-10-13),
and the number of ROIs whose mean k̄ reaches `KBAR_MIN` (user, 2026-10-14).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.saxs.config import DEFAULT_OUTPUT_ROOT as SAXS_ROOT
from analysis.triage.inventory import DAMNIT_ROOT, damnit_file
from analysis.triage.xpcs import ROI_MATCH_A, Beta, fixed_beta, measure_run, summarise

__all__ = [
    "KBAR_MIN",
    "XPCS_ROOT",
    "beta_by_roi",
    "kbar_rois",
    "load_beta",
    "rep_rate_hz",
    "xpcs_score",
    "xpcs_snr",
]

#: Mean photons per pixel per frame a ROI needs (user, 2026-10-14). The same number
#: is extra-speckle's default ``kbar_threshold`` (pipeline/pipeline.py:89), below which
#: it drops a frame pair from the TTCF; here it is applied to the ROI's mean over hits.
KBAR_MIN = 3e-3
#: extra-speckle's per-run output; ``kbar.nc`` is k̄ per (trainId, frame, roi_q).
XPCS_ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400/scratch/xpcs")


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
    beta: dict[float, Beta] | None = None,
    *,
    proposal: int = 10400,
    damnit_root: str | Path = DAMNIT_ROOT,
    saxs_root: str | Path = SAXS_ROOT,
    rep_rate: float | None = None,
    select: np.ndarray | None = None,
) -> dict[str, Any]:
    """One run's SNR per plain ROI and 30 s slice, and its keep verdict.

    :param beta: None for `REFERENCE_BETA`, or `load_beta` of a triage XPCS report.
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
    b = fixed_beta(m.roi_q_A) if beta is None else beta_by_roi(m.roi_q_A, beta)
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
    summary["beta_source"] = "REFERENCE_BETA" if beta is None else "given"
    summary["beta_known"] = bool(
        np.isfinite(beta_values[[k == "plain" for k in m.kinds]]).any()
    )
    return {"summary": summary, "slices": slices, "rois": rois, "measurement": m}


def kbar_rois(
    run: int,
    trains=None,
    *,
    xpcs_root: str | Path = XPCS_ROOT,
    proposal: int = 10400,
    damnit_root: str | Path = DAMNIT_ROOT,
    saxs_root: str | Path = SAXS_ROOT,
    kbar_min: float = KBAR_MIN,
):
    """Mean k̄ of each XPCS ROI over ``trains``, and whether it reaches ``kbar_min``.

    From extra-speckle's ``kbar.nc`` (photons per pixel per frame, every frame of the
    selected trains). Without it, from DAMNIT's agipd_saxs weighted by the pass's
    pixels per bin (`triage.xpcs.read_saxs_kbar`), our geometry. On r0423 the two
    agree to 0.4–3 % per ROI (2026-10-14).

    :param trains: train ids to average over, e.g. ``hits(run)["train_ids"]``; None
        for every train.
    :returns: a DataFrame per ROI (``roi_q_A``, ``kind``, ``kbar``, ``above``), with
        ``attrs``: ``source``, ``n_trains``, ``kbar_min``, ``n_above``,
        ``n_plain_above``.
    """
    import pandas as pd
    import xarray as xr

    from analysis.triage.xpcs import (
        read_npix_per_bin,
        read_saxs_kbar,
        roi_edges_nm,
        roi_kinds,
    )

    path = Path(xpcs_root) / f"r{run:04d}" / "kbar.nc"
    wanted = None if trains is None else np.asarray(trains, dtype=np.uint64)
    if path.exists():
        with xr.open_dataarray(path) as da:
            if wanted is not None:
                da = da.sel(trainId=da.trainId.isin(wanted))
            n_trains = int(da.sizes["trainId"])
            values = da.mean(("trainId", "frame"), skipna=True).values
            roi_q = da.roi_q.values
        source = str(path)
    else:
        damnit = damnit_file(run, Path(damnit_root), proposal)
        with h5py.File(damnit, "r") as f:
            q = f["agipd_saxs/q"][:]
        npix = read_npix_per_bin(Path(saxs_root) / f"r{run:04d}" / "agipd_saxs.h5", q)
        saxs = read_saxs_kbar(damnit, roi_edges_nm(), npix)
        keep = np.ones(saxs.train_ids.size, dtype=bool)
        if wanted is not None:
            keep = np.isin(saxs.train_ids, wanted)
        n_trains = int(keep.sum())
        frames = saxs.n_frames[keep][:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.nansum(saxs.kbar[keep] * frames, axis=0) / frames.sum()
        roi_q = roi_edges_nm().mean(axis=1) / 10.0  # centres, Å⁻¹, as DAMNIT stores
        source = f"{damnit} (our geometry; no {path.name})"
    kinds = roi_kinds(roi_edges_nm())
    table = pd.DataFrame(
        {"roi_q_A": roi_q, "kind": kinds[: len(roi_q)], "kbar": values}
    )
    table["above"] = table.kbar >= kbar_min
    table.attrs.update(
        source=source,
        n_trains=n_trains,
        kbar_min=kbar_min,
        n_above=int(table.above.sum()),
        n_plain_above=int((table.above & (table.kind == "plain")).sum()),
    )
    return table


def xpcs_score(
    run: int,
    *,
    beta=None,
    proposal: int = 10400,
    damnit_root: str | Path = DAMNIT_ROOT,
    saxs_root: str | Path = SAXS_ROOT,
    xpcs_root: str | Path = XPCS_ROOT,
) -> dict:
    """The per-run XPCS diagnostic score, over the run's droplet hits only.

    :returns: ``hits`` (the classification summary), ``n_hits``, ``kbar`` (the
        `kbar_rois` table over the hits), ``n_rois_kbar_ok`` and
        ``n_plain_rois_kbar_ok``, and ``snr`` (`xpcs_snr`'s summary over the hits,
        None when the run has no hit or no TTCF).
    """
    from analysis.triage.classify import hits

    roots = {"proposal": proposal, "damnit_root": damnit_root, "saxs_root": saxs_root}
    h = hits(run, proposal=proposal, damnit_root=damnit_root)
    trains = h["train_ids"]
    kbar = kbar_rois(run, trains, xpcs_root=xpcs_root, **roots) if trains.size else None
    snr = None
    if trains.size:
        try:
            snr = xpcs_snr(run, beta, select=trains, **roots)["summary"]
        except LookupError:  # no TTCF in DAMNIT for this run
            snr = None
    return {
        "run": int(run),
        "kind": h["kind"],
        "block": h["block"],
        "n_hits": int(trains.size),
        "hits": h["summary"],
        "kbar": kbar,
        "n_rois_kbar_ok": None if kbar is None else kbar.attrs["n_above"],
        "n_plain_rois_kbar_ok": None if kbar is None else kbar.attrs["n_plain_above"],
        "snr": snr,
    }
