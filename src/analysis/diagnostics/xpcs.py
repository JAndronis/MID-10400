"""Per-run XPCS SNR, the keep criterion (context/run-triage.md §4.6), for one run.

A thin layer over `analysis.triage.xpcs`: the same reduction (`measure_run`, from the
run's hit-train results file, `analysis.xpcs.result.XpcsHits`), the same
analytical SNR per 30 s slice (`summarise`),

    SNR = β · k̄ · sqrt(N_pix · (F − 1) · φ · n) / √2,

with β the fixed `REFERENCE_BETA` table, the same for every run (user, 2026-10-13):
4.8 % at 0.029 Å⁻¹ to 1.9 % at 0.0805, from r450 and r493–496. A β per run, or one
that moves with the set of runs it came from, would make the verdict depend on that
set. A report's ``beta`` block can still be passed in to compare.

An unknown β gives SNR NaN and ``extractable`` None — never False.

`xpcs_score` is the per-run diagnostic score: the SNR and the k̄ per XPCS ROI, both
over the run's droplet hits only (the results file's hit set, which the pass took from
`analysis.triage.classify.hits`; user, 2026-10-13), and the number of ROIs whose
mean k̄ reaches `KBAR_MIN` (user, 2026-10-14).

Nothing here reads DAMNIT's ``xpcs_ttcf`` or extra-speckle's ``kbar.nc``; the
slices are anchored to the first usable hit, because the results file knows only
its hits.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from analysis.triage.xpcs import (
    KBAR_MIN,
    ROI_MATCH_A,
    Beta,
    fixed_beta,
    measure_run,
    roi_edges_from,
    roi_kinds,
    summarise,
)
from analysis.xpcs.config import DEFAULT_XPCS_ROOT
from analysis.xpcs.result import XpcsHits, read

__all__ = [
    "KBAR_MIN",
    "beta_by_roi",
    "kbar_rois",
    "load_beta",
    "open_results",
    "xpcs_score",
    "xpcs_snr",
]


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


def open_results(
    run: int,
    *,
    proposal: int = 10400,
    output_root: str | Path = DEFAULT_XPCS_ROOT,
    hits: XpcsHits | None = None,
) -> XpcsHits:
    """The run's hit-train results file (``hits`` as given).

    :raises LookupError: the run has no results file; its hit-train pass has not run.
    """
    if hits is not None:
        return hits
    try:
        return read(proposal, run, output_root=output_root)
    except FileNotFoundError:
        raise LookupError(f"r{run}: no XPCS results file under {output_root}") from None


def xpcs_snr(
    run: int,
    beta: dict[float, Beta] | None = None,
    *,
    proposal: int = 10400,
    output_root: str | Path = DEFAULT_XPCS_ROOT,
    hits: XpcsHits | None = None,
    select: np.ndarray | None = None,
    span: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """One run's SNR per plain ROI and 30 s slice, and its keep verdict.

    :param beta: None for `REFERENCE_BETA`, or `load_beta` of a triage XPCS report.
    :param hits: an open results file, else the run's under ``output_root``.
    :param select: train ids to restrict to (e.g. a mesh run's `on` trains).
    :param span: first and last train id the slices are anchored to (default: the
        first and last usable hit).
    :returns: ``summary`` (the `summarise` dict without its slices), ``slices`` (a
        DataFrame: slice, start in s, usable trains, best plain SNR, SNR per ROI),
        ``rois`` (a DataFrame per ROI: kind, k̄, N_pix, φ, ρ, the fit, the β used and
        the run SNR) and ``measurement`` (the `RunMeasurement`).
    :raises LookupError: the run has no results file.
    """
    import pandas as pd

    from analysis.triage.xpcs import SLICE_TRAINS, snr

    opened = open_results(run, proposal=proposal, output_root=output_root, hits=hits)
    m = measure_run(run, opened, select=select, span=span)
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
    proposal: int = 10400,
    output_root: str | Path = DEFAULT_XPCS_ROOT,
    hits: XpcsHits | None = None,
    kbar_min: float = KBAR_MIN,
):
    """Mean k̄ of each XPCS ring over ``trains``, and whether it reaches ``kbar_min``.

    From the results file (`XpcsHits.kbar`: photons per pixel per frame, every frame
    of the usable trains).

    :param trains: train ids to average over (they must be hits); None for every
        usable hit.
    :returns: a DataFrame per ROI (``roi_q_A``, ``kind``, ``kbar``, ``above``), with
        ``attrs``: ``source``, ``n_trains``, ``kbar_min``, ``n_above``,
        ``n_plain_above``.
    """
    import pandas as pd

    opened = open_results(run, proposal=proposal, output_root=output_root, hits=hits)
    da = opened.kbar(None if trains is None else np.asarray(trains, dtype=np.uint64))
    n_trains = int(da.sizes["trainId"])
    values = (
        da.mean(("trainId", "frame"), skipna=True).values
        if n_trains
        else np.full(opened.n_rings, np.nan)
    )
    kinds = roi_kinds(roi_edges_from(opened.edges_nm, opened.n_rings))
    table = pd.DataFrame({"roi_q_A": opened.roi_q, "kind": kinds, "kbar": values})
    table["above"] = table.kbar >= kbar_min
    table.attrs.update(
        source=str(opened.path),
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
    output_root: str | Path = DEFAULT_XPCS_ROOT,
    hits: XpcsHits | None = None,
) -> dict:
    """The per-run XPCS diagnostic score, over the hits in the run's results file.

    :returns: ``n_hits`` (the file's hit set), ``hits`` (its `Coverage` as a dict),
        ``kbar`` (the `kbar_rois` table over the usable hits), ``n_rois_kbar_ok``
        and ``n_plain_rois_kbar_ok`` (None without a usable hit), and ``snr``
        (`xpcs_snr`'s summary, None without a usable hit).
    :raises LookupError: the run has no results file.
    """
    opened = open_results(run, proposal=proposal, output_root=output_root, hits=hits)
    coverage = opened.coverage()
    usable = coverage.n_usable > 0
    kbar = kbar_rois(run, hits=opened) if usable else None
    snr = xpcs_snr(run, beta, hits=opened)["summary"] if usable else None
    return {
        "run": int(run),
        "n_hits": coverage.n_hits,
        "hits": dataclasses.asdict(coverage),
        "kbar": kbar,
        "n_rois_kbar_ok": None if kbar is None else kbar.attrs["n_above"],
        "n_plain_rois_kbar_ok": None if kbar is None else kbar.attrs["n_plain_above"],
        "snr": snr,
    }
