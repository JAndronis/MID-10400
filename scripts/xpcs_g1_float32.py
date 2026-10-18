#!/usr/bin/env python
"""G1 g2 difference: is it extra-speckle's float32? (``context/xpcs-hits.md`` §10).

Reads the files an acceptance run left under ``scratch/xpcs_hits_acceptance/r0423/``
(extra-speckle's ``TTCF_raw.nc``, ``TTCF_off.nc``, ``means.nc``, ``g2_c2.nc`` and our
Kernel.ES results file), and runs no extra-speckle code: its g2 reduction is
re-implemented here from ``extra_speckle/xpcs/tools.py`` (``_reduce_ttcf_regular``,
``g2_to_bins``) and checked bit for bit against its saved ``g2_c2.nc`` first.

Three computations of the same g2, six rings, 30 binned lags:

    stored    extra-speckle's ``g2_c2.nc`` (float32 end to end)
    f32       the reimplementation on ``means.nc``; must equal ``stored`` exactly
    f64       the same reduction from extra-speckle's own per-train files, mean and
              subtraction in float64
    ours      ``XpcsHits.g2(Averaging.ES)``

Read-only; prints one JSON object. Run on a login node (a few seconds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from analysis.xpcs.result import ES_KBAR_MIN, Averaging, XpcsHits

ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400/scratch/xpcs_hits_acceptance/r0423")
SKIP_LAST = 10


def diagonals(c2: np.ndarray, n_lags: int) -> np.ndarray:
    """``nanmean(np.diagonal(c2, offset=k))`` per ring and lag, in c2's dtype."""
    return np.stack(
        [
            np.array([np.nanmean(np.diagonal(r, offset=k)) for r in c2])
            for k in range(1, n_lags + 1)
        ],
        axis=-1,
    ).astype(c2.dtype)


def binned(diag: np.ndarray, tau: np.ndarray, skip: int = 10, count: int = 20):
    """``g2_to_bins(50, 20, 10)``: the first ``skip`` lags, then log-bin medians."""
    edges = np.logspace(np.log10(tau[skip]), np.log10(tau[-1]), count + 1)
    tail = [
        np.nanmedian(diag[..., (tau >= lo) & (tau < hi)], axis=-1)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    return np.concatenate([diag[..., :skip], np.column_stack(tail)], -1)


def spread(a, b) -> dict:
    """Largest absolute, elementwise relative and to-the-maximum difference."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    d, scale = np.abs(a - b)[both], np.abs(b)[both]
    return {
        "nan_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum()),
        "max_abs": float(d.max()),
        "max_rel_elementwise": float((d / scale).max()),
        "max_rel_to_max": float(d.max() / scale.max()),
        "n_elements": int(both.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    es = args.root / "es"
    results = args.root / "g1/es/xpcs_hits/r0423/xpcs_hits.h5"

    stored = xr.open_dataset(es / "g2_c2.nc")["g2"]
    means = xr.open_dataset(es / "means.nc")
    raw = xr.open_dataarray(es / "TTCF_raw.nc")
    off = xr.open_dataarray(es / "TTCF_off.nc")
    kept = stored.roi_q.values
    n_frames = means["raw"].shape[-1]
    n_lags = n_frames - 1 - SKIP_LAST

    ours_reader = XpcsHits(results)
    tau = np.arange(1, n_lags + 1) * ours_reader.period_s
    # rings extra-speckle kept, matched by the roi_q centre it reports
    rings = [int(np.argmin(np.abs(means.roi_q.values - q))) for q in kept]

    c2_32 = (means["raw"].values - means["off"].values)[rings]
    f32 = binned(diagonals(c2_32, n_lags), tau)
    raw64 = np.nanmean(raw.values[:, rings].astype(np.float64), axis=0)
    off64 = np.nanmean(off.values[:, rings].astype(np.float64), axis=0)
    f64_diag = diagonals(raw64 - off64, n_lags)
    f64 = binned(f64_diag, tau)

    trains = np.sort(raw.trainId.values).astype(np.uint64)
    ours_full = ours_reader.g2(Averaging.ES, trains, kbar_min=ES_KBAR_MIN)
    ours = ours_full["g2"].values[rings]
    ours_unbinned = ours_reader.g2(
        Averaging.ES, trains, binned=False, kbar_min=ES_KBAR_MIN
    )["g2"].values[rings]

    # our per-train maps rounded to float32 as extra-speckle stores them, then the
    # same float64 mean: what is left is the rounding of the per-train inputs alone
    per = ours_reader.per_train(trains)
    lit = (ours_reader.kbar(trains).values > ES_KBAR_MIN).transpose(0, 2, 1)
    keep = (lit[..., :, None] & lit[..., None, :])[:, rings]
    own_raw = np.where(keep, per["TTCF_raw"].values[:, rings], np.nan).astype(
        np.float64
    )
    own_off = np.where(keep, per["TTCF_off"].values[:, rings], np.nan).astype(
        np.float64
    )
    own32 = binned(
        diagonals(np.nanmean(own_raw, axis=0) - np.nanmean(own_off, axis=0), n_lags),
        tau,
    )

    mean_gap = {
        "raw_mean_f32_vs_f64": spread(means["raw"].values[rings], raw64),
        "off_mean_f32_vs_f64": spread(means["off"].values[rings], off64),
    }
    out = {
        "rings_kept": rings,
        "tau_first_ns": float(tau[0] * 1e9),
        "check_reimplementation_equals_stored": spread(f32, stored.values),
        "es_float32_vs_float64": spread(stored.values, f64),
        "ours_vs_es_float64": spread(ours, f64),
        "ours_vs_es_stored_float32": spread(ours, stored.values),
        "our_float32_per_train_inputs_vs_es_float64": spread(own32, f64),
        "ours_unbinned_vs_es_float64_unbinned": spread(ours_unbinned, f64_diag),
        "ours_vs_float32_of_ours": spread(ours.astype(np.float32), ours),
        "largest_abs_g2": float(np.nanmax(np.abs(f64))),
        "smallest_abs_g2": float(np.nanmin(np.abs(f64))),
        "float32_eps": float(np.finfo(np.float32).eps),
        **mean_gap,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
