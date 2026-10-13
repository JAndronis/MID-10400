#!/usr/bin/env python
"""W7: the numbers photonising JUNGFRAU needs (CLAUDE.md open task 18, pitfalls 23-24).

Measurement only. Nothing here writes to the pass's output, and nothing here is a
gate: it fixes the parameters of the per-ASIC photon counting before
``analysis.waxs.photons`` exists. Verdict as JSON beside this script; the
per-train arrays behind it go to ``--out-dir`` as an ``.npz``.

    python scripts/w7_photonize_check.py --runs 423 370 --detectors jf1 jf2 --trains 100

**The estimator** is p010646's (``transfer_context/templates/jf_photon_count.py``,
validated bit for bit against fxs ``--photons``), copied here verbatim in
:func:`asic_view` and :func:`baseline`: per 256 x 256 ASIC, leaving out its 3 edge
pixels, the median of the pixels below 2 keV, then three times the mean of the
pixels within ±1 keV of the current estimate. It is run on three pixel sets, all
``data.mask``-good and finite:

* ``A`` every such pixel, the ``.edf``-masked ones included;
* ``K`` those the static ``.edf`` keeps — the pixels that are integrated;
* ``S`` those the ``.edf`` masks — shadowed or excluded, but read out all the same.

**What it measures**, per run and detector, over ``--trains`` evenly spaced trains:

1. ASIC tiling: do the ``NON_STANDARD_SIZE`` (bit 22) rows and columns sit at the
   256-pixel boundaries the estimator assumes?
2. Pixels per ASIC in each set, and how many ASICs are fully or mostly masked.
3. Dark cells (all but the one straight after the lit sequence): train-to-train rms
   of the level, the 8 x 8 ASIC correlation, the module-common share, the
   within-train against between-train spread, static offsets per cell and per
   64-column ADC group, and iteration 3 against iteration 10.
4. The cell after the last lit one against the rest: its level, and the slope of
   its mean image against the lit mean image (a leftover of the last lit cell).
5. Lit frames: each set's level against the mean of the same train's dark cells
   (bias, rms, by cell and by intensity), and against the shadow set ``S``.
6. Over-dispersion ``D = var/mean`` of photon counts in 0.1 nm⁻¹ x 1° (q, χ) cells.
7. ``keV / (E · photons)`` against q and along the run, raw and level-subtracted.
8. Counting at ``E/2`` (should double every photon) and at 9.000 keV (the XGM's
   value, CLAUDE.md open task 15) against the configured 9.04.
9. The spectrum of level-subtracted keV: the single-photon peak, the valley at
   ``E/2``, and how many photons a level error of 0.1 keV moves.

The σ_read the pass measures today (``waxs/cells.py``: one std over pixel x dark
cell x train around one global mean) is reproduced and split into what it lumps:
common mode, pixel offsets, cell offsets, the leftover, and pure pixel noise.

Needs Maxwell: the proc data, and the real PONI and ``.edf`` files.
"""

from __future__ import annotations

# Before numpy, pyFAI or EXtra-data import anything that builds a thread pool.
from _common import report_header, write_report  # noqa: E402

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from analysis.common.plan import evenly_spaced  # noqa: E402
from analysis.waxs.cells import CellAccumulator  # noqa: E402
from analysis.waxs.config import (  # noqa: E402
    CELLS_PER_TRAIN,
    MODULE_SHAPE,
    config_for,
)

# ── the template's estimator, verbatim apart from the parameters ─────────────
LEVEL_CUT = 2.0  # keV: the first estimate is the median of the pixels below this
LEVEL_WINDOW = 1.0  # keV: then the mean of the pixels within ±this of it
ITERATIONS = 3
EDGE = 3
ASIC = 256
N_ASICS = 8
ASIC_PIXELS_INNER = (ASIC - 2 * EDGE) ** 2

ROWS, COLS = np.indices(MODULE_SHAPE)
#: ASIC index per pixel, ``4 * row_block + column_block`` — the order
#: :func:`asic_view` produces.
ASIC_ID = (ROWS // ASIC) * 4 + COLS // ASIC
#: 64-column ADC group per pixel: 2 row blocks x 16 column groups.
ADC_ID = (ROWS // ASIC) * 16 + COLS // 64
IN_EDGE = (
    ((ROWS % ASIC) < EDGE)
    | ((ROWS % ASIC) >= ASIC - EDGE)
    | ((COLS % ASIC) < EDGE)
    | ((COLS % ASIC) >= ASIC - EDGE)
)

#: Spectrum histogram of level-subtracted keV.
SPEC_LO, SPEC_HI, SPEC_STEP = -3.0, 40.0, 0.05
N_SPEC = int(round((SPEC_HI - SPEC_LO) / SPEC_STEP))
#: Coarse q bands for the keV/photon ratio; fine (q, χ) cells for D.
N_Q = 100
D_Q_STEP, D_CHI_STEP_DEG, D_MIN_PIXELS, D_MIN_MEAN = 0.1, 1.0, 20, 0.02
D_OCC_EDGES = np.logspace(-2, 1, 16)
#: Per-pixel accumulators ignore |x - level| beyond this in dark cells, so one
#: stray photon does not set a pixel's offset.
DARK_CLIP_KEV = 2.0
XGM_ENERGY_KEV = 9.000


def asic_view(a: np.ndarray, edge: int = EDGE) -> np.ndarray:
    """``(..., 512, 1024)`` -> ``(..., 8, n)``: each ASIC's pixels, edges left out."""
    lead = a.shape[:-2]
    tiles = a.reshape(*lead, 2, ASIC, 4, ASIC).swapaxes(-3, -2)
    tiles = tiles[..., edge : ASIC - edge, edge : ASIC - edge]
    return tiles.reshape(*lead, N_ASICS, -1)


def baseline(
    a: np.ndarray,
    cut: float = LEVEL_CUT,
    window: float = LEVEL_WINDOW,
    iterations: int = ITERATIONS,
) -> np.ndarray:
    """Zero-photon level along the last axis, NaN where nothing qualifies."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        est = np.nanmedian(np.where(a < cut, a, np.nan), -1)
        for _ in range(iterations):
            w = np.abs(a - est[..., None]) < window
            est = np.nansum(np.where(w, a, 0), -1) / w.sum(-1)
    return est


def per_pixel(level: np.ndarray) -> np.ndarray:
    """``(..., 8)`` ASIC levels -> ``(..., 512, 1024)``."""
    return level[..., ASIC_ID]


def count(sub: np.ndarray, energy: float) -> np.ndarray:
    """``clip(rint(sub / E), 0)``; NaN stays NaN."""
    return np.clip(np.rint(sub / energy), 0, None)


# ── small statistics helpers ────────────────────────────────────────────────
def _f(value: Any) -> Any:
    """JSON-friendly floats: NaN becomes None, arrays become rounded lists."""
    if isinstance(value, np.ndarray):
        return [_f(v) for v in value.tolist()] if value.ndim else _f(value.item())
    if isinstance(value, list):
        return [_f(v) for v in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if not np.isfinite(value) else round(value, 6)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _nanstats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": _f(values.mean()),
        "std": _f(values.std(ddof=1) if values.size > 1 else np.nan),
        "median": _f(np.median(values)),
        "p02": _f(np.percentile(values, 2)),
        "p98": _f(np.percentile(values, 98)),
        "max_abs": _f(np.abs(values).max()),
    }


def _sequence_successor(lit: tuple[int, ...]) -> int | None:
    """The cell read straight after the lit storage-cell sequence, if any is dark."""
    if not lit or len(lit) >= CELLS_PER_TRAIN:
        return None
    wanted = set(lit)
    for start in lit:
        run = {(start + i) % CELLS_PER_TRAIN for i in range(len(lit))}
        if run == wanted:
            return (start + len(lit)) % CELLS_PER_TRAIN
    return None


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    xc = x[ok] - x[ok].mean()
    return float((xc * (y[ok] - y[ok].mean())).sum() / (xc**2).sum())


def _demean_trains(levels: np.ndarray) -> np.ndarray:
    """Subtract each (cell, ASIC)'s mean over trains, leaving the fluctuation."""
    return levels - np.nanmean(levels, axis=0, keepdims=True)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _peak(hist: np.ndarray, energy: float) -> dict[str, Any]:
    """Single-photon peak position, the valley at E/2 and the edge density."""
    centres = SPEC_LO + SPEC_STEP * (np.arange(N_SPEC) + 0.5)
    window = (centres > 0.7 * energy) & (centres < 1.3 * energy)
    if not hist[window].any():
        return {}
    top = np.flatnonzero(window)[np.argmax(hist[window])]
    around = slice(max(top - 12, 0), top + 13)  # ±0.6 keV
    peak = float((centres[around] * hist[around]).sum() / hist[around].sum())
    valley_zone = (centres > 0.3 * energy) & (centres < 0.7 * energy)
    valley = np.flatnonzero(valley_zone)[np.argmin(hist[valley_zone])]
    half = int(np.argmin(np.abs(centres - energy / 2)))
    zero = int(np.argmin(np.abs(centres)))
    one_photon = hist[(centres >= energy / 2) & (centres < 1.5 * energy)].sum()
    # Pixels per keV at the rounding edge, as a fraction of the one-photon
    # pixels: a level wrong by δ keV moves roughly this x δ of them.
    density_half = hist[half - 2 : half + 3].mean() / SPEC_STEP
    return {
        "peak_kev": peak,
        "peak_over_e": peak / energy,
        "valley_kev": float(centres[valley]),
        "valley_over_peak": float(hist[valley] / hist[top]),
        "half_e_over_peak": float(hist[half] / hist[top]),
        "zero_over_peak": float(hist[zero] / hist[top]),
        "one_photon_pixels_moved_per_0p1_kev_level_error": float(
            0.1 * density_half / one_photon
        )
        if one_photon
        else None,
    }


# ── one run, one detector ───────────────────────────────────────────────────
def measure(
    proposal: int, run: int, detector: str, n_trains: int, out_dir: str
) -> dict[str, Any]:
    """Every measurement for one (run, detector); see the module docstring."""
    from extra_data import by_id, open_run

    from analysis.waxs.operator import build_operator
    from analysis.waxs.plan import open_detector

    started = time.perf_counter()
    cfg = config_for(proposal, run, detector)
    energy = float(cfg.photon_energy_kev)
    op, ai = build_operator(cfg)
    static = op.static_bad.reshape(MODULE_SHAPE)
    q = np.asarray(ai.array_from_unit(MODULE_SHAPE, "center", "q_nm^-1"), np.float64)
    chi = np.degrees(np.asarray(ai.chiArray(MODULE_SHAPE), np.float64))

    dc = open_run(proposal, run, data="proc")
    det = open_detector(cfg, dc)
    all_trains = np.asarray(det.train_ids, dtype=np.uint64)

    def read(train_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = det.select_trains(by_id[[int(train_id)]])
        stored = selected["data.adc"].train_id_coordinates()
        if stored.size != 1 or int(stored[0]) != int(train_id):
            raise ValueError(f"asked for train {train_id}, got {stored}")
        data = np.asarray(selected["data.adc"].ndarray())[0, 0]
        mask = np.asarray(selected["data.mask"].ndarray())[0, 0]
        cells = np.asarray(selected["data.memoryCell"].ndarray()).reshape(-1)
        return data, mask, cells

    # Lit and dark exactly as the pass decides them, off its own sampled trains.
    accumulator = CellAccumulator(cfg, op.static_bad)
    for train_id in evenly_spaced(all_trains, cfg.cell_sample_trains).tolist():
        data, mask, cells = read(train_id)
        accumulator.update(data, mask, cells)
    classification = accumulator.finalise()
    lit = tuple(classification.lit)
    dark = tuple(classification.dark)
    successor = _sequence_successor(lit)
    reference_dark = tuple(c for c in dark if c != successor)

    # q bands over the kept region, and (q, χ) cells for D.
    kept_static = ~static
    q_lo, q_hi = float(q[kept_static].min()), float(q[kept_static].max())
    q_band = np.clip(((q - q_lo) / (q_hi - q_lo) * N_Q).astype(np.int64), 0, N_Q - 1)
    q_centres = q_lo + (np.arange(N_Q) + 0.5) * (q_hi - q_lo) / N_Q
    chi_lo = float(chi[kept_static].min())
    n_chi = int(np.ceil((float(chi[kept_static].max()) - chi_lo) / D_CHI_STEP_DEG)) + 1
    d_cell = np.floor((q - q_lo) / D_Q_STEP).astype(np.int64) * n_chi + np.floor(
        (chi - chi_lo) / D_CHI_STEP_DEG
    ).astype(np.int64)
    n_d_cells = int(d_cell[kept_static].max()) + 1

    sampled = evenly_spaced(all_trains, n_trains)
    position = np.searchsorted(all_trains, sampled) / max(all_trains.size - 1, 1)
    n_t = sampled.size
    sets = ("A", "K", "S")
    levels = {name: np.full((n_t, CELLS_PER_TRAIN, N_ASICS), np.nan) for name in sets}
    levels["A_iter10"] = np.full((n_t, CELLS_PER_TRAIN, N_ASICS), np.nan)
    levels["A_median_only"] = np.full((n_t, CELLS_PER_TRAIN, N_ASICS), np.nan)
    npix = {name: np.zeros((n_t, CELLS_PER_TRAIN, N_ASICS), np.int64) for name in sets}
    kept_per_asic = np.zeros((n_t, CELLS_PER_TRAIN, N_ASICS), np.int64)

    lit_index = {cell: i for i, cell in enumerate(lit)}
    n_lit = len(lit)
    variants = ("raw", "sub", "p", "p_half", "p_xgm", "p_levelK", "n")
    per_train = {v: np.zeros((n_t, N_Q)) for v in variants}
    occupancy = np.full((n_t, max(n_lit, 1)), np.nan)
    d_frame_median = np.full((n_t, max(n_lit, 1)), np.nan)
    d_sum = np.zeros(D_OCC_EDGES.size - 1)
    d_sumsq = np.zeros(D_OCC_EDGES.size - 1)
    d_n = np.zeros(D_OCC_EDGES.size - 1, np.int64)
    hist_lit = np.zeros(N_SPEC, np.int64)
    hist_dark = np.zeros(N_SPEC, np.int64)
    n_extreme = 0
    n_level_missing_pixels = 0

    npix_flat = MODULE_SHAPE[0] * MODULE_SHAPE[1]
    track = tuple(dark)  # per-pixel accumulators, every dark cell
    dark_sum = np.zeros((len(track), npix_flat))
    dark_sumsq = np.zeros((len(track), npix_flat))
    dark_n = np.zeros((len(track), npix_flat), np.int64)
    lit_sum = np.zeros(npix_flat)
    lit_n = np.zeros(npix_flat, np.int64)
    # cells.py's sigma_read, reproduced: unclipped x over the union mask.
    lumped = np.zeros(3)  # n, sum, sumsq
    after_level = np.zeros(3)  # same pixels, x - level, reference dark cells

    bit22 = None
    read_s = 0.0
    for t, train_id in enumerate(sampled.tolist()):
        tic = time.perf_counter()
        data, mask, cells = read(train_id)
        read_s += time.perf_counter() - tic
        order = np.argsort(cells)
        if not np.array_equal(cells[order], np.arange(CELLS_PER_TRAIN)):
            raise ValueError(f"train {train_id}: memory cells {cells.tolist()}")
        x = data[order].astype(np.float64)
        m = mask[order]
        if bit22 is None:
            bit22 = ((m >> 22) & 1).astype(bool).any(axis=0)

        finite = np.isfinite(x)
        good = ((m & np.uint32(cfg.mask_bits)) == 0) & finite
        kept = good & ~static & (np.abs(np.where(finite, x, 0)) <= cfg.max_abs_kev)
        n_extreme += int((good & ~static & ~kept).sum())
        masks = {"A": good, "K": good & ~static, "S": good & static}
        for name, chosen in masks.items():
            view = asic_view(np.where(chosen, x, np.nan))
            levels[name][t] = baseline(view)
            npix[name][t] = np.isfinite(view).sum(-1)
            if name == "A":
                levels["A_iter10"][t] = baseline(view, iterations=10)
                levels["A_median_only"][t] = baseline(view, iterations=0)
        kept_per_asic[t] = np.stack(
            [np.bincount(ASIC_ID[k], minlength=N_ASICS) for k in kept]
        )

        level_px = per_pixel(levels["A"][t])  # (16, 512, 1024)
        sub = x - level_px

        # dark cells: per-pixel accumulators and the two sigma decompositions
        for j, cell in enumerate(track):
            r = sub[cell]
            use = good[cell] & np.isfinite(r) & (np.abs(r) < DARK_CLIP_KEV)
            flat_r, flat_use = r.reshape(-1), use.reshape(-1)
            dark_sum[j] += np.where(flat_use, flat_r, 0.0)
            dark_sumsq[j] += np.where(flat_use, flat_r**2, 0.0)
            dark_n[j] += flat_use
            union_kept = good[cell] & ~static
            values = x[cell][union_kept]
            lumped += (values.size, values.sum(), (values**2).sum())
            if cell in reference_dark:
                residual = r[union_kept]
                residual = residual[np.isfinite(residual)]
                after_level += (residual.size, residual.sum(), (residual**2).sum())
                use_h = union_kept & np.isfinite(r)
                idx = ((r[use_h] - SPEC_LO) / SPEC_STEP).astype(np.int64)
                hist_dark += np.bincount(
                    idx[(idx >= 0) & (idx < N_SPEC)], minlength=N_SPEC
                )

        # lit cells: photons, ratios, D, spectrum
        level_k_px = per_pixel(levels["K"][t])
        for cell in lit:
            i = lit_index[cell]
            valid = kept[cell] & np.isfinite(level_px[cell])
            n_level_missing_pixels += int((kept[cell] & ~valid).sum())
            s = sub[cell][valid]
            p = count(s, energy)
            qb = q_band[valid]
            per_train["raw"][t] += np.bincount(qb, x[cell][valid], N_Q)
            per_train["sub"][t] += np.bincount(qb, s, N_Q)
            per_train["p"][t] += np.bincount(qb, p, N_Q)
            per_train["p_half"][t] += np.bincount(qb, count(s, energy / 2), N_Q)
            per_train["p_xgm"][t] += np.bincount(qb, count(s, XGM_ENERGY_KEV), N_Q)
            valid_k = valid & np.isfinite(level_k_px[cell])
            per_train["p_levelK"][t] += np.bincount(
                q_band[valid_k],
                count(x[cell][valid_k] - level_k_px[cell][valid_k], energy),
                N_Q,
            )
            per_train["n"][t] += np.bincount(qb, minlength=N_Q)
            occupancy[t, i] = p.mean() if p.size else np.nan

            dc_idx = d_cell[valid]
            n_c = np.bincount(dc_idx, minlength=n_d_cells)
            s1 = np.bincount(dc_idx, p, n_d_cells)
            s2 = np.bincount(dc_idx, p**2, n_d_cells)
            ok = n_c >= D_MIN_PIXELS
            mean = np.where(ok, s1 / np.maximum(n_c, 1), 0.0)
            var = np.where(ok, (s2 - n_c * mean**2) / np.maximum(n_c - 1, 1), 0.0)
            ok &= mean >= D_MIN_MEAN
            dispersion = var[ok] / mean[ok]
            if dispersion.size:
                d_frame_median[t, i] = np.median(dispersion)
                which = np.digitize(mean[ok], D_OCC_EDGES) - 1
                inside = (which >= 0) & (which < d_n.size)
                np.add.at(d_sum, which[inside], dispersion[inside])
                np.add.at(d_sumsq, which[inside], dispersion[inside] ** 2)
                np.add.at(d_n, which[inside], 1)

            idx = ((s - SPEC_LO) / SPEC_STEP).astype(np.int64)
            hist_lit += np.bincount(idx[(idx >= 0) & (idx < N_SPEC)], minlength=N_SPEC)
            flat_valid = valid.reshape(-1)
            lit_sum += np.where(flat_valid, sub[cell].reshape(-1), 0.0)
            lit_n += flat_valid

    result: dict[str, Any] = {
        "run": run,
        "detector": detector,
        "photon_energy_kev": energy,
        "n_trains_in_run": int(all_trains.size),
        "n_trains_sampled": int(n_t),
        "lit_cells": list(lit),
        "dark_cells": list(dark),
        "cell_after_lit_sequence": successor,
        "reference_dark_cells": list(reference_dark),
        "estimator": {
            "level_cut_kev": LEVEL_CUT,
            "level_window_kev": LEVEL_WINDOW,
            "iterations": ITERATIONS,
            "edge_px": EDGE,
            "pixel_sets": {
                "A": "data.mask good, finite (static .edf ignored)",
                "K": "A and kept by the static .edf",
                "S": "A and masked by the static .edf",
            },
        },
        "q_kept_range_nm": [q_lo, q_hi],
        "n_extreme_pixels_kept_region": n_extreme,
    }

    # 1. ASIC tiling against bit 22
    rows_full = np.flatnonzero(bit22.mean(axis=1) > 0.9).tolist()
    cols_full = np.flatnonzero(bit22.mean(axis=0) > 0.9).tolist()
    boundary_rows = [ASIC - 1, ASIC]
    boundary_cols = [ASIC * k + d for k in (1, 2, 3) for d in (-1, 0)]
    result["asic_tiling"] = {
        "bit22_rows_over_90pct": rows_full,
        "bit22_cols_over_90pct": cols_full,
        "expected_rows": boundary_rows,
        "expected_cols": boundary_cols,
        "matches": set(rows_full) - {0, MODULE_SHAPE[0] - 1} == set(boundary_rows)
        and set(cols_full) - {0, MODULE_SHAPE[1] - 1} == set(boundary_cols),
        "module_edges_flagged": {
            "rows": sorted(set(rows_full) & {0, MODULE_SHAPE[0] - 1}),
            "cols": sorted(set(cols_full) & {0, MODULE_SHAPE[1] - 1}),
        },
        "bit22_fraction": float(bit22.mean()),
        "bit22_outside_asic_edges": int((bit22 & ~IN_EDGE).sum()),
    }

    # 2. pixels per ASIC
    result["pixels_per_asic"] = {
        "inner_pixels_per_asic": ASIC_PIXELS_INNER,
        **{
            f"level_set_{name}_median": np.median(npix[name], axis=(0, 1))
            .astype(int)
            .tolist()
            for name in sets
        },
        "integrated_median": np.median(kept_per_asic, axis=(0, 1)).astype(int).tolist(),
        "level_set_K_min_over_frames": npix["K"].min(axis=(0, 1)).tolist(),
        "level_set_A_min_over_frames": npix["A"].min(axis=(0, 1)).tolist(),
    }

    level_a = levels["A"]
    # 3. dark cells
    if reference_dark:
        ref = list(reference_dark)
        dark_levels = {name: levels[name][:, ref] for name in sets}
        fluct = dark_levels["A"] - np.nanmean(dark_levels["A"], axis=0, keepdims=True)
        samples = fluct.reshape(-1, N_ASICS)
        usable = np.isfinite(samples).all(axis=0)
        corr = np.full((N_ASICS, N_ASICS), np.nan)
        if usable.sum() >= 2:
            corr[np.ix_(usable, usable)] = np.corrcoef(samples[:, usable].T)
        common = np.nanmean(fluct, axis=-1, keepdims=True)
        per_train_common = np.nanmean(fluct, axis=1)  # (t, asic)
        within = np.nanstd(fluct - per_train_common[:, None, :], axis=1, ddof=1)
        static_per_cell = np.nanmean(level_a, axis=0)  # (16, 8)
        result["dark"] = {
            "train_to_train_rms_kev": {
                name: _f(np.nanmedian(np.nanstd(dark_levels[name], axis=0, ddof=1), 0))
                for name in sets
            },
            "train_to_train_rms_all_asics_kev": _nanstats(
                np.nanstd(dark_levels["A"], axis=0, ddof=1)
            ),
            "asic_correlation": _f(corr),
            "asic_correlation_offdiag_median": _f(
                np.nanmedian(corr[~np.eye(N_ASICS, dtype=bool)])
            ),
            "module_common_variance_fraction": _f(
                1.0 - np.nanvar(fluct - common) / np.nanvar(fluct)
            ),
            "rms_after_removing_module_common_kev": _f(
                np.nanstd((fluct - common).reshape(-1, N_ASICS), axis=0, ddof=1)
            ),
            "between_train_rms_kev": _f(np.nanstd(per_train_common, axis=0, ddof=1)),
            "within_train_rms_kev": _f(np.nanmean(within, axis=0)),
            "static_level_per_cell_kev": _f(static_per_cell),
            "static_cell_to_cell_rms_reference_dark_kev": _f(
                np.nanstd(static_per_cell[ref], axis=0, ddof=1)
            ),
            "set_difference_K_minus_A_kev": _nanstats(
                dark_levels["K"] - dark_levels["A"]
            ),
            "set_difference_S_minus_A_kev": _nanstats(
                dark_levels["S"] - dark_levels["A"]
            ),
            "iteration10_minus_iteration3_kev": _nanstats(
                levels["A_iter10"][:, ref] - dark_levels["A"]
            ),
            "median_only_minus_iteration3_kev": _nanstats(
                levels["A_median_only"][:, ref] - dark_levels["A"]
            ),
        }

        # ADC groups and per-pixel offsets, from the per-pixel accumulators
        ref_rows = [track.index(c) for c in ref]
        offset = dark_sum[ref_rows].sum(0) / np.maximum(dark_n[ref_rows].sum(0), 1)
        has = dark_n[ref_rows].sum(0) > 0
        offset = np.where(has, offset, np.nan).reshape(MODULE_SHAPE)

        def group_offsets(image: np.ndarray, region: np.ndarray) -> np.ndarray:
            groups = np.full(32, np.nan)
            for g in range(32):
                sel = region & (ADC_ID == g) & np.isfinite(image)
                if sel.sum() > 100:
                    asic = int(ASIC_ID[sel][0])
                    whole = region & (ASIC_ID == asic) & np.isfinite(image)
                    groups[g] = image[sel].mean() - image[whole].mean()
            return groups

        region_a = np.isfinite(offset)
        groups_a = group_offsets(offset, region_a)
        per_cell_groups = []
        for c in ref:
            j = track.index(c)
            image = np.where(
                dark_n[j] > 0, dark_sum[j] / np.maximum(dark_n[j], 1), np.nan
            ).reshape(MODULE_SHAPE)
            per_cell_groups.append(group_offsets(image, region_a))
        # Per pixel and cell over trains: what is left once the frame's ASIC
        # level and the pixel's own offset are both taken out.
        temporal = np.full((len(ref), npix_flat), np.nan)
        for k, c in enumerate(ref):
            j = track.index(c)
            n = dark_n[j]
            ok = n >= 10
            var = (dark_sumsq[j][ok] - dark_sum[j][ok] ** 2 / n[ok]) / (n[ok] - 1)
            temporal[k, ok] = np.sqrt(np.maximum(var, 0))
        kept_flat = (~static).reshape(-1)
        n_l, s_l, ss_l = lumped
        n_a, s_a, ss_a = after_level
        result["dark"]["adc_group_offsets_kev"] = _f(groups_a)
        result["dark"]["adc_group_offsets_stats_kev"] = _nanstats(groups_a)
        result["dark"]["adc_group_offset_cell_to_cell_rms_kev"] = _f(
            np.nanmedian(np.nanstd(np.stack(per_cell_groups), axis=0, ddof=1))
        )
        result["dark"]["pixel_offset_rms_kept_kev"] = _f(
            np.nanstd(offset.reshape(-1)[kept_flat])
        )
        result["dark"]["pixel_offset_rms_all_good_kev"] = _f(np.nanstd(offset))
        result["dark"]["sigma_decomposition_kev"] = {
            "lumped_as_cells_py_all_dark_cells": _f(
                np.sqrt(ss_l / n_l - (s_l / n_l) ** 2)
            ),
            "lumped_mean_kev": _f(s_l / n_l),
            "after_per_asic_level_reference_dark": _f(
                np.sqrt(ss_a / n_a - (s_a / n_a) ** 2)
            ),
            "pure_per_pixel_per_cell_temporal_median_kept": _f(
                np.nanmedian(temporal[:, kept_flat])
            ),
            "pure_per_pixel_per_cell_temporal_mean_kept": _f(
                np.nanmean(temporal[:, kept_flat])
            ),
            "note": (
                "lumped: one std over pixel x dark cell x train around one mean, "
                "union mask, unclipped (waxs/cells.py). after_per_asic_level: the "
                "same pixels minus each frame's ASIC level, reference dark cells. "
                f"pure: per pixel and cell over trains, |x - level| < {DARK_CLIP_KEV}"
            ),
        }
        result["dark"]["dark_spectrum_counts"] = hist_dark.tolist()

        # 4. the cell after the lit sequence
        if successor is not None and successor in track:
            lit_image = np.where(lit_n > 0, lit_sum / np.maximum(lit_n, 1), np.nan)
            slopes, means = {}, {}
            for j, c in enumerate(track):
                image = np.where(
                    dark_n[j] > 0, dark_sum[j] / np.maximum(dark_n[j], 1), np.nan
                )
                image = np.where(kept_flat, image, np.nan)
                slopes[str(c)] = _f(
                    _slope(image, np.where(kept_flat, lit_image, np.nan))
                )
                means[str(c)] = _f(np.nanmean(image))
            result["leftover"] = {
                "cell": successor,
                "level_minus_reference_dark_kev": _f(
                    np.nanmean(
                        level_a[:, successor]
                        - np.nanmean(level_a[:, list(reference_dark)], axis=1),
                        axis=0,
                    )
                ),
                "level_minus_reference_dark_rms_kev": _f(
                    np.nanstd(
                        level_a[:, successor]
                        - np.nanmean(level_a[:, list(reference_dark)], axis=1),
                        axis=0,
                        ddof=1,
                    )
                ),
                "slope_vs_lit_image_per_cell": slopes,
                "mean_residual_kept_per_cell_kev": means,
                "lit_mean_residual_kept_kev": _f(np.nanmean(lit_image[kept_flat])),
            }
    else:
        result["dark"] = {"note": "no dark cell in this run: every cell is lit"}

    # 5. the estimator on lit frames
    lit_list = list(lit)
    occ_train = np.nanmean(occupancy, axis=1)
    lit_block: dict[str, Any] = {
        "occupancy_photons_per_pixel": _nanstats(occupancy),
        "train_to_train_rms_kev": {
            name: _f(
                np.nanmedian(np.nanstd(levels[name][:, lit_list], axis=0, ddof=1), 0)
            )
            for name in sets
        },
        "against_shadow_S_kev": {
            name: _nanstats(levels[name][:, lit_list] - levels["S"][:, lit_list])
            for name in ("A", "K")
        },
        "iteration10_minus_iteration3_kev": _nanstats(
            levels["A_iter10"][:, lit_list] - levels["A"][:, lit_list]
        ),
    }
    if reference_dark:
        ref = list(reference_dark)
        against = {}
        for name in sets:
            dark_mean = np.nanmean(levels[name][:, ref], axis=1, keepdims=True)
            diff = levels[name][:, lit_list] - dark_mean  # (t, lit, asic)
            tracking = diff - np.nanmean(diff, axis=0, keepdims=True)
            terciles = np.nanpercentile(occ_train, [33.3, 66.7])
            groups = np.digitize(occ_train, terciles)
            against[name] = {
                "bias_kev": _nanstats(diff),
                "bias_per_asic_kev": _f(np.nanmean(diff, axis=(0, 1))),
                "bias_per_lit_cell_kev": {
                    str(c): _f(np.nanmean(diff[:, i])) for i, c in enumerate(lit_list)
                },
                "tracking_rms_kev": _f(np.nanstd(tracking)),
                "tracking_rms_per_asic_kev": _f(
                    np.nanstd(tracking.reshape(-1, N_ASICS), axis=0, ddof=1)
                ),
                "bias_by_occupancy_tercile_kev": [
                    _f(np.nanmean(diff[groups == g])) for g in range(3)
                ],
                # Does the lit-frame level follow the dark cells' train-to-train
                # movement? Near 1 means it tracks the common mode.
                "corr_with_dark_fluctuation": _f(
                    _corr(
                        _demean_trains(levels[name][:, lit_list]).ravel(),
                        np.broadcast_to(_demean_trains(dark_mean), diff.shape).ravel(),
                    )
                ),
            }
        lit_block["against_same_train_dark_kev"] = against
        lit_block["occupancy_terciles"] = _f(np.nanpercentile(occ_train, [33.3, 66.7]))
    fluct_lit = level_a[:, lit_list] - np.nanmean(
        level_a[:, lit_list], 0, keepdims=True
    )
    samples = fluct_lit.reshape(-1, N_ASICS)
    usable = np.isfinite(samples).all(axis=0)
    corr_lit = np.full((N_ASICS, N_ASICS), np.nan)
    if usable.sum() >= 2:
        corr_lit[np.ix_(usable, usable)] = np.corrcoef(samples[:, usable].T)
    lit_block["asic_correlation"] = _f(corr_lit)
    lit_block["asic_correlation_offdiag_median"] = _f(
        np.nanmedian(corr_lit[~np.eye(N_ASICS, dtype=bool)])
    )
    common_lit = np.nanmean(fluct_lit, axis=-1, keepdims=True)
    lit_block["module_common_variance_fraction"] = _f(
        1.0 - np.nanvar(fluct_lit - common_lit) / np.nanvar(fluct_lit)
    )
    lit_block["static_level_per_lit_cell_kev"] = {
        str(c): _f(np.nanmean(level_a[:, c], axis=0)) for c in lit_list
    }
    result["lit"] = lit_block

    # 6. over-dispersion
    with np.errstate(invalid="ignore", divide="ignore"):
        d_mean = d_sum / d_n
        d_std = np.sqrt(d_sumsq / d_n - d_mean**2)
    result["dispersion"] = {
        "cell": f"{D_Q_STEP} nm^-1 x {D_CHI_STEP_DEG} deg, >= {D_MIN_PIXELS} px, "
        f"mean >= {D_MIN_MEAN}",
        "frame_median_D": _nanstats(d_frame_median),
        "occupancy_bin_edges": _f(D_OCC_EDGES),
        "mean_D_per_occupancy_bin": _f(d_mean),
        "std_D_per_occupancy_bin": _f(d_std),
        "n_cells_per_occupancy_bin": d_n.tolist(),
        "poisson_expectation": 1.0,
    }

    # 7-8. ratios
    tot = {v: per_train[v].sum(0) for v in variants}
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_raw_q = tot["raw"] / (energy * tot["p"])
        ratio_sub_q = tot["sub"] / (energy * tot["p"])
        half_q = tot["p_half"] / tot["p"]
        ratio_sub_t = per_train["sub"].sum(1) / (energy * per_train["p"].sum(1))
        ratio_raw_t = per_train["raw"].sum(1) / (energy * per_train["p"].sum(1))
    populated = tot["n"] > 0
    deciles = np.array_split(np.arange(n_t), 10)
    result["ratios"] = {
        "q_centres_nm": _f(q_centres),
        "pixels_per_band": tot["n"].astype(int).tolist(),
        "kev_raw_over_E_photons_per_q": _f(ratio_raw_q),
        "kev_sub_over_E_photons_per_q": _f(ratio_sub_q),
        "kev_raw_over_E_photons": _f(tot["raw"].sum() / (energy * tot["p"].sum())),
        "kev_sub_over_E_photons": _f(tot["sub"].sum() / (energy * tot["p"].sum())),
        "kev_sub_over_E_photons_q_p02_p98": _f(
            np.nanpercentile(ratio_sub_q[populated], [2, 50, 98])
        ),
        "kev_raw_over_E_photons_q_p02_p98": _f(
            np.nanpercentile(ratio_raw_q[populated], [2, 50, 98])
        ),
        "half_energy_count_ratio": _f(tot["p_half"].sum() / tot["p"].sum()),
        "half_energy_count_ratio_q_p02_p98": _f(
            np.nanpercentile(half_q[populated], [2, 50, 98])
        ),
        "xgm_9p000_over_9p04_count_ratio": _f(tot["p_xgm"].sum() / tot["p"].sum()),
        "levelK_over_levelA_count_ratio": _f(tot["p_levelK"].sum() / tot["p"].sum()),
        "along_run": {
            "train_position": _f(position),
            "occupancy": _f(occ_train),
            "kev_sub_over_E_photons": _f(ratio_sub_t),
            "kev_raw_over_E_photons": _f(ratio_raw_t),
            "corr_sub_ratio_vs_occupancy": _f(_corr(ratio_sub_t, occ_train)),
            "corr_raw_ratio_vs_occupancy": _f(_corr(ratio_raw_t, occ_train)),
            "slope_sub_ratio_per_occupancy": _f(_slope(ratio_sub_t, occ_train)),
            "occupancy_first_last_decile": _f(
                [np.nanmean(occ_train[deciles[0]]), np.nanmean(occ_train[deciles[-1]])]
            ),
            "sub_ratio_first_last_decile": _f(
                [
                    np.nanmean(ratio_sub_t[deciles[0]]),
                    np.nanmean(ratio_sub_t[deciles[-1]]),
                ]
            ),
        },
        "n_level_missing_pixels": int(n_level_missing_pixels),
    }

    # 9. spectrum
    result["spectrum"] = {
        "edges_kev": [SPEC_LO, SPEC_HI, SPEC_STEP],
        "lit": _peak(hist_lit, energy),
        "lit_counts": hist_lit.tolist(),
    }

    out = Path(out_dir) / f"w7_r{run:04d}_{detector}_arrays.npz"
    np.savez_compressed(
        out,
        train_ids=sampled,
        position=position,
        cells=np.arange(CELLS_PER_TRAIN),
        **{f"level_{k}": v for k, v in levels.items()},
        **{f"npix_{k}": v for k, v in npix.items()},
        kept_per_asic=kept_per_asic,
        occupancy=occupancy,
        d_frame_median=d_frame_median,
        **{f"per_train_{k}": v for k, v in per_train.items()},
        q_centres=q_centres,
        hist_lit=hist_lit,
        hist_dark=hist_dark,
    )
    result["arrays"] = str(out)
    result["timing_s"] = {"total": time.perf_counter() - started, "read": read_s}
    return result


def _safe(args: tuple) -> dict[str, Any]:
    try:
        return measure(*args)
    except Exception as error:  # noqa: BLE001 - recorded with its traceback
        formatted = traceback.format_exc()
        print(formatted, flush=True)
        return {
            "run": args[1],
            "detector": args[2],
            "error": repr(error),
            "traceback": formatted,
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--runs", type=int, nargs="+", default=[423, 370])
    parser.add_argument("--detectors", nargs="+", default=["jf1", "jf2"])
    parser.add_argument("--trains", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gpfs/exfel/exp/MID/202601/p010400/scratch/s2_jf_photonise/w7"),
    )
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (args.proposal, run, det, args.trains, str(args.out_dir))
        for run in args.runs
        for det in args.detectors
    ]
    report = report_header(
        proposal=args.proposal,
        runs=args.runs,
        detectors=args.detectors,
        trains=args.trains,
        results=[],
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=min(args.processes, len(jobs)), mp_context=get_context("spawn")
    ) as pool:
        report["results"] = list(pool.map(_safe, jobs))
    report["wall_s"] = time.perf_counter() - started
    write_report(report, args.json, "w7_photonize_check")
    failed = [r for r in report["results"] if "error" in r]
    for r in report["results"]:
        tag = "FAILED" if "error" in r else "ok"
        print(f"r{r['run']:04d} {r['detector']}: {tag}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
