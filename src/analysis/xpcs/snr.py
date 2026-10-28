"""XPCS signal-to-noise per run and per 30 s slice: the keep criterion and score.

The analytical multispeckle SNR of one g2 point (Möller, Sprung, Madsen & Gutt 2019,
IUCrJ 6:794, eq. 8, after Lumma et al. 2000 and Falus et al. 2006), as it applies to
the hit-train XPCS:

    SNR = β · k̄ · sqrt(N_pix · (F − 1) · φ · n) / √2

k̄ is photons per pixel per frame, N_pix the pixels of the ROI, F the frames per
train (F − 1 pairs at the shortest lag), φ the fraction of those pairs that survive
extra-speckle's kbar filter, n the trains, and √2 the extra variance of subtracting
TTCF_off.

Every input is read from the run's hit-train results file
(`analysis.xpcs.reader.XpcsHits`), which knows only the droplet hits, so the
slices are anchored to the first usable hit. The TTCF is reduced to per-train sums
and counts of its finite diagonal entries, from which g2 over any set of trains
follows exactly.

β is one table for every run, `REFERENCE_BETA`, the τ → 0 intercepts of the runs
whose g2 decays almost completely inside the lag window. A β per run, or one that
moves with the set of runs it came from, would make the verdict depend on that
set. A triage report's ``beta`` block (`load_beta`) can still be passed in to
compare. An unknown β gives SNR NaN and ``extractable`` None, never False.

`xpcs_score` is the per-run diagnostic score: the SNR and the k̄ per XPCS ROI,
both over the run's droplet hits only, and the number of ROIs whose mean k̄
reaches `KBAR_MIN`.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from analysis.xpcs.config import DEFAULT_XPCS_ROOT, roi_kinds
from analysis.xpcs.estimators import N_BLOCKS
from analysis.xpcs.reader import ES_KBAR_MIN, XpcsHits, read

__all__ = [
    "ES_REFERENCE_BETA",
    "REFERENCE",
    "KBAR_MIN",
    "MIN_SLICE_TRAINS",
    "N_BLOCKS",
    "OFF_SUBTRACTION",
    "REFERENCE_BETA",
    "REFERENCE_BETA_RUNS",
    "ROI_MATCH_A",
    "SLICE_TRAINS",
    "SNR_MIN",
    "Beta",
    "DecayFit",
    "RunMeasurement",
    "beta_for",
    "fit_decay",
    "fixed_beta",
    "jackknife_g2",
    "kbar_rois",
    "load_beta",
    "measure_run",
    "open_results",
    "roi_edges_from",
    "slice_index",
    "snr",
    "summarise",
    "xpcs_score",
    "xpcs_snr",
]

#: 30 s at 10 Hz. A shorter final part joins the slice before it.
SLICE_TRAINS, MIN_SLICE_TRAINS = 300, 150
#: Per-slice SNR a run must reach once to count as extractable.
SNR_MIN = 3.0
#: Noise factor of the pipeline's TTCF_raw − TTCF_off: two estimates, one variance each.
OFF_SUBTRACTION = float(np.sqrt(2.0))
#: Mean photons per pixel per frame below which a frame drops out of a pair:
#: extra-speckle's default ``kbar_threshold``.
KBAR_MIN = ES_KBAR_MIN

#: Runs whose g2 decays almost completely inside the 60 µs lag window, a late PEG 6K
#: droplet. Their τ → 0 intercepts fix one β for every run, rather than a contrast
#: fitted to each run.
REFERENCE_BETA_RUNS = (450, 493, 494, 495, 496)
#: β per extra-speckle ROI centre [Å⁻¹] (`analysis.xpcs.config.ES_RINGS_NM`) as
#: (median, p16, p84) of the fitted intercept A over `REFERENCE_BETA_RUNS`, from
#: this module's `measure_run` and `fit_decay`. The photon statistics
#: (`analysis.diagnostics.photon_stats`) bin on these ROIs.
ES_REFERENCE_BETA: dict[float, tuple[float, float, float]] = {
    0.02875: (0.047803, 0.047412, 0.048816),
    0.04: (0.039039, 0.038549, 0.039383),
    0.0505: (0.029474, 0.029348, 0.029969),
    0.0605: (0.022055, 0.021190, 0.027577),
    0.067: (0.019076, 0.018810, 0.019468),
    0.072: (0.019117, 0.018546, 0.021378),
    0.0805: (0.018556, 0.017375, 0.018862),
}
#: β per ROI centre [Å⁻¹] as (median, p16, p84) of the fitted intercept A over
#: `REFERENCE_BETA_RUNS`, for the pass's rings (`analysis.xpcs.config.RINGS_NM`),
#: from this module's `measure_run` and `fit_decay` on those runs merged from their
#: photon stores. It also carries `ES_REFERENCE_BETA`, whose centres differ from
#: these by more than `ROI_MATCH_A`, so a lookup by centre gives each ring set its
#: own β. `reference_beta` re-measures it.
REFERENCE_BETA: dict[float, tuple[float, float, float]] = {
    **ES_REFERENCE_BETA,
    0.02475: (0.048842, 0.048317, 0.050663),
    0.02925: (0.048951, 0.048686, 0.050053),
    0.03375: (0.045297, 0.044749, 0.045758),
    0.03825: (0.040283, 0.039870, 0.040961),
    0.04275: (0.037710, 0.036988, 0.037940),
    0.04725: (0.033020, 0.032943, 0.034061),
    0.05175: (0.028717, 0.028545, 0.029118),
    0.07925: (0.018553, 0.016734, 0.018980),
    0.08375: (0.017423, 0.016043, 0.018863),
}
#: Two ROI centres within this [Å⁻¹] are the same ROI (DAMNIT stores float centres).
ROI_MATCH_A = 1e-6


def jackknife_g2(
    sums: np.ndarray, counts: np.ndarray, n_blocks: int = N_BLOCKS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(g2, σ_g2, leave-one-out g2)`` over contiguous blocks of the given trains.

    g2(k) pools every finite entry at lag k, Σ sum / Σ count. Blocks are contiguous so
    a slow drift inflates σ rather than hiding in it.
    """
    n = sums.shape[0]
    blocks = np.array_split(np.arange(n), min(n_blocks, n))
    s_b = np.stack([sums[b].sum(axis=0) for b in blocks])
    c_b = np.stack([counts[b].sum(axis=0) for b in blocks]).astype(np.float64)
    s, c = s_b.sum(axis=0), c_b.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g2 = s / c
        loo = (s - s_b) / (c - c_b)
    b = len(blocks)
    sigma = np.sqrt(
        (b - 1) / b * np.nansum((loo - np.nanmean(loo, axis=0)) ** 2, axis=0)
    )
    sigma[~np.isfinite(g2)] = np.nan
    return g2, sigma, loo


@dataclass(frozen=True)
class DecayFit:
    """g2(τ) = A·exp(−2Γτ) + C, extra-speckle's model, with jackknife errors."""

    amplitude: float
    sigma_amplitude: float
    rate: float  # Γ [1/s]
    sigma_rate: float
    offset: float
    converged: bool


def _decay(tau, a, rate, c):
    return a * np.exp(-2.0 * rate * tau) + c


def _fit_once(tau, g2, sigma, p0):
    ok = np.isfinite(g2) & np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < 5:
        return None
    lower = [0.0, 1.0 / (100.0 * tau[-1]), -1.0]
    upper = [1.0, 10.0 / tau[0], 1.0]
    p0 = np.clip(p0, np.add(lower, 1e-12), np.subtract(upper, 1e-12))
    try:
        popt, _ = curve_fit(
            _decay, tau[ok], g2[ok], p0=p0, sigma=sigma[ok], bounds=(lower, upper),
            maxfev=20000,
        )  # fmt: skip
    except (RuntimeError, ValueError):
        return None
    return popt


def fit_decay(tau, g2, sigma, loo=None) -> DecayFit:
    """Weighted fit of one ROI's g2, with σ_A and σ_Γ from refitting ``loo`` rows."""
    tau, g2, sigma = (np.asarray(x, dtype=np.float64) for x in (tau, g2, sigma))
    quarter = tau[max(len(tau) // 4, 1)]
    p0 = [max(np.nanmean(g2[:3]), 1e-5), 1.0 / (4.0 * quarter), np.nanmean(g2[-5:])]
    popt = _fit_once(tau, g2, sigma, p0)
    if popt is None:
        nan = float("nan")
        return DecayFit(nan, nan, nan, nan, nan, False)
    sig_a = sig_rate = float("nan")
    if loo is not None and len(loo) > 1:
        refits = [_fit_once(tau, row, sigma, popt) for row in loo]
        refits = np.array([r for r in refits if r is not None])
        if len(refits) > 1:
            b = len(loo)
            spread = ((refits - refits.mean(axis=0)) ** 2).sum(axis=0)
            sig_a, sig_rate = np.sqrt((b - 1) / b * spread[:2])
    return DecayFit(float(popt[0]), float(sig_a), float(popt[1]), float(sig_rate),
                    float(popt[2]), True)  # fmt: skip


def slice_index(
    train_ids: np.ndarray,
    first_train: int,
    last_train: int,
    slice_trains: int = SLICE_TRAINS,
    min_trains: int = MIN_SLICE_TRAINS,
) -> np.ndarray:
    """Each train's slice, by train id (wall time): a dropped train shifts nothing."""
    offset = np.asarray(train_ids, dtype=np.int64) - int(first_train)
    idx = offset // slice_trains
    span = int(last_train) - int(first_train) + 1
    n_full, tail = divmod(span, slice_trains)
    if tail and tail < min_trains and n_full >= 1:
        idx = np.minimum(idx, n_full - 1)
    return idx


def snr(beta, kbar, npix, n_frames, phi, n_trains):
    """The analytical SNR of one g2 point at the shortest lag (module docstring)."""
    pairs = np.asarray(npix) * (n_frames - 1) * np.asarray(phi) * np.asarray(n_trains)
    return np.asarray(beta) * np.asarray(kbar) * np.sqrt(pairs) / OFF_SUBTRACTION


@dataclass
class RunMeasurement:
    """Everything the SNR verdict needs from one run, before β is known."""

    run: int
    roi_q_A: np.ndarray  # (R,) centres as DAMNIT stores them
    edges_nm: np.ndarray  # (R, 2)
    kinds: list[str]
    n_frames: int  # F
    rep_rate_hz: float
    npix: np.ndarray  # (R,)
    n_usable: int
    kbar: np.ndarray  # (R,) over the usable trains
    phi: np.ndarray  # (R,) fraction of finite k = 1 pairs
    tau: np.ndarray  # (K,)
    g2: np.ndarray  # (R, K)
    sigma_g2: np.ndarray  # (R, K)
    fits: list[DecayFit]
    rho: np.ndarray  # (R,) measured / analytical noise at k = 1…3
    slices: list[dict] = field(default_factory=list)  # slice, n_usable, kbar, phi


def _reduce(
    run: int,
    roi_q_A: np.ndarray,
    edges: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
    kbar_t: np.ndarray,
    train_ids: np.ndarray,
    npix: np.ndarray,
    f_frames: int,
    k_max: int,
    rep_rate_hz: float,
    span: tuple[int, int] | None,
    n_blocks: int,
) -> RunMeasurement:
    """The usable trains' diagonal sums, k̄ and N_pix reduced to a `RunMeasurement`.

    ``span`` is the first and last train id the slices are
    anchored to; None for the usable trains' own.
    """
    tau = np.arange(1, k_max + 1) / rep_rate_hz
    n = train_ids.size
    n_roi = len(edges)
    phi = counts[:, :, 0].sum(axis=0) / max(n * (f_frames - 1), 1)
    kbar = np.nanmean(kbar_t, axis=0) if n else np.full(n_roi, np.nan)
    if n >= 2:
        # A ROI whose every pair fell to the kbar filter is all NaN: its g2, σ and
        # fit come back NaN, which is the answer, not a warning per ROI and block.
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            g2, sigma, loo = jackknife_g2(
                sums[:, :, :k_max], counts[:, :, :k_max], n_blocks
            )
            fits = [fit_decay(tau, g2[r], sigma[r], loo[:, r]) for r in range(n_roi)]
            analytic = OFF_SUBTRACTION / (
                kbar * np.sqrt(npix * (f_frames - 1) * phi * n)
            )
            rho = np.nanmean(sigma[:, :3], axis=1) / analytic
    else:
        g2 = sigma = np.full((n_roi, k_max), np.nan)
        nan = float("nan")
        fits = [DecayFit(nan, nan, nan, nan, nan, False)] * n_roi
        rho = np.full(n_roi, np.nan)

    slices = []
    if n:
        first, last = span if span is not None else (train_ids[0], train_ids[-1])
        idx = slice_index(train_ids, int(first), int(last))
        for s in np.unique(idx):
            m = idx == s
            slices.append(
                {
                    "slice": int(s),
                    "n_usable": int(m.sum()),
                    "kbar": np.nanmean(kbar_t[m], axis=0),
                    "phi": counts[m, :, 0].sum(axis=0) / (m.sum() * (f_frames - 1)),
                }
            )
    return RunMeasurement(
        run=run, roi_q_A=roi_q_A, edges_nm=edges, kinds=roi_kinds(edges),
        n_frames=f_frames, rep_rate_hz=rep_rate_hz, npix=npix, n_usable=int(n),
        kbar=kbar, phi=phi, tau=tau, g2=g2, sigma_g2=sigma, fits=fits, rho=rho,
        slices=slices,
    )  # fmt: skip


def measure_run(
    run: int,
    hits: XpcsHits,
    *,
    select: np.ndarray | None = None,
    span: tuple[int, int] | None = None,
    kbar_min: float = KBAR_MIN,
    n_blocks: int = N_BLOCKS,
) -> RunMeasurement:
    """Reduce one run's hit-train results file to a `RunMeasurement`.

    Usable trains are the file's usable hits with a baseline (a train with no
    ``TTCF_off`` has no ``raw − off``); ``select`` (train ids, e.g. the ``on``
    trains of a mesh run) narrows them further. The diagonals come from
    `XpcsHits.diagonals` with extra-speckle's k̄ filter, k̄ from `XpcsHits.kbar`
    (the mean over frames, then trains), N_pix from the ring pixel counts the
    pass stores per train (`XpcsHits.pixels`, the median over the usable
    trains), and the lag from the file's pulse pattern. Nothing is read from
    DAMNIT.

    :param span: first and last train id the 30 s slices are anchored to; the
        file knows only the hits, so the default is the first and last usable
        hit.
    """
    chosen = hits.train_ids[hits.usable & hits.has_off]
    if select is not None:
        chosen = np.intersect1d(chosen, np.asarray(select, dtype=np.uint64))
    diag = hits.diagonals(kbar_min=kbar_min, trains=chosen)
    kbar_t = (
        hits.kbar(diag.train_ids).mean("frame", skipna=True).values
        if diag.train_ids.size
        else np.empty((0, hits.n_rings))
    )
    pixels = hits.pixels(diag.train_ids)
    npix = np.median(pixels, axis=0) if pixels.size else np.full(hits.n_rings, np.nan)
    return _reduce(
        run, diag.roi_q_A, roi_edges_from(hits.bounds_nm, hits.n_rings), diag.sums,
        diag.counts, kbar_t, diag.train_ids, npix, hits.n_frames, hits.lags().size,
        1.0 / hits.period_s, span, n_blocks,
    )  # fmt: skip


def roi_edges_from(bounds_nm: np.ndarray, n_rings: int) -> np.ndarray:
    """(R, 2) ROI edges in nm⁻¹ from a results file's ring bounds."""
    b = np.asarray(bounds_nm, dtype=np.float64).reshape(-1, 2)
    if len(b) != n_rings:
        return np.full((n_rings, 2), np.nan)  # rings from a masks file: no q edges
    return b


@dataclass(frozen=True)
class Beta:
    value: float
    n_eligible: int
    p16: float
    p84: float
    runs: tuple[int, ...]


#: `REFERENCE_BETA` as `Beta` values, each from every reference run.
REFERENCE: dict[float, Beta] = {
    q: Beta(value, len(REFERENCE_BETA_RUNS), p16, p84, REFERENCE_BETA_RUNS)
    for q, (value, p16, p84) in REFERENCE_BETA.items()
}


def beta_for(roi_q_A: np.ndarray, table: dict[float, Beta]) -> dict[int, Beta]:
    """β of each ROI by index, matched on the centre [Å⁻¹]; NaN for one ``table``
    lacks."""
    nan = float("nan")
    out = {}
    for r, q in enumerate(np.asarray(roi_q_A, dtype=np.float64)):
        hit = [b for key, b in table.items() if abs(key - q) < ROI_MATCH_A]
        out[r] = hit[0] if hit else Beta(nan, 0, nan, nan, ())
    return out


def fixed_beta(roi_q_A: np.ndarray) -> dict[int, Beta]:
    """`REFERENCE_BETA` by ROI index (`beta_for`)."""
    return beta_for(roi_q_A, REFERENCE)


def summarise(m: RunMeasurement, beta: dict[int, Beta], snr_min: float = SNR_MIN):
    """The run's keep verdict: per-slice SNR over plain ROIs, best slice, run SNR.

    ``extractable`` is None, not False, when no slice has a finite SNR in any plain
    ROI (β unknown, or no usable train). An unknown SNR is not a low one.
    """
    plain = [r for r, k in enumerate(m.kinds) if k == "plain"]
    b = np.array([beta[r].value for r in range(len(m.kinds))])
    per_roi_run = snr(b, m.kbar, m.npix, m.n_frames, m.phi, m.n_usable)
    rows = []
    for sl in m.slices:
        values = snr(b, sl["kbar"], m.npix, m.n_frames, sl["phi"], sl["n_usable"])
        on_plain = values[plain]
        j = int(np.nanargmax(on_plain)) if np.isfinite(on_plain).any() else None
        rows.append(
            {
                "slice": sl["slice"],
                "n_usable": sl["n_usable"],
                "snr": values,
                "best": float(on_plain[j]) if j is not None else float("nan"),
                "best_roi": plain[j] if j is not None else None,
            }
        )
    finite = [r for r in rows if np.isfinite(r["best"])]
    top = max(finite, key=lambda r: r["best"]) if finite else None
    best_snr = top["best"] if top else float("nan")
    n_ext = sum(r["best"] >= snr_min for r in finite)
    return {
        "run": m.run,
        "n_usable": m.n_usable,
        "n_slices": len(rows),
        "n_extractable_slices": int(n_ext),
        "extractable": bool(n_ext >= 1) if finite else None,
        "best_snr": best_snr,
        "best_slice": top["slice"] if top else None,
        "best_roi_q_A": float(m.roi_q_A[top["best_roi"]]) if top else None,
        "snr_run": {float(m.roi_q_A[r]): float(per_roi_run[r]) for r in plain},
        "t_needed_s": (
            SLICE_TRAINS / 10.0 * (snr_min / best_snr) ** 2
            if np.isfinite(best_snr) and best_snr > 0
            else float("inf")
        ),  # fmt: skip
        "slices": rows,
    }


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
    n_blocks: int = N_BLOCKS,
) -> dict[str, Any]:
    """One run's SNR per plain ROI and 30 s slice, and its keep verdict.

    :param beta: None for `REFERENCE_BETA`, or `load_beta` of a triage XPCS report.
    :param hits: an open results file, else the run's under ``output_root``.
    :param select: train ids to restrict to (e.g. a mesh run's `on` trains).
    :param span: first and last train id the slices are anchored to (default: the
        first and last usable hit).
    :param n_blocks: jackknife blocks of the g2 errors and of the fit's.
    :returns: ``summary`` (the `summarise` dict without its slices), ``slices`` (a
        DataFrame: slice, start in s, usable trains, best plain SNR, SNR per ROI),
        ``rois`` (a DataFrame per ROI: kind, k̄, N_pix, φ, ρ, the fit, the β used and
        the run SNR) and ``measurement`` (the `RunMeasurement`).
    :raises LookupError: the run has no results file.
    """
    import pandas as pd

    opened = open_results(run, proposal=proposal, output_root=output_root, hits=hits)
    m = measure_run(run, opened, select=select, span=span, n_blocks=n_blocks)
    b = beta_for(m.roi_q_A, REFERENCE if beta is None else beta)
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
    kinds = roi_kinds(roi_edges_from(opened.bounds_nm, opened.n_rings))
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
    n_blocks: int = N_BLOCKS,
) -> dict:
    """The per-run XPCS diagnostic score, over the hits in the run's results file.

    :returns: ``n_hits`` (the file's hit set), ``hits`` (its `Coverage` as a dict),
        ``n_rois`` and ``n_plain_rois`` (the file's rings, and those that touch no
        Bragg window), ``kbar`` (the `kbar_rois` table over the usable hits),
        ``n_rois_kbar_ok`` and ``n_plain_rois_kbar_ok`` (None without a usable
        hit), and ``snr`` (`xpcs_snr`'s summary, None without a usable hit).
    :raises LookupError: the run has no results file.
    """
    opened = open_results(run, proposal=proposal, output_root=output_root, hits=hits)
    coverage = opened.coverage()
    usable = coverage.n_usable > 0
    kbar = kbar_rois(run, hits=opened) if usable else None
    snr = (
        xpcs_snr(run, beta, hits=opened, n_blocks=n_blocks)["summary"]
        if usable
        else None
    )
    kinds = roi_kinds(roi_edges_from(opened.bounds_nm, opened.n_rings))
    return {
        "run": int(run),
        "n_hits": coverage.n_hits,
        "hits": dataclasses.asdict(coverage),
        "n_rois": len(kinds),
        "n_plain_rois": kinds.count("plain"),
        "kbar": kbar,
        "n_rois_kbar_ok": None if kbar is None else kbar.attrs["n_above"],
        "n_plain_rois_kbar_ok": None if kbar is None else kbar.attrs["n_plain_above"],
        "snr": snr,
    }
