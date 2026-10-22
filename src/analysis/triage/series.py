"""Consecutive runs grouped into droplets, and each run's droplet metrics.

Evaporation never increases a droplet's volume, so the question at every run
boundary is whether the volume went *up* by more than the droplet fit can scatter.
Each run supplies its own noise: the scatter of differences between adjacent
20-train medians of log10(volume) inside it, where continuity is certain. An
increase significant against that noise is a new droplet when it is large (above
the empty band of all significant increases) and a flagged glitch when it is small.
A pause between runs is not a break, and nor is a run without a fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analysis.triage.samples import CHECKED_LABEL_CHANGES, DROPLET_SAMPLES
from analysis.triage.stats import (
    empty_band,
    mad_dof,
    robust_sd,
    welch_dof,
    z_for_tests,
)

__all__ = [
    "DROPLET_CLASSES",
    "DROPLET_PX_UM",
    "WINDOW",
    "Boundary",
    "Droplet",
    "SeriesResult",
    "in_run_jump",
    "load_droplet",
    "run_noise",
    "segment",
    "window_medians",
]

#: Trains per median window, at run edges and inside runs alike.
WINDOW = 20
#: Droplet camera pixel size, the ``px`` of DAMNIT's droplet_fit. Same value as
#: analysis.utils.DROPLET_PX_MM, not imported because that module pulls in extra_data.
DROPLET_PX_UM = 13.9
#: Classes whose runs can hold a levitated droplet.
DROPLET_CLASSES = frozenset({"static_droplet", "mesh_sshex", "reference"})


@dataclass(frozen=True, slots=True)
class Droplet:
    """One run's droplet fit: finite, positive volumes only, in trainId order."""

    run: int
    train_ids: np.ndarray
    volume_mm3: np.ndarray
    center_px: np.ndarray  # (n, 2): x, y
    n_rows: int  # trains in the fit table, finite or not

    @property
    def log_volume(self) -> np.ndarray:
        return np.log10(self.volume_mm3)

    @property
    def coverage(self) -> float:
        return self.volume_mm3.size / self.n_rows if self.n_rows else 0.0


def load_droplet(path: Path, run: int) -> Droplet | None:
    """The ``droplet_fit`` group of one DAMNIT file; None when it is absent."""
    if not Path(path).exists():
        return None
    with h5py.File(path, "r") as f:
        if "droplet_fit/volume" not in f:
            return None
        g = f["droplet_fit"]
        tids = g["trainId"][:].astype(np.uint64)
        volume = g["volume"][:].astype(np.float64)
        dims = [d.decode() if isinstance(d, bytes) else str(d) for d in g["dim"][:]]
        center = g["center"][:].astype(np.float64)[
            :, [dims.index("x"), dims.index("y")]
        ]
    order = np.argsort(tids, kind="stable")
    tids, volume, center = tids[order], volume[order], center[order]
    ok = np.isfinite(volume) & (volume > 0)
    return Droplet(run, tids[ok], volume[ok], center[ok], int(tids.size))


def window_medians(values: np.ndarray, w: int = WINDOW) -> np.ndarray:
    """Medians of consecutive, non-overlapping windows of ``w``; drops a short tail."""
    n = values.size // w
    return np.median(values[: n * w].reshape(n, w), axis=1) if n else np.empty(0)


def run_noise(log_volume: np.ndarray, w: int = WINDOW) -> float:
    """Robust sd of the differences between adjacent window medians; NaN if < 3.

    This is the noise of a *difference* of two window medians. Taking it for both
    sides of a boundary overstates that boundary's noise by √2, deliberately. A
    boundary also carries a pause and any change of camera conditions, which no
    in-run null sees.
    """
    d = np.diff(window_medians(log_volume, w))
    return robust_sd(d - np.median(d)) if d.size >= 3 else float("nan")


def in_run_jump(
    log_volume: np.ndarray, sigma: float, w: int = WINDOW
) -> tuple[float, float, int]:
    """``(max |d − median d|, that divided by sigma, number of d)`` inside one run.

    ``d`` are the adjacent window-median differences; subtracting their median
    removes the steady evaporation, so a jump is measured against the run's trend.
    """
    d = np.diff(window_medians(log_volume, w))
    if d.size == 0:
        return float("nan"), float("nan"), 0
    jump = float(np.max(np.abs(d - np.median(d))))
    return jump, jump / sigma if sigma > 0 else float("nan"), int(d.size)


@dataclass(frozen=True, slots=True)
class Boundary:
    """The step between two consecutive fitted droplet runs."""

    before: int
    after: int
    d: float  # log10 volume step, start of `after` minus end of `before`
    sigma: float
    z: float
    z_cut: float  # Student t at the boundary's Welch–Satterthwaite dof
    kind: str  # continuing | increase | fresh | sample_change
    noise_fallback: bool
    #: the label changes but the volume does not jump: one droplet, two labels?
    label_suspect: bool = False
    #: a person's verdict on a suspect label change (samples.CHECKED_LABEL_CHANGES)
    label_checked: str | None = None
    #: a "continuing" verdict that even a fresh-sized step (d_fresh) could not have
    #: overturned against this noise, so "continuing" means "could not tell"
    underpowered: bool = False

    @property
    def breaks(self) -> bool:
        return self.kind in ("fresh", "sample_change")


@dataclass
class SeriesResult:
    runs: pd.DataFrame  # one row per droplet-class run
    series: pd.DataFrame  # one row per series
    boundaries: pd.DataFrame
    thresholds: dict


def _eligible(inventory: pd.DataFrame) -> pd.DataFrame:
    keep = inventory.cls.isin(DROPLET_CLASSES) & inventory["sample"].isin(
        DROPLET_SAMPLES
    )
    return inventory.loc[keep].sort_values("run")


def segment(
    inventory: pd.DataFrame,
    droplets: dict[int, Droplet],
    checked: dict[tuple[int, int], str] = CHECKED_LABEL_CHANGES,
) -> SeriesResult:
    """Group the droplet-class runs of `inventory` into series.

    ``droplets`` maps run -> `Droplet` for every run whose fit could be read. A run
    is *fitted* when it has at least two windows of finite volume. Unfitted runs
    become members of the series around them, or ``ambiguous`` when a series breaks
    across them. ``checked`` holds the verdicts on suspect label changes.
    """
    elig = _eligible(inventory)
    sample = dict(zip(elig.run, elig["sample"], strict=True))
    order = [int(r) for r in elig.run]
    fitted = [
        r for r in order if r in droplets and droplets[r].volume_mm3.size >= 2 * WINDOW
    ]

    sigma = {r: run_noise(droplets[r].log_volume) for r in fitted}
    n_diffs = {r: droplets[r].volume_mm3.size // WINDOW - 1 for r in fitted}
    finite = [s for s in sigma.values() if np.isfinite(s)]
    sigma_fallback = float(np.median(finite)) if finite else float("nan")
    fallback = {r: not np.isfinite(sigma[r]) for r in fitted}
    sigma = {r: sigma_fallback if fallback[r] else s for r, s in sigma.items()}
    # a fallback sigma is a median over many runs: taken as known (infinite dof)
    dof = {r: np.inf if fallback[r] else mad_dof(n_diffs[r]) for r in fitted}

    pairs = []
    for a, b in pairwise(fitted):
        between = order[order.index(a) : order.index(b) + 1]
        changed = any(sample[x] != sample[y] for x, y in pairwise(between))
        d = float(
            np.median(droplets[b].log_volume[:WINDOW])
            - np.median(droplets[a].log_volume[-WINDOW:])
        )
        s = float(np.hypot(sigma[a], sigma[b]))
        nu = welch_dof(sigma[a] ** 2, dof[a], sigma[b] ** 2, dof[b])
        pairs.append((a, b, d, s, d / s, nu, changed, fallback[a] or fallback[b]))
    cut = {(p[0], p[1]): z_for_tests(len(pairs), dof=p[5]) for p in pairs}
    significant = [p for p in pairs if p[4] > cut[p[0], p[1]] and not p[6]]
    band = empty_band(np.log10([p[2] for p in significant])) if significant else None
    d_fresh = float(10 ** np.mean([band.lower, band.upper])) if band else None

    boundaries = []
    for a, b, d, s, z, _, changed, fb in pairs:
        z_cut = cut[a, b]
        if changed:
            kind = "sample_change"
        elif z > z_cut and (d_fresh is None or d > d_fresh):
            kind = "fresh"
        elif z > z_cut:
            kind = "increase"
        else:
            kind = "continuing"
        boundaries.append(
            Boundary(
                a, b, d, s, z, z_cut, kind, fb,
                label_suspect=changed and z <= z_cut,
                label_checked=checked.get((a, b)),
                underpowered=kind == "continuing"
                and d_fresh is not None
                and d_fresh / s <= z_cut,
            )
        )  # fmt: skip
    after = {bd.after: bd for bd in boundaries}

    series_of: dict[int, str | None] = {}
    member: dict[int, str] = {}
    current: str | None = None
    pending: list[int] = []  # unfitted runs waiting for the next fitted run's verdict
    for r in order:
        if r not in sigma:
            pending.append(r)
            continue
        bd = after.get(r)
        starts = bd is None or bd.breaks
        if starts:
            for u in pending:  # unfitted runs before a fresh start cannot be placed
                series_of[u], member[u] = None, "ambiguous"
            current = f"s{r:03d}"
        else:
            for u in pending:
                series_of[u], member[u] = current, "no_fit"
        pending = []
        series_of[r], member[r] = current, "fitted"
    for u in pending:  # unfitted runs after the last fitted one
        same = current is not None and sample[u] == sample[int(current[1:])]
        series_of[u], member[u] = (current, "no_fit") if same else (None, "ambiguous")

    n_jump_tests = sum(max(n, 0) for n in n_diffs.values())
    rows = []
    v0 = {}
    for r in order:
        row = {"run": r, "series_id": series_of.get(r), "member": member.get(r)}
        bd = after.get(r)
        row["boundary_before"] = bd.kind if bd else None
        if r in sigma:
            dr = droplets[r]
            lv = dr.log_volume
            v_start, v_end = 10 ** np.median(lv[:WINDOW]), 10 ** np.median(lv[-WINDOW:])
            if row["series_id"] not in v0:
                v0[row["series_id"]] = v_start
            jump, jump_z, _ = in_run_jump(lv, sigma[r])
            jump_cut = z_for_tests(n_jump_tests, two_sided=True, dof=dof[r])
            significant_jump = bool(jump_z > jump_cut)
            # size decides what a significant step means: below d_fresh it is
            # glitch-sized, at or above it as large as an injection
            big = d_fresh is None or jump >= d_fresh
            row.update(
                v_start_mm3=float(v_start),
                v_end_mm3=float(v_end),
                v_over_v0_start=float(v_start / v0[row["series_id"]]),
                v_over_v0_end=float(v_end / v0[row["series_id"]]),
                fit_coverage=dr.coverage,
                center_jitter_x_um=robust_sd(dr.center_px[:, 0]) * DROPLET_PX_UM,
                center_jitter_y_um=robust_sd(dr.center_px[:, 1]) * DROPLET_PX_UM,
                noise_log10=sigma[r],
                noise_fallback=fallback[r],
                max_jump_dlog=jump,
                max_jump_z=jump_z,
                jump_z_cut=jump_cut,
                volume_glitch=significant_jump and not big,
                volume_jump=significant_jump and big,
            )
        rows.append(row)
    runs = pd.DataFrame(rows)

    series_rows = []
    for sid, grp in runs.dropna(subset=["series_id"]).groupby("series_id", sort=False):
        fit_rows = grp[grp.member == "fitted"]
        first = int(fit_rows.run.iloc[0])
        bd = after.get(first)
        series_rows.append(
            {
                "series_id": sid,
                "first_run": int(grp.run.min()),
                "last_run": int(grp.run.max()),
                "n_runs": len(grp),
                "runs": [int(r) for r in grp.run],
                "no_fit_runs": [int(r) for r in grp.run[grp.member == "no_fit"]],
                "sample": sample[first],
                "start_kind": bd.kind if bd else "first",
                "v0_mm3": float(v0[sid]),
                "v_min_mm3": float(fit_rows.v_end_mm3.min()),
                "n_increase": int((fit_rows.boundary_before == "increase").sum()),
                "n_underpowered": int(
                    sum(after[r].underpowered for r in fit_rows.run if r in after)
                ),
                "n_volume_glitch": int(fit_rows.volume_glitch.sum()),
                "n_volume_jump": int(fit_rows.volume_jump.sum()),
            }
        )
    thresholds = {
        "window_trains": WINDOW,
        "n_boundaries": len(pairs),
        "z_cut_rule": "Student t, alpha = 0.1/n_boundaries one-sided, Welch dof",
        "z_cut_normal": z_for_tests(len(pairs)) if pairs else float("nan"),
        "d_fresh": d_fresh,
        "d_fresh_band_log10": None if band is None else [band.lower, band.upper],
        "n_significant_increases": len(significant),
        "sigma_fallback": sigma_fallback,
        "n_jump_tests": n_jump_tests,
        "jump_cut_rule": "Student t, alpha = 0.1/n_jump_tests two-sided, MAD dof",
    }
    return SeriesResult(
        runs=runs,
        series=pd.DataFrame(series_rows),
        boundaries=pd.DataFrame([asdict(b) for b in boundaries]),
        thresholds=thresholds,
    )
