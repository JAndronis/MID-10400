"""Which trains hit the droplet (run-triage §4.4, xpcs-quality §5.1).

Every diagnostic of a run is computed over its droplet hits only (user, 2026-10-13).
The per-train signal is mid-q SAXS per unit flux,

    s_t = mean over the frames present of DAMNIT's agipd_saxs, 0.3–0.5 nm⁻¹ (D5)
    F_t = xgm_intensity × total_transmission           (per train, never per pulse)

which separates droplet from beam-only trains by 14–28× (§2.1). The JUNGFRAU ring
(jf1, 17–23 nm⁻¹) is read beside it as a cross-check (~2.5×).

This module reads DAMNIT's per-run file only, and caches each run's per-train signals
under scratch/run_triage (§6), keyed on the file's size and mtime.

Mesh runs (x = log10 s_t/F_t), measured 2026-10-14 over all 255 mesh runs:

    per run:    Otsu's two-class split of x (Otsu 1979). The off level m and its width
                σ (1.4826·MAD) come from the low class. The median of the run is no
                use: 29 runs hit the droplet on ≥ 40 % of their trains.
    per block:  runs of one attenuator and sample. rel = x − m of each run, pooled.
                The hit cut t_on is the centre of the widest empty band of the pooled
                rel between the off band and the hits' mode (§4.4's empty band, on the
                relative axis, where the off peaks of different runs coincide).
    classes:    off ⇔ rel ≤ 5σ;  on ⇔ rel ≥ t_on;  edge otherwise (partial overlaps)
    one class:  a run with no second class of its own (high − off < 2·5σ) is placed
                against its nearest two-class neighbour's off level: at the hit level
                every train is a hit (r309–311: the beam stayed on the droplet), in
                the off band every train is off, anywhere else the run is ambiguous
                (r140, 0.6 dex below its neighbours' off level).
    ambiguous:  such a run, or a block whose hits' mode lies less than one off band
                above the off band (on_mode < 2·5σ; silica r68–69 at 1.07 band
                widths, against 4.9 for the weakest real block, r307). No hits are
                reported for either.
    Off levels drift within a block (r308 → r335: −5.45 → −4.7) and differ between
    its sub-series (r290–306 sit 0.7 dex below r240–288), so no block-wide level is
    used, only each run's own or its neighbour's.

Pooling raw x per block, as §4.4 first said, fails: off levels differ between runs
by up to 0.9 dex within a block, and the r240–335 block mixes two samples.

Static runs: every train is a hit except a droplet out of the beam, a train whose
x falls more than 5 robust σ below a running 75th percentile over 301 trains (30 s),
which an absence of up to 22 s cannot drag down (`static_hits`).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from analysis.triage.inventory import DAMNIT_ROOT, damnit_file

__all__ = [
    "CACHE_DIR",
    "MID_BAND_NM",
    "OFF_SIGMA",
    "WAXS_RING_NM",
    "block_cut",
    "classify_block",
    "hits",
    "otsu_split",
    "run_blocks",
    "static_hits",
    "train_signals",
    "waxs_check",
]

CACHE_DIR = Path("/gpfs/exfel/exp/MID/202601/p010400/scratch/run_triage")
#: The classification band (§4.4, D5) and the WAXS cross-check ring.
MID_BAND_NM = (0.3, 0.5)
WAXS_RING_NM = (17.0, 23.0)
#: Trains per read of the pulse-resolved grid: bounds memory, matches its chunking.
TRAIN_CHUNK = 106
#: Bump when what a cached signal file holds changes.
SIGNALS_VERSION = 1
#: Gaussian-tail cut in robust σ (§4.3, 0.002 false outliers per 3000 trains): the off
#: band of a run, a run's off level against its block's, a static train's drop.
OFF_SIGMA = 5.0
#: Running-median window of §4.3, in trains (kept for reference; see `static_hits`).
RUNNING_TRAINS = 51
#: Static runs: the reference is this quantile over this many trains, the keep rule's
#: 30 s slice (user, 2026-10-01). An absence of up to 225 trains (22 s) is found.
STATIC_WINDOW, STATIC_QUANTILE = 301, 0.75
#: Hits need a mode beyond this many off-band widths above the off level.
SEPARATION_BANDS = 2.0


def _band_mean(path: Path, group: str, band) -> tuple | None:
    """Per train: mean over present frames of the mean over ``band``'s q bins."""
    with h5py.File(path, "r") as f:
        if group not in f or "intensity" not in f[group]:
            return None
        g = f[group]
        q = g["q"][:]
        cols = np.flatnonzero((q >= band[0]) & (q < band[1]))
        if cols.size == 0:
            raise ValueError(f"{path}:{group} has no q bin in {band}")
        lo, hi = int(cols[0]), int(cols[-1]) + 1
        if not np.array_equal(cols, np.arange(lo, hi)):
            raise ValueError(f"{path}:{group}: q is not sorted")
        tids = g["trainId"][:].astype(np.uint64)
        values = np.full(tids.size, np.nan)
        present = np.zeros(tids.size, dtype=np.int32)
        for s in range(0, tids.size, TRAIN_CHUNK):
            frames = g["n_frames"][s : s + TRAIN_CHUNK] > 0  # (n, P)
            band_mean = g["intensity"][s : s + TRAIN_CHUNK, :, lo:hi].mean(axis=-1)
            n = frames.sum(axis=1)
            total = np.where(frames, band_mean, 0.0).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                values[s : s + n.size] = np.where(n > 0, total / n, np.nan)
            present[s : s + n.size] = n
    return tids, values, present


def train_signals(
    run: int,
    *,
    damnit_root: str | Path = DAMNIT_ROOT,
    proposal: int = 10400,
    cache_dir: str | Path | None = CACHE_DIR,
    refresh: bool = False,
):
    """Per train of one run: ``saxs_mid``, ``n_frames``, ``xgm_uJ``,
    ``total_transmission``, ``flux`` and ``waxs_ring`` (NaN without jf1).

    Unscaled (memory, cache-holds-raw-data-only): the flux sits beside the signals.
    The trains are those of DAMNIT's agipd_saxs grid; a train without a frame, an
    XGM or a transmission carries NaN where the value is missing.

    :param cache_dir: where to cache, or None to read DAMNIT every time.
    :raises LookupError: the run has no agipd_saxs in DAMNIT.
    """
    import xarray as xr

    from analysis.cache import _attrs, _cache_path, _load, _source, _write
    from analysis.diagnostics.curves import per_train_flux

    path = damnit_file(int(run), Path(damnit_root), proposal)
    inputs = {
        "kind": "train_signals",
        "version": SIGNALS_VERSION,
        "run": int(run),
        "mid_band_nm": list(MID_BAND_NM),
        "waxs_ring_nm": list(WAXS_RING_NM),
    }
    sources = [_source(path)]
    cached = None
    if cache_dir is not None:
        cached = _cache_path(cache_dir, f"signals_r{int(run):04d}", inputs)
        if not refresh and (ds := _load(cached, inputs, sources)) is not None:
            return ds

    saxs = _band_mean(path, "agipd_saxs", MID_BAND_NM)
    if saxs is None:
        raise LookupError(f"r{run}: no agipd_saxs in {path}")
    tids, mid, present = saxs
    ds = xr.Dataset(
        {
            "saxs_mid": ("trainId", mid),
            "n_frames": ("trainId", present),
        },
        coords={"trainId": tids},
    )
    flux = per_train_flux(path).reindex(trainId=tids)
    for name in ("xgm_uJ", "total_transmission", "flux"):
        ds[name] = flux[name]
    waxs = _band_mean(path, "jungfrau_waxs_jf1", WAXS_RING_NM)
    if waxs is None:
        ds["waxs_ring"] = ("trainId", np.full(tids.size, np.nan))
    else:
        ds["waxs_ring"] = xr.DataArray(
            waxs[1], dims="trainId", coords={"trainId": waxs[0]}
        ).reindex(trainId=tids)
    ds.saxs_mid.attrs["description"] = (
        f"DAMNIT agipd_saxs, mean over {MID_BAND_NM} nm^-1 and the frames present"
    )
    ds.waxs_ring.attrs["description"] = (
        f"DAMNIT jungfrau_waxs_jf1, mean over {WAXS_RING_NM} nm^-1 and the cells kept"
    )
    ds.attrs.update(
        _attrs(inputs, sources, "per-train signals for droplet hits; unscaled"),
        run=int(run),
    )
    if cached is not None:
        _write(ds, cached)
    return ds


# ── classification ────────────────────────────────────────────────────────────


def otsu_split(x: np.ndarray) -> float:
    """The cut between two classes of ``x`` that maximises their between-class variance.

    Otsu 1979 on the sorted values themselves, no histogram: the cut is the midpoint
    of the two neighbouring values it falls between.
    """
    x = np.sort(np.asarray(x, dtype=np.float64)[np.isfinite(x)])
    if x.size < 2:
        raise ValueError(f"{x.size} finite values: nothing to split")
    n = x.size
    k = np.arange(1, n)
    c = np.cumsum(x)
    m0, m1 = c[:-1] / k, (c[-1] - c[:-1]) / (n - k)
    between = (k / n) * (1 - k / n) * (m0 - m1) ** 2
    i = int(np.argmax(between))
    return float(0.5 * (x[i] + x[i + 1]))


def _shorth_midpoint(x: np.ndarray) -> float:
    """Midpoint of the shortest interval holding half of ``x``: a robust mode."""
    x = np.sort(x)
    h = max(x.size // 2, 1)
    if x.size <= h:
        return float(np.median(x))
    i = int(np.argmin(x[h:] - x[: x.size - h]))
    return float(0.5 * (x[i] + x[i + h]))


def run_levels(x: np.ndarray) -> dict:
    """A run's off level ``m``, its width ``sigma`` and the high class's median."""
    from analysis.triage.stats import robust_sd

    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    cut = otsu_split(x)
    low, high = x[x <= cut], x[x > cut]
    return {
        "off": float(np.median(low)),
        "sigma": float(robust_sd(low)) if low.size >= 3 else float("nan"),
        "high": float(np.median(high)) if high.size else float("nan"),
        "n": int(x.size),
        "n_high": int(high.size),
    }


def block_cut(rel: np.ndarray, off_cut: float) -> dict:
    """The hit cut of one block from its pooled ``rel`` values.

    ``on_mode`` is the shorth midpoint of the values above the off band; ``t_on`` the
    centre of the widest empty band between ``off_cut`` and ``on_mode``. A block
    whose values above the off band do not form a mode above it is ambiguous.
    """
    from analysis.triage.stats import empty_band

    rel = np.asarray(rel, dtype=np.float64)
    rel = rel[np.isfinite(rel)]
    above = rel[rel > off_cut]
    out = {"off_cut": float(off_cut), "n_above": int(above.size)}
    if above.size < 2:
        return out | {"on_mode": float("nan"), "t_on": float("nan"), "ambiguous": True}
    on_mode = _shorth_midpoint(above)
    band = empty_band(rel, lo=off_cut, hi=on_mode)
    if band is None or not on_mode >= SEPARATION_BANDS * off_cut:
        return out | {"on_mode": on_mode, "t_on": float("nan"), "ambiguous": True}
    return out | {
        "on_mode": on_mode,
        "band": [band.lower, band.upper],
        "t_on": band.centre,
        "ambiguous": False,
    }


def classify_block(xs: dict[int, np.ndarray]) -> dict:
    """Classes for every train of one block's runs.

    A run with two classes of its own (high − off ≥ `SEPARATION_BANDS` off bands) is
    classified on its own off level. A run without (the beam stayed on the droplet,
    or never met it) is placed against the off level of its nearest two-class
    neighbour in run number: at their hit level all its trains are hits, inside
    their off band all are off, anywhere else it is ambiguous.

    :param xs: run -> per-train log10(s_t/F_t), NaN where a train has no value.
    :returns: ``classes`` (run -> array of "on"/"edge"/"off"/"none"), ``runs`` (run
        -> its levels, reference, counts and flags) and ``block`` (the cut).
    """
    nan = float("nan")
    levels = {}
    for r, x in xs.items():
        try:
            levels[r] = run_levels(x)
        except ValueError:  # fewer than two trains with a value: nothing to split
            levels[r] = {"off": nan, "sigma": nan, "high": nan, "n": 0, "n_high": 0}
    sigmas = [v["sigma"] for v in levels.values() if np.isfinite(v["sigma"])]
    sigma = float(np.median(sigmas)) if sigmas else nan
    off_cut = OFF_SIGMA * sigma
    two = sorted(
        r for r, v in levels.items()
        if v["high"] - v["off"] >= SEPARATION_BANDS * off_cut
    )  # fmt: skip
    for r, v in levels.items():
        v["two_classes"] = r in two
        if v["two_classes"]:
            v["reference"], v["off_ref"] = int(r), v["off"]
        elif two and np.isfinite(v["off"]):
            nearest = min(two, key=lambda t: (abs(t - r), t))
            v["reference"], v["off_ref"] = int(nearest), levels[nearest]["off"]
        else:
            v["reference"], v["off_ref"] = None, nan
    rel = np.concatenate(
        [np.asarray(xs[r], dtype=np.float64) - levels[r]["off"] for r in two]
        or [np.zeros(0)]
    )
    block = block_cut(rel, off_cut) | {
        "sigma": sigma,
        "runs": sorted(int(r) for r in xs),
        "two_class_runs": len(two),
    }
    classes, runs = {}, {}
    for r, x in xs.items():
        x = np.asarray(x, dtype=np.float64)
        v = levels[r]
        level = v["off"] - v["off_ref"]  # 0 for a two-class run
        ambiguous = (
            block["ambiguous"]
            or not np.isfinite(level)
            or (
                not v["two_classes"]
                and not (abs(level) <= off_cut or level >= block["t_on"])
            )
        )
        cls = np.full(x.size, "none", dtype="<U4")
        finite = np.isfinite(x)
        if not ambiguous:
            d = x - v["off_ref"]
            cls[finite & (d <= off_cut)] = "off"
            cls[finite & (d >= block["t_on"])] = "on"
            cls[finite & (d > off_cut) & (d < block["t_on"])] = "edge"
        classes[r] = cls
        runs[r] = v | {
            "ambiguous": bool(ambiguous),
            **{f"n_{k}": int((cls == k).sum()) for k in ("on", "edge", "off", "none")},
        }
    return {"classes": classes, "runs": runs, "block": block}


def static_hits(
    x: np.ndarray,
    window: int = STATIC_WINDOW,
    quantile: float = STATIC_QUANTILE,
    max_passes: int = 5,
) -> np.ndarray:
    """A static run's hits: finite trains not out of the beam.

    ``x`` is log10(s_t/F_t) in train order. The reference is a running `quantile`
    over `window` trains, so an absence of up to (1 − quantile)·window trains
    cannot pull it down. A 51-train running median (§4.3) misses any absence longer
    than 25 trains, and r0426 has two of 21 and 25. A train is out of the beam when
    its residual, centred on the median residual, lies below `OFF_SIGMA` robust σ.
    The reference is recomputed without the trains already flagged, until nothing
    new is flagged.

    Small dips (0.1–0.2 dex, the droplet moving slightly in the beam) stay below
    that cut at this window and remain hits. The 51-train median flagged them as well
    (r0436: 68 against 4 trains, r0406: 54 against 6; 2026-10-14).
    """
    import pandas as pd

    from analysis.triage.stats import robust_sd

    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.zeros(x.size, dtype=bool)
    if finite.sum() < 3:
        return out
    for _ in range(max_passes):
        kept = pd.Series(np.where(finite & ~out, x, np.nan))
        rolling = kept.rolling(window, center=True, min_periods=max(window // 4, 1))
        base = rolling.quantile(quantile).interpolate(limit_direction="both")
        resid = x - base.to_numpy()
        clean = finite & ~out
        resid = resid - np.median(resid[clean])
        new = finite & ~out & (resid < -OFF_SIGMA * robust_sd(resid[clean]))
        if not new.any():
            break
        out |= new
    return finite & ~out


def waxs_check(waxs: np.ndarray, classes: np.ndarray) -> dict:
    """How far the JUNGFRAU ring of a run's hits sits above its off trains.

    ``waxs`` is log10(ring/F) per train. The shift is the median over the hits minus
    the median over the off trains, in dex; ``z`` divides it by the standard error
    of a median, 1.2533 · σ_off / √n_hits, σ_off the off trains' robust σ.
    Measured 2026-10-14: hits sit 0.33–0.35 dex (×2.1–2.25) above off from
    attenuator 0.012 up; at 0.001 and 0.0047 the ring is noise-dominated and
    confirms nothing (r307: 0.05 dex, z 1.5). A reported number, never a cut.
    """
    from analysis.triage.stats import robust_sd

    w = np.asarray(waxs, dtype=np.float64)
    on, off = (
        w[(classes == "on") & np.isfinite(w)],
        w[(classes == "off") & np.isfinite(w)],
    )
    if on.size == 0 or off.size < 3:
        return {"waxs_shift_dex": float("nan"), "waxs_z": float("nan")}
    shift = float(np.median(on) - np.median(off))
    se = 1.2533 * robust_sd(off) / np.sqrt(on.size)
    return {
        "waxs_shift_dex": shift,
        "waxs_z": float(shift / se) if se > 0 else float("nan"),
    }


def run_blocks(runs_table):
    """Class and mesh block of every run of DAMNIT's run table (`read_run_table`).

    A mesh block is one attenuator (2 significant digits of the run-mean
    ``total_transmission``) and one effective sample (`samples.effective_sample`).
    """
    import pandas as pd

    from analysis.triage.inventory import classify
    from analysis.triage.samples import effective_sample

    rows = []
    for rec in runs_table.itertuples(index=False):
        run = int(rec.run)
        sample = effective_sample(run, rec.sample).name
        cls = classify(run, rec.run_type, rec.scan_type, sample)
        att = rec.total_transmission
        block = (
            f"{float(att):.2g}|{sample}"
            if cls == "mesh_sshex" and pd.notna(att)
            else None
        )
        rows.append({"run": run, "cls": cls, "sample": sample, "block": block})
    return pd.DataFrame(rows).set_index("run")


def _log_ratio(ds, name: str) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10((ds[name] / ds.flux).values)


def hits(
    run: int,
    *,
    damnit_root: str | Path = DAMNIT_ROOT,
    proposal: int = 10400,
    cache_dir: str | Path | None = CACHE_DIR,
    runs_table=None,
) -> dict:
    """The trains of one run that hit the droplet, and why.

    Mesh runs are classified with every run of their block (`classify_block`); any
    other run by `static_hits`. Per-train signals are cached, so a block costs a
    few seconds once its runs have been read.

    :param runs_table: DAMNIT's run table (`inventory.read_run_table`); read from
        ``damnit_root`` when not given.
    :returns: ``train_ids`` (the hits), ``classes`` (a DataArray over every train:
        "on"/"edge"/"off"/"none" for mesh runs, "on"/"off"/"none" for the others),
        ``kind`` ("mesh" or "static"), ``block`` and ``summary`` (counts, flags, the
        block's cut, the WAXS check).
    """
    import xarray as xr

    from analysis.triage.inventory import read_run_table

    if runs_table is None:
        runs_table = read_run_table(Path(damnit_root) / "runs.sqlite")
    table = run_blocks(runs_table)
    if run not in table.index:
        raise LookupError(f"r{run} is not in DAMNIT's run table")
    row = table.loc[run]
    signals = train_signals(run, damnit_root=damnit_root, proposal=proposal,
                            cache_dir=cache_dir)  # fmt: skip
    tids = signals.trainId.values
    block = row.block if isinstance(row.block, str) else None
    if block is not None:
        members = table.index[table.block == block]
        xs = {
            int(r): _log_ratio(
                train_signals(
                    int(r),
                    damnit_root=damnit_root,
                    proposal=proposal,
                    cache_dir=cache_dir,
                ),
                "saxs_mid",
            )  # fmt: skip
            for r in members
        }
        out = classify_block(xs)
        cls = out["classes"][run]
        summary = {k: v for k, v in out["runs"][run].items()}
        summary["block_cut"] = {
            k: v for k, v in out["block"].items() if k not in ("runs",)
        }
        kind = "mesh"
    else:
        x = _log_ratio(signals, "saxs_mid")
        on = static_hits(x)
        cls = np.where(on, "on", np.where(np.isfinite(x), "off", "none")).astype("<U4")
        summary = {f"n_{k}": int((cls == k).sum()) for k in ("on", "off", "none")}
        summary["ambiguous"] = False
        kind = "static"
    summary |= waxs_check(_log_ratio(signals, "waxs_ring"), cls)
    summary["n"] = int(tids.size)
    classes = xr.DataArray(cls, dims="trainId", coords={"trainId": tids}, name="class")
    return {
        "run": int(run),
        "kind": kind,
        "block": block,
        "train_ids": tids[cls == "on"],
        "classes": classes,
        "summary": summary,
    }
