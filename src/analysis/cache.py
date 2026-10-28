"""On-disk cache of per-train SAXS inputs, so notebooks re-run without re-reading runs.

Only what the sources hold is stored, reduced by averaging or summing over frames and
nothing else. The pulse-averaged I(q) is DAMNIT's ``agipd.saxs`` grid, neither scaled
nor background subtracted; the droplet scale factors, the droplet fit and the D^2-law
fit sit beside it as variables and attributes and are never applied; the 2D detector
sums are photon counts per pixel with no mask beyond the frame's own; the XPCS g2 and
two-time maps are the hit-train results file's own pooled estimates, with every hit
train listed beside them. Subtraction, normalisation and every other operation
belong in the notebook or script that loads the cache.

A cache file is rebuilt when its inputs, the DAMNIT files it was read from or
`CACHE_VERSION` change, and on ``refresh=True``. Files are written under a temporary
name and renamed into place, so a reader never sees a partial file.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import uuid
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from analysis import utils
from analysis.damnit_names import AGIPD_SAXS, stored_in_file

__all__ = [
    "CACHE_VERSION",
    "DEFAULT_CACHE_DIR",
    "detector_sums",
    "droplet_timeline",
    "file_source",
    "pool_g2",
    "saxs_run",
    "stale",
    "write_dataset",
    "xpcs_run",
]

DEFAULT_CACHE_DIR = Path("/gpfs/exfel/exp/MID/202601/p010400/scratch/saxs_cache")
PROC_ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400/proc")
#: Bump when what a cache file holds changes; older files are then rebuilt.
CACHE_VERSION = 1
#: Trains read per chunk when averaging the pulse-resolved grid (bounds memory).
TRAIN_CHUNK = 200
#: Trains per worker job when summing detector frames (~0.7 GB at 350 frames/train).
SUM_CHUNK = 5
#: AGIPD module geometry: modules per detector and pixels per module.
N_MODULES, MODULE_SHAPE = 16, (512, 128)


def stale(attrs, **expected) -> list[str]:
    """The keys of `expected` whose value differs from the JSON stored under the
    same key in `attrs`. A key missing from `attrs` is stale.

    Pass a file's attributes while the file is open: an h5py attribute manager kept
    past its ``with`` block returns the default, so every key would read as stale.
    """
    return [
        key
        for key, value in expected.items()
        if json.loads(attrs.get(key, "null"))
        != json.loads(json.dumps(value, sort_keys=True))
    ]


def file_source(path) -> dict:
    """What identifies a source file's version: its path, size and mtime."""
    stat = os.stat(path)
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _cache_path(cache_dir, stem: str, inputs: dict) -> Path:
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return Path(cache_dir) / f"{stem}_{digest[:12]}.nc"


def _load(path: Path, inputs: dict, sources: list) -> xr.Dataset | None:
    """The cached dataset, or None if absent or built from other inputs or sources."""
    if not path.exists():
        return None
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        # read every attribute inside the block; after it they read as absent
        current = int(ds.attrs.get("cache_version", -1)) == CACHE_VERSION and not (
            stale(ds.attrs, inputs=inputs, sources=sources)
        )
        return ds.load() if current else None


def write_dataset(ds: xr.Dataset, path: Path) -> None:
    """Write `ds` under a temporary name and rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        ds.to_netcdf(tmp, engine="h5netcdf")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _attrs(inputs: dict, sources: list, description: str) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "inputs": json.dumps(inputs, sort_keys=True),
        "sources": json.dumps(sources, sort_keys=True),
        "description": description,
    }


def _train_times(raw) -> xr.DataArray:
    """Train timestamps (UTC, tz-naive) labelled by trainId, never by position."""
    ts = raw.train_timestamps(labelled=True)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return xr.DataArray(
        ts.to_numpy(dtype="datetime64[ns]"),
        dims="trainId",
        coords={"trainId": ts.index.to_numpy(dtype=np.uint64)},
    )


def _pulse_mean(grid: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Mean of I(q) over the frames a train has, and how many it has.

    DAMNIT stores a missing frame (n_frames == 0) as I = 0, so it is excluded rather
    than averaged in; a train with no frame at all is NaN.
    """
    means, counts = [], []
    for start in range(0, grid.sizes["trainId"], TRAIN_CHUNK):
        part = grid.isel(trainId=slice(start, start + TRAIN_CHUNK)).load()
        has_frame = part.n_frames > 0
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # all-NaN: a train w/o frames
            means.append(part.intensity.where(has_frame).mean("pulseId"))
        counts.append(has_frame.sum("pulseId"))
    return xr.concat(means, "trainId"), xr.concat(counts, "trainId")


def _composition(
    ferritin_mg_per_ml_initial: float,
    peg_percent_wv_initial: float,
    nacl_mM_initial: float,
    n_fe: float,
) -> dict:
    """The `droplet_scaling` composition arguments, one canonical type each."""
    return {
        "ferritin_mg_per_ml_initial": float(ferritin_mg_per_ml_initial),
        "peg_percent_wv_initial": float(peg_percent_wv_initial),
        "nacl_mM_initial": float(nacl_mM_initial),
        "n_fe": float(n_fe),
    }


def _add_droplet(
    ds: xr.Dataset,
    tids,
    run_nr: int,
    db,
    raw,
    proposal: int,
    n_pulses: int,
    v0_run: int,
    composition: dict,
) -> dict:
    """Add ``time``, the `droplet_scaling` fields and ``droplet_*`` to `ds`.

    Everything is reindexed onto `tids` by label. `raw` is opened when None. Returns
    the attributes that describe them: ``v0_mm3`` and ``droplet_pixel_size_um``.
    """
    if raw is None:
        raw = utils.ex.open_run(proposal, run_nr, data="raw")
    scaling = utils.droplet_scaling(
        run_nr,
        db,
        proposal=proposal,
        n_pulses=n_pulses,
        v0_run=v0_run,
        raw=raw,
        **composition,
    )
    fit = xr.open_dataset(db[run_nr].file, group="droplet_fit")
    ds["time"] = _train_times(raw).reindex(trainId=tids)
    for name in scaling.data_vars:
        ds[name] = scaling[name].reindex(trainId=tids)
    ds["scale"].attrs["description"] = "1/(flux*path_mm); stored, not applied"
    for name in ("volume", "radius", "center", "phi"):
        if name in fit:
            var = fit[name].reindex(trainId=tids)
            ds[f"droplet_{name}"] = (
                var.rename({"dim": "axis"}) if "dim" in var.dims else var
            )
    return {
        "v0_mm3": scaling.attrs["v0_mm3"],
        "droplet_pixel_size_um": fit.attrs.get(
            "pixel_size_um", utils.DROPLET_PX_MM * 1e3
        ),
    }


def saxs_run(
    run_nr: int,
    db,
    *,
    v0_run: int | None = None,
    n_pulses: int = 155,
    ferritin_mg_per_ml_initial: float = 50.0,
    peg_percent_wv_initial: float = 5.0,
    nacl_mM_initial: float = 150.0,
    n_fe: float = 1800.0,
    proposal: int = 10400,
    raw=None,
    cache_dir=DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> xr.Dataset:
    """Per-train SAXS data of one run with everything needed to scale it, cached.

    Returns a Dataset over (trainId, q), one entry per train of DAMNIT's
    ``agipd_saxs`` grid:

    ``intensity``
        I(q) averaged over the first `n_pulses` pulseIds of each train, counting only
        frames that exist. Not scaled, not background subtracted.
    ``n_pulses_with_frames``
        How many frames went into that average.
    ``time``
        Train timestamp, UTC.
    the `analysis.utils.droplet_scaling` fields
        xgm_uJ, total_transmission, flux, volume_mm3, chord_mm, v_over_v0, mu_per_mm,
        droplet_transmission, path_mm, phi_ferritin and scale, NaN for a train the
        scaling had no input for. ``intensity * scale`` is I per flux*path; it is
        stored, never applied.
    ``droplet_*``
        The droplet fit as DAMNIT stores it, in camera pixels.

    The keyword arguments are those of `droplet_scaling`; they are part of the cache
    key, so the same run with another V0 or composition is a separate file.
    """
    # one canonical type per input, so 0 and 0.0 name the same cache file
    run_nr, n_pulses, proposal = int(run_nr), int(n_pulses), int(proposal)
    v0_run = run_nr if v0_run is None else int(v0_run)
    composition = _composition(
        ferritin_mg_per_ml_initial, peg_percent_wv_initial, nacl_mM_initial, n_fe
    )
    inputs = {
        "kind": "saxs_run",
        "proposal": proposal,
        "run": run_nr,
        "v0_run": v0_run,
        "n_pulses": n_pulses,
        **composition,
    }
    sources = [file_source(db[r].file) for r in sorted({run_nr, v0_run})]
    path = _cache_path(cache_dir, f"saxs_r{run_nr:04d}", inputs)
    if not refresh and (cached := _load(path, inputs, sources)) is not None:
        return cached

    grid = xr.open_dataset(
        db[run_nr].file, group=stored_in_file(db[run_nr].file, AGIPD_SAXS)
    )
    pulses = np.sort(grid.pulseId.values)
    if pulses.size < n_pulses:
        raise ValueError(
            f"r{run_nr} has {pulses.size} pulses per train, "
            f"fewer than n_pulses={n_pulses}"
        )
    intensity, n_with_frames = _pulse_mean(grid.sel(pulseId=pulses[:n_pulses]))
    tids = intensity.trainId

    ds = xr.Dataset(
        {
            "intensity": intensity.assign_attrs(
                description=(
                    f"I(q) from DAMNIT agipd_saxs, averaged over the frames present "
                    f"among the first {n_pulses} pulseIds; not scaled, not background "
                    "subtracted"
                )
            ),
            "n_pulses_with_frames": n_with_frames.astype(np.int16),
        }
    )
    droplet = _add_droplet(
        ds, tids, run_nr, db, raw, proposal, n_pulses, v0_run, composition
    )
    ds.attrs.update(
        _attrs(inputs, sources, "per-train SAXS inputs; see analysis.cache.saxs_run"),
        run=run_nr,
        v0_run=v0_run,
        **droplet,
    )
    write_dataset(ds, path)
    return ds


XPCS_AVERAGINGS = ("pooled", "es")


def _train_terms(hits, sums) -> dict[str, np.ndarray]:
    """The pooled sums' per-row lag terms placed on the hit trains, NaN where absent.

    ``sums.rows`` are positions in ``hits.train_ids``, so the terms follow the train,
    not the row order of the sums.
    """
    shape = (hits.train_ids.size, *sums.on.shape[1:])
    out = {
        name: np.full(shape, np.nan)
        for name in (
            "train_g2_num",
            "train_g2_den",
            "train_baseline_num",
            "train_baseline_den",
        )
    }
    out["train_g2_num"][sums.rows] = sums.on
    out["train_g2_den"][sums.rows] = sums.den_on
    base = sums.rows[sums.has_off]
    out["train_baseline_num"][base] = sums.off[sums.has_off]
    out["train_baseline_den"][base] = sums.den_off[sums.has_off]
    return out


def pool_g2(
    ds: xr.Dataset,
    trains=None,
    *,
    binned: bool = True,
    skip_last: int | None = None,
    n_blocks: int | None = None,
) -> xr.Dataset:
    """Pooled g2 of a selection of hit trains from an `xpcs_run` dataset.

    Gives `XpcsHits.g2(trains=...)` exactly for the same trains (same function,
    `analysis.xpcs.estimators.pooled_g2`): the lag terms of the usable selected trains
    are summed and divided, with the jackknife over ``n_blocks`` contiguous blocks
    of them in trainId order. ``trains=None`` takes every usable hit train, which is
    the dataset's own ``g2``, ``sigma_g2``, ``g2_paired``, ``baseline``, ``c2``,
    ``sigma``. Pool a selection (by time, droplet volume, ``kbar``, ...) with this;
    do not average ``train_g2_num / train_g2_den`` over trains, which is the mean of
    noisy per-train ratios, not the ratio of sums. The two-time maps of a subset come
    from ``XpcsHits.ttcf(trains=...)``.

    :param trains: trainIds, by label; unusable ones are skipped as the reader does.
    :param skip_last: lags dropped from the end; ``None`` for the reader's default.
    :param n_blocks: jackknife blocks; ``None`` for the reader's default.
    :raises ValueError: `ds` has no per-train terms (an ``averaging="es"`` cache, or
        an older one without them), or a trainId is not a hit train.
    """
    from analysis.common.status import FrameStatus
    from analysis.xpcs.estimators import (
        N_BLOCKS,
        SKIP_LAST_LAGS,
        lag_frames,
        pooled_g2,
    )

    skip_last = SKIP_LAST_LAGS if skip_last is None else skip_last
    n_blocks = N_BLOCKS if n_blocks is None else n_blocks
    if "train_g2_num" not in ds:
        raise ValueError(
            "no per-train lag terms in this dataset: they exist only for "
            "xpcs_run(averaging='pooled')"
        )
    wanted = ds["status"].values == int(FrameStatus.OK)
    if trains is not None:
        trains = np.asarray(trains, dtype=np.uint64)
        unknown = trains[~np.isin(trains, ds.trainId.values)]
        if unknown.size:
            raise ValueError(
                f"{unknown.size} train(s) are not in the hit set "
                f"(first: {unknown[:5].tolist()})"
            )
        wanted &= np.isin(ds.trainId.values, trains)
    rows = np.flatnonzero(wanted)
    terms = [
        ds[name].isel(trainId=rows).transpose("trainId", "roi_q", "lag_frame").values
        for name in (
            "train_g2_num",
            "train_g2_den",
            "train_baseline_num",
            "train_baseline_den",
        )
    ]
    has_off = np.isfinite(terms[2]).all(axis=(1, 2))
    on, den_on, off, den_off = (
        terms[0],
        terms[1],
        np.nan_to_num(terms[2]),
        np.nan_to_num(terms[3]),
    )
    return pooled_g2(
        on,
        den_on,
        off,
        den_off,
        has_off,
        lags=lag_frames(ds.sizes["lag_frame"] + 1, skip_last),
        period=ds.attrs.get("period_s"),
        roi_q=ds.roi_q.values,
        binned=binned,
        n_blocks=n_blocks,
    )


def xpcs_run(
    run_nr: int,
    db,
    *,
    averaging: str = "pooled",
    binned: bool = True,
    v0_run: int | None = None,
    n_pulses: int = 155,
    ferritin_mg_per_ml_initial: float = 50.0,
    peg_percent_wv_initial: float = 5.0,
    nacl_mM_initial: float = 150.0,
    n_fe: float = 1800.0,
    proposal: int = 10400,
    results_file=None,
    raw=None,
    cache_dir=DEFAULT_CACHE_DIR,
    refresh: bool = False,
    n_workers: int | None = None,
) -> xr.Dataset:
    """XPCS g2 and two-time maps of one run's hit trains with their scaling, cached.

    Reads the hit-train results file (`analysis.xpcs.reader.XpcsHits`), whose pooled
    two-time maps need every hit's ``(rings, P, P)`` cube, so a notebook loads this
    instead. Three families of variables share the ``roi_q`` axis:

    ``g2``, ``sigma_g2``, ``g2_paired``, ``baseline``, ``c2``, ``sigma``, (roi_q, tau)
        `XpcsHits.g2` with its ``lag`` coordinate on ``tau``. With ``averaging="es"``
        only ``g2``, and a ring extra-speckle's noisy-ring rule leaves out is NaN.
    ``ttcf_raw``, ``ttcf_off``, ``ttcf_c2`` (``ttcf_raw_paired`` pooled) over
    (roi_q, frame_1, frame_2)
        `XpcsHits.ttcf`, prefixed because its ``c2`` is not g2's.
    per hit train, labelled by trainId, every hit including the unusable ones
        ``status`` (`FrameStatus`), ``pair_status``, ``partner`` (trainId, 0 none),
        ``has_off``, ``kbar`` over (trainId, frame, roi_q), NaN for a train without
        sums, ``time``, the `analysis.utils.droplet_scaling` fields and ``droplet_*``
        as in `saxs_run`, stored, never applied.

    ``train_g2_num``, ``train_g2_den``, ``train_baseline_num``, ``train_baseline_den``
    over (trainId, roi_q, lag_frame), pooled averaging only
        Each hit train's lag terms of g2 and of its baseline, lags ``1 ... P - 1``
        (``lag_frame``, with ``lag_tau`` in seconds), the numerators and
        denominators `XpcsHits.g2` pools. NaN for an unusable train; the baseline
        terms are NaN for a usable train without a baseline. ``num / den`` is one
        train's g2, which is noisy: re-pool a selection of trains with `pool_g2`.
        ``averaging="es"`` files do not carry them.

    ``ring_bounds_nm`` holds the ring edges when the file has them. The attributes
    carry the results file's ``schema``, ``merge_hash``, ``hits_sha256`` and
    ``hits_source``, the ``coverage`` text and ``period_s``. The keyword arguments are
    part of the cache key, and the file's size and mtime are a source, so a rewritten
    results file rebuilds the cache. `n_workers` is not: it is the processes that
    read the pooled sums (once, for both g2 and the maps), ``None`` for the physical
    cores up to 16, and changes no value beyond the float grouping of the maps' sums.

    :raises FileNotFoundError: no results file.
    :raises IncompleteRun: a hit train is still unfinished.
    :raises ValueError: `averaging` is unknown, or no hit train is usable.
    """
    from analysis.common.cpu import physical_cores
    from analysis.common.status import UNFINISHED
    from analysis.common.writer import IncompleteRun
    from analysis.xpcs.damnit import result_file
    from analysis.xpcs.reader import Averaging, XpcsHits

    if averaging not in XPCS_AVERAGINGS:
        raise ValueError(
            f"averaging must be one of {XPCS_AVERAGINGS}, got {averaging!r}"
        )
    run_nr, n_pulses, proposal = int(run_nr), int(n_pulses), int(proposal)
    v0_run = run_nr if v0_run is None else int(v0_run)
    results_file = Path(
        result_file(proposal, run_nr) if results_file is None else results_file
    )
    if not results_file.exists():
        raise FileNotFoundError(f"no XPCS results for r{run_nr} at {results_file}")
    hits = XpcsHits(results_file)
    if not hits.train_ids.size:
        raise ValueError(f"r{run_nr}: the results file has no hit train")
    unfinished = np.isin(hits.status, [int(code) for code in UNFINISHED])
    if unfinished.any():
        raise IncompleteRun(f"{int(unfinished.sum())} hit trains are unfinished")
    if not hits.usable.any():
        raise ValueError(f"r{run_nr}: no usable hit train ({hits.coverage()})")

    composition = _composition(
        ferritin_mg_per_ml_initial, peg_percent_wv_initial, nacl_mM_initial, n_fe
    )
    inputs = {
        "kind": "xpcs_run",
        "proposal": proposal,
        "run": run_nr,
        "v0_run": v0_run,
        "n_pulses": n_pulses,
        "averaging": averaging,
        "binned": bool(binned),
        "layout": 2,
        **composition,
    }
    sources = [file_source(results_file)]
    sources += [file_source(db[r].file) for r in sorted({run_nr, v0_run})]
    path = _cache_path(cache_dir, f"xpcs_r{run_nr:04d}_{averaging}", inputs)
    if not refresh and (cached := _load(path, inputs, sources)) is not None:
        return cached

    mode = Averaging(averaging)
    if mode is Averaging.POOLED:
        if n_workers is None:
            n_workers = min(physical_cores() or 1, 16)
        sums = hits.pooled_sums(n_workers=n_workers)
        g2 = hits.g2(mode, binned=binned, sums=sums)
        ttcf = hits.ttcf(mode, sums=sums)
        terms = _train_terms(hits, sums)
    else:
        g2 = hits.g2(mode, binned=binned)
        ttcf = hits.ttcf(mode)
        terms = None
    ds = xr.merge(
        [g2, ttcf.rename({v: f"ttcf_{v}" for v in ttcf.data_vars})],
        join="outer",
        combine_attrs="drop",
    )

    tids = hits.train_ids
    if terms is not None:
        n_lags = terms["train_g2_num"].shape[-1]
        ds = ds.assign_coords(lag_frame=np.arange(1, n_lags + 1))
        if (period := g2.attrs.get("period_s")) is not None:
            ds = ds.assign_coords(lag_tau=("lag_frame", ds.lag_frame.values * period))
        for name, values in terms.items():
            ds[name] = xr.DataArray(
                values,
                dims=("trainId", "roi_q", "lag_frame"),
                coords={"trainId": tids},
            )
            ds[name].encoding.update(zlib=True, complevel=1, shuffle=True)
    by_train = {
        "status": hits.status.astype(np.uint8),
        "pair_status": hits.pair_status.astype(np.uint8),
        "partner": hits.partner_ids,
        "has_off": hits.has_off,
    }
    for name, values in by_train.items():
        ds[name] = xr.DataArray(
            values, dims="trainId", coords={"trainId": tids}, name=name
        )
    ds["kbar"] = hits.kbar().reindex(trainId=tids)
    if len(hits.bounds_nm):
        ds["ring_bounds_nm"] = (("roi_q", "edge"), hits.bounds_nm)
    droplet = _add_droplet(
        ds, tids, run_nr, db, raw, proposal, n_pulses, v0_run, composition
    )

    identity = {
        k: (v.item() if isinstance(v, np.generic) else str(v))
        for k in ("schema", "merge_hash", "hits_sha256", "hits_source")
        if (v := hits.attrs.get(k)) is not None
    }
    ds.attrs.update(
        _attrs(inputs, sources, "XPCS hit-train results; see analysis.cache.xpcs_run"),
        run=run_nr,
        v0_run=v0_run,
        averaging=averaging,
        binned=int(g2.attrs.get("binned", 0)),
        n_trains=ttcf.attrs["n_trains"],
        n_with_baseline=ttcf.attrs["n_with_baseline"],
        coverage=str(hits.coverage()),
        results_file=str(results_file),
        **({"period_s": g2.attrs["period_s"]} if "period_s" in g2.attrs else {}),
        **identity,
        **droplet,
    )
    write_dataset(ds, path)
    return ds


def droplet_timeline(
    runs,
    db,
    *,
    stop_at_min_volume: bool = False,
    t_star="fit",
    t_star_guess: float | None = None,
    proposal: int = 10400,
    raws: dict | None = None,
    cache_dir=DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> xr.Dataset:
    """One droplet's volume against time over consecutive runs, with its D^2-law fit.

    Every train with a droplet fit and a timestamp is stored, labelled by trainId;
    ``elapsed_s`` counts from the first of them. `analysis.utils.fit_D2_law` is fitted
    to the trains marked ``used_in_fit``: all of them, or with `stop_at_min_volume`
    only those up to the smallest volume, for a droplet that is lost or disturbed
    after it. Its parameters are attributes; nothing is derived from them here.

    `t_star` chooses the attribute ``t_star_s``:
    ``"fit"``, the fitted break; ``"end"``, the last train used in the fit, for a
    droplet whose D^2 law never breaks before it is lost; or a number of seconds.
    ``t_star_fit_s`` always holds the fitted value, and ``t_star_source`` says which
    one ``t_star_s`` is.
    """
    runs = [int(r) for r in runs]
    if isinstance(t_star, str) and t_star not in ("fit", "end"):
        raise ValueError(f"t_star must be 'fit', 'end' or seconds, got {t_star!r}")
    inputs = {
        "kind": "droplet_timeline",
        "proposal": int(proposal),
        "runs": runs,
        "stop_at_min_volume": bool(stop_at_min_volume),
        "t_star": t_star if isinstance(t_star, str) else float(t_star),
        "t_star_guess": None if t_star_guess is None else float(t_star_guess),
    }
    sources = [file_source(db[r].file) for r in runs]
    path = _cache_path(cache_dir, f"timeline_r{runs[0]:04d}-r{runs[-1]:04d}", inputs)
    if not refresh and (cached := _load(path, inputs, sources)) is not None:
        return cached

    parts = []
    for run_nr in runs:
        raw = raws[run_nr] if raws else utils.ex.open_run(proposal, run_nr, data="raw")
        volume = xr.open_dataset(db[run_nr].file, group="droplet_fit").volume
        part = xr.Dataset({"volume_mm3": volume, "time": _train_times(raw)})
        part = part.dropna("trainId", how="any")
        part["run"] = ("trainId", np.full(part.sizes["trainId"], run_nr, np.int32))
        parts.append(part)
    tl = xr.concat(parts, "trainId").sortby("trainId")
    if np.unique(tl.trainId).size != tl.sizes["trainId"]:
        raise ValueError(f"runs {runs} share trainIds; they cannot be one droplet")

    elapsed = ((tl.time - tl.time[0]) / np.timedelta64(1, "s")).values
    volume = tl.volume_mm3.values.astype(np.float64)
    used = np.ones(volume.size, dtype=bool)
    if stop_at_min_volume:
        used[int(np.argmin(volume)) + 1 :] = False
    popt, pcov = utils.fit_D2_law(
        elapsed[used], volume[used], t_star_guess=t_star_guess
    )
    if t_star == "fit":
        chosen = float(popt[3])
    elif t_star == "end":
        chosen = float(elapsed[used][-1])
    else:
        chosen = float(t_star)

    tl["elapsed_s"] = ("trainId", elapsed)
    tl["used_in_fit"] = ("trainId", used)
    tl.attrs.update(
        _attrs(
            inputs,
            sources,
            "droplet volume vs time; see analysis.cache.droplet_timeline",
        ),
        runs=runs,
        t0_utc=str(tl.time.values[0]),
        D0_sq_mm2=float(popt[0]),
        k1_mm2_per_s=float(popt[1]),
        k2_mm2_per_s=float(popt[2]),
        t_star_fit_s=float(popt[3]),
        t_star_fit_err_s=float(np.sqrt(pcov[3, 3])),
        fit_covariance=pcov.ravel(),
        t_star_s=chosen,
        t_star_source="override" if not isinstance(t_star, str) else t_star,
    )
    write_dataset(tl, path)
    return tl


def _dir_source(path) -> dict:
    """A proc run directory's identity: its file count, total size and newest mtime."""
    stats = [os.stat(p) for p in sorted(Path(path).glob("*.h5"))]
    if not stats:
        raise FileNotFoundError(f"no .h5 files under {path}")
    return {
        "path": str(path),
        "n_files": len(stats),
        "size": sum(s.st_size for s in stats),
        "mtime_ns": max(s.st_mtime_ns for s in stats),
    }


def _open_proc(run_dir):
    import extra_data as ed

    return ed.RunDirectory(str(run_dir))


def _sum_job(args) -> tuple[int, np.ndarray, np.ndarray, int]:
    """Worker: one module's counts and valid-frame counts over a few trains."""
    run_dir, train_ids, modno = args
    # EXtra-data's threaded decompression is slower than the HDF5 path here
    os.environ.setdefault("EXTRA_NUM_THREADS", "1")
    from extra_data import by_id
    from extra_data.components import AGIPD1M

    det = AGIPD1M(
        _open_proc(run_dir).select_trains(by_id[list(train_ids)]),
        min_modules=N_MODULES,
    )
    source = det.modno_to_source[modno]
    data = det.data[source, "image.data"].ndarray()
    valid = det.data[source, "image.mask"].ndarray() == 0
    counts = np.where(valid, data, 0).sum(axis=0, dtype=np.int64)
    return modno, counts, valid.sum(axis=0, dtype=np.int64), data.shape[0]


def _read_sums(run_dir, train_ids: list[int], n_workers: int | None):
    """Sum each module over `train_ids` in a pool; all modules must be present."""
    from extra_data import by_id
    from extra_data.components import AGIPD1M

    from analysis.common.cpu import default_pool, physical_cores

    det = AGIPD1M(
        _open_proc(run_dir).select_trains(by_id[train_ids]), min_modules=N_MODULES
    )
    missing = sorted(set(train_ids) - {int(t) for t in det.train_ids})
    if missing:
        raise ValueError(
            f"{len(missing)} requested trains have no frames from all {N_MODULES} "
            f"modules in {run_dir}: {missing[:10]}"
        )
    chunks = [train_ids[i : i + SUM_CHUNK] for i in range(0, len(train_ids), SUM_CHUNK)]
    jobs = [(str(run_dir), chunk, m) for chunk in chunks for m in range(N_MODULES)]
    workers = n_workers or min(len(jobs), physical_cores() or 8, 32)

    counts = np.zeros((N_MODULES, *MODULE_SHAPE), np.int64)
    valid = np.zeros_like(counts)
    frames = np.zeros(N_MODULES, np.int64)
    with default_pool(workers) as pool:
        for modno, c, v, n in pool.map(_sum_job, jobs):
            counts[modno] += c
            valid[modno] += v
            frames[modno] += n
    if np.unique(frames).size != 1:
        raise RuntimeError(f"modules returned different frame counts: {frames}")
    return counts, valid, int(frames[0])


def detector_sums(
    run_nr: int,
    train_ids,
    *,
    label: str = "",
    proposal: int = 10400,
    run_dir=None,
    n_workers: int | None = None,
    cache_dir=DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> xr.Dataset:
    """Photon counts per AGIPD pixel, summed over every frame of `train_ids`, cached.

    Returns a Dataset over (module, slow, fast):

    ``counts``
        Sum of ``image.data`` over the frames whose ``image.mask`` is 0 for that pixel.
    ``valid_frames``
        How many frames that is, per pixel; ``counts / valid_frames`` is the mean
        count per frame.

    Nothing else is applied: no static mask, no flux or path normalisation. The train
    ids are stored as ``train_id``; every one must have frames from all 16 modules,
    else this raises rather than summing fewer. `label` only names the file. The
    corrected data come from the proposal's ``proc`` tree unless `run_dir` is given.
    """
    run_nr, proposal = int(run_nr), int(proposal)
    train_ids = sorted({int(t) for t in train_ids})
    if not train_ids:
        raise ValueError("no train ids given")
    run_dir = Path(run_dir) if run_dir is not None else PROC_ROOT / f"r{run_nr:04d}"
    digest = hashlib.sha256(np.asarray(train_ids, np.uint64).tobytes()).hexdigest()
    inputs = {
        "kind": "detector_sums",
        "proposal": proposal,
        "run": run_nr,
        "n_trains": len(train_ids),
        "train_ids_sha256": digest,
    }
    sources = [_dir_source(run_dir)]
    stem = f"sums_r{run_nr:04d}" + (f"_{label}" if label else "")
    path = _cache_path(cache_dir, stem, inputs)
    if not refresh and (cached := _load(path, inputs, sources)) is not None:
        return cached

    counts, valid, n_frames = _read_sums(run_dir, train_ids, n_workers)
    dims = ("module", "slow", "fast")
    ds = xr.Dataset(
        {
            "counts": (dims, counts.astype(np.int32)),
            "valid_frames": (dims, valid.astype(np.int32)),
            "train_id": ("train", np.asarray(train_ids, np.uint64)),
        },
        coords={"module": np.arange(N_MODULES)},
    )
    ds["counts"].attrs["description"] = (
        "sum of image.data over the frames with image.mask == 0 for the pixel"
    )
    ds.attrs.update(
        _attrs(
            inputs, sources, "per-pixel photon sums; see analysis.cache.detector_sums"
        ),
        run=run_nr,
        n_frames=n_frames,
        label=label,
    )
    write_dataset(ds, path)
    return ds
