#!/usr/bin/env python
"""P4 node acceptance for the hit-train XPCS pass (``context/xpcs-hits.md``).

One SLURM job per run. Everything is written under
``scratch/xpcs_hits_acceptance/r####/`` and the verdicts to a JSON beside this
script. Nothing goes into ``usr/cached_files`` or any existing product directory.

Stages (``--stages``; the default depends on the run):

    hits   the hit ids (``analysis.triage.classify.hits``, signals from DAMNIT's
           file or, failing that, the signals cache), first N of them
    main   extract + merge the hits (Kernel.CLEAN) from nothing: B2 and B3 timings
    g1     extra-speckle ``xpcs_offline(selected_trains=...)`` in a separate
           process, then Kernel.ES + Averaging.ES on its ring masks: TTCF_raw,
           TTCF_off, kbar, TTCF, g2 to <= 1e-6 relative; and G3 on the same rings
    g2     every ring pixel that differs between extra-speckle's rings and ours,
           attributed to a named cause
    g3     CLEAN against the ES view per ring at lags 1, 10, 100 (own geometry)
    g4     8 against 36 workers, and block size: bit-identical sums
    g5     rerun of a complete run: inode, mtime, sha256 unchanged
    g6     sizes, timings, and B4 (partner gap 1, 2, 5, 10, 50)
    e3     our g2 against ``analysis.triage.xpcs`` on DAMNIT's ``xpcs_ttcf``
    a1     the store's per-frame ring photon sums against ``agipd_photon_stats``

    python scripts/xpcs_hits_acceptance.py --run 423 493 -w 36
    python scripts/xpcs_hits_acceptance.py --run 423 -w 36 --local --stages hits main

The ``es-child`` mode (``python scripts/xpcs_hits_acceptance.py es-child ...``)
is how the extra-speckle run is made: importing ``extra_speckle`` switches off
HDF5 file locking for the whole process (CLAUDE.md pitfall 29), so it never
happens in the process that writes our files.

Needs Maxwell: proc data, geometry, the pixel mask, DAMNIT's files. Not
unit-tested (CLAUDE.md working rule 7).
"""

from __future__ import annotations

# Before numpy, pyFAI or EXtra-data import anything that builds a thread pool.
from _common import add_job_args, dispatch, report_header, write_report  # noqa: E402

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common.cpu import file_sha256  # noqa: E402
from analysis.common.slurm import PROPOSAL_ROOT  # noqa: E402
from analysis.saxs.config import NPIX, SHAPE, AgipdSaxsConfig  # noqa: E402
from analysis.xpcs.config import RING_EDGES_NM, Kernel, XpcsConfig  # noqa: E402
from analysis.xpcs.hits import HitSet  # noqa: E402
from analysis.xpcs.kernel import train_sums  # noqa: E402
from analysis.xpcs.result import ES_KBAR_MIN, Averaging, XpcsHits  # noqa: E402
from analysis.xpcs.run import run_xpcs_hits  # noqa: E402
from analysis.xpcs.store import RingPhotonStore  # noqa: E402

log = logging.getLogger("scripts.xpcs_hits_acceptance")

SCRATCH = PROPOSAL_ROOT / "scratch"
ACCEPT = SCRATCH / "xpcs_hits_acceptance"
DAMNIT_FILE = PROPOSAL_ROOT / "usr/Shared/amore/extracted_data/p10400_r{run}.h5"
PHOTON_STATS = PROPOSAL_ROOT / "usr/cached_files/agipd_photon_stats"
GEOMETRY = PROPOSAL_ROOT / "usr/geometry/geom_latest.geom"
PIXEL_MASK = PROPOSAL_ROOT / "usr/masks/mask_2026-09-08_AGIPD_SAXS.npy"
#: What DAMNIT hands extra-speckle (src/amore/context.py): Å⁻¹ edges, px, py, mm.
ES_EDGES_A = (0.0225, 0.035, 0.045, 0.056, 0.065, 0.069, 0.075, 0.086)
ES_CENTRE = (607.4598, 672.0767)
ES_SDD_MM = 7531.5962
ES_TOLERANCE = 1e-6
GAPS = (1, 2, 5, 10, 50)
LAGS = (1, 10, 100)
#: Hit trains used per run (None: every hit). r0493 is the full-size run.
N_HITS = {423: 200, 493: None}
DEFAULT_N_HITS = 400
DEFAULT_STAGES = {
    423: ["hits", "main", "g1", "g2", "g3", "g4", "g5", "g6", "e3", "a1"],
    493: ["hits", "main", "g3", "g5", "g6", "e3", "a1"],
}
REFERENCE_STAGES = ["hits", "main", "g3", "g5", "g6", "e3"]


# ── small helpers ─────────────────────────────────────────────────────────────


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else str(value)
    return value


def fresh(path: Path) -> None:
    """Empty one of this script's own scratch directories (requeue leftovers)."""
    if ACCEPT in path.parents and path.exists():
        shutil.rmtree(path)


def stat(path: Path) -> dict[str, Any]:
    """Inode, mtime, size and sha256: what a rerun must leave alone."""
    s = path.stat()
    return {
        "inode": s.st_ino,
        "mtime_ns": s.st_mtime_ns,
        "size": s.st_size,
        "sha256": file_sha256(path),
    }


def compare(a: Any, b: Any) -> dict[str, Any]:
    """How far ``a`` is from ``b``: NaN pattern, elementwise and global relative."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    out: dict[str, Any] = {
        "n_finite_both": int(both.sum()),
        "nan_pattern_mismatch": int((fa != fb).sum()),
    }
    if not both.any():
        return out
    d, scale = np.abs(a - b)[both], np.abs(b)[both]
    pos = scale > 0
    out["max_abs"] = float(d.max())
    out["max_rel_elementwise"] = float((d[pos] / scale[pos]).max()) if pos.any() else 0
    out["max_rel_to_max"] = float(d.max() / scale.max()) if scale.max() > 0 else 0.0
    out["p99_rel_elementwise"] = (
        float(np.percentile(d[pos] / scale[pos], 99)) if pos.any() else 0.0
    )
    out["n_zero_in_reference_with_difference"] = int(((~pos) & (d > 0)).sum())
    return out


def provenance(path: Path) -> dict[str, Any]:
    """The last pass recorded in a store or a results file."""
    with h5py.File(path, "r") as f:
        attrs = f["provenance"].attrs
        if "history" in attrs:
            history = json.loads(attrs["history"])
            return history[-1] if history else {}
        return {
            k: (json.loads(v) if isinstance(v, str) and v[:1] in "{[" else v)
            for k, v in attrs.items()
        }


def group_digest(path: Path, prefixes: tuple[str, ...]) -> str:
    """sha256 over the datasets under ``prefixes`` in name order: contents only."""
    digest = hashlib.sha256()
    with h5py.File(path, "r") as f:
        names: list[str] = []
        f.visititems(
            lambda n, o: (
                names.append(n)
                if isinstance(o, h5py.Dataset) and n.startswith(prefixes)
                else None
            )
        )
        for name in sorted(names):
            data = np.ascontiguousarray(f[name][()])
            digest.update(name.encode())
            digest.update(str(data.dtype).encode())
            digest.update(str(data.shape).encode())
            digest.update(data.tobytes())
    return digest.hexdigest()


def sums_digests(results: Path, store: Path) -> dict[str, str]:
    return {
        "results_sums": group_digest(results, ("sums/",)),
        "results_hits": group_digest(results, ("hits/", "pattern/", "rings/")),
        "store_trains": group_digest(store, ("trains/",)),
    }


class Ctx:
    """What a run's stages share."""

    def __init__(self, args: argparse.Namespace, run: int) -> None:
        self.args = args
        self.run = run
        self.workers = args.workers
        self.dir = ACCEPT / f"r{run:04d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ids: np.ndarray | None = None
        self.report: dict[str, Any] = {}

    def saxs(self, **changes: Any) -> AgipdSaxsConfig:
        return AgipdSaxsConfig(proposal=self.args.proposal, run=self.run, **changes)

    def cfg(self, root: Path, **changes: Any) -> XpcsConfig:
        changes.setdefault("n_workers", self.workers)
        return XpcsConfig(saxs=self.saxs(), output_root=str(root), **changes)

    def hit_set(self, ids: np.ndarray | None = None) -> HitSet:
        return HitSet.of(self.ids if ids is None else ids, source=self.source)

    source = ""


def run_pass(cfg: XpcsConfig, hits: HitSet) -> dict[str, Any]:
    """One pass, timed; the paths of what it wrote."""
    started = time.perf_counter()
    path = run_xpcs_hits(cfg, hits)
    return {
        "wall_s": time.perf_counter() - started,
        "results": path,
        "store": cfg.store_file,
        "n_hits": len(hits),
    }


# ── hits ──────────────────────────────────────────────────────────────────────


def stage_hits(ctx: Ctx) -> dict[str, Any]:
    """The run's hit trains, from DAMNIT's signals or else the signals cache."""
    import xarray as xr

    from analysis.triage.classify import CACHE_DIR, hits

    info: dict[str, Any] = {}
    try:
        out = hits(ctx.run)
        info["signals"] = "DAMNIT file"
    except LookupError as error:
        info["damnit_error"] = str(error)
        cached = sorted(Path(CACHE_DIR).glob(f"signals_r{ctx.run:04d}_*.nc"))
        if not cached:
            raise LookupError(
                f"r{ctx.run}: DAMNIT has no agipd_saxs and there is no signals cache"
            ) from error
        signals = xr.load_dataset(cached[-1])
        out = hits(ctx.run, signals=signals)
        info["signals"] = f"signals cache {cached[-1].name}"
        info["signals_cache_created_utc"] = signals.attrs.get("created_utc")
        info["signals_cache_n_trains"] = int(signals.sizes["trainId"])
    ids = np.sort(np.asarray(out["train_ids"], dtype=np.uint64))
    n = N_HITS.get(ctx.run, DEFAULT_N_HITS)
    table = sorted((SCRATCH / "agipd_saxs" / f"r{ctx.run:04d}").glob("agipd_saxs*.h5"))
    if table:
        with h5py.File(table[-1], "r") as f:
            run_ids = f["trains/trainId"][()][f["trains/status"][()] == 0]
        foreign = int((~np.isin(ids, run_ids)).sum())
        info["hits_not_in_run"] = foreign
        if foreign:
            # a cache of another run's trains: use consecutive OK trains instead
            mid = run_ids.size // 3
            ids = np.sort(run_ids[mid : mid + (n or run_ids.size)]).astype(np.uint64)
            info["signals"] += f" REJECTED ({foreign} ids not in the run); stand-in"
            info["stand_in_table"] = table[-1].name
            out = {**out, "kind": "stand-in: consecutive OK trains"}
    ctx.source = f"classify.hits r{ctx.run} ({info['signals']}), first {n or 'all'}"
    ctx.ids = ids if n is None else ids[:n]
    np.save(ctx.dir / "hit_ids.npy", ctx.ids)
    steps = np.diff(ctx.ids.astype(np.int64))
    info |= {
        "kind": out["kind"],
        "n_hits_in_run": int(ids.size),
        "n_used": int(ctx.ids.size),
        "first_id": int(ctx.ids[0]),
        "last_id": int(ctx.ids[-1]),
        "consecutive_fraction": float((steps == 1).mean()) if steps.size else None,
        "summary": out["summary"],
    }
    return info


# ── main: extract + merge from nothing ────────────────────────────────────────


def stage_main(ctx: Ctx) -> dict[str, Any]:
    """CLEAN on the geometry's rings, store and results from nothing (B2, B3)."""
    root = ctx.dir / "main"
    fresh(root)
    cfg = ctx.cfg(root)
    hits = ctx.hit_set()
    first = run_pass(cfg, hits)
    # the same hits, another kernel: nothing to extract, so this is the merge alone
    es_cfg = ctx.cfg(root, kernel=Kernel.ES)
    es_out = root / "xpcs_hits" / f"r{ctx.run:04d}" / "xpcs_hits_es.h5"
    merge_started = time.perf_counter()
    run_xpcs_hits(es_cfg, hits, output_path=es_out)
    merge_wall = time.perf_counter() - merge_started

    store_prov, result_prov = provenance(first["store"]), provenance(first["results"])
    n = len(hits)
    store_bytes, result_bytes = first["store"].stat().st_size, es_out.stat().st_size
    with RingPhotonStore.open(first["store"]) as s:
        summary = s.status_summary()
    summary_out = {
        "n_hits": n,
        "workers": ctx.workers,
        "first_pass_wall_s": first["wall_s"],
        "merge_only_wall_s": merge_wall,
        "extract_wall_s_per_train": (first["wall_s"] - merge_wall) / n,
        "extract_core_s": store_prov.get("timings"),
        "extract_setup_s": store_prov.get("setup_timings"),
        "merge_core_s": result_prov.get("timings"),
        "store_status": summary,
        "paths": {"store": str(first["store"]), "results": str(first["results"])},
        "es_kernel_results": str(es_out),
        "store_bytes": store_bytes,
        "results_bytes": first["results"].stat().st_size,
        "es_results_bytes": result_bytes,
        "host": store_prov.get("host"),
    }
    (ctx.dir / "main.json").write_text(json.dumps(jsonable(summary_out), indent=1))
    return summary_out


def open_store(path: Path) -> RingPhotonStore:
    """Read-only, whatever hash it was made under."""
    return RingPhotonStore.open(path)


# ── extra-speckle, in its own process ─────────────────────────────────────────


def stage_es(ctx: Ctx, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    done = out / "es_setup.json"
    if done.exists() and np.array_equal(np.load(out / "ids.npy"), ctx.ids):
        return json.loads(done.read_text()) | {"reused": True}
    np.save(out / "ids.npy", ctx.ids)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "es-child",
        str(ctx.run),
        str(out / "ids.npy"),
        str(out),
        str(ctx.workers),
    ]
    started = time.perf_counter()
    done = subprocess.run(
        command, capture_output=True, text=True, env=os.environ | {"MPLBACKEND": "Agg"}
    )
    (out / "es_stdout.txt").write_text(done.stdout)
    (out / "es_stderr.txt").write_text(done.stderr)
    if done.returncode:
        raise RuntimeError(
            f"extra-speckle child failed ({done.returncode}); see {out}/es_stderr.txt: "
            f"{done.stderr[-1500:]}"
        )
    info = json.loads((out / "es_setup.json").read_text())
    info["wall_s"] = time.perf_counter() - started
    return info


def es_child(run: int, ids_file: str, out: str, workers: int) -> int:
    """The extra-speckle run, and what the comparisons need from it."""
    import xarray as xr
    from extra_speckle.pipeline import xpcs_offline
    from extra_speckle.setup import Setup
    from extra_speckle.utils import get_run
    from extra_speckle.xpcs.tools import (
        baseline_subtraction_and_averaging,
        g2_to_bins,
        reduce_ttcf,
    )

    out_dir = Path(out)
    ids = np.load(ids_file)
    q_range = np.array(ES_EDGES_A)
    results = xpcs_offline(
        10400,
        run,
        out_dir=str(out_dir),
        geom=str(GEOMETRY),
        mask=str(PIXEL_MASK),
        px=ES_CENTRE[0],
        py=ES_CENTRE[1],
        sdd=ES_SDD_MM,
        q_range=q_range,
        selected_trains=[int(t) for t in ids],
        calc_stats=True,
        run_xpcs_analysis=True,
        subtract_TTCF_off=False,
        cache_dir=str(out_dir / "cache"),
        workers_no=workers,
        save_files=True,
        show_plots=False,
    )
    setup = results["setup"]
    raw, off = results["TTCF_raw"], results["TTCF_off"]
    mean, _ = baseline_subtraction_and_averaging(
        raw, off, True, rep_rate=setup.rep_rate, out_dir=None
    )
    g2 = reduce_ttcf(mean, rep_rate=setup.rep_rate, mask_noisy=True, nan_thresh=0.025)
    g2 = g2_to_bins(g2, bin_thresh=50, bin_no=20, bin_skip=10)
    xr.Dataset({"g2": g2}).to_netcdf(out_dir / "g2_c2.nc")
    mean_raw = raw.mean("trainId", skipna=True)
    mean_off = off.mean("trainId", skipna=True)
    xr.Dataset({"raw": mean_raw, "off": mean_off}).to_netcdf(out_dir / "means.nc")
    np.save(out_dir / "qA.npy", np.asarray(setup.qA))
    np.save(out_dir / "setup_mask.npy", np.asarray(setup.mask))
    info: dict[str, Any] = {
        "photon_energy_kev": float(setup.photon_energy),
        "wavelength_m": float(setup.wavelength),
        "sdd_mm": float(setup.config.sdd),
        "px": float(setup.config.px),
        "py": float(setup.config.py),
        "rep_rate_hz": None if setup.rep_rate is None else float(setup.rep_rate),
        "q_range_A": q_range.tolist(),
        "geom": str(GEOMETRY),
        "mask_file": str(PIXEL_MASK),
        "n_trains": int(raw.sizes["trainId"]),
        "extra_speckle_file": __import__("extra_speckle").__file__,
    }
    try:
        run_obj, _ = get_run(10400, run, str(out_dir / "encoder_setup"))
        encoder = Setup(
            run=run_obj,
            out_dir=out_dir / "encoder_setup",
            detector="AGIPD 1M",
            configuration="SAXS",
            mask=str(PIXEL_MASK),
            geom=None,
            sdd=ES_SDD_MM,
            px=ES_CENTRE[0],
            py=ES_CENTRE[1],
            q_range=q_range,
        )
        rings = np.stack([~encoder.mask * ring for ring in encoder.index])
        np.save(out_dir / "all_masks_encoder.npy", rings)
        np.save(out_dir / "qA_encoder.npy", np.asarray(encoder.qA))
        info["encoder_setup"] = {
            "photon_energy_kev": float(encoder.photon_energy),
            "sdd_mm": float(encoder.config.sdd),
        }
    except Exception:  # noqa: BLE001 - recorded, the comparison is optional
        info["encoder_setup_error"] = traceback.format_exc()[-1500:]
    (out_dir / "es_setup.json").write_text(json.dumps(info, indent=2))
    return 0


# ── G1 ────────────────────────────────────────────────────────────────────────


def stage_g1(ctx: Ctx) -> dict[str, Any]:
    import xarray as xr

    es_dir = ctx.dir / "es"
    fresh(ctx.dir / "g1")
    es_info = stage_es(ctx, es_dir)
    es_ids = np.sort(xr.open_dataarray(es_dir / "TTCF_raw.nc").trainId.values)
    hits = ctx.hit_set(es_ids.astype(np.uint64))
    masks_file = es_dir / "all_masks.npy"
    root = ctx.dir / "g1"
    es_pass = run_pass(
        ctx.cfg(root / "es", kernel=Kernel.ES, ring_masks_file=str(masks_file)),
        hits,
    )
    # the CLEAN kernel on the same store, for G3 on identical rings
    clean_cfg = ctx.cfg(
        root / "es", kernel=Kernel.CLEAN, ring_masks_file=str(masks_file)
    )
    clean_out = clean_cfg.output_file.with_name("xpcs_hits_clean.h5")
    run_xpcs_hits(clean_cfg, hits, output_path=clean_out)

    ours = XpcsHits(es_pass["results"])
    cov = ours.coverage()
    es_raw = xr.open_dataarray(es_dir / "TTCF_raw.nc").sortby("trainId")
    es_off = xr.open_dataarray(es_dir / "TTCF_off.nc").sortby("trainId")
    es_kbar = xr.open_dataarray(es_dir / "kbar.nc").sortby("trainId")
    es_mean = xr.open_dataarray(es_dir / "TTCF.nc")
    es_g2 = xr.open_dataset(es_dir / "g2_c2.nc")["g2"]
    trains = es_raw.trainId.values.astype(np.uint64)
    per = ours.per_train(trains)
    # the pipeline's kbar filter (kbar_filter) is applied to the saved TTCF_raw/off
    lit = (ours.kbar(trains).values > ES_KBAR_MIN).transpose(0, 2, 1)  # (T, R, F)
    keep = lit[..., :, None] & lit[..., None, :]
    ours_raw = np.where(keep, per["TTCF_raw"].values, np.nan)
    ours_off = np.where(keep, per["TTCF_off"].values, np.nan)
    gate: dict[str, Any] = {
        "es": es_info,
        "n_es_trains": int(trains.size),
        "same_train_ids": bool(
            np.array_equal(per.trainId.values.astype(np.uint64), trains)
        ),
        "coverage": str(cov),
        "pair_status": cov.by_pairing,
        "TTCF_raw": compare(ours_raw, es_raw.values),
        "TTCF_off": compare(ours_off, es_off.values),
        "kbar": compare(
            ours.kbar(trains).values,
            es_kbar.transpose("trainId", "frame", "roi_q").values,
        ),
    }
    kbar_min = ES_KBAR_MIN
    ttcf = ours.ttcf(Averaging.ES, trains, kbar_min=kbar_min)
    gate["TTCF"] = compare(ttcf["raw"].values, es_mean.values)
    ours_g2 = ours.g2(Averaging.ES, trains, kbar_min=kbar_min)["g2"]
    # rings from a masks file are numbered, so ES's centres are matched by edges
    es_centres = 0.5 * (np.array(ES_EDGES_A[:-1]) + np.array(ES_EDGES_A[1:]))
    rings = [int(np.argmin(np.abs(es_centres - q))) for q in es_g2.roi_q.values]
    gate["g2"] = compare(ours_g2.values[rings], es_g2.values)
    gate["g2_rings_extra_speckle_kept"] = rings
    gate["g2_rings_extra_speckle_dropped_as_noisy"] = [
        r for r in range(ours.n_rings) if r not in rings
    ]
    # the filter the pipeline applied is the kbar one, so ours must not differ in NaNs
    gate["tau_first_ours_s"] = float(ours.g2(Averaging.ES, trains)["tau"].values[0])
    gate["tau_first_es_s"] = (
        float(es_g2[es_g2.dims[-1]].values[0]) if es_g2.ndim == 2 else None
    )
    worst = max(
        gate[k].get("max_rel_elementwise", np.inf)
        for k in ("TTCF_raw", "TTCF_off", "kbar", "TTCF", "g2")
    )
    gate["worst_max_rel_elementwise"] = worst
    gate["pass"] = bool(
        gate["same_train_ids"]
        and worst <= ES_TOLERANCE
        and all(
            gate[k].get("nan_pattern_mismatch", 1) == 0
            for k in ("TTCF_raw", "TTCF_off", "kbar", "TTCF", "g2")
        )
    )
    means = xr.open_dataset(es_dir / "means.nc")
    es_c2 = (means["raw"] - means["off"]).values
    gate["ES_view_off"] = compare(ttcf["off"].values, means["off"].values)
    gate["ES_view_c2"] = compare(ttcf["c2"].values, es_c2)
    chained = ours.train_ids[ours.pair_status == 0]
    no_first = ours.ttcf(Averaging.ES, chained, kbar_min=kbar_min)["c2"].values
    gate["g3_es_view_without_chain_first_trains"] = lag_table(
        XpcsHits(clean_out), no_first, "ES view without each chain's first train"
    )
    gate["g3_identical_rings"] = lag_table(
        XpcsHits(clean_out), es_c2, "extra-speckle xpcs_offline (real)"
    )
    return gate


def lag_table(clean: XpcsHits, es_c2: np.ndarray, es_label: str) -> dict[str, Any]:
    """CLEAN's pooled c2 and the ES c2 diagonal per ring at ``LAGS``."""
    g2 = clean.g2(Averaging.POOLED, binned=False)
    lags = g2["lag"].values.astype(int)
    rows = []
    for ring in range(clean.n_rings):
        for lag in LAGS:
            if lag not in lags:
                continue
            j = int(np.flatnonzero(lags == lag)[0])
            diag = np.diagonal(es_c2[ring], lag)
            es_value = float(np.nanmean(diag)) if np.isfinite(diag).any() else np.nan
            c = float(g2["c2"].values[ring, j])
            s = float(g2["sigma"].values[ring, j])
            rows.append(
                {
                    "ring": ring,
                    "q_A": float(clean.roi_q[ring]),
                    "lag": lag,
                    "c2_clean": c,
                    "sigma_clean": s,
                    "c2_es": es_value,
                    "difference": c - es_value,
                    "difference_over_sigma": (c - es_value) / s if s > 0 else np.nan,
                }
            )
    z = np.array([r["difference_over_sigma"] for r in rows], dtype=float)
    return {
        "reference": es_label,
        "rows": rows,
        "n_trains": g2.attrs["n_trains"],
        "n_with_baseline": g2.attrs["n_with_baseline"],
        "rms_z": float(np.sqrt(np.nanmean(z**2))) if np.isfinite(z).any() else None,
        "max_abs_z": float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else None,
    }


# ── G2 ────────────────────────────────────────────────────────────────────────


def q_nm(geom: Any, wavelength_m: float, sdd_m: float, centre: tuple[float, float]):
    """Pixel-centre q [nm⁻¹] as ``build_operator`` constructs it."""
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    detector = geom.to_pyfai_detector()
    detector.set_pixel_corners(geom.to_distortion_array())
    ai = AzimuthalIntegrator(detector=detector, wavelength=wavelength_m)
    ai.setFit2D(sdd_m * 1e3, *centre)
    return np.asarray(ai.array_from_unit(SHAPE, "center", "q_nm^-1")).ravel()


def labels_from_q(q: np.ndarray, edges: np.ndarray, *, open_edges: bool) -> np.ndarray:
    label = np.full(NPIX, -1, dtype=np.int16)
    for r, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        inside = (q > lo) & (q < hi) if open_edges else (q >= lo) & (q < hi)
        label[inside] = r
    return label


def stage_g2(ctx: Ctx) -> dict[str, Any]:
    from pyFAI.units import hc

    from analysis.saxs.masks import build_static_bad, load_pixel_mask
    from analysis.saxs.operator import geometry_from_config
    from analysis.xpcs.rings import pixel_map, static_bad_for

    es_dir = ctx.dir / "es"
    es = json.loads((es_dir / "es_setup.json").read_text())
    saxs = ctx.saxs()
    cfg = ctx.cfg(ctx.dir / "g2")
    geom = geometry_from_config(saxs)
    edges_nm = np.asarray(RING_EDGES_NM)

    ours_label = pixel_map(cfg).label.astype(np.int16).copy()
    static_ours = static_bad_for(cfg)
    ours_final = np.where(static_ours, -1, ours_label)
    es_masks = np.load(es_dir / "all_masks.npy").reshape(-1, NPIX)
    es_label = np.where(es_masks.any(0), es_masks.argmax(0), -1).astype(np.int16)

    es_static = np.load(es_dir / "setup_mask.npy").reshape(NPIX).astype(bool)
    pixel_file = load_pixel_mask(saxs.pixel_mask_file)
    seams = build_static_bad(replace(saxs, pixel_mask_file=None)).bad
    qa = np.load(es_dir / "qA.npy").reshape(NPIX) * 10.0

    lam_es = 12.3984 / es["photon_energy_kev"] * 1e-10
    lam_hc = hc / es["photon_energy_kev"] * 1e-10
    lam_ours = saxs.wavelength_m
    centre_es = (es["px"], es["py"])
    centre_ours = saxs.beam_center
    sdd_es, sdd_ours = es["sdd_mm"] * 1e-3, saxs.sdd_m

    def labels(lam, sdd, centre, *, open_edges, static):
        q = q_nm(geom, lam, sdd, centre)
        return np.where(
            static, -1, labels_from_q(q, edges_nm, open_edges=open_edges)
        ), q

    steps = [
        (
            "start: extra-speckle parameters, rebuilt",
            dict(
                lam=lam_es,
                sdd=sdd_es,
                centre=centre_es,
                open_edges=True,
                static=es_static,
            ),
        ),
        ("wavelength constant (12.3984 vs pyFAI hc)", dict(lam=lam_hc)),
        (
            f"photon energy {es['photon_energy_kev']:.3f} -> "
            f"{saxs.photon_energy_kev} keV",
            dict(lam=lam_ours),
        ),
        (f"sdd {es['sdd_mm']} mm -> {saxs.sdd_m * 1e3} mm", dict(sdd=sdd_ours)),
        (f"beam centre {centre_es} -> {centre_ours}", dict(centre=centre_ours)),
        ("ring edges: open -> half-open", dict(open_edges=False)),
        ("pixel mask file: extra-speckle's -> ours", dict(static=pixel_file)),
        ("ASIC seams added to the static mask", dict(static=pixel_file | seams)),
    ]
    params: dict[str, Any] = {}
    previous = None
    sequence = []
    changed_at: list[np.ndarray] = []
    for name, change in steps:
        params |= change
        label, q = labels(**params)
        if previous is None:
            sequence.append(
                {
                    "step": name,
                    "differs_from_extra_speckle_rings": int((label != es_label).sum()),
                    "max_abs_dq_vs_es_qA_nm": float(np.nanmax(np.abs(q - qa))),
                    "static_pixels": int(params["static"].sum()),
                }
            )
        else:
            moved = label != previous
            changed_at.append(moved)
            sequence.append(
                {
                    "step": name,
                    "pixels_changing_ring": int(moved.sum()),
                    "entering_a_ring": int((moved & (previous < 0)).sum()),
                    "leaving_all_rings": int((moved & (label < 0)).sum()),
                    "moving_between_rings": int(
                        (moved & (previous >= 0) & (label >= 0)).sum()
                    ),
                }
            )
        previous = label
    reconstruction_residual = int((previous != ours_final).sum())

    start_label = labels(lam_es, sdd_es, centre_es, open_edges=True, static=es_static)[
        0
    ]
    diff = es_label != ours_final
    first_cause: dict[str, int] = {}
    n_steps = [s["step"] for s in sequence[1:]]
    stack = np.stack(changed_at)  # (steps, NPIX)
    for pixel in np.flatnonzero(diff):
        causes = [n_steps[i] for i in np.flatnonzero(stack[:, pixel])]
        key = (
            causes[0]
            if len(causes) == 1
            else ("none" if not causes else "several: " + " + ".join(causes))
        )
        first_cause[key] = first_cause.get(key, 0) + 1

    # each cause alone, from the start
    alone = {}
    start_params = dict(steps[0][1])
    for name, change in steps[1:]:
        label, _ = labels(**{**start_params, **change})
        alone[name] = int((label != start_label).sum())

    out = {
        "pixels_in_extra_speckle_rings": int((es_label >= 0).sum()),
        "pixels_in_our_rings": int((ours_final >= 0).sum()),
        "pixels_that_differ": int(diff.sum()),
        "start_reconstruction_residual_vs_es": int((start_label != es_label).sum()),
        "end_reconstruction_residual_vs_ours": reconstruction_residual,
        "sequence": sequence,
        "attribution_of_differing_pixels": first_cause,
        "each_cause_alone_from_start": alone,
        "unexplained_pixels": int(first_cause.get("none", 0))
        + int((start_label != es_label).sum())
        + reconstruction_residual,
        "es_static_vs_our_pixel_mask_file": int((es_static != pixel_file).sum()),
        "energy": {
            "extra_speckle_kev": es["photon_energy_kev"],
            "ours_kev": saxs.photon_energy_kev,
        },
        "operator_geometry_source": "geom_latest.geom both sides",
    }
    # DAMNIT's own call passes geom=None: extra-speckle then builds from encoders
    encoder = es_dir / "all_masks_encoder.npy"
    if encoder.exists():
        enc = np.load(encoder).reshape(-1, NPIX)
        enc_label = np.where(enc.any(0), enc.argmax(0), -1).astype(np.int16)
        out["damnit_geometry_geom_none"] = {
            "info": es.get("encoder_setup"),
            "pixels_in_rings": int((enc_label >= 0).sum()),
            "differs_from_explicit_geom_rings": int((enc_label != es_label).sum()),
            "differs_from_ours": int((enc_label != ours_final).sum()),
            "max_abs_dq_nm_vs_explicit_geom": float(
                np.nanmax(
                    np.abs(np.load(es_dir / "qA_encoder.npy").reshape(NPIX) * 10 - qa)
                )
            ),
        }
    else:
        out["damnit_geometry_geom_none"] = {
            "unavailable": es.get("encoder_setup_error")
        }
    return out


# ── G3, G6: B4 ────────────────────────────────────────────────────────────────


def stage_g3(ctx: Ctx) -> dict[str, Any]:
    """CLEAN against the ES view of the same store (own geometry), lags 1, 10, 100."""
    root = ctx.dir / "main"
    cfg = ctx.cfg(root)
    clean = XpcsHits(cfg.output_file)
    es_view = XpcsHits(root / "xpcs_hits" / f"r{ctx.run:04d}" / "xpcs_hits_es.h5")
    c2 = es_view.ttcf(Averaging.ES, kbar_min=ES_KBAR_MIN)["c2"].values
    chained = es_view.train_ids[es_view.pair_status == 0]
    no_first = es_view.ttcf(Averaging.ES, chained, kbar_min=ES_KBAR_MIN)["c2"].values
    return {
        "without_chain_first_trains": lag_table(
            clean, no_first, "ES view without each chain's first train"
        ),
        "es_reference": "Kernel.ES + Averaging.ES on the same store (G1 validates it)",
        **lag_table(clean, c2, "Kernel.ES + Averaging.ES, same store"),
        "coverage_clean": str(clean.coverage()),
        "coverage_es": str(es_view.coverage()),
    }


_B4: dict[str, Any] = {}


def _b4_init(store_path: str, label: np.ndarray, n_rings: int, lags: np.ndarray):
    from analysis.xpcs.rings import Rings

    store = open_store(Path(store_path))
    label.flags.writeable = False
    _B4.update(
        store=store,
        rings=Rings((), label, n_rings, ""),
        lags=lags,
    )


def _b4_pair(job: tuple[int, int]) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Baseline numerator and denominator per ring and lag for ``t`` against ``t-g``."""
    t, g = job
    store, rings, lags = _B4["store"], _B4["rings"], _B4["lags"]
    sums = train_sums(store.get(t), store.get(t - g), rings, Kernel.CLEAN)
    off = sums.off.astype(np.float64)
    s_cur, s_prev = (
        sums.s_off_cur.astype(np.float64),
        sums.s_off_prev.astype(np.float64),
    )
    cnt = sums.cnt_pair.astype(np.float64)
    den = np.zeros_like(off)
    np.divide(
        s_cur[..., :, None] * s_prev[..., None, :],
        cnt[..., None, None],
        out=den,
        where=(cnt > 0)[..., None, None],
    )

    def diagonals(m: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.diagonal(m, k, -2, -1).sum(-1) + np.diagonal(m, -k, -2, -1).sum(-1)
                for k in lags
            ],
            -1,
        )

    return t, g, diagonals(off), diagonals(den)


def jackknife_ratio(num: np.ndarray, den: np.ndarray, blocks: list[np.ndarray]):
    """Pooled ratio over trains and leave-one-block-out replicates."""
    value = num.sum(0) / den.sum(0)
    loo = np.stack(
        [(num.sum(0) - num[b].sum(0)) / (den.sum(0) - den[b].sum(0)) for b in blocks]
    )
    return value, loo


def stage_b4(ctx: Ctx, store_path: Path, n_max: int = 1200) -> dict[str, Any]:
    from analysis.xpcs.rings import (
        build_envelope,
        build_rings,
        pixel_map,
        static_bad_for,
    )

    cfg = ctx.cfg(ctx.dir / "main")
    pixels = pixel_map(cfg)
    envelope = build_envelope(pixels, cfg)
    rings = build_rings(pixels, envelope, static_bad_for(cfg), cfg)
    with open_store(store_path) as store:
        ok = {int(t) for t in ctx.ids if store.status(np.array([t]))[0] == 0}
    ids = sorted(ok)[:n_max]
    targets = [t for t in ids if all(t - g in ok for g in GAPS)]
    if len(targets) < 20:
        return {"skipped": f"only {len(targets)} trains have all partners {GAPS}"}
    p = int(XpcsHits(ctx.cfg(ctx.dir / "main").output_file).n_frames)
    lags = np.arange(1, p - 10)
    jobs = [(t, g) for t in targets for g in GAPS]
    started = time.perf_counter()
    with mp.get_context("fork").Pool(
        ctx.workers,
        initializer=_b4_init,
        initargs=(str(store_path), rings.label.copy(), rings.n_rings, lags),
    ) as pool:
        results = pool.map(_b4_pair, jobs, chunksize=4)
    wall = time.perf_counter() - started
    index = {t: i for i, t in enumerate(targets)}
    num = {g: np.zeros((len(targets), rings.n_rings, lags.size)) for g in GAPS}
    den = {g: np.zeros_like(num[g]) for g in GAPS}
    for t, g, n_, d_ in results:
        num[g][index[t]], den[g][index[t]] = n_, d_
    blocks = np.array_split(np.arange(len(targets)), min(50, len(targets)))
    base, loo = {}, {}
    for g in GAPS:
        base[g], loo[g] = jackknife_ratio(num[g], den[g], blocks)
    k = len(blocks)
    out: dict[str, Any] = {
        "n_targets": len(targets),
        "n_blocks": k,
        "kernel_core_s_per_pair": wall * ctx.workers / len(jobs),
        "gaps": {},
    }
    window = slice(0, min(60, lags.size))
    for g in GAPS:
        sigma = np.sqrt((k - 1) / k * ((loo[g] - loo[g].mean(0)) ** 2).sum(0))
        entry: dict[str, Any] = {
            "baseline_lag1_by_ring": base[g][:, 0],
            "sigma_lag1_by_ring": sigma[:, 0],
        }
        if g != 1:
            delta_loo = loo[g] - loo[1]
            sigma_d = np.sqrt(
                (k - 1) / k * ((delta_loo - delta_loo.mean(0)) ** 2).sum(0)
            )
            z = (base[g] - base[1]) / sigma_d
            zw = z[:, window]
            entry |= {
                "rms_z_lags_1_60": float(np.sqrt(np.nanmean(zw**2))),
                "rms_z_by_ring": np.sqrt(np.nanmean(zw**2, axis=1)),
                "fraction_abs_z_gt_3": float(np.nanmean(np.abs(zw) > 3)),
                "mean_delta_lags_1_60": float(
                    np.nanmean(base[g][:, window] - base[1][:, window])
                ),
            }
        out["gaps"][str(g)] = entry
    passing = [
        g
        for g in GAPS
        if g == 1
        or (
            out["gaps"][str(g)]["rms_z_lags_1_60"] <= 1.0
            and max(out["gaps"][str(g)]["rms_z_by_ring"]) <= 1.5
        )
    ]
    out["criterion"] = (
        "largest gap with rms z <= 1.0 over (ring, lags 1..60) and every ring <= 1.5"
    )
    out["largest_gap_agreeing_with_gap_1"] = max(passing)
    return out


# ── G4, G5, G6 ────────────────────────────────────────────────────────────────


def stage_g4(ctx: Ctx) -> dict[str, Any]:
    """The same hits under other worker counts and block sizes."""
    ids = ctx.ids[:96]
    fresh(ctx.dir / "g4")
    hits = ctx.hit_set(ids)
    variants = {
        "w36_b8": dict(n_workers=ctx.workers, trains_per_block=8),
        "w8_b8": dict(n_workers=8, trains_per_block=8),
        "w36_b1": dict(n_workers=ctx.workers, trains_per_block=1),
        "w8_b3": dict(n_workers=8, trains_per_block=3),
    }
    out: dict[str, Any] = {"n_hits": int(ids.size), "variants": {}}
    digests = {}
    for name, change in variants.items():
        cfg = ctx.cfg(ctx.dir / "g4" / name, **change)
        done = run_pass(cfg, hits)
        digests[name] = sums_digests(done["results"], done["store"])
        out["variants"][name] = {"wall_s": done["wall_s"], **digests[name]}
    reference = digests["w36_b8"]
    out["bit_identical"] = {
        name: digests[name] == reference for name in variants if name != "w36_b8"
    }
    out["pass"] = all(out["bit_identical"].values())
    return out


def stage_g5(ctx: Ctx) -> dict[str, Any]:
    """A rerun of a complete run touches nothing."""
    cfg = ctx.cfg(ctx.dir / "main")
    paths = {"store": cfg.store_file, "results": cfg.output_file}
    before = {k: stat(p) for k, p in paths.items()}
    started = time.perf_counter()
    run_xpcs_hits(cfg, ctx.hit_set())
    wall = time.perf_counter() - started
    after = {k: stat(p) for k, p in paths.items()}
    unchanged = {k: before[k] == after[k] for k in paths}
    return {
        "rerun_wall_s": wall,
        "unchanged": unchanged,
        "pass": bool(all(unchanged.values())),
        "before": before,
        "after": after,
    }


def stage_g6(ctx: Ctx, main: dict[str, Any]) -> dict[str, Any]:
    cfg = ctx.cfg(ctx.dir / "main")
    n = main["n_hits"]
    reader = XpcsHits(cfg.output_file)
    with h5py.File(cfg.store_file, "r") as f:
        fixed = f["envelope/pixels"].id.get_storage_size() + sum(
            d.id.get_storage_size() for d in f["base"].values()
        )
    out: dict[str, Any] = {
        "n_hits": n,
        "extract_s_per_train_at_workers": main["extract_wall_s_per_train"],
        "workers": main["workers"],
        "kernel_core_ms_per_train": 1e3
        * float(main["merge_core_s"].get("train_sums", np.nan))
        / max(int(reader.usable.sum()), 1),
        "store_bytes_per_train_incl_base": main["store_bytes"] / n,
        "store_fixed_bytes_estimate": fixed,
        "store_bytes_per_train_excl_base": (main["store_bytes"] - fixed) / n,
        "results_bytes_per_train": main["results_bytes"] / n,
        "projection_gb_per_10000_hits": 1e4
        * (main["store_bytes"] / n + main["results_bytes"] / n)
        / 1e9,
    }
    started = time.perf_counter()
    g2 = reader.g2(Averaging.POOLED)
    out["g2_pooled_binned_s"] = time.perf_counter() - started
    started = time.perf_counter()
    reader.g2(Averaging.POOLED, binned=False)
    out["g2_pooled_unbinned_s"] = time.perf_counter() - started
    started = time.perf_counter()
    reader.g2(Averaging.ES)
    out["g2_es_s"] = time.perf_counter() - started
    started = time.perf_counter()
    reader.ttcf(Averaging.POOLED)
    out["ttcf_pooled_s"] = time.perf_counter() - started
    out["g2_n_trains"] = int(g2.attrs["n_trains"])
    out["b4"] = stage_b4(ctx, cfg.store_file)
    return out


# ── E3 ────────────────────────────────────────────────────────────────────────


def stage_e3(ctx: Ctx) -> dict[str, Any]:
    from analysis.triage.xpcs import jackknife_g2, read_ttcf_diagonals

    path = Path(str(DAMNIT_FILE).format(run=ctx.run))
    started = time.perf_counter()
    damnit = read_ttcf_diagonals(path)
    read_s = time.perf_counter() - started
    if damnit is None:
        return {"skipped": f"{path} has no xpcs_ttcf"}
    cfg = ctx.cfg(ctx.dir / "main")
    ours = XpcsHits(cfg.output_file)
    shared = np.intersect1d(damnit.train_ids, ours.train_ids[ours.usable])
    sel = np.searchsorted(damnit.train_ids, shared)
    g_d, s_d, _ = jackknife_g2(damnit.sums[sel], damnit.counts[sel])
    k_max = ours.n_frames - 1 - 10
    diag = ours.diagonals(kbar_min=ES_KBAR_MIN, trains=shared)
    g_o, s_o, _ = jackknife_g2(diag.sums, diag.counts)
    es_diag = None
    es_path = cfg.output_file.with_name("xpcs_hits_es.h5")
    if es_path.exists():
        es_view = XpcsHits(es_path)
        es_d = es_view.diagonals(kbar_min=ES_KBAR_MIN, trains=shared)
        g_e, _, _ = jackknife_g2(es_d.sums, es_d.counts)
        es_diag = g_e
    order = [
        int(np.argmin(np.abs(ours.roi_q - q))) for q in damnit.roi_q_A
    ]  # DAMNIT ring -> ours
    rows = []
    for j, r in enumerate(order):
        lags = [k for k in (1, 2, 5, 10, 50, 100) if k <= k_max]
        z = (g_o[r, :k_max] - g_d[j, :k_max]) / np.hypot(s_o[r, :k_max], s_d[j, :k_max])
        entry = {
            "damnit_roi_q_A": float(damnit.roi_q_A[j]),
            "ours_roi_q_A": float(ours.roi_q[r]),
            "lag": lags,
            "g2_damnit": [float(g_d[j, k - 1]) for k in lags],
            "g2_ours_clean": [float(g_o[r, k - 1]) for k in lags],
            "sigma_damnit": [float(s_d[j, k - 1]) for k in lags],
            "sigma_ours": [float(s_o[r, k - 1]) for k in lags],
            "max_abs_z_lags_1_10": float(np.nanmax(np.abs(z[:10]))),
            "rms_z_all_lags": float(np.sqrt(np.nanmean(z**2))),
        }
        if es_diag is not None:
            entry["g2_ours_es_kernel"] = [float(es_diag[r, k - 1]) for k in lags]
        rows.append(entry)
    return {
        "damnit_file": str(path),
        "read_ttcf_diagonals_s": read_s,
        "n_trains_shared": int(shared.size),
        "n_ours_usable": int(ours.usable.sum()),
        "n_damnit": int(damnit.train_ids.size),
        "note": (
            "DAMNIT's TTCF is extra-speckle on all trains (previous train as "
            "partner, outlier mask on, 9.000 keV, encoder geometry); ours is the "
            "CLEAN kernel on the geometry rings: differences are expected where "
            "G2 names a cause"
        ),
        "rings": rows,
    }


# ── A1 ────────────────────────────────────────────────────────────────────────


def stage_a1(ctx: Ctx) -> dict[str, Any]:
    from analysis.saxs.masks import build_static_bad
    from analysis.xpcs.rings import build_envelope, pixel_map

    stats_path = PHOTON_STATS / f"r{ctx.run:04d}" / "agipd_photon_stats.h5"
    if not stats_path.exists():
        return {"skipped": f"{stats_path} does not exist"}
    cfg = ctx.cfg(ctx.dir / "main")
    envelope = build_envelope(pixel_map(cfg), cfg).pixels
    static = build_static_bad(ctx.saxs()).bad
    with h5py.File(stats_path, "r") as f:
        labels = f["rings/labels"][()].astype(np.int16)
        edges = f["rings/edges_nm"][()]
        train_id = f["frames/trainId"][()]
        status = f["frames/status"][()]
        cell = f["frames/cellId"][()]
        s1, n_valid = f["frames/s1"], f["frames/n_valid"]
        rows_of = {}
        for t in ctx.ids[:100]:
            rows_of[int(t)] = np.flatnonzero(train_id == t)
        n_rings = labels.max() + 1
        outside = int(((labels >= 0) & ~np.isin(np.arange(NPIX), envelope)).sum())
        env_label = labels[envelope]
        env_static = static[envelope]
        in_ring = env_label >= 0
        mism_s1 = mism_nv = frames = skipped = 0
        max_ds1 = max_dnv = 0
        cell_mismatch = 0
        k_ratio_max = 0.0
        with open_store(cfg.store_file) as store:
            for t, rows in rows_of.items():
                photons = store.get(t)
                if photons.n_frames != rows.size or rows.size == 0:
                    skipped += 1
                    continue
                if not np.array_equal(photons.cell_ids, cell[rows]):
                    cell_mismatch += 1
                    continue
                file_s1, file_nv = s1[rows], n_valid[rows]
                for frame in range(photons.n_frames):
                    if status[rows[frame]] != 0:
                        continue
                    sl = slice(photons.frame_ptr[frame], photons.frame_ptr[frame + 1])
                    px, cnt = photons.pixel[sl], photons.count[sl].astype(np.int64)
                    flagged = np.zeros(envelope.size, dtype=bool)
                    flagged[photons.flagged_frame(frame)] = True
                    bad = flagged | env_static
                    keep = in_ring[px] & ~bad[px]
                    mine_s1 = np.bincount(
                        env_label[px[keep]], weights=cnt[keep], minlength=n_rings
                    )
                    mine_nv = np.bincount(env_label[in_ring & ~bad], minlength=n_rings)
                    d1 = np.abs(mine_s1 - file_s1[frame])
                    dn = np.abs(mine_nv - file_nv[frame])
                    frames += 1
                    mism_s1 += int((d1 > 0).any())
                    mism_nv += int((dn > 0).any())
                    max_ds1, max_dnv = max(max_ds1, d1.max()), max(max_dnv, dn.max())
                    with np.errstate(invalid="ignore", divide="ignore"):
                        k_mine = mine_s1 / mine_nv
                        k_file = file_s1[frame] / file_nv[frame]
                        k_ratio_max = max(
                            k_ratio_max, float(np.nanmax(np.abs(k_mine / k_file - 1)))
                        )
    return {
        "stats_file": str(stats_path),
        "n_photon_stat_rings": int(n_rings),
        "ring_edges_nm_first_last": [edges[0].tolist(), edges[-1].tolist()],
        "ring_pixels_outside_the_store_envelope": outside,
        "frames_compared": frames,
        "trains_skipped_frame_count": skipped,
        "trains_skipped_cell_ids_differ": cell_mismatch,
        "frames_with_any_s1_difference": mism_s1,
        "frames_with_any_n_valid_difference": mism_nv,
        "max_abs_s1_difference": float(max_ds1),
        "max_abs_n_valid_difference": float(max_dnv),
        "max_relative_kbar_difference": k_ratio_max,
        "pass": bool(frames > 0 and mism_s1 == 0 and mism_nv == 0),
        "rings": "the file's own labels, restricted to the store's envelope; static "
        "mask rebuilt from the same config (seams + pixel mask)",
    }


# ── driver ────────────────────────────────────────────────────────────────────


def run_one(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ctx = Ctx(args, args.run)
    stages = args.stages or DEFAULT_STAGES.get(ctx.run, REFERENCE_STAGES)
    report: dict[str, Any] = report_header(
        proposal=args.proposal, run=ctx.run, workers=ctx.workers, stages=stages
    )
    failed = False

    def attempt(name: str, call) -> Any:
        nonlocal failed
        started = time.perf_counter()
        try:
            value = call()
            report[name] = value
            report[f"{name}_seconds"] = time.perf_counter() - started
            return value
        except Exception:  # noqa: BLE001 - recorded in the report, never hidden
            failed = True
            report[name] = {"error": traceback.format_exc()}
            log.exception("stage %s failed", name)
            return None

    if "hits" in stages:
        attempt("hits", lambda: stage_hits(ctx))
    elif (ctx.dir / "hit_ids.npy").exists():
        ctx.ids = np.load(ctx.dir / "hit_ids.npy")
    if ctx.ids is None:
        report["error"] = "no hit ids"
        write_report(
            jsonable(report), args.json, f"xpcs_hits_acceptance_r{ctx.run:04d}"
        )
        return 1
    main = None
    if "main" in stages:
        main = attempt("main", lambda: stage_main(ctx))
    elif (ctx.dir / "main.json").exists():
        main = json.loads((ctx.dir / "main.json").read_text())
    for name, call in (
        ("g1", lambda: stage_g1(ctx)),
        ("g2", lambda: stage_g2(ctx)),
        ("g3", lambda: stage_g3(ctx)),
        ("g4", lambda: stage_g4(ctx)),
        ("g5", lambda: stage_g5(ctx)),
        ("g6", lambda: stage_g6(ctx, main)),
        ("e3", lambda: stage_e3(ctx)),
        ("a1", lambda: stage_a1(ctx)),
    ):
        if name not in stages:
            continue
        if name in ("g3", "g5", "g6", "e3", "a1") and not main:
            report[name] = {"skipped": "main stage did not complete"}
            continue
        attempt(name, call)
    write_report(jsonable(report), args.json, f"xpcs_hits_acceptance_r{ctx.run:04d}")
    return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--run", type=int, nargs="+", required=True)
    add_job_args(parser, workers=36, time="03:00:00")
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if len(args.run) > 1 and args.json:
        parser.error("--json names one run's file")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "es-child":
        return es_child(int(argv[1]), argv[2], argv[3], int(argv[4]))
    args = parse_args(argv)
    return dispatch(
        args,
        args.run,
        lambda run: run_one(argparse.Namespace(**vars(args) | {"run": run})),
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
