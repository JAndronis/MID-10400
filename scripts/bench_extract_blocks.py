#!/usr/bin/env python
"""Block-file extraction against the real r0493 store (``context/xpcs-hits.md`` §10 G6).

Two stages, all writes under ``--work-dir`` in scratch; the real store is opened
read-only and never copied back or modified.

``extract``
    The hit-train pass from nothing into a scratch store, on the real store's own
    train ids: ``analysis.xpcs.run.run_xpcs_hits`` end to end, so workers write
    block files and the parent merges them. Reports the extract stage's wall
    (from the store's provenance, against the 208 s serial-``put`` baseline), the
    parent's merge seconds, and then compares every train of the new store with
    the real one: ``TrainPhotons`` fields and stored bytes per dataset.

``scale``
    ``merge_block`` into a copy of the real store (~6 GB, ~3000 groups) against
    ``merge_block`` into an empty store, with the same blocks under new train ids:
    does the per-train merge cost grow with the size of the file?

    python scripts/bench_extract_blocks.py --run 493 -w 36 --constraint 9374F
    python scripts/bench_extract_blocks.py --run 493 --local --stages scale
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402

from _common import (  # noqa: E402
    add_job_args,
    configure_logging,
    dispatch,
    report_header,
    stamp,
    write_report,
)

from analysis.threadenv import set_thread_env  # noqa: E402

set_thread_env()

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common.slurm import PROPOSAL_ROOT  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.xpcs.config import XpcsConfig  # noqa: E402
from analysis.xpcs.hits import HitSet  # noqa: E402
from analysis.xpcs.run import run_xpcs_hits  # noqa: E402
from analysis.xpcs.store import RingPhotonStore, write_block_file  # noqa: E402

log = logging.getLogger("scripts.bench_extract_blocks")

SCRATCH = PROPOSAL_ROOT / "scratch"
CACHED = PROPOSAL_ROOT / "usr/cached_files"
FIELDS = (
    "pulse_ids",
    "cell_ids",
    "frame_ptr",
    "pixel",
    "count",
    "flag_ptr",
    "flag_pixel",
)
#: Added to train ids: a block then merges as new trains beside the originals.
NEW_ID_OFFSET = 10**10


def sizes(store: RingPhotonStore) -> dict[int, dict[str, int]]:
    return {
        int(name[1:]): {k: d.id.get_storage_size() for k, d in group.items()}
        for name, group in store._f["trains"].items()
    }


def compare(real_path: Path, new_path: Path, ids: np.ndarray) -> dict:
    """Every train of the new store against the real one, field by field."""
    bad: list[str] = []
    with RingPhotonStore.open(real_path) as real, RingPhotonStore.open(new_path) as new:
        for tid in ids.tolist():
            if real.status(np.array([tid]))[0] != new.status(np.array([tid]))[0]:
                bad.append(f"{tid}: status")
                continue
            a = real._f["trains"][f"t{tid:012d}"]
            b = new._f["trains"][f"t{tid:012d}"]
            if dict(a.attrs) != dict(b.attrs):
                bad.append(f"{tid}: attrs")
            if set(a) != set(b) or any(
                a[k].id.get_storage_size() != b[k].id.get_storage_size() for k in a
            ):
                bad.append(f"{tid}: stored layout")
            x, y = real.get(tid), new.get(tid)
            for name in FIELDS:
                u, v = getattr(x, name), getattr(y, name)
                if u.dtype != v.dtype or not np.array_equal(u, v):
                    bad.append(f"{tid}: {name}")
        extra = len(set(new.train_ids().tolist()) - set(ids.tolist()))
        same_base = all(
            np.array_equal(getattr(real.base, f), getattr(new.base, f))
            for f in ("cells", "ptr", "pixel")
        ) and np.array_equal(real.envelope(), new.envelope())
    return {
        "trains_checked": int(ids.size),
        "mismatches": bad[:20],
        "n_mismatches": len(bad),
        "extra_trains": extra,
        "base_and_envelope_identical": bool(same_base),
        "identical": not bad and extra == 0 and bool(same_base),
    }


def stage_extract(args: argparse.Namespace, work: Path, real_path: Path) -> dict:
    with RingPhotonStore.open(real_path) as real:
        ids = np.sort(real.train_ids()).astype(np.uint64)[: args.n_hits]
        real_summary = real.status_summary()
        real_hash = real.extract_hash
    cfg = XpcsConfig(
        saxs=AgipdSaxsConfig(proposal=10400, run=args.run),
        output_root=str(work),
        n_workers=args.workers,
    )
    hits = HitSet.of(ids, source=f"train ids of {real_path.name}")
    started = time.perf_counter()
    run_xpcs_hits(cfg, hits)
    wall = time.perf_counter() - started
    with h5py.File(cfg.store_file, "r") as f:
        import json

        history = json.loads(f["provenance"].attrs["history"])[-1]
    with RingPhotonStore.open(cfg.store_file) as new:
        new_hash, new_summary = new.extract_hash, new.status_summary()
    return {
        "n_hits": int(ids.size),
        "workers": args.workers,
        "whole_pass_wall_s": wall,
        "extract_stage_wall_s": history["wall_s"],
        "extract_s_per_train": history["wall_s"] / ids.size,
        "worker_core_s": history["timings"],
        "setup_s": history["setup_timings"],
        "parent_merge_s": history["setup_timings"].get("write_blocks"),
        "same_extract_hash_as_real_store": real_hash == new_hash,
        "real_status": real_summary,
        "new_status": new_summary,
        "store_bytes": cfg.store_file.stat().st_size,
        "real_store_bytes": real_path.stat().st_size,
        "comparison": compare(real_path, cfg.store_file, ids),
        "host": history.get("host"),
    }


def stage_scale(args: argparse.Namespace, work: Path, real_path: Path) -> dict:
    with RingPhotonStore.open(real_path) as real:
        ok = np.sort(real.train_ids())
        ok = ok[real.status(ok) == int(FrameStatus.OK)]
        pick = [
            int(t) for t in ok[np.linspace(0, ok.size - 1, args.n_scale).astype(int)]
        ]
        trains = [real.get(t) for t in pick]
        extract_hash, envelope, base = real.extract_hash, real.envelope(), real.base
    blocks = work / "scale_blocks"
    files = []
    k = args.block
    for i in range(0, len(trains), k):
        moved = [
            replace(t, train_id=t.train_id + NEW_ID_OFFSET) for t in trains[i : i + k]
        ]
        file = blocks / f"b{i // k:04d}.h5"
        write_block_file(file, extract_hash, moved)
        files.append(file)

    def merge_all(store_path: Path, label: str, copies: list[Path]) -> dict:
        wall = cpu = 0.0
        with RingPhotonStore.open(store_path, extract_hash, writable=True) as store:
            n_before = len(store._f["trains"])
            for src in copies:
                w0, c0 = time.perf_counter(), time.process_time()
                store.merge_block(src)
                wall += time.perf_counter() - w0
                cpu += time.process_time() - c0
            n_after = len(store._f["trains"])
        n = n_after - n_before
        return {
            "label": label,
            "groups_before": n_before,
            "merged": n,
            "merge_ms_per_train": 1e3 * wall / n,
            "merge_cpu_ms_per_train": 1e3 * cpu / n,
            "file_bytes_after": store_path.stat().st_size,
        }

    out: dict = {"n_scale_trains": len(trains), "block": k}
    runs = []
    for label in ("empty", "real_copy"):
        copies = []
        for f in (
            files
        ):  # merge_block deletes its file: one copy of the block set per target
            c = blocks / label / f.name
            c.parent.mkdir(exist_ok=True)
            shutil.copy(f, c)
            copies.append(c)
        target = work / f"scale_{label}.h5"
        if label == "empty":
            RingPhotonStore.create(target, extract_hash, envelope, base).close()
        else:
            started = time.perf_counter()
            shutil.copy(real_path, target)
            target.chmod(0o640)
            out["copy_real_store_s"] = time.perf_counter() - started
        runs.append(merge_all(target, label, copies))
        # same content merged twice must be a no-op at this size too
        again = [blocks / label / f.name for f in files]
        for src, f in zip(again, files, strict=True):
            shutil.copy(f, src)
        with RingPhotonStore.open(target, extract_hash, writable=True) as store:
            w0 = time.perf_counter()
            replayed = sum(store.merge_block(src) for src in again)
            out[f"{label}_replay_written"] = replayed
            out[f"{label}_replay_ms_per_train"] = (
                1e3 * (time.perf_counter() - w0) / len(trains)
            )
    out["runs"] = runs
    out["ratio_real_copy_over_empty"] = (
        runs[1]["merge_ms_per_train"] / runs[0]["merge_ms_per_train"]
    )
    with (
        RingPhotonStore.open(work / "scale_real_copy.h5") as merged,
        RingPhotonStore.open(real_path) as original,
    ):
        out["merged_reads_back"] = all(
            np.array_equal(merged.get(t + NEW_ID_OFFSET).pixel, original.get(t).pixel)
            for t in (pick[0], pick[-1])
        )
    return out


def run_one(args: argparse.Namespace) -> int:
    configure_logging("scripts.bench_extract_blocks")
    real_path = (
        CACHED / "agipd_ring_photons" / f"r{args.run:04d}" / "agipd_ring_photons.h5"
    )
    work = args.work_dir / f"r{args.run:04d}_{stamp()}"
    work.mkdir(parents=True)
    assert not str(work).startswith(str(CACHED)), "never write under cached_files"
    report = report_header(run=args.run, workers=args.workers, work=str(work))
    ok = True
    for name, fn in (("extract", stage_extract), ("scale", stage_scale)):
        if name not in args.stages:
            continue
        try:
            report[name] = fn(args, work, real_path)
            if name == "extract":
                ok &= report[name]["comparison"]["identical"]
        except Exception:  # noqa: BLE001 - into the report
            import traceback

            report[name] = {"error": traceback.format_exc()}
            ok = False
            log.exception("stage %s failed", name)
    write_report(report, args.json, f"bench_extract_blocks_r{args.run:04d}")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=int, nargs="+", default=[493])
    parser.add_argument("--stages", nargs="+", default=["extract", "scale"])
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--n-hits", type=int, default=None, help="first N (smoke test)")
    parser.add_argument("--n-scale", type=int, default=240)
    parser.add_argument(
        "--work-dir", type=Path, default=SCRATCH / "bench_extract_blocks"
    )
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    add_job_args(parser, workers=36, time="02:00:00")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return dispatch(
        args,
        args.run,
        lambda run: run_one(argparse.Namespace(**vars(args) | {"run": run})),
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
