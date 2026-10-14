#!/usr/bin/env python
"""Acceptance of the photon statistics written by the ``agipd_saxs`` pass itself.

One run, four passes into a scratch directory (never the real outputs):

1. ``off``: statistics off, cold page cache: frame table and window sums.
2. ``fused``: statistics on: all three files from one read.
3. ``off_warm``: statistics off again, warm, the fair comparison for 2.
4. ``backfill``: statistics on, into 3's directory, where the frame table and
   the sums are complete: only the statistics are written, and the frame table,
   the sums, ``operator.npz`` and ``masks.npz`` must keep inode, mtime and size.

Then: the fused and backfilled statistics equal each other and the standalone
file of Q1 (``usr/cached_files/agipd_photon_stats/r{run:04d}``) column by column,
and the fused frame table equals the warm one dataset by dataset.

    python scripts/photon_stats_fused_acceptance.py --run 423 --workers 32 \\
        --root /gpfs/exfel/exp/MID/202601/p010400/scratch/<somewhere>

Writes photon_stats_fused_acceptance_<stamp>.json beside itself. Not unit-tested
(CLAUDE.md rule 7): run it on a node.
"""

from __future__ import annotations

from _common import report_header, write_report  # noqa: E402

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.saxs.photon_stats import FILE_NAME as PHOTON  # noqa: E402
from analysis.saxs.pixel_sums import FILE_NAME as SUMS  # noqa: E402
from analysis.saxs.run import run_agipd_saxs  # noqa: E402

FRAME_TABLE = "agipd_saxs.h5"
UNTOUCHED = (FRAME_TABLE, SUMS, "operator.npz", "masks.npz")
GATES = (
    "backfill_left_the_other_files_untouched",
    "fused_equals_backfill",
    "fused_equals_q1_standalone",
    "fused_frame_table_equals_warm",
    "all_frames_ok",
)


def one_pass(run: int, workers: int, directory: Path, photon_stats: bool) -> dict:
    cfg = AgipdSaxsConfig(
        proposal=10400, run=run, n_workers=workers, photon_stats=photon_stats
    )
    started = time.perf_counter()
    run_agipd_saxs(cfg, output_path=directory / FRAME_TABLE, reduce="none")
    out: dict = {"wall_s": round(time.perf_counter() - started, 1)}
    for name in (FRAME_TABLE, SUMS, PHOTON):
        path = directory / name
        if path.exists():
            with h5py.File(path, "r") as f:
                prov = f["provenance"].attrs
                out[name] = {
                    "bytes": path.stat().st_size,
                    "wall_s": round(float(prov["wall_s"]), 1),
                    "timings": json.loads(prov["timings"]),
                    "setup_timings": json.loads(prov["setup_timings"]),
                }
    return out


def stat(path: Path) -> list | None:
    if not path.exists():
        return None
    s = path.stat()
    return [s.st_ino, s.st_mtime_ns, s.st_size]


def dataset_digests(path: Path, group: str) -> dict[str, str]:
    with h5py.File(path, "r") as f:
        return {
            name: hashlib.sha256(np.ascontiguousarray(f[group][name][()])).hexdigest()
            for name in sorted(f[group])
            if isinstance(f[group][name], h5py.Dataset)
        }


def statistics(path: Path) -> dict[str, str]:
    digests = dataset_digests(path, "frames")
    digests |= {f"rings/{k}": v for k, v in dataset_digests(path, "rings").items()}
    return digests


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, default=423)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root / f"r{args.run:04d}"
    passes: dict = {}
    checks: dict = {}

    passes["off"] = one_pass(args.run, args.workers, root / "off", False)
    passes["fused"] = one_pass(args.run, args.workers, root / "fused", True)
    passes["off_warm"] = one_pass(args.run, args.workers, root / "off_warm", False)
    before = {n: stat(root / "off_warm" / n) for n in UNTOUCHED}
    passes["backfill"] = one_pass(args.run, args.workers, root / "off_warm", True)
    after = {n: stat(root / "off_warm" / n) for n in UNTOUCHED}
    checks["backfill_left_the_other_files_untouched"] = before == after

    fused = statistics(root / "fused" / PHOTON)
    backfill = statistics(root / "off_warm" / PHOTON)
    checks["fused_equals_backfill"] = fused == backfill
    q1 = AgipdSaxsConfig(proposal=10400, run=args.run).photon_stats_file
    if q1.exists():
        standalone = statistics(q1)
        checks["fused_equals_q1_standalone"] = fused == standalone
        checks["differing_with_q1"] = sorted(
            k for k in fused if fused[k] != standalone.get(k)
        )
    checks["fused_frame_table_equals_warm"] = dataset_digests(
        root / "fused" / FRAME_TABLE, "frames"
    ) == dataset_digests(root / "off_warm" / FRAME_TABLE, "frames")
    with h5py.File(root / "fused" / PHOTON, "r") as f:
        status = f["frames/status"][:]
    checks["all_frames_ok"] = bool((status == 0).all())
    n_frames = int(status.size)
    for key in ("photon_stats", "integrate"):
        seconds = passes["fused"][FRAME_TABLE]["timings"].get(key, float("nan"))
        checks[f"fused_{key}_ms_per_frame_core"] = round(1e3 * seconds / n_frames, 3)
    seconds = passes["backfill"][PHOTON]["timings"].get("photon_stats", float("nan"))
    checks["backfill_photon_stats_ms_per_frame_core"] = round(
        1e3 * seconds / n_frames, 3
    )
    checks["fused_over_off_warm_wall"] = round(
        passes["fused"]["wall_s"] / passes["off_warm"]["wall_s"], 3
    )
    passed = all(checks.get(name) for name in GATES)
    report = report_header(
        script="photon_stats_fused_acceptance", run=args.run, workers=args.workers,
        root=str(root), n_frames=n_frames, passes=passes, checks=checks, passed=passed,
    )  # fmt: skip
    write_report(report, args.json, "photon_stats_fused_acceptance")
    for name, p in passes.items():
        print(f"{name:9s} wall {p['wall_s']:7.1f} s")
    for name, value in checks.items():
        print(f"{name}: {value}")
    print("passed:", passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
