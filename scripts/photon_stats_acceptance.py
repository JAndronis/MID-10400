#!/usr/bin/env python
"""Q1 acceptance for the photon-statistics pass (context/xpcs-quality.md §8).

For each run: run `analysis.saxs.photon_stats.run_photon_stats` (writes
usr/cached_files/agipd_photon_stats/r{run:04d}/), then check:
* every frame of the plan is OK, and the wall time and file size per run
* on the probe's runs (r0423, r0450): the 20 trains the probe read give its ring
  counts exactly (xpcs_quality_probe_20261013T165909Z.json)
* the single-frame contrast per XPCS ROI, weighted and not, with the shape checks;
  for a mesh run over its droplet hits only (`triage.classify.hits`)

    python scripts/photon_stats_acceptance.py --runs 423 450 318 --workers 32

Writes photon_stats_acceptance_<stamp>.json beside itself. Not unit-tested
(CLAUDE.md rule 7): run it on a node.
"""

from __future__ import annotations

# Before numpy or h5py import anything that builds a thread pool.
from _common import report_header, write_report  # noqa: E402

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from analysis.common.plan import evenly_spaced  # noqa: E402
from analysis.common.writer import status_counts  # noqa: E402
from analysis.diagnostics.photon_stats import (  # noqa: E402
    contrast,
    nb_checks,
    read_photon_stats,
    roi_groups,
    run_weights,
)
from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.saxs.photon_stats import PhotonStatsConfig, run_photon_stats  # noqa: E402

PROBE = Path(__file__).with_name("xpcs_quality_probe_20261013T165909Z.json")
PROBE_TRAINS = 20


def probe_counts(run: int) -> list | None:
    rows = {r["run"]: r for r in json.loads(PROBE.read_text())["runs"]}
    return None if run not in rows else [row["counts_k"] for row in rows[run]["narrow"]]


def _floats(da) -> list[float]:
    return [round(float(v), 6) for v in np.asarray(da)]


def one_run(run: int, workers: int) -> dict:
    import h5py
    from extra_data import open_run

    saxs = AgipdSaxsConfig(proposal=10400, run=run, n_workers=workers)
    cfg = PhotonStatsConfig(saxs=saxs)
    t0 = time.perf_counter()
    path = run_photon_stats(cfg)
    wall = time.perf_counter() - t0
    out: dict = {"run": run, "file": str(path), "wall_s": round(wall, 1),
                 "bytes": path.stat().st_size}  # fmt: skip
    with h5py.File(path, "r") as f:
        status = f["frames/status"][:]
        out["n_frames"] = int(status.size)
        out["status"] = status_counts(status, nonzero_only=True)
        out["bytes_per_frame"] = round(out["bytes"] / max(status.size, 1), 1)
        prov = f["provenance"].attrs
        out["timings"] = json.loads(prov["timings"])
        out["frames_per_s"] = round(status.size / wall, 1)

    # the probe's 20 trains, ring by ring, exactly
    expected = probe_counts(run)
    if expected is not None:
        trains = evenly_spaced(np.asarray(open_run(10400, run, data="proc").train_ids),
                               PROBE_TRAINS)  # fmt: skip
        ds = read_photon_stats(path, trains=trains)
        pooled = ds.hist.sum("frame").values
        same = [
            bool(
                np.array_equal(pooled[r, : len(c)], c)
                and pooled[r, len(c) :].sum() == 0
            )
            for r, c in enumerate(expected)
        ]
        out["probe_rings_reproduced"] = same

    # the contrast, over hits for a mesh run
    from analysis.triage.classify import hits

    h = hits(run)
    trains = h["train_ids"] if h["kind"] == "mesh" else None
    ds = read_photon_stats(path, trains=trains)
    weights = run_weights(
        path, pixel_sums_path=saxs.pixel_sums_file,
        masks_path=saxs.output_file.with_name("masks.npz"),
        trains=trains if h["kind"] == "mesh" else None,
    )  # fmt: skip
    groups = roi_groups(ds)
    weighted = contrast(ds, weights, groups=groups)
    plain = contrast(ds, None, groups=groups)
    rings = contrast(ds, weights)
    shape = nb_checks(ds, rings.beta.values)
    out["hits"] = {"kind": h["kind"], "n_hits": int(h["train_ids"].size),
                   "n_frames_used": int(ds.sizes["frame"]),
                   "weights_windows": int(weights.attrs["n_windows"])}  # fmt: skip
    out["roi_q_A"] = list(groups)
    out["beta_weighted"] = _floats(weighted.beta)
    out["sigma_weighted"] = _floats(weighted.sigma)
    out["beta_unweighted"] = _floats(plain.beta)
    out["kbar"] = _floats(weighted.kbar)
    out["rings"] = {
        "beta": _floats(rings.beta), "sigma": _floats(rings.sigma),
        "r3_z": _floats(shape.r3_z), "max_k": [int(v) for v in shape.max_k],
    }  # fmt: skip
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, nargs="+", default=[423, 450, 318])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    results, errors = [], {}
    for run in args.runs:
        try:
            results.append(one_run(run, args.workers))
        except Exception as error:  # noqa: BLE001 - recorded per run
            errors[run] = f"{type(error).__name__}: {error}"
    passed = not errors and all(
        r["status"] == {"OK": r["n_frames"]}
        and all(r.get("probe_rings_reproduced", [True]))
        for r in results
    )
    report = report_header(
        script="photon_stats_acceptance", elapsed_s=round(time.perf_counter() - t0, 1),
        workers=args.workers, probe=PROBE.name, runs=results, errors=errors,
        passed=bool(passed),
    )  # fmt: skip
    write_report(report, args.json, "photon_stats_acceptance")
    for r in results:
        print(f"r{r['run']}: {r['n_frames']} frames in {r['wall_s']} s "
              f"({r['frames_per_s']}/s), {r['bytes'] / 1e6:.1f} MB "
              f"({r['bytes_per_frame']} B/frame), status {r['status']}, "
              f"probe {r.get('probe_rings_reproduced')}")  # fmt: skip
        print(f"   beta_w {r['beta_weighted']} ± {r['sigma_weighted']}")
        print(f"   beta_u {r['beta_unweighted']}  kbar {r['kbar']}  hits {r['hits']}")
    print("errors:", errors, "\npassed:", passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
