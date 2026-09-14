#!/usr/bin/env python
"""W4 on-node acceptance for the JUNGFRAU WAXS pass (context file §7, W4).

Runs the pass on one detector of one run, then puts five gates on the result and
writes its verdict as JSON beside itself. Modelled on ``scripts/p4_acceptance.py``.

    python scripts/w4_acceptance.py --run 423 --detector jf1 --workers 36
    python scripts/w4_acceptance.py --run 423 --detector jf2 --workers 36

| Gate | Asks |
|---|---|
| 0 configuration | the whole run, on the workers claimed, at this config hash? |
| A self-test | did the NaN path equal the per-frame-mask reference on real frames? |
| B reference | does an *independent* integration reproduce what was stored? |
| C timing | how long did it take, and how much of the node did it use? |
| D ledger | is every frame accounted for, with the expected cells and bits? |

**Gate B is the one that earns its keep.** It re-reads the stored sums for a few
fully-OK trains and compares them against an integration done the slow way —
``integrate1d(mask = static | dynamic)`` per frame, the path
:mod:`analysis.waxs.selftest` calls the reference. That is a genuinely different
code path from the production NaN sentinel, and it goes through the writer's
rows, the ``f4`` storage and the row-to-train map, so the whole chain is inside
the comparison rather than just the kernel.

Needs a DAMNIT-partition node, the real PONI and ``.edf`` files, and the run.
"""

from __future__ import annotations

import os

# Before numpy, pyFAI or EXtra-data import anything that builds a thread pool.
for _name in (
    "EXTRA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common.masks import describe_bits, frame_bad  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.waxs.cells import UnexpectedLitCells  # noqa: E402
from analysis.waxs.config import (  # noqa: E402
    CELLS_PER_TRAIN,
    EXPECTED_BITS,
    config_for,
    is_storage_cell_sequence,
)
from analysis.waxs.integrate import ErrorModel  # noqa: E402
from analysis.waxs.operator import build_operator  # noqa: E402
from analysis.waxs.plan import open_detector  # noqa: E402
from analysis.waxs.selftest import SelfTestFailed, reference_frame  # noqa: E402

#: Measured on r0423 jf1, 36 workers, whole run: read_data 11.6, read_mask 5.0,
#: integrate 3.0 ms/frame/core. Set at roughly 1.5x each, so a real regression
#: fires - reverting the D5' NaN path would take integrate from 3.0 to ~17 ms -
#: without tripping on a merely busy node. A canary, not a verdict: exceeding
#: one is reported, only the wall time fails.
#:
#: ``integrate`` came in at 3.0 ms under full load against 2.3-2.9 ms on one idle
#: core, where the AGIPD pass found its stages 1.3-1.6x worse under load. Dense
#: pyFAI on 24 000 frames is simply not memory-bandwidth bound the way the AGIPD
#: sparse kernel on 465 000 frames is.
BUDGET_MS = {"read_data": 18.0, "read_mask": 8.0, "integrate": 5.0}

#: The whole r0423 jf1 pass took 17.5 s. 600 s is 34x that: loose enough to
#: survive a contended node, tight enough that something structural shows.
WALL_TARGET_S = 600.0


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def _lit_as_configured(cfg: Any, lit: list[int]) -> bool:
    """Does the file's lit set satisfy what this config asks of it?

    ``cfg.expected_lit_cells`` pins one named set and is ``None`` unless a
    caller asked for that, because the proposal used four different readout
    patterns and the pass measures the set per run. Unpinned, the acceptance
    cannot re-derive which cells *should* have been selected without reading
    the frames again — so it asks the run-invariant question instead: is the
    stored set non-empty, and is it a shape a JUNGFRAU can produce?
    """
    if cfg.expected_lit_cells is not None:
        return lit == list(cfg.expected_lit_cells)
    return bool(lit) and is_storage_cell_sequence(tuple(lit), CELLS_PER_TRAIN)


# ── gate A: run the pass ─────────────────────────────────────────────────────
def stage_run(cfg: Any, output: Path, workers: int) -> dict[str, Any]:
    """Run the pass. The self-test verdict is read back out of provenance."""
    import dataclasses

    from analysis.waxs.run import run_jungfrau_waxs

    started = time.perf_counter()
    try:
        run_jungfrau_waxs(
            dataclasses.replace(cfg, n_workers=workers),
            output_path=output,
            reduce="none",
        )
    except SelfTestFailed as error:
        return {
            "passed": False,
            "reason": "self-test",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    except UnexpectedLitCells as error:
        return {
            "passed": False,
            "reason": "lit cells",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    except Exception as error:  # noqa: BLE001 - the ledger explains it
        # The traceback, not just the repr. A gate that fails without saying
        # where costs a whole cluster round trip to read.
        formatted = traceback.format_exc()
        print(formatted, flush=True)
        return {
            "passed": False,
            "reason": "run raised",
            "error": repr(error),
            "traceback": formatted,
            "wall_s": time.perf_counter() - started,
        }
    return {"passed": True, "wall_s": time.perf_counter() - started}


def stage_selftest(output: Path, tolerance: float) -> dict[str, Any]:
    """Gate A: the NaN path against the per-frame-mask reference, on real frames."""
    with h5py.File(output, "r") as handle:
        stored = handle["provenance"].attrs.get("selftest")
    if stored is None:
        return {"passed": None, "reason": "no self-test recorded"}
    report = json.loads(stored)
    worst = max(
        report["max_rel_signal"],
        report["max_rel_normalization"],
        report["max_rel_variance"],
    )
    return {
        "passed": worst <= tolerance,
        "n_frames": report["n_frames"],
        "max_rel": worst,
        "tolerance": tolerance,
        **report,
    }


# ── gate 0: is this the run we think it is? ──────────────────────────────────
def stage_configuration(cfg: Any, output: Path, workers: int) -> dict[str, Any]:
    with h5py.File(output, "r") as handle:
        stored_hash = handle["provenance"].attrs.get("config_hash")
        n_workers = int(handle["provenance"].attrs.get("n_workers", 0))
        detector_name = handle["provenance"].attrs.get("detector_name")
        n_frames = int(handle["frames/status"].shape[0])
        n_trains = int(handle["trains/trainId"].shape[0])
        lit = [int(c) for c in handle["cells/lit"][:]]

    expected = cfg.config_hash()
    checks = {
        "config_hash_matches": stored_hash == expected,
        "workers_as_requested": n_workers == workers,
        "lit_matches_config": _lit_as_configured(cfg, lit),
    }
    return {
        "passed": all(checks.values()),
        "stored_config_hash": stored_hash,
        "expected_config_hash": expected,
        "n_workers": n_workers,
        "detector_name": detector_name,
        "n_trains": n_trains,
        "n_frames": n_frames,
        "lit_cells": lit,
        **checks,
    }


# ── gate B: an independent integration ───────────────────────────────────────
def stage_reference(
    cfg: Any, output: Path, n_trains: int, tolerance: float, dc: Any = None
) -> dict[str, Any]:
    """Re-integrate a few whole trains the slow way and compare to the file.

    The reference path passes ``static | dynamic`` to ``integrate1d(mask=)`` per
    frame — different code from the production NaN sentinel — and the comparison
    is against what the *writer stored*, pooled the way §9 pools it, so the row
    map and the ``f4`` storage are inside the gate.
    """
    from extra_data import by_id, open_run

    from analysis.common.plan import evenly_spaced

    op, ai = build_operator(cfg)

    with h5py.File(output, "r") as handle:
        if handle["operator"].attrs["sha256"] != op.sha256:
            return {
                "passed": False,
                "reason": "the operator rebuilt here differs from the stored one",
                "stored": handle["operator"].attrs["sha256"],
                "rebuilt": op.sha256,
            }
        model = ErrorModel(
            float(handle["cells"].attrs["read_noise_kev"]),
            float(handle["cells"].attrs["photon_energy_kev"]),
        )
        lit = [int(c) for c in handle["cells/lit"][:]]
        train_ids = handle["trains/trainId"][:]
        first = handle["trains/first"][:]
        count = handle["trains/count"][:]
        status = handle["frames/status"][:]

        fully_ok = []
        for index in range(train_ids.size):
            n = int(count[index])
            if n == 0:
                continue
            rows = slice(int(first[index]), int(first[index]) + n)
            if (status[rows] == FrameStatus.OK).all():
                fully_ok.append(index)
        if not fully_ok:
            return {"passed": False, "reason": "no fully-OK train to compare"}

        chosen = evenly_spaced(np.array(fully_ok, dtype=np.uint64), n_trains)
        dc = dc if dc is not None else open_run(cfg.proposal, cfg.run, data="proc")
        det = open_detector(cfg, dc)

        worst_intensity = 0.0
        bins_agree = True
        compared = []
        for index in chosen.tolist():
            index = int(index)
            train_id = int(train_ids[index])
            rows = slice(int(first[index]), int(first[index]) + int(count[index]))
            stored_signal = handle["frames/signal"][rows].astype(np.float64).sum(axis=0)
            stored_norm = (
                handle["frames/normalization"][rows].astype(np.float64).sum(axis=0)
            )

            selected = det.select_trains(by_id[[train_id]])
            data = np.asarray(selected["data.adc"].ndarray())[0, 0]
            mask = np.asarray(selected["data.mask"].ndarray())[0, 0]
            cells = np.asarray(selected["data.memoryCell"].ndarray()).reshape(-1)
            positions = {int(c): i for i, c in enumerate(cells.tolist())}

            ref_signal = np.zeros(op.npt)
            ref_norm = np.zeros(op.npt)
            for cell in lit:
                position = positions[cell]
                bad = frame_bad(mask[position], cfg.mask_bits, op.static_bad)
                signal, norm, _ = reference_frame(ai, op, model, data[position], bad)
                ref_signal += signal
                ref_norm += norm

            if not np.array_equal(stored_norm > 0, ref_norm > 0):
                bins_agree = False
            populated = ref_norm > 0
            stored_intensity = np.zeros(op.npt)
            ref_intensity = np.zeros(op.npt)
            np.divide(stored_signal, stored_norm, out=stored_intensity, where=populated)
            np.divide(ref_signal, ref_norm, out=ref_intensity, where=populated)
            relative = np.abs(stored_intensity - ref_intensity) / np.maximum(
                np.abs(ref_intensity), 1e-12
            )
            worst = float(relative[populated].max()) if populated.any() else 0.0
            worst_intensity = max(worst_intensity, worst)
            compared.append(
                {
                    "train": train_id,
                    "n_bins_populated": int(populated.sum()),
                    "max_rel_intensity": worst,
                }
            )

    return {
        "passed": bool(bins_agree and worst_intensity < tolerance),
        "trains": compared,
        "max_rel_intensity": worst_intensity,
        "tolerance": tolerance,
        "empty_bin_sets_equal": bins_agree,
    }


# ── gate C: timing ───────────────────────────────────────────────────────────
def stage_timing(output: Path, wall_s: float | None) -> dict[str, Any]:
    with h5py.File(output, "r") as handle:
        attrs = handle["provenance"].attrs
        timings = json.loads(attrs.get("timings", "{}"))
        setup = json.loads(attrs.get("setup_timings", "{}"))
        n_workers = int(attrs.get("n_workers", 1))
        n_blocks = int(attrs.get("n_blocks", 1))
        stored_wall = float(attrs.get("wall_s", 0.0)) or None
        n_ok = int((handle["frames/status"][:] == FrameStatus.OK).sum())

    wall = wall_s or stored_wall
    if not wall or not n_ok:
        return {"passed": None, "reason": "no wall time or no integrated frames"}

    per_frame = {k: 1000.0 * v / n_ok for k, v in timings.items()}
    over_budget = {
        k: {"measured_ms": v, "budget_ms": BUDGET_MS[k]}
        for k, v in per_frame.items()
        if k in BUDGET_MS and v > BUDGET_MS[k]
    }
    busy = min(n_workers, n_blocks)
    worker_seconds = sum(timings.values())
    return {
        # Only the wall time decides; a stage over budget is reported so the
        # next person knows where the time went, not failed.
        "passed": wall <= WALL_TARGET_S,
        "wall_s": wall,
        "wall_target_s": WALL_TARGET_S,
        "n_ok_frames": n_ok,
        "ms_per_frame_per_core": per_frame,
        "total_ms_per_frame_per_core": sum(per_frame.values()),
        "over_budget": over_budget,
        "parallel_efficiency": (worker_seconds / busy / wall) if wall else None,
        "setup_timings": setup,
        "serial_fraction": (sum(setup.values()) / wall) if wall else None,
    }


# ── gate D: the ledger ───────────────────────────────────────────────────────
def stage_ledger(cfg: Any, output: Path, pool: bool = True) -> dict[str, Any]:
    """The ledger, plus the one number that tests D3's actual claim.

    ``frames_with_negative_variance_bins`` says how often the unclamped
    estimator went non-positive on a *single* frame. That is expected and
    designed for. What matters is whether it survives pooling, which is the
    whole justification for storing it unclamped — so this pools the run the way
    §9 pools it and reports how many bins are still non-positive afterwards.
    """
    with h5py.File(output, "r") as handle:
        status = handle["frames/status"][:]
        count = handle["trains/count"][:]
        train_ids = handle["trains/trainId"][:]
        train_status = handle["trains/status"][:]
        bits = int(handle["provenance"].attrs.get("bits_present", 0))
        lit = [int(c) for c in handle["cells/lit"][:]]
        negative = handle["frames/n_negative_variance_bins"][:]

    counts = {
        code.name: int((status == code).sum())
        for code in FrameStatus
        if (status == code).any()
    }
    # Trains too, not only frames. A train that owns no rows - one the detector
    # never wrote - leaves no mark at all in the frame ledger, so a run can
    # reconcile perfectly while quietly having skipped a train.
    train_counts = {
        code.name: int((train_status == code).sum())
        for code in FrameStatus
        if (train_status == code).any()
    }
    offenders = {}
    for code in FrameStatus:
        if code is FrameStatus.OK:
            continue
        bad_trains = [
            int(train_ids[i])
            for i in range(train_ids.size)
            if int(train_status[i]) == int(code)
        ]
        if not bad_trains and not (status == code).any():
            continue
        offenders[code.name] = {
            "n_frames": counts.get(code.name, 0),
            "n_trains": len(bad_trains),
            "trains": bad_trains[:20],
        }

    # Does the unclamped variance survive pooling? That is D3's claim, and it is
    # cheap to check off the file that was just written.
    pooled: dict[str, Any] = {}
    if pool:
        from analysis.waxs.writer import pooled_per_train

        per_train = pooled_per_train(output)
        remaining = int(per_train.attrs.get("negative_variance_bins", 0))
        occupied = int((per_train["intensity"].values != 0).sum())
        pooled = {
            "pooled_negative_variance_bins": remaining,
            "pooled_occupied_bins": occupied,
            "pooled_negative_fraction": (remaining / occupied) if occupied else 0.0,
        }

    present = {b for b in range(32) if bits >> b & 1}
    unexpected = present - set(EXPECTED_BITS)
    reconciles = int(count.sum()) == status.size
    return {
        "passed": bool(
            reconciles
            and set(counts) == {"OK"}
            and not unexpected
            and _lit_as_configured(cfg, lit)
        ),
        "status_counts": counts,
        "train_status_counts": train_counts,
        "offending_trains": offenders,
        "frames_reconcile": reconciles,
        "n_frames": int(status.size),
        "n_trains": int(train_ids.size),
        "n_trains_with_rows": int((count > 0).sum()),
        "sum_train_counts": int(count.sum()),
        "bits_present": sorted(present),
        "bits_named": describe_bits(bits),
        "unexpected_bits": sorted(unexpected),
        "lit_cells": lit,
        # The unclamped variance of D3: how often a per-frame bin came out
        # non-positive. Reported, never failed - it is expected behaviour.
        "frames_with_negative_variance_bins": int((negative > 0).sum()),
        "fraction_of_frames_with_negative_variance_bins": (
            float((negative > 0).mean()) if negative.size else 0.0
        ),
        "max_negative_variance_bins_in_a_frame": int(negative.max())
        if negative.size
        else 0,
        **pooled,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--detector", required=True, choices=["jf1", "jf2"])
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--npt", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--poni-file", type=Path, default=None, help="override the config default"
    )
    parser.add_argument(
        "--static-mask-file", type=Path, default=None, help="override the default"
    )
    parser.add_argument("--ref-trains", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--selftest-tolerance",
        type=float,
        default=1e-9,
        help="the NaN path was measured exact; this leaves room for a different "
        "summation order, not a different answer",
    )
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.poni_file is not None:
        overrides["poni_file"] = str(args.poni_file)
    if args.static_mask_file is not None:
        overrides["static_mask_file"] = str(args.static_mask_file)
    cfg = config_for(
        args.proposal,
        args.run,
        args.detector,
        npt=args.npt,
        n_workers=args.workers,
        overwrite=args.overwrite,
        **overrides,
    )
    output = args.output or cfg.output_file

    # Before anything hashes them. A wrong path should say so, not surface as a
    # FileNotFoundError from inside config_hash.
    missing = {
        name: path
        for name, path in cfg.input_files.items()
        if not path or not Path(path).exists()
    }
    if missing:
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "run": args.run,
            "detector": args.detector,
            "passed": False,
            "gates": {
                "0_inputs": {
                    "passed": False,
                    "reason": "input files missing",
                    "missing": missing,
                }
            },
        }
        print(f"input files missing: {missing}")
        _write(report, args)
        return 2

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "proposal": args.proposal,
        "run": args.run,
        "detector": args.detector,
        "workers": args.workers,
        "config_hash": cfg.config_hash(),
        "output": str(output),
        "thread_env": {
            k: os.environ.get(k) for k in ("EXTRA_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "gates": {},
    }

    if args.skip_run:
        report["gates"]["A_run"] = {"passed": None, "reason": "--skip-run"}
        if not output.exists():
            report["passed"] = False
            report["gates"]["A_run"]["reason"] = f"{output} does not exist"
            _write(report, args)
            return 2
    else:
        report["gates"]["A_run"] = stage_run(cfg, output, args.workers)
        if report["gates"]["A_run"]["passed"] is False:
            report["passed"] = False
            _write(report, args)
            print(f"gate A run: FAIL — {report['gates']['A_run']}")
            return 1

    wall = report["gates"]["A_run"].get("wall_s")
    report["gates"]["A_selftest"] = stage_selftest(output, args.selftest_tolerance)
    report["gates"]["0_configuration"] = stage_configuration(cfg, output, args.workers)
    report["gates"]["B_reference"] = stage_reference(
        cfg, output, args.ref_trains, args.tolerance
    )
    report["gates"]["C_timing"] = stage_timing(output, wall)
    report["gates"]["D_ledger"] = stage_ledger(cfg, output)

    verdicts = {k: v.get("passed") for k, v in report["gates"].items()}
    report["passed"] = all(v is not False for v in verdicts.values())

    print(f"\n=== W4 acceptance, r{args.run:04d} {args.detector} ===")
    for name, verdict in verdicts.items():
        label = {True: "PASS", False: "FAIL", None: "SKIP"}[verdict]
        print(f"  {label}  {name}")
    timing = report["gates"]["C_timing"]
    if timing.get("wall_s"):
        print(
            f"\n  wall {timing['wall_s']:.1f} s (target {WALL_TARGET_S:.0f} s), "
            f"efficiency {timing.get('parallel_efficiency')}"
        )
        print(f"  per frame per core: {timing['ms_per_frame_per_core']}")
    selftest = report["gates"]["A_selftest"]
    if selftest.get("max_rel") is not None:
        print(
            f"  self-test max rel {selftest['max_rel']:.2e} over "
            f"{selftest['n_frames']} frames"
        )
    reference = report["gates"]["B_reference"]
    print(f"  gate B max rel intensity {reference.get('max_rel_intensity')}")
    ledger = report["gates"]["D_ledger"]
    print(f"  ledger frames {ledger.get('status_counts')}")
    print(
        f"  ledger trains {ledger.get('train_status_counts')} "
        f"({ledger.get('n_trains_with_rows')} of {ledger.get('n_trains')} own rows)"
    )
    negative_fraction = ledger.get("fraction_of_frames_with_negative_variance_bins", 0)
    print(
        f"  frames with a negative-variance bin: "
        f"{ledger.get('frames_with_negative_variance_bins')} "
        f"({100 * negative_fraction:.1f} %)"
    )
    print(
        f"  ...still negative after pooling per train: "
        f"{ledger.get('pooled_negative_variance_bins')} of "
        f"{ledger.get('pooled_occupied_bins')} occupied bins"
    )

    _write(report, args)
    return 0 if report["passed"] else 1


def _write(report: dict[str, Any], args) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.json or Path(__file__).with_name(
        f"w4_acceptance_r{args.run:04d}_{args.detector}_{stamp}.json"
    )
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
