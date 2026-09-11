#!/usr/bin/env python
"""P4 acceptance for the first-pass AGIPD SAXS integrator (context file §10).

Runs the full pass on one run and then checks, in order, the four things P4
asks for. Every number it decides on is written to a JSON next to this script
(CLAUDE.md working rule 4), so the verdict is reproducible and re-readable
without rerunning the pass.

    A  self-test passed, on real frames, before the pool started
    B  pooled I(q) for N trains matches a dense pyFAI reference to < tolerance
    C  per-stage timing against the §2 budget, and wall time
    D  every non-OK status accounted for

What gate B does and does not add over gate A. The self-test compares the
sparse path against the pyFAI engine frame by frame, in memory, in the parent.
Gate B re-reads what the *writer stored*, pools it the way §9 says to, and
compares that against an independently accumulated dense reference over whole
trains. So it is the row-to-train mapping, the float32 storage and
``pooled_per_train`` that gate B tests, on ~60x more frames; the kernel itself
is gate A's job. Both use ``selftest.reference_frame``, which is the "engine
with explicit ``variance``" the phase asks for — never ``ErrorModel.POISSON``
(CLAUDE.md pitfall 2).

Usage on a DAMNIT-partition node, from the uv environment:

    python scripts/p4_acceptance.py --proposal 10400 --run 423 --workers 36

To check an output file that already exists, without reintegrating:

    python scripts/p4_acceptance.py --run 423 --skip-run --output <path.h5>
"""

from __future__ import annotations

import os

# Before numpy, pyFAI or EXtra-data are imported. This process does the dense
# reference itself, and a multi-threaded BLAS here would make the stage timings
# in gate C describe a machine nobody will run the real pass on (CLAUDE.md:
# any process reading AGIPD data pins these to 1).
_THREAD_ENV = (
    "EXTRA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
for _name in _THREAD_ENV:
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.saxs import masks as masks_module  # noqa: E402
from analysis.saxs import operator as operator_module  # noqa: E402
from analysis.saxs.config import FirstPassConfig  # noqa: E402
from analysis.saxs.selftest import SelfTestFailed, reference_frame  # noqa: E402
from analysis.saxs.status import FrameStatus  # noqa: E402

#: Per-frame, per-core cost measured in the stage benchmark (context file §2).
#: The worker times ``read_data`` and ``read_mask`` around the EXtra-data reads
#: and ``integrate`` around the frame loop, so the three are directly
#: comparable; there is no measured budget for anything else.
BUDGET_MS = {"read_data": 4.6, "read_mask": 8.2, "integrate": 6.2}

#: Wall-time target for r0423 on 36 workers (context file §10, P4).
WALL_TARGET_S = 6 * 60

log = logging.getLogger("p4")


class _SelfTestCapture(logging.Handler):
    """Keep the parent's self-test line.

    ``run_first_pass`` gates on the self-test and then logs the worst relative
    differences, but does not return the report, so the numbers are recovered
    from the log record's arguments rather than by reimplementing the gate.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.args: tuple[Any, ...] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if str(record.msg).startswith("self-test passed") and record.args:
            self.args = tuple(record.args)


# ── gate A: the run itself ────────────────────────────────────────────────────
def stage_run(
    cfg: FirstPassConfig, output: Path | None, run_dir: str | None = None
) -> dict[str, Any]:
    """Integrate the whole run, and record what the self-test reported."""
    from analysis.saxs.run import run_first_pass

    capture = _SelfTestCapture()
    logging.getLogger("analysis.saxs.run").addHandler(capture)

    started = time.perf_counter()
    try:
        run_first_pass(cfg, run_dir=run_dir, output_path=output)
    except SelfTestFailed as error:
        return {
            "passed": False,
            "reason": "self-test failed; the pool was never started",
            "report": error.report.as_provenance(),
            "wall_s": time.perf_counter() - started,
        }
    finally:
        logging.getLogger("analysis.saxs.run").removeHandler(capture)

    wall_s = time.perf_counter() - started
    if capture.args is None:
        return {
            "passed": False,
            "reason": "the run completed but logged no self-test line",
            "wall_s": wall_s,
        }
    n_frames, rel_s, rel_n, rel_v = capture.args
    return {
        "passed": True,
        "selftest_frames": int(n_frames),
        "max_rel_signal": float(rel_s),
        "max_rel_normalization": float(rel_n),
        "max_rel_variance": float(rel_v),
        "wall_s": wall_s,
    }


# ── gate B: pooled I(q) against a dense reference ─────────────────────────────
def _fully_ok_trains(handle: h5py.File) -> np.ndarray:
    """Trains whose every stored frame reached ``OK``."""
    status = handle["frames/status"][:]
    train_ids = handle["trains/trainId"][:]
    first = handle["trains/first"][:]
    count = handle["trains/count"][:]
    keep = []
    for train_id, start, n in zip(train_ids, first, count, strict=True):
        if n and (status[int(start) : int(start) + int(n)] == FrameStatus.OK).all():
            keep.append(int(train_id))
    return np.array(keep, dtype=np.uint64)


def _pooled_from_file(handle: h5py.File, train_id: int) -> np.ndarray:
    """``ΣS / ΣN`` over one train's rows, exactly as §9 pools them."""
    index = int(np.flatnonzero(handle["trains/trainId"][:] == train_id)[0])
    start = int(handle["trains/first"][index])
    n = int(handle["trains/count"][index])
    rows = slice(start, start + n)
    signal = handle["frames/signal"][rows].astype(np.float64).sum(axis=0)
    normalization = handle["frames/normalization"][rows].astype(np.float64).sum(axis=0)
    return signal, normalization


def stage_reference(
    cfg: FirstPassConfig,
    output: Path,
    n_trains: int,
    tolerance: float,
    run_dir: str | None = None,
    geometry: Any = None,
) -> dict[str, Any]:
    """Compare stored, pooled I(q) against an independent dense accumulation.

    :param geometry: an ``AGIPD_1MGeometry`` to rebuild the reference operator
        from, instead of ``cfg.geometry_file``. Only the tests pass it, which
        is how this gate is exercised on the mock run before it is trusted on
        a node.
    """
    from extra_data import RunDirectory, by_id, open_run
    from extra_data.components import AGIPD1M

    geom = geometry or operator_module.geometry_from_config(cfg)
    op, ai = operator_module.build_operator(geom, cfg)
    static = masks_module.build_static_bad(cfg)
    engine = ai.engines[next(iter(ai.engines))].engine

    with h5py.File(output, "r") as handle:
        stored_operator = handle["operator"].attrs["sha256"]
        stored_static = handle["masks"].attrs["static_sha256"]
        candidates = _fully_ok_trains(handle)
        if candidates.size == 0:
            return {"passed": False, "reason": "no train has only OK frames"}
        chosen = masks_module.evenly_spaced(candidates, n_trains)
        pooled = {int(t): _pooled_from_file(handle, int(t)) for t in chosen.tolist()}

    # The reference is only a reference if it was built on the same geometry
    # and the same mask as the run it is checking.
    if stored_operator != op.sha256:
        return {
            "passed": False,
            "reason": "the operator rebuilt here differs from the one in the file",
            "file_operator_sha256": str(stored_operator),
            "rebuilt_operator_sha256": op.sha256,
        }
    if stored_static != static.sha256:
        return {
            "passed": False,
            "reason": "the static mask rebuilt here differs from the one in the file",
            "file_static_sha256": str(stored_static),
            "rebuilt_static_sha256": static.sha256,
        }

    dc = RunDirectory(run_dir) if run_dir else open_run(cfg.proposal, cfg.run, "proc")
    det = AGIPD1M(dc, detector_name=cfg.detector_name, min_modules=cfg.min_modules)

    per_train = []
    worst_intensity = 0.0
    worst_signal = 0.0
    worst_normalization = 0.0
    bins_agree = True

    for train_id, (signal_ours, norm_ours) in pooled.items():
        selected = det.select_trains(by_id[[train_id]])
        data = selected["image.data"].ndarray(decompress_threads=1)
        mask = selected["image.mask"].ndarray(decompress_threads=1)

        signal_ref = np.zeros(cfg.npt, dtype=np.float64)
        norm_ref = np.zeros(cfg.npt, dtype=np.float64)
        for frame in range(data.shape[1]):
            counts = data[:, frame].reshape(-1)
            bad = masks_module.frame_bad(mask[:, frame], cfg.mask_bits, static.bad)
            ref = reference_frame(engine, op, counts, bad)
            signal_ref += ref.signal
            norm_ref += ref.normalization

        populated_ours = norm_ours > 0
        populated_ref = norm_ref > 0
        agree = bool(np.array_equal(populated_ours, populated_ref))
        bins_agree = bins_agree and agree

        both = populated_ours & populated_ref
        intensity_ours = signal_ours[both] / norm_ours[both]
        intensity_ref = signal_ref[both] / norm_ref[both]
        rel_i = float(
            (
                np.abs(intensity_ours - intensity_ref)
                / np.maximum(np.abs(intensity_ref), 1e-12)
            ).max()
        )
        rel_s = float(
            (
                np.abs(signal_ours[both] - signal_ref[both])
                / np.maximum(np.abs(signal_ref[both]), 1e-12)
            ).max()
        )
        rel_n = float(
            (
                np.abs(norm_ours[both] - norm_ref[both])
                / np.maximum(np.abs(norm_ref[both]), 1e-12)
            ).max()
        )
        worst_intensity = max(worst_intensity, rel_i)
        worst_signal = max(worst_signal, rel_s)
        worst_normalization = max(worst_normalization, rel_n)
        per_train.append(
            {
                "trainId": int(train_id),
                "n_frames": int(data.shape[1]),
                "populated_bins": int(both.sum()),
                "empty_bins_agree": agree,
                "max_rel_intensity": rel_i,
                "max_rel_signal": rel_s,
                "max_rel_normalization": rel_n,
            }
        )
        log.info("train %d: max rel I(q) %.3e", train_id, rel_i)

    return {
        "passed": bool(bins_agree and worst_intensity < tolerance),
        "tolerance": tolerance,
        "trains": [int(t) for t in pooled],
        "empty_bins_agree": bins_agree,
        "max_rel_intensity": worst_intensity,
        "max_rel_signal": worst_signal,
        "max_rel_normalization": worst_normalization,
        # The sums are stored as float32, so ~1e-7 of the difference is
        # quantisation and not a disagreement between the two paths.
        "float32_eps": float(np.finfo(np.float32).eps),
        "per_train": per_train,
    }


# ── gate C: timings ───────────────────────────────────────────────────────────
def stage_timing(output: Path, measured_wall_s: float | None) -> dict[str, Any]:
    """Per-stage cost per frame per core, and wall time, against §2."""
    with h5py.File(output, "r") as handle:
        provenance = handle["provenance"].attrs
        timings = json.loads(provenance["timings"])
        summary = json.loads(provenance["status_summary"])
        n_workers = int(provenance["n_workers"])
        wall_s = float(provenance.get("wall_s", 0.0)) or (measured_wall_s or 0.0)

    ok_frames = int(summary.get(FrameStatus.OK.name, 0))
    if ok_frames == 0:
        return {"passed": False, "reason": "no OK frames to divide the timings by"}

    stages = {}
    for name, budget in BUDGET_MS.items():
        cpu_s = float(timings.get(name, 0.0))
        per_frame_ms = cpu_s * 1e3 / ok_frames
        stages[name] = {
            "cpu_s": cpu_s,
            "ms_per_frame_per_core": per_frame_ms,
            "budget_ms": budget,
            "ratio": per_frame_ms / budget,
        }
    total_cpu_s = sum(float(v) for v in timings.values())
    total_ms = total_cpu_s * 1e3 / ok_frames
    total_budget = sum(BUDGET_MS.values())

    # Everything the workers did not time: block setup, pickling results back,
    # the parent's writes. It is the gap worth looking at if the wall time
    # misses while every stage is inside its budget.
    ideal_wall_s = total_cpu_s / n_workers if n_workers else float("nan")

    return {
        # The budget is a measurement, not a contract: a stage over budget is
        # reported, and only the wall time decides the gate.
        "passed": bool(wall_s > 0 and wall_s <= WALL_TARGET_S),
        "ok_frames": ok_frames,
        "n_workers": n_workers,
        "wall_s": wall_s,
        "wall_target_s": WALL_TARGET_S,
        "stages": stages,
        "total_ms_per_frame_per_core": total_ms,
        "total_budget_ms": total_budget,
        "total_ratio": total_ms / total_budget,
        "worker_cpu_s": total_cpu_s,
        "ideal_wall_s": ideal_wall_s,
        "parallel_efficiency": (ideal_wall_s / wall_s) if wall_s else None,
        "untimed_wall_s": wall_s - ideal_wall_s if wall_s else None,
    }


# ── gate D: the ledger ────────────────────────────────────────────────────────
def stage_ledger(output: Path, max_listed: int = 20) -> dict[str, Any]:
    """Account for every frame and every train that is not ``OK``."""
    with h5py.File(output, "r") as handle:
        provenance = handle["provenance"].attrs
        status = handle["frames/status"][:]
        train_ids = handle["trains/trainId"][:]
        first = handle["trains/first"][:]
        count = handle["trains/count"][:]
        train_status = handle["trains/status"][:]
        masks_attrs = dict(handle["masks"].attrs)
        run_checks = json.loads(provenance["run_checks"])
        block_errors = json.loads(provenance.get("block_errors", "{}"))
        unseen = int(provenance.get("unseen_cell_frames", 0))
        bits_present = int(provenance.get("bits_present", 0))
        static_sources = json.loads(provenance["static_mask_sources"])

    frame_counts = {
        code.name: int((status == code).sum())
        for code in FrameStatus
        if (status == code).any()
    }
    train_counts = {
        code.name: int((train_status == code).sum())
        for code in FrameStatus
        if (train_status == code).any()
    }

    # Which trains carry each non-OK frame status, so a count is never left as
    # a bare number nobody can trace back to a train.
    offenders: dict[str, dict[str, Any]] = {}
    for code in FrameStatus:
        if code is FrameStatus.OK or not (status == code).any():
            continue
        rows = np.flatnonzero(status == code)
        index = np.searchsorted(first.astype(np.int64), rows, side="right") - 1
        affected = np.unique(train_ids[np.clip(index, 0, train_ids.size - 1)])
        offenders[code.name] = {
            "n_frames": int(rows.size),
            "n_trains": int(affected.size),
            "trains": [int(t) for t in affected[:max_listed]],
            "truncated": bool(affected.size > max_listed),
        }

    total = int(status.size)
    reconciles = sum(frame_counts.values()) == total == int(count.sum())

    return {
        "passed": bool(reconciles and set(frame_counts) <= {FrameStatus.OK.name}),
        "n_frames": total,
        "frame_status": frame_counts,
        "train_status": train_counts,
        "non_ok": offenders,
        "reconciles": reconciles,
        "block_errors": block_errors,
        "unseen_cell_frames": unseen,
        "bits_present": bits_present,
        "bits_present_named": masks_module.describe_bits(bits_present),
        "unexpected_bits": int(masks_attrs.get("unexpected_bits", 0)),
        "static_mask_sources": static_sources,
        "run_checks": run_checks,
    }


# ── driver ────────────────────────────────────────────────────────────────────
def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - recorded, never fatal
        return f"unavailable: {error!r}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument(
        "--workers", type=int, default=36, help="P4 asks for 36 (one per physical core)"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "open the run by directory. The workers and the gate B reader use "
            "it; the parent still resolves proposal/run for the plan and masks"
        ),
    )
    parser.add_argument("--geometry-file", default=None)
    parser.add_argument("--pixel-mask-file", default=None)
    parser.add_argument("--ref-trains", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="check an existing output file instead of integrating",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = FirstPassConfig(
        proposal=args.proposal,
        run=args.run,
        n_workers=args.workers,
        overwrite=args.overwrite,
    )
    if args.geometry_file:
        cfg = replace(cfg, geometry_file=args.geometry_file)
    if args.pixel_mask_file:
        cfg = replace(cfg, pixel_mask_file=args.pixel_mask_file)
    output = Path(args.output) if args.output else cfg.output_file

    report: dict[str, Any] = {
        "run": args.run,
        "proposal": args.proposal,
        "when": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": _git_commit(),
        "thread_env": {name: os.environ[name] for name in _THREAD_ENV},
        "config_hash": cfg.config_hash(),
        "output": str(output),
        "gates": {},
    }

    measured_wall_s: float | None = None
    if args.skip_run:
        if not output.exists():
            log.error("--skip-run needs an existing output file: %s", output)
            return 2
        report["gates"]["A_selftest"] = {"passed": None, "reason": "--skip-run"}
    else:
        gate_a = stage_run(cfg, args.output, args.run_dir)
        measured_wall_s = gate_a.get("wall_s")
        report["gates"]["A_selftest"] = gate_a
        if not gate_a["passed"]:
            _write(report, args)
            return 1

    report["gates"]["B_dense_reference"] = stage_reference(
        cfg, output, args.ref_trains, args.tolerance, args.run_dir
    )
    report["gates"]["C_timing"] = stage_timing(output, measured_wall_s)
    report["gates"]["D_ledger"] = stage_ledger(output)

    verdicts = {name: gate.get("passed") for name, gate in report["gates"].items()}
    report["passed"] = all(v is not False for v in verdicts.values())
    _write(report, args)

    print(f"\nP4 acceptance — run {args.run:04d}")
    for name, verdict in verdicts.items():
        label = {True: "PASS", False: "FAIL", None: "SKIP"}[verdict]
        print(f"  {label}  {name}")
    timing = report["gates"]["C_timing"]
    if "stages" in timing:
        print(
            f"\n  wall {timing['wall_s']:.1f} s on {timing['n_workers']} workers "
            f"(target {timing['wall_target_s']} s), "
            f"{timing['total_ms_per_frame_per_core']:.1f} ms/frame/core vs "
            f"{timing['total_budget_ms']:.1f} budget"
        )
    reference = report["gates"]["B_dense_reference"]
    if "max_rel_intensity" in reference:
        print(
            f"  pooled I(q) vs dense pyFAI: max rel "
            f"{reference['max_rel_intensity']:.2e} "
            f"(tolerance {reference['tolerance']:.0e})"
        )
    print(f"  ledger: {report['gates']['D_ledger']['frame_status']}")
    return 0 if report["passed"] else 1


def _write(report: dict[str, Any], args: argparse.Namespace) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.json or Path(__file__).with_name(
        f"p4_acceptance_r{args.run:04d}_{stamp}.json"
    )
    path.write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote %s", path)


if __name__ == "__main__":
    raise SystemExit(main())
