#!/usr/bin/env python
"""P4 acceptance for the AGIPD SAXS integrator, ``agipd_saxs`` (context file §10).

Runs the full pass on one run and then checks, in order, the four things P4
asks for. Every number it decides on is written to a JSON next to this script
(CLAUDE.md working rule 4), so the verdict is reproducible and re-readable
without rerunning the pass.

    A  self-test passed, on real frames, before the pool started
    B  pooled I(q) for N trains matches a dense pyFAI reference to < tolerance
    C  per-stage timing against the §2 budget, and wall time
    D  every non-OK status accounted for
    E  every per-pixel window sum equals ``analysis.cache.detector_sums`` over
       the window's trains, exactly (context file §15; skipped when
       ``--pixel-sum-trains none``)
    E2 the frame table pooled over each window equals the window sums gathered
       through the operator off the static mask (§15), to < tolerance

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

The window-sum overhead (S2 of §15) is the difference between two runs of this
script on one node, ``--pixel-sum-trains none`` and ``--pixel-sum-trains 10``.
``--sums-only`` instead times ``run_pixel_sums``, the backfill, and grades its
file with gate E; ``--output`` is then the window-sums file itself.
"""

from __future__ import annotations

# Before numpy, pyFAI or EXtra-data are imported. This process does the dense
# reference itself, and a multi-threaded BLAS here would make the stage timings
# in gate C describe a machine nobody will run the real pass on.
from _common import report_header, write_report  # noqa: E402

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common.arrays import relative_difference  # noqa: E402
from analysis.common.masks import frame_bad  # noqa: E402
from analysis.common.plan import evenly_spaced  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.saxs import masks as masks_module  # noqa: E402
from analysis.saxs import operator as operator_module  # noqa: E402
from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.saxs.pixel_sums import FILE_NAME, window_sums  # noqa: E402
from analysis.saxs.selftest import SelfTestFailed, reference_frame  # noqa: E402

#: Per-frame, per-core cost measured in the stage benchmark (context file §2).
#: The worker times ``read_data`` and ``read_mask`` around the EXtra-data reads
#: and ``integrate`` around the frame loop, so the three are directly
#: comparable; there is no measured budget for anything else.
BUDGET_MS = {"read_data": 4.6, "read_mask": 8.2, "integrate": 6.2}

#: Wall-time ceiling for r0423 on 36 workers.
#:
#: P4's "~6 min" was a design estimate; the accepted run came in at 377.6 s and
#: was signed off because the pipeline this replaces takes over an hour and
#: DAMNIT's default ``slurm_time`` is 2 h. So this is not that estimate — it is
#: a regression canary set with headroom above the accepted run (roughly 1.6x),
#: loose enough not to trip on a busier node and tight enough that a real
#: slowdown still fails. The hard operational limit is ``slurm_time``.
WALL_TARGET_S = 600

log = logging.getLogger("p4")


class _SelfTestCapture(logging.Handler):
    """Keep the parent's self-test line.

    ``run_agipd_saxs`` gates on the self-test and then logs the worst relative
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
    cfg: AgipdSaxsConfig, output: Path | None, run_dir: str | None = None
) -> dict[str, Any]:
    """Integrate the whole run, and record what the self-test reported."""
    from analysis.saxs.run import run_agipd_saxs

    capture = _SelfTestCapture()
    logging.getLogger("analysis.saxs.run").addHandler(capture)

    started = time.perf_counter()
    try:
        run_agipd_saxs(cfg, run_dir=run_dir, output_path=output)
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
    cfg: AgipdSaxsConfig,
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
        chosen = evenly_spaced(candidates, n_trains)
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
            bad = frame_bad(mask[:, frame], cfg.mask_bits, static.bad)
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
        rel_i = float(relative_difference(intensity_ours, intensity_ref).max())
        rel_s = float(relative_difference(signal_ours[both], signal_ref[both]).max())
        rel_n = float(relative_difference(norm_ours[both], norm_ref[both]).max())
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


# ── gate 0: is this the configuration P4 asks for? ────────────────────────────
#: P4 asks for one worker per physical core on the DAMNIT-partition node.
P4_WORKERS = 36


def stage_configuration(
    cfg: AgipdSaxsConfig, output: Path, run_dir: str | None = None
) -> dict[str, Any]:
    """Check that the file describes a full run integrated on 36 workers.

    Every other gate is happy to grade a ten-train trial, and would report
    numbers that look like an acceptance without being one. This one asks the
    question the others cannot: is the file in front of us the thing P4 is
    about? It also compares the file's own config hash against the config the
    reference gate is built from, which ``--skip-run`` otherwise leaves
    unchecked.
    """
    from extra_data import RunDirectory, open_run

    with h5py.File(output, "r") as handle:
        provenance = handle["provenance"].attrs
        n_workers = int(provenance["n_workers"])
        stored_hash = str(provenance["config_hash"])
        trains_in_file = int(handle["trains/trainId"].size)
        frames_in_file = int(handle["frames/status"].size)

    try:
        dc = (
            RunDirectory(run_dir)
            if run_dir
            else open_run(cfg.proposal, cfg.run, "proc")
        )
        trains_in_run = int(len(dc.train_ids))
    except Exception as error:  # noqa: BLE001 - recorded, not fatal
        trains_in_run = 0
        log.warning("could not count the run's trains: %r", error)

    full_run = bool(trains_in_run and trains_in_file == trains_in_run)
    hash_matches = stored_hash == cfg.config_hash()
    return {
        "passed": bool(full_run and n_workers == P4_WORKERS and hash_matches),
        "n_workers": n_workers,
        "workers_expected": P4_WORKERS,
        "trains_in_file": trains_in_file,
        "trains_in_run": trains_in_run,
        "frames_in_file": frames_in_file,
        "coverage": (trains_in_file / trains_in_run) if trains_in_run else None,
        "full_run": full_run,
        "config_hash_matches": hash_matches,
        "file_config_hash": stored_hash,
        "script_config_hash": cfg.config_hash(),
    }


# ── gate C: timings ───────────────────────────────────────────────────────────
def stage_timing(output: Path, measured_wall_s: float | None) -> dict[str, Any]:
    """Per-stage cost per frame per core, and wall time, against §2."""
    with h5py.File(output, "r") as handle:
        provenance = handle["provenance"].attrs
        timings = json.loads(provenance["timings"])
        summary = json.loads(provenance["status_summary"])
        n_workers = int(provenance["n_workers"])
        n_blocks = int(provenance.get("n_blocks", 0))
        setup = json.loads(provenance.get("setup_timings", "{}"))
        # Every attrs read stays inside the `with`: h5py's AttributeManager
        # outlives the file it came from, and `.get` on a closed file returns
        # the default instead of raising.
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
    # Stages with no measured budget — the window sums' ``pixel_sums`` — are
    # reported beside the others and enter the total, but have no ratio.
    for name in sorted(set(timings) - set(BUDGET_MS)):
        cpu_s = float(timings[name])
        stages[name] = {
            "cpu_s": cpu_s,
            "ms_per_frame_per_core": cpu_s * 1e3 / ok_frames,
            "budget_ms": None,
            "ratio": None,
        }
    total_cpu_s = sum(float(v) for v in timings.values())
    total_ms = total_cpu_s * 1e3 / ok_frames
    total_budget = sum(BUDGET_MS.values())

    # A run with fewer blocks than workers cannot occupy them all, so dividing
    # the CPU seconds by the worker count would invent an efficiency the job
    # never had a chance to reach.
    busy = min(n_workers, n_blocks) if n_blocks else n_workers
    ideal_wall_s = total_cpu_s / busy if busy else float("nan")

    # A missing wall time is not a slow run. The gate is undecided (None), not
    # failed, so a file written before wall_s was recorded does not read as a
    # performance miss.
    if wall_s <= 0:
        verdict: bool | None = None
        reason = (
            "no wall time recorded: this file predates provenance/wall_s, "
            "so rerun the pass to decide this gate"
        )
    else:
        verdict = bool(wall_s <= WALL_TARGET_S)
        reason = ""

    return {
        # The budget is a measurement, not a contract: a stage over budget is
        # reported, and only the wall time decides the gate.
        "passed": verdict,
        "reason": reason,
        "ok_frames": ok_frames,
        "n_workers": n_workers,
        "n_blocks": n_blocks,
        "busy_workers": busy,
        "wall_s": wall_s,
        "wall_target_s": WALL_TARGET_S,
        "stages": stages,
        "total_ms_per_frame_per_core": total_ms,
        "total_budget_ms": total_budget,
        "total_ratio": total_ms / total_budget,
        "worker_cpu_s": total_cpu_s,
        # The parent's serial phases: whatever the wall time exceeds the
        # workers' share by has to come from here or from the tail.
        "setup_s": setup,
        "setup_total_s": sum(setup.values()),
        "ideal_wall_s": ideal_wall_s,
        "parallel_efficiency": (ideal_wall_s / wall_s) if wall_s > 0 else None,
        "untimed_wall_s": (wall_s - ideal_wall_s) if wall_s > 0 else None,
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

    # A train with no rows of its own never appears in the frame ledger, so
    # its status is only ever visible here. Naming the trains is what makes
    # "accounted for" mean something.
    train_offenders: dict[str, dict[str, Any]] = {}
    for code in FrameStatus:
        if code is FrameStatus.OK or not (train_status == code).any():
            continue
        affected = train_ids[train_status == code]
        train_offenders[code.name] = {
            "n_trains": int(affected.size),
            "trains": [int(t) for t in affected[:max_listed]],
            "n_rows_owned": int(count[train_status == code].sum()),
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
        "non_ok_trains": train_offenders,
        "reconciles": reconciles,
        "block_errors": block_errors,
        "unseen_cell_frames": unseen,
        "bits_present": bits_present,
        "bits_present_named": masks_module.describe_bits(bits_present),
        "unexpected_bits": int(masks_attrs.get("unexpected_bits", 0)),
        "static_mask_sources": static_sources,
        "run_checks": run_checks,
    }


# ── gate E: window sums against the cache ─────────────────────────────────────
def _window_members(sums_path: Path) -> dict[int, list[int]]:
    """Each written window's summed trains, by label."""
    with h5py.File(sums_path, "r") as handle:
        train_ids = handle["trains/trainId"][:]
        window = handle["trains/window"][:]
        status = handle["trains/status"][:]
        written = handle["windows/written"][:].astype(bool)
    members: dict[int, list[int]] = {}
    for index in np.flatnonzero(written):
        trains = train_ids[(window == index) & (status == FrameStatus.OK)]
        if trains.size:
            members[int(index)] = [int(t) for t in trains]
    return members


def stage_window_sums(
    cfg: AgipdSaxsConfig,
    sums_path: Path,
    run_dir: str | None,
    n_workers: int,
    spot_checks: int = 3,
    max_listed: int = 10,
) -> dict[str, Any]:
    """Every window against ``analysis.cache.detector_sums``, exactly.

    ``detector_sums`` spawns a pool per call, and r0423 has 300 windows, so the
    comparison for *every* window runs its kernel, ``cache._sum_job``, in one
    pool over the same five-train chunks and modules ``cache._read_sums``
    hands it, and adds the parts the same way. ``spot_checks`` windows — the
    first, last and evenly between — then go through the public
    ``detector_sums`` itself, so the shortcut is checked against the function
    it stands in for.
    """
    from analysis import cache
    from analysis.common.cpu import default_pool

    with h5py.File(sums_path, "r") as handle:
        written = handle["windows/written"][:].astype(bool)
        n_frames = handle["windows/n_frames"][:]
        status = handle["trains/status"][:]
        provenance = handle["provenance"].attrs
        window_trains = int(provenance["window_trains"])
    ledger = {
        code.name: int((status == code).sum())
        for code in FrameStatus
        if (status == code).any()
    }
    members = _window_members(sums_path)
    proc_dir = Path(run_dir) if run_dir else cache.PROC_ROOT / f"r{cfg.run:04d}"

    jobs, owners = [], []
    for index, trains in members.items():
        for start in range(0, len(trains), cache.SUM_CHUNK):
            for module in range(cache.N_MODULES):
                jobs.append(
                    (str(proc_dir), trains[start : start + cache.SUM_CHUNK], module)
                )
                owners.append(index)
    remaining = {index: owners.count(index) for index in members}
    parts: dict[int, list[np.ndarray]] = {}
    mismatched: list[dict[str, Any]] = []
    checked = 0
    started = time.perf_counter()
    with default_pool(n_workers) as pool:
        for index, (module, counts, valid, frames) in zip(
            owners, pool.map(cache._sum_job, jobs, chunksize=8), strict=True
        ):
            total = parts.setdefault(
                index,
                [
                    np.zeros((cache.N_MODULES, *cache.MODULE_SHAPE), np.int64),
                    np.zeros((cache.N_MODULES, *cache.MODULE_SHAPE), np.int64),
                    np.zeros(cache.N_MODULES, np.int64),
                ],
            )
            total[0][module] += counts
            total[1][module] += valid
            total[2][module] += frames
            remaining[index] -= 1
            if remaining[index]:
                continue
            ref_counts, ref_valid, ref_frames = parts.pop(index)
            ours = window_sums(sums_path, [index])
            same = (
                np.unique(ref_frames).size == 1
                and int(ref_frames[0]) == int(n_frames[index])
                and np.array_equal(ours.counts.values, ref_counts)
                and np.array_equal(ours.valid_frames.values, ref_valid)
            )
            checked += 1
            if not same:
                mismatched.append(
                    {
                        "window": index,
                        "frames_ours": int(n_frames[index]),
                        "frames_reference": ref_frames.tolist(),
                        "max_abs_counts": int(
                            np.abs(ours.counts.values - ref_counts).max()
                        ),
                        "max_abs_valid": int(
                            np.abs(ours.valid_frames.values - ref_valid).max()
                        ),
                    }
                )
    reference_s = time.perf_counter() - started

    spots = []
    if members and spot_checks > 0:
        chosen = evenly_spaced(np.array(sorted(members), dtype=np.uint64), spot_checks)
        for index in chosen.tolist():
            ref = cache.detector_sums(
                cfg.run,
                members[index],
                proposal=cfg.proposal,
                run_dir=run_dir,
                n_workers=n_workers,
                cache_dir=sums_path.parent / "detector_sums_reference",
                refresh=True,
            )
            ours = window_sums(sums_path, [index])
            spots.append(
                {
                    "window": int(index),
                    "equal": bool(
                        np.array_equal(ours.counts.values, ref.counts.values)
                        and np.array_equal(
                            ours.valid_frames.values, ref.valid_frames.values
                        )
                        and np.array_equal(ours.train_id.values, ref.train_id.values)
                        and ours.attrs["n_frames"] == ref.attrs["n_frames"]
                    ),
                }
            )

    failed = {
        name: n
        for name, n in ledger.items()
        if name
        in {"LABEL_MISMATCH", "DATA_CHECK_FAILED", "WORKER_ERROR", "NOT_PROCESSED"}
    }
    return {
        "passed": bool(
            written.all()
            and not failed
            and checked == len(members)
            and not mismatched
            and all(spot["equal"] for spot in spots)
        ),
        "window_trains": window_trains,
        "n_windows": int(written.size),
        "windows_written": int(written.sum()),
        "windows_compared": checked,
        "windows_mismatched": len(mismatched),
        "mismatched": mismatched[:max_listed],
        "spot_checks_with_detector_sums": spots,
        "train_status": ledger,
        "trains_summed": int(sum(len(t) for t in members.values())),
        "frames_summed": int(n_frames.sum()),
        "reference_wall_s": reference_s,
    }


# ── gate E2: the frame table and the window sums tell one story ───────────────
def stage_window_identity(
    output: Path, sums_path: Path, work_dir: Path, tolerance: float
) -> dict[str, Any]:
    """Pooled S, N and V per window against the window sums through the operator.

    Σ S = Σ c·counts, Σ N = Σ c·Ω·valid_frames and Σ V = Σ c²·counts over the
    pixels off the static mask (context file §15). It holds only with every
    mask bit in ``mask_bits``, which is the default, and it is what makes the
    window sums a record the 1D result can be rebuilt from under any geometry.
    The products go through a scipy CSC matrix rather than ``sparse.gather``,
    so the pass's own kernel is not grading itself.
    """
    from scipy.sparse import csc_matrix

    op = operator_module.load_operator(work_dir / "operator.npz")
    static = masks_module.load_masks(work_dir / "masks.npz").static_bad

    with h5py.File(output, "r") as handle:
        stored_operator = str(handle["operator"].attrs["sha256"])
        mask_bits = int(json.loads(handle["provenance"].attrs["config"])["mask_bits"])
        train_ids = handle["trains/trainId"][:]
        first = handle["trains/first"][:]
        count = handle["trains/count"][:]
        frame_status = handle["frames/status"][:]
    if stored_operator != op.sha256:
        return {"passed": False, "reason": "operator.npz is not the file's operator"}
    if mask_bits != 0xFFFFFFFF:
        return {
            "passed": None,
            "reason": f"mask_bits {mask_bits:#x}: the identity needs every bit",
        }

    npix = op.indptr.size - 1
    matrix = csc_matrix(
        (op.coef.astype(np.float64), op.bins, op.indptr), shape=(op.npt, npix)
    )
    squared = matrix.multiply(matrix).tocsc()
    keep = (~static).astype(np.float64)
    row_of = {
        int(t): (int(f), int(c))
        for t, f, c in zip(train_ids, first, count, strict=True)
    }

    worst = {"signal": 0.0, "normalization": 0.0, "variance": 0.0}
    bins_agree = True
    skipped = []
    compared = 0
    with h5py.File(output, "r") as handle:
        frames = handle["frames"]
        for index, trains in _window_members(sums_path).items():
            start = row_of[trains[0]][0]
            stop = sum(row_of[trains[-1]])
            if not (frame_status[start:stop] == FrameStatus.OK).all():
                skipped.append(index)  # rows of a train the window did sum
                continue
            pooled = {
                name: frames[name][start:stop].astype(np.float64).sum(axis=0)
                for name in worst
            }
            sums = window_sums(sums_path, [index])
            counts = sums.counts.values.reshape(-1) * keep
            valid = sums.valid_frames.values.reshape(-1) * keep
            expected = {
                "signal": matrix @ counts,
                "normalization": matrix @ (op.omega * valid),
                "variance": squared @ counts,
            }
            agree = np.array_equal(
                pooled["normalization"] > 0, expected["normalization"] > 0
            )
            bins_agree = bins_agree and agree
            both = (pooled["normalization"] > 0) & (expected["normalization"] > 0)
            for name in worst:
                worst[name] = max(
                    worst[name],
                    float(
                        relative_difference(
                            pooled[name][both], expected[name][both]
                        ).max(initial=0.0)
                    ),
                )
            compared += 1

    return {
        "passed": bool(
            compared and not skipped and bins_agree and max(worst.values()) < tolerance
        ),
        "tolerance": tolerance,
        "windows_compared": compared,
        "windows_skipped_for_non_ok_rows": skipped,
        "empty_bins_agree": bins_agree,
        "max_rel": worst,
        "float32_eps": float(np.finfo(np.float32).eps),
    }


# ── the window sums' own timing ───────────────────────────────────────────────
def stage_sums_timing(sums_path: Path, measured_wall_s: float | None) -> dict[str, Any]:
    """Per-stage cost per summed frame, from the window-sums file alone."""
    with h5py.File(sums_path, "r") as handle:
        provenance = handle["provenance"].attrs
        timings = json.loads(provenance["timings"])
        setup = json.loads(provenance.get("setup_timings", "{}"))
        wall_s = float(provenance.get("wall_s", 0.0)) or (measured_wall_s or 0.0)
        n_workers = int(provenance["n_workers"])
        frames = int(handle["windows/n_frames"][:].sum())
    if frames == 0:
        return {"passed": None, "reason": "no frames summed"}
    total_cpu_s = sum(float(v) for v in timings.values())
    return {
        "passed": None,  # a measurement for the decision, not a gate
        "frames": frames,
        "n_workers": n_workers,
        "wall_s": wall_s,
        "stages_ms_per_frame_per_core": {
            name: float(v) * 1e3 / frames for name, v in timings.items()
        },
        "total_ms_per_frame_per_core": total_cpu_s * 1e3 / frames,
        "setup_s": setup,
        "parallel_efficiency": (total_cpu_s / n_workers / wall_s) if wall_s else None,
    }


def run_sums_only(cfg: AgipdSaxsConfig, sums_path: Path, run_dir: str | None) -> float:
    """``run_pixel_sums`` on the whole run; returns its wall time."""
    from analysis.saxs.run import run_pixel_sums

    started = time.perf_counter()
    run_pixel_sums(
        cfg, run_dir=Path(run_dir) if run_dir else None, pixel_sums_path=sums_path
    )
    return time.perf_counter() - started


# ── driver ────────────────────────────────────────────────────────────────────


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
    parser.add_argument(
        "--pixel-sum-trains",
        default=None,
        help="window length for the per-pixel sums, or 'none' to write none "
        "(default: the config's)",
    )
    parser.add_argument(
        "--sums-only",
        action="store_true",
        help="run run_pixel_sums instead of the pass; --output is then the "
        "window-sums file, and only gate E and the timing are graded",
    )
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = AgipdSaxsConfig(
        proposal=args.proposal,
        run=args.run,
        n_workers=args.workers,
        overwrite=args.overwrite,
    )
    if args.geometry_file:
        cfg = replace(cfg, geometry_file=args.geometry_file)
    if args.pixel_mask_file:
        cfg = replace(cfg, pixel_mask_file=args.pixel_mask_file)
    if args.pixel_sum_trains is not None:
        trains = args.pixel_sum_trains
        cfg = replace(
            cfg, pixel_sum_trains=None if trains.lower() == "none" else int(trains)
        )
    if args.sums_only:
        return main_sums_only(cfg, args)
    output = Path(args.output) if args.output else cfg.output_file

    report: dict[str, Any] = report_header(
        run=args.run,
        proposal=args.proposal,
        config_hash=cfg.config_hash(),
        output=str(output),
        gates={},
    )

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

    report["gates"]["0_configuration"] = stage_configuration(cfg, output, args.run_dir)
    report["gates"]["B_dense_reference"] = stage_reference(
        cfg, output, args.ref_trains, args.tolerance, args.run_dir
    )
    report["gates"]["C_timing"] = stage_timing(output, measured_wall_s)
    report["gates"]["D_ledger"] = stage_ledger(output)
    report["pixel_sum_trains"] = cfg.pixel_sum_trains
    if cfg.pixel_sum_trains is not None:
        # run_agipd_saxs puts them beside the output it is given
        sums_path = output.parent / FILE_NAME
        report["window_sums_file"] = str(sums_path)
        report["gates"]["E_window_sums"] = stage_window_sums(
            cfg, sums_path, args.run_dir, cfg.workers
        )
        report["gates"]["E2_window_identity"] = stage_window_identity(
            output, sums_path, output.parent, args.tolerance
        )

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
    configuration = report["gates"]["0_configuration"]
    print(
        f"  configuration: {configuration['trains_in_file']} of "
        f"{configuration['trains_in_run'] or '?'} trains on "
        f"{configuration['n_workers']} workers "
        f"(P4 asks for the full run on {P4_WORKERS})"
    )
    print(f"  ledger: {report['gates']['D_ledger']['frame_status']}")
    if "pixel_sums" in timing.get("stages", {}):
        print(
            "  window sums: "
            f"{timing['stages']['pixel_sums']['ms_per_frame_per_core']:.2f} "
            "ms/frame/core in the workers"
        )
    return 0 if report["passed"] else 1


def main_sums_only(cfg: AgipdSaxsConfig, args: Any) -> int:
    """Time ``run_pixel_sums`` and grade its file with gate E."""
    if cfg.pixel_sum_trains is None:
        log.error("--sums-only needs windows; --pixel-sum-trains is none")
        return 2
    sums_path = Path(args.output) if args.output else cfg.pixel_sums_file
    report: dict[str, Any] = report_header(
        run=args.run,
        proposal=args.proposal,
        mode="sums_only",
        pixel_sum_trains=cfg.pixel_sum_trains,
        window_sums_file=str(sums_path),
        gates={},
    )
    measured = None if args.skip_run else run_sums_only(cfg, sums_path, args.run_dir)
    report["gates"]["E_window_sums"] = stage_window_sums(
        cfg, sums_path, args.run_dir, cfg.workers
    )
    report["sums_timing"] = stage_sums_timing(sums_path, measured)
    report["passed"] = report["gates"]["E_window_sums"]["passed"] is not False
    _write(report, args, suffix="_sums_only")
    timing = report["sums_timing"]
    print(f"\nsums only — run {args.run:04d}")
    print(f"  {'PASS' if report['passed'] else 'FAIL'}  E_window_sums")
    if "wall_s" in timing:
        print(
            f"  wall {timing['wall_s']:.1f} s on {timing['n_workers']} workers, "
            f"{timing['total_ms_per_frame_per_core']:.2f} ms/frame/core"
        )
    return 0 if report["passed"] else 1


def _write(report: dict[str, Any], args: Any, suffix: str = "") -> None:
    """Write the report beside this script, stamped with the time."""
    write_report(report, args.json, f"p4_acceptance_r{args.run:04d}{suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
