#!/usr/bin/env python
"""pyFAI against pyFAI: two versions, one node, the same r0423 frames (task 19a).

The job runs ``--measure`` once under each interpreter in ``--pythons``, one
after the other, and compares what they return. Each measurement reads its own
frames from proc — the same trains, through the same EXtra-data — so the
inputs are identical and only pyFAI differs between the two venvs.

What one measurement records:

* **AGIPD operator** — build time, sha256 and the CSC arrays, which the parent
  compares element by element. A changed operator means a reprocess, not a
  resume (CLAUDE.md open task 17b).
* **AGIPD per frame** — the pass's sparse kernel (pure numpy, so a control that
  should not move) and the self-test's dense pyFAI reference, in ms/frame, and
  the self-test verdict on these frames.
* **JUNGFRAU per frame** — the pass's ``integrate_frame`` (dense pyFAI, the
  per-frame mask carried as NaN) in ms/frame; the same frames through
  ``mask=`` instead, timed (pitfall 17: a rebuild per frame) and compared bit
  for bit with the NaN path.
* **integrate2d on AGIPD frames** for three method tuples: the first call,
  which builds the matrix, and the steady ms/frame; the parent compares the 2D
  results across versions, since 2026.9 re-indexes pixels by detector
  orientation and ``chi`` may move.
* **Pitfall probes** — what a method string resolves to and what an OpenCL
  request falls back to (1), the Poisson floor (2), engine keys (5), read-only
  inputs and the solid-angle cache (18).
* **Imports** — extra_speckle (with xcca) and the EXtra components.

Everything runs in one single-threaded process, as one worker of a pass does.
Under a full node's load the numbers rise (P4 measured ~1.3x for AGIPD), so
they compare versions; they are not a pass's operating point.

Usage, from the uv environment (the job lands on a whole node; ``-w`` cores
are only what it asks SLURM for):

    python scripts/bench_pyfai.py --run 423 -w 4 --constraint 9374F \\
        --pythons $S/venv-pyfai2026.5/bin/python $S/venv-pyfai2026.9/bin/python

The verdict goes to JSON beside this script; the arrays it compared stay in
``--work-dir``.
"""

from __future__ import annotations

# Before numpy, pyFAI or EXtra-data are imported: every timing here is one
# single-threaded process, as one worker of a pass is.
from _common import (  # noqa: E402
    add_job_args,
    configure_logging,
    dispatch,
    report_header,
    stamp,
    write_report,
)

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from collections.abc import Callable, Iterator, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

log = logging.getLogger("bench_pyfai")

#: Where the per-version JSON and arrays go, one directory per job.
WORK_ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400/scratch/s3_speed/bench_pyfai")

#: Frames per measurement. AGIPD: the first ``AGIPD_FRAMES_PER_TRAIN`` of each
#: of ``AGIPD_TRAINS`` trains. JUNGFRAU: the lit cells of ``JF_TRAINS`` trains.
AGIPD_TRAINS = 2
AGIPD_FRAMES_PER_TRAIN = 16
JF_TRAINS = 4
JF_DETECTOR = "jf1"
#: Frames timed through ``mask=``; each one rebuilds the matrix (pitfall 17).
JF_MASK_FRAMES = 4
#: integrate2d: steady-state frames per method, and the grid.
FRAMES_2D = 8
NPT_RAD_2D = 500
NPT_AZIM_2D = 360
METHODS_2D = (
    ("full", "csr", "cython"),
    ("bbox", "csr", "cython"),
    ("no", "histogram", "cython"),
)
#: Timing repeats over the frame list, after one warm-up call.
REPEATS = 3


# ── the measurement, under one interpreter ────────────────────────────────────


def time_per_frame(
    fn: Callable[[Any], Any], items: Sequence[Any], repeats: int = REPEATS
) -> dict[str, float]:
    """ms per item over ``repeats`` passes, after one warm-up call."""
    fn(items[0])
    walls = []
    for _ in range(repeats):
        started = time.perf_counter()
        for item in items:
            fn(item)
        walls.append(time.perf_counter() - started)
    per = [1e3 * wall / len(items) for wall in walls]
    return {
        "ms_per_frame_median": statistics.median(per),
        "ms_per_frame_min": min(per),
        "ms_per_frame_max": max(per),
        "n_frames": len(items),
        "repeats": repeats,
    }


@contextlib.contextmanager
def captured_pyfai_messages() -> Iterator[list[str]]:
    """Warnings and pyFAI log records raised inside the block, as text."""
    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(f"{record.levelname} {record.name}: {record.getMessage()}")

    handler = Collect(level=logging.DEBUG)
    pyfai_logger = logging.getLogger("pyFAI")
    previous = pyfai_logger.level
    pyfai_logger.addHandler(handler)
    pyfai_logger.setLevel(logging.INFO)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield messages
        finally:
            pyfai_logger.removeHandler(handler)
            pyfai_logger.setLevel(previous)
            messages.extend(f"warning: {w.message}" for w in caught)


def attempt(fn: Callable[[], Any]) -> dict[str, Any]:
    """``{"ok": True, "value": ...}`` or the exception, never a raise."""
    try:
        return {"ok": True, "value": fn()}
    except Exception as error:  # noqa: BLE001 - the probe's answer
        return {"ok": False, "error": repr(error)}


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def read_agipd_frames(cfg: Any) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Flat int16 frames, their uint32 masks and train ids, from mid-run."""
    from extra_data import by_id, open_run
    from extra_data.components import AGIPD1M

    det = AGIPD1M(
        open_run(cfg.proposal, cfg.run, data="proc"),
        detector_name=cfg.detector_name,
        min_modules=cfg.min_modules,
    )
    middle = len(det.train_ids) // 2
    frames, masks, trains = [], [], []
    for train_id in det.train_ids[middle : middle + AGIPD_TRAINS]:
        selected = det.select_trains(by_id[[train_id]])
        data = selected["image.data"].ndarray(decompress_threads=1)
        mask = selected["image.mask"].ndarray(decompress_threads=1)
        for k in range(min(AGIPD_FRAMES_PER_TRAIN, data.shape[1])):
            frames.append(np.ascontiguousarray(data[:, k]).reshape(-1))
            masks.append(np.ascontiguousarray(mask[:, k]).reshape(-1))
        trains.append(int(train_id))
    return frames, masks, trains


def read_jf_frames(cfg: Any) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """The lit cells of a few mid-run trains, flat keV frames and masks."""
    from extra_data import by_id, open_run

    from analysis.waxs.config import CELLS_PER_TRAIN, EXPECTED_LIT_CELLS, NPIX
    from analysis.waxs.plan import open_detector

    det = open_detector(cfg, open_run(cfg.proposal, cfg.run, data="proc"))
    middle = len(det.train_ids) // 2
    frames, masks, trains = [], [], []
    for train_id in det.train_ids[middle : middle + JF_TRAINS]:
        selected = det.select_trains(by_id[[train_id]])
        data = selected["data.adc"].ndarray().reshape(CELLS_PER_TRAIN, NPIX)
        mask = selected["data.mask"].ndarray().reshape(CELLS_PER_TRAIN, NPIX)
        cells = selected["data.memoryCell"].ndarray().reshape(-1)
        # by label, never by position (pitfall 4); r0423's lit set
        for position, cell in enumerate(cells.tolist()):
            if cell in EXPECTED_LIT_CELLS:
                frames.append(np.array(data[position]))
                masks.append(np.array(mask[position]))
        trains.append(int(train_id))
    return frames, masks, trains


def measure_agipd(proposal: int, run: int, arrays: dict[str, np.ndarray]) -> dict:
    from analysis.common.masks import frame_bad
    from analysis.saxs.config import SHAPE, AgipdSaxsConfig
    from analysis.saxs.masks import build_static_bad
    from analysis.saxs.operator import build_operator, geometry_from_config
    from analysis.saxs.selftest import compare_frame, reference_frame
    from analysis.saxs.sparse import denominator, integrate_frame

    report: dict[str, Any] = {}
    cfg = AgipdSaxsConfig(proposal=proposal, run=run)

    started = time.perf_counter()
    op, ai = build_operator(geometry_from_config(cfg), cfg)
    report["operator_build_s"] = time.perf_counter() - started
    report["operator_sha256"] = op.sha256
    report["operator_nnz"] = op.nnz
    report["detector_orientation"] = int(getattr(ai.detector, "orientation", -1))
    for name in ("coef", "bins", "indptr", "omega", "q"):
        arrays[f"agipd_operator_{name}"] = np.asarray(getattr(op, name))

    static = build_static_bad(cfg)
    static_denominator = denominator(op, static.bad)
    frames, masks, trains = read_agipd_frames(cfg)
    report["trains"] = trains
    bads = [frame_bad(m, cfg.mask_bits, static.bad) for m in masks]
    items = list(zip(frames, masks, bads, strict=True))

    def sparse(item):
        x, m, _ = item
        bad = frame_bad(m, cfg.mask_bits, static.bad)
        return integrate_frame(op, x, bad, static.bad, static_denominator)

    report["sparse_kernel"] = time_per_frame(sparse, items)

    probe = ai.integrate1d(
        np.ones(SHAPE, np.float32), cfg.npt, method=cfg.method, unit=cfg.unit
    )
    engine = ai.engines[probe.method].engine
    report["dense_reference"] = time_per_frame(
        lambda item: reference_frame(engine, op, item[0], item[2]), items
    )
    worst = [0.0, 0.0, 0.0]
    empty_agree = True
    for x, _, bad in items:
        *rel, agree = compare_frame(engine, op, x, bad, static.bad, static_denominator)
        worst = [max(a, b) for a, b in zip(worst, rel, strict=True)]
        empty_agree &= agree
    report["selftest"] = {
        "max_rel_signal": worst[0],
        "max_rel_normalization": worst[1],
        "max_rel_variance": worst[2],
        "empty_bins_agree": empty_agree,
        "passed": empty_agree and max(worst) < 1e-6,
        "n_frames": len(items),
    }
    reference = reference_frame(engine, op, frames[0], bads[0])
    for name in ("signal", "normalization", "variance"):
        arrays[f"agipd_reference_{name}"] = np.asarray(getattr(reference, name))

    report["integrate2d"] = measure_2d(ai, cfg, frames, bads, arrays)
    report["pitfalls"] = agipd_pitfalls(ai, cfg, frames[0], bads[0])
    return report


def measure_2d(ai, cfg, frames, bads, arrays) -> dict[str, Any]:
    """integrate2d per method: first call (matrix build) and steady state."""
    from analysis.saxs.config import SHAPE

    images = [
        np.where(bad, np.nan, x).astype(np.float32).reshape(SHAPE)
        for x, bad in zip(frames[:FRAMES_2D], bads[:FRAMES_2D], strict=True)
    ]
    results = {}
    for method in METHODS_2D:
        label = "_".join(method)

        def integrate(image, method=method):
            return ai.integrate2d(
                image, NPT_RAD_2D, NPT_AZIM_2D, unit=cfg.unit, method=method
            )

        try:
            started = time.perf_counter()
            first = integrate(images[0])
            first_s = time.perf_counter() - started
            steady = time_per_frame(integrate, images)
        except Exception as error:  # noqa: BLE001 - recorded per method
            results[label] = {"error": repr(error), "traceback": traceback.format_exc()}
            continue
        results[label] = {
            "resolved": list(
                (
                    first.method.split_lower,
                    first.method.algo_lower,
                    first.method.impl_lower,
                )
            ),
            "first_call_s": first_s,
            **steady,
        }
        for name in (
            "intensity",
            "radial",
            "azimuthal",
            "sum_signal",
            "sum_normalization",
        ):
            arrays[f"agipd_2d_{label}_{name}"] = np.asarray(getattr(first, name))
    return results


def agipd_pitfalls(ai, cfg, x, bad) -> dict[str, Any]:
    """Pitfalls 1, 2 and 5 on the AGIPD integrator, after everything else."""
    from analysis.saxs.config import SHAPE

    image = np.where(bad, np.nan, x).astype(np.float32).reshape(SHAPE)
    report: dict[str, Any] = {}

    # 1: a method string, and an OpenCL request on a node without OpenCL
    for label, method in (
        ("string_csc", "csc"),
        ("tuple_full_csr_opencl", ("full", "csr", "opencl")),
    ):
        with captured_pyfai_messages() as messages:
            outcome = attempt(
                lambda method=method: (
                    ai.integrate1d(image, cfg.npt, unit=cfg.unit, method=method).method
                )
            )
        if outcome["ok"]:
            resolved = outcome["value"]
            outcome["value"] = [
                resolved.split_lower,
                resolved.algo_lower,
                resolved.impl_lower,
            ]
        report[f"pitfall1_{label}"] = outcome | {"messages": messages[:10]}

    # 2: the Poisson model's floor, against the explicit variance=x
    counts = np.where(bad, np.nan, x).astype(np.float32).reshape(SHAPE)
    explicit = ai.integrate1d(
        counts, cfg.npt, unit=cfg.unit, method=cfg.method, variance=counts
    )
    poisson = ai.integrate1d(
        counts, cfg.npt, unit=cfg.unit, method=cfg.method, error_model="poisson"
    )
    ratio = np.sum(poisson.sum_variance) / np.sum(explicit.sum_variance)
    report["pitfall2_poisson_over_explicit_variance"] = float(ratio)

    # 5: what the engine cache is keyed on
    ai.integrate1d(image, cfg.npt - 100, unit=cfg.unit, method=cfg.method)
    report["pitfall5_engine_keys"] = [str(key) for key in ai.engines]
    return report


def measure_jf(proposal: int, run: int, arrays: dict[str, np.ndarray]) -> dict:
    from analysis.common.masks import frame_bad
    from analysis.waxs.config import DEFAULT_READ_NOISE_KEV, MODULE_SHAPE, config_for
    from analysis.waxs.integrate import ErrorModel, extreme_pixels, integrate_frame
    from analysis.waxs.operator import build_operator

    report: dict[str, Any] = {}
    cfg = config_for(proposal, run, JF_DETECTOR)
    started = time.perf_counter()
    op, ai = build_operator(cfg)
    report["operator_build_s"] = time.perf_counter() - started
    report["operator_sha256"] = op.sha256
    report["detector_orientation"] = int(getattr(ai.detector, "orientation", -1))
    model = ErrorModel(
        read_noise_kev=DEFAULT_READ_NOISE_KEV[JF_DETECTOR],
        photon_energy_kev=cfg.photon_energy_kev,
    )
    frames, masks, trains = read_jf_frames(cfg)
    report["trains"] = trains
    bads = [frame_bad(m, cfg.mask_bits, op.static_bad) for m in masks]
    items = list(zip(frames, bads, strict=True))

    def nan_path(item):
        x, bad = item
        return integrate_frame(ai, op, model, x, bad, max_abs_kev=cfg.max_abs_kev)

    report["nan_path"] = time_per_frame(nan_path, items)
    nan_results = [nan_path(item) for item in items]
    for name in ("signal", "normalization", "variance"):
        arrays[f"jf_{name}"] = np.asarray(getattr(nan_results[0], name))

    # The same frames through mask=, which pitfall 17 says rebuilds the matrix
    # every time the mask changes. Run last: it leaves the engine rebuilt.
    def mask_path(item):
        x, bad = item
        excluded = bad | extreme_pixels(x, bad, cfg.max_abs_kev)
        signal = x.astype(np.float32, copy=True)
        variance = model.variance(signal)
        return ai.integrate1d(
            signal.reshape(MODULE_SHAPE),
            op.npt,
            method=op.method,
            unit=op.unit,
            mask=np.array(excluded.reshape(MODULE_SHAPE), dtype=bool, order="C"),
            variance=variance.reshape(MODULE_SHAPE),
        )

    worst = {"signal": 0.0, "normalization": 0.0, "variance": 0.0}
    identical = True
    started = time.perf_counter()
    for item, nan_result in zip(items[:JF_MASK_FRAMES], nan_results, strict=False):
        result = mask_path(item)
        for name, ours in (
            ("signal", result.sum_signal),
            ("normalization", result.sum_normalization),
            ("variance", result.sum_variance),
        ):
            ours = np.asarray(ours, np.float64)
            theirs = getattr(nan_result, name)
            identical &= bool(np.array_equal(ours, theirs))
            scale = np.maximum(np.abs(theirs), 1e-30)
            worst[name] = max(worst[name], float(np.max(np.abs(ours - theirs) / scale)))
    n_mask = min(JF_MASK_FRAMES, len(items))
    report["mask_path"] = {
        "ms_per_frame": 1e3 * (time.perf_counter() - started) / n_mask,
        "n_frames": n_mask,
        "bit_identical_to_nan_path": identical,
        "max_rel_difference": worst,
    }
    report["pitfall18"] = jf_readonly_probes(ai, op, frames[0], bads[0], model)
    return report


def jf_readonly_probes(ai, op, x, bad, model) -> dict[str, Any]:
    """Pitfall 18: read-only inputs, and the cached solid-angle array."""
    from analysis.waxs.config import MODULE_SHAPE

    signal = np.where(bad, np.nan, x).astype(np.float32).reshape(MODULE_SHAPE)
    variance = model.variance(signal)

    def frozen(array):
        copy = np.array(array)
        copy.flags.writeable = False
        return copy

    def run(**overrides):
        kwargs = {
            "mask": np.array(op.static_mask_2d),
            "variance": variance,
            "data": signal,
        } | overrides
        data = kwargs.pop("data")
        return ai.integrate1d(
            data, op.npt, method=op.method, unit=op.unit, **kwargs
        ).method.algo_lower

    report = {
        "readonly_mask": attempt(lambda: run(mask=frozen(op.static_mask_2d))),
        "readonly_data": attempt(lambda: run(data=frozen(signal))),
        "readonly_variance": attempt(lambda: run(variance=frozen(variance))),
    }
    first = ai.solidAngleArray(MODULE_SHAPE)
    report["solid_angle_array_is_cached_object"] = bool(
        ai.solidAngleArray(MODULE_SHAPE) is first
    )
    report["ascontiguousarray_returns_same_object"] = bool(
        np.ascontiguousarray(first, dtype=first.dtype) is first
    )
    report["solid_angle_array_writeable"] = bool(first.flags.writeable)
    return report


def import_checks() -> dict[str, Any]:
    """Modules the DAMNIT context and the notebooks import."""
    import importlib

    checks = {}
    for name in (
        "extra_speckle",
        "extra_speckle.xcca",
        "extra_speckle.saxs",
        "extra.components",
        "extra.calibration",
        "extra.applications.xcca",
    ):
        checks[name] = attempt(lambda name=name: bool(importlib.import_module(name)))
    return checks


def measure(args: argparse.Namespace) -> int:
    """One version's numbers, written to ``<work-dir>/<label>.json`` + ``.npz``."""
    import pyFAI

    from analysis.common.cpu import package_versions

    work = Path(args.work_dir)
    arrays: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {
        "label": args.measure,
        "python": sys.executable,
        "pyFAI": pyFAI.version,
        "package_versions": package_versions(),
        "cpu": cpu_model(),
    }
    for name, step in (
        ("agipd", lambda: measure_agipd(args.proposal, args.run, arrays)),
        ("jf", lambda: measure_jf(args.proposal, args.run, arrays)),
        ("imports", import_checks),
    ):
        started = time.perf_counter()
        try:
            report[name] = step()
        except Exception as error:  # noqa: BLE001 - recorded, the other steps run
            report[name] = {"error": repr(error), "traceback": traceback.format_exc()}
        log.info(
            "%s: %s done in %.1f s", args.measure, name, time.perf_counter() - started
        )
    np.savez(work / f"{args.measure}.npz", **arrays)
    (work / f"{args.measure}.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return 0


# ── the job: both versions, then the comparison ───────────────────────────────


def compare_arrays(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict:
    """Per array: identical, or the largest difference and where it is."""
    out = {}
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            out[name] = {"present_in_both": False}
            continue
        x, y = np.asarray(a[name]), np.asarray(b[name])
        entry: dict[str, Any] = {
            "identical": bool(np.array_equal(x, y, equal_nan=True))
        }
        if x.shape != y.shape:
            entry["shapes"] = [list(x.shape), list(y.shape)]
        elif not entry["identical"] and x.dtype.kind in "fiu":
            diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
            finite = np.isfinite(diff)
            scale = np.maximum(np.abs(x.astype(np.float64)), 1e-30)
            entry["n_different"] = int(np.count_nonzero(diff[finite]))
            entry["max_abs_difference"] = float(diff[finite].max(initial=0.0))
            entry["max_rel_difference"] = float((diff / scale)[finite].max(initial=0.0))
            entry["nan_patterns_equal"] = bool(
                np.array_equal(
                    np.isnan(x.astype(np.float64)), np.isnan(y.astype(np.float64))
                )
            )
        out[name] = entry
    return out


def speedups(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """``a`` / ``b`` for every per-frame timing both measured."""

    def ms(report, *path):
        for key in path:
            report = report.get(key, {}) if isinstance(report, dict) else {}
        return report if isinstance(report, int | float) else None

    rows = {
        "agipd_operator_build_s": ("agipd", "operator_build_s"),
        "agipd_sparse_kernel": ("agipd", "sparse_kernel", "ms_per_frame_median"),
        "agipd_dense_reference": ("agipd", "dense_reference", "ms_per_frame_median"),
        "jf_operator_build_s": ("jf", "operator_build_s"),
        "jf_nan_path": ("jf", "nan_path", "ms_per_frame_median"),
        "jf_mask_path": ("jf", "mask_path", "ms_per_frame"),
    }
    for method in METHODS_2D:
        label = "_".join(method)
        rows[f"agipd_2d_{label}"] = (
            "agipd",
            "integrate2d",
            label,
            "ms_per_frame_median",
        )
        rows[f"agipd_2d_{label}_first_call_s"] = (
            "agipd",
            "integrate2d",
            label,
            "first_call_s",
        )
    table = {}
    for name, path in rows.items():
        old, new = ms(a, *path), ms(b, *path)
        table[name] = {
            "old": old,
            "new": new,
            "old_over_new": old / new if old and new else None,
        }
    return table


def run_one(args: argparse.Namespace) -> int:
    configure_logging("bench_pyfai")
    if len(args.pythons) != 2:
        raise SystemExit("--pythons takes two interpreters: old, then new")
    work = WORK_ROOT / f"r{args.run:04d}_{stamp()}"
    work.mkdir(parents=True, exist_ok=True)
    labels = []
    for index, python in enumerate(args.pythons):
        label = f"v{index}"
        command = [
            python,
            str(Path(__file__).resolve()),
            "--measure",
            label,
            "--run",
            str(args.run),
            "--proposal",
            str(args.proposal),
            "--work-dir",
            str(work),
            "--local",
        ]
        log.info("measuring %s with %s", label, python)
        result = subprocess.run(command)
        if result.returncode:
            log.error("%s exited %d", label, result.returncode)
            return 1
        labels.append(label)

    reports = [json.loads((work / f"{label}.json").read_text()) for label in labels]
    arrays = [dict(np.load(work / f"{label}.npz")) for label in labels]
    old, new = reports
    report = report_header(
        run=args.run,
        proposal=args.proposal,
        work_dir=str(work),
        pythons=list(args.pythons),
        pyfai=[old["pyFAI"], new["pyFAI"]],
        cpu=new["cpu"],
        operator_sha256_equal=(
            old.get("agipd", {}).get("operator_sha256")
            == new.get("agipd", {}).get("operator_sha256")
        ),
        jf_operator_sha256_equal=(
            old.get("jf", {}).get("operator_sha256")
            == new.get("jf", {}).get("operator_sha256")
        ),
        speedups=speedups(old, new),
        arrays=compare_arrays(*arrays),
        old=old,
        new=new,
    )
    write_report(report, args.json, f"bench_pyfai_r{args.run:04d}")
    print(
        f"\npyFAI {old['pyFAI']} -> {new['pyFAI']} on {new['cpu']}"
        f"\n  AGIPD operator sha256 equal: {report['operator_sha256_equal']}"
        f"\n  JF operator sha256 equal:    {report['jf_operator_sha256_equal']}"
    )
    for name, row in report["speedups"].items():
        if row["old_over_new"] is not None:
            print(
                f"  {name:45s} {row['old']:10.3f} -> {row['new']:10.3f}  "
                f"x{row['old_over_new']:.2f}"
            )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--run", type=int, nargs="+", required=True)
    add_job_args(parser, workers=4, time="01:00:00")
    parser.add_argument(
        "--pythons",
        nargs="+",
        default=[],
        help="two interpreters, old then new, each in a venv with its own pyFAI",
    )
    parser.add_argument("--json", type=Path, default=None)
    # internal: one version's measurement, run by the job under each interpreter
    parser.add_argument("--measure", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.measure:
        configure_logging("bench_pyfai")
        args.run = args.run[0]
        return measure(args)
    return dispatch(
        args,
        args.run,
        lambda run: run_one(argparse.Namespace(**vars(args) | {"run": run})),
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
