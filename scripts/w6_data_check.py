#!/usr/bin/env python
"""W6: why a run's frames reach ``DATA_CHECK_FAILED`` (WAXS context file §3 D6).

r0480/jf1 finished 23 519 OK and 473 ``DATA_CHECK_FAILED``, and the output file
cannot say why: ``process_block`` catches ``DataCheckFailed``, keeps the row's
labels and discards the message, so ``max_kev`` stays 0 on exactly the rows
whose value is in question. This reads those frames back from proc and measures
the one distinction the fix depends on.

**The check and the integrator do not use the same mask.**
:func:`analysis.waxs.integrate.frame_data_status` tests the pixels the *static*
``.edf`` keeps, while the integration excludes the union ``((data.mask &
mask_bits) != 0) | static_bad``. So a wild pixel that ``data.mask`` flags fails
the frame without being able to affect a single stored number. That asymmetry is
deliberate — D6 is a tripwire on the arrangement "the ``.edf`` keeps those
pixels out", not a correctness check — but it means a failing frame can mean
three different things:

* **(a)** the extreme pixel is dynamically flagged and merely outside the
  ``.edf``: nothing wild reaches the integrator and the frame is usable as it
  stands. This is what r0423 would predict — all 130 jf1 / 204 jf2 pixels above
  1000 keV there were flagged (§6, corrected 2026-09-13).
* **(b)** it is unflagged and of artifact magnitude (~1e5 keV, ~20 000 photons,
  at or past JUNGFRAU's G2 ceiling): a genuine gap in the ``.edf``.
* **(c)** it is unflagged and modest — the bound is 1000 keV, which is only
  **110 photons** at 9.04 keV, against a kept-region maximum of ~74 keV on the
  non-crystallised r0423. jf1 covers q = 11.5–23.7 nm⁻¹, where a grainy salt
  powder ring puts far more than 110 photons in a pixel. Under (c) the check is
  deleting the frames with the strongest crystalline scattering, and nothing
  about the run would say so.

The decisive number is **how many failing frames carry an extreme or non-finite
value on a pixel that survives the union mask**. Zero means (a) outright. The
control sample answers (c) from the other side: if the OK frames' own maxima
press up against the bound, it is cutting through the signal distribution rather
than sitting in a gap.

    python scripts/w6_data_check.py --run 480 --detector jf1

Needs Maxwell: the proc data, the real PONI and ``.edf``, and the run's output
file from the failed pass. Reads only — it writes nothing but its own JSON, next
to this script or wherever ``--json`` says.

Not unit-tested (CLAUDE.md working rule 7): run it on a node.
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
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common.masks import describe_bits, frame_bad  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.waxs.config import (  # noqa: E402
    DETECTORS,
    MODULE_SHAPE,
    config_for,
)
from analysis.waxs.operator import build_operator  # noqa: E402
from analysis.waxs.plan import open_detector  # noqa: E402

#: Extreme values at least this many times the bound are artifact-scale rather
#: than bright-pixel-scale: the population D6 was written against sits at 1.8e5
#: keV, 180× the 1000 keV bound, and at ~20 000 photons is at or past what a
#: JUNGFRAU pixel can hold. Only used to *label* the verdict; every magnitude is
#: reported raw, so a reader can draw the line somewhere else.
ARTIFACT_RATIO = 50.0

#: Below this multiple of the bound an unflagged extreme value is in the range
#: real scattering can reach — 10× the bound is ~1100 photons.
SIGNAL_RATIO = 10.0

#: Extreme pixels listed per frame in the JSON. The aggregate counts every one;
#: this only caps the per-frame detail so the file stays readable.
MAX_PIXELS_PER_FRAME = 8


def _read_train(det: Any, train_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One train's ``(adc, mask, memoryCell)``, module and train axes dropped."""
    from extra_data import by_id

    selected = det.select_trains(by_id[[int(train_id)]])
    data = np.asarray(selected["data.adc"].ndarray())
    mask = np.asarray(selected["data.mask"].ndarray())
    cells = np.asarray(selected["data.memoryCell"].ndarray()).reshape(-1)
    if data.shape[:2] != (1, 1) or mask.shape != data.shape:
        raise ValueError(
            f"train {train_id}: adc {data.shape} and mask {mask.shape} are not "
            "one module of one train"
        )
    return data[0, 0], mask[0, 0], cells


def stage_configuration(cfg: Any, op: Any, ai: Any) -> dict[str, Any]:
    """What the check is being asked to enforce, in photons as well as keV."""
    static_fraction = float(op.static_bad.mean())
    return {
        "run": cfg.run,
        "detector": cfg.detector,
        "detector_name": cfg.detector_name,
        "poni_file": cfg.poni_file,
        "static_mask_file": cfg.static_mask_file,
        "poni_sha256": op.poni_sha256,
        "static_sha256": op.static_sha256,
        "operator_sha256": op.sha256,
        "config_hash": cfg.config_hash(),
        "max_abs_kev": cfg.max_abs_kev,
        "max_abs_photons": cfg.max_abs_kev / cfg.photon_energy_kev,
        "photon_energy_kev": cfg.photon_energy_kev,
        "mask_bits": f"0x{cfg.mask_bits:08x}",
        "npix": int(op.static_bad.size),
        "n_statically_excluded": int(op.static_bad.sum()),
        "static_excluded_fraction": static_fraction,
        "n_kept": int((~op.static_bad).sum()),
        "q_populated_nm": list(op.q_populated),
        "method": list(op.method),
    }


def stage_ledger(cfg: Any, output_path: Path) -> dict[str, Any]:
    """Which rows failed, and what the file already says about the run."""
    if not output_path.exists():
        raise FileNotFoundError(
            f"{output_path} does not exist; point --output-file at the file the "
            "failed pass wrote, or rerun the pass with allow_incomplete=True"
        )
    with h5py.File(output_path, "r") as handle:
        frames = handle["frames"]
        status = frames["status"][:]
        train_id = frames["trainId"][:]
        cell_id = frames["cellId"][:]
        max_kev = frames["max_kev"][:]
        provenance = handle["provenance"].attrs
        recorded = {
            key: provenance.get(key)
            for key in (
                "config_hash",
                "config_operational_fields",
                "status_summary",
                "bits_present",
                "detector_name",
                "run_checks",
                "block_errors",
            )
        }

    failed = status == FrameStatus.DATA_CHECK_FAILED
    ok = status == FrameStatus.OK
    counts = {
        code.name: int((status == code).sum())
        for code in FrameStatus
        if (status == code).any()
    }
    fail_trains = train_id[failed]
    fail_cells = cell_id[failed]
    per_cell = Counter(int(c) for c in fail_cells.tolist())
    distinct_trains = np.unique(fail_trains)
    all_trains = np.unique(train_id)

    bits = recorded.get("bits_present")
    return {
        "output_file": str(output_path),
        "status_counts": counts,
        "n_frames": int(status.size),
        "n_failed": int(failed.sum()),
        "failed_fraction": float(failed.mean()),
        "n_ok": int(ok.sum()),
        "failed_per_cell": dict(sorted(per_cell.items())),
        "n_distinct_trains_with_a_failure": int(distinct_trains.size),
        "n_trains_in_run": int(all_trains.size),
        "first_failing_train": int(distinct_trains[0])
        if distinct_trains.size
        else None,
        "last_failing_train": int(distinct_trains[-1])
        if distinct_trains.size
        else None,
        # A failure count that tracks position in the run is evaporation or
        # dose; one that is flat is the detector.
        "failures_by_run_decile": _by_decile(all_trains, distinct_trains, fail_trains),
        "max_kev_on_failed_rows_all_zero": bool(
            failed.any() and not np.any(max_kev[failed])
        ),
        "provenance": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in recorded.items()
            if value is not None
        },
        "bits_present_named": describe_bits(int(bits)) if bits is not None else None,
        "config_hash_matches_current": recorded.get("config_hash") == cfg.config_hash(),
        "_failed_train_ids": fail_trains,
        "_failed_cell_ids": fail_cells,
        "_ok_train_ids": train_id[ok],
        "_ok_cell_ids": cell_id[ok],
    }


def _by_decile(
    all_trains: np.ndarray, distinct: np.ndarray, fail_trains: np.ndarray
) -> list[int]:
    """Failing frames per tenth of the run, by train position."""
    if all_trains.size == 0 or fail_trains.size == 0:
        return [0] * 10
    position = np.searchsorted(all_trains, fail_trains)
    decile = np.clip((10 * position) // max(all_trains.size, 1), 0, 9)
    return [int((decile == d).sum()) for d in range(10)]


def _examine_frame(
    x: np.ndarray,
    mask_frame: np.ndarray,
    cfg: Any,
    op: Any,
    q_per_pixel: np.ndarray,
    edge_distance: np.ndarray,
) -> dict[str, Any]:
    """One frame, split by which mask would have removed the offending pixel."""
    flat = np.asarray(x).reshape(-1)
    bad = frame_bad(mask_frame, cfg.mask_bits, op.static_bad)
    kept = ~op.static_bad  # what the D6 check looks at
    reaching = ~bad  # what the integrator actually sees

    absolute = np.abs(flat)
    finite = np.isfinite(flat)
    over = absolute > cfg.max_abs_kev

    extreme_kept = np.flatnonzero(kept & over & finite)
    extreme_reaching = np.flatnonzero(reaching & over & finite)
    order = extreme_kept[np.argsort(-absolute[extreme_kept])]

    pixels = []
    for index in order[:MAX_PIXELS_PER_FRAME].tolist():
        bits = int(mask_frame.reshape(-1)[index])
        pixels.append(
            {
                "pixel": int(index),
                "row": int(index // MODULE_SHAPE[1]),
                "col": int(index % MODULE_SHAPE[1]),
                "kev": float(flat[index]),
                "photons": float(flat[index] / cfg.photon_energy_kev),
                "mask_bits": int(bits),
                "mask_bits_named": describe_bits(bits),
                "flagged_by_data_mask": bool(bits & np.uint32(cfg.mask_bits)),
                "q_nm": float(q_per_pixel[index]),
                "px_to_static_mask": float(edge_distance[index]),
            }
        )

    return {
        "max_abs_kept_kev": float(absolute[kept & finite].max(initial=0.0)),
        "max_abs_reaching_kev": float(absolute[reaching & finite].max(initial=0.0)),
        "n_extreme_kept": int(extreme_kept.size),
        "n_extreme_reaching": int(extreme_reaching.size),
        "n_nonfinite_kept": int((kept & ~finite).sum()),
        "n_nonfinite_reaching": int((reaching & ~finite).sum()),
        "n_bad_pixels": int(bad.sum()),
        "pixels": pixels,
        "_extreme_kept": extreme_kept,
        "_extreme_reaching": extreme_reaching,
        "_values": flat[extreme_kept],
    }


def stage_frames(
    cfg: Any,
    op: Any,
    ai: Any,
    det: Any,
    ledger: dict[str, Any],
    *,
    max_frames: int,
) -> dict[str, Any]:
    """Re-read the failing frames and locate the value that failed each one."""
    q_per_pixel = np.asarray(
        ai.array_from_unit(MODULE_SHAPE, typ="center", unit=cfg.unit, scale=True)
    ).reshape(-1)
    edge_distance = _edge_distance(op)

    trains = ledger["_failed_train_ids"]
    cells = ledger["_failed_cell_ids"]
    by_train: dict[int, list[int]] = {}
    for train, cell in zip(trains.tolist(), cells.tolist(), strict=True):
        by_train.setdefault(int(train), []).append(int(cell))

    examined: list[dict[str, Any]] = []
    recurrence: Counter[int] = Counter()
    values: list[float] = []
    reaching_values: list[float] = []
    n_examined = 0
    started = time.perf_counter()

    for train in sorted(by_train):
        if max_frames and n_examined >= max_frames:
            break
        data, mask, cell_ids = _read_train(det, train)
        positions = {int(c): i for i, c in enumerate(cell_ids.tolist())}
        for cell in sorted(by_train[train]):
            if max_frames and n_examined >= max_frames:
                break
            position = positions.get(cell)
            if position is None:
                examined.append(
                    {
                        "trainId": train,
                        "cellId": cell,
                        "error": "cell not in this train",
                    }
                )
                continue
            frame = _examine_frame(
                data[position], mask[position], cfg, op, q_per_pixel, edge_distance
            )
            recurrence.update(frame.pop("_extreme_kept").tolist())
            reaching_values.extend(
                np.abs(
                    np.asarray(data[position]).reshape(-1)[
                        frame.pop("_extreme_reaching")
                    ]
                ).tolist()
            )
            values.extend(np.abs(frame.pop("_values")).tolist())
            examined.append({"trainId": train, "cellId": cell, **frame})
            n_examined += 1
        if n_examined % 50 == 0:
            print(f"  {n_examined} frames examined", flush=True)

    usable = [
        frame
        for frame in examined
        if "error" not in frame
        and frame["n_extreme_reaching"] == 0
        and frame["n_nonfinite_reaching"] == 0
    ]
    magnitudes = np.asarray(values, dtype=np.float64)
    reaching = np.asarray(reaching_values, dtype=np.float64)

    return {
        "n_failing_frames_examined": n_examined,
        "n_failing_frames_in_ledger": int(ledger["n_failed"]),
        "read_s": time.perf_counter() - started,
        # The number Phase 2 turns on: frames whose offending pixel the union
        # mask already removes are frames D6 failed over nothing.
        "n_frames_clean_under_the_union_mask": len(usable),
        "fraction_clean_under_the_union_mask": (
            len(usable) / n_examined if n_examined else None
        ),
        "n_extreme_pixels_total": int(magnitudes.size),
        "n_extreme_pixels_reaching_the_integrator": int(reaching.size),
        "n_distinct_extreme_pixels": len(recurrence),
        "most_recurrent_pixels": [
            {"pixel": int(pixel), "n_frames": int(count)}
            for pixel, count in recurrence.most_common(12)
        ],
        "extreme_kev_percentiles": _percentiles(magnitudes),
        "extreme_reaching_kev_percentiles": _percentiles(reaching),
        "extreme_kev_max": float(magnitudes.max(initial=0.0)),
        "extreme_reaching_kev_max": float(reaching.max(initial=0.0)),
        "frames": examined,
    }


def _edge_distance(op: Any) -> np.ndarray:
    """Distance in pixels from each kept pixel to the nearest masked one.

    A wild pixel hard against the ``.edf`` boundary is one the mask nearly
    caught; one in open kept territory is a different animal.
    """
    from scipy.ndimage import distance_transform_edt

    kept_2d = ~op.static_bad.reshape(MODULE_SHAPE)
    return np.asarray(distance_transform_edt(kept_2d)).reshape(-1)


def stage_control(
    cfg: Any, op: Any, det: Any, ledger: dict[str, Any], *, n_frames: int
) -> dict[str, Any]:
    """What the *passing* frames' maxima look like, which answers (c).

    If the OK frames' own ``max |x|`` crowds the bound, 1000 keV is a cut
    through the signal distribution and not a gap in it.
    """
    trains = ledger["_ok_train_ids"]
    cells = ledger["_ok_cell_ids"]
    if trains.size == 0:
        return {"n_frames": 0}

    take = np.unique(np.linspace(0, trains.size - 1, num=n_frames).astype(int))
    by_train: dict[int, list[int]] = {}
    for index in take.tolist():
        by_train.setdefault(int(trains[index]), []).append(int(cells[index]))

    kept_max: list[float] = []
    reaching_max: list[float] = []
    started = time.perf_counter()
    for train in sorted(by_train):
        data, mask, cell_ids = _read_train(det, train)
        positions = {int(c): i for i, c in enumerate(cell_ids.tolist())}
        for cell in sorted(by_train[train]):
            position = positions.get(cell)
            if position is None:
                continue
            flat = np.asarray(data[position]).reshape(-1)
            bad = frame_bad(mask[position], cfg.mask_bits, op.static_bad)
            absolute = np.abs(flat)
            finite = np.isfinite(flat)
            kept_max.append(float(absolute[(~op.static_bad) & finite].max(initial=0.0)))
            reaching_max.append(float(absolute[(~bad) & finite].max(initial=0.0)))

    kept = np.asarray(kept_max)
    reaching = np.asarray(reaching_max)
    return {
        "n_frames": int(kept.size),
        "read_s": time.perf_counter() - started,
        "max_abs_kept_kev_percentiles": _percentiles(kept),
        "max_abs_reaching_kev_percentiles": _percentiles(reaching),
        "headroom_to_bound": (
            float(cfg.max_abs_kev / reaching.max())
            if reaching.size and reaching.max()
            else None
        ),
        "n_within_10x_of_the_bound": int((reaching > cfg.max_abs_kev / 10).sum()),
    }


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    keys = (0, 50, 90, 99, 100)
    return {f"p{key}": float(np.percentile(values, key)) for key in keys} | {
        "n": int(values.size)
    }


def verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Name the cause, or say why the evidence does not settle it."""
    cfg_stage = report["configuration"]
    frames = report["frames"]
    control = report.get("control", {})
    bound = float(cfg_stage["max_abs_kev"])

    examined = frames["n_failing_frames_examined"]
    if examined == 0:
        return {"cause": "none", "why": "no failing frame was examined"}

    reaching_pixels = frames["n_extreme_pixels_reaching_the_integrator"]
    nonfinite_reaching = sum(
        frame.get("n_nonfinite_reaching", 0) for frame in frames["frames"]
    )
    reaching_max = frames["extreme_reaching_kev_max"]
    control_p100 = (control.get("max_abs_reaching_kev_percentiles") or {}).get(
        "p100", 0
    )

    if reaching_pixels == 0 and nonfinite_reaching == 0:
        return {
            "cause": "a",
            "why": (
                f"none of the {frames['n_extreme_pixels_total']} extreme pixels in "
                f"{examined} failing frames survives the union mask, so no failing "
                "frame has a wild value reaching the integrator: D6 is failing "
                "frames over pixels that data.mask already removes"
            ),
            "implies": (
                "Phase 2 option (a): fail a frame only on a pixel the union mask "
                "keeps, and count wild values under a mask as a diagnostic"
            ),
        }

    if reaching_max >= ARTIFACT_RATIO * bound:
        return {
            "cause": "b",
            "why": (
                f"{reaching_pixels} extreme pixels reach the integrator, up to "
                f"{reaching_max:.3g} keV — {reaching_max / bound:.0f}x the bound, "
                "artifact scale, and neither mask removes them"
            ),
            "implies": "Phase 2 option (b): the .edf has a genuine gap; extend it",
        }

    if reaching_max <= SIGNAL_RATIO * bound:
        return {
            "cause": "c",
            "why": (
                f"the extreme values reaching the integrator top out at "
                f"{reaching_max:.3g} keV "
                f"({reaching_max / cfg_stage['photon_energy_kev']:.0f} photons), "
                "within reach of real scattering, and the passing frames reach "
                f"{control_p100:.3g} keV — the bound sits inside the signal "
                "distribution, not in a gap"
            ),
            "implies": (
                "Phase 2 option (c): re-derive max_abs_kev from the measured gap; "
                "the current 1000 keV has no traceable source"
            ),
        }

    return {
        "cause": "mixed",
        "why": (
            f"{reaching_pixels} extreme pixels reach the integrator, up to "
            f"{reaching_max:.3g} keV ({reaching_max / bound:.1f}x the bound) — "
            "between signal scale and artifact scale; read frames[] before choosing"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--detector", choices=DETECTORS, required=True)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="the pass's .h5; defaults to cfg.output_file",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="cap the failing frames re-read; 0 (default) reads all of them",
    )
    parser.add_argument(
        "--control-frames",
        type=int,
        default=200,
        help="OK frames sampled evenly across the run, for the (c) comparison",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from extra_data import open_run

    cfg = config_for(args.proposal, args.run, args.detector)
    output_path = args.output_file or Path(cfg.output_file)

    report: dict[str, Any] = {
        "script": Path(__file__).name,
        "when": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "argv": vars(args),
    }

    try:
        op, ai = build_operator(cfg)
        report["configuration"] = stage_configuration(cfg, op, ai)
        _print_configuration(report["configuration"])

        report["ledger"] = stage_ledger(cfg, output_path)
        _print_ledger(report["ledger"])

        dc = open_run(args.proposal, args.run, data="proc")
        det = open_detector(cfg, dc)

        print("\n=== failing frames ===", flush=True)
        report["frames"] = stage_frames(
            cfg, op, ai, det, report["ledger"], max_frames=args.max_frames
        )
        _print_frames(report["frames"])

        print("\n=== control sample ===", flush=True)
        report["control"] = stage_control(
            cfg, op, det, report["ledger"], n_frames=args.control_frames
        )
        _print_control(report["control"], cfg)

        report["verdict"] = verdict(report)
        report["passed"] = True
    except Exception as error:  # noqa: BLE001 - the JSON is the deliverable
        formatted = traceback.format_exc()
        print(f"FAILED: {error!r}\n{formatted}", flush=True)
        report["passed"] = False
        report["error"] = repr(error)
        report["traceback"] = formatted

    # The private arrays exist only to pass rows between stages.
    for key in list(report.get("ledger", {})):
        if key.startswith("_"):
            del report["ledger"][key]

    print("\n=== verdict ===")
    if "verdict" in report:
        print(f"  cause {report['verdict']['cause']}: {report['verdict']['why']}")
        if "implies" in report["verdict"]:
            print(f"  -> {report['verdict']['implies']}")

    path = args.json or Path(__file__).with_name(
        f"w6_data_check_r{args.run:04d}_{args.detector}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {path}")
    return 0 if report["passed"] else 1


def _print_configuration(stage: dict[str, Any]) -> None:
    print("=== configuration ===")
    print(f"  run           r{stage['run']:04d} {stage['detector']}")
    print(
        f"  bound         {stage['max_abs_kev']} keV = "
        f"{stage['max_abs_photons']:.1f} photons at {stage['photon_energy_kev']} keV"
    )
    print(
        f"  static mask   {100 * stage['static_excluded_fraction']:.1f} % excluded, "
        f"{stage['n_kept']} px kept"
    )
    print(f"  q populated   {stage['q_populated_nm']} nm^-1")


def _print_ledger(stage: dict[str, Any]) -> None:
    print("\n=== ledger ===")
    print(f"  file          {stage['output_file']}")
    print(f"  status        {stage['status_counts']}")
    print(
        f"  failed        {stage['n_failed']} of {stage['n_frames']} "
        f"({100 * stage['failed_fraction']:.2f} %) over "
        f"{stage['n_distinct_trains_with_a_failure']} of "
        f"{stage['n_trains_in_run']} trains"
    )
    print(f"  per cell      {stage['failed_per_cell']}")
    print(f"  by decile     {stage['failures_by_run_decile']}")
    print(f"  bits present  {stage['bits_present_named']}")
    if not stage["config_hash_matches_current"]:
        print("  WARNING: the file's config_hash is not this config's")


def _print_frames(stage: dict[str, Any]) -> None:
    print(
        f"  examined      {stage['n_failing_frames_examined']} of "
        f"{stage['n_failing_frames_in_ledger']} in {stage['read_s']:.1f} s"
    )
    print(
        f"  clean under the union mask: "
        f"{stage['n_frames_clean_under_the_union_mask']} "
        f"({stage['fraction_clean_under_the_union_mask']})"
    )
    print(
        f"  extreme px    {stage['n_extreme_pixels_total']} total, "
        f"{stage['n_extreme_pixels_reaching_the_integrator']} reaching the "
        f"integrator, {stage['n_distinct_extreme_pixels']} distinct"
    )
    print(f"  |x| kept      {stage['extreme_kev_percentiles']}")
    print(f"  |x| reaching  {stage['extreme_reaching_kev_percentiles']}")
    for entry in stage["most_recurrent_pixels"][:5]:
        print(f"    px {entry['pixel']:>8} in {entry['n_frames']} frames")


def _print_control(stage: dict[str, Any], cfg: Any) -> None:
    if not stage.get("n_frames"):
        print("  no OK frames to sample")
        return
    print(f"  sampled       {stage['n_frames']} OK frames in {stage['read_s']:.1f} s")
    print(f"  |x| kept      {stage['max_abs_kept_kev_percentiles']}")
    print(f"  |x| reaching  {stage['max_abs_reaching_kev_percentiles']}")
    print(
        f"  headroom      bound / brightest passing frame = "
        f"{stage['headroom_to_bound']}"
    )
    print(f"  within 10x of the bound: {stage['n_within_10x_of_the_bound']} frames")


if __name__ == "__main__":
    raise SystemExit(main())
