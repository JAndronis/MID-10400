#!/usr/bin/env python
"""W1 fact-finding for the JUNGFRAU WAXS pass (context file §7, W1).

Answers, per run and per detector, the questions the spec leaves open after O1:

* **O4 — is the lit-cell split run-invariant?** The set was measured on one train
  of r0423. Run this over r0423 *and* a crystallised run (r0426) and compare.
* **Are the geometry and mask files where the config expects them?** Existence
  plus sha256, so a later result can be tied to the files it used.
* **Is ``data.mask`` train-invariant?** It is uint32 and the same size as the
  data, so reading it doubles the pass's I/O. If the dynamic bits never change
  between trains, one read per run could replace three thousand.
* **Does an ROI over the lit cells actually save I/O?** ``ndarray(roi=...)``
  slices the array axis; whether that reads less depends on the HDF5 chunk
  layout, which is why this times both instead of assuming.

Nothing here writes to the pass's output; it only reads. The verdict goes to
JSON beside this script, or wherever ``--json`` says.

    python scripts/w1_facts.py --runs 423 426 --detectors jf1 jf2

Needs Maxwell: the proc data, and the real PONI and ``.edf`` files.
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
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from analysis.common.cpu import file_sha256  # noqa: E402
from analysis.common.masks import describe_bits  # noqa: E402
from analysis.common.plan import evenly_spaced  # noqa: E402
from analysis.waxs.cells import CellAccumulator  # noqa: E402
from analysis.waxs.config import (  # noqa: E402
    DETECTORS,
    EXPECTED_BITS,
    config_for,
)
from analysis.waxs.operator import build_operator  # noqa: E402
from analysis.waxs.plan import open_detector  # noqa: E402


def _read_train(det: Any, train_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from extra_data import by_id

    selected = det.select_trains(by_id[[int(train_id)]])
    data = np.asarray(selected["data.adc"].ndarray())[0, 0]
    mask = np.asarray(selected["data.mask"].ndarray())[0, 0]
    cells = np.asarray(selected["data.memoryCell"].ndarray()).reshape(-1)
    return data, mask, cells


def stage_files(cfg: Any) -> dict[str, Any]:
    """Are the PONI and the static mask where the config expects them?"""
    found = {}
    for name, path in cfg.input_files.items():
        entry: dict[str, Any] = {"path": path}
        if path and Path(path).exists():
            entry.update(
                exists=True, sha256=file_sha256(path), bytes=Path(path).stat().st_size
            )
        else:
            entry.update(exists=False)
        found[name] = entry
    return {"passed": all(e["exists"] for e in found.values()), "files": found}


def stage_sources(cfg: Any, dc: Any, det: Any) -> dict[str, Any]:
    """The source names, the entry shape and the legacy alias."""
    counts = det.frame_counts
    legacy = {
        src: canonical
        for src, canonical in dc.legacy_sources.items()
        if cfg.detector_name in src
    }
    return {
        "passed": len(det.source_to_modno) == 1,
        "detector_name": det.detector_name,
        "sources": sorted(det.source_to_modno),
        "source_to_modno": {k: int(v) for k, v in det.source_to_modno.items()},
        "n_modules": int(det.n_modules),
        "frames_per_entry": int(det._frames_per_entry),
        "entries_per_train": sorted({int(c) for c in counts}),
        "n_trains": int(len(det.train_ids)),
        "legacy_aliases": legacy,
    }


def stage_cells(cfg: Any, det: Any, op: Any, n_trains: int) -> dict[str, Any]:
    """O4: the lit-cell split and the readout noise, over sampled trains."""
    sampled = evenly_spaced(np.asarray(det.train_ids, dtype=np.uint64), n_trains)
    accumulator = CellAccumulator(cfg, op.static_bad)
    for train_id in sampled.tolist():
        accumulator.update(*_read_train(det, train_id))
    result = accumulator.finalise()

    bits = {b for b in range(32) if accumulator.bits_present >> b & 1}
    unexpected = bits - set(EXPECTED_BITS)
    matches = result.lit == tuple(cfg.expected_lit_cells)
    return {
        "passed": matches and not unexpected,
        "sampled_trains": [int(t) for t in sampled],
        "lit": list(result.lit),
        "dark": list(result.dark),
        "expected_lit": list(cfg.expected_lit_cells),
        "lit_matches_expected": matches,
        "lit_fraction": {
            int(c): float(f)
            for c, f in zip(result.cells, result.lit_fraction, strict=True)
        },
        "read_noise_kev": result.read_noise_kev,
        "read_noise_samples": result.read_noise_samples,
        "bits_present": sorted(bits),
        "bits_named": describe_bits(accumulator.bits_present),
        "unexpected_bits": sorted(unexpected),
    }


def stage_mask_variability(cfg: Any, det: Any, n_trains: int) -> dict[str, Any]:
    """Is ``data.mask`` the same in every train, or does it really vary?

    If it is invariant, the pass could read it once per run instead of once per
    train and halve its I/O. Reported, not acted on: the decision belongs to a
    benchmark on the whole run, not to a handful of sampled trains.
    """
    sampled = evenly_spaced(np.asarray(det.train_ids, dtype=np.uint64), n_trains)
    reference = None
    ever_differs = None
    per_train = []
    for train_id in sampled.tolist():
        _, mask, _ = _read_train(det, train_id)
        if reference is None:
            reference = mask
            ever_differs = np.zeros(mask.shape, dtype=bool)
            continue
        differs = mask != reference
        ever_differs |= differs
        per_train.append({"train": int(train_id), "n_differing_px": int(differs.sum())})
    total = int(ever_differs.sum()) if ever_differs is not None else 0
    size = int(reference.size) if reference is not None else 0
    return {
        "passed": None,  # a measurement, not a gate
        "n_trains_compared": len(sampled),
        "reference_train": int(sampled[0]),
        "identical_in_every_sampled_train": total == 0,
        "n_px_that_ever_differ": total,
        "fraction_that_ever_differ": (total / size) if size else 0.0,
        "per_train": per_train,
        "note": (
            "if this is 0 over many trains, one mask read per run could replace "
            "one per train; confirm over the whole run before relying on it"
        ),
    }


def stage_roi(cfg: Any, det: Any, lit: tuple[int, ...], repeats: int) -> dict[str, Any]:
    """Does restricting the read to the lit cells actually save anything?"""
    from extra_data import by_id

    train_id = int(det.train_ids[len(det.train_ids) // 2])
    selected = det.select_trains(by_id[[train_id]])
    adc = selected["data.adc"]

    contiguous = tuple(lit) == tuple(range(min(lit), max(lit) + 1))
    # A *tuple* of slices, not a bare slice: EXtra-data does
    # `source_sel=(chunk_slice,) + roi` and `np.index_exp[:] + roi`, both of
    # which need a tuple. `np.s_[0:8]` alone raises TypeError.
    roi = (np.s_[min(lit) : max(lit) + 1],) if contiguous else None

    chunks = {}
    for key in ("data.adc", "data.mask"):
        keydata = selected[key]
        first = next(iter(keydata.modno_to_keydata.values()))
        dataset = first._data_chunks[0].dataset if first._data_chunks else None
        chunks[key] = {
            "chunks": list(dataset.chunks)
            if dataset is not None and dataset.chunks
            else None,
            "compression": (dataset.compression if dataset is not None else None),
            "dtype": str(keydata.dtype),
            "entry_shape": list(first.entry_shape),
        }

    def timed(fn):
        best = float("inf")
        for _ in range(repeats):
            started = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - started)
        return best

    full = timed(lambda: adc.ndarray())
    windowed = timed(lambda: adc.ndarray(roi=roi)) if roi is not None else None
    return {
        "passed": None,
        "train": train_id,
        "lit_cells_contiguous": contiguous,
        "roi": [int(min(lit)), int(max(lit)) + 1] if roi is not None else None,
        "full_read_s": full,
        "roi_read_s": windowed,
        "speedup": (full / windowed) if windowed else None,
        "datasets": chunks,
        "note": (
            "the ROI slices the array axis, while the lit set is defined by "
            "data.memoryCell values; only equal when memoryCell == arange"
        ),
    }


def stage_values(cfg: Any, det: Any, op: Any) -> dict[str, Any]:
    """§3 D6: extreme pixels, and whether either mask actually catches them."""
    train_id = int(det.train_ids[0])
    data, mask, _ = _read_train(det, train_id)
    extreme = np.abs(data) > cfg.max_abs_kev
    unflagged = extreme & (mask == 0)
    covered = unflagged & op.static_bad.reshape(1, *op.shape)
    kept = (mask == 0) & ~op.static_bad.reshape(1, *op.shape)
    return {
        "passed": None,
        "train": train_id,
        "max_abs_kev_bound": cfg.max_abs_kev,
        "n_extreme": int(extreme.sum()),
        "n_extreme_unflagged_by_data_mask": int(unflagged.sum()),
        "n_extreme_unflagged_but_under_edf": int(covered.sum()),
        "n_extreme_reaching_the_integrator": int(
            (unflagged & ~covered.astype(bool)).sum()
        ),
        "max_abs_kev_in_kept_region": float(np.abs(data[kept]).max())
        if kept.any()
        else 0.0,
        "dtype": str(data.dtype),
        "negative_fraction_kept": float((data[kept] < 0).mean()) if kept.any() else 0.0,
        "exactly_zero_fraction": float((data == 0).mean()),
    }


def inspect(proposal: int, run: int, detector: str, args) -> dict[str, Any]:
    from extra_data import open_run

    report: dict[str, Any] = {"run": run, "detector": detector}
    cfg = config_for(proposal, run, detector, npt=args.npt)
    report["config_hash"] = (
        cfg.config_hash()
        if all(p and Path(p).exists() for p in cfg.input_files.values())
        else None
    )

    report["files"] = stage_files(cfg)
    if not report["files"]["passed"]:
        report["passed"] = False
        return report

    op, ai = build_operator(cfg)
    report["geometry"] = {
        "passed": True,
        "operator_sha256": op.sha256,
        "dist_m": op.dist_m,
        "wavelength_m": op.wavelength_m,
        "photon_energy_kev": cfg.photon_energy_kev,
        "q_range_nm": list(op.q_populated),
        "static_excluded_fraction": float(op.static_bad.mean()),
    }

    dc = open_run(proposal, run, data="proc")
    det = open_detector(cfg, dc)
    report["sources"] = stage_sources(cfg, dc, det)
    report["cells"] = stage_cells(cfg, det, op, args.trains)
    lit = tuple(report["cells"]["lit"]) or tuple(cfg.expected_lit_cells)
    report["mask_variability"] = stage_mask_variability(cfg, det, args.trains)
    report["roi"] = stage_roi(cfg, det, lit, args.repeats)
    report["values"] = stage_values(cfg, det, op)

    verdicts = [
        section.get("passed")
        for section in report.values()
        if isinstance(section, dict) and "passed" in section
    ]
    report["passed"] = all(v is not False for v in verdicts)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--runs", type=int, nargs="+", default=[423, 426])
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    parser.add_argument("--trains", type=int, default=8, help="trains to sample")
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats")
    parser.add_argument("--npt", type=int, default=500)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "proposal": args.proposal,
        "runs": args.runs,
        "detectors": args.detectors,
        "results": [],
    }

    for run in args.runs:
        for detector in args.detectors:
            print(f"\n=== r{run:04d} {detector} ===", flush=True)
            try:
                result = inspect(args.proposal, run, detector, args)
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                print(f"  FAILED: {error!r}")
                result = {
                    "run": run,
                    "detector": detector,
                    "passed": False,
                    "error": repr(error),
                }
            report["results"].append(result)
            _print(result)

    # O4: does the lit set agree across every run and detector that was read?
    sets = {
        (r["run"], r["detector"]): tuple(r["cells"]["lit"])
        for r in report["results"]
        if "cells" in r
    }
    report["o4_lit_cells_invariant"] = len(set(sets.values())) <= 1
    report["o4_lit_cells_by_run"] = {
        f"r{k[0]:04d}/{k[1]}": list(v) for k, v in sets.items()
    }
    report["passed"] = all(r.get("passed") is not False for r in report["results"])

    print("\n=== O4 ===")
    for key, value in report["o4_lit_cells_by_run"].items():
        print(f"  {key}: {value}")
    print(
        f"  lit-cell set invariant across everything read: "
        f"{report['o4_lit_cells_invariant']}"
    )

    path = args.json or Path(__file__).with_name(
        f"w1_facts_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {path}")
    return 0 if report["passed"] else 1


def _print(result: dict[str, Any]) -> None:
    if "error" in result:
        return
    cells = result.get("cells", {})
    sources = result.get("sources", {})
    geometry = result.get("geometry", {})
    roi = result.get("roi", {})
    variability = result.get("mask_variability", {})
    values = result.get("values", {})
    print(f"  source        {sources.get('sources')}")
    print(
        f"  trains        {sources.get('n_trains')}, entries/train "
        f"{sources.get('entries_per_train')}, cells/entry "
        f"{sources.get('frames_per_entry')}"
    )
    print(
        f"  q populated   {geometry.get('q_range_nm')} nm^-1, static mask "
        f"{100 * geometry.get('static_excluded_fraction', 0):.1f} %"
    )
    print(
        f"  lit cells     {cells.get('lit')} (expected {cells.get('expected_lit')}) "
        f"-> {'MATCH' if cells.get('lit_matches_expected') else 'DIFFERENT'}"
    )
    print(f"  read noise    {cells.get('read_noise_kev')} keV")
    print(
        f"  bits          {cells.get('bits_present')} unexpected "
        f"{cells.get('unexpected_bits')}"
    )
    print(
        f"  data.mask     identical across sampled trains: "
        f"{variability.get('identical_in_every_sampled_train')} "
        f"({variability.get('n_px_that_ever_differ')} px ever differ)"
    )
    print(
        f"  roi read      full {roi.get('full_read_s')}, roi {roi.get('roi_read_s')}, "
        f"speedup {roi.get('speedup')}"
    )
    print(
        f"  extreme px    {values.get('n_extreme')}, unflagged "
        f"{values.get('n_extreme_unflagged_by_data_mask')}, reaching the "
        f"integrator {values.get('n_extreme_reaching_the_integrator')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
