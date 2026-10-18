#!/usr/bin/env python
"""P0 fact-finding for the hit-train XPCS pass (``context/xpcs-hits.md``).

Answers, from a handful of trains of one run:

* **B0, sequence boundaries.** Do the 16 AGIPD modules cut a run into the same
  proc sequence files, train for train? ``Kernel.ES`` chains a train to its
  predecessor per sequence file, so a module that cuts elsewhere would give a
  different chain.
* **B0, the mask.** How much of the envelope does ``image.mask`` flag per frame,
  per memory cell and per train; what share of it is the same in every cell;
  how many envelope pixels an AND over a train's frames throws away; how short
  the per-frame difference from the train's majority mask is.
* **B0, the pulse pattern.** Pulse ids and their step in proc, the same from
  ``XrayPulses`` on raw, the period they imply (4.5 MHz base clock), and whether
  the frame count and the pattern are constant over the run's OK trains.
* **B1, bytes per hit train.** The store's layout and the alternatives for its
  two large parts (the photon events, the mask flags), each written to HDF5 with
  gzip, group per train, and measured by file size.

The trains are spread evenly over the run's OK trains in an existing
``agipd_saxs`` frame table (a stand-in for the hit set, which only the
classifier knows). Reads only; the candidate files go to a temporary directory.

    python scripts/xpcs_hits_facts.py --run 423 -w 2          # one SLURM job per run
    python scripts/xpcs_hits_facts.py --run 423 --local --n-trains 10

Needs Maxwell: proc data, the geometry and the pixel mask.
"""

from __future__ import annotations

# Before numpy, pyFAI or EXtra-data import anything that builds a thread pool.
from _common import add_job_args, dispatch, report_header, write_report  # noqa: E402

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common.plan import evenly_spaced  # noqa: E402
from analysis.common.slurm import PROPOSAL_ROOT  # noqa: E402
from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.saxs.masks import build_static_bad  # noqa: E402
from analysis.xpcs.config import XpcsConfig  # noqa: E402
from analysis.xpcs.extract import encode_train, train_sequences  # noqa: E402
from analysis.xpcs.rings import (  # noqa: E402
    build_envelope,
    build_rings,
    pixel_map,
)
from analysis.xpcs.store import FlagBase, delta_encode  # noqa: E402

log = logging.getLogger("scripts.xpcs_hits_facts")

#: The base clock pulse ids count in; a step of 2 is 444 ns.
BASE_CLOCK_HZ = 4.5e6
SCRATCH = PROPOSAL_ROOT / "scratch"
GZIP = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
PROJECTION_TRAINS = (1_000, 10_000, 30_000)


def frame_table_trains(cfg: AgipdSaxsConfig, given: Path | None) -> tuple[Path, dict]:
    """OK trains and their frame counts from an ``agipd_saxs`` file."""
    path = given or cfg.output_file
    if not path.exists():
        candidates = sorted(path.parent.glob("agipd_saxs*.h5"))
        if not candidates:
            raise FileNotFoundError(f"no agipd_saxs frame table under {path.parent}")
        path = candidates[-1]
    with h5py.File(path, "r") as handle:
        ids = handle["trains/trainId"][:]
        count = handle["trains/count"][:]
        status = handle["trains/status"][:]
    ok = status == 0
    return path, {"all": ids, "ok": ids[ok], "count": count[ok]}


# ── B0: sequence files ────────────────────────────────────────────────────────


def stage_sequences(run_dir: Path, run: int) -> dict[str, Any]:
    """Per module and sequence file, the first and last train id and the count."""
    pattern = re.compile(rf"CORR-R{run:04d}-AGIPD(\d\d)-S(\d{{5}})\.h5")
    table: dict[int, dict[int, tuple[int, int, int]]] = {}
    for path in sorted(run_dir.glob("CORR-*.h5")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        module, sequence = int(match[1]), int(match[2])
        with h5py.File(path, "r") as handle:
            ids = handle["INDEX/trainId"][:]
        ids = ids[ids > 0]
        table.setdefault(module, {})[sequence] = (
            int(ids.min()) if ids.size else 0,
            int(ids.max()) if ids.size else 0,
            int(ids.size),
        )
    reference = table[min(table)]
    disagreeing = {
        f"module {module}": {
            f"S{sequence:05d}": {"module": spans.get(sequence), "reference": span}
            for sequence, span in reference.items()
            if spans.get(sequence) != span
        }
        | (
            {"sequences_only_here": sorted(set(spans) - set(reference))}
            if set(spans) - set(reference)
            else {}
        )
        for module, spans in table.items()
        if spans != reference
    }
    return {
        "n_modules": len(table),
        "n_sequences_per_module": {m: len(s) for m, s in table.items()},
        "sequences": {
            f"S{s:05d}": {"first": a, "last": b, "n_trains": n}
            for s, (a, b, n) in reference.items()
        },
        "boundaries_agree_across_modules": not disagreeing,
        "disagreements": disagreeing,
    }


# ── B0: pulses ────────────────────────────────────────────────────────────────


def stage_xray_pulses(cfg: AgipdSaxsConfig) -> dict[str, Any]:
    """The pulse period the control data give (raw), for comparison with proc."""
    try:
        from extra.components import XrayPulses
        from extra_data import open_run

        pulses = XrayPulses(open_run(cfg.proposal, cfg.run, data="raw"))
        periods = np.asarray(pulses.pulse_periods(labelled=False))
        counts = np.asarray(pulses.pulse_counts(labelled=False))
        values, n = np.unique(periods, return_counts=True)
        counts_values, counts_n = np.unique(counts, return_counts=True)
        return {
            "pulse_periods_in_base_clocks": dict(
                zip(values.tolist(), n.tolist(), strict=True)
            ),
            "pulses_per_train": dict(
                zip(counts_values.tolist(), counts_n.tolist(), strict=True)
            ),
            "constant_pattern": bool(pulses.is_constant_pattern()),
            "bunch_repetition_rate_hz": float(pulses.bunch_repetition_rate),
        }
    except Exception as error:  # noqa: BLE001 - recorded, not fatal
        return {"unavailable": repr(error)}


# ── the read ──────────────────────────────────────────────────────────────────


class Tally:
    """What the read loop accumulates over the trains."""

    def __init__(self, n_env: int) -> None:
        self.n_env = n_env
        self.cell_votes: dict[int, np.ndarray] = {}
        self.cell_trains: dict[int, int] = {}
        self.trains: list[Any] = []
        #: per train: cell ids and each frame's complete flagged set
        self.flag_sets: list[tuple[np.ndarray, list[np.ndarray]]] = []
        self.rows: list[dict[str, Any]] = []
        self.patterns: set[bytes] = set()
        self.sequences: list[set[int]] = []
        self.timings = {"read_data": 0.0, "read_mask": 0.0, "encode": 0.0}
        self.dtype: set[str] = set()
        self.max_count = 0
        self.bits = 0


def stage_read(
    cfg: XpcsConfig, train_ids: np.ndarray, envelope: np.ndarray, ring_good: np.ndarray
) -> Tally:
    """Read and encode the trains with no flag base, so every flag is listed."""
    from extra_data import by_id, open_run
    from extra_data.components import AGIPD1M

    dc = open_run(cfg.proposal, cfg.run, data="proc")
    det = AGIPD1M(dc, detector_name=cfg.saxs.detector_name, min_modules=16)
    tally = Tally(envelope.size)

    for train_id in train_ids.tolist():
        selected = det.select_trains(by_id[[train_id]])
        key = selected["image.data"]
        started = time.perf_counter()
        data = key.ndarray(decompress_threads=1)
        tally.timings["read_data"] += time.perf_counter() - started
        started = time.perf_counter()
        mask = selected["image.mask"].ndarray(decompress_threads=1)
        tally.timings["read_mask"] += time.perf_counter() - started

        pulse_ids = key.pulse_id_coordinates()
        cell_ids = key.cell_id_coordinates()
        sequences = set(train_sequences(key).values())
        tally.sequences.append(sequences)
        tally.patterns.add(pulse_ids.astype(np.uint32).tobytes())
        tally.dtype.add(str(data.dtype))
        tally.bits |= int(np.bitwise_or.reduce(mask, axis=None))
        tally.max_count = max(tally.max_count, int(data.max()))

        started = time.perf_counter()
        photons = encode_train(
            data,
            mask,
            envelope,
            cfg.saxs.mask_bits,
            FlagBase.empty(),
            train_id=train_id,
            sequence=min(sequences),
            pulse_ids=pulse_ids,
            cell_ids=cell_ids,
        )
        tally.timings["encode"] += time.perf_counter() - started

        flagged = [photons.flagged_frame(f) for f in range(photons.n_frames)]
        for cell, pixels in zip(cell_ids.tolist(), flagged, strict=True):
            votes = tally.cell_votes.setdefault(
                cell, np.zeros(envelope.size, dtype=np.uint16)
            )
            votes[pixels] += 1
            tally.cell_trains[cell] = tally.cell_trains.get(cell, 0) + 1
        tally.flag_sets.append((cell_ids.copy(), flagged))
        any_frame = photons.flagged_any(envelope.size)
        tally.rows.append(
            {
                "train_id": train_id,
                "n_frames": photons.n_frames,
                "nnz": int(photons.pixel.size),
                "mean_flagged_per_frame": float(np.mean([f.size for f in flagged])),
                "any_frame_flagged": int(any_frame.sum()),
                "any_frame_flagged_in_good_pixels": int((any_frame & ring_good).sum()),
                "max_count": int(photons.count.max()) if photons.count.size else 0,
            }
        )
        tally.trains.append(photons)
        log.info("train %d: %d events", train_id, photons.pixel.size)
    return tally


def majority_base(flag_set: list[np.ndarray], n_env: int) -> np.ndarray:
    """Pixels flagged in more than half of one train's frames."""
    votes = np.zeros(n_env, dtype=np.uint16)
    for pixels in flag_set:
        votes[pixels] += 1
    return np.flatnonzero(votes * 2 > len(flag_set)).astype(np.uint32)


def summarise_read(tally: Tally, n_trains: int) -> dict[str, Any]:
    rows = tally.rows
    steps = np.unique(
        np.concatenate([np.diff(t.pulse_ids.astype(np.int64)) for t in tally.trains])
    )
    pulse_ids = tally.trains[0].pulse_ids
    cells = sorted(tally.cell_votes)
    always = np.array(
        [(tally.cell_votes[c] == tally.cell_trains[c]).sum() for c in cells]
    )
    varying = np.array(
        [
            (
                (tally.cell_votes[c] > 0) & (tally.cell_votes[c] < tally.cell_trains[c])
            ).sum()
            for c in cells
        ]
    )
    in_every_cell = np.all(
        [tally.cell_votes[c] == tally.cell_trains[c] for c in cells], axis=0
    )
    in_any_cell = np.any([tally.cell_votes[c] > 0 for c in cells], axis=0)
    train_bases = [majority_base(sets, tally.n_env) for _, sets in tally.flag_sets]
    train_base_difference = np.mean(
        [
            np.setxor1d(pixels, base, assume_unique=True).size
            for base, (_, sets) in zip(train_bases, tally.flag_sets, strict=True)
            for pixels in sets
        ]
    )
    n_env = tally.n_env

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    step = int(steps[0]) if steps.size == 1 else None
    return {
        "n_trains": n_trains,
        "frames_per_train": sorted({r["n_frames"] for r in rows}),
        "pulse_id_step": steps.tolist(),
        "pulse_period_ns_if_4p5_MHz": step / 4.5e6 * 1e9 if step else None,
        "pulse_pattern_identical_in_all_trains": len(tally.patterns) == 1,
        "pulse_ids_head": pulse_ids[:6].tolist(),
        "pulse_ids_tail": pulse_ids[-3:].tolist(),
        "trains_with_a_single_sequence_file": all(len(s) == 1 for s in tally.sequences),
        "sequence_of_trains": [min(s) for s in tally.sequences],
        "image_data_dtype": sorted(tally.dtype),
        "max_count_in_data": tally.max_count,
        "max_count_in_envelope": max(r["max_count"] for r in rows),
        "bits_present_in_mask": f"{tally.bits:#x}",
        "n_envelope": n_env,
        "mask": {
            "mean_flagged_per_frame": mean("mean_flagged_per_frame"),
            "mean_flagged_fraction_per_frame": mean("mean_flagged_per_frame") / n_env,
            "per_cell_always_flagged_mean": float(always.mean()),
            "per_cell_varying_between_trains_mean": float(varying.mean()),
            "per_cell_varying_between_trains_max": int(varying.max()),
            "flagged_in_every_cell": int(in_every_cell.sum()),
            "flagged_in_some_cell": int(in_any_cell.sum()),
            "flagged_in_some_but_not_every_cell": int(
                (in_any_cell & ~in_every_cell).sum()
            ),
            "any_frame_flagged_mean": mean("any_frame_flagged"),
            "any_frame_flagged_fraction": mean("any_frame_flagged") / n_env,
            "any_frame_flagged_in_good_pixels_mean": mean(
                "any_frame_flagged_in_good_pixels"
            ),
            "per_frame_difference_from_train_majority_mask_mean": float(
                train_base_difference
            ),
        },
        "events": {
            "nnz_per_train_mean": mean("nnz"),
            "nnz_per_frame_mean": mean("nnz") / rows[0]["n_frames"],
        },
        "per_train": rows,
    }


# ── B1: sizes ─────────────────────────────────────────────────────────────────


def write_candidates(tally: Tally, directory: Path) -> dict[str, Any]:
    """Each candidate encoding, one HDF5 file apiece, group per train.

    The per-cell base comes from the even-numbered trains and is applied to the
    odd-numbered ones, so it is measured on trains it was not built from.
    """
    n_env, n = tally.n_env, len(tally.trains)
    base = FlagBase.from_samples(tally.flag_sets[::2])
    names = (
        "events_absolute",
        "events_delta",
        "flags_full",
        "flags_diff_train_majority_mask",
        "flags_diff_cell_base",
        "flags_packed_bits",
        "flag_base_once",
    )
    paths = {name: directory / f"{name}.h5" for name in names}
    files = {name: h5py.File(path, "w") for name, path in paths.items()}
    held_out = 0

    for index, photons in enumerate(tally.trains):
        group = f"t{photons.train_id:012d}"
        pointers = photons.frame_ptr.astype(np.uint32)
        out = files["events_absolute"].create_group(group)
        out.create_dataset("pixel", data=photons.pixel, **GZIP)
        out.create_dataset("count", data=photons.count, **GZIP)
        out.create_dataset("frame_ptr", data=pointers)

        out = files["events_delta"].create_group(group)
        out.create_dataset(
            "pixel_step", data=delta_encode(photons.pixel, photons.frame_ptr), **GZIP
        )
        out.create_dataset("count", data=photons.count, **GZIP)
        out.create_dataset("frame_ptr", data=pointers)

        cells, sets = tally.flag_sets[index]
        out = files["flags_full"].create_group(group)
        out.create_dataset("pixel", data=photons.flag_pixel, **GZIP)
        out.create_dataset("frame_ptr", data=photons.flag_ptr.astype(np.uint32))

        majority = majority_base(sets, n_env)
        differences = [np.setxor1d(p, majority, assume_unique=True) for p in sets]
        out = files["flags_diff_train_majority_mask"].create_group(group)
        out.create_dataset("mask", data=majority, **GZIP)
        out.create_dataset(
            "pixel", data=np.concatenate(differences).astype(np.uint32), **GZIP
        )
        out.create_dataset("frame_ptr", data=_pointers(differences))

        out = files["flags_packed_bits"].create_group(group)
        packed = np.zeros((len(sets), (n_env + 7) // 8), dtype=np.uint8)
        for frame, flagged in enumerate(sets):
            dense = np.zeros(n_env, dtype=bool)
            dense[flagged] = True
            packed[frame] = np.packbits(dense)
        out.create_dataset("bits", data=packed, chunks=(1, packed.shape[1]), **GZIP)

        if index % 2 == 1:
            held_out += 1
            differences = [
                base.difference(int(cell), flagged)
                for cell, flagged in zip(cells.tolist(), sets, strict=True)
            ]
            out = files["flags_diff_cell_base"].create_group(group)
            out.create_dataset("pixel", data=np.concatenate(differences), **GZIP)
            out.create_dataset("frame_ptr", data=_pointers(differences))

    once = files["flag_base_once"]
    once.create_dataset("cells", data=base.cells)
    once.create_dataset("ptr", data=base.ptr.astype(np.uint32))
    once.create_dataset("pixel", data=base.pixel, **GZIP)
    for handle in files.values():
        handle.close()

    sizes = {}
    for name in names:
        trains = {"flags_diff_cell_base": held_out, "flag_base_once": 1}.get(name, n)
        size = paths[name].stat().st_size
        sizes[name] = {
            "bytes_total": size,
            "n_trains": trains,
            "bytes_per_train": size / trains,
        }
    return sizes


def _pointers(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum([p.size for p in parts])]).astype(np.uint32)


def stage_sizes(tally: Tally, tmp_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(dir=tmp_dir) as directory:
        sizes = write_candidates(tally, Path(directory))
    per_train = (
        sizes["events_delta"]["bytes_per_train"]
        + sizes["flags_diff_cell_base"]["bytes_per_train"]
    )
    once = sizes["flag_base_once"]["bytes_total"]
    return {
        "candidates": sizes,
        "chosen_layout": "events_delta + flags_diff_cell_base (+ flag_base_once)",
        "chosen_bytes_per_train": per_train,
        "projection_gigabytes": {
            str(n): (n * per_train + once) / 1e9 for n in PROJECTION_TRAINS
        },
    }


# ── driver ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--run", type=int, nargs="+", required=True)
    # one process reads the trains: -w only sizes the job
    add_job_args(parser, workers=2, time="00:40:00")
    parser.add_argument("--n-trains", type=int, default=50)
    parser.add_argument("--frame-table", type=Path, default=None)
    parser.add_argument("--tmp-dir", type=Path, default=SCRATCH)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if len(args.run) > 1 and args.json:
        parser.error("--json names one run's file")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return dispatch(
        args,
        args.run,
        lambda run: run_one(argparse.Namespace(**vars(args) | {"run": run})),
        argv=argv,
    )


def run_one(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    saxs = AgipdSaxsConfig(proposal=args.proposal, run=args.run)
    cfg = XpcsConfig(saxs=saxs)
    report: dict[str, Any] = report_header(
        proposal=args.proposal, run=args.run, config=cfg.payload()
    )

    table_path, table = frame_table_trains(saxs, args.frame_table)
    counts, n_counts = np.unique(table["count"], return_counts=True)
    chosen = evenly_spaced(table["ok"], args.n_trains)
    report["frame_table"] = {
        "path": str(table_path),
        "n_trains": int(table["all"].size),
        "n_ok_trains": int(table["ok"].size),
        "frames_per_ok_train": dict(
            zip(counts.tolist(), n_counts.tolist(), strict=True)
        ),
        "chosen_trains": chosen.tolist(),
    }

    started = time.perf_counter()
    pixels = pixel_map(cfg)
    envelope = build_envelope(pixels, cfg)
    static = build_static_bad(saxs)
    rings = build_rings(pixels, envelope, static.bad, cfg)
    ring_good = rings.label >= 0
    report["envelope"] = {
        "n_envelope": envelope.size,
        "envelope_sha256": envelope.sha256,
        "margin_mm": cfg.envelope_margin_mm,
        "good_pixels_in_rings": int(ring_good.sum()),
        "good_pixels_per_ring": np.bincount(
            rings.label[ring_good], minlength=rings.n_rings
        ).tolist(),
        "rings_sha256": rings.sha256,
        "seconds": time.perf_counter() - started,
    }

    run_dir = PROPOSAL_ROOT / "proc" / f"r{args.run:04d}"
    report["sequences"] = stage_sequences(run_dir, args.run)
    report["xray_pulses_raw"] = stage_xray_pulses(saxs)

    tally = stage_read(cfg, chosen, envelope.pixels, ring_good)
    summary = summarise_read(tally, chosen.size)
    summary["seconds_per_train"] = {
        k: v / chosen.size for k, v in tally.timings.items()
    }
    report["facts"] = summary
    report["constant_frames_per_ok_train"] = len(counts) == 1
    report["sizes"] = stage_sizes(tally, args.tmp_dir)

    write_report(report, args.json, "xpcs_hits_facts")
    print(
        f"\nB1: {report['sizes']['chosen_bytes_per_train'] / 1e6:.2f} MB per hit train"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
