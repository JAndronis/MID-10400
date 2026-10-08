#!/usr/bin/env python
"""S3 audit of the per-pixel window sums (context file §15.5) over every run.

For each run with AGIPD proc data, classifies its window-sums file and checks it
against the run's frame table, from the two files alone:

    complete     finalised, every window written, no train left out for a
                 failure, and the trains and frames it summed are exactly the
                 frame table's fully-OK trains and their frames
    missing      no window-sums file
    incomplete   a window unwritten, a train left out for a failure, or the file
                 never finalised
    mismatch     complete on its own, but disagreeing with the frame table
    busy         a DAMNIT job for the run is queued or running; not opened, so
                 nothing reads a file still being written
    no_agipd     no CORR-*-AGIPD*.h5 in the run's proc directory

``missing`` and ``incomplete`` are what ``pixel_sums_backfill.py`` takes.
``mismatch`` is for a person to look at. Writes its verdict as JSON beside
itself (CLAUDE.md working rule 4). Read-only; runs on a login node.

    python scripts/pixel_sums_audit.py
    python scripts/pixel_sums_audit.py --runs 17 423 426
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from _common import report_header, write_report

from analysis.common.status import FrameStatus
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.pixel_sums import FAILED

PROC_ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400/proc")


def busy_runs(proposal: int) -> set[int]:
    """Runs with a DAMNIT job of this user queued or running."""
    try:
        names = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%j"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except Exception as error:  # noqa: BLE001 - refuse to guess
        raise SystemExit(
            f"cannot read the Slurm queue ({error!r}); not auditing"
        ) from error
    pattern = re.compile(rf"^r(\d+)-p{proposal}-damnit$")
    return {int(m.group(1)) for name in names if (m := pattern.match(name))}


def proc_runs() -> list[int]:
    return sorted(int(p.name[1:]) for p in PROC_ROOT.glob("r[0-9][0-9][0-9][0-9]"))


def has_agipd(run: int) -> bool:
    return any((PROC_ROOT / f"r{run:04d}").glob(f"CORR-R{run:04d}-AGIPD*.h5"))


def frame_table(path: Path) -> dict[str, Any]:
    """The frame table's fully-OK trains and their frame count."""
    with h5py.File(path, "r") as handle:
        status = handle["frames/status"][:]
        train_ids = handle["trains/trainId"][:]
        first = handle["trains/first"][:]
        count = handle["trains/count"][:]
        npt = int(json.loads(handle["provenance"].attrs["config"])["npt"])
    ok_trains, ok_frames = [], 0
    for train_id, start, n in zip(train_ids, first, count, strict=True):
        rows = status[int(start) : int(start) + int(n)]
        if n and (rows == FrameStatus.OK).all():
            ok_trains.append(int(train_id))
            ok_frames += int(n)
    return {
        "npt": npt,
        "frames": int(status.size),
        "frames_not_ok": int((status != FrameStatus.OK).sum()),
        "ok_trains": ok_trains,
        "ok_frames": ok_frames,
    }


def window_sums(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        written = handle["windows/written"][:].astype(bool)
        n_frames = handle["windows/n_frames"][:]
        train_ids = handle["trains/trainId"][:]
        status = handle["trains/status"][:]
        provenance = handle["provenance"].attrs
        finalised = "status_summary" in provenance
        window_trains = int(provenance["window_trains"])
    return {
        "finalised": finalised,
        "window_trains": window_trains,
        "windows": int(written.size),
        "windows_written": int(written.sum()),
        "summed_trains": [int(t) for t in train_ids[status == FrameStatus.OK]],
        "summed_frames": int(n_frames[written].sum()),
        "failed_trains": int(np.isin(status, [int(c) for c in FAILED]).sum()),
        "train_status": {
            FrameStatus(int(code)).name: int(n)
            for code, n in zip(*np.unique(status, return_counts=True), strict=True)
        },
    }


def audit_run(run: int, proposal: int, busy: set[int]) -> dict[str, Any]:
    if not has_agipd(run):
        return {"run": run, "class": "no_agipd"}
    if run in busy:
        return {"run": run, "class": "busy"}
    cfg = AgipdSaxsConfig(proposal=proposal, run=run)
    record: dict[str, Any] = {"run": run, "sums_file": str(cfg.pixel_sums_file)}
    if not cfg.pixel_sums_file.exists():
        return record | {"class": "missing"}
    try:
        sums = window_sums(cfg.pixel_sums_file)
    except Exception as error:  # noqa: BLE001 - recorded
        return record | {"class": "incomplete", "error": repr(error)}
    summed = set(sums.pop("summed_trains"))
    record |= sums | {"n_summed_trains": len(summed)}
    if (
        not sums["finalised"]
        or sums["windows_written"] < sums["windows"]
        or sums["failed_trains"]
    ):
        return record | {"class": "incomplete"}

    if not cfg.output_file.exists():
        return record | {"class": "complete", "frame_table": "absent"}
    table = frame_table(cfg.output_file)
    ok = set(table.pop("ok_trains"))
    record["frame_table"] = table | {"n_ok_trains": len(ok)}
    agrees = summed == ok and sums["summed_frames"] == table["ok_frames"]
    record["agrees_with_frame_table"] = agrees
    if not agrees:
        record["only_in_sums"] = sorted(summed - ok)[:10]
        record["only_in_frame_table"] = sorted(ok - summed)[:10]
    return record | {"class": "complete" if agrees else "mismatch"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--runs", type=int, nargs="*", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    busy = busy_runs(args.proposal)
    runs = args.runs or proc_runs()
    records = []
    for run in runs:
        record = audit_run(run, args.proposal, busy)
        records.append(record)
        print(f"r{run:04d} {record['class']}", flush=True)

    classes: dict[str, list[int]] = {}
    for record in records:
        classes.setdefault(record["class"], []).append(record["run"])
    complete = [r for r in records if r["class"] == "complete"]
    report = report_header(
        proposal=args.proposal,
        n_runs=len(records),
        counts={name: len(runs) for name, runs in sorted(classes.items())},
        runs_by_class=classes,
        backfill=sorted(classes.get("missing", []) + classes.get("incomplete", [])),
        windows_complete=sum(r["windows"] for r in complete),
        trains_summed=sum(r["n_summed_trains"] for r in complete),
        frames_summed=sum(r["summed_frames"] for r in complete),
        bytes_complete=sum(Path(r["sums_file"]).stat().st_size for r in complete),
        runs=records,
    )
    write_report(report, args.json, "pixel_sums_audit")
    print(json.dumps(report["counts"]))
    print(f"to backfill: {report['backfill']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
