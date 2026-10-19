#!/usr/bin/env python
"""Reprocess runs through DAMNIT without ever truncating an output (open task 17f).

The pass refuses to resume into a file it cannot extend, and answers a complete
one from the file. So reprocessing means moving the output aside first, letting
DAMNIT write a fresh one, checking it, and only then deleting the old one:

``status``
    per run and detector: the output, whether it is complete and why not, a
    lock, a DAMNIT job, backups and their check verdict.
``start``
    refuses unless every run is idle — no DAMNIT job queued or running, no lock,
    no leftover backup — and the fileset has room for the new files. Then
    renames each output to ``<stem>_before_reprocess_<stamp>.h5`` (instant, on
    the same fileset) and runs ``damnit reprocess --match <title> <runs>``.
``check``
    passes a run when the new output is complete
    (:func:`analysis.saxs.complete.is_complete` / ``analysis.waxs``), every
    dataset in it reads back, and it has the backup's frames — unless
    ``--accept-count-change``. Writes the verdict beside the backup.
``cleanup``
    deletes only backups whose check passed and whose new output is still the
    file that was checked (same inode, mtime and size).
``restore``
    puts a backup back, keeping the new output as ``<stem>_failed_reprocess_…``.

Never moves the AGIPD window sums (``usr/cached_files/agipd_pixel_sums``) or
photon statistics (``usr/cached_files/agipd_photon_stats``): the pass never
truncates them, and once proc is on tape they cannot be rebuilt.

``-d xpcs`` is the hit-train XPCS product (``analysis.xpcs``, ``context/xpcs-hits.md``).
The output moved aside, rebuilt and checked is the result file ``xpcs_hits.h5``,
which is derived from the ring-photon store. The **store**
(``agipd_ring_photons.h5``, ``cfg.store_file``) is the hit trains' only 2D record
once proc is taped, so it is a companion like the window sums: test-read by
``check``, never renamed, never in ``--match``. The pass itself also moves a
result made for other inputs to ``xpcs_hits.prev.h5``; that file is not one of
this script's backups. ``xpcs_hits_preprocessing`` depends on ``agipd_saxs``, so
asking DAMNIT for it would run the AGIPD pass: ``start -d xpcs`` therefore
refuses a run unless its AGIPD output is complete, and for a run whose store is
absent unless its proc directory exists. What this script cannot judge for xpcs:
whether the hit set is the classifier's current one and whether the merge hash
matches today's config. Both need the classifier and the geometry, which a
script must not import, so ``complete`` means only that the file is readable,
finalised, has no unfinished or label-mismatched hit train, and that the store
holds every hit train under the hash the result recorded. Final non-OK trains
(NO_FRAMES and the like) are reported as notes. ``xpcs_hits_preprocessing`` is
the judge of the rest, and answers a complete run from its file.
``--dry-run`` prints what would happen and changes nothing. Writes its report as
JSON beside itself.

DAMNIT jobs: one run is a job named ``r<run>-p10400-damnit``; several are a job
array ``p10400-damnit`` whose runs squeue cannot show, so any active array
counts every run as busy. Non-cluster jobs run on cluster ``solaris``, hence
``squeue --clusters=all``.

    python scripts/reprocess_safe.py status 423 426 -d agipd -d jf1
    python scripts/reprocess_safe.py start 423-428 -d agipd --dry-run
    python scripts/reprocess_safe.py check 423-428 -d agipd -j 8
    python scripts/reprocess_safe.py start 493-496 -d xpcs --dry-run
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from _common import report_header, stamp, write_report

PROPOSAL = 10400
DAMNIT_DIR = Path("/gpfs/exfel/exp/MID/202601/p010400/usr/Shared/amore")
#: Space to leave free beyond the new outputs, as a fraction of their size.
QUOTA_MARGIN = 0.2
#: Bytes read at a time when test-reading a dataset.
READ_BYTES = 1 << 28


@dataclass(frozen=True)
class Detector:
    """One DAMNIT-written output kind, and how to judge it."""

    name: str
    #: DAMNIT variable titles to ``--match``. A JF detector also matches the
    #: overview and the combined curve, which depend on both detectors; the one
    #: not reprocessed is answered from its file, as is the other detector in
    #: ``JUNGFRAU preprocessing``, which runs both.
    titles: tuple[str, ...]
    #: ``--root``: where the outputs live instead of the config's default, for
    #: exercising this script on copies. ``output_root`` is operational, so it
    #: does not change the config hash a file is judged against.
    root_override: str | None = None

    def config(self, run: int) -> Any:
        roots = {}
        if self.root_override:
            roots = {"output_root": self.root_override}
        if self.name == "xpcs":
            from analysis.xpcs.damnit import config_for

            return config_for(PROPOSAL, run, **roots)
        if self.name == "agipd":
            from analysis.saxs.damnit import config_for

            if self.root_override:
                roots["pixel_sums_root"] = self.root_override
                roots["photon_stats_root"] = self.root_override
            return config_for(PROPOSAL, run, **roots)
        from analysis.waxs.damnit import damnit_config

        return damnit_config(PROPOSAL, run, self.name, **roots)

    def output(self, run: int) -> Path:
        return self.config(run).output_file

    def root(self) -> Path:
        return self.output(0).parent.parent

    def complete(self, run: int, path: Path) -> Any:
        if self.name == "xpcs":
            return xpcs_completeness(path, self.config(run).store_file)
        if self.name == "agipd":
            from analysis.saxs.complete import is_complete

            return is_complete(path, self.config(run))
        from analysis.waxs.complete import is_complete

        return is_complete(path, self.config(run))

    def companions(self, run: int) -> list[Path]:
        """Files the pass also writes, test-read but never moved."""
        if self.name == "agipd":
            cfg = self.config(run)
            return [cfg.pixel_sums_file] + (
                [cfg.photon_stats_file] if cfg.photon_stats else []
            )
        if self.name == "xpcs":
            store = self.config(run).store_file
            return [store] if store.exists() else []
        return []

    def caches(self, run: int) -> list[Path]:
        """`companions`, whether or not they exist now: what ``--fresh`` moves."""
        if self.name == "xpcs":
            return [self.config(run).store_file]
        return self.companions(run)


DETECTORS = {
    # The preprocessing variable reads proc and writes the files; the others
    # read them (user, 2026-10-14). --match takes case-insensitive title
    # substrings, so "AGIPD I(q)" also matches its overview. Titles are the
    # group's title plus the variable's (src/amore/context.py groups,
    # 2026-10-19); `scripts/match_titles.py --self-test` asserts each set
    # against the context file.
    "agipd": Detector(
        "agipd",
        (
            "AGIPD preprocessing",
            "AGIPD I(q)",
            "AGIPD SVD and chi2",
            "XPCS photon statistics",
        ),
    ),
    "jf1": Detector("jf1", ("JUNGFRAU preprocessing", "JF1 I(q)", "JUNGFRAU I(q)")),
    "jf2": Detector("jf2", ("JUNGFRAU preprocessing", "JF2 I(q)", "JUNGFRAU I(q)")),
    # Every cluster variable of the hit-train pass and its figures. Not "XPCS
    # frames used" and "XPCS score", which read the file the cluster job is still
    # writing: `reprocess_all.sh refresh` does those afterwards. "XPCS g2" also
    # selects the g2 figure and fit, "XPCS TTCF" the per-train TTCF.
    "xpcs": Detector(
        "xpcs",
        (
            "XPCS preprocessing",
            "XPCS g2",
            "XPCS TTCF",
            "XPCS overview",
            "XPCS q-rings",
            "XPCS beamcenter",
            "XPCS outliers",
            "XPCS intensity outliers",
            "XPCS kbar",
            "XPCS correction",
            "XPCS SAXS",
            "XPCS mean dataset",
        ),
    ),
}

#: Where a run's proc data live, for judging whether a missing store can be rebuilt.
PROC_DIR = Path("/gpfs/exfel/exp/MID/202601/p010400/proc")


# ── the xpcs result file ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class XpcsVerdict:
    """What the result file and its store record; see the module docstring."""

    path: Path
    reasons: tuple[str, ...]
    notes: tuple[str, ...] = ()


def xpcs_completeness(path: Path, store: Path) -> XpcsVerdict:
    """Why ``path`` is not complete, from what it and its store record alone.

    Imports no classifier and builds no geometry, so it cannot tell a hit set or
    a merge hash from the current ones (module docstring).
    """
    import h5py
    import numpy as np

    from analysis.common.status import UNFINISHED, FrameStatus
    from analysis.xpcs.result import SCHEMA
    from analysis.xpcs.store import ABSENT, RingPhotonStore, StoreMismatch

    if not path.exists():
        return XpcsVerdict(path, ("no file",))
    reasons: list[str] = []
    notes: list[str] = []
    try:
        with h5py.File(path, "r") as handle:
            schema = handle.attrs.get("schema")
            extract_hash = handle.attrs.get("extract_hash")
            finalised = "finalised" in handle.attrs
            train_ids = handle["hits/train_id"][()]
            status = handle["hits/status"][()]
    except (OSError, KeyError, ValueError) as error:
        return XpcsVerdict(path, (f"unreadable: {error!r}",))
    if schema != SCHEMA:
        reasons.append(f"schema {schema}, this code's is {SCHEMA}")
    if not finalised:
        reasons.append("never finalised")
    bad = (*sorted(UNFINISHED), FrameStatus.LABEL_MISMATCH)
    for code in bad:
        if n := int((status == code).sum()):
            reasons.append(f"{n} trains {code.name}")
    for code in FrameStatus:
        if code not in bad and code != FrameStatus.OK and (status == code).any():
            notes.append(f"{int((status == code).sum())} trains {code.name}")
    if not train_ids.size:
        notes.append("no hit train")
        return XpcsVerdict(path, tuple(reasons), tuple(notes))
    if not store.exists():
        reasons.append(f"store {store} is missing")
        return XpcsVerdict(path, tuple(reasons), tuple(notes))
    try:
        with RingPhotonStore.open(store) as photons:
            if photons.extract_hash != extract_hash:
                reasons.append("the store was extracted under another hash")
            usable = status == FrameStatus.OK
            held = photons.status(train_ids[usable])
            lacking = int(np.isin(held, [int(c) for c in UNFINISHED] + [ABSENT]).sum())
            if lacking:
                reasons.append(f"the store lacks {lacking} of the OK hit trains")
    except (OSError, KeyError, StoreMismatch) as error:
        reasons.append(f"store unreadable: {error!r}")
    return XpcsVerdict(path, tuple(reasons), tuple(notes))


# ── names ─────────────────────────────────────────────────────────────────────
def backup_name(output: Path, when: str, kind: str = "before") -> Path:
    return output.with_name(f"{output.stem}_{kind}_reprocess_{when}{output.suffix}")


def backups_of(output: Path) -> list[Path]:
    pattern = re.compile(
        rf"^{re.escape(output.stem)}_before_reprocess_\d{{8}}T\d{{6}}Z"
        rf"{re.escape(output.suffix)}$"
    )
    if not output.parent.exists():
        return []
    return sorted(p for p in output.parent.iterdir() if pattern.match(p.name))


def stamp_of(backup: Path) -> str:
    """The ``YYYYMMDDTHHMMSSZ`` a backup was made at, from its name."""
    return re.search(r"_before_reprocess_(\d{8}T\d{6}Z)", backup.name).group(1)


def companion_backup(companion: Path, when: str) -> Path:
    return backup_name(companion, when)


def marker_of(backup: Path) -> Path:
    return backup.with_name(f"{backup.name}.check.json")


def file_stat(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_ino, stat.st_mtime_ns, stat.st_size]


# ── the cluster ───────────────────────────────────────────────────────────────
def damnit_jobs() -> tuple[set[int], list[str]]:
    """Runs with a DAMNIT job queued or running, and any active job arrays.

    :raises SystemExit: squeue failed, so nothing can be said to be idle.
    """
    command = ["squeue", "--clusters=all", "--noheader", "--format=%i|%j|%u|%T"]
    result = subprocess.run(command, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if "|" in line]
    if result.returncode and not lines:
        raise SystemExit(f"cannot read the Slurm queue: {result.stderr.strip()}")
    runs, arrays = set(), []
    single = re.compile(rf"^r(\d+)-p{PROPOSAL}-damnit$")
    for line in lines:
        job, name, user, state = (part.strip() for part in line.split("|"))
        if match := single.match(name):
            runs.add(int(match.group(1)))
        elif name == f"p{PROPOSAL}-damnit":
            arrays.append(f"{job} ({user}, {state})")
    return runs, arrays


def free_bytes(directory: Path) -> tuple[int, str]:
    """Bytes left under the quota of ``directory``'s GPFS fileset, and its name."""
    attrs = subprocess.run(
        ["mmlsattr", "-L", str(directory)], capture_output=True, text=True, check=True
    ).stdout
    fileset = re.search(r"fileset name:\s*(\S+)", attrs).group(1)
    filesystem = subprocess.run(
        ["df", "--output=source", str(directory)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[-1]
    quota = subprocess.run(
        ["mmlsquota", "-j", fileset, "--block-size", "K", filesystem],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fields = next(
        line.split() for line in quota.splitlines() if line.split()[:1] == [filesystem]
    )
    used, soft, hard, doubt = (int(value) for value in fields[2:6])
    cap = min(value for value in (soft, hard) if value > 0)
    return (cap - used - doubt) * 1024, fileset


# ── reading ───────────────────────────────────────────────────────────────────
def test_read(path: Path) -> list[str]:
    """Read every dataset of ``path`` in full; one entry per one that fails.

    ``visititems`` stops at the first callback returning anything but ``None``,
    so failures are collected, never returned.
    """
    import h5py

    failures: list[str] = []

    def read(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        try:
            if obj.ndim == 0 or obj.shape[0] == 0:
                obj[()]
                return
            row = max(1, obj.dtype.itemsize * int(obj.size // obj.shape[0]))
            step = max(1, READ_BYTES // row)
            for start in range(0, obj.shape[0], step):
                obj[start : start + step]
        except Exception as error:  # noqa: BLE001 - every failure is reported
            failures.append(f"{name}: {error!r}")

    try:
        with h5py.File(path, "r") as handle:
            handle.visititems(read)
    except Exception as error:  # noqa: BLE001
        failures.append(f"opening: {error!r}")
    return failures


def frames_of(path: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        if "hits/train_id" in handle:
            hits = handle["hits/train_id"]
            return {"frames": int(hits.shape[0]), "trains": hits[()].tolist()}
        return {
            "frames": int(handle["frames/status"].shape[0]),
            "trains": np.asarray(handle["trains/trainId"][:]).tolist(),
        }


# ── commands ──────────────────────────────────────────────────────────────────
def status_one(run: int, detector: Detector, busy: set[int]) -> dict[str, Any]:
    output = detector.output(run)
    lock = output.with_name(f"{output.name}.lock")
    record: dict[str, Any] = {
        "run": run,
        "detector": detector.name,
        "output": str(output),
        "exists": output.exists(),
        "busy": run in busy,
        "lock": lock.read_text() if lock.exists() else None,
        "backups": {
            str(b): (
                json.loads(marker_of(b).read_text())["verdict"]
                if marker_of(b).exists()
                else "unchecked"
            )
            for b in backups_of(output)
        },
    }
    if record["exists"] and not record["busy"] and record["lock"] is None:
        verdict = detector.complete(run, output)
        record["complete_reasons"] = list(verdict.reasons)
        if notes := getattr(verdict, "notes", ()):
            record["notes"] = list(notes)
    return record


def check_one(
    run: int, detector: Detector, busy: set[int], accept_count_change: bool
) -> dict[str, Any]:
    output = detector.output(run)
    backups = backups_of(output)
    reasons: list[str] = []
    record: dict[str, Any] = {"run": run, "detector": detector.name}
    if len(backups) != 1:
        reasons.append(f"{len(backups)} backups, expected exactly one")
    if run in busy:
        reasons.append("a DAMNIT job for the run is queued or running")
    if output.with_name(f"{output.name}.lock").exists():
        reasons.append("the output is locked")
    if not output.exists():
        reasons.append("no new output")
    if reasons:
        return record | {"verdict": "fail", "reasons": reasons}

    backup = backups[0]
    reasons += [f"not complete: {r}" for r in detector.complete(run, output).reasons]
    for path in (output, *detector.companions(run)):
        reasons += [f"{path.name}: {failure}" for failure in test_read(path)]
    try:
        old, new = frames_of(backup), frames_of(output)
    except (OSError, KeyError) as error:
        reasons.append(f"frames unreadable: {error!r}")
        old = new = None
    if old is not None:
        record["frames"] = {"before": old["frames"], "after": new["frames"]}
    if old != new:
        change = (
            f"frames {old['frames']} -> {new['frames']}, trains "
            f"{len(old['trains'])} -> {len(new['trains'])}"
        )
        if accept_count_change:
            record["accepted_change"] = change
        else:
            reasons.append(f"{change}; pass --accept-count-change if that is meant")

    verdict = "pass" if not reasons else "fail"
    marker = {
        "verdict": verdict,
        "reasons": reasons,
        "output": str(output),
        "output_stat": file_stat(output),
        "checked_at": stamp(),
    } | record
    marker_of(backup).write_text(json.dumps(marker, indent=2))
    return record | {"verdict": verdict, "reasons": reasons, "backup": str(backup)}


def command_status(items, args, busy, arrays) -> dict[str, Any]:
    results = run_each(items, status_one, args.jobs, busy)
    for r in results:
        state = (
            "BUSY"
            if r["busy"]
            else "LOCKED"
            if r["lock"]
            else "missing"
            if not r["exists"]
            else "complete"
            if not r.get("complete_reasons")
            else "INCOMPLETE"
        )
        backups = ", ".join(f"{Path(b).name}={v}" for b, v in r["backups"].items())
        why = "; ".join(r.get("complete_reasons") or [])
        notes = f" [{'; '.join(r['notes'])}]" if r.get("notes") else ""
        print(f"r{r['run']:04d} {r['detector']:5s} {state:10s} {why}{notes} {backups}")
    return {"arrays": arrays, "results": results}


def xpcs_start_problems(run: int, detector: Detector) -> list[str]:
    """Why ``damnit reprocess`` must not be asked for the XPCS variables of ``run``.

    ``xpcs_hits_preprocessing`` depends on ``agipd_saxs``: DAMNIT would run the
    AGIPD pass, outside this script's protections, for a run whose AGIPD output
    is not complete. The AGIPD output is always judged at its real location,
    whatever ``--root`` says about the XPCS files.
    """
    tag = f"r{run:04d} xpcs"
    agipd = DETECTORS["agipd"]
    path = agipd.output(run)
    if not path.exists():
        return [f"{tag}: no AGIPD output {path}; DAMNIT would run the AGIPD pass"]
    if path.with_name(f"{path.name}.lock").exists():
        return [f"{tag}: the AGIPD output is locked"]
    reasons = agipd.complete(run, path).reasons
    problems = (
        [f"{tag}: AGIPD output not complete ({'; '.join(reasons)})"] if reasons else []
    )
    store = detector.config(run).store_file
    if not store.exists() and not (PROC_DIR / f"r{run:04d}").exists():
        problems.append(f"{tag}: no photon store {store} and no proc directory")
    return problems


def command_start(items, args, busy, arrays) -> dict[str, Any]:
    problems: list[str] = []
    if arrays:
        problems.append(f"DAMNIT job arrays are active ({arrays}); runs unknown")
    names = {detector.name for _, detector in items}
    if "xpcs" in names and names - {"xpcs"}:
        problems.append(
            "-d xpcs needs the AGIPD output complete before it starts: reprocess "
            "the AGIPD and JUNGFRAU outputs first, wait for the jobs, then xpcs"
        )
    moves: list[tuple[Path, Path]] = []
    when = stamp()
    size = 0
    for run, detector in items:
        output = detector.output(run)
        tag = f"r{run:04d} {detector.name}"
        if run in busy:
            problems.append(f"{tag}: a DAMNIT job is queued or running")
        if output.with_name(f"{output.name}.lock").exists():
            problems.append(f"{tag}: the output is locked")
        if leftover := backups_of(output):
            problems.append(f"{tag}: leftover backup {leftover[0].name}")
        if output.exists():
            moves.append((output, backup_name(output, when)))
            size += output.stat().st_size
        if args.fresh:
            for companion in detector.caches(run):
                if backups_of(companion):
                    problems.append(
                        f"{tag}: leftover backup of {companion.name}: "
                        f"{backups_of(companion)[0].name}"
                    )
                if companion.exists():
                    moves.append((companion, companion_backup(companion, when)))
                    size += companion.stat().st_size
        if detector.name == "xpcs":
            problems += xpcs_start_problems(run, detector)

    # A product not written yet has no directory: ask about the first one that exists.
    roots = {
        next(p for p in (detector.root(), *detector.root().parents) if p.exists())
        for _, detector in items
    }
    quota = {}
    for root in roots:
        try:
            free, fileset = free_bytes(root)
        except (subprocess.CalledProcessError, StopIteration, AttributeError) as e:
            problems.append(f"cannot read the quota of {root}: {e!r}")
            continue
        quota[str(root)] = {"fileset": fileset, "free_bytes": free}
        if free < size * (1 + QUOTA_MARGIN):
            problems.append(
                f"{fileset} has {free / 1e9:.1f} GB free; the new outputs need "
                f"{size / 1e9:.1f} GB plus {QUOTA_MARGIN:.0%}"
            )

    titles = sorted({t for _, detector in items for t in detector.titles})
    runs = sorted({run for run, _ in items})
    damnit = (
        ["damnit", "reprocess", "--in", str(DAMNIT_DIR)]
        + [part for title in titles for part in ("--match", title)]
        + (
            ["--concurrent-jobs", str(args.concurrent_jobs)]
            if args.concurrent_jobs
            else []
        )
        + [str(run) for run in runs]
    )
    shell = "module load exfel damnit/stable && " + shlex.join(damnit)
    record = {
        "moves": [[str(a), str(b)] for a, b in moves],
        "never_moved": []
        if args.fresh
        else sorted({str(c) for run, d in items for c in d.companions(run)}),
        "bytes": size,
        "quota": quota,
        "command": shell,
        "problems": problems,
    }
    for source, target in moves:
        print(f"rename {source} -> {target.name}")
    print(f"then: bash -lc {shlex.quote(shell)}")
    if problems:
        print("\nnot starting:\n- " + "\n- ".join(problems))
        return record | {"started": False}
    if args.dry_run:
        print("\n--dry-run: nothing moved, nothing started")
        return record | {"started": False}

    for source, target in moves:
        source.rename(target)
    if args.no_trigger:
        print(
            "\n--no-trigger: moved aside; start DAMNIT yourself with the command above"
        )
        return record | {"started": False, "moved": True}
    result = subprocess.run(["bash", "-lc", shell])
    record["damnit_exit"] = result.returncode
    if result.returncode:
        print(
            f"damnit reprocess exited {result.returncode}; the outputs are moved "
            "aside — `restore` puts them back"
        )
    return record | {"started": result.returncode == 0}


def command_check(items, args, busy, arrays) -> dict[str, Any]:
    if arrays:
        print(f"note: DAMNIT job arrays are active ({arrays})")
    results = run_each(items, check_one, args.jobs, busy, args.accept_count_change)
    for r in results:
        reasons = "; ".join(r["reasons"])
        print(f"r{r['run']:04d} {r['detector']:5s} {r['verdict']}: {reasons}")
    return {"arrays": arrays, "results": results}


def command_cleanup(items, args, busy, arrays) -> dict[str, Any]:
    deleted, kept = [], []
    for run, detector in items:
        output = detector.output(run)
        for backup in backups_of(output):
            marker = marker_of(backup)
            verdict = json.loads(marker.read_text()) if marker.exists() else None
            why = (
                "never checked"
                if verdict is None
                else "its check failed"
                if verdict["verdict"] != "pass"
                else "the output changed since its check"
                if not output.exists() or file_stat(output) != verdict["output_stat"]
                else None
            )
            if why:
                kept.append([str(backup), why])
                print(f"keep   {backup.name}: {why}")
                continue
            print(f"delete {backup}")
            deleted.append(str(backup))
            if not args.dry_run:
                backup.unlink()
                marker.unlink()
            for companion in detector.caches(run):
                old = companion_backup(companion, stamp_of(backup))
                if old.exists():
                    print(f"delete {old}")
                    deleted.append(str(old))
                    if not args.dry_run:
                        old.unlink()
    return {"deleted": deleted, "kept": kept, "dry_run": args.dry_run}


def command_restore(items, args, busy, arrays) -> dict[str, Any]:
    actions, refused = [], []
    when = stamp()
    for run, detector in items:
        output = detector.output(run)
        tag = f"r{run:04d} {detector.name}"
        backups = backups_of(output)
        if len(backups) != 1:
            refused.append(f"{tag}: {len(backups)} backups, expected exactly one")
        elif run in busy or arrays:
            refused.append(f"{tag}: a DAMNIT job is queued or running")
        elif output.with_name(f"{output.name}.lock").exists():
            refused.append(f"{tag}: the output is locked")
        else:
            steps = []
            if output.exists():
                steps.append((output, backup_name(output, when, kind="failed")))
            steps.append((backups[0], output))
            for companion in detector.caches(run):
                old = companion_backup(companion, stamp_of(backups[0]))
                if old.exists():
                    if companion.exists():
                        steps.append(
                            (companion, backup_name(companion, when, kind="failed"))
                        )
                    steps.append((old, companion))
            actions.append([[str(a), str(b)] for a, b in steps])
            for source, target in steps:
                print(f"rename {source} -> {target}")
                if not args.dry_run:
                    source.rename(target)
            if not args.dry_run:
                marker_of(backups[0]).unlink(missing_ok=True)
    for line in refused:
        print(f"refused {line}")
    return {"actions": actions, "refused": refused, "dry_run": args.dry_run}


COMMANDS: dict[str, Callable[..., dict[str, Any]]] = {
    "status": command_status,
    "start": command_start,
    "check": command_check,
    "cleanup": command_cleanup,
    "restore": command_restore,
}


# ── plumbing ──────────────────────────────────────────────────────────────────
def _call(job: tuple[Callable[..., Any], int, Detector, tuple[Any, ...]]) -> Any:
    function, run, detector, extra = job
    return function(run, detector, *extra)


def run_each(items, function, jobs: int, *extra: Any) -> list[dict[str, Any]]:
    """``function(run, detector, *extra)`` per item, in ``jobs`` processes."""
    work = [(function, run, detector, extra) for run, detector in items]
    if jobs <= 1:
        return [_call(job) for job in work]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(jobs, mp_context=context) as pool:
        return list(pool.map(_call, work))


def parse_runs(specs: list[str], detectors: list[Detector]) -> list[int]:
    runs: set[int] = set()
    for spec in specs:
        if spec == "all":
            for detector in detectors:
                root = detector.root()
                runs |= {int(p.name[1:]) for p in root.glob("r[0-9][0-9][0-9][0-9]")}
        elif "-" in spec:
            first, last = (int(part) for part in spec.split("-"))
            runs |= set(range(first, last + 1))
        else:
            runs.add(int(spec))
    return sorted(runs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("runs", nargs="+", help="run numbers, ranges a-b, or all")
    parser.add_argument(
        "-d",
        "--detector",
        action="append",
        choices=sorted(DETECTORS),
        required=True,
        help="repeat for several",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="processes")
    parser.add_argument("--accept-count-change", action="store_true")
    parser.add_argument(
        "--concurrent-jobs", type=int, help="passed to damnit reprocess"
    )
    parser.add_argument(
        "--no-trigger",
        action="store_true",
        help="start: move the outputs aside but do not run damnit reprocess",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start: also move the companion caches aside (window sums, photon "
        "histograms, the XPCS photon store), so the pass rewrites them from proc "
        "instead of finding them complete; cleanup and restore follow them",
    )
    parser.add_argument(
        "--root", type=Path, help="outputs under this root instead (testing)"
    )
    parser.add_argument("--json", type=Path, help="where to write the report")
    args = parser.parse_args()

    root = str(args.root.resolve()) if args.root else None
    detectors = [
        replace(DETECTORS[name], root_override=root)
        for name in dict.fromkeys(args.detector)
    ]
    runs = parse_runs(args.runs, detectors)
    items = [(run, detector) for run in runs for detector in detectors]
    busy, arrays = damnit_jobs()
    report = report_header(
        script="reprocess_safe",
        command=args.command,
        runs=runs,
        detectors=[d.name for d in detectors],
        dry_run=args.dry_run,
        busy_runs=sorted(busy),
    )
    report |= COMMANDS[args.command](items, args, busy, arrays)
    write_report(report, args.json, f"reprocess_safe_{args.command}")
    failed = (
        report.get("problems")
        or report.get("refused")
        or any(r.get("verdict") == "fail" for r in report.get("results", []))
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
