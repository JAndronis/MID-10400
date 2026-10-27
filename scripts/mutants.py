#!/usr/bin/env python
"""Run the mutation tables in scripts/mutants/*.toml against the test suite.

Each table entry replaces one exact string in a copy of ``src/`` and runs the
listed tests with ``PYTHONPATH`` at the copy. A mutant is *killed* when its tests
fail and *survived* when they pass. The exit status is non-zero when any result
differs from the entry's ``expect`` (``killed`` unless the entry says
``survived``), or when an ``old`` string no longer matches exactly once in the
real ``src/``; that check runs first, so a table cannot rot unnoticed.

    python scripts/mutants.py --list
    python scripts/mutants.py                     # every table
    python scripts/mutants.py --slice common -j 4
    python scripts/mutants.py --only phi_ferritin_zero

Entry fields: ``name``, ``file`` (relative to the repository root, under
``src/``), ``old``, ``new``, ``tests`` (pytest paths or node ids), ``expect``.
Not unit-tested; the check is running it.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TABLES = Path(__file__).resolve().parent / "mutants"
SHM = Path("/dev/shm")


@dataclass(frozen=True)
class Mutant:
    slice: str
    name: str
    file: str
    old: str
    new: str
    tests: tuple[str, ...]
    expect: str


def load(slice_names: list[str] | None) -> list[Mutant]:
    mutants: list[Mutant] = []
    for path in sorted(TABLES.glob("*.toml")):
        if slice_names and path.stem not in slice_names:
            continue
        for entry in tomllib.loads(path.read_text()).get("mutant", []):
            expect = entry.get("expect", "killed")
            if expect not in ("killed", "survived"):
                sys.exit(f"{path.name}: {entry['name']}: bad expect {expect!r}")
            mutants.append(
                Mutant(
                    path.stem,
                    entry["name"],
                    entry["file"],
                    entry["old"],
                    entry["new"],
                    tuple(entry["tests"]),
                    expect,
                )
            )
    names = [m.name for m in mutants]
    for name in {n for n in names if names.count(n) > 1}:
        sys.exit(f"duplicate mutant name {name!r}")
    return mutants


def stale(mutants: list[Mutant]) -> list[str]:
    """Entries whose ``old`` does not occur exactly once in the real source."""
    problems = []
    for m in mutants:
        path = REPO / m.file
        if not m.file.startswith("src/") or not path.is_file():
            problems.append(f"{m.name}: {m.file} is not a file under src/")
            continue
        count = path.read_text().count(m.old)
        if count != 1:
            problems.append(f"{m.name}: `old` matches {count} times in {m.file}")
    return problems


def scratch_root() -> Path:
    base = (
        SHM if SHM.is_dir() and os.access(SHM, os.W_OK) else Path(tempfile.gettempdir())
    )
    return Path(tempfile.mkdtemp(prefix=f"mutants-{getpass.getuser()}-", dir=base))


def run_one(m: Mutant, root: Path, python: str) -> tuple[str, float, str]:
    """Return ``(result, seconds, detail)``; result is killed, survived or error."""
    start = time.monotonic()
    work = root / m.name
    shutil.copytree(
        REPO / "src", work / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    target = work / m.file
    target.write_text(target.read_text().replace(m.old, m.new))
    env = dict(
        os.environ,
        PYTHONPATH=str(work / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(work),
    )
    resolved = subprocess.run(
        [python, "-c", "import analysis; print(analysis.__file__)"],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not resolved.startswith(str(work / "src")):
        return "error", time.monotonic() - start, f"import resolved to {resolved!r}"
    proc = subprocess.run(
        [
            python,
            "-m",
            "pytest",
            "-q",
            "-x",
            "-p",
            "no:cacheprovider",
            f"--basetemp={work / 'bt'}",
            *m.tests,
        ],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
    detail = lines[-1] if lines else ""
    (work / "pytest.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode == 0:
        return "survived", elapsed, detail
    if proc.returncode == 1:
        return "killed", elapsed, detail
    return "error", elapsed, f"pytest exit {proc.returncode}: {detail}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--slice", action="append", help="table name (file stem); repeatable"
    )
    ap.add_argument("--only", action="append", help="mutant name; repeatable")
    ap.add_argument("-j", type=int, default=1, help="mutants in parallel (default 1)")
    ap.add_argument("--list", action="store_true", help="list the mutants and exit")
    ap.add_argument(
        "--keep", action="store_true", help="keep the mutated copies and logs"
    )
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    mutants = load(args.slice)
    if args.only:
        unknown = set(args.only) - {m.name for m in mutants}
        if unknown:
            sys.exit(f"no such mutant: {sorted(unknown)}")
        mutants = [m for m in mutants if m.name in args.only]
    if args.list:
        for m in mutants:
            print(f"{m.slice}/{m.name}\texpect={m.expect}\t{m.file}")
        return 0
    if not mutants:
        sys.exit("no mutants selected")

    problems = stale(mutants)
    if problems:
        print("stale mutant table:", *problems, sep="\n  ", file=sys.stderr)
        return 2

    root = scratch_root()
    print(f"copies under {root}", flush=True)
    bad = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.j)) as pool:
            futures = [(m, pool.submit(run_one, m, root, args.python)) for m in mutants]
            for m, future in futures:
                result, seconds, detail = future.result()
                ok = result == m.expect
                bad += not ok
                flag = "ok  " if ok else "FAIL"
                print(
                    f"{flag} {m.slice}/{m.name}: {result} "
                    f"(expected {m.expect}) {seconds:.1f}s | {detail}",
                    flush=True,
                )
    finally:
        if args.keep:
            print(f"kept {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
    print(f"{len(mutants) - bad}/{len(mutants)} as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
