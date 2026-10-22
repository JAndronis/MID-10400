#!/usr/bin/env python
"""Install the repo's DAMNIT context into the live directory.

DAMNIT reads ``usr/Shared/amore/context.py``, not ``src/amore/context.py``, and
nothing copies one to the other. Others edit the live directory too, so an
install must neither overwrite their change unseen nor leave DAMNIT with a
context that does not evaluate. In order, this script:

1. refuses a dirty ``src/amore`` (the commit recorded must be what is installed);
2. reads the live ``context.py`` and ``analysis_helpers.py`` and their sha256;
3. guard: the live files must be exactly what the last install left
   (``backups/installed.json``). The first time there is no record, so it
   shows the diff and requires ``--accept-live <id>``, where ``<id>`` names the
   live state it printed. Accepting a state that has since changed fails;
4. evaluates the repo's context and the live one with ctxrunner the way DAMNIT
   does (``<context_python> -m ctxrunner ctx`` in a scratch copy, with the
   listener's own ``ctxsupport`` on ``PYTHONPATH``), and fails unless the
   pickled ``error_info`` is ``None``. ``ctx`` exits 0 even for a context that
   does not evaluate, so its exit status says nothing;
5. deploy gate: the commit installed from must already be in the deploy
   clone (``usr/Software/MID-10400``), whose venv DAMNIT runs. Otherwise the new
   context would call code that is not there yet;
6. only with ``--install``: writes a timestamped backup of both live files, the
   diff and the sha256s to ``backups/``, re-checks the live sha256, replaces
   each file atomically keeping its mode, and records what it installed.

It never runs ``damnit read-context``, which adds columns to the shared
database; it opens the database read-only, for two settings. Writes its report
as JSON beside itself. Runs on a login node.

    python scripts/install_context.py                       # dry run
    python scripts/install_context.py --accept-live 1a2b3c4d5e6f
    python scripts/install_context.py --install [--accept-live ...]
"""

from __future__ import annotations

import argparse
import difflib
import getpass
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common import report_header, stamp, write_report

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src" / "amore"
FILES = ("context.py", "analysis_helpers.py")
PROPOSAL = Path("/gpfs/exfel/exp/MID/202601/p010400")
LIVE = PROPOSAL / "usr" / "Shared" / "amore"
DEPLOY = PROPOSAL / "usr" / "Software" / "MID-10400"
SCRATCH = PROPOSAL / "scratch" / "install_context"
#: Characters of the live-state id ``--accept-live`` must repeat.
ID_LENGTH = 12

#: Evaluates a context the way ``damnit.backend.extract_data.get_context_file``
#: does, then unpickles the result with the same name mapping DAMNIT uses (under
#: ``-m`` the class pickles as ``__main__.ContextFile``).
_UNPICKLE = """
import json, pickle, sys, ctxrunner
class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "ContextFile":
            return ctxrunner.ContextFile
        return super().find_class(module, name)
with open(sys.argv[1], "rb") as handle:
    ctx, error = Unpickler(handle).load()
print(json.dumps({
    "error": None if error is None else str(error),
    "variables": sorted(ctx.vars) if ctx is not None else None,
    "titles": {n: v.title for n, v in ctx.vars.items()} if ctx is not None else None,
}))
"""


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def live_state_id(shas: dict[str, str | None]) -> str:
    """One short id for the live files together, for ``--accept-live``."""
    joined = "\n".join(f"{name}:{shas[name]}" for name in FILES)
    return hashlib.sha256(joined.encode()).hexdigest()[:ID_LENGTH]


def damnit_settings() -> dict[str, str]:
    """``context_python`` and ``damnit_python`` from the DB, opened read-only.

    Always the real DAMNIT directory's database, whatever ``--live-dir`` says:
    the settings describe how DAMNIT evaluates a context, not where it lives.
    """
    uri = f"file:{LIVE / 'runs.sqlite'}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        rows = dict(db.execute("SELECT key, value FROM metameta").fetchall())
    return {key: str(rows[key]) for key in ("context_python", "damnit_python")}


def ctxsupport_of(damnit_python: str) -> Path:
    """The ``ctxsupport`` directory of the damnit the listener runs."""
    package = subprocess.run(
        [
            damnit_python,
            "-c",
            "import damnit, os; print(os.path.dirname(damnit.__file__))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(package) / "ctxsupport"


def evaluate(files: dict[str, bytes], python: str, ctxsupport: Path) -> dict[str, Any]:
    """Run ctxrunner ``ctx`` on a scratch copy of ``files`` and unpickle it."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
        work = Path(directory)
        for name, content in files.items():
            (work / name).write_bytes(content)
        env = os.environ | {"PYTHONPATH": str(ctxsupport), "MPLBACKEND": "Agg"}
        evaluated = subprocess.run(
            [python, "-m", "ctxrunner", "ctx", "context.py", "out.pkl"],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
        )
        if evaluated.returncode or not (work / "out.pkl").exists():
            tail = evaluated.stderr[-2000:]
            return {"error": f"ctxrunner exited {evaluated.returncode}: {tail}"}
        read = subprocess.run(
            [python, "-c", _UNPICKLE, str(work / "out.pkl")],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
        )
        if read.returncode:
            return {"error": f"unpickling failed: {read.stderr[-2000:]}"}
        return json.loads(read.stdout.splitlines()[-1])


def unified(old: bytes | None, new: bytes, name: str) -> str:
    before = old.decode().splitlines(keepends=True) if old is not None else []
    return "".join(
        difflib.unified_diff(
            before,
            new.decode().splitlines(keepends=True),
            f"live/{name}",
            f"repo/{name}",
        )
    )


def deployed(commit: str, deploy: Path) -> tuple[bool, str]:
    """Whether the deploy clone's checkout already contains ``commit``."""
    try:
        head = git("rev-parse", "HEAD", cwd=deploy)
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=deploy,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return False, f"the deploy clone ({deploy}) does not contain {commit[:12]}"
    return True, f"the deploy clone's HEAD {head[:12]} contains {commit[:12]}"


def install(
    live_dir: Path,
    new: dict[str, bytes],
    old: dict[str, bytes | None],
    expected: dict[str, str | None],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Back up, re-check, replace atomically and record. Returns the record."""
    backups = live_dir / "backups"
    backups.mkdir(exist_ok=True)
    when = stamp()
    saved = {}
    for name in FILES:
        if old[name] is not None:
            target = backups / f"{Path(name).stem}_{when}.py"
            target.write_bytes(old[name])
            saved[name] = target.name
    (backups / f"install_{when}.diff").write_text(
        "".join(unified(old[name], new[name], name) for name in FILES)
    )
    (backups / f"install_{when}.sha256").write_text(
        "".join(
            f"{sha}  {label}/{name}\n"
            for name in FILES
            for label, sha in (
                ("before", expected[name]),
                ("after", hashlib.sha256(new[name]).hexdigest()),
            )
            if sha
        )
    )

    for name in FILES:
        live = live_dir / name
        if sha256(live) != expected[name]:
            raise SystemExit(f"{live} changed while installing; nothing more replaced")
        temporary = live_dir / f".{name}.install-{os.getpid()}"
        temporary.write_bytes(new[name])
        if live.exists():
            shutil.copymode(live, temporary)
        os.replace(temporary, live)
        if sha256(live) != hashlib.sha256(new[name]).hexdigest():
            raise SystemExit(f"{live} does not hold what was installed")

    installed = {
        "files": {name: hashlib.sha256(new[name]).hexdigest() for name in FILES},
        "commit": record["commit"],
        "installed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "installed_by": getpass.getuser(),
        "host": socket.gethostname(),
        "backups": saved,
        "diff": f"install_{when}.diff",
        "previous": expected,
    }
    (backups / "installed.json").write_text(json.dumps(installed, indent=2))
    with open(backups / "install_log.jsonl", "a") as log:
        log.write(json.dumps(installed) + "\n")
    return installed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--install", action="store_true", help="really install")
    parser.add_argument(
        "--accept-live",
        metavar="ID",
        help="first install only: the live-state id this script printed",
    )
    parser.add_argument("--json", type=Path, help="where to write the report")
    parser.add_argument(
        "--live-dir",
        type=Path,
        default=LIVE,
        help="the directory DAMNIT reads (default: %(default)s); another one "
        "exercises the script on a copy",
    )
    parser.add_argument(
        "--deploy-dir",
        type=Path,
        default=DEPLOY,
        help="the checkout DAMNIT's venv runs (default: %(default)s)",
    )
    args = parser.parse_args()
    live_dir = args.live_dir.resolve()

    report: dict[str, Any] = report_header(
        script="install_context", install=args.install, live_dir=str(live_dir)
    )
    problems: list[str] = []

    dirty = git("status", "--porcelain", "--", str(SOURCE))
    commit = git("rev-parse", "HEAD")
    report["commit"] = commit
    if dirty:
        problems.append(f"src/amore has uncommitted changes:\n{dirty}")

    new = {name: (SOURCE / name).read_bytes() for name in FILES}
    old = {
        name: ((live_dir / name).read_bytes() if (live_dir / name).exists() else None)
        for name in FILES
    }
    live_shas = {name: sha256(live_dir / name) for name in FILES}
    state = live_state_id(live_shas)
    report["live"] = {"sha256": live_shas, "state_id": state}
    report["repo_sha256"] = {
        name: hashlib.sha256(new[name]).hexdigest() for name in FILES
    }

    diffs = {name: unified(old[name], new[name], name) for name in FILES}
    for name in FILES:
        print(f"\n===== {name}: live -> repo =====")
        print(diffs[name] or "(identical)")
    report["diff_lines"] = {name: diffs[name].count("\n") for name in FILES}
    print(f"\nlive state id: {state}")

    record_path = live_dir / "backups" / "installed.json"
    if record_path.exists():
        last = json.loads(record_path.read_text())
        report["last_install"] = last
        changed = [name for name in FILES if last["files"].get(name) != live_shas[name]]
        if changed:
            problems.append(
                f"{changed} changed since the last install ({last['installed_at']}, "
                f"{last['commit'][:12]}): someone edited the live directory. "
                "Reconcile their change into src/amore first"
            )
        report["guard"] = "matches the last install" if not changed else "changed"
    else:
        report["guard"] = "first install"
        if args.accept_live != state:
            problems.append(
                "first install: there is no record of what was installed before, "
                f"so read the diff above and pass --accept-live {state}"
                + (f" (got {args.accept_live!r})" if args.accept_live else "")
            )

    settings = damnit_settings()
    ctxsupport = ctxsupport_of(settings["damnit_python"])
    report["damnit"] = settings | {"ctxsupport": str(ctxsupport)}
    checks = {
        "repo": evaluate(new, settings["context_python"], ctxsupport),
        "live": evaluate(
            {n: c for n, c in old.items() if c is not None},
            settings["context_python"],
            ctxsupport,
        ),
    }
    for label, check in checks.items():
        verdict = "evaluates" if check["error"] is None else "FAILS"
        count = len(check.get("variables") or [])
        print(f"ctxrunner on the {label} context: {verdict}, {count} variables")
    if checks["repo"]["error"] is not None:
        problems.append(
            f"the repo context does not evaluate: {checks['repo']['error']}"
        )
    if checks["repo"].get("variables") and checks["live"].get("variables"):
        added = sorted(
            set(checks["repo"]["variables"]) - set(checks["live"]["variables"])
        )
        removed = sorted(
            set(checks["live"]["variables"]) - set(checks["repo"]["variables"])
        )
        report["variables"] = {"added": added, "removed": removed}
        print(f"variables added: {added or 'none'}; removed: {removed or 'none'}")
    report["ctxrunner"] = checks

    ok, why = deployed(commit, args.deploy_dir)
    report["deploy"] = why
    print(why)
    if not ok:
        problems.append(
            f"{why}: merge to main first, so DAMNIT has the code this context calls"
        )

    report["problems"] = problems
    if problems:
        print("\nnot installing:\n- " + "\n- ".join(problems))
    elif not args.install:
        print("\nevery check passes; rerun with --install to install")
    else:
        report["installed"] = install(live_dir, new, old, live_shas, report)
        print(f"\ninstalled {commit[:12]} into {live_dir}; backups in backups/")
    write_report(report, args.json, "install_context")
    return 1 if problems and args.install else 0


if __name__ == "__main__":
    sys.exit(main())
