"""What the scripts share: the figure tree, and a comparable header for JSON reports.

Not part of the package: these run as ``python scripts/<name>.py``, which puts
this directory on ``sys.path``.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.threadenv import THREAD_ENV

__all__ = [
    "FIGURES",
    "REPO",
    "droplets_dir",
    "git_commit",
    "overview_dir",
    "report_header",
    "stamp",
    "write_index",
    "write_report",
]


REPO = Path(__file__).resolve().parents[1]
#: The figure scripts' output (gitignored). A droplet's figures go to
#: ``droplets/<technique>/<Droplet.name>/``, figures across droplets to
#: ``overview/<kind>/``.
FIGURES = REPO / "figures"


def droplets_dir(technique: str) -> Path:
    """The folder of every droplet's `technique` figures ("saxs", "xpcs")."""
    return FIGURES / "droplets" / technique


def overview_dir(kind: str) -> Path:
    """The folder of one kind of overview ("saxs", "saxs_matched_tstar", ...)."""
    return FIGURES / "overview" / kind


def write_index(technique: str, attempts: int = 120) -> None:
    """Gather every droplet's ``run.json`` into ``droplets/<technique>/index.json``.

    An entry already in the index stays for a folder without a ``run.json``.
    Parallel jobs each call this after writing their own ``run.json``: the lock
    makes the gathers take turns, so the last one sees every record, and the
    rename means a reader never sees a partial index.

    :param attempts: one-second waits for another job's lock before giving up.
    :raises OutputBusy: the lock stayed held for every attempt.
    """
    from analysis.common.lock import OutputBusy, output_lock

    index = droplets_dir(technique) / "index.json"
    for _ in range(attempts):
        try:
            with output_lock(index):
                merged = (
                    {r["id"]: r for r in json.loads(index.read_text())}
                    if index.exists()
                    else {}
                )
                for path in index.parent.glob("*/run.json"):
                    record = json.loads(path.read_text())
                    merged[record["id"]] = record
                tmp = index.with_name(f".{index.name}.{uuid.uuid4().hex}.tmp")
                tmp.write_text(
                    json.dumps(sorted(merged.values(), key=lambda r: r["id"]), indent=1)
                )
                os.replace(tmp, index)
            return
        except OutputBusy:
            time.sleep(1)
    raise OutputBusy(f"{index} stayed locked for {attempts} s")


def git_commit() -> str:
    """The commit this tree is on, or why it could not be read.

    :returns: the hex sha, or an ``"unavailable: ..."`` string. Never raises:
        a report that cannot name its commit is still worth writing.
    """
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


def stamp() -> str:
    """UTC timestamp for a report filename, ``20260914T131500Z``."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def report_header(**extra: Any) -> dict[str, Any]:
    """The keys every script's JSON starts with, so reports are comparable.

    :param extra: script-specific keys, merged on top.
    :returns: the header. ``thread_env`` records what was actually pinned
        rather than a hand-copied subset of it.
    """
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV},
    } | extra


def write_report(report: dict[str, Any], path: Path | None, stem: str) -> Path:
    """Write ``report`` as JSON, defaulting to a stamped file beside the script.

    :param report: the report.
    :param path: an explicit destination, usually ``args.json``.
    :param stem: filename stem used when ``path`` is None, e.g. ``"run_triage"``.
    :returns: where it was written.
    """
    destination = path or Path(__file__).with_name(f"{stem}_{stamp()}.json")
    destination.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {destination}")
    return destination
