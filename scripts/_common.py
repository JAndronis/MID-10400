"""Boilerplate the four acceptance and probe scripts share.

Not part of the package: these run as ``python scripts/<name>.py``, which puts
this directory on ``sys.path``. Only pure functions of their arguments live
here — nothing that opens a run or touches detector data, so a change here
cannot alter what a gate decides.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.threadenv import THREAD_ENV

__all__ = ["git_commit", "report_header", "stamp", "write_report"]


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
    """The keys every script's JSON starts with, so the four are comparable.

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
    :param stem: filename stem used when ``path`` is None, e.g. ``"w1_facts"``.
    :returns: where it was written.
    """
    destination = path or Path(__file__).with_name(f"{stem}_{stamp()}.json")
    destination.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {destination}")
    return destination
