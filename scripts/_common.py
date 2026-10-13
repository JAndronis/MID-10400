"""Boilerplate the acceptance, probe and benchmark scripts share.

Not part of the package: these run as ``python scripts/<name>.py``, which puts
this directory on ``sys.path``. Only pure functions of their arguments live
here — nothing that opens a run or touches detector data, so a change here
cannot alter what a gate decides.

Node scripts submit themselves (:func:`add_job_args`, :func:`dispatch`)::

    python scripts/x.py --run 423 426 -w 36            # one SLURM job per run
    python scripts/x.py --run 423 -w 36 --dry-run      # print the batch script
    python scripts/x.py --run 423 -w 36 --local        # here, now

Inside a job (``SLURM_JOB_ID`` set) a script works in place, as with
``--local``. The testable parts live in :mod:`analysis.common.slurm`, imported
here only inside functions: importing it runs ``analysis.common``'s
``__init__``, which imports numpy, and every script imports this module before
it pins the thread count.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.threadenv import THREAD_ENV

__all__ = [
    "add_job_args",
    "configure_logging",
    "dispatch",
    "git_commit",
    "report_header",
    "stamp",
    "write_report",
]

log = logging.getLogger("scripts")

#: Where the jobs' ``slurm-<jobid>.out`` logs go, one directory per script:
#: scratch, so that they stay out of the repository.
SLURM_LOG_ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400/scratch/slurm_logs")


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


def configure_logging(*names: str) -> None:
    """Our loggers at INFO, everything else at WARNING.

    :param names: loggers besides ``analysis`` and ``scripts`` to open up,
        e.g. a script's own.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in ("analysis", "scripts", *names):
        logging.getLogger(name).setLevel(logging.INFO)


def add_job_args(
    parser: argparse.ArgumentParser,
    *,
    workers: int = 36,
    time: str = "01:00:00",
) -> None:
    """Add ``-w``, ``-t``, ``--local``, ``--dry-run`` and the SLURM options.

    Also turns off argparse's prefix matching: :func:`dispatch` rewrites the
    command line by exact option name, so ``--dry`` must not be accepted as
    ``--dry-run``.

    :param workers: the default for ``-w``.
    :param time: the default for ``-t``.
    """
    from analysis.common.slurm import DEFAULT_PARTITION

    parser.allow_abbrev = False
    group = parser.add_argument_group("SLURM")
    group.add_argument(
        "-w",
        "--workers",
        type=int,
        default=workers,
        help=(
            "worker processes, and the physical cores each job asks for "
            f"(default {workers})"
        ),
    )
    group.add_argument(
        "-t", "--time", default=time, help=f"time limit of each job (default {time})"
    )
    group.add_argument(
        "--local",
        action="store_true",
        help="run here instead of submitting; implied inside a SLURM job",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="print each job's batch script and submit nothing",
    )
    group.add_argument("--partition", default=DEFAULT_PARTITION)
    group.add_argument(
        "--constraint",
        default=None,
        help="node features, e.g. 9374F or 'Gold-6240|Gold-6140'",
    )
    group.add_argument("--reservation", default=None)
    group.add_argument(
        "--after", default=None, help="a job id that must succeed first (afterok)"
    )


def dispatch(
    args: argparse.Namespace,
    runs: Sequence[int],
    work: Callable[[int], int | None],
    *,
    run_option: str = "--run",
    argv: Sequence[str] | None = None,
) -> int:
    """Do ``work`` for each run here, or submit one job per run.

    :param args: parsed with :func:`add_job_args`'s options.
    :param runs: the runs asked for.
    :param work: one run's work; returns an exit status, as ``main`` does.
    :param run_option: the option that names the runs; each job's command line
        carries it with that job's run alone.
    :param argv: the command line to rewrite, defaulting to ``sys.argv[1:]``.
    :returns: the exit status: non-zero when any run failed here, or when a
        submission was refused. Whether a submitted job succeeds is in its log.
    """
    from analysis.common import slurm

    if args.local or "SLURM_JOB_ID" in os.environ:
        _warn_on_oversubscription(args.workers)
        failed = slurm.run_each(runs, work)
        return 1 if failed else 0

    script = Path(sys.argv[0])
    rest = list(sys.argv[1:] if argv is None else argv)
    for name, nargs in (("--dry-run", 0), ("--local", 0), ("--after", 1)):
        rest = slurm.replace_option(rest, [name], nargs=nargs)
    out = SLURM_LOG_ROOT / script.stem
    status = 0
    for run in runs:
        command = [
            slurm.prop_path(sys.executable),
            slurm.prop_path(script),
            *slurm.replace_option(rest, [run_option], [str(run)]),
            "--local",
        ]
        try:
            job = slurm.submit(
                command,
                f"{script.stem}_r{run:04d}",
                out,
                cpus=args.workers,
                limit=args.time,
                partition=args.partition,
                dry_run=args.dry_run,
                after=args.after,
                constraint=args.constraint,
                reservation=args.reservation,
                workdir=slurm.prop_path(Path.cwd()),
            )
        except slurm.SubmitError as error:
            log.error("%s", error)
            status = 1
            continue
        if job is not None:
            print(f"r{run:04d}: job {job}, log {out}/slurm-{job}.out")
    return status


def _warn_on_oversubscription(workers: int) -> None:
    """Say so when there are more workers than physical cores to run them."""
    from analysis.common.cpu import physical_cores

    cores = physical_cores()
    if cores is not None and workers > cores:
        log.warning(
            "%d workers on %d physical cores: the extra ones share hyperthreads "
            "(CLAUDE.md pitfall 10)",
            workers,
            cores,
        )
