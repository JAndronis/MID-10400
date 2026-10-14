"""Submitting node scripts to SLURM, and the proposal's canonical paths.

The node scripts submit themselves: without ``--local`` each run becomes one
job, whose batch script calls the same script again with ``--local``. This
module holds the parts of that which can be tested without a cluster — the
path rewriting, the batch script, the ``sbatch`` call behind an injectable
runner, the per-run failure ledger and the command-line surgery.
``scripts/_common.py`` holds the argparse glue.

Nothing here calls ``srun``. On a whole-node partition ``srun`` without
``-n 1`` starts one copy per CPU group (CLAUDE.md pitfall 28); a batch script
with one task runs the command exactly once.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from analysis.threadenv import THREAD_ENV

__all__ = [
    "DEFAULT_PARTITION",
    "PROPOSAL_ROOT",
    "PROPOSAL_SUBDIRS",
    "THREADS_PER_CORE",
    "SubmitError",
    "batch_script",
    "prop_path",
    "replace_option",
    "run_each",
    "submit",
]

log = logging.getLogger(__name__)

#: The proposal directory, in the ``/gpfs/exfel/exp`` form every path is
#: written back to.
PROPOSAL_ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400")

#: Its subdirectories, each a symlink into ``/gpfs/exfel/u`` or ``/gpfs/exfel/d``.
PROPOSAL_SUBDIRS = ("usr", "proc", "raw", "red", "scratch")

#: The partition DAMNIT submits this proposal's cluster variables to: the
#: ``slurm_partition`` key of ``usr/Shared/amore/runs.sqlite``'s ``metameta``
#: table (checked 2026-10-13). It mixes node classes — 2 × 10 to 2 × 64 cores —
#: so a benchmark that must be comparable with an earlier one passes
#: ``constraint``.
DEFAULT_PARTITION = "allcpu"

#: Hardware threads per core on every ``allcpu`` node (``sinfo -p allcpu -o
#: %z`` gives ``S:C:2`` for every node class, 2026-10-13). SLURM counts CPUs as
#: threads, so a job that needs N physical cores asks for 2N CPUs: asked for
#: 36, it may be given a 2 × 10-core node and its 36 workers 20 cores.
THREADS_PER_CORE = 2

#: What ``sbatch`` accepts as ``--time``: ``MM``, ``MM:SS``, ``HH:MM:SS``,
#: ``D-HH``, ``D-HH:MM`` or ``D-HH:MM:SS``.
_TIME_LIMIT = re.compile(r"^(\d+-\d{1,2}(:\d{2}){0,2}|\d+(:\d{2}){0,2})$")


class SubmitError(RuntimeError):
    """``sbatch`` refused a job or answered with something that is not a job id."""


def prop_path(path: str | os.PathLike[str], root: Path = PROPOSAL_ROOT) -> str:
    """``path`` made absolute, written in the proposal's canonical form.

    The proposal's subdirectories are symlinks, so a resolved path names a
    storage tier (``/gpfs/exfel/u/usr/…``, ``/gpfs/exfel/d/proc/…``) rather
    than the proposal. A batch script written with those, or with a private
    alias such as ``~/p010400``, works only for as long as the data stay where
    they are and only for whoever can follow the alias.

    The parent directories are resolved and the final component is kept as
    given: a venv's ``bin/python`` is a symlink to the interpreter it was
    built from, and resolving it would drop the venv (CLAUDE.md pitfall 22).

    :param path: any path; relative ones are taken from the working directory.
    :param root: the proposal directory, in its ``/gpfs/exfel/exp`` form.
    :returns: ``<root>/<subdir>/…`` for a path inside one of the proposal's
        subdirectories, otherwise the path with its parents resolved.
    """
    given = Path(os.path.abspath(path))
    resolved = Path(os.path.realpath(given.parent)) / given.name
    for sub in PROPOSAL_SUBDIRS:
        real = Path(os.path.realpath(root / sub))
        if resolved == real:
            return str(root / sub)
        if real in resolved.parents:
            return str(root / sub / resolved.relative_to(real))
    return str(resolved)


def batch_script(
    command: Sequence[str],
    *,
    name: str,
    log_dir: str | os.PathLike[str],
    cpus: int,
    limit: str,
    partition: str = DEFAULT_PARTITION,
    after: str | Iterable[str] | None = None,
    constraint: str | None = None,
    reservation: str | None = None,
    workdir: str | os.PathLike[str] | None = None,
    threads_per_core: int = THREADS_PER_CORE,
) -> str:
    """The batch script that runs ``command`` once, as one single-node task.

    :param command: the program and its arguments; quoted with ``shlex``.
    :param name: the job name, also the log's subject.
    :param log_dir: where ``slurm-<jobid>.out`` goes; it must exist when the
        job starts, which :func:`submit` sees to.
    :param cpus: physical cores for the one task — the worker count of the
        command. The job asks for ``cpus * threads_per_core`` CPUs, which is
        what makes SLURM pick a node with that many cores. ``allcpu`` hands
        over the whole node either way (``select/linear``): the task's affinity
        is every CPU of the node, not the ones asked for (probed 2026-10-13 on a
        Gold 6140 and an EPYC 9374F node with ``--cpus-per-task=36``).
    :param limit: the time limit, in any form ``sbatch --time`` accepts.
    :param partition: defaults to DAMNIT's.
    :param after: job id(s) that must succeed first. A job that fails cancels
        this one rather than leaving it pending forever.
    :param constraint: node features, e.g. ``"9374F"`` or ``"Gold-6240|Gold-6140"``.
    :param reservation: e.g. the proposal's beamtime reservation.
    :param workdir: the job's working directory, so that relative paths in
        ``command`` mean what they meant where it was submitted.
    :param threads_per_core: of the nodes the job may land on.
    :raises ValueError: an option that ``sbatch`` would misread.
    """
    for label, count in (("cpus", cpus), ("threads_per_core", threads_per_core)):
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"{label} must be a positive integer, got {count!r}")
    if not _TIME_LIMIT.match(limit):
        raise ValueError(f"time limit {limit!r} is not in a form sbatch accepts")
    for label, value in (
        ("name", name),
        ("partition", partition),
        ("constraint", constraint),
        ("reservation", reservation),
    ):
        # #SBATCH lines are not shell-parsed, so there is nothing to quote with
        if value is not None and (not value or re.search(r"\s", value)):
            raise ValueError(f"{label} {value!r} is empty or contains whitespace")

    options = [
        f"--partition={partition}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cpus * threads_per_core}",
        f"--time={limit}",
        f"--job-name={name}",
        f"--output={log_dir}/slurm-%j.out",
    ]
    if workdir is not None:
        options.append(f"--chdir={workdir}")
    if constraint is not None:
        options.append(f"--constraint={constraint}")
    if reservation is not None:
        options.append(f"--reservation={reservation}")
    dependencies = _job_ids(after)
    if dependencies:
        options += [
            f"--dependency=afterok:{':'.join(dependencies)}",
            "--kill-on-invalid-dep=yes",
        ]
    pinned = " ".join(f"{variable}=1" for variable in THREAD_ENV)
    return "\n".join(
        [
            "#!/bin/bash",
            f"# {cpus} physical cores, {threads_per_core} threads each",
            *(f"#SBATCH {option}" for option in options),
            f"export {pinned}",
            'echo "job $SLURM_JOB_ID on $(hostname), $(date -Is)"',
            shlex.join(command),
            "",
        ]
    )


def submit(
    command: Sequence[str],
    name: str,
    out: str | os.PathLike[str],
    cpus: int = 1,
    limit: str = "01:00:00",
    partition: str = DEFAULT_PARTITION,
    dry_run: bool = False,
    after: str | Iterable[str] | None = None,
    *,
    constraint: str | None = None,
    reservation: str | None = None,
    workdir: str | os.PathLike[str] | None = None,
    threads_per_core: int = THREADS_PER_CORE,
    runner: Callable[..., Any] = subprocess.run,
) -> str | None:
    """Submit ``command`` as one job through a temporary batch script.

    The script is deleted once ``sbatch`` has read it; SLURM keeps its own
    copy. The arguments are those of :func:`batch_script`, with ``out`` as its
    ``log_dir``.

    :param dry_run: print the batch script and submit nothing.
    :param runner: ``subprocess.run`` or a stand-in; the tests inject one.
    :returns: the job id, or ``None`` on a dry run.
    :raises SubmitError: ``sbatch`` failed or did not answer with a job id.
    """
    script = batch_script(
        command,
        name=name,
        log_dir=out,
        cpus=cpus,
        limit=limit,
        partition=partition,
        after=after,
        constraint=constraint,
        reservation=reservation,
        workdir=workdir,
        threads_per_core=threads_per_core,
    )
    if dry_run:
        print(script)
        return None
    # SLURM does not create the log directory; a job whose log cannot be
    # opened fails without a word anywhere
    Path(out).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.sh"
        path.write_text(script)
        result = runner(
            ["sbatch", "--parsable", str(path)], capture_output=True, text=True
        )
    if result.returncode:
        raise SubmitError(f"{name}: sbatch failed: {result.stderr.strip()}")
    # --parsable answers "<jobid>" or "<jobid>;<cluster>"
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise SubmitError(f"{name}: sbatch answered {result.stdout!r}, not a job id")
    log.info(
        "%s: job %s (%d cores, %s, %s) -> %s/slurm-%s.out",
        name,
        job,
        cpus,
        limit,
        partition,
        out,
        job,
    )
    return job


def run_each[T](
    items: Iterable[T], work: Callable[[T], int | None], *, label: str = "run"
) -> list[T]:
    """Call ``work`` on every item, and carry on past the ones that fail.

    One bad run is not a reason to lose the others in a multi-run job; nor is
    it a reason for the job to report success.

    :param work: returns an exit status, as a script's ``main`` does — ``0`` or
        ``None`` is success, anything else a failure. An exception is logged
        with its traceback and counted as a failure. ``KeyboardInterrupt``
        still stops the loop.
    :param label: what an item is, for the log.
    :returns: the items that failed, in order.
    """
    items = list(items)
    failed = []
    for item in items:
        try:
            status = work(item)
        except Exception:  # noqa: BLE001 - logged, counted, never swallowed
            log.exception("%s %s failed", label, item)
            failed.append(item)
            continue
        if status not in (None, 0):
            log.error("%s %s failed with status %r", label, item, status)
            failed.append(item)
    if failed:
        log.error(
            "%d of %d %ss failed: %s",
            len(failed),
            len(items),
            label,
            " ".join(map(str, failed)),
        )
    return failed


def replace_option(
    argv: Sequence[str],
    names: Iterable[str],
    values: Sequence[str] | None = None,
    *,
    nargs: int | str = "+",
) -> list[str]:
    """``argv`` without every occurrence of an option, then with it set anew.

    Rewrites a command line for one job: the run list becomes one run, and
    ``--dry-run`` goes. Matching is exact, so the parser must not accept
    abbreviations (``allow_abbrev=False``); otherwise ``--dry`` would survive
    and ``--run`` could not be told from a prefix of ``--run-dir``.

    :param names: the option's spellings, e.g. ``("--run",)`` or
        ``("-w", "--workers")``. Both ``--run 1 2`` and ``--run=1`` are removed.
    :param values: what to append under ``names[0]``; ``None`` appends nothing
        (a flag to drop), ``[]`` appends the bare flag.
    :param nargs: how many tokens follow the option: an int, or ``"+"`` for
        every token up to the next one that starts with ``-``.
    """
    names = tuple(names)
    kept: list[str] = []
    tokens = list(argv)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in names:
            i += 1
            if nargs == "+":
                while i < len(tokens) and not tokens[i].startswith("-"):
                    i += 1
            else:
                i += int(nargs)
            continue
        if any(token.startswith(f"{name}=") for name in names):
            i += 1
            continue
        kept.append(token)
        i += 1
    if values is not None:
        kept += [names[0], *values]
    return kept


def _job_ids(after: str | Iterable[str] | None) -> list[str]:
    """Job ids to depend on, each checked to be one."""
    if after is None:
        return []
    ids = [after] if isinstance(after, str | int) else list(after)
    ids = [str(job) for job in ids]
    for job in ids:
        if not job.isdigit():
            raise ValueError(f"dependency {job!r} is not a job id")
    return ids
