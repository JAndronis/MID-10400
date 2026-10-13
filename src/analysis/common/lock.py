"""One writer per output file: a lock file beside it (CLAUDE.md pitfall 26).

DAMNIT can launch a cluster variable twice for the same run: requesting a
variable that depends on it starts it again while the first instance still
runs. Two jobs then write one file. :func:`output_lock` makes the second one
refuse instead.

The lock is ``<output>.lock``, created with ``O_CREAT | O_EXCL`` — atomic on
GPFS across nodes — and holding who took it: host, pid, user, Slurm job and
time. A second process that finds it asks whether the holder is still alive:

* the holder ran in a Slurm job → ``squeue --clusters=all``. Listed and not
  finished means alive. DAMNIT submits non-cluster variables to another cluster
  (``solaris``), hence ``--clusters=all``;
* the holder ran outside Slurm on this host → whether its pid exists;
* anything else cannot be checked from here, and is treated as alive.

An alive holder raises :class:`OutputBusy`. A dead one — a job that hit its
time limit, a killed kernel — leaves a stale lock behind, because neither
SIGKILL nor Slurm's SIGTERM runs ``finally`` blocks; that lock is taken over,
logged, and recorded by the pass in the file's provenance.

``squeue --clusters=all -j <id>`` exits 1 even when the job is running, since
every other cluster answers "Invalid job id specified". The verdict therefore
comes from what it prints, never from its exit status.
"""

from __future__ import annotations

import json
import logging
import os
import pwd
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "JobStateUnknown",
    "LockHolder",
    "OutputBusy",
    "OutputLock",
    "lock_path",
    "output_lock",
    "slurm_job_state",
]

log = logging.getLogger(__name__)

#: Seconds a lock file may stay unparsable before it counts as abandoned. The
#: holder writes it immediately after creating it, so only a holder that died
#: between the two leaves it empty for longer than a moment.
UNPARSABLE_GRACE_S = 60.0

#: Slurm states after which no process of the job is left. ``COMPLETING`` is
#: deliberately absent: the job's processes may still be exiting.
FINISHED_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)

#: Attempts at creating the lock before giving up. Each failed attempt has
#: either found a live holder (and raised) or taken a dead one's lock over, so
#: more than two means the lock keeps changing hands under us.
_ATTEMPTS = 3


class OutputBusy(RuntimeError):
    """Another process holds, or may hold, the lock on this output."""


class JobStateUnknown(RuntimeError):
    """``squeue`` could not say whether a job is still alive."""


def lock_path(output: str | Path) -> Path:
    """The lock file guarding ``output``: ``<output>.lock`` beside it."""
    output = Path(output)
    return output.with_name(f"{output.name}.lock")


#: What :meth:`LockHolder.parse` fills in for a field a lock file lacks.
_DEFAULTS: dict[str, Any] = {
    "output": "",
    "user": "?",
    "slurm_job_id": None,
    "slurm_cluster": None,
    "started_at": 0.0,
    "started": "?",
}


def _user() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return os.environ.get("USER", str(os.getuid()))


@dataclass(frozen=True, slots=True)
class LockHolder:
    """Who took a lock, as written into the lock file."""

    output: str
    host: str
    pid: int
    user: str
    slurm_job_id: str | None
    slurm_cluster: str | None
    started_at: float
    started: str
    #: Random per acquisition, so a process only ever removes its own lock.
    token: str

    @classmethod
    def current(cls, output: str | Path) -> LockHolder:
        """This process, about to take the lock on ``output``."""
        now = time.time()
        return cls(
            output=str(output),
            host=socket.gethostname(),
            pid=os.getpid(),
            user=_user(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_cluster=os.environ.get("SLURM_CLUSTER_NAME"),
            started_at=now,
            started=datetime.fromtimestamp(now, UTC).isoformat(timespec="seconds"),
            token=uuid.uuid4().hex,
        )

    @classmethod
    def parse(cls, text: str) -> LockHolder | None:
        """The holder a lock file names, or ``None`` if it names none.

        Only ``host``, ``pid`` and ``token`` are required — they are what the
        liveness check and the release need — so that a lock written by another
        version of this module is still judged on its holder, not taken for an
        abandoned one.
        """
        try:
            record = json.loads(text)
            return cls(
                **{
                    f.name: record.get(f.name, _DEFAULTS.get(f.name))
                    for f in fields(cls)
                }
                | {name: record[name] for name in ("host", "pid", "token")}
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            return None

    def __post_init__(self) -> None:
        if not isinstance(self.pid, int) or not self.host or not self.token:
            raise TypeError("a lock holder needs a host, an integer pid and a token")

    def describe(self) -> str:
        job = f", Slurm job {self.slurm_job_id}" if self.slurm_job_id else ""
        return f"pid {self.pid} on {self.host}{job} ({self.user}, since {self.started})"

    def record(self) -> dict[str, Any]:
        """For provenance: everything but the token."""
        record = asdict(self)
        del record["token"]
        return record


@dataclass(frozen=True, slots=True)
class OutputLock:
    """A held lock, and the stale one it replaced, if any."""

    path: Path
    holder: LockHolder
    #: The previous holder's record and why it counted as dead, when this
    #: acquisition took a stale lock over.
    takeover: dict[str, Any] | None = None

    def provenance(self) -> dict[str, Any]:
        """What a pass merges into its provenance record."""
        record: dict[str, Any] = {"lock": self.holder.record()}
        if self.takeover is not None:
            record["lock_takeover"] = self.takeover
        return record


def slurm_job_state(job_id: str) -> str | None:
    """The state Slurm reports for a job, or ``None`` once it has forgotten it.

    Slurm forgets a finished job ``MinJobAge`` after it ends (90 s on Maxwell),
    so ``None`` and a finished state mean the same thing to a caller.

    :raises JobStateUnknown: ``squeue`` failed for any reason other than not
        knowing the job, so the job's state is unknown — not absent.
    """
    command = [
        "squeue",
        "--clusters=all",
        "--noheader",
        f"--jobs={job_id}",
        "--format=%i|%T",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        raise JobStateUnknown(f"{' '.join(command)} failed: {error!r}") from error
    # Every line is this job: an array task is shown as <array>_<task> rather
    # than by the id it was asked for, so the id column cannot be matched.
    states = [
        line.rpartition("|")[2].strip()
        for line in result.stdout.splitlines()
        if "|" in line
    ]
    if states:
        unfinished = [state for state in states if state not in FINISHED_STATES]
        return unfinished[0] if unfinished else states[0]
    complaints = [line for line in result.stderr.splitlines() if line.strip()]
    if all("Invalid job id" in line for line in complaints):
        return None
    raise JobStateUnknown(
        f"{' '.join(command)} exited {result.returncode} and listed nothing: "
        + " / ".join(complaints)
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, it is just someone else's
    return True


def _holder_alive(holder: LockHolder) -> tuple[bool, str]:
    """Whether the holder may still be writing, and the evidence.

    :raises OutputBusy: the holder cannot be checked from here.
    """
    own_job = os.environ.get("SLURM_JOB_ID")
    if holder.slurm_job_id and holder.slurm_job_id != own_job:
        try:
            state = slurm_job_state(holder.slurm_job_id)
        except JobStateUnknown as error:
            raise OutputBusy(
                f"{holder.output} is locked by {holder.describe()}, and whether "
                f"that job is still running cannot be checked: {error}"
            ) from error
        if state is None:
            return False, f"Slurm no longer knows job {holder.slurm_job_id}"
        if state in FINISHED_STATES:
            return False, f"Slurm job {holder.slurm_job_id} is {state}"
        return True, f"Slurm job {holder.slurm_job_id} is {state}"
    if holder.host == socket.gethostname():
        if holder.pid == os.getpid():
            return True, "this very process already holds it"
        if _pid_alive(holder.pid):
            return True, f"pid {holder.pid} is running on {holder.host}"
        return False, f"pid {holder.pid} no longer exists on {holder.host}"
    raise OutputBusy(
        f"{holder.output} is locked by {holder.describe()}, outside Slurm on "
        "another host, so whether it is still running cannot be checked from "
        f"here. If that process is gone, delete {lock_path(holder.output)} by hand"
    )


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except FileNotFoundError:
        return None


def _take_over(path: Path, seen: str) -> None:
    """Remove a stale lock, but only the one that was judged stale.

    Between reading the lock and removing it, another process may have taken it
    over and written its own. Renaming first and comparing afterwards means a
    fresh lock is put back rather than deleted.

    :raises OutputBusy: the lock changed hands while it was being taken over.
    """
    aside = path.with_name(f"{path.name}.stale-{uuid.uuid4().hex[:12]}")
    try:
        os.rename(path, aside)
    except FileNotFoundError:
        return  # someone else removed it first; the caller simply retries
    if (_read(aside) or "") == seen:
        aside.unlink()
        return
    try:
        os.link(aside, path)  # back where it was, unless the slot is taken
    except FileExistsError:
        log.error(
            "%s changed hands while being taken over; the lock that was moved "
            "aside is kept as %s for inspection",
            path,
            aside,
        )
    else:
        aside.unlink()
    raise OutputBusy(f"{path} changed hands while it was being taken over")


def _release(path: Path, holder: LockHolder) -> None:
    current = LockHolder.parse(_read(path) or "")
    if current is not None and current.token == holder.token:
        path.unlink()
        return
    # Someone judged this process dead and took the lock: two writers may have
    # overlapped, so this is loud rather than quietly tidied up.
    log.error(
        "the lock %s is no longer this process's (now %s); leaving it in place",
        path,
        current.describe() if current else "absent or unreadable",
    )


@contextmanager
def output_lock(output: str | Path) -> Iterator[OutputLock]:
    """Hold the lock on ``output`` for the duration of the block.

    :param output: the file to be written; the lock goes beside it.
    :yields: the held lock, whose :meth:`OutputLock.provenance` a pass records.
    :raises OutputBusy: another process holds the lock and is, or may be,
        still running.
    """
    path = lock_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    holder = LockHolder.current(output)
    takeover: dict[str, Any] | None = None

    for _ in range(_ATTEMPTS):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(asdict(holder), handle)
                handle.flush()
                os.fsync(handle.fileno())
            break

        seen = _read(path)
        if seen is None:
            continue  # released between the create and the read
        previous = LockHolder.parse(seen)
        if previous is None:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < UNPARSABLE_GRACE_S:
                raise OutputBusy(f"{path} is being written by another process")
            reason = f"the lock file has been unparsable for {age:.0f} s"
            record: dict[str, Any] = {"unparsable": seen[:200]}
        else:
            alive, reason = _holder_alive(previous)
            if alive:
                raise OutputBusy(
                    f"{output} is being written by {previous.describe()}: {reason}"
                )
            record = previous.record()
        log.warning("taking over the stale lock %s: %s", path, reason)
        _take_over(path, seen)
        takeover = {
            "previous": record,
            "reason": reason,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    else:
        raise OutputBusy(f"{path} kept changing hands; not writing {output}")

    try:
        yield OutputLock(path, holder, takeover)
    finally:
        _release(path, holder)
