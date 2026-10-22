"""The output lock: one writer per file, stale locks taken over.

Slurm is never called: ``slurm_job_state`` is replaced, and its own parsing is
tested against recorded ``squeue`` output, including the exit status 1 that
``--clusters=all`` gives for a job that is running.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import pytest

from analysis.common import lock as lock_module
from analysis.common.lock import (
    JobStateUnknown,
    OutputBusy,
    lock_path,
    output_lock,
    slurm_job_state,
)


def write_holder(output, **changes):
    """A lock file on ``output`` as another process would have left it."""
    values = {
        "output": str(output),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "user": "someone",
        "slurm_job_id": None,
        "slurm_cluster": None,
        "slurm_restart_count": 0,
        "started_at": time.time(),
        "started": "2026-10-13T12:00:00+00:00",
        "token": "theirs",
    } | changes
    lock_path(output).write_text(json.dumps(values))
    return values


@pytest.fixture
def job_state(monkeypatch):
    """Replace Slurm: ``states[job_id]`` is what squeue would report."""
    states: dict[str, str | None] = {}

    def fake(job_id):
        value = states.get(job_id, None)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(lock_module, "slurm_job_state", fake)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    return states


def dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


# ── acquiring and releasing ───────────────────────────────────────────────────
def test_the_lock_names_its_holder_and_goes_away_after(tmp_path, job_state):
    output = tmp_path / "out.h5"
    with output_lock(output) as held:
        record = json.loads(lock_path(output).read_text())
        assert record["pid"] == os.getpid()
        assert record["host"] == socket.gethostname()
        assert record["output"] == str(output)
        assert held.takeover is None
        assert "token" not in held.provenance()["lock"]
        assert "lock_takeover" not in held.provenance()
    assert not lock_path(output).exists()


def test_the_lock_is_released_when_the_block_raises(tmp_path, job_state):
    output = tmp_path / "out.h5"
    with pytest.raises(RuntimeError, match="boom"), output_lock(output):
        raise RuntimeError("boom")
    assert not lock_path(output).exists()


def test_a_lock_someone_else_took_is_left_in_place(tmp_path, job_state):
    """Never remove a lock that is not this process's own."""
    output = tmp_path / "out.h5"
    with output_lock(output):
        write_holder(output, token="usurper")
    assert json.loads(lock_path(output).read_text())["token"] == "usurper"


# ── conflicts ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "COMPLETING"])
def test_a_live_slurm_job_holding_the_lock_is_refused(tmp_path, job_state, state):
    output = tmp_path / "out.h5"
    before = write_holder(output, slurm_job_id="4242", host="max-exfl001")
    job_state["4242"] = state
    with pytest.raises(OutputBusy, match=f"4242 is {state}"), output_lock(output):
        pytest.fail("acquired a lock a live job holds")
    assert json.loads(lock_path(output).read_text()) == before


def test_a_live_process_on_this_host_holding_the_lock_is_refused(tmp_path, job_state):
    output = tmp_path / "out.h5"
    write_holder(output, pid=os.getppid())
    with pytest.raises(OutputBusy, match="is running"), output_lock(output):
        pytest.fail("acquired a lock a live process holds")


def test_this_process_cannot_take_its_own_lock_twice(tmp_path, job_state):
    output = tmp_path / "out.h5"
    with output_lock(output), pytest.raises(OutputBusy, match="this very process"):
        with output_lock(output):
            pytest.fail("acquired the same lock twice")


def test_a_holder_that_cannot_be_checked_is_treated_as_alive(tmp_path, job_state):
    """Another host and no Slurm job: nothing here can say it has finished."""
    output = tmp_path / "out.h5"
    write_holder(output, host="elsewhere.desy.de")
    with pytest.raises(OutputBusy, match="delete .* by hand"), output_lock(output):
        pytest.fail("took over a lock whose holder could not be checked")
    assert lock_path(output).exists()


def test_a_failed_squeue_is_not_taken_for_a_finished_job(tmp_path, job_state):
    output = tmp_path / "out.h5"
    write_holder(output, slurm_job_id="4242", host="max-exfl001")
    job_state["4242"] = JobStateUnknown("slurm_load_jobs error: timed out")
    with pytest.raises(OutputBusy, match="cannot be checked"), output_lock(output):
        pytest.fail("took over a lock on a guess")


def test_a_freshly_created_empty_lock_is_busy(tmp_path, job_state):
    """The holder writes its record just after creating the file."""
    output = tmp_path / "out.h5"
    lock_path(output).write_text("")
    with pytest.raises(OutputBusy, match="being written"), output_lock(output):
        pytest.fail("took over a lock that is still being written")


# ── stale locks ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("state", [None, "TIMEOUT", "CANCELLED", "COMPLETED"])
def test_a_finished_job_s_lock_is_taken_over_and_recorded(tmp_path, job_state, state):
    output = tmp_path / "out.h5"
    previous = write_holder(output, slurm_job_id="4242", host="max-exfl001")
    job_state["4242"] = state
    with output_lock(output) as held:
        record = json.loads(lock_path(output).read_text())
        assert record["pid"] == os.getpid()
        takeover = held.provenance()["lock_takeover"]
        assert takeover["previous"]["slurm_job_id"] == "4242"
        assert takeover["previous"]["started"] == previous["started"]
        assert "token" not in takeover["previous"]
        assert "4242" in takeover["reason"]
    assert not lock_path(output).exists()
    assert not list(tmp_path.glob("*.stale-*"))


def test_a_dead_process_s_lock_on_this_host_is_taken_over(tmp_path, job_state):
    output = tmp_path / "out.h5"
    pid = dead_pid()
    write_holder(output, pid=pid)
    with output_lock(output) as held:
        assert held.takeover["reason"] == (
            f"pid {pid} no longer exists on {socket.gethostname()}"
        )


def test_an_abandoned_unparsable_lock_is_taken_over(tmp_path, job_state):
    output = tmp_path / "out.h5"
    path = lock_path(output)
    path.write_text("{trunc")
    old = time.time() - 2 * lock_module.UNPARSABLE_GRACE_S
    os.utime(path, (old, old))
    with output_lock(output) as held:
        assert held.takeover["previous"] == {"unparsable": "{trunc"}


def test_a_lock_that_changed_hands_is_put_back_not_deleted(tmp_path):
    """Between judging a lock stale and removing it, another process took it."""
    path = tmp_path / "out.h5.lock"
    path.write_text('{"token": "fresh"}')
    with pytest.raises(OutputBusy, match="changed hands"):
        lock_module._take_over(path, seen='{"token": "stale"}')
    assert path.read_text() == '{"token": "fresh"}'
    assert not list(tmp_path.glob("*.stale-*"))


# ── squeue ────────────────────────────────────────────────────────────────────
INVALID = "slurm_load_jobs error: Invalid job id specified\n"


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "expected"),
    [
        # --clusters=all: the job's own cluster lists it, the others complain
        # and squeue exits 1 regardless.
        (1, "25284145|RUNNING\n", INVALID * 2, "RUNNING"),
        (0, "25284145|PENDING\n", "", "PENDING"),
        # An array task is shown as <array>_<task>, not by its own job id.
        (1, "25272945_68|RUNNING\n", INVALID, "RUNNING"),
        (1, "1|COMPLETED\n", INVALID, "COMPLETED"),
        # Forgotten after MinJobAge: every cluster says so.
        (1, "", INVALID * 3, None),
    ],
)
def test_squeue_output_is_read_from_what_it_prints(
    monkeypatch, returncode, stdout, stderr, expected
):
    def fake_run(command, **kwargs):
        assert "--clusters=all" in command
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(lock_module.subprocess, "run", fake_run)
    assert slurm_job_state("25284145") == expected


def test_a_squeue_that_says_anything_else_is_unknown(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, "", "slurm_load_jobs error: Unable to contact slurm controller"
        )

    monkeypatch.setattr(lock_module.subprocess, "run", fake_run)
    with pytest.raises(JobStateUnknown, match="Unable to contact"):
        slurm_job_state("1")


def test_a_missing_squeue_is_unknown(monkeypatch):
    def fake_run(command, **kwargs):
        raise FileNotFoundError("squeue")

    monkeypatch.setattr(lock_module.subprocess, "run", fake_run)
    with pytest.raises(JobStateUnknown):
        slurm_job_state("1")


def test_a_lock_from_another_version_is_judged_on_its_holder(tmp_path, job_state):
    """Missing optional fields must not make a live holder's lock look abandoned."""
    output = tmp_path / "out.h5"
    path = lock_path(output)
    path.write_text(
        json.dumps({"host": socket.gethostname(), "pid": os.getppid(), "token": "t"})
    )
    old = time.time() - 2 * lock_module.UNPARSABLE_GRACE_S
    os.utime(path, (old, old))
    with pytest.raises(OutputBusy, match="is running"), output_lock(output):
        pytest.fail("took over a live holder's lock")


# ── requeued jobs ─────────────────────────────────────────────────────────────
def test_a_requeued_job_takes_over_the_lock_its_first_attempt_left(
    tmp_path, job_state, monkeypatch
):
    """allcpu preempts and requeues under the same job id.

    Asking Slurm about its own id would say RUNNING, and the first attempt ran
    on another node, so only the restart count can say the lock is stale.
    """
    output = tmp_path / "out.h5"
    write_holder(output, slurm_job_id="4242", slurm_restart_count=0, host="max-exfl001")
    job_state["4242"] = "RUNNING"  # it is: this is the requeued attempt
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    with output_lock(output) as held:
        assert held.takeover["reason"].endswith(
            "before it was requeued (now restart 1)"
        )
        assert json.loads(lock_path(output).read_text())["slurm_restart_count"] == 1


def test_a_lock_from_before_restart_counts_were_recorded_counts_as_restart_0(
    tmp_path, job_state, monkeypatch
):
    output = tmp_path / "out.h5"
    record = write_holder(output, slurm_job_id="4242", host="max-exfl001")
    del record["slurm_restart_count"]
    lock_path(output).write_text(json.dumps(record))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    with output_lock(output) as held:
        assert "at restart 0" in held.takeover["reason"]


def test_the_same_attempt_of_this_job_on_another_host_is_not_taken_over(
    tmp_path, job_state, monkeypatch
):
    output = tmp_path / "out.h5"
    write_holder(output, slurm_job_id="4242", slurm_restart_count=1, host="max-exfl001")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    with pytest.raises(OutputBusy, match="same Slurm job attempt"), output_lock(output):
        pytest.fail("took over a lock this job's live attempt may hold")
