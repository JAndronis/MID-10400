"""``analysis.common.slurm``: paths, batch scripts and the per-run ledger.

No cluster is involved. ``sbatch`` is a stand-in runner, and the proposal is a
symlink tree under ``tmp_path`` shaped like the real one: ``exp/…/p010400/usr``
pointing at ``u/usr/…`` and ``proc`` at ``d/proc/…``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from analysis.common.slurm import (
    DEFAULT_PARTITION,
    THREADS_PER_CORE,
    SubmitError,
    batch_script,
    prop_path,
    replace_option,
    run_each,
    submit,
)
from analysis.threadenv import THREAD_ENV


@pytest.fixture
def proposal(tmp_path: Path) -> Path:
    """``exp/MID/202601/p010400`` with ``usr`` and ``proc`` on other tiers."""
    root = tmp_path / "exp" / "MID" / "202601" / "p010400"
    root.mkdir(parents=True)
    for sub, tier in (("usr", "u"), ("proc", "d")):
        real = tmp_path / tier / sub / "MID" / "202601" / "p010400"
        real.mkdir(parents=True)
        (root / sub).symlink_to(real)
    return root


# -- prop_path ---------------------------------------------------------------


def test_a_resolved_path_is_written_back_in_the_proposal_form(proposal, tmp_path):
    script = tmp_path / "u" / "usr" / "MID" / "202601" / "p010400" / "x" / "s.py"
    assert prop_path(script, root=proposal) == str(proposal / "usr" / "x" / "s.py")


def test_a_private_alias_is_resolved_first(proposal, tmp_path):
    """The worktree is reached through ``~/p010400``; a job must not need it."""
    alias = tmp_path / "home" / "p010400"
    alias.parent.mkdir()
    alias.symlink_to(proposal)
    assert prop_path(alias / "proc" / "r0423", root=proposal) == str(
        proposal / "proc" / "r0423"
    )


def test_the_subdirectory_itself_maps_to_its_proposal_form(proposal, tmp_path):
    real = tmp_path / "d" / "proc" / "MID" / "202601" / "p010400"
    assert prop_path(real, root=proposal) == str(proposal / "proc")


def test_a_venv_python_keeps_its_name_and_so_its_venv(proposal, tmp_path):
    """Resolving ``bin/python`` would run the bare interpreter (pitfall 22)."""
    interpreter = tmp_path / "uv-python" / "python3.12"
    interpreter.parent.mkdir()
    interpreter.write_text("")
    venv_bin = proposal / "usr" / "Software" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(interpreter)
    assert prop_path(venv_bin / "python", root=proposal) == str(
        proposal / "usr" / "Software" / ".venv" / "bin" / "python"
    )


def test_a_relative_path_is_taken_from_the_working_directory(
    proposal, tmp_path, monkeypatch
):
    monkeypatch.chdir(proposal / "usr")
    assert prop_path("scripts/p4.py", root=proposal) == str(
        proposal / "usr" / "scripts" / "p4.py"
    )


def test_a_path_outside_the_proposal_is_only_resolved(proposal, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (tmp_path / "link").symlink_to(other)
    assert prop_path(tmp_path / "link" / "f.txt", root=proposal) == str(other / "f.txt")


def test_a_sibling_with_a_shared_prefix_is_not_inside(proposal, tmp_path):
    """``…/p010400-old`` must not match ``…/p010400`` as a string prefix would."""
    sibling = tmp_path / "u" / "usr" / "MID" / "202601" / "p010400-old"
    sibling.mkdir()
    assert prop_path(sibling / "f", root=proposal) == str(sibling / "f")


# -- batch_script --------------------------------------------------------------


def _options(script: str) -> list[str]:
    return [
        line.removeprefix("#SBATCH ")
        for line in script.splitlines()
        if line.startswith("#SBATCH ")
    ]


def test_the_batch_script_runs_the_command_once_on_one_node():
    script = batch_script(
        ["/p/.venv/bin/python", "/p/scripts/x.py", "--run", "423", "--local"],
        name="x_r0423",
        log_dir="/p/logs",
        cpus=36,
        limit="00:30:00",
    )
    options = _options(script)
    assert f"--partition={DEFAULT_PARTITION}" in options
    assert {"--nodes=1", "--ntasks=1", "--time=00:30:00"} <= set(options)
    assert "--job-name=x_r0423" in options
    assert "--output=/p/logs/slurm-%j.out" in options
    assert script.startswith("#!/bin/bash\n")
    assert script.rstrip().splitlines()[-1] == (
        "/p/.venv/bin/python /p/scripts/x.py --run 423 --local"
    )
    assert "srun" not in script


def test_cpus_are_physical_cores_and_the_job_asks_for_their_threads():
    """``-c 36`` alone may land on a 2 x 10-core node of ``allcpu``."""
    options = _options(batch_script(["x"], name="j", log_dir="/l", cpus=36, limit="1"))
    assert f"--cpus-per-task={36 * THREADS_PER_CORE}" in options
    options = _options(
        batch_script(
            ["x"], name="j", log_dir="/l", cpus=8, limit="1", threads_per_core=1
        )
    )
    assert "--cpus-per-task=8" in options


def test_the_batch_script_pins_every_library_to_one_thread():
    script = batch_script(["x"], name="j", log_dir="/l", cpus=1, limit="10")
    export = next(line for line in script.splitlines() if line.startswith("export "))
    assert set(export.split()[1:]) == {f"{name}=1" for name in THREAD_ENV}
    # before the command, so the command's imports see it
    lines = script.splitlines()
    assert lines.index(export) < len(lines) - 1


def test_a_dependency_waits_for_success_and_dies_with_a_failure():
    options = _options(
        batch_script(["x"], name="j", log_dir="/l", cpus=1, limit="10", after="123")
    )
    assert "--dependency=afterok:123" in options
    assert "--kill-on-invalid-dep=yes" in options
    options = _options(
        batch_script(["x"], name="j", log_dir="/l", cpus=1, limit="10", after=[1, "2"])
    )
    assert "--dependency=afterok:1:2" in options


def test_no_dependency_lines_without_one():
    script = batch_script(["x"], name="j", log_dir="/l", cpus=1, limit="10")
    assert "dependency" not in script and "kill-on-invalid-dep" not in script


def test_constraint_reservation_and_workdir_are_passed_through():
    options = _options(
        batch_script(
            ["x"],
            name="j",
            log_dir="/l",
            cpus=1,
            limit="10",
            constraint="Gold-6240|Gold-6140",
            reservation="upex_010400",
            workdir="/p/usr",
        )
    )
    assert "--constraint=Gold-6240|Gold-6140" in options
    assert "--reservation=upex_010400" in options
    assert "--chdir=/p/usr" in options


def test_arguments_with_spaces_are_quoted():
    script = batch_script(
        ["python", "-c", "print('a b')"], name="j", log_dir="/l", cpus=1, limit="10"
    )
    assert script.rstrip().splitlines()[-1] == "python -c 'print('\"'\"'a b'\"'\"')'"


@pytest.mark.parametrize("limit", ["10", "10:00", "01:00:00", "2-00", "1-02:03:04"])
def test_every_time_form_sbatch_accepts_is_accepted(limit):
    batch_script(["x"], name="j", log_dir="/l", cpus=1, limit=limit)


@pytest.mark.parametrize(
    "bad",
    [
        {"limit": "1h"},
        {"cpus": 0},
        {"cpus": 1.5},
        {"name": "two words"},
        {"partition": ""},
        {"constraint": "a b"},
        {"after": "123;cluster"},
        {"threads_per_core": 0},
    ],
)
def test_options_sbatch_would_misread_are_refused(bad):
    kwargs = {"name": "j", "log_dir": "/l", "cpus": 1, "limit": "10"} | bad
    with pytest.raises(ValueError):
        batch_script(["x"], **kwargs)


# -- submit --------------------------------------------------------------------


class FakeSbatch:
    """Records what ``sbatch`` was given and answers like it."""

    def __init__(self, stdout: str = "4242\n", returncode: int = 0, stderr: str = ""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr
        self.calls: list[list[str]] = []
        self.scripts: list[str] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        self.scripts.append(Path(argv[-1]).read_text())
        return subprocess.CompletedProcess(
            argv, self.returncode, self.stdout, self.stderr
        )


def test_submit_hands_sbatch_the_script_and_returns_the_job_id(tmp_path, caplog):
    fake = FakeSbatch()
    caplog.set_level(logging.INFO, logger="analysis")
    job = submit(["echo", "hi"], "j", tmp_path / "logs", cpus=4, runner=fake)
    assert job == "4242"
    assert fake.calls[0][:2] == ["sbatch", "--parsable"]
    assert fake.scripts[0] == batch_script(
        ["echo", "hi"], name="j", log_dir=tmp_path / "logs", cpus=4, limit="01:00:00"
    )
    assert "job 4242" in caplog.text


def test_submit_creates_the_log_directory_slurm_would_not(tmp_path):
    submit(["x"], "j", tmp_path / "a" / "b", runner=FakeSbatch())
    assert (tmp_path / "a" / "b").is_dir()


def test_the_temporary_script_is_gone_once_sbatch_has_it(tmp_path):
    fake = FakeSbatch()
    submit(["x"], "j", tmp_path, runner=fake)
    assert not Path(fake.calls[0][-1]).exists()


def test_a_cluster_suffix_is_stripped_from_the_job_id(tmp_path):
    assert submit(["x"], "j", tmp_path, runner=FakeSbatch("77;maxwell\n")) == "77"


def test_a_refusal_raises_with_sbatch_s_reason(tmp_path):
    fake = FakeSbatch("", returncode=1, stderr="invalid partition")
    with pytest.raises(SubmitError, match="invalid partition"):
        submit(["x"], "j", tmp_path, runner=fake)


def test_an_answer_that_is_not_a_job_id_raises(tmp_path):
    with pytest.raises(SubmitError, match="not a job id"):
        submit(["x"], "j", tmp_path, runner=FakeSbatch("Submitted batch job 5\n"))


def test_a_dry_run_prints_and_submits_nothing(tmp_path, capsys):
    def never(*args, **kwargs):
        raise AssertionError("sbatch called on a dry run")

    assert submit(["x"], "j", tmp_path / "logs", dry_run=True, runner=never) is None
    assert (
        capsys.readouterr().out.strip()
        == batch_script(
            ["x"], name="j", log_dir=tmp_path / "logs", cpus=1, limit="01:00:00"
        ).strip()
    )
    assert not (tmp_path / "logs").exists()


# -- run_each ------------------------------------------------------------------


def test_every_run_is_tried_and_the_failures_are_returned(caplog):
    seen = []

    def work(run):
        seen.append(run)
        if run == 2:
            raise RuntimeError("boom")
        return {3: 1, 4: None}.get(run, 0)

    with caplog.at_level(logging.ERROR, logger="analysis"):
        failed = run_each([1, 2, 3, 4, 5], work)
    assert seen == [1, 2, 3, 4, 5]
    assert failed == [2, 3]
    assert "boom" in caplog.text  # with the traceback
    assert "2 of 5 runs failed: 2 3" in caplog.text


def test_nothing_failed_is_an_empty_list(caplog):
    with caplog.at_level(logging.ERROR, logger="analysis"):
        assert run_each([1, 2], lambda run: 0) == []
    assert caplog.text == ""


def test_an_interrupt_still_stops_the_loop():
    def work(run):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_each([1, 2], work)


# -- replace_option ------------------------------------------------------------


def test_the_run_list_becomes_one_run():
    argv = ["--run", "423", "426", "-w", "36"]
    assert replace_option(argv, ["--run"], ["426"]) == ["-w", "36", "--run", "426"]


def test_the_equals_form_is_removed_too():
    assert replace_option(["--run=423", "-w", "8"], ["--run"], ["1"]) == [
        "-w",
        "8",
        "--run",
        "1",
    ]


def test_a_flag_is_dropped_without_eating_the_next_option():
    argv = ["--dry-run", "--run", "1", "--local"]
    assert replace_option(argv, ["--dry-run"], nargs=0) == ["--run", "1", "--local"]


def test_an_option_with_one_value_takes_exactly_one():
    argv = ["--after", "123", "456"]
    assert replace_option(argv, ["--after"], nargs=1) == ["456"]


def test_every_spelling_and_occurrence_goes():
    argv = ["-w", "8", "--workers", "16", "x"]
    assert replace_option(argv, ["-w", "--workers"], ["36"], nargs=1) == [
        "x",
        "-w",
        "36",
    ]


def test_an_option_that_only_shares_a_prefix_is_kept():
    argv = ["--run-dir", "/r", "--run", "1"]
    assert replace_option(argv, ["--run"], ["2"]) == ["--run-dir", "/r", "--run", "2"]


def test_an_absent_option_is_appended():
    assert replace_option(["-w", "8"], ["--run"], ["1"]) == ["-w", "8", "--run", "1"]
