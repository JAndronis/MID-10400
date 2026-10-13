"""Pass integrity on a mock run (open task 17; CLAUDE.md pitfalls 26 and 27).

A complete run is answered from its files and nothing is written; a second
writer is refused while the first holds the lock, and takes a dead one's lock
over; ``WORKER_ERROR`` blocks are retried; a file integrated by another
operator or pyFAI is refused rather than resumed into.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("extra_data")

import h5py  # noqa: E402
import xarray as xr  # noqa: E402
from test_run import InlinePool, run_on_mock  # noqa: E402  (tests/saxs on sys.path)

from analysis.common import lock as lock_module  # noqa: E402
from analysis.common.lock import OutputBusy, lock_path  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.common.writer import IncompleteRun, OperatorMismatch  # noqa: E402
from analysis.saxs import masks as masks_module  # noqa: E402
from analysis.saxs import operator as operator_module  # noqa: E402
from analysis.saxs import run as run_module  # noqa: E402
from analysis.saxs import worker  # noqa: E402
from analysis.saxs.complete import is_complete, operator_differences  # noqa: E402
from analysis.saxs.pixel_sums import FILE_NAME, PixelSumWriter  # noqa: E402
from analysis.saxs.writer import AgipdSaxsWriter, per_pulse  # noqa: E402


@pytest.fixture
def pipeline(mock_pipeline, geom):
    mock_pipeline.geometry = geom
    return mock_pipeline


@pytest.fixture(scope="module")
def complete_run(mock_pipeline, geom, tmp_path_factory):
    """One finished run, copied by the tests that alter it."""
    mock_pipeline.geometry = geom
    directory = tmp_path_factory.mktemp("complete")
    pooled = run_on_mock(mock_pipeline, directory)
    return directory, pooled


def copy_of(complete_run, tmp_path):
    source, pooled = complete_run
    for name in ("out.h5", FILE_NAME):
        shutil.copy2(source / name, tmp_path / name)
    return pooled


def stats(directory):
    """What any write — even an attribute — would change, per file."""
    return {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(directory.iterdir())
    }


def forbid_writing(monkeypatch):
    """Make every step that could write raise, so a write cannot pass silently."""

    def forbidden(*args, **kwargs):
        raise AssertionError("a complete run must write nothing")

    monkeypatch.setattr(run_module, "build_plan", forbidden)
    monkeypatch.setattr(run_module, "output_lock", forbidden)
    monkeypatch.setattr(operator_module, "save_operator", forbidden)
    monkeypatch.setattr(masks_module, "save_masks", forbidden)
    monkeypatch.setattr(AgipdSaxsWriter, "open_or_create", forbidden)
    monkeypatch.setattr(PixelSumWriter, "open_or_create", forbidden)
    monkeypatch.setattr(worker, "process_block", forbidden)


@pytest.fixture
def job_state(monkeypatch):
    """Slurm, replaced: ``states[job_id]`` is what squeue would report."""
    states: dict[str, str | None] = {}
    monkeypatch.setattr(lock_module, "slurm_job_state", lambda job: states.get(job))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return states


def hold(output, job_id):
    """A lock on ``output`` left by Slurm job ``job_id`` on another node."""
    record = {
        "output": str(output),
        "host": "max-exfl001.desy.de",
        "pid": 12345,
        "user": "xdamnprd",
        "slurm_job_id": job_id,
        "slurm_cluster": "maxwell",
        "started_at": 0.0,
        "started": "2026-10-13T12:00:00+00:00",
        "token": "theirs",
    }
    lock_path(output).write_text(json.dumps(record))


# ── a complete run writes nothing ─────────────────────────────────────────────
def test_a_complete_run_is_complete(pipeline, complete_run):
    directory, _ = complete_run
    verdict = is_complete(
        directory / "out.h5",
        pipeline.cfg,
        pixel_sums_path=directory / FILE_NAME,
        geometry=pipeline.geometry,
    )
    assert verdict, verdict.reasons


def test_a_complete_run_is_answered_from_the_file_and_writes_nothing(
    pipeline, complete_run, tmp_path, monkeypatch
):
    """The inode and mtime of both files stay put, and no npz reappears."""
    pooled = copy_of(complete_run, tmp_path)
    before = stats(tmp_path)
    forbid_writing(monkeypatch)

    again = run_on_mock(pipeline, tmp_path)

    xr.testing.assert_identical(again, pooled)
    assert stats(tmp_path) == before
    assert not list(tmp_path.glob("*.npz"))
    assert not list(tmp_path.glob("*.lock"))


def test_the_per_pulse_grid_comes_from_the_file_too(
    pipeline, complete_run, tmp_path, monkeypatch
):
    copy_of(complete_run, tmp_path)
    before = stats(tmp_path)
    forbid_writing(monkeypatch)
    grid = run_module.run_agipd_saxs(
        pipeline.cfg,
        dc=pipeline.dc,
        geometry=pipeline.geometry,
        output_path=tmp_path / "out.h5",
        reduce="per_pulse",
    )
    xr.testing.assert_identical(grid, per_pulse(tmp_path / "out.h5"))
    assert stats(tmp_path) == before


def test_a_complete_file_with_final_failures_raises_as_the_pass_would(
    pipeline, complete_run, tmp_path, monkeypatch
):
    """``LABEL_MISMATCH`` is final, so the file is complete — and still refused."""
    copy_of(complete_run, tmp_path)
    with h5py.File(tmp_path / "out.h5", "r+") as handle:
        handle["frames/status"][:3] = FrameStatus.LABEL_MISMATCH
    before = stats(tmp_path)
    forbid_writing(monkeypatch)

    with pytest.raises(IncompleteRun, match="LABEL_MISMATCH"):
        run_on_mock(pipeline, tmp_path)
    run_on_mock(pipeline, tmp_path, allow_incomplete=True)
    assert stats(tmp_path) == before


# ── why a file is not complete ────────────────────────────────────────────────
def _set_status(code):
    def change(directory):
        with h5py.File(directory / "out.h5", "r+") as handle:
            handle["frames/status"][5:7] = code

    return change


def _set_pyfai(directory):
    with h5py.File(directory / "out.h5", "r+") as handle:
        attrs = handle["provenance"].attrs
        versions = json.loads(attrs["package_versions"])
        attrs["package_versions"] = json.dumps(versions | {"pyFAI": "2000.1.0"})


def _unfinalise(directory):
    with h5py.File(directory / "out.h5", "r+") as handle:
        del handle["provenance"].attrs["status_summary"]


def _unwrite_window(directory):
    with h5py.File(directory / FILE_NAME, "r+") as handle:
        handle["windows/written"][0] = 0


def _lose_sums(directory):
    (directory / FILE_NAME).unlink()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (_set_status(FrameStatus.NOT_PROCESSED), "2 frames NOT_PROCESSED"),
        (_set_status(FrameStatus.WORKER_ERROR), "2 frames WORKER_ERROR"),
        (_set_pyfai, "written with pyFAI 2000.1.0 -> "),
        (_unfinalise, "never finalised"),
        (_unwrite_window, "window sums: 1 of 1 windows unwritten"),
        (_lose_sums, "window sums: no file"),
    ],
)
def test_is_complete_says_why_not(pipeline, complete_run, tmp_path, change, expected):
    copy_of(complete_run, tmp_path)
    change(tmp_path)
    verdict = is_complete(
        tmp_path / "out.h5",
        pipeline.cfg,
        pixel_sums_path=tmp_path / FILE_NAME,
        geometry=pipeline.geometry,
    )
    assert not verdict
    assert any(expected in reason for reason in verdict.reasons), verdict.reasons


def test_another_operator_or_config_is_not_complete(pipeline, complete_run):
    directory, _ = complete_run
    kwargs = {"pixel_sums_path": directory / FILE_NAME}
    moved = replace(pipeline.op, q=pipeline.op.q * 1.001)
    other_operator = is_complete(
        directory / "out.h5", pipeline.cfg, operator=moved, **kwargs
    )
    assert other_operator.reasons == ("operator q differs from the one built now",)
    other_npt = is_complete(
        directory / "out.h5",
        replace(pipeline.cfg, npt=pipeline.cfg.npt + 1),
        geometry=pipeline.geometry,
        **kwargs,
    )
    assert any("config hash" in reason for reason in other_npt.reasons)


# ── node classes ──────────────────────────────────────────────────────────────
def _nudge_omega(directory, relative=None):
    """Move every tenth stored solid angle: one ULP, or by ``relative``."""
    with h5py.File(directory / "out.h5", "r+") as handle:
        omega = handle["operator/omega"][()]
        if relative is None:
            omega[::10] = np.nextafter(omega[::10], np.inf)
        else:
            omega[::10] *= 1 + relative
        handle["operator/omega"][...] = omega


def test_a_one_ulp_solid_angle_from_another_node_class_is_complete(
    pipeline, complete_run, tmp_path, monkeypatch
):
    """AVX2 and AVX-512 nodes build omega one ULP apart on ~10 % of pixels
    (S3, 2026-10-13), and allcpu mixes both: that must not refuse a file."""
    copy_of(complete_run, tmp_path)
    _nudge_omega(tmp_path)
    before = stats(tmp_path)
    kwargs = {"pixel_sums_path": tmp_path / FILE_NAME, "geometry": pipeline.geometry}
    assert is_complete(tmp_path / "out.h5", pipeline.cfg, **kwargs)
    AgipdSaxsWriter.check_resumable(
        pipeline.cfg,
        tmp_path / "out.h5",
        operator_check=lambda handle: operator_differences(handle, pipeline.op),
    )
    forbid_writing(monkeypatch)
    run_on_mock(pipeline, tmp_path)
    assert stats(tmp_path) == before


def test_a_real_change_to_the_solid_angle_is_not(pipeline, complete_run, tmp_path):
    copy_of(complete_run, tmp_path)
    _nudge_omega(tmp_path, relative=1e-9)
    verdict = is_complete(
        tmp_path / "out.h5",
        pipeline.cfg,
        pixel_sums_path=tmp_path / FILE_NAME,
        geometry=pipeline.geometry,
    )
    assert verdict.reasons == ("operator omega differs by more than 1e-12 relative",)


def test_a_missing_file_is_not_complete(pipeline, tmp_path):
    verdict = is_complete(tmp_path / "out.h5", pipeline.cfg, operator=pipeline.op)
    assert verdict.reasons[0] == "no file"


# ── the lock ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("locked", ["out.h5", FILE_NAME])
def test_a_live_writer_s_lock_refuses_the_run_before_anything_is_written(
    pipeline, tmp_path, job_state, locked
):
    hold(tmp_path / locked, "4242")
    job_state["4242"] = "RUNNING"
    with pytest.raises(OutputBusy, match="4242 is RUNNING"):
        run_on_mock(pipeline, tmp_path)
    # Only the lock that refused is left: none of ours, and no output at all.
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{locked}.lock"]


def test_a_live_lock_refuses_even_a_complete_run(
    pipeline, complete_run, tmp_path, job_state
):
    """The file may be mid-write, so its completeness is not trusted."""
    copy_of(complete_run, tmp_path)
    hold(tmp_path / "out.h5", "4242")
    job_state["4242"] = "RUNNING"
    with pytest.raises(OutputBusy):
        run_on_mock(pipeline, tmp_path)


def test_a_dead_writer_s_lock_is_taken_over_and_recorded(pipeline, tmp_path, job_state):
    hold(tmp_path / "out.h5", "4242")
    job_state["4242"] = "TIMEOUT"
    run_on_mock(pipeline, tmp_path)

    with h5py.File(tmp_path / "out.h5", "r") as handle:
        takeover = json.loads(handle["provenance"].attrs["lock_takeover"])
        lock = json.loads(handle["provenance"].attrs["lock"])
    assert takeover["previous"]["slurm_job_id"] == "4242"
    assert takeover["reason"] == "Slurm job 4242 is TIMEOUT"
    assert lock["host"] and "token" not in lock
    with h5py.File(tmp_path / FILE_NAME, "r") as handle:
        assert "lock_takeover" not in handle["provenance"].attrs
        assert "lock" in handle["provenance"].attrs
    assert not list(tmp_path.glob("*.lock"))


def test_a_stale_lock_on_a_complete_run_is_cleared_without_writing(
    pipeline, complete_run, tmp_path, job_state, monkeypatch
):
    copy_of(complete_run, tmp_path)
    hold(tmp_path / "out.h5", "4242")
    job_state["4242"] = None  # long gone
    data_files = {
        name: value
        for name, value in stats(tmp_path).items()
        if not name.endswith(".lock")
    }
    monkeypatch.setattr(run_module, "build_plan", pytest.fail)

    run_on_mock(pipeline, tmp_path)
    assert stats(tmp_path) == data_files


# ── retry and resume ──────────────────────────────────────────────────────────
def test_a_rerun_retries_worker_error_blocks(pipeline, tmp_path, monkeypatch):
    real = worker.process_block

    def explode(block):
        if block.index == 0:
            raise RuntimeError("transient")
        return real(block)

    monkeypatch.setattr(worker, "process_block", explode)
    with pytest.raises(IncompleteRun):
        run_on_mock(pipeline, tmp_path, pixel_sum_trains=None)

    seen = []
    monkeypatch.setattr(
        worker, "process_block", lambda b: seen.append(b.index) or real(b)
    )
    run_on_mock(pipeline, tmp_path, pixel_sum_trains=None)

    assert seen == [0]
    with h5py.File(tmp_path / "out.h5", "r") as handle:
        assert (handle["frames/status"][:] == FrameStatus.OK).all()
        provenance = handle["provenance"].attrs
        assert "block_errors" not in provenance
        assert "transient" in json.loads(provenance["block_errors_resolved"])["0"]


def test_a_file_from_another_pyfai_is_refused_not_resumed(
    pipeline, tmp_path, monkeypatch
):
    """Resuming would integrate the rest with another pyFAI, then stamp it on all."""
    real = worker.process_block
    monkeypatch.setattr(
        worker,
        "process_block",
        lambda b: (_ for _ in ()).throw(RuntimeError("x")) if b.index == 0 else real(b),
    )
    with pytest.raises(IncompleteRun):
        run_on_mock(pipeline, tmp_path)
    monkeypatch.setattr(worker, "process_block", real)
    _set_pyfai(tmp_path)
    before = stats(tmp_path)

    with pytest.raises(OperatorMismatch, match="pyFAI 2000.1.0 -> "):
        run_on_mock(pipeline, tmp_path)
    assert stats(tmp_path) == before


def test_a_file_from_another_operator_is_refused_not_resumed(
    pipeline, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        run_module,
        "fan_out",
        lambda out, todo, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        run_on_mock(pipeline, tmp_path, pixel_sum_trains=None)
    monkeypatch.undo()
    with h5py.File(tmp_path / "out.h5", "r+") as handle:
        handle["operator/coef"][0] *= 2
    before = stats(tmp_path)

    with pytest.raises(OperatorMismatch, match="operator coef differs"):
        run_on_mock(pipeline, tmp_path, pixel_sum_trains=None)
    assert stats(tmp_path) == before


def test_the_npz_writes_leave_no_temporary_files(pipeline, tmp_path):
    run_on_mock(pipeline, tmp_path, pool_factory=InlinePool)
    assert not [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    np.testing.assert_array_equal(
        operator_module.load_operator(tmp_path / "operator.npz").q, pipeline.op.q
    )
