"""Pass integrity for one JUNGFRAU, as for AGIPD.

A complete file is answered from and nothing is written; a live lock refuses;
``WORKER_ERROR`` blocks are retried; a file from another pyFAI is refused.
"""

from __future__ import annotations

import dataclasses
import json

import h5py
import numpy as np
import pytest

pytest.importorskip("extra_data")

import xarray as xr  # noqa: E402
from test_waxs_run import FailingPool, run  # noqa: E402  (tests/waxs on sys.path)

from analysis.common import lock as lock_module  # noqa: E402
from analysis.common.lock import OutputBusy, lock_path  # noqa: E402
from analysis.common.status import FrameStatus  # noqa: E402
from analysis.common.writer import IncompleteRun, OperatorMismatch  # noqa: E402
from analysis.waxs import run as run_module  # noqa: E402
from analysis.waxs import worker  # noqa: E402
from analysis.waxs.cells import UnexpectedLitCells  # noqa: E402
from analysis.waxs.complete import expected_operator, is_complete  # noqa: E402
from analysis.waxs.writer import JungfrauWaxsWriter  # noqa: E402


def stats(directory):
    return {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(directory.iterdir())
    }


def forbid_writing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a complete run must write nothing")

    monkeypatch.setattr(run_module, "output_lock", forbidden)
    monkeypatch.setattr(run_module, "open_detector", forbidden)
    monkeypatch.setattr(JungfrauWaxsWriter, "open_or_create", forbidden)
    monkeypatch.setattr(worker, "process_block", forbidden)


def set_pyfai(path, version):
    with h5py.File(path, "r+") as handle:
        attrs = handle["provenance"].attrs
        versions = json.loads(attrs["package_versions"])
        attrs["package_versions"] = json.dumps(versions | {"pyFAI": version})


def test_a_complete_file_is_answered_from_and_nothing_is_written(
    cfg, mock_run_factory, tmp_path, monkeypatch
):
    mock, dc = mock_run_factory()
    path = tmp_path / "run.h5"
    first = run(cfg, mock, dc, output_path=path)
    assert is_complete(path, cfg)
    before = stats(tmp_path)
    forbid_writing(monkeypatch)

    again = run(cfg, mock, dc, output_path=path)

    xr.testing.assert_identical(again, first)
    assert stats(tmp_path) == before


def test_a_pin_the_stored_cells_contradict_still_refuses(
    cfg, mock_run_factory, tmp_path, monkeypatch
):
    mock, dc = mock_run_factory(lit_cells=(0, 1, 2, 3))
    path = tmp_path / "run.h5"
    run(cfg, mock, dc, output_path=path)
    before = stats(tmp_path)
    forbid_writing(monkeypatch)

    pinned = dataclasses.replace(cfg, expected_lit_cells=(0, 1, 2, 3, 4, 5, 6, 15))
    with pytest.raises(UnexpectedLitCells, match=r"holds cells \[0, 1, 2, 3\]"):
        run(pinned, mock, dc, output_path=path)
    assert stats(tmp_path) == before


def test_another_pyfai_or_operator_is_not_complete(cfg, mock_run_factory, tmp_path):
    mock, dc = mock_run_factory()
    path = tmp_path / "run.h5"
    run(cfg, mock, dc, output_path=path)
    op = expected_operator(cfg)
    moved = is_complete(
        path, cfg, operator=dataclasses.replace(op, q=op.q * 1.01, sha256="b" * 64)
    )
    assert moved.reasons == ("operator q differs by more than 1e-06 relative",)
    set_pyfai(path, "2000.1.0")
    verdict = is_complete(path, cfg)
    assert verdict.reasons == (f"written with pyFAI 2000.1.0 -> {_installed()}",)


def test_a_float32_ulp_in_the_stored_solid_angle_is_complete(
    cfg, mock_run_factory, tmp_path
):
    """The file stores omega as float32; a one-ULP float64 difference between
    node classes can round to a one-ULP float32 one."""
    mock, dc = mock_run_factory()
    path = tmp_path / "run.h5"
    run(cfg, mock, dc, output_path=path)
    with h5py.File(path, "r+") as handle:
        omega = handle["operator/omega"][()]
        omega[::10] = np.nextafter(omega[::10], np.float32(np.inf))
        handle["operator/omega"][...] = omega
        # The other node class's file carries its own sha256, everywhere.
        handle["operator"].attrs["sha256"] = "a" * 64
        handle["provenance"].attrs["operator_sha256"] = "a" * 64
    assert is_complete(path, cfg)


def _installed():
    from importlib.metadata import version

    return version("pyFAI")


def test_a_live_lock_refuses_and_nothing_is_written(
    cfg, mock_run_factory, tmp_path, monkeypatch
):
    mock, dc = mock_run_factory()
    path = tmp_path / "run.h5"
    lock_path(path).write_text(
        json.dumps(
            {
                "output": str(path),
                "host": "max-exfl001.desy.de",
                "pid": 1,
                "user": "xdamnprd",
                "slurm_job_id": "4242",
                "slurm_cluster": "maxwell",
                "started_at": 0.0,
                "started": "2026-10-13T12:00:00+00:00",
                "token": "theirs",
            }
        )
    )
    monkeypatch.setattr(lock_module, "slurm_job_state", lambda job: "RUNNING")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    with pytest.raises(OutputBusy, match="4242 is RUNNING"):
        run(cfg, mock, dc, output_path=path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.h5.lock"]


def test_a_rerun_retries_worker_error_blocks(cfg, mock_run_factory, tmp_path):
    mock, dc = mock_run_factory()
    path = tmp_path / "run.h5"
    with pytest.raises(IncompleteRun, match="WORKER_ERROR"):
        run(cfg, mock, dc, output_path=path, pool_factory=FailingPool)

    grid = run(cfg, mock, dc, output_path=path)

    assert (grid["n_frames"] == 1).all()
    with h5py.File(path, "r") as handle:
        assert (handle["frames/status"][:] == FrameStatus.OK).all()
        assert "block_errors" not in handle["provenance"].attrs
    assert is_complete(path, cfg)


def test_a_file_from_another_pyfai_is_refused_not_resumed(
    cfg, mock_run_factory, tmp_path
):
    mock, dc = mock_run_factory()
    path = tmp_path / "run.h5"
    with pytest.raises(IncompleteRun):
        run(cfg, mock, dc, output_path=path, pool_factory=FailingPool)
    set_pyfai(path, "2000.1.0")
    before = stats(tmp_path)

    with pytest.raises(OperatorMismatch, match="pyFAI 2000.1.0 -> "):
        run(cfg, mock, dc, output_path=path)
    assert stats(tmp_path) == before
