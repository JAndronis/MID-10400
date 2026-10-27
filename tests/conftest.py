"""Fixtures shared by every suite.

A content digest of a pass output file, a session-cached mock-run factory and
the plan-status helper both plan suites use.
The heavy imports (h5py, EXtra-data) happen inside the fixtures that need them.
"""

from __future__ import annotations

import getpass
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

#: Default basetemp root. ``/tmp`` is a small volume shared with other users, and
#: the mock runs are hundreds of megabytes, so a session without ``--basetemp``
#: puts its scratch in RAM-backed ``/dev/shm`` when that is writable.
_SHM = Path("/dev/shm")


def pytest_configure(config: pytest.Config) -> None:
    """Choose a per-session basetemp under ``/dev/shm`` when the user gave none."""
    config._default_basetemp = None  # type: ignore[attr-defined]
    if config.option.basetemp is not None:
        return
    if not (_SHM.is_dir() and os.access(_SHM, os.W_OK)):
        return
    chosen = _SHM / f"pytest-of-{getpass.getuser()}" / f"session-{os.getpid()}"
    chosen.parent.mkdir(exist_ok=True)
    config.option.basetemp = str(chosen)
    config._default_basetemp = chosen  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Free the RAM-backed basetemp of a passing session; keep a failing one."""
    chosen = session.config._default_basetemp  # type: ignore[attr-defined]
    if chosen is not None and exitstatus == 0:
        shutil.rmtree(chosen, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _session_tempdir(tmp_path_factory: pytest.TempPathFactory):
    """Point ``tempfile`` and ``TMPDIR`` at a directory inside the basetemp.

    ``TMPDIR`` is an environment variable so that spawned pool workers, which
    are fresh interpreters, inherit it. The directory is made after pytest has
    cleared an explicit basetemp, which is why this is a fixture and not part
    of ``pytest_configure``.
    """
    scratch = tmp_path_factory.getbasetemp() / "tmp"
    scratch.mkdir(exist_ok=True)
    saved_env = os.environ.get("TMPDIR")
    saved_dir = tempfile.tempdir
    os.environ["TMPDIR"] = str(scratch)
    tempfile.tempdir = str(scratch)
    yield scratch
    tempfile.tempdir = saved_dir
    if saved_env is None:
        os.environ.pop("TMPDIR", None)
    else:
        os.environ["TMPDIR"] = saved_env


#: Attributes that differ between two runs of the same pass on the same input,
#: and so cannot be pinned: wall-clock, the machine, per-stage timings, and the
#: digests of the synthetic input files the fixtures build. ``poni`` is in the
#: list because pyFAI stamps the PONI it writes with the time it wrote it, so
#: the fixture's bytes, and every sha256 taken over them, move every session.
#: ``lock`` and ``lock_takeover`` name the process that held the output's lock.
#: Everything else an output file carries is a function of its inputs.
VOLATILE_ATTRS = frozenset(
    {
        "config_hash",
        "host",
        "input_file_sha256",
        "lock",
        "lock_takeover",
        "package_versions",
        "platform",
        "poni",
        "poni_sha256",
        "run_checks",
        "setup_timings",
        "started_at",
        "timings",
        "wall_s",
    }
)


#: Matches an absolute POSIX path inside an attribute's rendering. The fixtures
#: write their geometry, masks and output under ``tmp_path``, so a recorded path
#: carries a directory that changes every session; the basename does not.
_ABSOLUTE_PATH = re.compile(r"/(?:[^/\s\"',]+/)+([^/\s\"',]*)")


def _h5_digest(path: Path, skip_attrs: frozenset[str] = VOLATILE_ATTRS) -> str:
    """sha256 over every dataset and every non-volatile attribute of an HDF5 file.

    Datasets and attributes are visited in sorted-name order, so the digest is a
    function of the file's content alone and not of the order it was written in.
    Absolute paths are reduced to their basename first, so the recorded config
    (the record of every result-affecting field) stays in the digest without the
    fixtures' temporary directories entering it.

    :param skip_attrs: attribute names to leave out, for values that differ
        between two runs of the same pass on the same input.
    """
    import hashlib

    import h5py
    import numpy as np

    digest = hashlib.sha256()

    def visit(name: str, obj: object) -> None:
        digest.update(f"\n@{name}".encode())
        for key in sorted(obj.attrs):  # type: ignore[attr-defined]
            if key in skip_attrs:
                continue
            value = obj.attrs[key]  # type: ignore[attr-defined]
            if isinstance(value, np.ndarray):
                rendered = np.array2string(value, threshold=1 << 30)
            else:
                rendered = repr(value)
            digest.update(f"|{key}={_ABSOLUTE_PATH.sub(r'\1', rendered)}".encode())
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            digest.update(f"|dtype={obj.dtype!s}|shape={obj.shape}".encode())
            digest.update(np.ascontiguousarray(data).tobytes())

    with h5py.File(path, "r") as handle:
        visit("/", handle)
        handle.visititems(visit)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def h5_digest() -> Callable[..., str]:
    """A canonical content digest of a pass output file (:func:`_h5_digest`)."""
    return _h5_digest


@pytest.fixture(scope="session")
def memoised_run_factory(tmp_path_factory):
    """Build a session-cached mock-run factory for one pass.

    Writing a mock run is the slowest thing in either suite and each one lives
    until the session ends, so two requests for the same bytes must share a
    directory. The key is the *resolved* signature, not the raw keywords:
    passing a default explicitly is asking for the default run, and keying on
    keywords alone silently writes a second identical copy of it.

    :param write_mock_run: the pass's writer, called as ``write(root, **kwargs)``.
    :param prefix: ``mktemp`` prefix, so the two suites' runs stay tellable apart.
    :param open_run: how to open what was written; injected by the tests that
        cover the caching itself, which have no HDF5 to open.
    :returns: a factory returning ``(mock, DataCollection)``.
    """
    import inspect

    def make(
        write_mock_run: Callable[..., object],
        prefix: str,
        open_run: Callable[[str], object] | None = None,
    ):
        signature = inspect.signature(write_mock_run)
        cache: dict[tuple, tuple] = {}

        def factory(**kwargs):
            opener = open_run
            if opener is None:
                from extra_data import RunDirectory

                opener = RunDirectory

            bound = signature.bind_partial(**kwargs)
            bound.apply_defaults()
            key = tuple(
                sorted(
                    (name, tuple(value) if isinstance(value, list) else value)
                    for name, value in bound.arguments.items()
                )
            )
            if key not in cache:
                root = tmp_path_factory.mktemp(prefix)
                run = write_mock_run(root, **kwargs)
                cache[key] = (run, opener(str(root)))
            return cache[key]

        return factory

    return make


@pytest.fixture
def status_of():
    """``plan.record(train_id).status``, the assertion both plan suites make."""

    def of(plan, train_id: int):
        return plan.record(train_id).status

    return of
