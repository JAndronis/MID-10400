"""Shared fixtures: synthetic EuXFEL ``raw/`` trees and canonical elogs.

Everything here is filesystem-only — no EXtra-data, no HDF5 content — so the
unit tests exercise the reader's pure paths (helpers, ``can_read``,
``list_runs``, ``get_run_path``) without the EuXFEL stack installed.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest


def _make_raw_tree(
    root: Path, run_ids: Iterable[int], *, sentinel: bool = True
) -> Path:
    """Create ``root/raw/r####`` dirs, each with a ``RAW-*.h5`` sentinel file."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for run_id in run_ids:
        run_dir = raw / f"r{run_id:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if sentinel:
            (run_dir / f"RAW-R{run_id:04d}-AGIPD00-S00000.h5").write_bytes(b"")
    return root


@pytest.fixture
def make_raw_tree() -> Callable[..., Path]:
    """Factory to build a ``raw/r####`` tree (optionally without sentinels)."""
    return _make_raw_tree


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """A ``p010400`` root with raw/r0001, r0007, r0010 plus ignorable noise."""
    root = tmp_path / "p010400"
    _make_raw_tree(root, [1, 7, 10])
    (root / "raw" / "scratch").mkdir()  # non-run directory → ignored
    (root / "raw" / "r0003.txt").write_text("")  # not a directory → ignored
    return root


@pytest.fixture
def write_elog() -> Callable[[Path, Iterable[Mapping[str, object]]], Path]:
    """Write a canonical ``elog.csv`` (columns ``scan_id``, ``sample``, + extras)."""

    def _write(root: Path, rows: Iterable[Mapping[str, object]]) -> Path:
        rows = list(rows)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        path = root / "elog.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    return _write


#: Attributes that differ between two runs of the same pass on the same input,
#: and so cannot be pinned: wall-clock, the machine, per-stage timings, and the
#: digests of the synthetic input files the fixtures build. ``poni`` is in the
#: list because pyFAI stamps the PONI it writes with the time it wrote it, so
#: the fixture's bytes — and every sha256 taken over them — move every session.
#: Everything else an output file carries is a function of its inputs.
VOLATILE_ATTRS = frozenset(
    {
        "config_hash",
        "host",
        "input_file_sha256",
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
    Absolute paths are reduced to their basename first, so that the recorded
    config stays in the digest — it is the record of every result-affecting
    field — without the fixtures' temporary directories entering it.

    :param path: the HDF5 file to digest.
    :param skip_attrs: attribute names to leave out, for values that differ
        between two runs of the same pass on the same input.
    :returns: the hex digest.
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
    """A canonical content digest of a pass output file — see :func:`_h5_digest`."""
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
    """``plan.record(train_id).status`` — the assertion both plan suites make."""

    def of(plan, train_id: int):
        return plan.record(train_id).status

    return of
