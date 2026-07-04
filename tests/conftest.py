"""Shared fixtures: synthetic EuXFEL ``raw/`` trees and canonical elogs.

Everything here is filesystem-only — no EXtra-data, no HDF5 content — so the
unit tests exercise the reader's pure paths (helpers, ``can_read``,
``list_runs``, ``get_run_path``) without the EuXFEL stack installed.
"""

from __future__ import annotations

import csv
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
