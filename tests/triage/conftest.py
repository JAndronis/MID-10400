"""Fixtures for the triage tests; the fakes themselves live in `triage_fakes`."""

from __future__ import annotations

from pathlib import Path

import pytest
from triage_fakes import write_mesh_store


@pytest.fixture
def damnit_root(tmp_path: Path) -> Path:
    root = tmp_path / "amore"
    (root / "extracted_data").mkdir(parents=True)
    return root


@pytest.fixture(scope="session")
def mesh_store(tmp_path_factory) -> Path:
    """`write_mesh_store`, shared read-only: copy it before changing a file."""
    return write_mesh_store(tmp_path_factory.mktemp("mesh_store") / "amore")
