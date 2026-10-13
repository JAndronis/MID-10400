"""Fixtures for the triage tests; the fakes themselves live in `triage_fakes`."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def damnit_root(tmp_path: Path) -> Path:
    root = tmp_path / "amore"
    (root / "extracted_data").mkdir(parents=True)
    return root
