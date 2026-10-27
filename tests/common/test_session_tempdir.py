"""Pins that the suite's scratch space lives inside the session's basetemp."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def test_gettempdir_is_inside_the_basetemp(tmp_path_factory) -> None:
    basetemp = tmp_path_factory.getbasetemp().resolve()
    assert basetemp in Path(tempfile.gettempdir()).resolve().parents


def test_a_child_process_sees_the_same_tempdir() -> None:
    """A spawned pool worker is a fresh interpreter that inherits only TMPDIR."""
    code = "import tempfile; print(tempfile.gettempdir())"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert Path(out.stdout.strip()).resolve() == Path(tempfile.gettempdir()).resolve()
