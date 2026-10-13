"""The XPCS diagnostics reuse the triage's synthetic XPCS run (`triage_fakes`)."""

from __future__ import annotations

import sys
from pathlib import Path

# pytest puts each test file's own directory on sys.path; the triage fakes live in
# the sibling directory, and are needed whether or not tests/triage is collected.
TRIAGE_TESTS = str(Path(__file__).resolve().parents[1] / "triage")
if TRIAGE_TESTS not in sys.path:
    sys.path.append(TRIAGE_TESTS)
