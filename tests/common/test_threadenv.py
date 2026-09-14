"""``analysis.threadenv`` must stay importable before the numerical stack is.

The scripts set these variables as their very first act, because every library
in the stack fixes its thread count at import time. A module that pulled in
numpy on the way to telling numpy how many threads to use would be useless, so
the leaf-ness is asserted rather than assumed.
"""

from __future__ import annotations

import subprocess
import sys

from analysis.threadenv import THREAD_ENV, set_thread_env


def test_importing_threadenv_pulls_in_nothing_heavy():
    code = (
        "import sys, analysis.threadenv; "
        "print([m for m in ('numpy', 'h5py', 'pyFAI', 'extra_data') "
        "if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_every_library_in_the_stack_is_pinned(monkeypatch):
    """OpenBLAS included: the scripts pinned it while the library did not."""
    for name in THREAD_ENV:
        monkeypatch.delenv(name, raising=False)
    set_thread_env()

    import os

    assert {name: os.environ[name] for name in THREAD_ENV} == dict.fromkeys(
        THREAD_ENV, "1"
    )
    assert "OPENBLAS_NUM_THREADS" in THREAD_ENV
