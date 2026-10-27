"""``analysis.threadenv`` must stay importable before the numerical stack is.

The scripts set these variables as their very first act, because every library
in the stack fixes its thread count at import time. A module that pulled in
numpy on the way to telling numpy how many threads to use would be useless, so
the leaf-ness is asserted rather than assumed.
"""

from __future__ import annotations

import subprocess
import sys


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
