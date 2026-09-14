"""Thread pinning, with no imports of its own.

Every numerical library in this stack decides its thread count at import time,
so the variables have to be set before anything imports numpy, pyFAI or
EXtra-data. That is why this is a leaf module: importing
:mod:`analysis.common.cpu` would pull in numpy and h5py through the package,
which is exactly what the caller is trying to get ahead of.
"""

from __future__ import annotations

import os

__all__ = ["THREAD_ENV", "set_thread_env"]

#: Environment variables that pin every numerical library to one thread.
THREAD_ENV = (
    "EXTRA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def set_thread_env() -> None:
    """Pin every numerical library to one thread, before the pool is built.

    Spawned children inherit the environment at process start.
    """
    for name in THREAD_ENV:
        os.environ[name] = "1"
