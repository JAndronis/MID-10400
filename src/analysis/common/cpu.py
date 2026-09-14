"""Process, thread and file-digest helpers shared by every pass.

Both passes spawn one single-threaded worker per physical core and hash their
input files into the config hash, so the counting, the thread pinning and the
digest live here rather than in either detector's ``config``.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from analysis.threadenv import THREAD_ENV, set_thread_env

__all__ = [
    "CPU_TOPOLOGY_ROOT",
    "THREAD_ENV",
    "default_pool",
    "file_sha256",
    "package_versions",
    "phase",
    "physical_cores",
    "set_thread_env",
]

#: Linux CPU topology, where a core's hyperthread siblings are listed.
CPU_TOPOLOGY_ROOT = Path("/sys/devices/system/cpu")


#: Packages whose versions are recorded in provenance.
RECORDED_PACKAGES = (
    "numpy",
    "h5py",
    "EXtra-data",
    "EXtra-geom",
    "euxfel-EXtra",
    "pyFAI",
    "fabio",
)


def physical_cores(root: Path = CPU_TOPOLOGY_ROOT) -> int | None:
    """Physical cores available to this process, or ``None`` off Linux.

    Counts distinct hyperthread-sibling groups over the CPUs in this process's
    affinity mask, so a core contributes once however many threads it exposes
    and a cgroup-restricted job is not told about cores it cannot use.

    :param root: sysfs CPU topology root; the tests point it at a fixture.
    """
    if not hasattr(os, "sched_getaffinity"):
        return None
    groups: set[str] = set()
    for cpu in os.sched_getaffinity(0):
        try:
            siblings = (root / f"cpu{cpu}/topology/thread_siblings_list").read_text()
        except OSError:
            return None
        groups.add(siblings.strip())
    return len(groups) or None


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_pool(n_workers: int, **kwargs: Any) -> ProcessPoolExecutor:
    """The real pool: spawned processes only, never forked."""
    return ProcessPoolExecutor(
        max_workers=n_workers, mp_context=get_context("spawn"), **kwargs
    )


@contextmanager
def phase(into: dict[str, float], name: str) -> Iterator[None]:
    """Time one parent-side setup phase into ``into``.

    Everything before the pool starts is serial, and recording it is what keeps
    that time attributable instead of leaving it as the unexplained gap between
    the wall time and the workers' own timings.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        into[name] = time.perf_counter() - started


def package_versions(names: tuple[str, ...] = RECORDED_PACKAGES) -> dict[str, str]:
    """Installed versions of the packages a result depends on."""
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "not installed"
    return found
