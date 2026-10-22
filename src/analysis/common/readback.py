"""Read a written HDF5 file back from disk before calling it finished.

A quota that fills during a write can leave a finalised file whose chunks
raise ``filter returned failure during read`` while its attributes and
statuses read fine. A completeness check that reads only those calls such a
file complete. A pass that reads its own output back, from a fresh handle after
an ``fsync``, cannot finalise it.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py

__all__ = ["READ_BYTES", "UnreadableOutput", "sync", "unreadable"]

#: Largest slab read at once, rounded down to whole chunks along the first axis.
READ_BYTES = 1 << 28


class UnreadableOutput(RuntimeError):
    """A file just written does not read back."""


def sync(path: str | Path) -> None:
    """Push ``path``'s dirty pages to disk; a failed write-back raises here.

    Without it a full quota can surface only at write-back, after every read of
    the page cache has already succeeded.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def unreadable(path: str | Path) -> list[str]:
    """Read every dataset of ``path`` in full; one entry for each that fails.

    ``visititems`` stops at the first callback returning anything but ``None``,
    so failures are collected, never returned.
    """
    failures: list[str] = []

    def read(name: str, obj: object) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        try:
            if obj.ndim == 0 or obj.shape[0] == 0:
                obj[()]
                return
            row = max(1, obj.dtype.itemsize * int(obj.size // obj.shape[0]))
            step = max(1, READ_BYTES // row)
            if obj.chunks is not None:
                step = max(obj.chunks[0], step - step % obj.chunks[0])
            for start in range(0, obj.shape[0], step):
                obj[start : start + step]
        except Exception as error:  # noqa: BLE001 - every failure is reported
            failures.append(f"{name}: {error!r}")

    try:
        with h5py.File(path, "r") as handle:
            handle.visititems(read)
    except Exception as error:  # noqa: BLE001
        failures.append(f"opening: {error!r}")
    return failures
