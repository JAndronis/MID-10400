"""The names DAMNIT stores the analysis products under.

A DAMNIT ``Group`` names its variables ``<group>.<var>``, so the AGIPD and
JUNGFRAU columns became ``agipd.saxs``, ``jungfrau.jf1`` and so on in
``src/amore/context.py``. Runs processed before that still hold the flat names,
until they are reprocessed, so a reader asks `stored` which of the two a file
carries instead of hardcoding one.
"""

from __future__ import annotations

from collections.abc import Container
from pathlib import Path

import h5py

__all__ = [
    "AGIPD_SAXS",
    "JF1",
    "JF2",
    "JF_COMBINED",
    "LEGACY",
    "stored",
    "stored_in_file",
]

AGIPD_SAXS = "agipd.saxs"
JF1 = "jungfrau.jf1"
JF2 = "jungfrau.jf2"
JF_COMBINED = "jungfrau.combined"

#: Current name -> the flat name runs processed before the groups carry.
LEGACY: dict[str, str] = {
    AGIPD_SAXS: "agipd_saxs",
    JF1: "jungfrau_waxs_jf1",
    JF2: "jungfrau_waxs_jf2",
    JF_COMBINED: "jungfrau_waxs_combined",
}
_CURRENT = {old: new for new, old in LEGACY.items()}


def stored(names: Container[str], name: str) -> str:
    """The name under which ``names`` holds this product: current, else legacy.

    :param names: anything supporting ``in``, such as an open h5py file.
    :param name: the current name, or a legacy one (translated first).
    :returns: the current name when present, else the legacy name when present,
        else the current name, so a missing product reports the name now in use.
    """
    current = _CURRENT.get(name, name)
    if current in names:
        return current
    legacy = LEGACY.get(current)
    if legacy is not None and legacy in names:
        return legacy
    return current


def stored_in_file(path: str | Path, name: str) -> str:
    """`stored` for a DAMNIT results file on disk."""
    with h5py.File(path, "r") as f:
        return stored(f, name)
