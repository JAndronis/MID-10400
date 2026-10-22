"""A read-only check of whether a run's output is already complete.

DAMNIT re-runs a pass on a finished run whenever a variable that depends on it
is requested. Such a call must return what the file holds and write nothing (no
operator, no masks, no provenance attribute), so that a reader of the file is
never disturbed by someone else's request.

A frame table is complete when all of these hold:

* it exists, opens and was finalised;
* its config hash is the current config's, and it has every column this pass
  writes;
* no row is :data:`~analysis.common.status.UNFINISHED`, and the train table
  accounts for every row;
* every :data:`~analysis.common.writer.GATING_PACKAGES` version is the
  installed one;
* its stored operator equals the one this pass would build now, array by
  array.

The last two make a pyFAI upgrade force a reprocess instead of leaving a file
half integrated by each version. The operator is compared array by array, not
by its sha256. The same build differs by one ULP in the solid angle between
AVX2 and AVX-512 nodes, and the ``allcpu`` partition mixes both, so a sha256
match would refuse a valid file whenever DAMNIT landed on the other class.

The pass-specific checks (the operator's layout, and the window sums for AGIPD)
live in ``analysis.saxs.complete`` and ``analysis.waxs.complete``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from analysis.common.status import UNFINISHED
from analysis.common.writer import (
    FrameTableWriter,
    OperatorCheck,
    PassConfig,
    version_changes,
)

__all__ = ["Completeness", "frame_table_reasons", "unfinished_reasons"]


@dataclass(frozen=True, slots=True)
class Completeness:
    """The verdict on one output file: true when complete, else why not."""

    path: Path
    reasons: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return not self.reasons

    def __str__(self) -> str:
        if not self.reasons:
            return f"{self.path}: complete"
        return f"{self.path}: not complete — " + "; ".join(self.reasons)


def unfinished_reasons(status: Any, noun: str) -> list[str]:
    """One reason per :data:`UNFINISHED` code present in a status column."""
    reasons = []
    for code in sorted(UNFINISHED):
        count = int((status == code).sum())
        if count:
            reasons.append(f"{count} {noun} {code.name}")
    return reasons


def frame_table_reasons(
    path: str | Path,
    cfg: PassConfig,
    writer: type[FrameTableWriter],
    operator_check: OperatorCheck | None,
) -> list[str]:
    """Why the frame table at ``path`` is not complete; empty when it is.

    :param writer: the pass's writer class, whose schema the file must have.
    :param operator_check: compares the stored operator with the one the pass
        would build now; ``None`` skips that comparison.
    """
    path = Path(path)
    if not path.exists():
        return ["no file"]
    try:
        with h5py.File(path, "r") as handle:
            return _reasons(handle, cfg, writer, operator_check)
    except (OSError, KeyError, ValueError) as error:
        return [f"unreadable: {error!r}"]


def _reasons(
    handle: h5py.File,
    cfg: PassConfig,
    writer: type[FrameTableWriter],
    operator_check: OperatorCheck | None,
) -> list[str]:
    if "provenance" not in handle:
        return ["no provenance group"]
    attrs = handle["provenance"].attrs
    reasons = []

    stored_hash, current_hash = attrs.get("config_hash"), cfg.config_hash()
    if stored_hash != current_hash:
        reasons.append(
            f"config hash {stored_hash}, the current config's is {current_hash}"
        )
    missing = writer._missing_columns(handle)
    if missing:
        reasons.append(f"the frame table lacks {missing}")
    stored_units = attrs.get("units")
    if writer.UNITS is not None and stored_units != writer.UNITS:
        reasons.append(
            f"it stores {stored_units or 'no units attribute'}, the pass writes "
            f"{writer.UNITS}"
        )
    if "status_summary" not in attrs:
        reasons.append("never finalised")

    if "frames/status" in handle and "trains/count" in handle:
        status = handle["frames/status"][:]
        reasons += unfinished_reasons(status, "frames")
        accounted = int(handle["trains/count"][:].sum())
        if accounted != status.size:
            reasons.append(
                f"the train table accounts for {accounted} rows, the frame "
                f"table has {status.size}"
            )

    reasons += [
        f"written with {change}"
        for change in version_changes(attrs.get("package_versions"))
    ]

    recorded = attrs.get("operator_sha256")
    stored = handle["operator"].attrs.get("sha256") if "operator" in handle else None
    if recorded != stored:
        reasons.append(
            f"provenance names operator {recorded}, /operator holds {stored}"
        )
    if operator_check is not None:
        reasons += operator_check(handle)
    return reasons
