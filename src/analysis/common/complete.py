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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.common.status import UNFINISHED
from analysis.common.writer import (
    FrameTableWriter,
    OperatorCheck,
    PassConfig,
    version_changes,
)

__all__ = [
    "Completeness",
    "expected_operator",
    "frame_table_reasons",
    "operator_differences",
    "unfinished_reasons",
]


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


def expected_operator[Op](
    cache: dict[str, Op], cfg: Any, build: Callable[[], Op]
) -> Op:
    """The operator ``build`` makes for ``cfg``, memoised in ``cache`` per process.

    Built for real rather than inferred from library versions, because it is
    cheap and needs no assumption about what a pyFAI release changes. No input
    of an operator depends on the run, so the key is the config hash with the
    proposal and the run zeroed, and a script checking hundreds of runs builds
    it once.
    """
    key = replace(cfg, proposal=0, run=0).config_hash()
    if key not in cache:
        cache[key] = build()
    return cache[key]


def operator_differences(
    handle: h5py.File,
    sha256: str,
    *,
    exact: Mapping[str, np.ndarray],
    toleranced: Mapping[str, np.ndarray],
    rtol: float,
) -> list[str]:
    """How the operator stored in ``handle["operator"]`` differs; empty if not.

    An equal sha256 settles it without reading the arrays, since it is written
    with them and covers all of them. An unequal one does not, because the same
    build differs between node classes in its computed floats.

    :param sha256: the sha256 of the operator built now.
    :param exact: stored dataset name to the array it must equal bit for bit.
    :param toleranced: stored dataset name to the array it must equal to
        ``rtol`` relative, for the computed floats.
    """
    group = handle.get("operator")
    if group is None or any(name not in group for name in (*exact, *toleranced)):
        return ["the file stores no complete operator"]
    if group.attrs.get("sha256") == sha256:
        return []
    reasons = [
        f"operator {name} differs from the one built now"
        for name, built in exact.items()
        if not np.array_equal(group[name][()], built)
    ]
    for name, built in toleranced.items():
        stored = group[name][()]
        if stored.shape != built.shape or not np.allclose(
            stored, built, rtol=rtol, atol=0.0
        ):
            reasons.append(f"operator {name} differs by more than {rtol:g} relative")
    return reasons
