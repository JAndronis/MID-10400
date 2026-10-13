"""Whether one detector's JUNGFRAU output is complete.

See :mod:`analysis.common.complete` for what complete means. A run with no lit
cell writes no file at all, so it is never complete here; the pass measures its
cells again — a few seconds — and returns ``None`` again, writing nothing.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import h5py
import numpy as np

from analysis.common.complete import Completeness, frame_table_reasons
from analysis.waxs.config import JungfrauWaxsConfig
from analysis.waxs.operator import WaxsOperator, build_operator
from analysis.waxs.writer import JungfrauWaxsWriter

__all__ = ["expected_operator", "is_complete", "operator_differences"]

#: Relative tolerance on the q axis and on the solid angle, which the file
#: stores as float32. Both are computed floats, which can differ by one ULP
#: between node classes as AGIPD's solid angle does
#: (:data:`analysis.saxs.complete.OMEGA_RTOL`); float32 rounding can turn that
#: into one float32 ULP, 6e-8. The static mask is compared exactly.
RTOL = 1e-6

#: Operators by run-neutral config hash, for this process.
_EXPECTED: dict[str, WaxsOperator] = {}


def expected_operator(
    cfg: JungfrauWaxsConfig, poni_file: str | Path | None = None
) -> WaxsOperator:
    """The operator this pass would build for ``cfg`` now, on this node.

    Built for real — 0.17 s on r0423 — and memoised per process with the
    proposal and run zeroed, as for AGIPD
    (:func:`analysis.saxs.complete.expected_operator`).

    :param poni_file: overrides ``cfg.poni_file``; not memoised.
    """
    if poni_file is not None:
        return build_operator(cfg, poni_file)[0]
    key = replace(cfg, proposal=0, run=0).config_hash()
    if key not in _EXPECTED:
        _EXPECTED[key] = build_operator(cfg)[0]
    return _EXPECTED[key]


def operator_differences(handle: h5py.File, op: WaxsOperator) -> list[str]:
    """How the operator stored in ``handle`` differs from ``op``; empty if not."""
    group = handle.get("operator")
    names = ("q", "omega", "static_bad_packed")
    if group is None or any(name not in group for name in names):
        return ["the file stores no complete operator"]
    reasons = []
    if not np.array_equal(group["static_bad_packed"][()], np.packbits(op.static_bad)):
        reasons.append("operator static mask differs from the one built now")
    for name in ("q", "omega"):
        stored, built = group[name][()], getattr(op, name)
        if stored.shape != built.shape or not np.allclose(
            stored, built, rtol=RTOL, atol=0.0
        ):
            reasons.append(f"operator {name} differs by more than {RTOL:g} relative")
    return reasons


def is_complete(
    path: str | Path | None,
    cfg: JungfrauWaxsConfig,
    *,
    operator: WaxsOperator | None = None,
    poni_file: str | Path | None = None,
) -> Completeness:
    """Whether one detector's frame table is complete. Read-only.

    :param path: the file; ``None`` is ``cfg.output_file``.
    :param operator: the operator to compare against, when the caller has
        already built it; ``None`` builds it (:func:`expected_operator`).
    :param poni_file: passed to :func:`expected_operator`.
    """
    path = Path(path) if path is not None else cfg.output_file
    op = operator if operator is not None else expected_operator(cfg, poni_file)
    reasons = frame_table_reasons(
        path, cfg, JungfrauWaxsWriter, partial(operator_differences, op=op)
    )
    return Completeness(path, tuple(reasons))
