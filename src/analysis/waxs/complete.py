"""Whether one detector's JUNGFRAU output is complete.

:mod:`analysis.common.complete` defines complete. A run with no lit cell writes
no file, so it is never complete here. The pass measures its cells again, which
takes seconds, and returns ``None`` again without writing anything.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import h5py
import numpy as np

from analysis.common import complete as common
from analysis.common.complete import Completeness, frame_table_reasons
from analysis.waxs.config import JungfrauWaxsConfig
from analysis.waxs.operator import WaxsOperator, build_operator
from analysis.waxs.writer import JungfrauWaxsWriter

__all__ = ["expected_operator", "is_complete", "operator_differences"]

#: Relative tolerance on the q axis and on the solid angle, which the file
#: stores as float32. Both are computed floats that can differ by one ULP
#: between node classes, as AGIPD's solid angle does
#: (:data:`analysis.saxs.operator.OMEGA_RTOL`). Rounding to float32 can turn
#: that into one float32 ULP, 6e-8. The static mask is compared exactly.
RTOL = 1e-6

#: Operators by run-neutral config hash, for this process.
_EXPECTED: dict[str, WaxsOperator] = {}


def expected_operator(
    cfg: JungfrauWaxsConfig, poni_file: str | Path | None = None
) -> WaxsOperator:
    """The operator this pass would build for ``cfg`` now, on this node.

    :param poni_file: overrides ``cfg.poni_file``. Not memoised.
    """
    if poni_file is not None:
        return build_operator(cfg, poni_file)[0]
    return common.expected_operator(_EXPECTED, cfg, lambda: build_operator(cfg)[0])


def operator_differences(handle: h5py.File, op: WaxsOperator) -> list[str]:
    """How the operator stored in ``handle`` differs from ``op``; empty if not.

    The static mask must be equal, ``q`` and ``omega`` equal to :data:`RTOL`.
    """
    return common.operator_differences(
        handle,
        op.sha256,
        exact={"static_bad_packed": np.packbits(op.static_bad)},
        toleranced={"q": op.q, "omega": op.omega},
        rtol=RTOL,
    )


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
