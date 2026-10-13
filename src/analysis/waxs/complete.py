"""Whether one detector's JUNGFRAU output is complete.

See :mod:`analysis.common.complete` for what complete means. A run with no lit
cell writes no file at all, so it is never complete here; the pass measures its
cells again — a few seconds — and returns ``None`` again, writing nothing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from analysis.common.complete import Completeness, frame_table_reasons
from analysis.waxs.config import JungfrauWaxsConfig
from analysis.waxs.operator import build_operator
from analysis.waxs.writer import JungfrauWaxsWriter

__all__ = ["expected_operator_sha256", "is_complete"]

#: Operator sha256 by run-neutral config hash, for this process.
_EXPECTED: dict[str, str] = {}


def expected_operator_sha256(
    cfg: JungfrauWaxsConfig, poni_file: str | Path | None = None
) -> str:
    """The sha256 of the operator this pass would build for ``cfg`` now.

    Built for real — 0.17 s on r0423 — and memoised per process with the
    proposal and run zeroed, as for AGIPD
    (:func:`analysis.saxs.complete.expected_operator_sha256`).

    :param poni_file: overrides ``cfg.poni_file``; not memoised.
    """
    if poni_file is not None:
        return build_operator(cfg, poni_file)[0].sha256
    key = replace(cfg, proposal=0, run=0).config_hash()
    if key not in _EXPECTED:
        _EXPECTED[key] = build_operator(cfg)[0].sha256
    return _EXPECTED[key]


def is_complete(
    path: str | Path | None,
    cfg: JungfrauWaxsConfig,
    *,
    operator_sha256: str | None = None,
    poni_file: str | Path | None = None,
) -> Completeness:
    """Whether one detector's frame table is complete. Read-only.

    :param path: the file; ``None`` is ``cfg.output_file``.
    :param operator_sha256: the operator to compare against, when the caller has
        already built it; ``None`` builds it.
    :param poni_file: passed to :func:`expected_operator_sha256`.
    """
    path = Path(path) if path is not None else cfg.output_file
    if operator_sha256 is None:
        operator_sha256 = expected_operator_sha256(cfg, poni_file)
    reasons = frame_table_reasons(path, cfg, JungfrauWaxsWriter, operator_sha256)
    return Completeness(path, tuple(reasons))
