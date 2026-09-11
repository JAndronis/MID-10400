"""Sparse-vs-pyFAI gate (context file §6.6).

The sparse path and the dense pyFAI engine must agree to a relative difference
below ``tolerance`` on every populated bin, and must agree exactly on *which*
bins are populated. In P1 this runs on synthetic frames; from P4 it runs on
real frames in the parent process before the worker pool is started, so a
geometry or masking mistake aborts the job instead of producing a run's worth
of wrong sums.

The reference is called the way CLAUDE.md pitfall 2 requires: an explicit
``variance=`` array, never ``ErrorModel.POISSON``, which floors the per-pixel
variance at 1 and inflates it by ~87x on 1 %-occupancy frames.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from analysis.saxs.operator import SparseOperator
from analysis.saxs.sparse import integrate_frame

__all__ = [
    "SelfTestFailed",
    "SelfTestReport",
    "compare_frame",
    "reference_frame",
    "run_selftest",
]

#: One frame's inputs: counts, this frame's bad mask, its cell's base mask and
#: the base denominator for that cell.
SelfTestFrame = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class SelfTestReport:
    """Worst-case agreement over the compared frames."""

    n_frames: int
    empty_bins_agree: bool
    max_rel_signal: float
    max_rel_normalization: float
    max_rel_variance: float
    tolerance: float
    passed: bool

    def as_provenance(self) -> dict[str, object]:
        """A JSON-serialisable record for the provenance group (§7)."""
        return asdict(self)


class SelfTestFailed(AssertionError):
    """The sparse path disagreed with the pyFAI engine."""

    def __init__(self, report: SelfTestReport) -> None:
        super().__init__(
            f"sparse/pyFAI self-test failed over {report.n_frames} frame(s): "
            f"empty_bins_agree={report.empty_bins_agree}, "
            f"max rel S={report.max_rel_signal:.3e}, "
            f"N={report.max_rel_normalization:.3e}, "
            f"V={report.max_rel_variance:.3e} "
            f"(tolerance {report.tolerance:.1e})"
        )
        self.report = report


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative difference ``|a - b| / max(|b|, 1e-12)`` (context file §6.6)."""
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def reference_frame(engine, op: SparseOperator, x: np.ndarray, bad: np.ndarray):
    """Integrate one frame densely with the pyFAI engine.

    Bad pixels are NaN in ``weights``, which is what removes them from both the
    signal and the normalization; ``variance`` is passed as the raw counts
    (Poisson, unfloored) and ``solidangle`` as the operator's own Ω, so that
    both paths normalise by the same quantity.
    """
    frame = np.where(bad, np.nan, x).astype(np.float32).reshape(op.shape)
    variance = x.astype(np.float32).reshape(op.shape)
    solidangle = op.omega.reshape(op.shape)
    return engine.integrate_ng(frame, variance=variance, solidangle=solidangle)


def compare_frame(
    engine,
    op: SparseOperator,
    x: np.ndarray,
    bad: np.ndarray,
    base_bad: np.ndarray,
    base_denominator: np.ndarray,
) -> tuple[float, float, float, bool]:
    """Compare one frame: ``(rel_S, rel_N, rel_V, empty_bins_agree)``.

    The relative differences are the maxima over the bins with ``N > 0``; bins
    are never compared where the reference normalization is zero, since the
    ratio there is meaningless.
    """
    result = integrate_frame(op, x, bad, base_bad, base_denominator)
    ref = reference_frame(engine, op, x, bad)

    populated = ref.normalization > 0
    empty_bins_agree = bool(np.array_equal(result.normalization > 0, populated))
    if not populated.any():
        return 0.0, 0.0, 0.0, empty_bins_agree

    return (
        float(_rel(result.signal, ref.signal)[populated].max()),
        float(_rel(result.normalization, ref.normalization)[populated].max()),
        float(_rel(result.variance, ref.variance)[populated].max()),
        empty_bins_agree,
    )


def run_selftest(
    engine,
    op: SparseOperator,
    frames: Iterable[SelfTestFrame],
    *,
    tolerance: float = 1e-6,
) -> SelfTestReport:
    """Compare every frame and raise unless all of them agree.

    :raises SelfTestFailed: carrying the :class:`SelfTestReport`, so the caller
        can write it into provenance before re-raising. The worker pool must
        never be started after this raises.
    """
    n_frames = 0
    worst = [0.0, 0.0, 0.0]
    empty_bins_agree = True

    for x, bad, base_bad, base_denominator in frames:
        n_frames += 1
        rel_s, rel_n, rel_v, agree = compare_frame(
            engine, op, x, bad, base_bad, base_denominator
        )
        worst = [max(worst[0], rel_s), max(worst[1], rel_n), max(worst[2], rel_v)]
        empty_bins_agree = empty_bins_agree and agree

    if n_frames == 0:
        raise ValueError("run_selftest needs at least one frame")

    passed = empty_bins_agree and max(worst) < tolerance
    report = SelfTestReport(
        n_frames=n_frames,
        empty_bins_agree=empty_bins_agree,
        max_rel_signal=worst[0],
        max_rel_normalization=worst[1],
        max_rel_variance=worst[2],
        tolerance=tolerance,
        passed=passed,
    )
    if not passed:
        raise SelfTestFailed(report)
    return report
