"""The NaN-equivalence gate (WAXS context file §4, replacing the AGIPD §6.6 gate).

The AGIPD self-test compares a sparse kernel against pyFAI. Here both sides are
pyFAI, so that comparison would be an identity check and the context file says
as much. This gate has content instead: it is the correctness argument for the
whole design in :mod:`analysis.waxs.operator`.

The **reference** masks each frame the obvious way — ``mask = static | dynamic``
passed to ``integrate1d`` — which is correct and slow, because pyFAI rebuilds
its sparse matrix whenever the mask checksum changes. The **production** path
builds the matrix once with the static mask and carries the dynamic mask as NaN.
Measured on r0423, the two agree exactly. If a future pyFAI changes how
non-finite values are preprocessed, that equivalence is the thing that breaks,
and this gate is what catches it before a whole run is integrated.

The reference runs on its own ``AzimuthalIntegrator``, rebuilt from the
operator's PONI text, so that hammering it with per-frame masks never evicts
the production engine's cached matrix.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from analysis.waxs.config import MODULE_SHAPE
from analysis.waxs.integrate import ErrorModel, extreme_pixels, integrate_frame
from analysis.waxs.operator import WaxsOperator

__all__ = [
    "SelfTestFailed",
    "SelfTestReport",
    "compare_frame",
    "reference_frame",
    "reference_integrator",
    "run_selftest",
]

#: The two paths were measured to agree exactly; this leaves room for a
#: different summation order without leaving room for a different answer.
TOLERANCE = 1e-9


class SelfTestFailed(RuntimeError):
    """The NaN path and the per-frame-mask reference disagree."""


@dataclass(frozen=True, slots=True)
class SelfTestReport:
    n_frames: int
    max_rel_signal: float
    max_rel_normalization: float
    max_rel_variance: float
    tolerance: float


def reference_integrator(op: WaxsOperator) -> AzimuthalIntegrator:
    """A second integrator on the same geometry, for the reference path."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "reference.poni"
        path.write_text(op.poni_text)
        return AzimuthalIntegrator.sload(path)


def reference_frame(
    ai: AzimuthalIntegrator,
    op: WaxsOperator,
    model: ErrorModel,
    x: np.ndarray,
    bad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(signal, normalization, variance)`` with the union passed as ``mask=``."""
    flat = np.asarray(x).reshape(-1).astype(np.float32)
    result = ai.integrate1d(
        flat.reshape(MODULE_SHAPE),
        op.npt,
        method=op.method,
        unit=op.unit,
        # A fresh writable array: ``frame_bad`` builds one per frame, so this
        # is never a read-only view of the operator's static mask.
        mask=np.ascontiguousarray(bad.reshape(MODULE_SHAPE)),
        variance=model.variance(flat).reshape(MODULE_SHAPE),
    )
    return (
        np.asarray(result.sum_signal, dtype=np.float64),
        np.asarray(result.sum_normalization, dtype=np.float64),
        np.asarray(result.sum_variance, dtype=np.float64),
    )


def _relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12)))


def compare_frame(
    ai: AzimuthalIntegrator,
    reference_ai: AzimuthalIntegrator,
    op: WaxsOperator,
    model: ErrorModel,
    x: np.ndarray,
    bad: np.ndarray,
    *,
    max_abs_kev: float,
) -> tuple[float, float, float]:
    """Relative difference in S, N and V between the two paths, on one frame.

    :raises SelfTestFailed: the two paths disagree about which bins are
        populated at all, which no tolerance would catch.
    """
    # ``integrate_frame`` excludes extreme pixels itself; the reference has to
    # be given the same union or the two would differ over the policy rather
    # than over the NaN-vs-mask equivalence this gate exists to test.
    bad = bad | extreme_pixels(x, bad, max_abs_kev)
    got = integrate_frame(ai, op, model, x, bad, max_abs_kev=max_abs_kev)
    want_s, want_n, want_v = reference_frame(reference_ai, op, model, x, bad)
    if not np.array_equal(got.normalization > 0, want_n > 0):
        raise SelfTestFailed(
            "the NaN path and the per-frame-mask reference populate different "
            f"bins: {int((got.normalization > 0).sum())} against "
            f"{int((want_n > 0).sum())} of {op.npt}"
        )
    occupied = want_n > 0
    return (
        _relative(got.signal[occupied], want_s[occupied]),
        _relative(got.normalization[occupied], want_n[occupied]),
        _relative(got.variance[occupied], want_v[occupied]),
    )


def run_selftest(
    ai: AzimuthalIntegrator,
    op: WaxsOperator,
    model: ErrorModel,
    frames: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_abs_kev: float,
    tolerance: float = TOLERANCE,
) -> SelfTestReport:
    """Gate the pass on real frames before any pool is started.

    :param frames: ``(x, bad)`` pairs, ``x`` in keV and ``bad`` the flat union
        of the static mask and this frame's ``data.mask``.
    :raises SelfTestFailed: any frame exceeds ``tolerance``.
    """
    if not frames:
        raise SelfTestFailed("no frames to self-test on")
    reference_ai = reference_integrator(op)
    worst = [0.0, 0.0, 0.0]
    for x, bad in frames:
        for index, value in enumerate(
            compare_frame(ai, reference_ai, op, model, x, bad, max_abs_kev=max_abs_kev)
        ):
            worst[index] = max(worst[index], value)

    report = SelfTestReport(len(frames), worst[0], worst[1], worst[2], tolerance)
    if max(worst) > tolerance:
        raise SelfTestFailed(
            f"the NaN path disagrees with the per-frame-mask reference on "
            f"{len(frames)} frames: max relative S {worst[0]:.2e}, "
            f"N {worst[1]:.2e}, V {worst[2]:.2e}, tolerance {tolerance:.1e}. "
            "pyFAI's handling of non-finite values may have changed; the "
            "per-frame-mask path is still correct, and slow."
        )
    return report
