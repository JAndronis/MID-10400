"""The NaN-equivalence gate.

This gate checks the design in :mod:`analysis.waxs.operator`. The reference
path passes every excluded pixel of a frame as ``integrate1d(mask=)``, which is
correct and slow, because pyFAI rebuilds its sparse matrix whenever the mask
changes. The production path builds the matrix once with the static mask and
carries the rest as NaN. Both integrate the same photon counts, so the gate
tests that equivalence and nothing else. A pyFAI change in how non-finite
values are preprocessed would break it, and this gate catches that before a run
is integrated. The reference runs on its own ``AzimuthalIntegrator``, rebuilt
from the operator's PONI text, so its per-frame masks never evict the
production engine's cached matrix.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from analysis.common.arrays import relative_difference
from analysis.waxs.config import MODULE_SHAPE
from analysis.waxs.integrate import integrate_prepared, prepare_frame
from analysis.waxs.operator import WaxsOperator
from analysis.waxs.photons import PhotonCounting

__all__ = [
    "SelfTestFailed",
    "SelfTestReport",
    "compare_frame",
    "reference_frame",
    "reference_integrator",
    "run_selftest",
]

#: The two paths agree exactly on real frames. This leaves room for a different
#: summation order and none for a different answer.
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
    photons: np.ndarray,
    excluded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(signal, normalization, variance)`` with ``excluded`` passed as ``mask=``.

    :param photons: float32 counts, flat. Values under ``excluded`` are replaced
        by zero, so the mask alone removes them.
    :param excluded: flat bool, the pixels to leave out.
    """
    counts = np.where(excluded, np.float32(0), np.asarray(photons, np.float32))
    result = ai.integrate1d(
        counts.reshape(MODULE_SHAPE),
        op.npt,
        method=op.method,
        unit=op.unit,
        # A fresh writable array, because pyFAI refuses a read-only mask.
        mask=np.ascontiguousarray(excluded.reshape(MODULE_SHAPE)),
        variance=counts.copy().reshape(MODULE_SHAPE),
    )
    return (
        np.asarray(result.sum_signal, dtype=np.float64),
        np.asarray(result.sum_normalization, dtype=np.float64),
        np.asarray(result.sum_variance, dtype=np.float64),
    )


def _relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(relative_difference(got, want)))


def compare_frame(
    ai: AzimuthalIntegrator,
    reference_ai: AzimuthalIntegrator,
    op: WaxsOperator,
    counting: PhotonCounting,
    x: np.ndarray,
    dynamic: np.ndarray,
    *,
    max_abs_kev: float,
) -> tuple[float, float, float]:
    """Relative difference in S, N and V between the two paths, on one frame.

    Both paths integrate the same counted frame, the one
    :func:`~analysis.waxs.integrate.prepare_frame` produces, so the comparison
    tests the NaN-versus-mask equivalence and not the counts themselves.

    :raises SelfTestFailed: the two paths disagree about which bins are
        populated at all, which no tolerance would catch.
    """
    prepared = prepare_frame(op, counting, x, dynamic, max_abs_kev=max_abs_kev)
    got_s, got_n, got_v = integrate_prepared(ai, op, prepared)
    want_s, want_n, want_v = reference_frame(
        reference_ai, op, prepared.photons, prepared.excluded
    )
    if not np.array_equal(got_n > 0, want_n > 0):
        raise SelfTestFailed(
            "the NaN path and the per-frame-mask reference populate different "
            f"bins: {int((got_n > 0).sum())} against "
            f"{int((want_n > 0).sum())} of {op.npt}"
        )
    occupied = want_n > 0
    return (
        _relative(got_s[occupied], want_s[occupied]),
        _relative(got_n[occupied], want_n[occupied]),
        _relative(got_v[occupied], want_v[occupied]),
    )


def run_selftest(
    ai: AzimuthalIntegrator,
    op: WaxsOperator,
    counting: PhotonCounting,
    frames: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_abs_kev: float,
    tolerance: float = TOLERANCE,
) -> SelfTestReport:
    """Gate the pass on real frames before any pool is started.

    :param frames: ``(x, dynamic)`` pairs, ``x`` in keV and ``dynamic`` this
        frame's flat ``data.mask`` exclusion, without the static mask.
    :raises SelfTestFailed: any frame exceeds ``tolerance``.
    """
    if not frames:
        raise SelfTestFailed("no frames to self-test on")
    reference_ai = reference_integrator(op)
    worst = [0.0, 0.0, 0.0]
    for x, dynamic in frames:
        for index, value in enumerate(
            compare_frame(
                ai, reference_ai, op, counting, x, dynamic, max_abs_kev=max_abs_kev
            )
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
