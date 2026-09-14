"""The error model and the pure per-frame integration.

Kept free of I/O and of EXtra-data so the self-test and the unit tests can
reach it without a run, the way :mod:`analysis.saxs.sparse` is for AGIPD.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from analysis.common.status import DataCheckFailed
from analysis.waxs.config import MODULE_SHAPE
from analysis.waxs.operator import WaxsOperator

__all__ = [
    "ErrorModel",
    "FrameResult",
    "extreme_pixels",
    "frame_maxima",
    "integrate_frame",
]


@dataclass(frozen=True, slots=True)
class ErrorModel:
    """``Var(x) = σ_read² + E·x``, in keV².

    ``Σc²·x`` is the Poisson variance of *integer photon counts*; on keV-valued
    data with a negative tail it is not a variance at all. For a pixel holding
    ``x`` keV deposited by ``E``-keV photons the count is ``x/E``, whose Poisson
    variance in keV² is ``E·x``; the readout term adds ``σ_read²``.

    **The result is deliberately not clamped.** Clamping at zero would bias the
    noise floor upward on every negative-noise pixel and that bias does not
    cancel, where the unbiased form's negatives do once a bin pools many frames.
    The per-frame negative bins are counted into the ledger instead.
    """

    read_noise_kev: float
    photon_energy_kev: float
    #: Where ``read_noise_kev`` came from: ``"configured"`` from
    #: ``cfg.read_noise_kev``, ``"measured"`` from this run's own dark cells, or
    #: ``"fallback"`` from ``cfg.read_noise_fallback_kev`` for a run that reads
    #: every storage cell and so has no dark one. See
    #: :func:`analysis.waxs.run._error_model` for the order of precedence.
    source: str = "configured"
    #: Frames the measurement averaged over; 0 when configured or fallen back.
    n_samples: int = 0
    dark_cells: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.read_noise_kev <= 0:
            raise ValueError(
                f"read_noise_kev must be positive, got {self.read_noise_kev}"
            )
        if self.photon_energy_kev <= 0:
            raise ValueError(
                f"photon_energy_kev must be positive, got {self.photon_energy_kev}"
            )

    def variance(self, x: np.ndarray) -> np.ndarray:
        """Per-pixel variance in keV², same shape and dtype family as ``x``."""
        return np.asarray(
            self.read_noise_kev**2 + self.photon_energy_kev * x, dtype=np.float32
        )

    @property
    def sha256(self) -> str:
        payload = (
            f"{self.read_noise_kev!r}:{self.photon_energy_kev!r}:"
            f"{self.source}:{self.n_samples}:{sorted(self.dark_cells)}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Per-frame sufficient statistics and scalars."""

    signal: np.ndarray  # float64 (npt,)  Σ c·x
    normalization: np.ndarray  # float64 (npt,)  Σ c·Ω over valid pixels
    variance: np.ndarray  # float64 (npt,)  Σ c²·Var(x)
    energy_valid: float  # keV summed over the kept pixels
    n_bad_pixels: int
    n_negative_variance_bins: int
    #: Largest value over the pixels that reach the integrator (the union mask).
    max_kev: float
    #: Largest value over the pixels the *static* mask keeps — the set D6 tests.
    #: With an extreme pixel excluded this is the value that was excluded, which
    #: is what makes the pair readable: see :func:`frame_maxima`.
    max_kev_static: float
    #: Kept pixels dropped for being non-finite or outside ``max_abs_kev``.
    n_extreme_pixels: int


def frame_maxima(
    x: np.ndarray, bad: np.ndarray, static_bad: np.ndarray
) -> tuple[float, float]:
    """``(max over the union mask, max over the static mask)``, in keV.

    The pair lets a ``DATA_CHECK_FAILED`` row explain itself: a large
    static-mask maximum beside an ordinary union one means the offending pixel
    never reached the integrator, where two large values mean it did.

    :returns: the two maxima, each 0.0 if its mask keeps no pixel. Non-finite
        values propagate, since a NaN maximum is itself why a frame failed.
    """
    flat = np.asarray(x).reshape(-1)
    return (
        float(flat[~bad].max(initial=0.0)),
        float(flat[~static_bad].max(initial=0.0)),
    )


def extreme_pixels(
    x: np.ndarray, excluded: np.ndarray, max_abs_kev: float
) -> np.ndarray:
    """Pixels carrying a value the pass will not integrate.

    Non-finite values go because the dynamic mask is itself carried as NaN.
    These pixels are excluded, **not** grounds for discarding the frame: that
    would throw away every good pixel in it, in step with crystallisation.
    ``FrameResult.n_extreme_pixels`` is the filter a reader applies instead.

    :param x: one frame, keV.
    :param excluded: the union bad-pixel mask, flat. Not the static mask alone:
        a wild value the masks already drop never reaches the integrator, so it
        is not this check's doing and must not be counted as such.
    :param max_abs_kev: values beyond ``±`` this are excluded.
    :returns: flat bool over the whole frame, True where a pixel is to be
        dropped by this check.
    """
    flat = np.asarray(x).reshape(-1)
    finite = np.isfinite(flat)
    return (~excluded) & (~finite | (np.abs(flat) > max_abs_kev))


def integrate_frame(
    ai: object,
    op: WaxsOperator,
    model: ErrorModel,
    x: np.ndarray,
    bad: np.ndarray,
    *,
    max_abs_kev: float,
) -> FrameResult:
    """Integrate one frame densely through pyFAI.

    ``bad`` is carried into pyFAI as NaN in both the data and the variance,
    which drops those pixels from the numerator and the denominator exactly.
    Pixels failing :func:`extreme_pixels` join it locally, leaving the caller's
    array untouched, and the frame is still integrated.

    :param ai: the ``AzimuthalIntegrator`` built by
        :func:`analysis.waxs.operator.build_operator`, with ``mask`` already set
        to ``op.static_bad``. Passing a mask here instead would rebuild pyFAI's
        sparse matrix on every frame.
    :param op: the operator whose q axis and static mask the engine was built on.
    :param model: the variance model applied to the kept pixels.
    :param x: one frame, ``(512, 1024)`` or flat, in keV.
    :param bad: this frame's bad-pixel mask, flat, as
        ``((m & mask_bits) != 0) | static_bad``.
    :param max_abs_kev: value bound handed to :func:`extreme_pixels`.
    :returns: the per-bin sums, maxima and pixel counts for this frame.
    :raises DataCheckFailed: *every* pixel is excluded, so there is nothing to
        integrate. Raised rather than returned, because a frame of zeros placed
        as if it had been measured is the one outcome no ledger column shows.
    """
    flat = np.asarray(x).reshape(-1)
    if flat.size != op.static_bad.size:
        raise ValueError(
            f"frame has {flat.size} pixels, the operator has {op.static_bad.size}"
        )
    if bad.shape != op.static_bad.shape:
        raise ValueError(f"bad has shape {bad.shape}, expected {op.static_bad.shape}")

    extreme = extreme_pixels(flat, bad, max_abs_kev)
    n_extreme = int(extreme.sum())
    if n_extreme:
        bad = bad | extreme
    if bad.all():
        raise DataCheckFailed(
            f"every one of the {bad.size} pixels is excluded: {n_extreme} of them "
            f"for being non-finite or outside ±{max_abs_kev} keV, the rest by the "
            "static mask and data.mask. There is nothing left to integrate"
        )

    signal = flat.astype(np.float32, copy=True)
    variance = model.variance(signal)
    signal[bad] = np.nan
    variance[bad] = np.nan

    result = ai.integrate1d(  # type: ignore[attr-defined]
        signal.reshape(MODULE_SHAPE),
        op.npt,
        method=op.method,
        unit=op.unit,
        mask=op.static_mask_2d,
        variance=variance.reshape(MODULE_SHAPE),
    )
    sum_signal = np.asarray(result.sum_signal, dtype=np.float64)
    sum_normalization = np.asarray(result.sum_normalization, dtype=np.float64)
    sum_variance = np.asarray(result.sum_variance, dtype=np.float64)

    kept_values = flat[~bad]
    occupied = sum_normalization > 0
    max_kev, max_kev_static = frame_maxima(flat, bad, op.static_bad)
    return FrameResult(
        signal=sum_signal,
        normalization=sum_normalization,
        variance=sum_variance,
        energy_valid=float(kept_values.sum()),
        n_bad_pixels=int(bad.sum()),
        n_negative_variance_bins=int((occupied & (sum_variance < 0)).sum()),
        max_kev=max_kev,
        max_kev_static=max_kev_static,
        n_extreme_pixels=n_extreme,
    )
