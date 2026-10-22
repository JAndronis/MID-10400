"""The per-frame path: masks, value check, photon counting, integration.

Free of I/O and of EXtra-data, so the self-test and the unit tests can reach it
without a run. Per frame, in this order:

1. the union mask, ``data.mask`` (dynamic) or the static ``.edf``;
2. :func:`extreme_pixels` drops kept pixels that are non-finite or beyond
   ``±max_abs_kev``, before counting, so a gain-switched pixel or a salt Bragg
   spot never reaches a level or a count;
3. :mod:`analysis.waxs.photons` estimates each ASIC's zero level from the
   frame itself and counts photons. An ASIC with no level is not counted;
4. pyFAI integrates the counts with the excluded pixels carried as NaN, and
   ``variance = counts``, which is Poisson and gives ``Σc²·n`` per bin.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from analysis.common.status import DataCheckFailed
from analysis.waxs.config import MODULE_SHAPE
from analysis.waxs.operator import WaxsOperator
from analysis.waxs.photons import (
    ASIC_ID,
    PhotonCounting,
    count_photons,
    zero_levels,
)

__all__ = [
    "ErrorModel",
    "FrameResult",
    "PreparedFrame",
    "extreme_pixels",
    "frame_maxima",
    "integrate_frame",
    "integrate_prepared",
    "prepare_frame",
]


@dataclass(frozen=True, slots=True)
class ErrorModel:
    """The readout noise and where it came from, recorded with every file.

    A diagnostic only. The stored variance is Poisson on the counts and does
    not use it. It is one standard deviation over pixel, dark cell and train
    around a single mean, so pixel offsets, cell offsets and any common mode
    all add to it. It therefore overstates the pure per-pixel noise.
    """

    read_noise_kev: float
    photon_energy_kev: float
    #: Where ``read_noise_kev`` came from: ``"configured"`` from
    #: ``cfg.read_noise_kev``, ``"measured"`` from this run's own dark cells, or
    #: ``"fallback"`` from ``cfg.read_noise_fallback_kev`` for a run that reads
    #: every storage cell and so has no dark one. The order of precedence is in
    #: :func:`analysis.waxs.run._error_model`.
    source: str = "configured"
    #: Samples the measurement averaged over. 0 when configured or fallen back.
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

    @property
    def sha256(self) -> str:
        payload = (
            f"{self.read_noise_kev!r}:{self.photon_energy_kev!r}:"
            f"{self.source}:{self.n_samples}:{sorted(self.dark_cells)}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    """One frame counted and masked, ready for either integration path."""

    #: float32 ``(npix,)`` photon counts, NaN wherever :attr:`excluded`.
    photons: np.ndarray
    #: bool ``(npix,)``, the union mask, extreme pixels and uncounted ASICs.
    excluded: np.ndarray
    #: float32 ``(8,)`` zero level per ASIC in keV, NaN where there was none.
    asic_level: np.ndarray
    #: ASICs with pixels to integrate but no level, whose pixels were dropped.
    n_level_missing: int
    #: Kept pixels dropped for being non-finite or outside ``max_abs_kev``.
    n_extreme_pixels: int


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Per-frame sufficient statistics and scalars."""

    signal: np.ndarray  # float64 (npt,)  Σ c·n, photons
    normalization: np.ndarray  # float64 (npt,)  Σ c·Ω over valid pixels
    variance: np.ndarray  # float64 (npt,)  Σ c²·n, photons²
    #: Raw keV summed over the integrated pixels, before any level is taken out.
    energy_valid: float
    #: Photons summed over the integrated pixels.
    n_photons: int
    n_bad_pixels: int
    n_negative_variance_bins: int
    #: Largest value over the pixels that reach the integrator.
    max_kev: float
    #: Largest value over the pixels the static mask keeps. With an extreme
    #: pixel excluded this is the value that was excluded (:func:`frame_maxima`).
    max_kev_static: float
    #: Kept pixels dropped for being non-finite or outside ``max_abs_kev``.
    n_extreme_pixels: int
    #: float32 ``(8,)`` zero level per ASIC in keV, NaN where there was none.
    asic_level: np.ndarray
    n_level_missing: int
    #: The energy the counts were taken at, as stored.
    count_energy_kev: float


def frame_maxima(
    x: np.ndarray, bad: np.ndarray, static_bad: np.ndarray
) -> tuple[float, float]:
    """``(max over the union mask, max over the static mask)``, in keV.

    The pair says what an excluded pixel held. A large static-mask maximum
    beside an ordinary union one means the large value never reached the
    integrator. Two large values mean it did.

    :returns: the two maxima, each 0.0 if its mask keeps no pixel. Non-finite
        values propagate, since a NaN maximum is itself a reason to exclude.
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

    Non-finite values go because the excluded pixels are themselves carried as
    NaN. Only the pixels are excluded and the frame is kept, since discarding
    it would lose every good pixel in it. ``FrameResult.n_extreme_pixels`` is
    the filter a reader applies instead.

    :param x: one frame, keV.
    :param excluded: the union bad-pixel mask, flat. A value the masks already
        drop never reaches the integrator, so this check does not count it.
    :param max_abs_kev: values beyond ``±`` this are excluded.
    :returns: flat bool over the whole frame, True where a pixel is to be
        dropped by this check.
    """
    flat = np.asarray(x).reshape(-1)
    finite = np.isfinite(flat)
    return (~excluded) & (~finite | (np.abs(flat) > max_abs_kev))


def prepare_frame(
    op: WaxsOperator,
    counting: PhotonCounting,
    x: np.ndarray,
    dynamic: np.ndarray,
    *,
    max_abs_kev: float,
) -> PreparedFrame:
    """Mask, check and count one frame. Leaves the caller's arrays untouched.

    :param op: the operator whose static mask the engine was built on.
    :param counting: the photon-counting settings.
    :param x: one frame, ``(512, 1024)`` or flat, in keV.
    :param dynamic: this frame's own exclusion, flat,
        ``(data.mask & mask_bits) != 0``. Not ORed with the static mask, which
        the operator supplies, because the zero level can be estimated over
        ``.edf``-masked pixels too.
    :param max_abs_kev: value bound handed to :func:`extreme_pixels`.
    """
    flat = np.asarray(x).reshape(-1)
    if flat.size != op.static_bad.size:
        raise ValueError(
            f"frame has {flat.size} pixels, the operator has {op.static_bad.size}"
        )
    if dynamic.shape != op.static_bad.shape:
        raise ValueError(
            f"dynamic has shape {dynamic.shape}, expected {op.static_bad.shape}"
        )

    bad = dynamic | op.static_bad
    extreme = extreme_pixels(flat, bad, max_abs_kev)
    n_extreme = int(extreme.sum())
    if n_extreme:
        bad = bad | extreme

    usable = ~dynamic & np.isfinite(flat)
    if counting.level_pixels == "kept":
        usable &= ~op.static_bad
    levels, _ = zero_levels(flat, usable, counting)
    asic_level = levels.astype(np.float32)

    no_level = ~np.isfinite(asic_level)
    uncounted = no_level[ASIC_ID] & ~bad
    n_level_missing = int(np.unique(ASIC_ID[uncounted]).size) if uncounted.any() else 0
    excluded = bad | uncounted

    photons = count_photons(flat, asic_level, counting.energy_f4)
    photons[excluded] = np.nan
    return PreparedFrame(
        photons=photons,
        excluded=excluded,
        asic_level=asic_level,
        n_level_missing=n_level_missing,
        n_extreme_pixels=n_extreme,
    )


def integrate_prepared(
    ai: object, op: WaxsOperator, prepared: PreparedFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(signal, normalization, variance)`` of a counted frame, the NaN path.

    The excluded pixels are already NaN in the counts, and the variance is the
    counts themselves, NaN in the same places, so pyFAI drops them from the
    numerator and the denominator exactly. Every call passes the same static
    mask, because a mask that changes per frame rebuilds pyFAI's sparse matrix.
    """
    signal = prepared.photons.copy()  # pyFAI's kernels need writable buffers
    variance = prepared.photons.copy()
    result = ai.integrate1d(  # type: ignore[attr-defined]
        signal.reshape(MODULE_SHAPE),
        op.npt,
        method=op.method,
        unit=op.unit,
        mask=op.static_mask_2d,
        variance=variance.reshape(MODULE_SHAPE),
    )
    return (
        np.asarray(result.sum_signal, dtype=np.float64),
        np.asarray(result.sum_normalization, dtype=np.float64),
        np.asarray(result.sum_variance, dtype=np.float64),
    )


def integrate_frame(
    ai: object,
    op: WaxsOperator,
    counting: PhotonCounting,
    x: np.ndarray,
    dynamic: np.ndarray,
    *,
    max_abs_kev: float,
    timings: dict[str, float] | None = None,
) -> FrameResult:
    """Count one frame's photons and integrate them densely through pyFAI.

    :param ai: the ``AzimuthalIntegrator`` built by
        :func:`analysis.waxs.operator.build_operator`.
    :param op: the operator whose q axis and static mask the engine was built on.
    :param counting: the photon-counting settings.
    :param x: one frame, ``(512, 1024)`` or flat, in keV.
    :param dynamic: this frame's ``data.mask`` exclusion, flat
        (:func:`prepare_frame`).
    :param max_abs_kev: value bound handed to :func:`extreme_pixels`.
    :param timings: if given, ``"photonise"`` and ``"integrate"`` seconds are
        added to it.
    :returns: the per-bin sums, the levels, maxima and pixel counts.
    :raises DataCheckFailed: every pixel is excluded, so there is nothing to
        integrate. Raised so that a frame of zeros is never stored as measured.
    """
    started = time.perf_counter()
    prepared = prepare_frame(op, counting, x, dynamic, max_abs_kev=max_abs_kev)
    counted = time.perf_counter()
    if prepared.excluded.all():
        raise DataCheckFailed(
            f"every one of the {prepared.excluded.size} pixels is excluded: "
            f"{prepared.n_extreme_pixels} of them for being non-finite or outside "
            f"±{max_abs_kev} keV, {prepared.n_level_missing} ASIC(s) for having "
            "no zero level, the rest by the static mask and data.mask. There is "
            "nothing left to integrate"
        )
    signal, normalization, variance = integrate_prepared(ai, op, prepared)
    if timings is not None:
        timings["photonise"] = timings.get("photonise", 0.0) + counted - started
        timings["integrate"] = (
            timings.get("integrate", 0.0) + time.perf_counter() - counted
        )

    flat = np.asarray(x).reshape(-1)
    kept = ~prepared.excluded
    occupied = normalization > 0
    max_kev, max_kev_static = frame_maxima(flat, prepared.excluded, op.static_bad)
    return FrameResult(
        signal=signal,
        normalization=normalization,
        variance=variance,
        energy_valid=float(flat[kept].sum()),
        n_photons=int(prepared.photons[kept].sum(dtype=np.float64)),
        n_bad_pixels=int(prepared.excluded.sum()),
        n_negative_variance_bins=int((occupied & (variance < 0)).sum()),
        max_kev=max_kev,
        max_kev_static=max_kev_static,
        n_extreme_pixels=prepared.n_extreme_pixels,
        asic_level=prepared.asic_level,
        n_level_missing=prepared.n_level_missing,
        count_energy_kev=float(counting.energy_f4),
    )
