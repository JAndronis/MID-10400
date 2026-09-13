"""The error model and the pure per-frame integration (WAXS context file §3).

Kept free of I/O and of EXtra-data so the self-test and the unit tests can
reach it without a run, the way ``analysis.saxs.sparse`` is for AGIPD.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from analysis.common.status import DataCheckFailed, FrameStatus
from analysis.waxs.config import MODULE_SHAPE
from analysis.waxs.operator import WaxsOperator

__all__ = [
    "ErrorModel",
    "FrameResult",
    "frame_data_status",
    "integrate_frame",
]


@dataclass(frozen=True, slots=True)
class ErrorModel:
    """``Var(x) = σ_read² + E·x``, in keV² (context file §3 D3).

    ``Σc²·x`` is the Poisson variance of *integer photon counts*; on keV-valued
    data with a negative tail it is not a variance at all. For a pixel holding
    ``x`` keV deposited by ``E``-keV photons the count is ``x/E``, whose Poisson
    variance is ``x/E``, which in keV² is ``E·x``; the readout term adds
    ``σ_read²``.

    **The result is deliberately not clamped.** On r0423 it is negative for
    23–26 % of kept pixels, and integrating leaves 5 of 4000 occupied per-frame
    bins with a non-positive ``sum_variance`` on jf1. Clamping at zero would
    bias the noise floor upward on every negative-noise pixel, and that bias
    does *not* cancel; the unbiased form's negatives do, once a bin pools many
    frames. So the sums are stored as they come and the per-frame negative bins
    are counted into the ledger, where they stay visible.
    """

    read_noise_kev: float
    photon_energy_kev: float
    #: ``"measured"`` from the run's dark cells, or ``"configured"``.
    source: str = "configured"
    #: Frames the measurement averaged over; 0 when configured.
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
    max_kev: float


def frame_data_status(
    x: np.ndarray, static_bad: np.ndarray, max_abs_kev: float
) -> FrameStatus:
    """Classify a frame's values as ``OK`` or ``DATA_CHECK_FAILED`` (§3 D6).

    Two checks, both on the pixels the static mask keeps — a wild value under
    the static mask never reaches the integrator, so it is not this frame's
    problem:

    * **Non-finite.** The dynamic mask is carried as NaN, so a NaN already in
      the data would be indistinguishable from a masked pixel.
    * **Out of range.** jf2 carries 15 pixels reaching ±1.8e5 keV — some 20 000
      photons where the lit-cell mean is 5.3 — in *every* memory cell, and
      ``data.mask`` flags none of them. The ``.edf`` happens to cover all of
      them, but relying on that silently is the failure mode the ledger exists
      to prevent.

    The AGIPD check — integer dtype, no negative counts — is wrong here in both
    halves, so this replaces it rather than adding to it.
    """
    kept = x.reshape(-1)[~static_bad]
    if not np.isfinite(kept).all():
        return FrameStatus.DATA_CHECK_FAILED
    if kept.size and np.abs(kept).max() > max_abs_kev:
        return FrameStatus.DATA_CHECK_FAILED
    return FrameStatus.OK


def integrate_frame(
    ai: object,
    op: WaxsOperator,
    model: ErrorModel,
    x: np.ndarray,
    bad: np.ndarray,
    *,
    max_abs_kev: float,
) -> FrameResult:
    """Integrate one frame densely through pyFAI (§3 D1).

    :param ai: the ``AzimuthalIntegrator`` whose engine was built by
        :func:`analysis.waxs.operator.build_operator` — that is, with ``mask``
        already set to ``op.static_bad``. Passing any other mask here would
        rebuild the sparse matrix; see that module's docstring.
    :param x: one frame, ``(512, 1024)`` or flat, in keV.
    :param bad: this frame's bad-pixel mask, ``((m & mask_bits) != 0) |
        static_bad``, flat.

    ``bad`` is carried into pyFAI as NaN in both the data and the variance,
    which drops those pixels from the numerator and the denominator exactly.
    The static half of ``bad`` is already excluded by the engine's own mask, so
    NaN-ing it again is a no-op; it is included so that ``n_bad_pixels`` counts
    the union a reader would expect.

    :raises DataCheckFailed: the frame is non-finite or out of range on a pixel
        the static mask keeps. Raised rather than returned: there is no
        fallback, so a failing frame must never yield arrays.
    """
    flat = np.asarray(x).reshape(-1)
    if flat.size != op.static_bad.size:
        raise ValueError(
            f"frame has {flat.size} pixels, the operator has {op.static_bad.size}"
        )
    if bad.shape != op.static_bad.shape:
        raise ValueError(f"bad has shape {bad.shape}, expected {op.static_bad.shape}")
    if frame_data_status(flat, op.static_bad, max_abs_kev) is not FrameStatus.OK:
        kept = flat[~op.static_bad]
        finite = kept[np.isfinite(kept)]
        raise DataCheckFailed(
            "frame fails the value check on pixels the static mask keeps: "
            f"{int((~np.isfinite(kept)).sum())} non-finite, max |x| "
            f"{float(np.abs(finite).max(initial=0.0)):.1f} keV against a "
            f"bound of {max_abs_kev} keV"
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
    return FrameResult(
        signal=sum_signal,
        normalization=sum_normalization,
        variance=sum_variance,
        energy_valid=float(kept_values.sum()),
        n_bad_pixels=int(bad.sum()),
        n_negative_variance_bins=int((occupied & (sum_variance < 0)).sum()),
        max_kev=float(kept_values.max(initial=0.0)),
    )
