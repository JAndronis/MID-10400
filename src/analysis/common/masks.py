"""Mask vocabulary shared by both passes.

A *static* mask excludes the same pixels for every frame of a run; a *dynamic*
one comes from the correction pipeline's per-frame bad-pixel bitfield. What
goes into each is detector business — AGIPD needs ASIC seams because
``NON_STANDARD_SIZE`` is never set in its files, JUNGFRAU does not because it
is — but the provenance record, the bit vocabulary and the per-frame union are
the same in both.

Every mask here follows one convention: **non-zero means excluded**, which is
also pyFAI's own (``integrate1d_ng``: "array with 0 for valid pixels, all other
are masked"). Nothing in either pass inverts a mask.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MaskSource",
    "StaticMask",
    "UnexpectedMaskBits",
    "bits_to_mask",
    "describe_bits",
    "frame_bad",
]


class UnexpectedMaskBits(UserWarning):
    """A ``BadPixels`` bit outside the configured expected set was present."""


def bits_to_mask(bits: Iterable[int]) -> int:
    """Turn bit *positions* into a uint32 bitmask."""
    value = 0
    for bit in bits:
        if not 0 <= bit < 32:
            raise ValueError(f"bit position out of range for uint32: {bit}")
        value |= 1 << bit
    return value


def describe_bits(value: int) -> str:
    """Name the set bits of a ``BadPixels`` field, for warnings and provenance.

    The bit *set* is recorded exactly by the caller; only this human-readable
    rendering is best-effort, so a missing ``euxfel-EXtra`` degrades to bare bit
    numbers instead of failing. The import is deferred because ``extra`` is a
    large package and workers have no use for it.
    """
    positions = [bit for bit in range(32) if value & (1 << bit)]
    try:
        from extra.calibration import BadPixels
    except ImportError:
        return ", ".join(f"bit {bit}" for bit in positions) or "none"
    names = {member.value.bit_length() - 1: member.name for member in BadPixels}
    return (
        ", ".join(f"{bit} {names.get(bit, 'UNKNOWN')}" for bit in positions) or "none"
    )


@dataclass(frozen=True, slots=True)
class MaskSource:
    """One contribution to a static mask, recorded for provenance.

    ``n_excluded`` is this source's own count, before the OR with the others,
    so each source's contribution stays visible in the provenance record.
    """

    name: str
    path: str | None
    sha256: str | None
    n_excluded: int


@dataclass(frozen=True, slots=True)
class StaticMask:
    """Pixels excluded for every frame of the run."""

    bad: np.ndarray  # bool (npix,), read-only
    sha256: str
    sources: tuple[MaskSource, ...]

    @property
    def n_excluded(self) -> int:
        return int(self.bad.sum())


def frame_bad(
    mask_frame: np.ndarray, mask_bits: int, static_bad: np.ndarray
) -> np.ndarray:
    """This frame's bad-pixel mask: ``((m & mask_bits) != 0) | static_bad``.

    ``mask_frame`` is one frame of the correction pipeline's mask, in any shape
    that flattens to the static mask's length. A blanket ``mask_bits`` is only
    correct while every bit present marks an unusable pixel (CLAUDE.md pitfall
    6), which is why both passes record the bit set actually seen.
    """
    flat = mask_frame.reshape(-1)
    if flat.size != static_bad.size:
        raise ValueError(
            f"mask frame has {flat.size} pixels, static mask has {static_bad.size}"
        )
    return ((flat & np.uint32(mask_bits)) != 0) | static_bad
