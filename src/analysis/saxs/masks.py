"""Static and per-cell base masks.

Three things are combined here:

1. **The static mask.** ASIC seams OR'd with the single hand-maintained pixel
   mask, which covers both the generally bad pixels and the anisotropic low-q
   lobe (integrator I4, option (a)). There is deliberately only one such file:
   two overlapping masks would have to be kept in step with each other.
   ``image.mask`` never sets ``NON_STANDARD_SIZE``, so the double-width
   ASIC-edge pixels are unflagged in the data and must come from
   ``agipd_asic_seams()``.
2. **The per-cell base mask.** The static bits in ``image.mask`` are per memory
   cell (4.05 % always flagged, 0.022 % varying), so a majority vote over a few
   sampled trains gives each cell a base mask and a precomputed denominator
   ``D = Σ c·Ω``. Because every statically-bad pixel votes on every sample, the
   base mask is always a superset of the static mask; this is asserted, not
   assumed.
3. **The per-frame correction.** ``sparse.integrate_frame`` corrects ``D`` by
   only the pixels where a frame disagrees with its cell, in either direction,
   so the result is exact rather than approximate.

This module does not import EXtra-data. It consumes mask samples handed to it
by the caller — the train selection and the ``image.mask`` read belong to the
plan and the worker — which also keeps it testable without data.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from extra_geom import agipd_asic_seams

# Re-exported: the mask vocabulary and the train sampling are detector-agnostic
# and now live under ``analysis.common``, but the SAXS package, its tests and
# ``scripts/p4_acceptance.py`` reach them through this module.
from analysis.common.masks import (
    MaskSource,
    StaticMask,
    UnexpectedMaskBits,
    bits_to_mask,
    describe_bits,
    frame_bad,
)
from analysis.common.plan import evenly_spaced
from analysis.saxs.config import NPIX, AgipdSaxsConfig, file_sha256
from analysis.saxs.operator import SparseOperator
from analysis.saxs.sparse import denominator

__all__ = [
    "BaseMaskAccumulator",
    "BaseMasks",
    "MaskSource",
    "StaticMask",
    "UnexpectedMaskBits",
    "bits_to_mask",
    "build_static_bad",
    "describe_bits",
    "evenly_spaced",
    "frame_bad",
    "load_masks",
    "load_pixel_mask",
    "save_masks",
]

MODULE_SHAPE = (512, 128)
STACKED_SHAPE = (16, 512, 128)
#: Shapes a stored pixel mask may have. ``(8192, 128)`` is the flattened pixel
#: order module × ss × fs, which is how the earlier prototype stored it.
ACCEPTED_MASK_SHAPES = (STACKED_SHAPE, MODULE_SHAPE, (16 * 512, 128))


# ── static mask ───────────────────────────────────────────────────────────────
def load_pixel_mask(path: str | Path) -> np.ndarray:
    """Load a stored pixel mask as a flat boolean array, non-zero = excluded.

    Accepts ``(16, 512, 128)``, ``(512, 128)`` (one module, broadcast over all
    sixteen) and ``(8192, 128)``.

    :raises ValueError: unaccepted shape, or a mask that excludes every pixel —
        which would leave nothing to integrate and is always a mistake.
    :raises TypeError: a non-boolean, non-integer dtype, which would make the
        non-zero convention ambiguous.
    """
    array = np.load(path)
    if array.shape not in ACCEPTED_MASK_SHAPES:
        raise ValueError(
            f"mask {path} has shape {array.shape}; expected one of "
            f"{ACCEPTED_MASK_SHAPES}"
        )
    if array.dtype != np.bool_ and not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"mask {path} has dtype {array.dtype}; expected bool or an integer "
            "type, so that 'non-zero = excluded' is unambiguous"
        )
    if array.shape == MODULE_SHAPE:
        array = np.broadcast_to(array, STACKED_SHAPE)
    bad = array.ravel() != 0
    if bad.all():
        raise ValueError(f"mask {path} excludes every pixel")
    return bad


def build_static_bad(cfg: AgipdSaxsConfig) -> StaticMask:
    """ASIC seams ∪ the pixel mask."""
    bad = np.zeros(NPIX, dtype=bool)
    sources: list[MaskSource] = []

    if cfg.use_asic_seams:
        seams = np.broadcast_to(agipd_asic_seams(), STACKED_SHAPE).ravel()
        sources.append(MaskSource("asic_seams", None, None, int(seams.sum())))
        bad |= seams

    if cfg.pixel_mask_file is not None:
        path = cfg.pixel_mask_file
        contribution = load_pixel_mask(path)
        sources.append(
            MaskSource(
                "pixel_mask", str(path), file_sha256(path), int(contribution.sum())
            )
        )
        bad |= contribution

    bad.flags.writeable = False
    return StaticMask(
        bad=bad,
        sha256=hashlib.sha256(np.packbits(bad).tobytes()).hexdigest(),
        sources=tuple(sources),
    )


# ── per-cell base masks ───────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BaseMasks:
    """Per-memory-cell base masks and their precomputed denominators."""

    cells: np.ndarray  # uint16 (n_cells,), sorted
    base_bad: np.ndarray  # bool    (n_cells, NPIX)
    denominators: np.ndarray  # float64 (n_cells, npt)
    n_samples: np.ndarray  # uint32  (n_cells,)
    static_bad: np.ndarray  # bool    (NPIX,)
    static_denominator: np.ndarray  # float64 (npt,)
    static_sha256: str
    bits_present: int
    unexpected_bits: int
    operator_sha256: str
    sha256: str

    def for_cell(self, cell: int) -> tuple[np.ndarray, np.ndarray]:
        """``(base_bad, denominator)`` for ``cell``.

        A cell absent from the sampled trains falls back to the static mask and
        is counted in provenance. The per-frame correction stays exact either
        way; the fallback only corrects more pixels per frame.
        """
        index = int(np.searchsorted(self.cells, cell))
        if index < self.cells.size and int(self.cells[index]) == cell:
            return self.base_bad[index], self.denominators[index]
        return self.static_bad, self.static_denominator

    def has_cell(self, cell: int) -> bool:
        index = int(np.searchsorted(self.cells, cell))
        return index < self.cells.size and int(self.cells[index]) == cell


class BaseMaskAccumulator:
    """Majority-vote accumulator over sampled trains.

    The caller feeds one train at a time; ``finalise`` turns the votes into base
    masks and denominators. Votes are uint8, so a cell may be sampled at most
    255 times — far above the ``base_mask_trains`` default of 8.
    """

    def __init__(
        self,
        op: SparseOperator,
        static: StaticMask,
        *,
        mask_bits: int,
        expected_bits: Iterable[int],
    ) -> None:
        self._op = op
        self._static = static
        self._mask_bits = int(mask_bits)
        self._expected = bits_to_mask(expected_bits)
        self._votes: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}
        self._bits_present = 0
        self.n_trains = 0

    def update(self, mask: np.ndarray, cell_ids: np.ndarray) -> None:
        """Accumulate one train of ``image.mask``.

        :param mask: ``(16, n_frames, 512, 128)`` uint32 ``BadPixels`` field.
        :param cell_ids: ``(n_frames,)`` memory-cell id per frame, from
            ``cell_id_coordinates()`` — never inferred from array position.
        """
        mask = np.asarray(mask)
        cell_ids = np.asarray(cell_ids)
        if mask.ndim != 4 or mask.shape[0] != 16 or mask.shape[2:] != MODULE_SHAPE:
            raise ValueError(
                f"mask must have shape (16, n_frames, 512, 128), got {mask.shape}"
            )
        if mask.dtype != np.uint32:
            raise TypeError(f"image.mask must be uint32, got {mask.dtype}")
        if cell_ids.shape != (mask.shape[1],):
            raise ValueError(
                f"cell_ids has shape {cell_ids.shape}, expected ({mask.shape[1]},)"
            )

        self._bits_present |= int(np.bitwise_or.reduce(mask, axis=None))

        for frame, cell in enumerate(cell_ids.tolist()):
            votes = self._votes.get(cell)
            if votes is None:
                votes = self._votes[cell] = np.zeros(NPIX, dtype=np.uint8)
                self._counts[cell] = 0
            if self._counts[cell] == np.iinfo(np.uint8).max:
                raise ValueError(
                    f"cell {cell} sampled more than {np.iinfo(np.uint8).max} times; "
                    "the uint8 vote counter would overflow"
                )
            bad = frame_bad(mask[:, frame], self._mask_bits, self._static.bad)
            votes += bad.view(np.uint8)
            self._counts[cell] += 1

        self.n_trains += 1

    def finalise(self) -> BaseMasks:
        """Majority-vote the samples into base masks and denominators."""
        if not self._votes:
            raise ValueError("no frames accumulated; cannot build base masks")

        cells = np.array(sorted(self._votes), dtype=np.uint16)
        base_bad = np.zeros((cells.size, NPIX), dtype=bool)
        denominators = np.zeros((cells.size, self._op.npt), dtype=np.float64)
        n_samples = np.zeros(cells.size, dtype=np.uint32)

        for index, cell in enumerate(cells.tolist()):
            count = self._counts[cell]
            # Strict majority. Written as ``votes > count // 2`` rather than
            # ``votes * 2 > count`` is identical for every integer count,
            # and it cannot overflow the uint8 vote array.
            voted = self._votes[cell] > count // 2
            if (self._static.bad & ~voted).any():
                raise AssertionError(
                    f"base mask for cell {cell} is not a superset of the static "
                    "mask; the votes did not include static_bad"
                )
            base_bad[index] = voted
            denominators[index] = denominator(self._op, voted)
            n_samples[index] = count

        unexpected = self._bits_present & ~self._expected
        if unexpected:
            warnings.warn(
                f"image.mask carries unexpected BadPixels bits: "
                f"{describe_bits(unexpected)}. A blanket mask_bits is only "
                "correct while every bit present marks an unusable pixel — "
                "check these before trusting the run.",
                UnexpectedMaskBits,
                stacklevel=2,
            )

        base_bad.flags.writeable = False
        denominators.flags.writeable = False
        static_denominator = denominator(self._op, self._static.bad)
        static_denominator.flags.writeable = False

        return BaseMasks(
            cells=cells,
            base_bad=base_bad,
            denominators=denominators,
            n_samples=n_samples,
            static_bad=self._static.bad,
            static_denominator=static_denominator,
            static_sha256=self._static.sha256,
            bits_present=self._bits_present,
            unexpected_bits=unexpected,
            operator_sha256=self._op.sha256,
            sha256=_base_masks_sha256(
                cells, base_bad, n_samples, self._static.sha256, self._op.sha256
            ),
        )


def _base_masks_sha256(
    cells: np.ndarray,
    base_bad: np.ndarray,
    n_samples: np.ndarray,
    static_sha256: str,
    operator_sha256: str,
) -> str:
    digest = hashlib.sha256(f"{static_sha256}:{operator_sha256}".encode())
    digest.update(np.ascontiguousarray(cells).tobytes())
    digest.update(np.ascontiguousarray(n_samples).tobytes())
    digest.update(np.packbits(base_bad, axis=-1).tobytes())
    return digest.hexdigest()


# ── persistence ───────────────────────────────────────────────────────────────
def save_masks(bm: BaseMasks, path: str | Path) -> Path:
    """Write base masks to ``path`` as an ``.npz``, boolean arrays packed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        cells=bm.cells,
        base_bad_packed=np.packbits(bm.base_bad, axis=-1),
        denominators=bm.denominators,
        n_samples=bm.n_samples,
        static_bad_packed=np.packbits(bm.static_bad),
        static_denominator=bm.static_denominator,
        scalars=np.array(
            [
                bm.static_sha256,
                str(bm.bits_present),
                str(bm.unexpected_bits),
                bm.operator_sha256,
                bm.sha256,
            ]
        ),
    )
    return path


def load_masks(path: str | Path) -> BaseMasks:
    """Load base masks written by :func:`save_masks`, verifying the hash."""
    with np.load(path) as handle:
        cells = handle["cells"]
        base_bad = np.unpackbits(handle["base_bad_packed"], axis=-1, count=NPIX).astype(
            bool
        )
        denominators = handle["denominators"]
        n_samples = handle["n_samples"]
        static_bad = np.unpackbits(handle["static_bad_packed"], count=NPIX).astype(bool)
        static_denominator = handle["static_denominator"]
        scalars = [str(value) for value in handle["scalars"]]

    static_sha256, bits_present, unexpected_bits, operator_sha256, sha256 = scalars
    for array in (base_bad, denominators, static_bad, static_denominator):
        array.flags.writeable = False

    recomputed = _base_masks_sha256(
        cells, base_bad, n_samples, static_sha256, operator_sha256
    )
    if recomputed != sha256:
        raise ValueError(
            f"base masks at {path} are corrupt: stored sha256 {sha256}, "
            f"recomputed {recomputed}"
        )
    return BaseMasks(
        cells=cells,
        base_bad=base_bad,
        denominators=denominators,
        n_samples=n_samples,
        static_bad=static_bad,
        static_denominator=static_denominator,
        static_sha256=static_sha256,
        bits_present=int(bits_present),
        unexpected_bits=int(unexpected_bits),
        operator_sha256=operator_sha256,
        sha256=sha256,
    )
