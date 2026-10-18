"""``agipd_ring_photons.h5``: the hit trains' raw photons, kept after proc is taped.

One group per train, so the file is the union of whichever trains were extracted
and its content does not depend on the order or the number of workers that
wrote it. Everything in it is raw (counts and ``image.mask`` flags over the
envelope, no ring, no static mask, no kernel), which is why it is keyed on the
envelope alone.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from analysis.common.status import FrameStatus

__all__ = [
    "FlagBase",
    "TrainPhotons",
    "delta_decode",
    "delta_encode",
]


def _empty(dtype: type) -> np.ndarray:
    return np.zeros(0, dtype=dtype)


@dataclass(frozen=True, slots=True)
class FlagBase:
    """Each memory cell's usual set of flagged pixels, as envelope positions.

    The cells of a train share nearly all their bad pixels with the same cell of
    every other train (0.022 % vary between trains, CLAUDE.md), so a frame's
    flags are stored as their difference from this: about 100 pixels a frame
    instead of 37 000. Any base is lossless; a better one only makes the file
    smaller, which is why it is fixed once, when the store is created, and why it
    is not part of the store's identity.
    """

    cells: np.ndarray  # (n,) uint16, ascending
    ptr: np.ndarray  # (n + 1,) int64
    pixel: np.ndarray  # (nnz,) uint32, ascending within a cell

    @classmethod
    def empty(cls) -> FlagBase:
        """No base: every flagged pixel of every frame is listed."""
        return cls(_empty(np.uint16), np.zeros(1, dtype=np.int64), _empty(np.uint32))

    @classmethod
    def from_samples(
        cls, samples: Iterable[tuple[np.ndarray, list[np.ndarray]]]
    ) -> FlagBase:
        """The pixels flagged in more than half of a cell's sampled frames.

        :param samples: per sample train, its cell ids and each frame's flagged
            pixels. A cell never seen gets an empty base.
        """
        seen: dict[int, list[np.ndarray]] = {}
        for cells, flagged in samples:
            for cell, pixels in zip(cells.tolist(), flagged, strict=True):
                seen.setdefault(int(cell), []).append(pixels)
        ids = sorted(seen)
        parts = []
        for cell in ids:
            frames = seen[cell]
            pixels, votes = np.unique(np.concatenate(frames), return_counts=True)
            parts.append(pixels[votes * 2 > len(frames)].astype(np.uint32))
        return cls(
            np.asarray(ids, dtype=np.uint16),
            np.concatenate([[0], np.cumsum([p.size for p in parts])]).astype(np.int64),
            np.concatenate(parts) if parts else _empty(np.uint32),
        )

    def of_cell(self, cell: int) -> np.ndarray:
        index = int(np.searchsorted(self.cells, cell))
        if index == self.cells.size or self.cells[index] != cell:
            return _empty(np.uint32)
        return self.pixel[self.ptr[index] : self.ptr[index + 1]]

    def difference(self, cell: int, flagged: np.ndarray) -> np.ndarray:
        """A frame's flagged pixels as their symmetric difference from the base."""
        return np.setxor1d(flagged, self.of_cell(cell), assume_unique=True).astype(
            np.uint32
        )


@dataclass(frozen=True, slots=True)
class TrainPhotons:
    """One hit train: photons per frame and the pixels ``image.mask`` flagged.

    Pixel positions are indices into the envelope, not the detector. A frame's
    flagged set is its ``flag_pixel`` entries XOR the ``base`` of its memory
    cell. A train that is not ``OK`` carries no arrays.
    """

    train_id: int
    status: FrameStatus
    #: Proc sequence file the train sits in (the ``Kernel.ES`` chain).
    sequence: int
    pulse_ids: np.ndarray  # (P,) uint32
    cell_ids: np.ndarray  # (P,) uint16
    frame_ptr: np.ndarray  # (P + 1,) int64
    pixel: np.ndarray  # (nnz,) uint32, ascending within a frame
    count: np.ndarray  # (nnz,) uint8
    flag_ptr: np.ndarray  # (P + 1,) int64
    flag_pixel: np.ndarray  # (nf,) uint32, ascending within a frame
    base: FlagBase

    @property
    def n_frames(self) -> int:
        return int(self.pulse_ids.size)

    @classmethod
    def failed(cls, train_id: int, status: FrameStatus, sequence: int = -1):
        """A train that gave no photons, and why."""
        return cls(
            train_id,
            status,
            sequence,
            _empty(np.uint32),
            _empty(np.uint16),
            np.zeros(1, dtype=np.int64),
            _empty(np.uint32),
            _empty(np.uint8),
            np.zeros(1, dtype=np.int64),
            _empty(np.uint32),
            FlagBase.empty(),
        )

    def frames(self, n_env: int) -> sp.csr_matrix:
        """Photons as a ``(P, n_env)`` int32 sparse matrix."""
        return sp.csr_matrix(
            (
                self.count.astype(np.int32),
                self.pixel.astype(np.int32),
                self.frame_ptr.astype(np.int64),
            ),
            shape=(self.n_frames, n_env),
        )

    def flagged_frame(self, frame: int) -> np.ndarray:
        """One frame's flagged pixels, ascending."""
        listed = self.flag_pixel[self.flag_ptr[frame] : self.flag_ptr[frame + 1]]
        return np.setxor1d(
            listed, self.base.of_cell(int(self.cell_ids[frame])), assume_unique=True
        )

    def flagged_any(self, n_env: int) -> np.ndarray:
        """Pixels flagged in at least one frame, as a bool array over the envelope."""
        flagged = np.zeros(n_env, dtype=bool)
        for frame in range(self.n_frames):
            flagged[self.flagged_frame(frame)] = True
        return flagged


def delta_encode(pixel: np.ndarray, frame_ptr: np.ndarray) -> np.ndarray:
    """Positions as steps from the previous event, restarting at each frame.

    The first event of a frame keeps its absolute position. Gzip then finds far
    more to remove in the steps than in the positions (1.6 against 2.2 MB per
    train on r0423, P0 of ``context/xpcs-hits.md``).
    """
    step = np.diff(pixel.astype(np.int64), prepend=0)
    starts = frame_ptr[:-1]
    starts = starts[starts < pixel.size]
    step[starts] = pixel[starts]
    return step.astype(np.uint32)


def delta_decode(step: np.ndarray, frame_ptr: np.ndarray) -> np.ndarray:
    """Inverse of :func:`delta_encode`."""
    if step.size == 0:
        return step.astype(np.uint32)
    total = np.cumsum(step.astype(np.int64))
    counts = np.diff(frame_ptr)
    # what the running sum held when each frame began, taken off that frame
    before = np.concatenate([[0], total])[frame_ptr[:-1]]
    return (total - np.repeat(before, counts)).astype(np.uint32)
