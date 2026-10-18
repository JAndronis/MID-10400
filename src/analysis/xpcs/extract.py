"""Read hit trains from proc into :class:`TrainPhotons`.

``encode_train`` is the pure array step; the worker around it (added with the
store) is the read-and-loop wrapper, as in ``analysis.saxs.worker``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from analysis.common.status import DataCheckFailed, FrameStatus
from analysis.xpcs.store import FlagBase, TrainPhotons

__all__ = ["MAX_COUNT", "encode_train", "train_sequences"]

#: Photons per pixel the store can hold. The probe saw no count above 12 in any
#: ring (r0423, r0450, 2026-10-13).
MAX_COUNT = 255


_SEQUENCE = re.compile(r"-S(\d+)\.h5$")


def train_sequences(key: Any) -> dict[int, int]:
    """Module number to the proc sequence file number holding the selected train.

    :param key: the ``image.data`` key of a detector selected to one train.
    """
    return {
        module: int(_SEQUENCE.search(str(paths[0]))[1])
        for module, keydata in key.modno_to_keydata.items()
        if (paths := keydata.source_file_paths)
    }


def encode_train(
    data: np.ndarray,
    mask: np.ndarray,
    envelope: np.ndarray,
    mask_bits: int,
    base: FlagBase,
    *,
    train_id: int,
    sequence: int,
    pulse_ids: np.ndarray,
    cell_ids: np.ndarray,
) -> TrainPhotons:
    """Photons and flags of one train over the envelope.

    :param data: ``(16, P, 512, 128)`` integer photon counts.
    :param mask: the same shape, ``image.mask``.
    :param envelope: flat detector indices of the stored pixels.
    :param mask_bits: the ``BadPixels`` bits that mark a pixel unusable.
    :param base: what each cell's flags are differenced against.
    :raises DataCheckFailed: counts that are not integers, are negative, or
        exceed :data:`MAX_COUNT`, in the envelope.
    """
    if not np.issubdtype(data.dtype, np.integer):
        raise DataCheckFailed(f"train {train_id}: data are {data.dtype}, not photons")
    n_frames = data.shape[1]
    bits = np.uint32(mask_bits)
    pixels, counts, differences = [], [], []
    for frame in range(n_frames):
        values = data[:, frame].reshape(-1)[envelope]
        hit = np.flatnonzero(values)
        if hit.size:
            seen = values[hit]
            if seen.min() < 0 or seen.max() > MAX_COUNT:
                raise DataCheckFailed(
                    f"train {train_id} frame {frame}: counts {seen.min()} … "
                    f"{seen.max()} are outside 0 … {MAX_COUNT}"
                )
            counts.append(seen.astype(np.uint8))
        else:
            counts.append(np.zeros(0, dtype=np.uint8))
        pixels.append(hit.astype(np.uint32))
        flagged = np.flatnonzero((mask[:, frame].reshape(-1)[envelope] & bits) != 0)
        differences.append(base.difference(int(cell_ids[frame]), flagged))
    return TrainPhotons(
        train_id=train_id,
        status=FrameStatus.OK,
        sequence=sequence,
        pulse_ids=np.asarray(pulse_ids, dtype=np.uint32),
        cell_ids=np.asarray(cell_ids, dtype=np.uint16),
        frame_ptr=_pointers(pixels),
        pixel=np.concatenate(pixels) if pixels else np.zeros(0, dtype=np.uint32),
        count=np.concatenate(counts) if counts else np.zeros(0, dtype=np.uint8),
        flag_ptr=_pointers(differences),
        flag_pixel=(
            np.concatenate(differences) if differences else np.zeros(0, dtype=np.uint32)
        ),
        base=base,
    )


def _pointers(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum([part.size for part in parts])]).astype(
        np.int64
    )
