"""Read hit trains from proc into :class:`TrainPhotons`.

``encode_train`` is the pure array step; the worker around it (added with the
store) is the read-and-loop wrapper, as in ``analysis.saxs.worker``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common.cpu import set_thread_env
from analysis.common.plan import Block
from analysis.common.status import DataCheckFailed, FrameStatus
from analysis.xpcs.config import XpcsConfig
from analysis.xpcs.store import FlagBase, TrainPhotons

__all__ = [
    "MAX_COUNT",
    "ExtractResult",
    "encode_train",
    "init",
    "init_from_detector",
    "open_detector",
    "process_block",
    "sample_flag_base",
    "train_sequences",
]

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


def open_detector(cfg: XpcsConfig, dc: Any = None, run_dir: str | Path | None = None):
    """The AGIPD detector of the proc run, all 16 modules required."""
    from extra_data import RunDirectory, open_run
    from extra_data.components import AGIPD1M

    if dc is None:
        dc = (
            RunDirectory(str(run_dir))
            if run_dir
            else open_run(cfg.proposal, cfg.run, data="proc")
        )
    return AGIPD1M(dc, detector_name=cfg.saxs.detector_name, min_modules=16)


def sample_flag_base(
    cfg: XpcsConfig, detector: Any, train_ids: np.ndarray, envelope: np.ndarray
) -> FlagBase:
    """The flag base from the masks of ``train_ids`` (their data are not read)."""
    from extra_data import by_id

    bits = np.uint32(cfg.saxs.mask_bits)
    samples = []
    for train_id in train_ids.tolist():
        key = detector.select_trains(by_id[[train_id]])["image.mask"]
        mask = key.ndarray(decompress_threads=1)
        flagged = [
            np.flatnonzero((mask[:, frame].reshape(-1)[envelope] & bits) != 0)
            for frame in range(mask.shape[1])
        ]
        samples.append((key.cell_id_coordinates(), flagged))
    return FlagBase.from_samples(samples)


@dataclass(slots=True)
class ExtractResult:
    """One block's trains, each carrying its own identity."""

    block_index: int
    photons: list[TrainPhotons]
    bits_present: int = 0
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _State:
    cfg: XpcsConfig
    envelope: np.ndarray
    base: FlagBase
    detector: Any


_STATE: _State | None = None


def init(
    cfg: XpcsConfig,
    envelope: np.ndarray,
    base: FlagBase,
    run_dir: str | None = None,
) -> None:
    """Process initializer: open the run."""
    set_thread_env()
    init_from_detector(cfg, envelope, base, open_detector(cfg, run_dir=run_dir))


def init_from_detector(
    cfg: XpcsConfig, envelope: np.ndarray, base: FlagBase, detector: Any
) -> None:
    """In-process initializer for tests and single-process runs."""
    global _STATE
    _STATE = _State(cfg, envelope, base, detector)


def process_block(block: Block) -> ExtractResult:
    """Read and encode one block's trains.

    A train whose labels or frame count disagree with the plan is
    ``LABEL_MISMATCH`` and one whose counts are not photons ``DATA_CHECK_FAILED``;
    both are final and carry no photons. Anything else raises, so the block is
    recorded as ``WORKER_ERROR`` and retried.
    """
    if _STATE is None:
        raise RuntimeError("extract.init has not run in this process")
    from extra_data import by_id

    state = _STATE
    timings = {"read_data": 0.0, "read_mask": 0.0, "encode": 0.0}
    photons: list[TrainPhotons] = []
    bits_present = 0
    for train_id, expected in zip(block.train_ids, block.expected_frames, strict=True):
        selected = state.detector.select_trains(by_id[[train_id]])
        key = selected["image.data"]
        started = time.perf_counter()
        data = key.ndarray(decompress_threads=1)
        timings["read_data"] += time.perf_counter() - started
        started = time.perf_counter()
        mask = selected["image.mask"].ndarray(decompress_threads=1)
        timings["read_mask"] += time.perf_counter() - started

        sequences = train_sequences(key)
        sequence = min(sequences.values()) if sequences else -1
        train_ids = key.train_id_coordinates()
        if (
            train_ids.size != expected
            or data.shape[1] != expected
            or mask.shape[1] != expected
            or not np.all(train_ids == train_id)
        ):
            photons.append(
                TrainPhotons.failed(train_id, FrameStatus.LABEL_MISMATCH, sequence)
            )
            continue
        bits_present |= int(np.bitwise_or.reduce(mask, axis=None))
        started = time.perf_counter()
        try:
            photons.append(
                encode_train(
                    data,
                    mask,
                    state.envelope,
                    state.cfg.saxs.mask_bits,
                    state.base,
                    train_id=train_id,
                    sequence=sequence,
                    pulse_ids=key.pulse_id_coordinates(),
                    cell_ids=key.cell_id_coordinates(),
                )
            )
        except DataCheckFailed:
            photons.append(
                TrainPhotons.failed(train_id, FrameStatus.DATA_CHECK_FAILED, sequence)
            )
        timings["encode"] += time.perf_counter() - started
    return ExtractResult(block.index, photons, bits_present, timings)
