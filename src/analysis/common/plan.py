"""The row model: trains, frame counts, row offsets and blocks.

A run plan is its identity table. Every row of an output file is addressed by
the train it belongs to, never by its position in an array: a dropped train
would otherwise shift every later frame onto the wrong trainId (CLAUDE.md
pitfall 4).

Nothing here knows which detector it is describing. Each pass builds its own
:class:`RunPlan` — how many rows a train owns, and which run checks apply, are
detector questions — but the records, the blocks and the train sampling are
the same either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from analysis.common.status import FrameStatus

__all__ = ["Block", "RunPlan", "TrainRecord", "build_blocks", "evenly_spaced"]


@dataclass(frozen=True, slots=True)
class TrainRecord:
    """One train's place in the flat frame table."""

    train_id: int
    n_frames: int
    first_row: int
    status: FrameStatus


@dataclass(frozen=True, slots=True)
class Block:
    """A unit of scheduling, ledger and resume — not a unit of memory."""

    index: int
    train_ids: tuple[int, ...]
    expected_frames: tuple[int, ...]
    first_rows: tuple[int, ...]

    @property
    def n_frames(self) -> int:
        return sum(self.expected_frames)

    def rows(self) -> np.ndarray:
        """Every output row this block owns, in train order."""
        return np.concatenate(
            [
                np.arange(first, first + count, dtype=np.int64)
                for first, count in zip(
                    self.first_rows, self.expected_frames, strict=True
                )
            ]
        )

    def frames_for(self, train_id: int) -> tuple[int, int]:
        """``(first_row, n_frames)`` for one train of this block."""
        index = self.train_ids.index(train_id)
        return self.first_rows[index], self.expected_frames[index]


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Everything the workers and the writer need to address rows by label."""

    trains: tuple[TrainRecord, ...]
    blocks: tuple[Block, ...]
    n_frames: int
    detector_name: str
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def train_ids(self) -> np.ndarray:
        return np.array([t.train_id for t in self.trains], dtype=np.uint64)

    def record(self, train_id: int) -> TrainRecord:
        for train in self.trains:
            if train.train_id == train_id:
                return train
        raise KeyError(f"train {train_id} is not in the plan")

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for train in self.trains:
            counts[train.status.name] = counts.get(train.status.name, 0) + 1
        return counts

    def ok_train_ids(self) -> np.ndarray:
        """The trains that own rows, as a uint64 array."""
        return np.array(
            [t.train_id for t in self.trains if t.status is FrameStatus.OK],
            dtype=np.uint64,
        )


def build_blocks(
    records: list[TrainRecord] | tuple[TrainRecord, ...], trains_per_block: int
) -> list[Block]:
    """Group *consecutive* OK trains into blocks of at most ``trains_per_block``.

    A non-OK train ends the current run of consecutive trains, so a block never
    straddles a gap and a block's rows are always contiguous.
    """
    blocks: list[Block] = []
    current: list[TrainRecord] = []
    runs: list[list[TrainRecord]] = []
    for record in records:
        if record.status is FrameStatus.OK:
            current.append(record)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    for consecutive in runs:
        for start in range(0, len(consecutive), trains_per_block):
            chunk = consecutive[start : start + trains_per_block]
            blocks.append(
                Block(
                    index=len(blocks),
                    train_ids=tuple(t.train_id for t in chunk),
                    expected_frames=tuple(t.n_frames for t in chunk),
                    first_rows=tuple(t.first_row for t in chunk),
                )
            )
    return blocks


def evenly_spaced(train_ids: np.ndarray, n: int) -> np.ndarray:
    """Pick up to ``n`` train ids spread evenly over ``train_ids``.

    Endpoints included. Returns fewer than ``n`` only when the run has fewer
    trains than that.
    """
    train_ids = np.asarray(train_ids)
    if train_ids.ndim != 1:
        raise ValueError(f"train_ids must be 1-D, got shape {train_ids.shape}")
    if train_ids.size == 0:
        raise ValueError("no trains to sample from")
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    positions = np.linspace(0, train_ids.size - 1, min(n, train_ids.size))
    return train_ids[np.unique(np.rint(positions).astype(np.int64))]
