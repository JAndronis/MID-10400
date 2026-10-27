"""A three-bin frame table over five trains, for the common-layer tests.

One train has no frames, so the blocks have a gap, and the trains differ in
length, so a row placed by position instead of by label lands on the wrong
train. Every value a test expects follows from the tables below by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.common.config import config_sha256
from analysis.common.plan import RunPlan, TrainRecord, build_blocks
from analysis.common.status import FrameStatus
from analysis.common.writer import FrameTableWriter

NPT = 3
OK = FrameStatus.OK

#: (trainId, frames). 10001 has none, so the blocks of two trains are
#: [10000], [10002, 10003], [10004]: no block straddles the gap.
TRAINS = ((10000, 3), (10001, 0), (10002, 2), (10003, 3), (10004, 1))
#: The trainId of every row, in row order.
ROW_TRAIN = [10000] * 3 + [10002] * 2 + [10003] * 3 + [10004]
#: The pulse each row's frame came from: its index within its train.
ROW_PULSE = [0, 1, 2, 0, 1, 0, 1, 2, 0]


@dataclass(frozen=True)
class TinyConfig:
    output_dir: str
    tag: str = "a"
    npt: int = NPT
    overwrite: bool = False

    @property
    def output_file(self) -> Path:
        return Path(self.output_dir) / "tiny.h5"

    @property
    def operational_fields(self) -> frozenset[str]:
        return frozenset({"output_dir", "overwrite"})

    def config_hash(self) -> str:
        return config_sha256(self, {}, self.operational_fields)


class TinyWriter(FrameTableWriter):
    FRAME_VECTORS = {"trainId": np.uint64, "status": np.uint8, "pulse": np.uint16}
    FRAME_VECTOR_SOURCES = {"status": "status", "pulse": "pulse"}


class WiderWriter(TinyWriter):
    FRAME_VECTORS = {**TinyWriter.FRAME_VECTORS, "extra": np.float32}
    FRAME_VECTOR_SOURCES = {**TinyWriter.FRAME_VECTOR_SOURCES, "extra": "pulse"}


def make_plan() -> RunPlan:
    records, row = [], 0
    for train_id, n in TRAINS:
        status = OK if n else FrameStatus.NO_FRAMES
        records.append(TrainRecord(train_id, n, row, status))
        row += n
    return RunPlan(
        trains=tuple(records),
        blocks=tuple(build_blocks(records, 2)),
        n_frames=row,
        detector_name="TINY",
    )


PLAN = make_plan()


def signal_of(row: int) -> np.ndarray:
    """A value per bin that names its row, so a misplaced row shows."""
    return np.array([row + 1.0, 10 * (row + 1.0), 100 * (row + 1.0)])


def result_for(block, *, status=None):
    """What a worker returns for ``block``: its labels and per-row sums."""
    rows = block.rows()
    n = rows.size
    return SimpleNamespace(
        n_frames=n,
        status=np.full(n, OK, np.uint8) if status is None else np.asarray(status),
        train_id=np.array([ROW_TRAIN[r] for r in rows], np.uint64),
        pulse=np.array([ROW_PULSE[r] for r in rows], np.uint16),
        signal=np.stack([signal_of(r) for r in rows]).astype(np.float32),
        normalization=np.full((n, NPT), 2.0, np.float32),
        variance=np.full((n, NPT), 4.0, np.float32),
    )
