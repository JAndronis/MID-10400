"""Trains, frame counts, row offsets, blocks and run checks (context file §6.8).

The plan is the run's identity table. Every row of the output file is addressed
by the train it belongs to, never by its position in an array: a dropped train
would otherwise shift every later frame onto the wrong trainId (CLAUDE.md
pitfall 4).

``build_plan`` takes an optional ``DataCollection`` so callers can supply a run
that is already open — or a mock one — instead of going through ``open_run``.
Nothing here imports DAMNIT, so this module can back the pyBeamtime EuXFEL
reader later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from analysis.saxs.config import FirstPassConfig
from analysis.saxs.status import FrameStatus

__all__ = ["Block", "RunPlan", "TrainRecord", "build_plan", "run_checks"]

log = logging.getLogger(__name__)


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


def _open_detector(cfg: FirstPassConfig, dc: Any):
    from extra_data.components import AGIPD1M

    return AGIPD1M(dc, detector_name=cfg.detector_name, min_modules=cfg.min_modules)


def build_plan(cfg: FirstPassConfig, dc: Any = None) -> RunPlan:
    """Build the run plan (context file §6.8).

    :param dc: an open ``DataCollection``. When ``None``, the proc run named by
        ``cfg`` is opened.

    Trains present in the run but missing from the detector selection get
    ``MISSING_MODULES``; trains with no frames get ``NO_FRAMES``. Neither owns
    any row of the frame table, so the table spans only the detector's trains
    while the ledger spans every train of the run.
    """
    if dc is None:
        from extra_data import open_run

        dc = open_run(cfg.proposal, cfg.run, data="proc")

    det = _open_detector(cfg, dc)
    counts = det.frame_counts
    # Computed here rather than via det.train_id_to_ix, which is deprecated.
    first = (counts.cumsum() - counts).astype(np.int64)
    detector_trains = {int(tid): index for index, tid in enumerate(counts.index)}
    modules_present = _modules_present(dc, det)

    records: list[TrainRecord] = []
    row = 0
    for raw_train_id in dc.train_ids:
        train_id = int(raw_train_id)
        index = detector_trains.get(train_id)
        if index is None:
            # No rows of its own; the offset is where the next train starts, so
            # the ledger stays monotonic. A train where *no* module wrote a
            # frame is NO_FRAMES; one where some but too few did is
            # MISSING_MODULES. AGIPD1M cannot tell them apart — it drops both
            # before frame_counts exists — so the per-module counts are
            # consulted directly.
            status = (
                FrameStatus.NO_FRAMES
                if modules_present.get(train_id, 0) == 0
                else FrameStatus.MISSING_MODULES
            )
            records.append(TrainRecord(train_id, 0, row, status))
            continue
        n_frames = int(counts.iloc[index])
        if row != int(first.iloc[index]):
            raise AssertionError(
                f"row offset disagreement for train {train_id}: counted {row}, "
                f"frame_counts says {int(first.iloc[index])}"
            )
        records.append(
            TrainRecord(
                train_id,
                n_frames,
                row,
                FrameStatus.OK if n_frames else FrameStatus.NO_FRAMES,
            )
        )
        row += n_frames

    blocks = _build_blocks(records, cfg.trains_per_block)
    n_frames = int(counts.sum())
    log.info(
        "run %d: %d trains, %d frames, %d blocks",
        cfg.run,
        len(records),
        n_frames,
        len(blocks),
    )
    return RunPlan(
        trains=tuple(records),
        blocks=tuple(blocks),
        n_frames=n_frames,
        detector_name=det.detector_name,
        checks=run_checks(dc, det, counts),
    )


def _modules_present(dc: Any, det: Any) -> dict[int, int]:
    """How many detector modules wrote at least one frame, per train.

    ``det.frame_counts`` cannot answer this: ``AGIPD1M`` has already discarded
    every train below ``min_modules`` by the time it exists.
    """
    import pandas as pd

    per_module = (
        pd.DataFrame(
            {src: dc.get_data_counts(src, "image.data") for src in det.source_to_modno}
        )
        .fillna(0)
        .astype(np.uint64)
    )
    present = (per_module > 0).sum(axis=1)
    return {int(tid): int(value) for tid, value in present.items()}


def _build_blocks(records: list[TrainRecord], trains_per_block: int) -> list[Block]:
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


def run_checks(dc: Any, det: Any, counts: Any) -> dict[str, Any]:
    """Provenance flags for the run (context file §6.8).

    None of these select frames in v1 (integrator I1); each is recorded and a
    disagreement is flagged. Every check is best-effort: a missing source makes
    the check unavailable, which is itself recorded, rather than failing the
    run before any data is read.
    """
    checks: dict[str, Any] = {}

    def attempt(name: str, fn) -> None:
        try:
            checks[name] = fn()
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            checks[name] = f"unavailable: {error!r}"
            log.warning("run check %r unavailable: %r", name, error)

    def pulse_check() -> dict[str, Any]:
        from extra.components import XrayPulses

        pulses = XrayPulses(dc)
        pulse_counts = pulses.pulse_counts()
        shared = counts.index.intersection(pulse_counts.index)
        mismatched = int((counts[shared] != pulse_counts[shared]).sum())
        return {
            "constant_pattern": bool(pulses.is_constant_pattern()),
            "trains_compared": int(len(shared)),
            "frames_ne_xray_pulses": mismatched,
        }

    def quadrant_check() -> dict[str, Any]:
        from extra.components import AGIPD1MQuadrantMotors

        positions = AGIPD1MQuadrantMotors(dc).positions(compressed=True)
        moved = len(positions) > 1
        return {"n_positions": int(len(positions)), "quadrants_moved": bool(moved)}

    def energy_check() -> dict[str, Any]:
        from extra.components import XGM

        energies = np.asarray(XGM(dc).photon_energy_by_train())
        finite = energies[np.isfinite(energies)]
        if finite.size == 0:
            return {"available": False}
        return {
            "available": True,
            "mean_kev": float(finite.mean()) / 1000.0,
            "spread_fraction": float(
                (finite.max() - finite.min()) / max(abs(finite.mean()), 1e-12)
            ),
        }

    attempt("xray_pulses", pulse_check)
    attempt("quadrant_motors", quadrant_check)
    attempt("xgm_photon_energy", energy_check)
    return checks
