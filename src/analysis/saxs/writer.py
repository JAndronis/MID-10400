"""Output file, ledger and resume (context file §7).

Only the parent process opens this file for writing (§3 rule 8). Rows are
addressed by label: ``write_block`` checks every frame's trainId and every
train's frame count against the plan before anything is stored, so a dropped
train can never shift frames onto the wrong train (CLAUDE.md pitfall 4).

There are no NaN sentinels anywhere in the file. A frame that was not
integrated carries a status code and zeros (§3 rule 7); ``status`` is what
distinguishes the two, and pooling must select on it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.saxs.config import FirstPassConfig
from analysis.saxs.plan import Block, RunPlan
from analysis.saxs.status import FrameStatus
from analysis.saxs.worker import BlockResult

__all__ = ["ConfigHashMismatch", "FirstPassWriter", "IncompleteRun"]

log = logging.getLogger(__name__)

FRAME_VECTORS: dict[str, Any] = {
    "trainId": np.uint64,
    "reader_pulseId": np.uint64,
    "cellId": np.uint16,
    "status": np.uint8,
    "photons_valid": np.float32,
    "n_bad_pixels": np.uint32,
    "n_frame_specific": np.uint32,
    "max_count": np.uint16,
}
FRAME_MATRICES = ("signal", "normalization", "variance")


class ConfigHashMismatch(RuntimeError):
    """An existing output file was written with a different configuration."""


class IncompleteRun(RuntimeError):
    """Some frames did not reach ``OK`` and ``allow_incomplete`` is False."""


class FirstPassWriter:
    """The single writer for one run's output file."""

    def __init__(self, handle: h5py.File, cfg: FirstPassConfig, plan: RunPlan) -> None:
        self._f = handle
        self._cfg = cfg
        self._plan = plan

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def open_or_create(
        cls, cfg: FirstPassConfig, plan: RunPlan, path: Path | None = None
    ) -> FirstPassWriter:
        """Open the run's output file, creating and initialising it if needed.

        An existing file whose stored config hash differs is refused unless
        ``cfg.overwrite``: resuming into a file built under a different
        configuration would mix incompatible rows.
        """
        path = Path(path) if path is not None else cfg.output_file
        path.parent.mkdir(parents=True, exist_ok=True)
        config_hash = cfg.config_hash()

        if path.exists() and not cfg.overwrite:
            with h5py.File(path, "r") as existing:
                stored = existing["provenance"].attrs.get("config_hash")
            if stored != config_hash:
                raise ConfigHashMismatch(
                    f"{path} was written with config hash {stored}, "
                    f"this run has {config_hash}; pass overwrite=True to replace it"
                )
            return cls(h5py.File(path, "r+"), cfg, plan)

        handle = h5py.File(path, "w")
        cls._create_layout(handle, cfg, plan, config_hash)
        return cls(handle, cfg, plan)

    @staticmethod
    def _create_layout(
        handle: h5py.File, cfg: FirstPassConfig, plan: RunPlan, config_hash: str
    ) -> None:
        n, npt = plan.n_frames, cfg.npt
        frames = handle.create_group("frames")
        # One chunk per train keeps a block's writes contiguous; the largest
        # train in the run sets the chunk length.
        chunk_rows = max(
            (train.n_frames for train in plan.trains if train.n_frames), default=1
        )
        for name, dtype in FRAME_VECTORS.items():
            fill = FrameStatus.NOT_PROCESSED if name == "status" else 0
            frames.create_dataset(
                name,
                shape=(n,),
                dtype=dtype,
                fillvalue=fill,
                chunks=(min(chunk_rows, n),) if n else None,
                compression="gzip",
                compression_opts=1,
            )
        for name in FRAME_MATRICES:
            frames.create_dataset(
                name,
                shape=(n, npt),
                dtype=np.float32,
                fillvalue=0.0,
                chunks=(min(chunk_rows, n), npt) if n else None,
                compression="gzip",
                compression_opts=1,
            )

        trains = handle.create_group("trains")
        trains["trainId"] = np.array([t.train_id for t in plan.trains], dtype=np.uint64)
        trains["first"] = np.array([t.first_row for t in plan.trains], dtype=np.uint64)
        trains["count"] = np.array([t.n_frames for t in plan.trains], dtype=np.uint32)
        trains["status"] = np.array(
            [int(t.status) for t in plan.trains], dtype=np.uint8
        )

        provenance = handle.create_group("provenance")
        provenance.attrs["config_hash"] = config_hash
        provenance.attrs["config"] = json.dumps(
            {
                **{
                    key: (sorted(value) if isinstance(value, frozenset) else value)
                    for key, value in asdict(cfg).items()
                },
                "method": list(cfg.method),
            },
            sort_keys=True,
            default=str,
        )
        provenance.attrs["detector_name"] = plan.detector_name
        provenance.attrs["run_checks"] = json.dumps(plan.checks, default=str)
        handle.create_group("masks")
        handle.create_group("operator")
        handle.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> FirstPassWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── resume ───────────────────────────────────────────────────────────────
    def block_complete(self, block: Block) -> bool:
        """A block is complete when none of its frames is ``NOT_PROCESSED``."""
        rows = block.rows()
        if rows.size == 0:
            return True
        status = self._f["frames/status"][rows[0] : rows[-1] + 1]
        return bool((status != FrameStatus.NOT_PROCESSED).all())

    # ── writing ──────────────────────────────────────────────────────────────
    def write_block(self, block: Block, result: BlockResult) -> None:
        """Validate the block's labels, then store its rows.

        :raises ValueError: the result's shape does not match the block. Label
            disagreements are not an error — they are recorded as
            ``LABEL_MISMATCH`` by the worker and stored as such — but a result
            of the wrong length is a programming fault.
        """
        rows = block.rows()
        if result.n_frames != rows.size:
            raise ValueError(
                f"block {block.index} expects {rows.size} frames, "
                f"result carries {result.n_frames}"
            )

        integrated = result.status == FrameStatus.OK
        expected_trains = (
            np.concatenate(
                [
                    np.full(count, train_id, dtype=np.uint64)
                    for train_id, count in zip(
                        block.train_ids, block.expected_frames, strict=True
                    )
                ]
            )
            if rows.size
            else np.zeros(0, dtype=np.uint64)
        )
        mislabelled = integrated & (result.train_id != expected_trains)
        if mislabelled.any():
            log.error(
                "block %d: %d frames carry a trainId the plan does not expect",
                block.index,
                int(mislabelled.sum()),
            )
            result.status[mislabelled] = FrameStatus.LABEL_MISMATCH
            integrated = result.status == FrameStatus.OK

        start, stop = int(rows[0]), int(rows[-1]) + 1
        frames = self._f["frames"]
        frames["trainId"][start:stop] = expected_trains
        frames["reader_pulseId"][start:stop] = result.pulse_id
        frames["cellId"][start:stop] = result.cell_id
        frames["status"][start:stop] = result.status
        frames["photons_valid"][start:stop] = result.photons_valid
        frames["n_bad_pixels"][start:stop] = result.n_bad_pixels
        frames["n_frame_specific"][start:stop] = result.n_frame_specific
        frames["max_count"][start:stop] = result.max_count
        for name in FRAME_MATRICES:
            frames[name][start:stop] = getattr(result, name)
        self._f.flush()

    def mark(self, block: Block, status: FrameStatus, message: str = "") -> None:
        """Record a status for every frame of a block, with no data."""
        rows = block.rows()
        if rows.size == 0:
            return
        self._f["frames/status"][int(rows[0]) : int(rows[-1]) + 1] = status
        if message:
            errors = self._f["provenance"].attrs.get("block_errors", "{}")
            recorded = json.loads(errors)
            recorded[str(block.index)] = message
            self._f["provenance"].attrs["block_errors"] = json.dumps(recorded)
        self._f.flush()

    def mark_remaining(self, status: FrameStatus) -> None:
        """Stamp every still-unprocessed frame, e.g. after a broken pool."""
        current = self._f["frames/status"][:]
        current[current == FrameStatus.NOT_PROCESSED] = status
        self._f["frames/status"][:] = current
        self._f.flush()

    # ── finishing ────────────────────────────────────────────────────────────
    def store_operator(self, op: Any) -> None:
        group = self._f["operator"]
        for name in ("coef", "bins", "indptr", "omega", "q"):
            if name in group:
                del group[name]
            group.create_dataset(name, data=getattr(op, name))
        group.attrs["sha256"] = op.sha256
        if "q" not in self._f:
            centers = self._f.create_group("q")
            centers["centers"] = np.asarray(op.q, dtype=np.float64)
            centers["centers"].attrs["unit"] = "nm^-1"

    def store_masks(self, masks: Any, sampled_trains: np.ndarray) -> None:
        group = self._f["masks"]
        payload = {
            "cells": masks.cells,
            "base_bad_packed": np.packbits(masks.base_bad, axis=-1),
            "D": masks.denominators,
            "n_samples": masks.n_samples,
            "static_bad_packed": np.packbits(masks.static_bad),
            "D_static": masks.static_denominator,
            "sampled_trains": np.asarray(sampled_trains, dtype=np.uint64),
        }
        for name, value in payload.items():
            if name in group:
                del group[name]
            group.create_dataset(
                name, data=value, compression="gzip", compression_opts=1
            )
        group.attrs["sha256"] = masks.sha256
        group.attrs["static_sha256"] = masks.static_sha256
        group.attrs["operator_sha256"] = masks.operator_sha256
        group.attrs["bits_present"] = int(masks.bits_present)
        group.attrs["unexpected_bits"] = int(masks.unexpected_bits)

    def finalise(self, provenance: dict[str, Any]) -> None:
        group = self._f["provenance"]
        for key, value in provenance.items():
            group.attrs[key] = (
                value
                if isinstance(value, (str, int, float))
                else json.dumps(value, default=str)
            )
        self._f.flush()

    # ── inspection ───────────────────────────────────────────────────────────
    def status_summary(self) -> dict[str, int]:
        status = self._f["frames/status"][:]
        return {
            code.name: int((status == code).sum())
            for code in FrameStatus
            if (status == code).any()
        }

    def any_not_ok(self) -> bool:
        return bool((self._f["frames/status"][:] != FrameStatus.OK).any())

    def pooled_per_train(self) -> Any:
        """Per-train pooled ``I(q)`` over the ``OK`` frames (context file §9).

        ``I(q) = Σ S / Σ N`` over each train's integrated frames. Trains with no
        integrated frame are returned as zeros with ``count == 0``; there are no
        NaN sentinels, so callers select on ``count``.
        """
        import xarray as xr

        status = self._f["frames/status"][:]
        train_ids = np.array([t.train_id for t in self._plan.trains], dtype=np.uint64)
        q = (
            np.asarray(self._f["q/centers"][:])
            if "q" in self._f
            else np.arange(self._cfg.npt, dtype=np.float64)
        )

        intensity = np.zeros((train_ids.size, self._cfg.npt), dtype=np.float64)
        counts = np.zeros(train_ids.size, dtype=np.uint32)
        for index, train in enumerate(self._plan.trains):
            if train.n_frames == 0:
                continue
            rows = slice(train.first_row, train.first_row + train.n_frames)
            ok = status[rows] == FrameStatus.OK
            counts[index] = int(ok.sum())
            if not ok.any():
                continue
            signal = self._f["frames/signal"][rows][ok].sum(axis=0)
            normalization = self._f["frames/normalization"][rows][ok].sum(axis=0)
            valid = normalization > 0
            intensity[index, valid] = signal[valid] / normalization[valid]

        return xr.Dataset(
            {
                "intensity": (("trainId", "q"), intensity),
                "n_frames": (("trainId",), counts),
            },
            coords={"trainId": train_ids, "q": q},
        )
