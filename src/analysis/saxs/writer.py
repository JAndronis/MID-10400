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
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.plan import Block, RunPlan
from analysis.saxs.status import FrameStatus
from analysis.saxs.worker import BlockResult

__all__ = [
    "ConfigHashMismatch",
    "AgipdSaxsWriter",
    "IncompleteRun",
    "per_pulse",
    "pooled_per_train",
]

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


class AgipdSaxsWriter:
    """The single writer for one run's output file."""

    def __init__(self, handle: h5py.File, cfg: AgipdSaxsConfig, plan: RunPlan) -> None:
        self._f = handle
        self._cfg = cfg
        self._plan = plan

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def open_or_create(
        cls, cfg: AgipdSaxsConfig, plan: RunPlan, path: Path | None = None
    ) -> AgipdSaxsWriter:
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
        handle: h5py.File, cfg: AgipdSaxsConfig, plan: RunPlan, config_hash: str
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

    def __enter__(self) -> AgipdSaxsWriter:
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
        """Per-train pooled ``I(q)`` — see :func:`pooled_per_train`."""
        return pooled_per_train(self._f)

    def per_pulse(self) -> Any:
        """Per-frame ``I(q)`` on a (train, pulse) grid — see :func:`per_pulse`."""
        return per_pulse(self._f)


# ── reducers (context file §9) ────────────────────────────────────────────────
# These take the output file rather than a live writer, so the same code serves
# the run that produced it and any post hoc analysis of it months later. They
# read the ``/trains`` table for the row-to-train map: ``frames/trainId`` is
# only filled for blocks that were actually written, so a run with an
# unprocessed block has rows carrying trainId 0 (CLAUDE.md pitfall 4).


@contextmanager
def _as_handle(source: Any) -> Any:
    """Accept an open file or a path, and yield an open file either way."""
    if isinstance(source, h5py.File):
        yield source
    else:
        with h5py.File(source, "r") as handle:
            yield handle


def _q_centers(handle: h5py.File, npt: int) -> np.ndarray:
    if "q" in handle:
        return np.asarray(handle["q/centers"][:])
    return np.arange(npt, dtype=np.float64)


def pooled_per_train(source: Any) -> Any:
    """Per-train pooled ``I(q)`` and ``σ(q)`` over the ``OK`` frames (§9).

    ``I(q) = Σ S / Σ N`` and ``σ(q) = sqrt(Σ V) / Σ N`` over each train's
    integrated frames. Trains with no integrated frame are returned as zeros
    with ``n_frames == 0``; there are no NaN sentinels, so callers select on
    ``n_frames``.
    """
    import xarray as xr

    with _as_handle(source) as handle:
        frames = handle["frames"]
        status = frames["status"][:]
        npt = int(frames["signal"].shape[1])
        q = _q_centers(handle, npt)
        train_ids = handle["trains/trainId"][:]
        first = handle["trains/first"][:]
        count = handle["trains/count"][:]

        intensity = np.zeros((train_ids.size, npt), dtype=np.float64)
        sigma = np.zeros((train_ids.size, npt), dtype=np.float64)
        counts = np.zeros(train_ids.size, dtype=np.uint32)

        for index in range(train_ids.size):
            n = int(count[index])
            if n == 0:
                continue
            rows = slice(int(first[index]), int(first[index]) + n)
            ok = status[rows] == FrameStatus.OK
            counts[index] = int(ok.sum())
            if not ok.any():
                continue
            signal = frames["signal"][rows][ok].sum(axis=0)
            normalization = frames["normalization"][rows][ok].sum(axis=0)
            variance = frames["variance"][rows][ok].sum(axis=0)
            valid = normalization > 0
            intensity[index, valid] = signal[valid] / normalization[valid]
            sigma[index, valid] = np.sqrt(variance[valid]) / normalization[valid]

    return xr.Dataset(
        {
            "intensity": (("trainId", "q"), intensity),
            "sigma": (("trainId", "q"), sigma),
            "n_frames": (("trainId",), counts),
        },
        coords={"trainId": train_ids, "q": q},
    )


def per_pulse(source: Any, *, dtype: Any = np.float32, chunk_rows: int = 20_000) -> Any:
    """Per-frame ``I(q)`` on a ``(trainId, pulseId, q)`` grid (context file §9).

    Every frame placed on the grid is placed by its *stored* trainId and
    pulseId, never by its position in the frame table, so a dropped train or a
    short one cannot slide frames onto the wrong train (CLAUDE.md pitfall 4).
    The pulse axis is the sorted set of pulse ids actually seen.

    Only ``OK`` frames carry trustworthy labels — a frame whose train failed
    label validation was never given a pulse id, and an unwritten row carries
    zeros — so those are counted in ``attrs["unplaced"]`` rather than guessed
    onto a slot. Slots that received no frame are zero with ``n_frames == 0``,
    the same selection idiom as :func:`pooled_per_train` and, like it, free of
    NaN sentinels (context file §3 rule 7).

    :param dtype: storage for the intensity grid. The default f4 costs ~0.9 GB
        for a 3000-train, 155-pulse run at ``npt`` 500; f8 doubles that.
    """
    import xarray as xr

    with _as_handle(source) as handle:
        frames = handle["frames"]
        status = frames["status"][:]
        row_train = frames["trainId"][:]
        row_pulse = frames["reader_pulseId"][:]
        npt = int(frames["signal"].shape[1])
        q = _q_centers(handle, npt)
        train_ids = handle["trains/trainId"][:]
        if not np.all(np.diff(train_ids.astype(np.int64)) > 0):
            raise ValueError("the train table is not strictly increasing")

        placeable = status == FrameStatus.OK
        pulse_ids = np.unique(row_pulse[placeable])
        unplaced = {
            code.name: int(((status == code) & ~placeable).sum())
            for code in FrameStatus
            if code is not FrameStatus.OK and (status == code).any()
        }

        intensity = np.zeros((train_ids.size, pulse_ids.size, npt), dtype=dtype)
        n_frames = np.zeros((train_ids.size, pulse_ids.size), dtype=np.uint8)

        for start in range(0, status.size, chunk_rows):
            stop = min(start + chunk_rows, status.size)
            keep = placeable[start:stop]
            if not keep.any():
                continue
            trains = row_train[start:stop][keep]
            pulses = row_pulse[start:stop][keep]
            # searchsorted returns size for anything past the last train, so
            # the result is clamped before it is used as an index: an unknown
            # trainId must reach the check below, not raise IndexError first.
            train_index = np.clip(
                np.searchsorted(train_ids, trains), 0, train_ids.size - 1
            )
            if not np.array_equal(train_ids[train_index], trains):
                raise ValueError(
                    "a frame carries a trainId that is not in the train table"
                )
            pulse_index = np.searchsorted(pulse_ids, pulses)

            signal = frames["signal"][start:stop][keep]
            normalization = frames["normalization"][start:stop][keep]
            value = np.zeros_like(signal)
            np.divide(signal, normalization, out=value, where=normalization > 0)
            intensity[train_index, pulse_index] = value
            n_frames[train_index, pulse_index] = 1

    placed = int(n_frames.sum())
    if placed != int(placeable.sum()):
        raise ValueError(
            f"{int(placeable.sum()) - placed} frame(s) shared a (train, pulse) "
            "slot with another; the pulse ids do not identify frames uniquely"
        )

    result = xr.Dataset(
        {
            "intensity": (("trainId", "pulseId", "q"), intensity),
            "n_frames": (("trainId", "pulseId"), n_frames),
        },
        coords={"trainId": train_ids, "pulseId": pulse_ids, "q": q},
    )
    result.attrs["unplaced"] = json.dumps(unplaced, sort_keys=True)
    result.attrs["n_placed"] = placed
    return result
