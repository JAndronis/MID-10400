"""Output file, ledger and resume.

The AGIPD schema and the two AGIPD-specific stores. Everything detector-
agnostic — the layout, the ledger, resume, the label checks, ``mark`` and
``pooled_per_train`` — is :class:`analysis.common.writer.FrameTableWriter`.

There are no NaN sentinels anywhere in the file. A frame that was not
integrated carries a status code and zeros; ``status`` is what
distinguishes the two, and pooling must select on it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

# Re-exported: the SAXS tests and ``run.py`` name these through this module.
from analysis.common.writer import (
    ConfigHashMismatch,
    FrameTableWriter,
    IncompleteRun,
    as_handle,
    pooled_per_train,
    q_centers,
)
from analysis.saxs.status import FrameStatus

__all__ = [
    "ConfigHashMismatch",
    "AgipdSaxsWriter",
    "IncompleteRun",
    "per_pulse",
    "pooled_per_train",
]

log = logging.getLogger(__name__)


class AgipdSaxsWriter(FrameTableWriter):
    """The single writer for one AGIPD run's output file."""

    FRAME_VECTORS = {
        "trainId": np.uint64,
        "reader_pulseId": np.uint64,
        "cellId": np.uint16,
        "status": np.uint8,
        "photons_valid": np.float32,
        "n_bad_pixels": np.uint32,
        "n_frame_specific": np.uint32,
        "max_count": np.uint16,
    }
    FRAME_VECTOR_SOURCES = {
        "reader_pulseId": "pulse_id",
        "cellId": "cell_id",
        "status": "status",
        "photons_valid": "photons_valid",
        "n_bad_pixels": "n_bad_pixels",
        "n_frame_specific": "n_frame_specific",
        "max_count": "max_count",
    }
    EXTRA_GROUPS = ("masks", "operator")

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

    def per_pulse(self) -> Any:
        """Per-frame ``I(q)`` on a (train, pulse) grid — see :func:`per_pulse`."""
        return per_pulse(self._f)


def per_pulse(source: Any, *, dtype: Any = np.float32, chunk_rows: int = 20_000) -> Any:
    """Per-frame ``I(q)`` as a ``(trainId, pulseId, q)`` DataArray.

    A ``DataArray`` rather than a ``Dataset``, matching what
    ``analysis_helpers.integrate_run`` returned, with ``n_frames`` carried as a
    non-dimension coordinate.

    Every frame placed on the grid is placed by its *stored* trainId and
    pulseId, never by its position in the frame table, so a dropped train or a
    short one cannot slide frames onto the wrong train.
    The pulse axis is the sorted set of pulse ids actually seen.

    Only ``OK`` frames carry trustworthy labels — a frame whose train failed
    label validation was never given a pulse id, and an unwritten row carries
    zeros — so those are counted in ``attrs["unplaced"]`` rather than guessed
    onto a slot. Slots that received no frame are zero with ``n_frames == 0``,
    the same selection idiom as :func:`pooled_per_train` and, like it, free of
    NaN sentinels.

    :param dtype: storage for the intensity grid. The default f4 costs ~0.9 GB
        for a 3000-train, 155-pulse run at ``npt`` 500; f8 doubles that.
    """
    import xarray as xr

    with as_handle(source) as handle:
        frames = handle["frames"]
        status = frames["status"][:]
        row_train = frames["trainId"][:]
        row_pulse = frames["reader_pulseId"][:]
        npt = int(frames["signal"].shape[1])
        q = q_centers(handle, npt)
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

    result = xr.DataArray(
        intensity,
        dims=("trainId", "pulseId", "q"),
        coords={
            "trainId": train_ids,
            "pulseId": pulse_ids,
            "q": q,
            # A non-dimension coordinate rather than a second data variable,
            # which would make this a Dataset. DAMNIT renders a 3-D DataArray
            # in the table as "float32: (n, m, npt)" — the cell
            # ``analysis_helpers.integrate_run`` produced — but a Dataset only
            # as "Dataset (930.49MB)".
            "n_frames": (("trainId", "pulseId"), n_frames),
        },
        name="intensity",
    )
    result.attrs["unplaced"] = json.dumps(unplaced, sort_keys=True)
    result.attrs["n_placed"] = placed
    return result
