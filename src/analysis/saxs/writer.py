"""Output file, ledger and resume.

The AGIPD schema and the two AGIPD-specific stores. Everything detector-
agnostic — the layout, the ledger, resume, the label checks, ``mark`` and
``pooled_per_train`` — is :class:`analysis.common.writer.FrameTableWriter`.

There are no NaN sentinels anywhere in the file. A frame that was not
integrated carries a status code and zeros; ``status`` is what
distinguishes the two, and pooling must select on it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

# Re-exported: the SAXS tests and ``run.py`` name these through this module.
from analysis.common.writer import (
    ConfigHashMismatch,
    FrameTableWriter,
    IncompleteRun,
    per_label,
    pooled_per_train,
)

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

    :param source: an open handle or a path to a finished output file.
    :param dtype: storage for the intensity grid.
    :param chunk_rows: rows read per pass, bounding peak memory on this grid,
        which is ~0.9 GB as f4 for a 3000-train, 155-pulse run at ``npt`` 500.
    :returns: the grid — see :func:`analysis.common.writer.per_label`.
    """
    return per_label(
        source,
        label_column="reader_pulseId",
        label_dim="pulseId",
        dtype=dtype,
        chunk_rows=chunk_rows,
    )
