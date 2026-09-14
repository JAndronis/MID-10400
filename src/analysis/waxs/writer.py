"""Output file, ledger and resume for one detector (WAXS context file §3 D2).

The layout, the ledger, resume, the label checks and ``pooled_per_train`` are
:class:`analysis.common.writer.FrameTableWriter`. What is here is the JUNGFRAU
schema and the two JUNGFRAU stores.

Two columns differ from AGIPD in a way worth stating:

* **There is no ``reader_pulseId``.** ``JUNGFRAU`` is a ``MultimodDetectorBase``
  and its ``MultimodKeyData`` has ``train_id_coordinates()`` and nothing else —
  no pulse ids exist in the reader at all (§5 R5). Which of a train's X-ray
  pulses each memory cell sampled has to be aligned from ``XrayPulses``/LITFRM
  onto the cell axis, and until that alignment exists the pass stores the cell
  id and leaves pulse identity absent rather than synthesising one from
  position (CLAUDE.md pitfall 4).
* **``max_count`` becomes ``max_kev``**, a float. The data are energies, not
  counts.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from analysis.common.status import FrameStatus

# Re-exported: the tests and ``run.py`` name these through this module.
from analysis.common.writer import (
    ConfigHashMismatch,
    FrameTableWriter,
    IncompleteRun,
    SchemaMismatch,
    as_handle,
    pooled_per_train,
    q_centers,
)

__all__ = [
    "ConfigHashMismatch",
    "IncompleteRun",
    "JungfrauWaxsWriter",
    "SchemaMismatch",
    "per_cell",
    "pooled_per_train",
]

log = logging.getLogger(__name__)


class JungfrauWaxsWriter(FrameTableWriter):
    """The single writer for one detector's output file."""

    FRAME_VECTORS = {
        "trainId": np.uint64,
        "cellId": np.uint16,
        "status": np.uint8,
        "energy_valid": np.float32,
        "n_bad_pixels": np.uint32,
        "n_negative_variance_bins": np.uint32,
        "max_kev": np.float32,
        "max_kev_static": np.float32,
        "n_extreme_pixels": np.uint32,
    }
    FRAME_VECTOR_SOURCES = {
        "cellId": "cell_id",
        "status": "status",
        "energy_valid": "energy_valid",
        "n_bad_pixels": "n_bad_pixels",
        "n_negative_variance_bins": "n_negative_variance_bins",
        "max_kev": "max_kev",
        "max_kev_static": "max_kev_static",
        "n_extreme_pixels": "n_extreme_pixels",
    }
    EXTRA_GROUPS = ("operator", "cells")

    def store_operator(self, op: Any) -> None:
        """The q axis, the solid angle, the static mask and the PONI verbatim.

        The PONI text is stored rather than only its path: it is a few hundred
        bytes and it makes the file self-describing, so a reader months later
        can rebuild the identical geometry without the file still being on
        GPFS under the same name.
        """
        group = self._f["operator"]
        payload = {
            "q": np.asarray(op.q, dtype=np.float64),
            "omega": np.asarray(op.omega, dtype=np.float32),
            "static_bad_packed": np.packbits(op.static_bad),
        }
        for name, value in payload.items():
            if name in group:
                del group[name]
            group.create_dataset(
                name, data=value, compression="gzip", compression_opts=1
            )
        group.attrs["sha256"] = op.sha256
        group.attrs["static_sha256"] = op.static_sha256
        group.attrs["poni_sha256"] = op.poni_sha256 or ""
        group.attrs["poni"] = op.poni_text
        group.attrs["dist_m"] = op.dist_m
        group.attrs["wavelength_m"] = op.wavelength_m
        group.attrs["method"] = json.dumps(list(op.method))
        if "q" not in self._f:
            centers = self._f.create_group("q")
            centers["centers"] = np.asarray(op.q, dtype=np.float64)
            centers["centers"].attrs["unit"] = op.unit

    def store_cells(
        self, classification: Any, model: Any, sampled_trains: np.ndarray
    ) -> None:
        """The lit/dark split, the per-cell evidence and the error model."""
        group = self._f["cells"]
        payload = {
            "cells": np.asarray(classification.cells, dtype=np.uint16),
            "lit_fraction": np.asarray(classification.lit_fraction, dtype=np.float64),
            "n_samples": np.asarray(classification.n_samples, dtype=np.uint32),
            "lit": np.asarray(classification.lit, dtype=np.uint16),
            "dark": np.asarray(classification.dark, dtype=np.uint16),
            "sampled_trains": np.asarray(sampled_trains, dtype=np.uint64),
        }
        for name, value in payload.items():
            if name in group:
                del group[name]
            group.create_dataset(name, data=value)
        group.attrs["read_noise_kev"] = model.read_noise_kev
        group.attrs["photon_energy_kev"] = model.photon_energy_kev
        group.attrs["error_model_source"] = model.source
        group.attrs["error_model_samples"] = model.n_samples
        group.attrs["error_model_sha256"] = model.sha256

    def data_check_summary(self) -> dict[str, Any]:
        """Pixels the value check excluded, and frames it refused (§3 D6′).

        The per-run record of what D6 cost. Since 2026-09-14 the check drops the
        offending *pixel* and keeps the frame, so the number that matters is how
        many frames lost pixels and how many — on r0480/jf1 that was 473 frames
        losing a median of 3 each, NaCl Bragg spots from the evaporating
        droplet. ``n_frames_refused`` is the residue: a frame with nothing left
        to integrate at all, which should not happen and is loud if it does.

        Empty when the run lost nothing, so a clean run carries no attribute
        rather than an attribute full of zeros.
        """
        frames = self._f["frames"]
        status = frames["status"][:]
        excluded = np.asarray(frames["n_extreme_pixels"][:], dtype=np.int64)
        integrated = status == FrameStatus.OK
        affected = integrated & (excluded > 0)
        refused = status == FrameStatus.DATA_CHECK_FAILED
        if not affected.any() and not refused.any():
            return {}
        checked = np.asarray(frames["max_kev_static"][:], dtype=np.float64)
        worst = checked[affected]
        worst = worst[np.isfinite(worst)]
        return {
            "max_abs_kev": float(self._cfg.max_abs_kev),
            "n_frames_with_excluded_pixels": int(affected.sum()),
            "n_pixels_excluded": int(excluded[affected].sum()),
            "worst_frame_n_pixels": int(excluded[affected].max(initial=0)),
            "worst_excluded_kev": float(worst.max(initial=0.0)),
            "n_frames_refused": int(refused.sum()),
            "note": (
                "the pixel is dropped and the frame kept; its q bin reads low in "
                "that frame, so filter on frames/n_extreme_pixels before treating "
                "an affected bin quantitatively"
            ),
        }

    def per_cell(self) -> Any:
        """Per-frame ``I(q)`` on a (train, cell) grid — see :func:`per_cell`."""
        return per_cell(self._f)


def per_cell(source: Any, *, dtype: Any = np.float32) -> Any:
    """Per-frame ``I(q)`` as a ``(trainId, cellId, q)`` DataArray.

    The JUNGFRAU counterpart of ``analysis.saxs.writer.per_pulse``, and the same
    discipline: every frame is placed by its *stored* trainId and cellId, never
    by its row position, so a dropped or short train cannot slide frames onto
    the wrong train (CLAUDE.md pitfall 4). Only ``OK`` frames carry trustworthy
    labels, so anything else is counted in ``attrs["unplaced"]`` rather than
    guessed onto a slot. Empty slots are zeros with ``n_frames == 0``, never
    NaN (AGIPD context file §3 rule 7).

    ``n_frames`` is a non-dimension coordinate rather than a second variable,
    which keeps this a DataArray: DAMNIT renders a 3-D DataArray in the table
    as ``float32: (n, m, npt)`` but a Dataset only as ``Dataset (48MB)``.

    The grid is small — 3000 trains × 8 lit cells × npt 500 is 48 MB as f4,
    against 0.93 GB for the AGIPD per-pulse grid — so there is no reason to
    return anything coarser.
    """
    import xarray as xr

    with as_handle(source) as handle:
        frames = handle["frames"]
        status = frames["status"][:]
        row_train = frames["trainId"][:]
        row_cell = frames["cellId"][:]
        npt = int(frames["signal"].shape[1])
        q = q_centers(handle, npt)
        train_ids = handle["trains/trainId"][:]
        if not np.all(np.diff(train_ids.astype(np.int64)) > 0):
            raise ValueError("the train table is not strictly increasing")
        # Inside the ``with``: an AttributeManager kept past it returns the
        # default rather than raising (CLAUDE.md pitfall 13).
        data_check = handle["provenance"].attrs.get("data_check", "")

        placeable = status == FrameStatus.OK
        cell_ids = np.unique(row_cell[placeable])
        unplaced = {
            code.name: int(((status == code) & ~placeable).sum())
            for code in FrameStatus
            if code is not FrameStatus.OK and (status == code).any()
        }

        intensity = np.zeros((train_ids.size, cell_ids.size, npt), dtype=dtype)
        n_frames = np.zeros((train_ids.size, cell_ids.size), dtype=np.uint8)

        if placeable.any():
            trains = row_train[placeable]
            cells = row_cell[placeable]
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
            cell_index = np.searchsorted(cell_ids, cells)

            signal = frames["signal"][:][placeable]
            normalization = frames["normalization"][:][placeable]
            value = np.zeros_like(signal)
            np.divide(signal, normalization, out=value, where=normalization > 0)
            intensity[train_index, cell_index] = value
            n_frames[train_index, cell_index] = 1

    placed = int(n_frames.sum())
    if placed != int(placeable.sum()):
        raise ValueError(
            f"{int(placeable.sum()) - placed} frame(s) shared a (train, cell) "
            "slot with another; the cell ids do not identify frames uniquely"
        )

    result = xr.DataArray(
        intensity,
        dims=("trainId", "cellId", "q"),
        coords={
            "trainId": train_ids,
            "cellId": cell_ids,
            "q": q,
            "n_frames": (("trainId", "cellId"), n_frames),
        },
        name="intensity",
    )
    result.attrs["unplaced"] = json.dumps(unplaced, sort_keys=True)
    result.attrs["n_placed"] = placed
    # Why the DATA_CHECK_FAILED slots are empty, on the object a DAMNIT user
    # actually has in front of them rather than only in the file.
    result.attrs["data_check"] = data_check
    return result
