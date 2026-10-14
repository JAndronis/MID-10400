"""Output file, ledger and resume for one detector.

The layout, the ledger, resume, the label checks and ``pooled_per_train`` are
:class:`analysis.common.writer.FrameTableWriter`. What is here is the JUNGFRAU
schema and the two JUNGFRAU stores.

Two columns differ from AGIPD in a way worth stating:

* **There is no ``reader_pulseId``.** ``JUNGFRAU`` is a ``MultimodDetectorBase``
  and its ``MultimodKeyData`` has ``train_id_coordinates()`` and nothing else —
  no pulse ids exist in the reader at all. Which of a train's X-ray
  pulses each memory cell sampled has to be aligned from ``XrayPulses``/LITFRM
  onto the cell axis, and until that alignment exists the pass stores the cell
  id and leaves pulse identity absent rather than synthesising one from
  position.
* **``max_count`` becomes ``max_kev``**, a float. ``max_kev``, ``max_kev_static``
  and ``energy_valid`` describe the raw keV values the frame arrived with;
  ``signal`` and ``variance`` are photon counts (``provenance.attrs["units"]``),
  counted per ASIC by :mod:`analysis.waxs.photons`, whose levels are stored per
  frame in ``asic_level`` so any frame can be recounted from proc.

A file written before photon counting has no ``units`` attribute and counts
keV. :func:`file_units` says which, and every reader that combines two files
refuses to mix them.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from analysis.common.status import FrameStatus
from analysis.common.writer import (
    FrameTableWriter,
    as_handle,
    per_label,
)
from analysis.waxs.photons import N_ASICS

__all__ = [
    "KEV",
    "PHOTONS",
    "JungfrauWaxsWriter",
    "file_units",
    "per_cell",
]

#: What ``signal`` counts in a file this writer produces.
PHOTONS = "photons"
#: What it counts in a file written before photon counting, which carries no
#: ``units`` attribute at all.
KEV = "keV"

log = logging.getLogger(__name__)


class JungfrauWaxsWriter(FrameTableWriter):
    """The single writer for one detector's output file."""

    FRAME_VECTORS = {
        "trainId": np.uint64,
        "cellId": np.uint16,
        "status": np.uint8,
        "energy_valid": np.float32,
        "n_photons": np.uint32,
        "n_bad_pixels": np.uint32,
        "n_negative_variance_bins": np.uint32,
        "max_kev": np.float32,
        "max_kev_static": np.float32,
        "n_extreme_pixels": np.uint32,
        "n_level_missing": np.uint8,
        "count_energy_kev": np.float32,
    }
    FRAME_VECTOR_SOURCES = {
        "cellId": "cell_id",
        "status": "status",
        "energy_valid": "energy_valid",
        "n_photons": "n_photons",
        "n_bad_pixels": "n_bad_pixels",
        "n_negative_variance_bins": "n_negative_variance_bins",
        "max_kev": "max_kev",
        "max_kev_static": "max_kev_static",
        "n_extreme_pixels": "n_extreme_pixels",
        "n_level_missing": "n_level_missing",
        "count_energy_kev": "count_energy_kev",
    }
    #: The zero level of each ASIC, keV, as counted with. NaN for an ASIC that
    #: had none in an integrated frame — a measured level of 0.0 is a real
    #: value, so 0 cannot stand for "none" — and 0 in rows never integrated.
    FRAME_ARRAYS = {"asic_level": (N_ASICS, np.float32)}
    EXTRA_GROUPS = ("operator", "cells")
    UNITS = PHOTONS

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
        group.attrs["read_noise_role"] = (
            "diagnostic: the stored variance is Poisson on the photon counts"
        )

    def level_summary(self) -> dict[str, Any]:
        """How the per-ASIC zero levels behaved over the integrated frames.

        :returns: per ASIC the median and the frame-to-frame rms of the level in
            keV, and how many frames lost an ASIC for having none. Empty when
            nothing was integrated.
        """
        frames = self._f["frames"]
        ok = frames["status"][:] == FrameStatus.OK
        if not ok.any():
            return {}
        levels = frames["asic_level"][:][ok].astype(np.float64)
        missing = frames["n_level_missing"][:][ok]
        with np.errstate(invalid="ignore"):
            present = np.isfinite(levels)
            median = [
                float(np.median(levels[present[:, a], a]))
                if present[:, a].any()
                else None
                for a in range(levels.shape[1])
            ]
            rms = [
                float(np.std(levels[present[:, a], a]))
                if present[:, a].sum() > 1
                else None
                for a in range(levels.shape[1])
            ]
        return {
            "median_kev": median,
            "frame_to_frame_rms_kev": rms,
            "n_frames_with_a_missing_level": int((missing > 0).sum()),
            "n_asic_levels_missing": int(missing.astype(np.int64).sum()),
        }

    def data_check_summary(self) -> dict[str, Any]:
        """Pixels the value check excluded, and frames it refused.

        The check drops the offending *pixel* and keeps the frame, so what
        matters is how many frames lost pixels and how many they lost.
        ``n_frames_refused`` is the residue: a frame with nothing left to
        integrate at all, which should not happen and is loud if it does.

        :returns: the record, empty when the run lost nothing, so a clean run
            carries no attribute rather than one full of zeros.
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

    The grid is small — 3000 trains x 8 lit cells at ``npt`` 500 is 48 MB as f4,
    against 0.93 GB for the AGIPD per-pulse grid — so it is returned whole.

    :param source: an open handle or a path to a finished output file.
    :param dtype: storage for the intensity grid.
    :returns: the grid, carrying ``data_check`` so a DAMNIT user can see why a
        slot is empty — see :func:`analysis.common.writer.per_label` — and
        ``units``, so nothing downstream has to guess what it counts.
    """
    grid = per_label(
        source,
        label_column="cellId",
        label_dim="cellId",
        dtype=dtype,
        carry_attrs=("data_check", "units"),
    )
    # A keV file carries no attribute, which per_label renders as "".
    grid.attrs["units"] = grid.attrs.get("units") or KEV
    return grid


def file_units(source: Any) -> str:
    """What ``signal`` counts in a finished file: :data:`PHOTONS` or :data:`KEV`.

    :param source: an open handle or a path.
    """
    with as_handle(source) as handle:
        stored = handle["provenance"].attrs.get("units")
    return str(stored) if stored else KEV
