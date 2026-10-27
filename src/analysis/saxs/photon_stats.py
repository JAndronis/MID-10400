"""Photon-count histograms per frame and ring, the XPCS quality product.

Written by the ``agipd_saxs`` pass from the same read of proc as the frame table
and the window sums (``AgipdSaxsConfig.photon_stats``, on by default). For every
frame and each narrow ring tiling the plain XPCS ROIs, over the frame's
valid pixels:

    n_valid        valid pixels
    hist[k]        pixels with k photons, k = 0 … K; hist[K+1] = more than K
    s1 = Σ k,  s2 = Σ k(k − 1)      exact, whatever K is
    max_k          the largest count, for the outlier rule

Raw counts only: the per-pixel weights and the contrast are computed afterwards
(`analysis.diagnostics.photon_stats`).

The geometry, the static mask and each frame's bad pixels are the SAXS pass's
own (`build_operator`, `build_static_bad`, `frame_bad`); a pixel belongs to the
ring its centre's q falls in. Frames are labelled from the reader's coordinates
and laid out as the `agipd_saxs` frame table is (`build_plan`), one row per frame,
so the two files match row for row.

Proc goes to tape: this must run before it does. ``overwrite`` replaces a file
written under another configuration; it never touches anything else.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common.config import restore_by_name, state_by_name
from analysis.common.cpu import file_sha256
from analysis.common.plan import Block
from analysis.common.run import base_provenance
from analysis.common.status import FrameStatus
from analysis.common.writer import FrameTableWriter
from analysis.saxs.config import (
    DEFAULT_PHOTON_STATS_ROOT,
    NPIX,
    SHAPE,
    AgipdSaxsConfig,
)

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "FILE_NAME",
    "K_MAX",
    "PhotonBlockResult",
    "PhotonSpec",
    "PhotonStatsConfig",
    "PhotonStatsWriter",
    "default_rings",
    "empty_result",
    "frame_stats",
    "histogram_frame",
    "photon_stats_reasons",
    "ring_labels_from",
]

log = logging.getLogger(__name__)

FILE_NAME = "agipd_photon_stats.h5"
#: Beside the per-pixel window sums, so it survives the move to tape.
DEFAULT_OUTPUT_ROOT = DEFAULT_PHOTON_STATS_ROOT
#: Counts above this go to the overflow bin; s1 and s2 stay exact.
K_MAX = 15
#: Width of the rings that tile the plain XPCS ROIs [nm⁻¹].
RING_WIDTH_NM = 0.025
#: Bump when what a frame record holds changes.
PHOTON_STATS_VERSION = 1


def default_rings(width: float = RING_WIDTH_NM) -> tuple[tuple[float, float], ...]:
    """Rings of about ``width`` tiling each plain XPCS ROI, clear of Bragg windows."""
    from analysis.triage.xpcs import roi_edges_nm, roi_kinds
    from analysis.xpcs.config import touches_bragg

    edges = roi_edges_nm()
    rings = []
    for (lo, hi), kind in zip(edges, roi_kinds(edges), strict=True):
        if kind != "plain":
            continue
        n = max(int(round((hi - lo) / width)), 1)
        bounds = np.linspace(lo, hi, n + 1)
        rings += [
            (float(a), float(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        ]
    for lo, hi in rings:
        if touches_bragg(lo, hi):
            raise ValueError(f"ring {lo}-{hi} overlaps a Bragg window")
    return tuple(rings)


@dataclass(frozen=True, slots=True)
class PhotonStatsConfig:
    """The run's configuration: the SAXS pass's geometry and masks, plus the rings.

    Only ``saxs``'s geometry, mask, detector and module fields matter here; its
    ``npt``, method and outputs do not, and do not enter `config_hash`.
    """

    saxs: AgipdSaxsConfig
    rings_nm: tuple[tuple[float, float], ...] = field(default_factory=default_rings)
    k_max: int = K_MAX
    output_root: str = DEFAULT_OUTPUT_ROOT
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.k_max < 3:
            raise ValueError(f"k_max {self.k_max}: the shape checks need k ≤ 3 bins")
        lows = [lo for lo, _ in self.rings_nm]
        if lows != sorted(lows) or any(hi <= lo for lo, hi in self.rings_nm):
            raise ValueError("rings must be sorted and of positive width")

    @classmethod
    def for_saxs(cls, saxs: AgipdSaxsConfig) -> PhotonStatsConfig:
        """The statistics the ``agipd_saxs`` pass writes for ``saxs``."""
        return cls(saxs=saxs, output_root=saxs.photon_stats_root)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle by field name, so a field added later cannot shift the others."""
        return state_by_name(self)

    def __setstate__(self, state: Any) -> None:
        restore_by_name(self, state)

    @property
    def npt(self) -> int:
        """The number of rings (the `PassConfig` protocol's second axis)."""
        return len(self.rings_nm)

    @property
    def output_file(self) -> Path:
        """``{output_root}/r{run:04d}/agipd_photon_stats.h5``."""
        return Path(self.output_root) / f"r{self.saxs.run:04d}" / FILE_NAME

    @property
    def operational_fields(self) -> frozenset[str]:
        """What `config_hash` leaves out: nothing here changes a stored count."""
        return frozenset(
            {"output_root", "overwrite", "saxs.n_workers", "saxs.trains_per_block",
             "saxs.allow_incomplete"}
        )  # fmt: skip

    def config_hash(self) -> str:
        """sha256 over what changes a count: rings, K, geometry, masks, modules."""
        s = self.saxs
        payload = {
            "version": PHOTON_STATS_VERSION,
            "rings_nm": [list(r) for r in self.rings_nm],
            "k_max": self.k_max,
            "proposal": s.proposal,
            "run": s.run,
            "sdd_m": s.sdd_m,
            "photon_energy_kev": s.photon_energy_kev,
            "beam_center": s.beam_center,
            "unit": s.unit,
            "mask_bits": s.mask_bits,
            "use_asic_seams": s.use_asic_seams,
            "min_modules": s.min_modules,
            "detector_name": s.detector_name,
            "input_file_sha256": {
                name: file_sha256(path) if path else None
                for name, path in s.input_files.items()
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


# ── rings ─────────────────────────────────────────────────────────────────────


def ring_labels_from(ai: Any, cfg: PhotonStatsConfig) -> np.ndarray:
    """Each pixel's ring (−1: none) from the integrator ``build_operator`` returned."""
    q = np.asarray(ai.array_from_unit(SHAPE, "center", cfg.saxs.unit)).ravel()
    labels = np.full(NPIX, -1, dtype=np.int16)
    for r, (lo, hi) in enumerate(cfg.rings_nm):
        labels[(q >= lo) & (q < hi)] = r
    return labels


def labels_sha256(labels: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(labels, dtype=np.int16).tobytes()
    ).hexdigest()


# ── one frame ─────────────────────────────────────────────────────────────────


def frame_stats(
    counts: np.ndarray,
    bad: np.ndarray,
    labels: np.ndarray,
    ring_n: np.ndarray,
    in_ring: np.ndarray,
    k_max: int = K_MAX,
) -> tuple[np.ndarray, ...]:
    """``(n_valid, hist, s1, s2, max_k)`` of one frame, from its nonzero pixels only.

    ``ring_n`` is each ring's pixel count and ``in_ring`` ``labels >= 0``, both
    built once. The zero bin is what is left of the valid pixels, so a frame costs
    a pass over its ~1 % nonzero pixels and over its bad pixels in rings.
    """
    n_rings = ring_n.size
    nz = np.flatnonzero(counts)
    keep = nz[in_ring[nz] & ~bad[nz]]
    lab = labels[keep].astype(np.intp)
    val = counts[keep].astype(np.int64)
    hist = np.bincount(
        lab * (k_max + 2) + np.minimum(val, k_max + 1), minlength=n_rings * (k_max + 2)
    ).reshape(n_rings, k_max + 2)
    s1 = np.bincount(lab, weights=val, minlength=n_rings)
    s2 = np.bincount(lab, weights=val * (val - 1), minlength=n_rings)
    max_k = np.zeros(n_rings, dtype=np.int64)
    np.maximum.at(max_k, lab, val)
    gone = np.flatnonzero(bad & in_ring)
    n_valid = ring_n - np.bincount(labels[gone].astype(np.intp), minlength=n_rings)
    hist[:, 0] = n_valid - hist[:, 1:].sum(axis=1)
    return n_valid, hist, s1, s2, max_k


# ── the frame table ───────────────────────────────────────────────────────────


#: The per-frame, per-ring arrays besides the base frame table, name -> (dtype, has k).
RING_ARRAYS: dict[str, tuple[Any, bool]] = {
    "n_valid": (np.uint32, False),
    "hist": (np.uint32, True),
    "s1": (np.uint32, False),
    "s2": (np.uint32, False),
    "max_k": (np.uint16, False),
}


class PhotonStatsWriter(FrameTableWriter):
    """One run's photon-statistics file: the frame-table ledger plus ring arrays."""

    FRAME_VECTORS = {
        "trainId": np.uint64,
        "reader_pulseId": np.uint64,
        "cellId": np.uint16,
        "status": np.uint8,
    }
    FRAME_VECTOR_SOURCES = {
        "reader_pulseId": "pulse_id",
        "cellId": "cell_id",
        "status": "status",
    }
    FRAME_MATRICES: tuple[str, ...] = ()
    EXTRA_GROUPS = ("rings",)

    @classmethod
    def _missing_columns(cls, handle) -> list[str]:
        missing = super()._missing_columns(handle)
        frames = handle.get("frames")
        return missing + sorted(
            name for name in RING_ARRAYS if frames is None or name not in frames
        )

    @classmethod
    def _create_layout(cls, handle, cfg, plan, config_hash) -> None:
        super()._create_layout(handle, cfg, plan, config_hash)
        n, n_rings, k = plan.n_frames, cfg.npt, cfg.k_max + 2
        rows = max((t.n_frames for t in plan.trains if t.n_frames), default=1)
        frames = handle["frames"]
        for name, (dtype, with_k) in RING_ARRAYS.items():
            shape = (n, n_rings, k) if with_k else (n, n_rings)
            frames.create_dataset(
                name, shape=shape, dtype=dtype, fillvalue=0,
                chunks=(min(rows, n), *shape[1:]) if n else None,
                compression="gzip", compression_opts=1,
            )  # fmt: skip
        handle.flush()

    def store_rings(self, cfg: PhotonStatsConfig, labels: np.ndarray, op_sha: str):
        group = self._f["rings"]
        for name in ("edges_nm", "labels"):
            if name in group:
                del group[name]
        group["edges_nm"] = np.asarray(cfg.rings_nm, dtype=np.float64)
        group["edges_nm"].attrs["unit"] = "nm^-1"
        group.create_dataset("labels", data=labels, compression="gzip",
                             compression_opts=1)  # fmt: skip
        group.attrs["labels_sha256"] = labels_sha256(labels)
        group.attrs["operator_sha256"] = op_sha
        group.attrs["k_max"] = cfg.k_max
        self._f.flush()

    def write_block(self, block: Block, result: Any) -> None:
        super().write_block(block, result)
        rows = block.rows()
        if rows.size == 0:
            return
        start, stop = int(rows[0]), int(rows[-1]) + 1
        frames = self._f["frames"]
        for name in RING_ARRAYS:
            frames[name][start:stop] = getattr(result, name)
        self._f.flush()

    def finalise(self, provenance: dict[str, Any]) -> None:
        """As the frame table, but the operator's sha256 lives in ``/rings``.

        This file stores no ``/operator``, and the completeness check compares
        a provenance ``operator_sha256`` with one. The ring map is what matters
        here, and :func:`photon_stats_reasons` compares it label by label.
        """
        provenance = {k: v for k, v in provenance.items() if k != "operator_sha256"}
        self._f["provenance"].attrs.pop("operator_sha256", None)
        super().finalise(provenance)


# ── workers ───────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PhotonBlockResult:
    train_id: np.ndarray
    pulse_id: np.ndarray
    cell_id: np.ndarray
    status: np.ndarray
    n_valid: np.ndarray
    hist: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    max_k: np.ndarray
    bits_present: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.status.size)


@dataclass(frozen=True)
class PhotonSpec:
    """What a worker needs to histogram frames: the ring map and the static mask.

    Handed to the ``agipd_saxs`` workers as an initarg (pickled by attribute,
    not position: no slots). ``blocks`` are the block indices still to do.
    """

    labels: np.ndarray
    static_bad: np.ndarray
    k_max: int
    blocks: frozenset[int]

    @property
    def n_rings(self) -> int:
        return int(self.labels.max()) + 1 if self.labels.size else 0

    def ring_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """``(ring_n, in_ring)`` for :func:`frame_stats`, built once per worker."""
        in_ring = self.labels >= 0
        ring_n = np.bincount(
            self.labels[in_ring].astype(np.intp), minlength=self.n_rings
        )
        return ring_n, in_ring


def empty_result(block: Block, n_rings: int, k: int) -> PhotonBlockResult:
    """A block's records, every frame ``NOT_PROCESSED`` until histogrammed."""
    n = block.n_frames
    return PhotonBlockResult(
        train_id=np.zeros(n, np.uint64),
        pulse_id=np.zeros(n, np.uint64),
        cell_id=np.zeros(n, np.uint16),
        status=np.full(n, FrameStatus.NOT_PROCESSED, np.uint8),
        n_valid=np.zeros((n, n_rings), np.uint32),
        hist=np.zeros((n, n_rings, k), np.uint32),
        s1=np.zeros((n, n_rings), np.uint32),
        s2=np.zeros((n, n_rings), np.uint32),
        max_k=np.zeros((n, n_rings), np.uint16),
    )


def histogram_frame(
    result: PhotonBlockResult,
    row: int,
    counts: np.ndarray,
    bad: np.ndarray,
    labels: np.ndarray,
    ring_n: np.ndarray,
    in_ring: np.ndarray,
    k_max: int,
) -> None:
    """Fill ``row`` of ``result`` from one frame that passed the data check."""
    n_valid, hist, s1, s2, max_k = frame_stats(
        counts, bad, labels, ring_n, in_ring, k_max
    )
    result.n_valid[row] = n_valid
    result.hist[row] = hist
    result.s1[row] = s1
    result.s2[row] = s2
    result.max_k[row] = np.minimum(max_k, np.iinfo(np.uint16).max)
    result.status[row] = FrameStatus.OK


# ── completeness ──────────────────────────────────────────────────────────────


def photon_stats_reasons(
    path: str | Path, cfg: PhotonStatsConfig, labels: np.ndarray | None
) -> list[str]:
    """Why the photon-statistics file at ``path`` is not complete; empty if it is.

    The frame table's checks (``analysis.common.complete``: hash, columns,
    finalised, no unfinished row, package versions), then the ring map: its
    edges, K and every pixel's label must be what this pass would use now.

    :param labels: the ring map built now (:func:`ring_labels_from`); ``None``
        skips that comparison.
    """
    from analysis.common.complete import frame_table_reasons

    def rings_differ(handle) -> list[str]:
        group = handle.get("rings")
        if group is None or "labels" not in group or "edges_nm" not in group:
            return ["the file stores no ring map"]
        reasons = []
        if not np.array_equal(group["edges_nm"][()], np.asarray(cfg.rings_nm)):
            reasons.append("its ring edges are not the config's")
        if int(group.attrs.get("k_max", -1)) != cfg.k_max:
            reasons.append(
                f"k_max {group.attrs.get('k_max')}, the config's {cfg.k_max}"
            )
        if labels is not None and not np.array_equal(group["labels"][()], labels):
            reasons.append("its pixel → ring map differs from the one built now")
        return reasons

    return frame_table_reasons(path, cfg, PhotonStatsWriter, rings_differ)


# ── orchestration ─────────────────────────────────────────────────────────────


def photon_provenance(
    cfg: PhotonStatsConfig, plan: Any, out: Any, op_sha: str, **kwargs: Any
) -> dict[str, Any]:
    """What a photon-statistics file records about the run that wrote it."""
    provenance = base_provenance(
        cfg.saxs, plan, out, operator_sha256=op_sha,
        input_file_sha256={k: file_sha256(v) if v else None
                           for k, v in cfg.saxs.input_files.items()},
        run_checks=plan.checks, **kwargs,
    )  # fmt: skip
    provenance["rings_nm"] = [list(r) for r in cfg.rings_nm]
    provenance["photon_stats_config"] = json.dumps(asdict(cfg), default=str)
    return provenance
