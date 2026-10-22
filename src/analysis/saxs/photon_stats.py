"""Photon-count histograms per frame and ring, the XPCS quality product.

Written by the ``agipd_saxs`` pass from the same read of proc as the frame table
and the window sums (``AgipdSaxsConfig.photon_stats``, on by default), or by
:func:`run_photon_stats` alone for a run whose other products are complete. For
every frame and each narrow ring tiling the plain XPCS ROIs, over the frame's
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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common.config import restore_by_name, state_by_name
from analysis.common.cpu import file_sha256, set_thread_env
from analysis.common.masks import frame_bad
from analysis.common.plan import Block
from analysis.common.run import base_provenance, fan_out
from analysis.common.status import FrameStatus
from analysis.common.writer import FrameTableWriter, IncompleteRun
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
    "ring_labels",
    "ring_labels_from",
    "run_photon_stats",
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
    from analysis.triage.xpcs import BRAGG_WINDOWS_NM, roi_edges_nm, roi_kinds

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
        for w_lo, w_hi in BRAGG_WINDOWS_NM:
            if lo < w_hi and hi > w_lo:
                raise ValueError(f"ring {lo}-{hi} overlaps Bragg window {w_lo}-{w_hi}")
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


def ring_labels(cfg: PhotonStatsConfig, geometry: Any = None) -> tuple[np.ndarray, str]:
    """Each pixel's ring (−1: none) in the SAXS pass's geometry, and the operator sha.

    The q of a pixel is that of its centre, from the same `build_operator` the
    integrator uses, with its beam centre and pixel corners.
    """
    from analysis.saxs.operator import build_operator, geometry_from_config

    geom = geometry if geometry is not None else geometry_from_config(cfg.saxs)
    op, ai = build_operator(geom, cfg.saxs)
    return ring_labels_from(ai, cfg), op.sha256


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


@dataclass(slots=True)
class _State:
    cfg: PhotonStatsConfig
    labels: np.ndarray
    static_bad: np.ndarray
    ring_n: np.ndarray
    in_ring: np.ndarray
    detector: Any


_STATE: _State | None = None


def init(cfg: PhotonStatsConfig, labels, static_bad, run_dir: str | None = None):
    """Process initializer: open the run, keep the ring map and the static mask."""
    set_thread_env()
    from extra_data import RunDirectory, open_run
    from extra_data.components import AGIPD1M

    dc = (
        RunDirectory(run_dir) if run_dir
        else open_run(cfg.saxs.proposal, cfg.saxs.run, data="proc")
    )  # fmt: skip
    detector = AGIPD1M(
        dc, detector_name=cfg.saxs.detector_name, min_modules=cfg.saxs.min_modules
    )
    labels = np.asarray(labels, dtype=np.int16)
    in_ring = labels >= 0
    global _STATE
    _STATE = _State(
        cfg=cfg,
        labels=labels,
        static_bad=np.asarray(static_bad, dtype=bool),
        ring_n=np.bincount(labels[in_ring].astype(np.intp), minlength=cfg.npt),
        in_ring=in_ring,
        detector=detector,
    )


def process_block(block: Block) -> PhotonBlockResult:
    """Read one block's trains and histogram every frame (the standalone pass).

    Identity comes from the reader's coordinates; a train whose labels or frame
    count disagree with the plan is ``LABEL_MISMATCH``, a frame that is not integer
    photon counts ``DATA_CHECK_FAILED``, exactly as in the integrator, whose
    worker does the same from its own read (``analysis.saxs.worker``).
    """
    if _STATE is None:
        raise RuntimeError("photon_stats.init has not run in this process")
    from extra_data import by_id

    from analysis.saxs.sparse import frame_data_status

    st = _STATE
    k_max = st.cfg.k_max
    result = empty_result(block, st.cfg.npt, k_max + 2)
    timings = {"read_data": 0.0, "read_mask": 0.0, "histogram": 0.0}
    offset = 0
    for train_id, expected in zip(block.train_ids, block.expected_frames, strict=True):
        rows = slice(offset, offset + expected)
        offset += expected
        selected = st.detector.select_trains(by_id[[train_id]])
        started = time.perf_counter()
        data = selected["image.data"].ndarray(decompress_threads=1)
        timings["read_data"] += time.perf_counter() - started
        started = time.perf_counter()
        mask = selected["image.mask"].ndarray(decompress_threads=1)
        timings["read_mask"] += time.perf_counter() - started
        key = selected["image.data"]
        train_ids = key.train_id_coordinates()
        if (
            train_ids.size != expected
            or data.shape[1] != expected
            or mask.shape[1] != expected
            or not np.all(train_ids == train_id)
        ):
            result.status[rows] = FrameStatus.LABEL_MISMATCH
            continue
        result.train_id[rows] = train_ids
        result.pulse_id[rows] = key.pulse_id_coordinates()
        result.cell_id[rows] = key.cell_id_coordinates()
        result.bits_present |= int(np.bitwise_or.reduce(mask, axis=None))
        started = time.perf_counter()
        for frame in range(expected):
            row = rows.start + frame
            counts = data[:, frame].reshape(-1)
            if frame_data_status(counts) != FrameStatus.OK:
                result.status[row] = FrameStatus.DATA_CHECK_FAILED
                continue
            bad = frame_bad(mask[:, frame], st.cfg.saxs.mask_bits, st.static_bad)
            histogram_frame(
                result, row, counts, bad, st.labels, st.ring_n, st.in_ring, k_max
            )
        timings["histogram"] += time.perf_counter() - started
    result.timings = timings
    return result


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


def run_photon_stats(
    cfg: PhotonStatsConfig,
    *,
    dc: Any = None,
    geometry: Any = None,
    static_bad: np.ndarray | None = None,
    run_dir: Path | None = None,
    output_path: Path | None = None,
    pool_factory: Any = None,
) -> Path:
    """Write only a run's photon statistics; resume a file of the same hash.

    For a run whose frame table and window sums are complete. Otherwise the
    ``agipd_saxs`` pass writes the statistics from its own read
    (``AgipdSaxsConfig.photon_stats``). A complete file is left untouched, and
    the file is locked while written.

    :param geometry: an AGIPD geometry object instead of ``cfg.saxs.geometry_file``
        (the tests' synthetic one).
    :param static_bad: the static mask instead of `build_static_bad` (tests).
    :raises IncompleteRun: some frame did not reach ``OK`` and
        ``cfg.saxs.allow_incomplete`` is False.
    """
    from analysis.common.lock import output_lock
    from analysis.saxs.masks import build_static_bad
    from analysis.saxs.plan import build_plan

    set_thread_env()
    started_at, started = time.time(), time.perf_counter()
    setup: dict[str, float] = {}
    t = time.perf_counter()
    labels, op_sha = ring_labels(cfg, geometry)
    setup["rings"] = time.perf_counter() - t
    path = Path(output_path) if output_path else cfg.output_file

    with output_lock(path):
        if not cfg.overwrite and not photon_stats_reasons(path, cfg, labels):
            log.info("r%d: %s is complete; nothing written", cfg.saxs.run, path)
            return _check_ok(cfg, path)
        t = time.perf_counter()
        plan = build_plan(cfg.saxs, dc=dc)
        setup["plan"] = time.perf_counter() - t
        if static_bad is None:
            static_bad = build_static_bad(cfg.saxs).bad
        with PhotonStatsWriter.open_or_create(cfg, plan, path) as out:
            out.store_rings(cfg, labels, op_sha)
            todo = [b for b in plan.blocks if not out.block_complete(b)]
            log.info(
                "r%d: %d of %d blocks to do", cfg.saxs.run, len(todo), len(plan.blocks)
            )
            totals = fan_out(
                out,
                todo,
                process=process_block,
                n_workers=cfg.saxs.workers,
                initializer=init,
                initargs=(cfg, labels, static_bad, str(run_dir) if run_dir else None),
                pool_factory=pool_factory,
            )
            out.finalise(
                photon_provenance(
                    cfg,
                    plan,
                    out,
                    op_sha,
                    started_at=started_at,
                    started=started,
                    totals=totals,
                    setup_timings=setup,
                )  # fmt: skip
            )
    return _check_ok(cfg, path)


def _check_ok(cfg: PhotonStatsConfig, path: Path) -> Path:
    """Raise ``IncompleteRun`` for a file holding frames that are not ``OK``."""
    import h5py

    from analysis.common.writer import status_counts

    with h5py.File(path, "r") as handle:
        summary = status_counts(handle["frames/status"][:], nonzero_only=True)
    if set(summary) - {FrameStatus.OK.name} and not cfg.saxs.allow_incomplete:
        raise IncompleteRun(
            f"r{cfg.saxs.run}: not every frame was histogrammed: {summary}"
        )
    return path
