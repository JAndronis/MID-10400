"""extra-speckle's pixel-outlier mask, computed from the hit trains in the store.

``extra_speckle.stats.tools.mask_outlier`` removes every pixel whose RMS over
frames of its per-frame mean photon count lies outside given multiples of the
median of its q bin. The mask is a property of the run's pixels, so it is made
once from every usable hit train and folded into the ring labels like the static
mask. The store is not touched.

Per frame position ``f`` and envelope pixel ``p``, over the trains ``t``::

    S[f,p] = Σ_t x[t,f,p] · g[t,f,p]        g: image.mask does not flag p in that frame
    C[f,p] = Σ_t g[t,f,p]
    s[p]   = sqrt(nanmean_f (S/C)²)         S/C is NaN where C = 0
    ratio  = s / nanmedian(s over the pixels of p's q interval)
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.common.cpu import set_thread_env
from analysis.common.plan import Block
from analysis.common.status import FrameStatus
from analysis.xpcs.rings import Rings, sha256_of_array
from analysis.xpcs.store import FlagBase, RingPhotonStore, TrainPhotons

__all__ = [
    "BLOCK_LIMIT",
    "FrameSums",
    "OutlierFailed",
    "OutlierMask",
    "OutlierSink",
    "accumulate",
    "outlier_mask",
    "q_interval",
]

#: Trains a worker sums into uint16: 257 × 255 (the store's largest count) < 2¹⁶.
BLOCK_LIMIT = 257


class OutlierFailed(RuntimeError):
    """A block of the statistic failed; a mask without it would depend on which."""


@dataclass(frozen=True, slots=True)
class OutlierMask:
    """The applied mask over the envelope, and the statistic behind it."""

    ratio: np.ndarray  # float32 (n_env,): NaN where the pixel has no value
    mask: np.ndarray  # bool (n_env,)
    thresholds: tuple[float, float]
    q_bins: int
    n_trains: int

    @property
    def sha256(self) -> str:
        return sha256_of_array(self.mask)

    def apply(self, rings: Rings) -> Rings:
        """``rings`` with the outliers set to -1."""
        label = rings.label.copy()
        label[self.mask] = -1
        label.flags.writeable = False
        return Rings(
            rings.bounds_nm,
            label,
            rings.n_rings,
            hashlib.sha256((rings.sha256 + self.sha256).encode()).hexdigest(),
        )


@dataclass(slots=True)
class FrameSums:
    """Photons and flags of some trains, per frame position and envelope pixel.

    ``flag_cells`` counts, per frame position, how often each memory cell sat
    there (its flag base is flagged in all those trains); ``flag_diff`` holds the
    signed corrections of the frames' listed differences from that base.
    """

    photons: np.ndarray  # (P, n_env) uint16 in a worker, uint32 once added up
    n_trains: int
    flag_cells: dict[tuple[int, int], int]
    flag_diff_key: np.ndarray  # int64 f·n_env + p, unique
    flag_diff: np.ndarray  # int32
    bits_present: int = 0
    timings: dict[str, float] = field(default_factory=dict)


class _BaseMap:
    """The flag base as a dense ``(cell, pixel)`` bool table; unknown cells empty."""

    def __init__(self, base: FlagBase, n_env: int) -> None:
        n = base.cells.size
        self.table = np.zeros((n + 1, n_env), dtype=bool)
        for i in range(n):
            self.table[i, base.pixel[base.ptr[i] : base.ptr[i + 1]]] = True
        self.lookup = np.full(2**16, n, dtype=np.int64)
        self.lookup[base.cells] = np.arange(n)

    def rows(self, cell_ids: np.ndarray) -> np.ndarray:
        return self.lookup[np.asarray(cell_ids, dtype=np.int64)]


def accumulate(
    trains: Iterable[TrainPhotons], base: FlagBase, n_env: int, n_frames: int
) -> FrameSums:
    """Sum at most :data:`BLOCK_LIMIT` trains' unflagged photons, frame by frame."""
    table = _BaseMap(base, n_env)
    photons = np.zeros((n_frames, n_env), dtype=np.uint16)
    flat = photons.reshape(-1)
    cells: dict[tuple[int, int], int] = {}
    keys, signs = [], []
    n = 0
    for train in trains:
        if train.n_frames != n_frames:
            raise ValueError(
                f"train {train.train_id} has {train.n_frames} frames, not {n_frames}"
            )
        n += 1
        if n > BLOCK_LIMIT:
            raise ValueError(f"more than {BLOCK_LIMIT} trains in one block")
        rows = table.rows(train.cell_ids)
        for f, cell in enumerate(train.cell_ids.tolist()):
            cells[(f, cell)] = cells.get((f, cell), 0) + 1

        frame = np.repeat(np.arange(n_frames), np.diff(train.frame_ptr))
        event = frame * n_env + train.pixel.astype(np.int64)
        flag_frame = np.repeat(np.arange(n_frames), np.diff(train.flag_ptr))
        listed = flag_frame * n_env + train.flag_pixel.astype(np.int64)
        in_listed = np.zeros(event.size, dtype=bool)
        at = np.searchsorted(event, listed)
        found = at < event.size
        found[found] = event[at[found]] == listed[found]
        in_listed[at[found]] = True
        flagged = table.table[rows[frame], train.pixel] ^ in_listed
        keep = ~flagged
        flat[event[keep]] += train.count[keep]

        on_base = table.table[rows[flag_frame], train.flag_pixel]
        keys.append(listed)
        signs.append(np.where(on_base, -1, 1).astype(np.int32))

    key = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    sign = np.concatenate(signs) if signs else np.zeros(0, np.int32)
    unique, inverse = np.unique(key, return_inverse=True)
    diff = np.zeros(unique.size, np.int32)
    np.add.at(diff, inverse, sign)
    return FrameSums(photons, n, cells, unique, diff)


def q_interval(q: np.ndarray, q_range: tuple[float, float], q_bins: int) -> np.ndarray:
    """extra-speckle's q intervals: index ``i`` for ``c_i < q ≤ c_{i+1}``, else -1.

    ``c`` are the centres of ``q_bins`` equal bins over ``q_range``, as
    ``saxs.get``'s 300 pyFAI bins over the detector.
    """
    low, high = q_range
    width = (high - low) / q_bins
    centres = low + (np.arange(q_bins) + 0.5) * width
    index = np.searchsorted(centres, q, side="left") - 1
    with np.errstate(invalid="ignore"):
        inside = (index >= 0) & (index <= q_bins - 2) & np.isfinite(q)
    return np.where(inside, index, -1)


def outlier_mask(
    sums: FrameSums,
    base: FlagBase,
    candidate: np.ndarray,
    q_env: np.ndarray,
    q_range: tuple[float, float],
    thresholds: tuple[float, float],
    q_bins: int,
) -> OutlierMask:
    """The statistic and the mask from the summed photons.

    :param candidate: envelope pixels that may hold a value (not statically
        bad); the others are NaN and in no median.
    :param q_env: each envelope pixel's q, in the unit of ``q_range``.
    """
    n_frames, n_env = sums.photons.shape
    table = _BaseMap(base, n_env)
    diff = np.zeros(n_frames * n_env, np.int32)
    diff[sums.flag_diff_key] = sums.flag_diff
    diff = diff.reshape(n_frames, n_env)
    square = np.zeros(n_env, np.float64)
    finite = np.zeros(n_env, np.int64)
    for f in range(n_frames):
        flagged = diff[f].astype(np.int64)
        for (frame, cell), times in sums.flag_cells.items():
            if frame == f:
                flagged += times * table.table[table.lookup[cell]]
        count = sums.n_trains - flagged
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums.photons[f] / count
        ok = count > 0
        square[ok] += mean[ok] ** 2
        finite += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(square / finite)
    rms[~np.asarray(candidate, bool)] = np.nan

    interval = q_interval(q_env, q_range, q_bins)
    ratio = np.full(n_env, float(np.mean(thresholds)))
    for i in np.unique(interval[interval >= 0]):
        inside = interval == i
        values = rms[inside]
        median = np.nanmedian(values) if np.isfinite(values).any() else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio[inside] = values / median
    low, high = thresholds
    with np.errstate(invalid="ignore"):
        mask = (ratio < low) | (ratio > high)
    return OutlierMask(
        ratio.astype(np.float32), mask, tuple(thresholds), q_bins, sums.n_trains
    )


class OutlierSink:
    """The ``out`` :func:`fan_out` writes through: it adds the blocks up."""

    def __init__(self, n_frames: int, n_env: int) -> None:
        self.total = FrameSums(
            np.zeros((n_frames, n_env), np.uint32),
            0,
            {},
            np.zeros(0, np.int64),
            np.zeros(0, np.int32),
        )
        self._diff = np.zeros(n_frames * n_env, np.int32)
        self.failed: list[str] = []

    def write_block(self, block: Block, result: FrameSums) -> None:
        self.total.photons += result.photons
        self.total.n_trains += result.n_trains
        for key, times in result.flag_cells.items():
            self.total.flag_cells[key] = self.total.flag_cells.get(key, 0) + times
        np.add.at(self._diff, result.flag_diff_key, result.flag_diff)

    def mark(self, block: Block, status: FrameStatus, message: str = "") -> None:
        self.failed.append(f"block {block.index}: {status.name} {message}")

    def mark_remaining(self, status: FrameStatus) -> None:
        self.failed.append(f"remaining blocks: {status.name}")

    def result(self) -> FrameSums:
        if self.failed:
            raise OutlierFailed(
                "the outlier statistic is incomplete, so no mask and no sums: "
                + "; ".join(self.failed)
            )
        key = np.flatnonzero(self._diff)
        self.total.flag_diff_key = key.astype(np.int64)
        self.total.flag_diff = self._diff[key]
        return self.total


# ── the worker ──────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class _State:
    store: RingPhotonStore
    n_frames: int
    n_env: int


_STATE: _State | None = None


def init(store_path: str | Path, extract_hash: str, n_frames: int) -> None:
    """Process initializer: open the store read-only."""
    set_thread_env()
    global _STATE
    store = RingPhotonStore.open(Path(store_path), extract_hash)
    _STATE = _State(store, n_frames, int(store.envelope().size))


def close() -> None:
    """Close the store this process opened (see :func:`analysis.xpcs.merge.close`)."""
    global _STATE
    if _STATE is not None:
        _STATE.store.close()
        _STATE = None


def process_block(block: Block) -> FrameSums:
    if _STATE is None:
        raise RuntimeError("outliers.init has not run in this process")
    started = time.perf_counter()
    store = _STATE.store
    sums = accumulate(
        (store.get(int(t)) for t in block.train_ids),
        store.base,
        _STATE.n_env,
        _STATE.n_frames,
    )
    sums.timings["outlier_sums"] = time.perf_counter() - started
    return sums
