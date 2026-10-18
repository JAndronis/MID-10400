"""``xpcs_hits.h5``: the per-train integer sums, and the reader that turns them into g2.

Everything in the file is derived from the photon store and the hit set, so
changed inputs rebuild it (the previous file kept as ``xpcs_hits.prev.h5``)
rather than refuse. Rows are the hit set's, in order, and are addressed by
position: a train that gave no photons has a status and zero sums.

The file holds exact integers only. Every float is made here, from sums that
pool in any order: the pooled estimates divide summed numerators by summed
denominators (a ratio of sums, CLAUDE.md pitfall 8), and a quantity with no
photons behind it is NaN, never 1. ``Averaging.ES`` is extra-speckle's mean of
per-train ratios with its ``norm == 0 -> 1`` rule and k-bar filter, kept for
comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

from analysis.common.complete import Completeness
from analysis.common.plan import Block
from analysis.common.status import UNFINISHED, FrameStatus
from analysis.xpcs.config import DEFAULT_XPCS_ROOT
from analysis.xpcs.hits import HitSet
from analysis.xpcs.kernel import Pairing, PairStatus, TrainSums
from analysis.xpcs.rings import Rings

__all__ = [
    "BASE_CLOCK_HZ",
    "ES_KBAR_MIN",
    "SCHEMA",
    "Averaging",
    "Coverage",
    "NonUniformPattern",
    "Prelude",
    "RepRateMismatch",
    "ResultWriter",
    "XpcsHits",
    "read",
    "result_reasons",
]

log = logging.getLogger(__name__)

SCHEMA = 1
GZIP = {"compression": "gzip", "compression_opts": 1, "shuffle": True}


@dataclass(frozen=True, slots=True)
class Prelude:
    """What the merge stage decides from the store before any sum is made."""

    status: np.ndarray  # uint8 (T,): the store's status, or the pattern check's
    pairing: Pairing
    sequence: np.ndarray  # int32 (T,)
    pattern: np.ndarray  # uint32 (P,) pulse ids shared by the usable trains

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for array in (self.status, self.pairing.partner, self.pairing.status):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(np.ascontiguousarray(self.sequence).tobytes())
        digest.update(np.ascontiguousarray(self.pattern).tobytes())
        return digest.hexdigest()


def _create(handle: h5py.File, name: str, shape: tuple[int, ...], dtype, chunks=None):
    """A zero-filled dataset; HDF5 cannot chunk or compress an empty one."""
    if all(shape) and chunks is not None:
        return handle.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks, fillvalue=0, **GZIP
        )
    return handle.create_dataset(name, shape=shape, dtype=dtype, fillvalue=0)


class ResultWriter:
    """The single writer of one run's ``xpcs_hits.h5``."""

    def __init__(self, handle: h5py.File, hits: HitSet) -> None:
        self._f = handle
        self._hits = hits

    @classmethod
    def open_or_create(
        cls,
        path: Path,
        *,
        hits: HitSet,
        prelude: Prelude,
        rings: Rings,
        merge_hash: str,
        extract_hash: str,
        attrs: dict[str, Any],
        overwrite: bool = False,
    ) -> ResultWriter:
        """Resume a file made for these exact inputs, else build it afresh.

        A file made for other inputs is moved to ``xpcs_hits.prev.h5`` first: it
        is derived, so replacing it loses minutes, not data.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = {
            "schema": SCHEMA,
            "merge_hash": merge_hash,
            "hits_sha256": hits.sha256,
            "prelude_sha256": prelude.sha256(),
        }
        if path.exists():
            if not overwrite and _identity(path) == identity:
                return cls(h5py.File(path, "r+"), hits)
            os.replace(path, path.with_name("xpcs_hits.prev.h5"))
            log.info("%s was made for other inputs; kept as xpcs_hits.prev.h5", path)
        handle = h5py.File(path, "w")
        cls._layout(handle, hits, prelude, rings, extract_hash, identity, attrs)
        return cls(handle, hits)

    @staticmethod
    def _layout(handle, hits, prelude, rings, extract_hash, identity, attrs) -> None:
        n, n_rings, n_frames = len(hits), rings.n_rings, prelude.pattern.size
        for key, value in identity.items():
            handle.attrs[key] = value
        handle.attrs["extract_hash"] = extract_hash
        handle.attrs["hits_source"] = hits.source
        handle.attrs["n_rings"] = n_rings
        handle.attrs["n_frames"] = n_frames
        for key, value in attrs.items():
            handle.attrs[key] = value

        initial = prelude.status.copy()
        initial[initial == FrameStatus.OK] = FrameStatus.NOT_PROCESSED
        handle.create_dataset("hits/train_id", data=hits.train_ids)
        handle.create_dataset("hits/status", data=initial)
        handle.create_dataset("hits/partner", data=prelude.pairing.partner)
        handle.create_dataset("hits/pair_status", data=prelude.pairing.status)
        handle.create_dataset("hits/sequence", data=prelude.sequence)
        handle.create_dataset("hits/has_off", data=np.zeros(n, dtype=np.uint8))
        handle.create_dataset("pattern/pulse_ids", data=prelude.pattern)
        handle.create_dataset("rings/edges_nm", data=np.asarray(rings.edges_nm))
        handle.create_dataset("rings/label", data=rings.label, **GZIP)
        handle["rings"].attrs["sha256"] = rings.sha256

        cube = (n, n_rings, n_frames, n_frames)
        plane = (n, n_rings, n_frames)
        row = (n, n_rings)
        for name in ("on", "off"):
            _create(handle, f"sums/{name}", cube, np.int32, (1, *cube[1:]))
        for name in ("s_on", "s_off_cur", "s_off_prev"):
            _create(handle, f"sums/{name}", plane, np.int32, (min(n, 64), *plane[1:]))
        for name in ("cnt", "cnt_pair"):
            _create(handle, f"sums/{name}", row, np.int32, (min(n, 256), n_rings))
        handle.create_group("provenance")
        handle.flush()

    # ── resume ───────────────────────────────────────────────────────────────
    def block_complete(self, block: Block) -> bool:
        rows = [self._row(block, t) for t in block.train_ids]
        status = self._f["hits/status"][rows]
        return not bool(np.isin(status, [int(code) for code in UNFINISHED]).any())

    def _row(self, block: Block, train_id: int) -> int:
        return block.frames_for(train_id)[0]

    # ── writing ──────────────────────────────────────────────────────────────
    def write_block(self, block: Block, result: Any) -> None:
        for sums in result.sums:
            self._write(self._row(block, sums.train_id), sums)
        self._f.flush()

    def _write(self, row: int, sums: TrainSums) -> None:
        f = self._f
        if int(f["hits/train_id"][row]) != sums.train_id:
            raise ValueError(
                f"row {row} is train {f['hits/train_id'][row]}, not {sums.train_id}"
            )
        if int(f["hits/partner"][row]) != sums.partner_id:
            raise ValueError(
                f"train {sums.train_id}: summed against {sums.partner_id}, the "
                f"pairing says {int(f['hits/partner'][row])}"
            )
        for name in ("on", "off", "s_on", "s_off_cur", "s_off_prev", "cnt", "cnt_pair"):
            f[f"sums/{name}"][row] = getattr(sums, name)
        f["hits/has_off"][row] = sums.has_off
        f["hits/status"][row] = FrameStatus.OK

    def mark(self, block: Block, status: FrameStatus, message: str = "") -> None:
        for train_id in block.train_ids:
            self._f["hits/status"][self._row(block, train_id)] = status
        if message:
            attrs = self._f["provenance"].attrs
            errors = json.loads(attrs.get("block_errors", "{}"))
            errors[str(block.index)] = message
            attrs["block_errors"] = json.dumps(errors)
        self._f.flush()

    def mark_remaining(self, status: FrameStatus) -> None:
        """Stamp every still-unprocessed row, e.g. after a broken pool."""
        current = self._f["hits/status"][()]
        current[current == FrameStatus.NOT_PROCESSED] = status
        self._f["hits/status"][...] = current
        self._f.flush()

    def unfinished(self) -> int:
        """Rows left ``NOT_PROCESSED`` or ``WORKER_ERROR``."""
        status = self._f["hits/status"][()]
        return int(np.isin(status, [int(code) for code in UNFINISHED]).sum())

    def status_summary(self) -> dict[str, int]:
        status = self._f["hits/status"][()]
        return {
            code.name: int((status == code).sum())
            for code in FrameStatus
            if (status == code).any()
        }

    def finalise(self, provenance: dict[str, Any]) -> None:
        group = self._f["provenance"]
        for key, value in provenance.items():
            group.attrs[key] = (
                value
                if isinstance(value, str | int | float)
                else json.dumps(value, default=str)
            )
        group.attrs["status_summary"] = json.dumps(self.status_summary())
        self._f.attrs["finalised"] = 1
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> ResultWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _identity(path: Path) -> dict[str, Any] | None:
    try:
        with h5py.File(path, "r") as handle:
            return {
                key: handle.attrs.get(key)
                for key in ("schema", "merge_hash", "hits_sha256", "prelude_sha256")
            }
    except OSError:
        return None


def result_reasons(path: str | Path, hits: HitSet, merge_hash: str) -> Completeness:
    """Why the results file at ``path`` is not complete; reasons empty if it is.

    Read-only, so a complete run is answered from the file with nothing written
    (CLAUDE.md pitfall 26). The prelude is not compared here: it is a function of
    the store and the hit set, and the merge hash already names the config.
    """
    path = Path(path)
    if not path.exists():
        return Completeness(path, ("no file",))
    try:
        with h5py.File(path, "r") as handle:
            attrs = handle.attrs
            reasons = []
            if attrs.get("schema") != SCHEMA:
                reasons.append(f"schema {attrs.get('schema')}, this code's is {SCHEMA}")
            if attrs.get("merge_hash") != merge_hash:
                reasons.append("it was merged under another configuration")
            if attrs.get("hits_sha256") != hits.sha256:
                reasons.append("it was made for another hit set")
            if "finalised" not in attrs:
                reasons.append("never finalised")
            status = handle["hits/status"][()]
            for code in sorted(UNFINISHED):
                if (status == code).any():
                    reasons.append(f"{int((status == code).sum())} trains {code.name}")
            return Completeness(path, tuple(reasons))
    except (OSError, KeyError, ValueError) as error:
        return Completeness(path, (f"unreadable: {error!r}",))


# ── the reader ────────────────────────────────────────────────────────────────

#: The clock pulse ids count in: the 1.3 GHz master over 288, 4.5139 MHz, as
#: ``XrayPulses.bunch_repetition_rate`` reports it and extra-speckle's
#: ``convert_pulse_to_mhz`` uses it. A pulse-id step of 2 is 443.1 ns.
BASE_CLOCK_HZ = 1.3e9 / 288
#: extra-speckle's k-bar threshold. No source for it has been found in this
#: repository, so it is offered only in the ``Averaging.ES`` view.
ES_KBAR_MIN = 3e-3
#: extra-speckle drops the last lags of g2 unless that would leave fewer than 20.
SKIP_LAST_LAGS = 10
#: Jackknife blocks (``analysis.triage.xpcs.N_BLOCKS``).
N_BLOCKS = 50
#: Largest relative disagreement between the pulse pattern's repetition rate and
#: a DAMNIT ``rep_rate`` before the lag axis is refused. DAMNIT's own figure
#: (2.2569 MHz) agrees with the pattern to better than 0.1 %.
REP_RATE_RTOL = 0.01
#: Rows read at a time when pooling.
CHUNK = 16
#: extra-speckle bins g2 only above this many lags, in this many log bins after
#: this many unbinned ones (``g2_to_bins(50, 20, 10)``).
BIN_THRESHOLD, BIN_COUNT, BIN_SKIP = 50, 20, 10

SUM_NAMES = ("on", "off", "s_on", "s_off_cur", "s_off_prev", "cnt", "cnt_pair")


class Averaging(Enum):
    POOLED = "pooled"
    """Ratios of summed numerators and denominators over the trains."""

    ES = "es"
    """extra-speckle's: the plain mean of each train's own normalised TTCF."""


class NonUniformPattern(ValueError):
    """The pulses are not evenly spaced, so a frame lag is not a time."""


class RepRateMismatch(ValueError):
    """The pulse pattern and a given repetition rate disagree."""


@dataclass(frozen=True, slots=True)
class Coverage:
    """Where every hit train went, so none drops silently."""

    n_hits: int
    by_status: dict[str, int]
    by_pairing: dict[str, int]
    n_usable: int
    n_with_baseline: int

    @property
    def accounted(self) -> bool:
        """Whether every hit has exactly one status."""
        return sum(self.by_status.values()) == self.n_hits

    def __str__(self) -> str:
        return (
            f"{self.n_hits} hits: {self.n_usable} usable, {self.n_with_baseline} "
            f"with a baseline; status {self.by_status}; pairing {self.by_pairing}"
        )


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator / denominator``, NaN where there is nothing behind the ratio."""
    out = np.full(np.shape(numerator), np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _outer(s: np.ndarray, t: np.ndarray, count: np.ndarray) -> np.ndarray:
    """``s_a t_b / count`` over the last axis pair; zero where ``count`` is."""
    out = np.zeros((*s.shape, s.shape[-1]))
    np.divide(
        s[..., :, None] * t[..., None, :],
        count[..., None, None],
        out=out,
        where=(count > 0)[..., None, None],
    )
    return out


def _jackknife(
    numerator: np.ndarray, denominator: np.ndarray, n_blocks: int = N_BLOCKS
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled ratio and its leave-one-block-out standard error.

    The blocks are contiguous in train order, so a slow drift inflates the
    error instead of hiding in it. Same estimator as
    ``analysis.triage.xpcs.jackknife_g2``, which takes the same pooled sums.
    """
    n = numerator.shape[0]
    value = _safe_divide(numerator.sum(0), denominator.sum(0))
    if n < 2:
        return value, np.full_like(value, np.nan)
    blocks = np.array_split(np.arange(n), min(n_blocks, n))
    num_b = np.stack([numerator[b].sum(0) for b in blocks])
    den_b = np.stack([denominator[b].sum(0) for b in blocks])
    loo = _safe_divide(numerator.sum(0) - num_b, denominator.sum(0) - den_b)
    count = len(blocks)
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(
            (count - 1) / count * np.nansum((loo - np.nanmean(loo, axis=0)) ** 2, 0)
        )
    sigma[~np.isfinite(value)] = np.nan
    return value, sigma


class XpcsHits:
    """A run's hit-train sums, read on demand; every float is derived here."""

    def __init__(self, path: str | Path, rep_rate_mhz: float | None = None) -> None:
        self.path = Path(path)
        self.rep_rate_mhz = rep_rate_mhz
        with h5py.File(self.path, "r") as f:
            self.train_ids: np.ndarray = f["hits/train_id"][()]
            self.status: np.ndarray = f["hits/status"][()]
            self.partner_ids: np.ndarray = f["hits/partner"][()]
            self.pair_status: np.ndarray = f["hits/pair_status"][()]
            self.has_off: np.ndarray = f["hits/has_off"][()].astype(bool)
            self.pulse_ids: np.ndarray = f["pattern/pulse_ids"][()]
            self.edges_nm: np.ndarray = f["rings/edges_nm"][()]
            self.n_rings = int(f.attrs["n_rings"])
            self.n_frames = int(f.attrs["n_frames"])
            self.attrs: dict[str, Any] = {k: v for k, v in f.attrs.items()}

    # ── what is in it ────────────────────────────────────────────────────────
    @property
    def roi_q(self) -> np.ndarray:
        """Ring centres in Å⁻¹, the unit DAMNIT's ``roi_q`` uses (ring numbers
        for rings that came from a masks file)."""
        if self.edges_nm.size < 2:
            return np.arange(self.n_rings, dtype=np.float64)
        return 0.5 * (self.edges_nm[:-1] + self.edges_nm[1:]) / 10.0

    @property
    def usable(self) -> np.ndarray:
        """Rows whose sums exist."""
        return self.status == FrameStatus.OK

    @property
    def period_s(self) -> float:
        """Seconds between consecutive frames, from the pulse ids.

        :raises NonUniformPattern: the pulses are not evenly spaced.
        :raises RepRateMismatch: ``rep_rate_mhz`` was given and disagrees.
        """
        if self.pulse_ids.size < 2:
            raise NonUniformPattern("fewer than two pulses: there is no lag")
        steps = np.unique(np.diff(self.pulse_ids.astype(np.int64)))
        if steps.size != 1:
            raise NonUniformPattern(
                f"pulse-id steps {steps.tolist()}: a frame lag is not one time"
            )
        period = float(steps[0]) / BASE_CLOCK_HZ
        if self.rep_rate_mhz is not None:
            implied = 1e-6 / period
            if abs(implied - self.rep_rate_mhz) > REP_RATE_RTOL * self.rep_rate_mhz:
                raise RepRateMismatch(
                    f"the pulse pattern implies {implied:.4f} MHz, the given "
                    f"repetition rate is {self.rep_rate_mhz:.4f} MHz"
                )
        return period

    def coverage(self) -> Coverage:
        """How many hits are usable, paired, and why the rest are not."""
        by_status = {
            FrameStatus(int(code)).name: int((self.status == code).sum())
            for code in np.unique(self.status)
        }
        by_pairing = {
            PairStatus(int(code)).name: int((self.pair_status == code).sum())
            for code in np.unique(self.pair_status)
        }
        return Coverage(
            n_hits=int(self.train_ids.size),
            by_status=by_status,
            by_pairing=by_pairing,
            n_usable=int(self.usable.sum()),
            n_with_baseline=int((self.usable & self.has_off).sum()),
        )

    # ── reading the sums ─────────────────────────────────────────────────────
    def _rows(self, trains: np.ndarray | None, *, baseline: bool = False) -> np.ndarray:
        """Row numbers of the usable trains in ``trains`` (all when ``None``)."""
        wanted = self.usable & (self.has_off if baseline else True)
        if trains is not None:
            trains = np.asarray(trains, dtype=np.uint64)
            unknown = trains[~np.isin(trains, self.train_ids)]
            if unknown.size:
                raise ValueError(
                    f"{unknown.size} train(s) are not in the hit set "
                    f"(first: {unknown[:5].tolist()})"
                )
            wanted = wanted & np.isin(self.train_ids, trains)
        return np.flatnonzero(wanted)

    def _chunks(self, rows: np.ndarray) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
        """The sums of ``rows``, a few at a time, as float64."""
        with h5py.File(self.path, "r") as f:
            for start in range(0, rows.size, CHUNK):
                part = rows[start : start + CHUNK]
                yield (
                    part,
                    {
                        name: f[f"sums/{name}"][part.tolist()].astype(np.float64)
                        for name in SUM_NAMES
                    },
                )

    # ── pooled maps ──────────────────────────────────────────────────────────
    def ttcf(
        self,
        averaging: Averaging = Averaging.POOLED,
        trains: np.ndarray | None = None,
        kbar_min: float | None = ES_KBAR_MIN,
    ) -> xr.Dataset:
        """The two-time maps ``raw``, ``off`` and ``c2 = raw − off``.

        ``Averaging.POOLED``: ``raw`` is Σ on over Σ s_a s_b / cnt, over all the
        usable trains; ``off`` the same for the baseline, symmetrised after the
        sums; ``raw_paired`` is ``raw`` over the trains that have a baseline, so
        that ``c2`` subtracts like from like. Anything with no photons behind it
        is NaN.

        ``Averaging.ES``: each train normalised on its own (``norm == 0 → 1``),
        entries with k-bar under ``kbar_min`` set to NaN, then the mean over
        trains. ``kbar_min`` is extra-speckle's unsourced default; ``None``
        disables the filter.
        """
        if averaging is Averaging.ES:
            return self._ttcf_es(trains, kbar_min)
        n_frames, n_rings = self.n_frames, self.n_rings
        shape = (n_rings, n_frames, n_frames)
        raw_num, raw_den, pair_num, pair_den, off_num, off_den = (
            np.zeros(shape) for _ in range(6)
        )
        rows = self._rows(trains)
        for part, c in self._chunks(rows):
            den_on = _outer(c["s_on"], c["s_on"], c["cnt"])
            raw_num += c["on"].sum(0)
            raw_den += den_on.sum(0)
            base = self.has_off[part]
            if base.any():
                den_off = _outer(c["s_off_cur"], c["s_off_prev"], c["cnt_pair"])
                pair_num += c["on"][base].sum(0)
                pair_den += den_on[base].sum(0)
                off_num += c["off"][base].sum(0)
                off_den += den_off[base].sum(0)
        raw = _safe_divide(raw_num, raw_den)
        paired = _safe_divide(pair_num, pair_den)
        off = _safe_divide(
            off_num + off_num.swapaxes(-1, -2), off_den + off_den.swapaxes(-1, -2)
        )
        return self._map_dataset(
            {"raw": raw, "raw_paired": paired, "off": off, "c2": paired - off},
            rows,
            Averaging.POOLED,
        )

    def _map_dataset(
        self, variables: dict[str, np.ndarray], rows: np.ndarray, averaging: Averaging
    ) -> xr.Dataset:
        frames = np.arange(self.n_frames)
        dataset = xr.Dataset(
            {
                name: (("roi_q", "frame_1", "frame_2"), array)
                for name, array in variables.items()
            },
            coords={"roi_q": self.roi_q, "frame_1": frames, "frame_2": frames},
        )
        dataset.attrs.update(
            averaging=averaging.value,
            n_trains=int(rows.size),
            n_with_baseline=int(self.has_off[rows].sum()),
        )
        return dataset

    def _normalised(self, c: dict[str, Any], rows: np.ndarray):
        """extra-speckle's per-train normalisation of a chunk (``convert_TTCF``).

        :returns: ``(raw, off, kbar)``; ``off`` is NaN for a train with no
            baseline. ``norm == 0 → 1`` as there, which turns a frame with no
            photons into a 0 entry rather than an undefined one.
        """
        cnt, cnt_pair = c["cnt"], c["cnt_pair"]
        with np.errstate(invalid="ignore", divide="ignore"):
            kbar = c["s_on"] / cnt[..., None]
            norm = kbar[..., :, None] * kbar[..., None, :]
            norm[norm == 0] = 1.0
            raw = c["on"] / (norm * cnt[..., None, None])
            cur = c["s_off_cur"] / cnt_pair[..., None]
            prev = c["s_off_prev"] / cnt_pair[..., None]
            norm_off = cur[..., :, None] * prev[..., None, :]
            norm_off[norm_off == 0] = 1.0
            off = c["off"] / (norm_off * cnt_pair[..., None, None])
            off = 0.5 * (off + off.swapaxes(-1, -2))
        off[~self.has_off[rows]] = np.nan
        return raw, off, kbar

    def _ttcf_es(self, trains, kbar_min) -> xr.Dataset:
        shape = (self.n_rings, self.n_frames, self.n_frames)
        total_raw, total_off = np.zeros(shape), np.zeros(shape)
        n_raw, n_off = np.zeros(shape), np.zeros(shape)
        rows = self._rows(trains)
        for part, c in self._chunks(rows):
            raw, off, kbar = self._normalised(c, part)
            if kbar_min is not None:
                lit = kbar > kbar_min
                keep = lit[..., :, None] & lit[..., None, :]
                raw[~keep] = np.nan
                off[~keep] = np.nan
            for value, total, count in (
                (raw, total_raw, n_raw),
                (off, total_off, n_off),
            ):
                finite = np.isfinite(value)
                total += np.where(finite, value, 0.0).sum(0)
                count += finite.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_mean = np.where(n_raw > 0, total_raw / n_raw, np.nan)
            off_mean = np.where(n_off > 0, total_off / n_off, np.nan)
        dataset = self._map_dataset(
            {"raw": raw_mean, "off": off_mean, "c2": raw_mean - off_mean},
            rows,
            Averaging.ES,
        )
        dataset.attrs["kbar_min"] = -1.0 if kbar_min is None else kbar_min
        return dataset

    # ── per train ────────────────────────────────────────────────────────────
    def kbar(self, trains: np.ndarray | None = None) -> xr.DataArray:
        """Mean photons per pixel per frame and ring, ``(trainId, frame, roi_q)``."""
        rows = self._rows(trains)
        out = np.full((rows.size, self.n_frames, self.n_rings), np.nan)
        for part, c in self._chunks(rows):
            at = np.searchsorted(rows, part)
            out[at] = _safe_divide(c["s_on"], c["cnt"][..., None]).swapaxes(1, 2)
        return xr.DataArray(
            out,
            dims=("trainId", "frame", "roi_q"),
            coords={
                "trainId": self.train_ids[rows],
                "frame": np.arange(self.n_frames),
                "roi_q": self.roi_q,
            },
            name="kbar",
        )

    def per_train(self, trains: np.ndarray | None = None) -> xr.Dataset:
        """Each train's own normalised maps, in extra-speckle's layout and rules.

        ``TTCF_raw`` and ``TTCF_off`` are ``(trainId, roi_q, frame_1, frame_2)``
        float32; a train without a baseline has a NaN ``TTCF_off``. Large: pass
        ``trains``.
        """
        rows = self._rows(trains)
        shape = (rows.size, self.n_rings, self.n_frames, self.n_frames)
        raw_all, off_all = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        for part, c in self._chunks(rows):
            at = np.searchsorted(rows, part)
            raw, off, _ = self._normalised(c, part)
            raw_all[at], off_all[at] = raw, off
        frames = np.arange(self.n_frames)
        coords = {
            "trainId": self.train_ids[rows],
            "roi_q": self.roi_q,
            "frame_1": frames,
            "frame_2": frames,
        }
        dims = ("trainId", "roi_q", "frame_1", "frame_2")
        return xr.Dataset(
            {"TTCF_raw": (dims, raw_all), "TTCF_off": (dims, off_all)}, coords=coords
        )

    # ── g2 ───────────────────────────────────────────────────────────────────
    def lags(self, skip_last: int = SKIP_LAST_LAGS) -> np.ndarray:
        """Lags (in frames) g2 is evaluated at: extra-speckle's ``1 … P − 1 − skip``."""
        skip = skip_last if self.n_frames - skip_last >= 20 else 0
        return np.arange(1, self.n_frames - skip)

    def _lag_terms(self, trains, lags: np.ndarray):
        """Per-train numerators and denominators of g2 and of the baseline per lag.

        :returns: rows, then ``(n, R, L)`` arrays ``on, den_on`` for every usable
            row and ``off, den_off`` for rows with a baseline (zeros elsewhere),
            and which rows those are.
        """
        rows = self._rows(trains)
        shape = (rows.size, self.n_rings, lags.size)
        on, den_on, off, den_off = (np.zeros(shape) for _ in range(4))

        def diagonals(matrix: np.ndarray, both: bool = False) -> np.ndarray:
            return np.stack(
                [
                    np.diagonal(matrix, k, -2, -1).sum(-1)
                    + (np.diagonal(matrix, -k, -2, -1).sum(-1) if both else 0)
                    for k in lags
                ],
                axis=-1,
            )

        for part, c in self._chunks(rows):
            at = np.searchsorted(rows, part)
            on[at] = diagonals(c["on"])
            den_on[at] = diagonals(_outer(c["s_on"], c["s_on"], c["cnt"]))
            off[at] = diagonals(c["off"], both=True)
            den_off[at] = diagonals(
                _outer(c["s_off_cur"], c["s_off_prev"], c["cnt_pair"]), both=True
            )
        return rows, on, den_on, off, den_off, self.has_off[rows]

    def g2(
        self,
        averaging: Averaging = Averaging.POOLED,
        trains: np.ndarray | None = None,
        binned: bool = True,
        skip_last: int = SKIP_LAST_LAGS,
        n_blocks: int = N_BLOCKS,
        kbar_min: float | None = ES_KBAR_MIN,
    ) -> xr.Dataset:
        """g2 against lag time, with a jackknife error.

        Pooled: ``g2(τ) = Σ_t Σ_a on_t(a, a+τ) / Σ_t Σ_a s_t(a) s_t(a+τ) / cnt_t``
        over the usable trains, so each pair of frames weighs with its photon
        statistics. ``baseline`` is the same for the off-correlation over the
        trains that have one, and ``c2 = g2_paired − baseline`` over those same
        trains. ``sigma`` is the jackknife error of ``c2`` over ``n_blocks``
        contiguous blocks of trains, ``sigma_g2`` that of ``g2``. Binning sums
        numerators and denominators over a bin's lags before dividing.

        ``Averaging.ES`` is extra-speckle's: the diagonals of the mean
        baseline-subtracted map (``ttcf(Averaging.ES)``), then medians over log
        bins. It has no error.

        :param skip_last: lags dropped from the end, as extra-speckle does.
        :raises NonUniformPattern: the pulses are not evenly spaced.
        :raises RepRateMismatch: the pulse pattern disagrees with ``rep_rate_mhz``.
        """
        lags = self.lags(skip_last)
        if lags.size == 0:
            return self._empty_g2(averaging)
        period = self.period_s
        if averaging is Averaging.ES:
            return self._g2_es(trains, lags, period, binned, kbar_min)

        rows, on, den_on, off, den_off, has_off = self._lag_terms(trains, lags)
        groups = (
            _lag_bins(lags) if binned else [np.array([i]) for i in range(lags.size)]
        )
        pooled = lambda a: np.stack([a[..., g].sum(-1) for g in groups], -1)  # noqa: E731
        on, den_on, off, den_off = (pooled(a) for a in (on, den_on, off, den_off))
        g2, sigma_g2 = _jackknife(on, den_on, n_blocks)
        g2_paired, _ = _jackknife(on[has_off], den_on[has_off], n_blocks)
        baseline, _ = _jackknife(off[has_off], den_off[has_off], n_blocks)
        c2, sigma = self._c2_error(on[has_off], den_on[has_off], off[has_off],
                                   den_off[has_off], n_blocks)  # fmt: skip
        centre = np.array([lags[g].mean() for g in groups])
        dims = ("roi_q", "tau")
        dataset = xr.Dataset(
            {
                "g2": (dims, g2),
                "sigma_g2": (dims, sigma_g2),
                "g2_paired": (dims, g2_paired),
                "baseline": (dims, baseline),
                "c2": (dims, c2),
                "sigma": (dims, sigma),
            },
            coords={
                "roi_q": self.roi_q,
                "tau": centre * period,
                "lag": ("tau", centre),
            },
        )
        dataset.attrs.update(
            averaging="pooled",
            n_trains=int(rows.size),
            n_with_baseline=int(has_off.sum()),
            period_s=period,
            binned=int(binned and lags.size > BIN_THRESHOLD),
        )
        return dataset

    def _empty_g2(self, averaging: Averaging) -> xr.Dataset:
        """g2 of a run with fewer than two frames: no lag, so nothing to report."""
        none = np.zeros((self.n_rings, 0))
        names = ["g2"] if averaging is Averaging.ES else [
            "g2", "sigma_g2", "g2_paired", "baseline", "c2", "sigma"
        ]  # fmt: skip
        return xr.Dataset(
            {name: (("roi_q", "tau"), none) for name in names},
            coords={"roi_q": self.roi_q, "tau": np.zeros(0)},
            attrs={"averaging": averaging.value},
        )

    @staticmethod
    def _c2_error(on, den_on, off, den_off, n_blocks):
        """``c2 = g2 − baseline`` and its jackknife error, blocks shared by both."""
        n = on.shape[0]
        value = _safe_divide(on.sum(0), den_on.sum(0)) - _safe_divide(
            off.sum(0), den_off.sum(0)
        )
        if n < 2:
            return value, np.full_like(value, np.nan)
        blocks = np.array_split(np.arange(n), min(n_blocks, n))
        totals = [a.sum(0) for a in (on, den_on, off, den_off)]
        parts = [
            np.stack([a[b].sum(0) for b in blocks]) for a in (on, den_on, off, den_off)
        ]
        loo = _safe_divide(totals[0] - parts[0], totals[1] - parts[1]) - _safe_divide(
            totals[2] - parts[2], totals[3] - parts[3]
        )
        count = len(blocks)
        with np.errstate(invalid="ignore"):
            sigma = np.sqrt(
                (count - 1) / count * np.nansum((loo - np.nanmean(loo, 0)) ** 2, 0)
            )
        sigma[~np.isfinite(value)] = np.nan
        return value, sigma

    def _g2_es(self, trains, lags, period, binned, kbar_min) -> xr.Dataset:
        mean = self.ttcf(Averaging.ES, trains, kbar_min)["c2"].values
        diag = np.stack(
            [np.nanmean(np.diagonal(mean, k, -2, -1), axis=-1) for k in lags],
            axis=-1,
        )
        tau = lags * period
        if binned and lags.size > BIN_THRESHOLD:
            edges = np.logspace(
                np.log10(tau[BIN_SKIP]), np.log10(tau[-1]), BIN_COUNT + 1
            )
            tail = [
                np.nanmedian(diag[..., (tau >= lo) & (tau < hi)], axis=-1)
                for lo, hi in zip(edges[:-1], edges[1:], strict=True)
            ]
            diag = np.concatenate([diag[..., :BIN_SKIP], np.column_stack(tail)], -1)
            tau = np.concatenate([tau[:BIN_SKIP], 0.5 * (edges[:-1] + edges[1:])])
        return xr.Dataset(
            {"g2": (("roi_q", "tau"), diag)},
            coords={"roi_q": self.roi_q, "tau": tau},
            attrs={"averaging": "es", "period_s": period},
        )

    # ── for the triage ───────────────────────────────────────────────────────
    def diagonals(
        self, kbar_min: float | None = None, trains: np.ndarray | None = None
    ):
        """Per train and ring, the sum and count of finite ``raw − off`` at each lag.

        The shape ``analysis.triage.xpcs.measure_run`` reads from DAMNIT's
        ``xpcs_ttcf``: each train normalised on its own, here with a 0/0 entry
        NaN instead of 1 and no k-bar filter unless ``kbar_min`` is given. Only
        trains with a baseline have a ``raw − off``, so only they get a row.
        """
        from analysis.triage.xpcs import TtcfDiagonals, diagonal_sums

        rows = self._rows(trains, baseline=True)
        sums = np.zeros((rows.size, self.n_rings, max(self.n_frames - 1, 0)))
        counts = np.zeros(sums.shape, dtype=np.int32)
        for part, c in self._chunks(rows):
            at = np.searchsorted(rows, part)
            with np.errstate(invalid="ignore", divide="ignore"):
                kbar = c["s_on"] / c["cnt"][..., None]
                norm = kbar[..., :, None] * kbar[..., None, :]
                raw = np.where(
                    norm > 0, c["on"] / (norm * c["cnt"][..., None, None]), np.nan
                )
                cur = c["s_off_cur"] / c["cnt_pair"][..., None]
                prev = c["s_off_prev"] / c["cnt_pair"][..., None]
                norm_off = cur[..., :, None] * prev[..., None, :]
                off = np.where(
                    norm_off > 0,
                    c["off"] / (norm_off * c["cnt_pair"][..., None, None]),
                    np.nan,
                )
                off = 0.5 * (off + off.swapaxes(-1, -2))
            c2 = raw - off
            if kbar_min is not None:
                lit = kbar > kbar_min
                c2[~(lit[..., :, None] & lit[..., None, :])] = np.nan
            sums[at], counts[at] = diagonal_sums(c2)
        return TtcfDiagonals(self.train_ids[rows], self.roi_q, sums, counts)


def _lag_bins(lags: np.ndarray) -> list[np.ndarray]:
    """Index groups of ``lags`` for binning, or one group per lag when few.

    As extra-speckle's ``g2_to_bins``: above ``BIN_THRESHOLD`` lags the first
    ``BIN_SKIP`` stay alone and the rest fall into ``BIN_COUNT`` logarithmic bins
    (the last one closed, so the final lag is not lost).
    """
    if lags.size <= BIN_THRESHOLD:
        return [np.array([i]) for i in range(lags.size)]
    edges = np.logspace(np.log10(lags[BIN_SKIP]), np.log10(lags[-1]), BIN_COUNT + 1)
    groups = [np.array([i]) for i in range(BIN_SKIP)]
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        inside = (lags >= lo) & ((lags < hi) | (b == BIN_COUNT - 1))
        if inside.any():
            groups.append(np.flatnonzero(inside))
    return groups


def read(
    proposal: int,
    run: int,
    *,
    output_root: str | Path = DEFAULT_XPCS_ROOT,
    rep_rate_mhz: float | None = None,
) -> XpcsHits:
    """Open a run's results, written by :func:`analysis.xpcs.xpcs_hits`.

    :param rep_rate_mhz: DAMNIT's ``rep_rate``. When given, a pulse pattern that
        implies another repetition rate refuses to make a lag axis.
    """
    del proposal  # one proposal per repository today; the path carries the run
    path = Path(output_root) / "xpcs_hits" / f"r{run:04d}" / "xpcs_hits.h5"
    return XpcsHits(path, rep_rate_mhz)
