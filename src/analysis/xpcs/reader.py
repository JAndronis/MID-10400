"""The reader of ``xpcs_hits.h5``: every float of the hit-train XPCS is made here.

`XpcsHits` reads the integer sums of `analysis.xpcs.results_file` on demand and
turns them into g2 and two-time maps with `analysis.xpcs.estimators`. The pooled
estimates divide summed numerators by summed denominators, so they pool in any
order, and a quantity with no photons behind it is NaN, never 1.
``Averaging.ES`` is extra-speckle's mean of per-train ratios, with its rule that
a zero norm counts as 1 and its k-bar filter, kept for comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import xarray as xr

from analysis.common.cpu import default_pool
from analysis.common.status import FrameStatus
from analysis.xpcs.config import DEFAULT_XPCS_ROOT
from analysis.xpcs.estimators import (
    N_BLOCKS,
    SKIP_LAST_LAGS,
    TtcfDiagonals,
    diagonal_sums,
    es_g2,
    lag_frames,
    lag_sums,
    normalised,
    outer,
    pooled_g2,
    safe_divide,
)
from analysis.xpcs.kernel import PairStatus
from analysis.xpcs.outliers import OutlierMask
from analysis.xpcs.results_file import (
    SUM_NAMES,
    read_header,
    read_outlier,
    read_ring_label,
    read_sums,
)

__all__ = [
    "BASE_CLOCK_HZ",
    "ES_KBAR_MIN",
    "ES_NAN_THRESH",
    "Averaging",
    "Coverage",
    "NonUniformPattern",
    "PooledSums",
    "RepRateMismatch",
    "XpcsHits",
    "read",
]

#: The clock pulse ids count in: the 1.3 GHz master over 288, 4.5139 MHz, as
#: ``XrayPulses.bunch_repetition_rate`` reports it and extra-speckle's
#: ``convert_pulse_to_mhz`` uses it. A pulse-id step of 2 is 443.1 ns.
BASE_CLOCK_HZ = 1.3e9 / 288
#: extra-speckle's k-bar threshold. It has no known source, so it is offered
#: only in the ``Averaging.ES`` view.
ES_KBAR_MIN = 3e-3
#: extra-speckle's ``reduce_ttcf(mask_noisy=True, nan_thresh=0.025)``: a ring is
#: left out of g2 when more than this share of its (frame, frame) map is NaN. The
#: pipeline's ``mask_noisy`` defaults to True; offered only in the ES view.
ES_NAN_THRESH = 0.025
#: Largest relative disagreement between the pulse pattern's repetition rate and
#: a DAMNIT ``rep_rate`` before the lag axis is refused.
REP_RATE_RTOL = 0.01
#: Rows read at a time when pooling.
CHUNK = 16


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


@dataclass(frozen=True, slots=True)
class PooledSums:
    """What one read of the sums yields for the pooled g2 and two-time maps.

    ``rows`` are the usable rows read, ascending, and ``has_off`` theirs. The
    ``*_num`` / ``*_den`` maps are ``(R, P, P)``: ``raw`` over every row, ``pair``
    and ``off`` over the rows with a baseline (``off`` not yet symmetrised). The
    lag terms ``on, den_on, off, den_off`` are per row, ``(n, R, P − 1)``
    over the lags ``1 … P − 1``; the baseline ones sum both diagonals and are zero
    for a row without a baseline.
    """

    rows: np.ndarray
    has_off: np.ndarray
    raw_num: np.ndarray
    raw_den: np.ndarray
    pair_num: np.ndarray
    pair_den: np.ndarray
    off_num: np.ndarray
    off_den: np.ndarray
    on: np.ndarray
    den_on: np.ndarray
    off: np.ndarray
    den_off: np.ndarray


def _pooled_slab(path: Path, rows: np.ndarray) -> PooledSums:
    """A pool worker's share: the pooled sums of contiguous ``rows``."""
    return XpcsHits(path)._sums_of(rows)


def _merge(slabs: list[PooledSums]) -> PooledSums:
    """Slabs in row order as one: maps summed in that order, lag terms stacked."""
    first = slabs[0]
    maps = {
        name: getattr(first, name).copy()
        for name in ("raw_num", "raw_den", "pair_num", "pair_den", "off_num", "off_den")
    }
    for slab in slabs[1:]:
        for name, total in maps.items():
            total += getattr(slab, name)
    stacked = {
        name: np.concatenate([getattr(slab, name) for slab in slabs])
        for name in ("rows", "has_off", "on", "den_on", "off", "den_off")
    }
    return PooledSums(**maps, **stacked)


def _check_sums(averaging: Averaging, trains, sums: PooledSums | None) -> None:
    if sums is None:
        return
    if averaging is not Averaging.POOLED:
        raise ValueError("precomputed sums are pooled ones; ES reads its own")
    if trains is not None:
        raise ValueError("pass trains or precomputed sums, not both")


class XpcsHits:
    """A run's hit-train sums, read on demand; every float is derived here."""

    def __init__(self, path: str | Path, rep_rate_mhz: float | None = None) -> None:
        self.path = Path(path)
        self.rep_rate_mhz = rep_rate_mhz
        header = read_header(self.path)
        self.train_ids: np.ndarray = header["train_ids"]
        self.status: np.ndarray = header["status"]
        self.partner_ids: np.ndarray = header["partner_ids"]
        self.pair_status: np.ndarray = header["pair_status"]
        self.has_off: np.ndarray = header["has_off"]
        self.pulse_ids: np.ndarray = header["pulse_ids"]
        #: ``(R, 2)`` ring bounds [nm⁻¹]; ``(0, 2)`` for rings from a masks file.
        self.bounds_nm: np.ndarray = header["bounds_nm"]
        self.n_rings: int = header["n_rings"]
        self.n_frames: int = header["n_frames"]
        self.attrs: dict[str, Any] = header["attrs"]

    def outlier(self) -> tuple[OutlierMask, np.ndarray] | None:
        """The applied pixel-outlier mask and the ring labels before it; None if off."""
        return read_outlier(self.path)

    def ring_label(self) -> np.ndarray:
        """The ring label of every envelope pixel the sums use; -1 for none."""
        return read_ring_label(self.path)

    @property
    def roi_q(self) -> np.ndarray:
        """Ring centres in Å⁻¹, the unit of DAMNIT's ``roi_q``; ring numbers for
        rings from a masks file."""
        if not len(self.bounds_nm):
            return np.arange(self.n_rings, dtype=np.float64)
        return self.bounds_nm.mean(axis=1) / 10.0

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

    def _chunks(
        self, rows: np.ndarray, names: tuple[str, ...] = SUM_NAMES
    ) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
        """The sums ``names`` of ``rows``, a few rows at a time, as float64."""
        with h5py.File(self.path, "r") as f:
            for start in range(0, rows.size, CHUNK):
                part = rows[start : start + CHUNK]
                yield part, read_sums(f, part, names, np.float64)

    def pooled_sums(
        self,
        trains: np.ndarray | None = None,
        n_workers: int = 1,
        pool_factory: Callable[..., Any] | None = None,
    ) -> PooledSums:
        """Everything the pooled ``g2`` and ``ttcf`` need, in one read of the sums.

        With ``n_workers > 1`` contiguous slabs of rows go to a pool and are
        merged in row order: the lag terms are the serial ones, the summed maps
        differ from them by float grouping only.

        :param pool_factory: builds the executor (spawned processes by default);
            the tests inject an inline one.
        """
        rows = self._rows(trains)
        n_chunks = -(-rows.size // CHUNK)
        if n_workers <= 1 or n_chunks <= 1:
            return self._sums_of(rows)
        size = -(-n_chunks // n_workers) * CHUNK
        slabs = [rows[start : start + size] for start in range(0, rows.size, size)]
        factory = pool_factory or default_pool
        with factory(len(slabs)) as pool:
            futures = [pool.submit(_pooled_slab, self.path, slab) for slab in slabs]
            return _merge([future.result() for future in futures])

    def _sums_of(self, rows: np.ndarray) -> PooledSums:
        shape = (self.n_rings, self.n_frames, self.n_frames)
        raw_num, raw_den, pair_num, pair_den, off_num, off_den = (
            np.zeros(shape) for _ in range(6)
        )
        lags = np.arange(1, self.n_frames)
        per_row = (rows.size, self.n_rings, lags.size)
        on, den_on_terms, off_terms, den_off_terms = (
            np.zeros(per_row) for _ in range(4)
        )
        for part, c in self._chunks(rows):
            at = np.searchsorted(rows, part)
            den_on = outer(c["s_on"], c["s_on"], c["cnt"])
            den_off = outer(c["s_off_cur"], c["s_off_prev"], c["cnt_pair"])
            raw_num += c["on"].sum(0)
            raw_den += den_on.sum(0)
            base = self.has_off[part]
            if base.any():
                pair_num += c["on"][base].sum(0)
                pair_den += den_on[base].sum(0)
                off_num += c["off"][base].sum(0)
                off_den += den_off[base].sum(0)
            on[at] = lag_sums(c["on"], lags)
            den_on_terms[at] = lag_sums(den_on, lags)
            off_terms[at] = lag_sums(c["off"], lags, both=True)
            den_off_terms[at] = lag_sums(den_off, lags, both=True)
        return PooledSums(
            rows=rows,
            has_off=self.has_off[rows],
            raw_num=raw_num,
            raw_den=raw_den,
            pair_num=pair_num,
            pair_den=pair_den,
            off_num=off_num,
            off_den=off_den,
            on=on,
            den_on=den_on_terms,
            off=off_terms,
            den_off=den_off_terms,
        )

    def ttcf(
        self,
        averaging: Averaging = Averaging.POOLED,
        trains: np.ndarray | None = None,
        kbar_min: float | None = ES_KBAR_MIN,
        sums: PooledSums | None = None,
    ) -> xr.Dataset:
        """The two-time maps ``raw``, ``off`` and ``c2 = raw − off``.

        ``Averaging.POOLED``: ``raw`` is Σ on over Σ s_a s_b / cnt, over all the
        usable trains; ``off`` the same for the baseline, symmetrised after the
        sums; ``raw_paired`` is ``raw`` over the trains that have a baseline, so
        that ``c2`` subtracts like from like. Anything with no photons behind it
        is NaN.

        ``Averaging.ES``: each train normalised on its own (a zero norm is 1),
        entries with k-bar under ``kbar_min`` set to NaN, then the mean over
        trains. ``kbar_min`` is extra-speckle's unsourced default; ``None``
        disables the filter.

        :param sums: ``pooled_sums()`` already made, so ``g2`` and ``ttcf`` share
            one read; pooled only, and not together with ``trains``.
        """
        _check_sums(averaging, trains, sums)
        if averaging is Averaging.ES:
            return self._ttcf_es(trains, kbar_min)
        if sums is None:
            sums = self.pooled_sums(trains)
        raw = safe_divide(sums.raw_num, sums.raw_den)
        paired = safe_divide(sums.pair_num, sums.pair_den)
        off = safe_divide(
            sums.off_num + sums.off_num.swapaxes(-1, -2),
            sums.off_den + sums.off_den.swapaxes(-1, -2),
        )
        return self._map_dataset(
            {"raw": raw, "raw_paired": paired, "off": off, "c2": paired - off},
            sums.rows,
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

    def _ttcf_es(self, trains, kbar_min) -> xr.Dataset:
        shape = (self.n_rings, self.n_frames, self.n_frames)
        total_raw, total_off = np.zeros(shape), np.zeros(shape)
        n_raw, n_off = np.zeros(shape), np.zeros(shape)
        rows = self._rows(trains)
        for part, c in self._chunks(rows):
            raw, off, kbar = normalised(c, self.has_off[part], zero_norm=1.0)
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

    def kbar(self, trains: np.ndarray | None = None) -> xr.DataArray:
        """Mean photons per pixel per frame and ring, ``(trainId, frame, roi_q)``."""
        rows = self._rows(trains)
        out = np.full((rows.size, self.n_frames, self.n_rings), np.nan)
        for part, c in self._chunks(rows, ("s_on", "cnt")):
            at = np.searchsorted(rows, part)
            out[at] = safe_divide(c["s_on"], c["cnt"][..., None]).swapaxes(1, 2)
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

    def pixels(self, trains: np.ndarray | None = None) -> np.ndarray:
        """Pixels summed into each ring of each usable train, ``(train, ring)``."""
        rows = self._rows(trains)
        if rows.size == 0:
            return np.zeros((0, self.n_rings), dtype=np.int64)
        with h5py.File(self.path, "r") as f:
            return read_sums(f, rows, ("cnt",), np.int64)["cnt"]

    def ring_sums(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(rows, s_on (n, R, P), cnt (n, R))`` of the usable trains, as float64."""
        rows = np.flatnonzero(self.usable)
        with h5py.File(self.path, "r") as f:
            c = read_sums(f, rows, ("s_on", "cnt"), np.float64)
        return rows, c["s_on"], c["cnt"]

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
            raw, off, _ = normalised(c, self.has_off[part], zero_norm=1.0)
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

    def lags(self, skip_last: int = SKIP_LAST_LAGS) -> np.ndarray:
        """Lags (in frames) g2 is evaluated at: extra-speckle's ``1 … P − 1 − skip``."""
        return lag_frames(self.n_frames, skip_last)

    def g2(
        self,
        averaging: Averaging = Averaging.POOLED,
        trains: np.ndarray | None = None,
        binned: bool = True,
        skip_last: int = SKIP_LAST_LAGS,
        n_blocks: int = N_BLOCKS,
        kbar_min: float | None = ES_KBAR_MIN,
        nan_thresh: float | None = ES_NAN_THRESH,
        sums: PooledSums | None = None,
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
        bins. It has no error. As in ``reduce_ttcf(mask_noisy=True)`` it leaves
        out a ring whose map has more than ``nan_thresh`` NaN (``None`` keeps
        every ring); ``roi_q`` then lists the rings kept.

        :param skip_last: lags dropped from the end, as extra-speckle does.
        :param sums: ``pooled_sums()`` already made, as for ``ttcf``.
        :raises NonUniformPattern: the pulses are not evenly spaced.
        :raises RepRateMismatch: the pulse pattern disagrees with ``rep_rate_mhz``.
        """
        _check_sums(averaging, trains, sums)
        lags = self.lags(skip_last)
        if lags.size == 0:
            return self._empty_g2(averaging)
        period = self.period_s
        if averaging is Averaging.ES:
            return self._g2_es(trains, lags, period, binned, kbar_min, nan_thresh)

        if sums is None:
            sums = self.pooled_sums(trains)
        return pooled_g2(
            sums.on,
            sums.den_on,
            sums.off,
            sums.den_off,
            sums.has_off,
            lags=lags,
            period=period,
            roi_q=self.roi_q,
            binned=binned,
            n_blocks=n_blocks,
        )

    def _empty_g2(self, averaging: Averaging) -> xr.Dataset:
        """g2 of a run with fewer than two frames: no lag, so nothing to report."""
        none = np.zeros((self.n_rings, 0))
        names = (
            ["g2"]
            if averaging is Averaging.ES
            else ["g2", "sigma_g2", "g2_paired", "baseline", "c2", "sigma"]
        )
        return xr.Dataset(
            {name: (("roi_q", "tau"), none) for name in names},
            coords={"roi_q": self.roi_q, "tau": np.zeros(0)},
            attrs={"averaging": averaging.value},
        )

    def _g2_es(self, trains, lags, period, binned, kbar_min, nan_thresh) -> xr.Dataset:
        mean = self.ttcf(Averaging.ES, trains, kbar_min)["c2"].values
        return es_g2(mean, self.roi_q, lags, period, binned, nan_thresh)

    def diagonals(
        self, kbar_min: float | None = None, trains: np.ndarray | None = None
    ):
        """Per train and ring, the sum and count of finite ``raw − off`` at each lag.

        The shape the SNR reduction (``measure_run``) takes: each train
        normalised on its own, here with a 0/0 entry NaN instead of 1 and no
        k-bar filter unless ``kbar_min`` is given. Only
        trains with a baseline have a ``raw − off``, so only they get a row.
        """
        rows = self._rows(trains, baseline=True)
        sums = np.zeros((rows.size, self.n_rings, max(self.n_frames - 1, 0)))
        counts = np.zeros(sums.shape, dtype=np.int32)
        for part, c in self._chunks(rows):
            at = np.searchsorted(rows, part)
            raw, off, kbar = normalised(c, self.has_off[part], zero_norm=np.nan)
            c2 = raw - off
            if kbar_min is not None:
                lit = kbar > kbar_min
                c2[~(lit[..., :, None] & lit[..., None, :])] = np.nan
            sums[at], counts[at] = diagonal_sums(c2)
        return TtcfDiagonals(self.train_ids[rows], self.roi_q, sums, counts)


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
    del proposal  # the output root belongs to one proposal
    path = Path(output_root) / "xpcs_hits" / f"r{run:04d}" / "xpcs_hits.h5"
    return XpcsHits(path, rep_rate_mhz)
