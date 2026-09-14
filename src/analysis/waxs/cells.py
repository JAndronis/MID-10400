"""Lit-cell selection and the readout-noise measurement.

Both come off the same sampled trains, so they share one streaming accumulator:
a cell is *lit* if a useful fraction of its kept pixels holds at least half a
photon, and the readout noise is the spread of the *dark* cells, which hold none.

Which cells are lit is not fixed between runs — the proposal used four readout
patterns — so the pass gates on the *shape* instead: every valid pattern is a
run of consecutive memory cells mod 16, which :func:`is_storage_cell_sequence`
checks. The cell ids come from the reader, never from a frame's array position,
which is a different sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.common.masks import frame_bad
from analysis.waxs.config import JungfrauWaxsConfig, is_storage_cell_sequence

#: Guards the ratio arithmetic in :func:`split_lit_dark` against the exactly
#: zero fractions that dark cells reach on short samples.
_FLOOR_EPS = 1e-9

__all__ = [
    "CellAccumulator",
    "CellClassification",
    "ImplausibleLitCells",
    "UnexpectedLitCells",
    "is_storage_cell_sequence",
    "split_lit_dark",
]


class UnexpectedLitCells(RuntimeError):
    """The lit cells measured in the data are not the ones the config expects."""


class ImplausibleLitCells(UnexpectedLitCells):
    """The measured lit cells are not a storage-cell sequence.

    A JUNGFRAU reads consecutive storage cells mod 16, so any other shape means
    the classification is wrong rather than that the run is unusual. Raised
    rather than logged: a tripwire that returns quietly is not a tripwire.
    """


def split_lit_dark(
    fraction: np.ndarray, floor: float, gap_ratio: float
) -> tuple[np.ndarray, float | None]:
    """Split per-cell lit fractions into lit and dark.

    :param fraction: one lit fraction per cell, in cell order.
    :param floor: a cell below this is dark. Placed in the empty multiplicative
        band that separates the two populations.
    :param gap_ratio: among the cells that clear the floor, a step larger than
        this splits them again, so an attenuated run still separates on the
        step rather than on the absolute level.
    :returns: a boolean lit mask, and the ratio of the step it cut at —
        ``None`` when every cell above the floor was kept.
    """
    order = np.argsort(fraction)[::-1]
    values = np.maximum(fraction[order], _FLOOR_EPS)
    above = np.flatnonzero(values > floor)
    lit = np.zeros(fraction.shape, dtype=bool)
    if above.size == 0:
        return lit, None

    last = int(above[-1])
    cut, ratio = last, 1.0
    for index in range(last):
        step = values[index] / values[index + 1]
        if step > ratio:
            cut, ratio = index, step
    if ratio < gap_ratio:
        cut, ratio = last, None

    lit[order[: cut + 1]] = True
    return lit, ratio


@dataclass(frozen=True, slots=True)
class CellClassification:
    """Which memory cells carry photons, and how confidently."""

    cells: np.ndarray  # uint16 (n_cells,), sorted
    lit_fraction: np.ndarray  # float64 (n_cells,)
    n_samples: np.ndarray  # uint32  (n_cells,)
    lit: tuple[int, ...]
    dark: tuple[int, ...]
    read_noise_kev: float | None
    read_noise_samples: int
    #: The step :func:`split_lit_dark` cut at, or ``None`` when it kept every
    #: cell above the floor. Recorded so a marginal split is visible rather
    #: than having to be re-derived from ``lit_fraction``.
    gap_ratio: float | None = None

    def summary(self) -> dict[str, object]:
        """The provenance record: every cell's fraction, and the two sets."""
        return {
            "cells": [int(c) for c in self.cells],
            "lit_fraction": [float(f) for f in self.lit_fraction],
            "n_samples": [int(n) for n in self.n_samples],
            "lit": list(self.lit),
            "dark": list(self.dark),
            "read_noise_kev": self.read_noise_kev,
            "read_noise_samples": self.read_noise_samples,
            "gap_ratio": self.gap_ratio,
        }

    def check_structure(self, n_cells: int) -> None:
        """Fail when the measured set is not a storage-cell sequence.

        Unlike :meth:`check_expected`, which pins one named set, this holds for
        every run — including one whose pattern nobody has looked at yet.

        :raises ImplausibleLitCells: the shape is impossible for a JUNGFRAU
            readout, so the classification — not the run — is what is wrong.
        """
        if is_storage_cell_sequence(self.lit, n_cells):
            return
        raise ImplausibleLitCells(
            f"cells {list(self.lit)} came out lit, which is not a run of "
            f"consecutive cells mod {n_cells} and so cannot be a storage-cell "
            f"sequence; per-cell fractions were {self._fractions()}. The "
            "classification is wrong: look at lit_fraction_min and "
            "lit_gap_ratio before trusting any I(q) from this run."
        )

    def _fractions(self) -> dict[int, float]:
        return {
            int(cell): round(float(value), 6)
            for cell, value in zip(self.cells, self.lit_fraction, strict=True)
        }

    def check_expected(self, expected: tuple[int, ...]) -> None:
        """Fail loudly when the measured set is not the pinned one.

        Only reached when a caller pinned ``cfg.expected_lit_cells``, which is
        how a reprocess refuses a run whose pattern has drifted;
        :meth:`check_structure` is the check that runs unconditionally.

        :raises UnexpectedLitCells: an unnoticed change in the cell pattern
            would halve or double I(q) with nothing in the output saying so.
        """
        if self.lit != tuple(expected):
            fractions = self._fractions()
            raise UnexpectedLitCells(
                f"the data says cells {list(self.lit)} are lit, the config "
                f"expects {list(expected)}; per-cell fractions above the "
                f"threshold were {fractions}. Either this run genuinely reads a "
                "different storage-cell pattern — in which case clear "
                "expected_lit_cells, or pin it to this run's set — or the "
                "threshold is wrong."
            )


class CellAccumulator:
    """Streaming per-cell statistics over sampled trains.

    The caller feeds one train at a time; :meth:`finalise` turns the totals into
    a :class:`CellClassification`. Everything accumulates in float64, so the
    number of sampled trains is not bounded by the accumulator.
    """

    def __init__(self, cfg: JungfrauWaxsConfig, static_bad: np.ndarray) -> None:
        self._cfg = cfg
        self._static = np.asarray(static_bad)
        self._kept: dict[int, int] = {}
        self._above: dict[int, int] = {}
        self._sum: dict[int, float] = {}
        self._sumsq: dict[int, float] = {}
        self._frames: dict[int, int] = {}
        self.bits_present = 0
        self.n_trains = 0

    def update(self, data: np.ndarray, mask: np.ndarray, cell_ids: np.ndarray) -> None:
        """Accumulate one train.

        :param data: ``(n_cells, 512, 1024)`` float32 keV.
        :param mask: ``(n_cells, 512, 1024)`` uint32 ``BadPixels``.
        :param cell_ids: ``(n_cells,)`` memory-cell id per frame, read from
            ``data.memoryCell`` — never inferred from array position, which is
            a different sequence.
        """
        data = np.asarray(data)
        mask = np.asarray(mask)
        cell_ids = np.asarray(cell_ids)
        if data.ndim != 3 or data.shape[1:] != (512, 1024):
            raise ValueError(
                f"data must have shape (n_cells, 512, 1024), got {data.shape}"
            )
        if mask.shape != data.shape:
            raise ValueError(f"mask has shape {mask.shape}, expected {data.shape}")
        if mask.dtype != np.uint32:
            raise TypeError(f"data.mask must be uint32, got {mask.dtype}")
        if cell_ids.shape != (data.shape[0],):
            raise ValueError(
                f"cell_ids has shape {cell_ids.shape}, expected ({data.shape[0]},)"
            )

        self.bits_present |= int(np.bitwise_or.reduce(mask, axis=None))

        for index, cell in enumerate(cell_ids.tolist()):
            bad = frame_bad(mask[index], self._cfg.mask_bits, self._static)
            values = data[index].reshape(-1)[~bad].astype(np.float64)
            self._kept[cell] = self._kept.get(cell, 0) + values.size
            self._above[cell] = self._above.get(cell, 0) + int(
                (values > self._cfg.lit_threshold_kev).sum()
            )
            self._sum[cell] = self._sum.get(cell, 0.0) + float(values.sum())
            self._sumsq[cell] = self._sumsq.get(cell, 0.0) + float((values**2).sum())
            self._frames[cell] = self._frames.get(cell, 0) + 1

        self.n_trains += 1

    def finalise(self) -> CellClassification:
        """Classify the cells and measure the readout noise off the dark ones."""
        if not self._kept:
            raise ValueError("no frames accumulated; cannot classify cells")

        cells = np.array(sorted(self._kept), dtype=np.uint16)
        fraction = np.array(
            [
                self._above[int(c)] / self._kept[int(c)] if self._kept[int(c)] else 0.0
                for c in cells
            ],
            dtype=np.float64,
        )
        n_samples = np.array([self._frames[int(c)] for c in cells], dtype=np.uint32)
        lit_flag, gap_ratio = split_lit_dark(
            fraction, self._cfg.lit_fraction_min, self._cfg.lit_gap_ratio
        )
        lit = tuple(int(c) for c, f in zip(cells, lit_flag, strict=True) if f)
        dark = tuple(int(c) for c, f in zip(cells, lit_flag, strict=True) if not f)

        read_noise, read_noise_samples = self._read_noise(dark)
        return CellClassification(
            cells=cells,
            lit_fraction=fraction,
            n_samples=n_samples,
            lit=lit,
            dark=dark,
            read_noise_kev=read_noise,
            read_noise_samples=read_noise_samples,
            gap_ratio=gap_ratio,
        )

    def _read_noise(self, dark: tuple[int, ...]) -> tuple[float | None, int]:
        """Pooled standard deviation over the dark cells' kept pixels.

        The dark cells hold no photons, so their spread is the readout noise
        alone, which is why it is worth measuring per run rather than hardcoding.

        :returns: the noise in keV and the number of pixels it came from, or
            ``(None, 0)`` when the run reads all 16 cells and so has no dark
            one. That is not an error here — :func:`analysis.waxs.run._error_model`
            falls back to ``cfg.read_noise_fallback_kev``.
        """
        total = sum(self._kept[c] for c in dark)
        if not dark or total < 2:
            return None, 0
        mean = sum(self._sum[c] for c in dark) / total
        mean_square = sum(self._sumsq[c] for c in dark) / total
        variance = max(mean_square - mean**2, 0.0) * total / (total - 1)
        return float(np.sqrt(variance)), int(total)
