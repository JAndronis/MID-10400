"""Lit-cell selection and the readout-noise measurement (context file §3 D4, D3).

Both come off the same sampled trains, so they share one streaming accumulator:
a cell is *lit* if a useful fraction of its kept pixels holds at least half a
photon, and the readout noise is the spread of the *dark* cells, which by
definition hold none.

Unlike AGIPD's open task 3, the lit-cell split does not wait on LITFRM: it is
unambiguous in the frames themselves. Measured over r0423 and r0426 on both
detectors, cells **0–6 and 15** hold 33–40 % of kept pixels above half a photon
while every other cell sits at 1e-6 — five orders of magnitude, so the threshold
has an enormous margin on either side. Integrating all 16 cells would halve I(q)
and add a dark-frame background, which is why the selection is mandatory and why
a change in the pattern between runs must fail loudly rather than pass.

Note what the set is **not**: 0–7. The lit *array positions* are 0–7, but the
``data.memoryCell`` values they carry are 0–6 and 15. Reading the split off
positions rather than off the reader is the inference CLAUDE.md pitfall 4
forbids, and it is how the wrong set was first recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.common.masks import frame_bad
from analysis.waxs.config import JungfrauWaxsConfig

__all__ = [
    "CellAccumulator",
    "CellClassification",
    "UnexpectedLitCells",
]


class UnexpectedLitCells(RuntimeError):
    """The lit cells measured in the data are not the ones the config expects."""


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
        }

    def check_expected(self, expected: tuple[int, ...]) -> None:
        """Fail loudly when the measured set is not the configured one (§3 D4).

        :raises UnexpectedLitCells: a silent change in the cell pattern between
            runs would halve or double I(q) with nothing in the output saying
            so, which is exactly what this exists to prevent.
        """
        if self.lit != tuple(expected):
            fractions = {
                int(cell): round(float(value), 5)
                for cell, value in zip(self.cells, self.lit_fraction, strict=True)
            }
            raise UnexpectedLitCells(
                f"the data says cells {list(self.lit)} are lit, the config "
                f"expects {list(expected)}; per-cell fractions above the "
                f"threshold were {fractions}. Either the cell pattern changed "
                "for this run — in which case set expected_lit_cells for it — "
                "or the threshold is wrong."
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
            ``data.memoryCell``. Never inferred from array position (CLAUDE.md
            pitfall 4).
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
        lit_flag = fraction >= self._cfg.lit_fraction_min
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
        )

    def _read_noise(self, dark: tuple[int, ...]) -> tuple[float | None, int]:
        """Pooled standard deviation over the dark cells' kept pixels (§3 D3).

        The dark cells hold no photons, so their spread is the readout noise
        alone: 0.3230 keV on jf1 and 0.3175 on jf2, measured on r0423 and in
        agreement, which is what makes them worth measuring per run rather than
        hardcoding.

        Returns ``(None, 0)`` when the run has no dark cell. That is not an
        error here — :func:`analysis.waxs.run` decides, because a config that
        names ``read_noise_kev`` explicitly does not need the measurement.
        """
        total = sum(self._kept[c] for c in dark)
        if not dark or total < 2:
            return None, 0
        mean = sum(self._sum[c] for c in dark) / total
        mean_square = sum(self._sumsq[c] for c in dark) / total
        variance = max(mean_square - mean**2, 0.0) * total / (total - 1)
        return float(np.sqrt(variance)), int(total)
