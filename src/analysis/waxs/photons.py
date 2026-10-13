"""Photon counting of corrected JUNGFRAU frames, per ASIC (CLAUDE.md open task 18).

Proc data are keV with a zero level that no correction removes (pitfall 23).
W7 measured it on r0423 and r0370 (``context/jungfrau-photonising.md``):

* it jumps 0.13–0.16 keV rms from train to train on jf1, the whole module
  together, and 0.03–0.04 keV from frame to frame on jf2;
* lit cells sit 0.06–0.17 keV above the same train's dark cells, depending on
  the cell and rising with intensity, in shadowed pixels as much as in lit ones.

So the level has to come from each frame itself: no dark cell and no per-run
constant can stand in for it. Per frame and per 256 x 256 ASIC this module
estimates it with p010646's estimator (``transfer_context/templates/
jf_photon_count.py``, validated bit for bit against fxs ``--photons``), then
counts ``clip(rint((x - level) / E), 0)``.

**The estimator.** Over the ASIC's pixels minus its ``edge_px`` border, the
median of the values below ``level_cut_kev``, then ``level_iterations`` times the
mean of the values within ``±level_window_kev`` of the current estimate.

**Which pixels feed it** is :attr:`PhotonCounting.level_pixels`. ``"all_good"``,
the default, takes every ``data.mask``-good finite pixel, the static ``.edf``
ignored: shadowed pixels are good zero-level pixels, and W7 found the level they
give agrees with the shadow alone to 0.004 keV while keeping >= 55 k pixels on
every ASIC. ``"kept"`` restricts it to the pixels the ``.edf`` keeps.

**An ASIC with fewer than** ``min_pixels`` **usable pixels has no level**, and
its pixels are not counted at all rather than counted against a guess. 1000 is
derived, not tuned: the window mean of n pixels of 0.30 keV noise has σ ≈
0.30/√n = 0.0095 keV at n = 1000, which moves ≤ 0.1 % of the counts (W7: a
0.1 keV level error moves 1 % of the one-photon pixels).

**Reproducibility.** The level is estimated in float64 and stored as float32,
and the counts are computed from the *stored* float32 level and the float32
energy, so a reader can recount any frame from proc bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from analysis.waxs.config import MODULE_SHAPE

__all__ = [
    "ASIC_ID",
    "ASIC_SIZE",
    "LEVEL_PIXEL_SETS",
    "N_ASICS",
    "PhotonCounting",
    "asic_view",
    "count_photons",
    "zero_levels",
]

#: One JUNGFRAU ASIC is 256 x 256 pixels; a module is 2 x 4 of them.
ASIC_SIZE: int = 256
N_ASICS: int = 8
LEVEL_PIXEL_SETS: tuple[str, ...] = ("all_good", "kept")

_rows, _cols = np.indices(MODULE_SHAPE)
#: The ASIC each pixel belongs to, flat: ``4 * row_block + column_block``, the
#: order :func:`asic_view` produces. W7 checked the tiling against the
#: ``NON_STANDARD_SIZE`` bit, which covers exactly rows 255–256 and columns
#: 255–256, 511–512 and 767–768 of the proc layout.
ASIC_ID: np.ndarray = ((_rows // ASIC_SIZE) * 4 + _cols // ASIC_SIZE).reshape(-1)
ASIC_ID.flags.writeable = False
del _rows, _cols


@dataclass(frozen=True, slots=True)
class PhotonCounting:
    """The counting settings, all of them config fields and so in the hash."""

    energy_kev: float
    level_cut_kev: float = 2.0
    level_window_kev: float = 1.0
    level_iterations: int = 3
    edge_px: int = 3
    min_pixels: int = 1000
    level_pixels: str = "all_good"

    def __post_init__(self) -> None:
        if self.energy_kev <= 0:
            raise ValueError(f"energy_kev must be positive, got {self.energy_kev}")
        if self.level_window_kev <= 0:
            raise ValueError(
                f"level_window_kev must be positive, got {self.level_window_kev}"
            )
        if self.level_iterations < 0:
            raise ValueError(
                f"level_iterations must be >= 0, got {self.level_iterations}"
            )
        if not 0 <= self.edge_px < ASIC_SIZE // 2:
            raise ValueError(
                f"edge_px must lie in [0, {ASIC_SIZE // 2}), got {self.edge_px}"
            )
        if self.min_pixels < 1:
            raise ValueError(f"min_pixels must be positive, got {self.min_pixels}")
        if self.level_pixels not in LEVEL_PIXEL_SETS:
            raise ValueError(
                f"level_pixels must be one of {LEVEL_PIXEL_SETS}, "
                f"got {self.level_pixels!r}"
            )

    @classmethod
    def from_config(cls, cfg: Any) -> PhotonCounting:
        """The settings a :class:`~analysis.waxs.config.JungfrauWaxsConfig` names."""
        return cls(
            energy_kev=float(cfg.count_energy_kev),
            level_cut_kev=float(cfg.level_cut_kev),
            level_window_kev=float(cfg.level_window_kev),
            level_iterations=int(cfg.level_iterations),
            edge_px=int(cfg.level_edge_px),
            min_pixels=int(cfg.level_min_pixels),
            level_pixels=str(cfg.level_pixels),
        )

    @property
    def energy_f4(self) -> np.float32:
        """The energy as the counts are computed with it, and as it is stored."""
        return np.float32(self.energy_kev)

    @property
    def method(self) -> str:
        """One line a reader can act on without this source to hand."""
        return (
            f"per frame and 256x256 ASIC ({self.edge_px}-px edge left out), over "
            f"{self.level_pixels} pixels: level = median of values < "
            f"{self.level_cut_kev} keV, then {self.level_iterations} x mean of "
            f"values within +-{self.level_window_kev} keV; no level below "
            f"{self.min_pixels} pixels; photons = clip(rint((x - float32(level)) / "
            f"float32({self.energy_kev})), 0) (p010646 jf_photon_count)"
        )

    def summary(self) -> dict[str, Any]:
        """The provenance record."""
        return {
            "energy_kev": self.energy_kev,
            "level_cut_kev": self.level_cut_kev,
            "level_window_kev": self.level_window_kev,
            "level_iterations": self.level_iterations,
            "edge_px": self.edge_px,
            "min_pixels": self.min_pixels,
            "level_pixels": self.level_pixels,
            "method": self.method,
        }


def asic_view(a: np.ndarray, edge: int) -> np.ndarray:
    """``(..., 512, 1024)`` -> ``(..., 8, n)``: each ASIC's pixels, edges left out."""
    lead = a.shape[:-2]
    tiles = a.reshape(*lead, 2, ASIC_SIZE, 4, ASIC_SIZE).swapaxes(-3, -2)
    tiles = tiles[..., edge : ASIC_SIZE - edge, edge : ASIC_SIZE - edge]
    return tiles.reshape(*lead, N_ASICS, -1)


def zero_levels(
    x: np.ndarray, usable: np.ndarray, counting: PhotonCounting
) -> tuple[np.ndarray, np.ndarray]:
    """Each ASIC's zero-photon level, in keV.

    :param x: one frame, keV, ``(512, 1024)`` or flat.
    :param usable: which pixels may feed the estimate, same size as ``x``.
        Non-finite values must already be excluded here.
    :param counting: the estimator's settings.
    :returns: ``(levels, n_pixels)``: float64 ``(8,)``, NaN for an ASIC with
        fewer than ``min_pixels`` usable pixels or with no value below the cut
        or inside the window; and int ``(8,)``, the usable pixels per ASIC.
    """
    values = asic_view(np.asarray(x).reshape(MODULE_SHAPE), counting.edge_px)
    chosen = asic_view(np.asarray(usable).reshape(MODULE_SHAPE), counting.edge_px)
    n_pixels = chosen.sum(axis=-1)
    levels = np.full(N_ASICS, np.nan)
    for asic in range(N_ASICS):
        if n_pixels[asic] < counting.min_pixels:
            continue
        pixels = values[asic][chosen[asic]].astype(np.float64)
        below = pixels[pixels < counting.level_cut_kev]
        if below.size == 0:
            continue
        estimate = float(np.median(below))
        for _ in range(counting.level_iterations):
            # Exact, not an approximation: when the window ends below the cut,
            # every value inside it is already in ``below``, which is a third
            # smaller at r0423's occupancy and halves the cost.
            fits = estimate + counting.level_window_kev <= counting.level_cut_kev
            pool = below if fits else pixels
            inside = pool[np.abs(pool - estimate) < counting.level_window_kev]
            if inside.size == 0:
                estimate = float("nan")
                break
            estimate = float(inside.sum() / inside.size)
        levels[asic] = estimate
    return levels, n_pixels


def count_photons(x: np.ndarray, levels: np.ndarray, energy: np.float32) -> np.ndarray:
    """``clip(rint((x - level) / E), 0)`` per pixel, in float32, flat.

    :param x: one frame, keV, any shape that flattens to the module.
    :param levels: float32 ``(8,)``, one per ASIC; a NaN level gives NaN counts
        on its ASIC, which the caller excludes.
    :param energy: the photon energy as float32 — the same value that is stored.
    :returns: a new float32 array; NaN in ``x`` stays NaN.
    """
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    level = np.asarray(levels, dtype=np.float32)[ASIC_ID]
    photons = np.rint((flat - level) / np.float32(energy))
    return np.clip(photons, np.float32(0), None)
