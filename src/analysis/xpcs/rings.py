"""The pixel envelope the photon store covers, and the rings inside it.

Geometry is the SAXS pass's own (``build_operator``, whose pixel-centre q and
beam centre every other product uses), not a second one. The *envelope* is the
annulus the rings span, dilated by ``envelope_margin_mm`` and with bad pixels
kept: it is what the store holds, so it fixes the store's identity, while the
rings, the static mask and the kernel are recomputable from it after proc is
taped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from analysis.saxs.config import NPIX, SHAPE
from analysis.saxs.photon_stats import PhotonStatsConfig, ring_labels_from
from analysis.xpcs.config import XpcsConfig

__all__ = [
    "Envelope",
    "PixelMap",
    "Rings",
    "build_envelope",
    "build_rings",
    "pixel_map",
    "sha256_of_array",
    "static_bad_for",
]

RING_MASK_SHAPE = (16, 512, 128)


def sha256_of_array(array: Any) -> str:
    """sha256 over an array's dtype, shape and bytes."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.dtype.str}{array.shape}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Envelope:
    """The detector pixels the store keeps, as flat indices into ``(16·512, 128)``."""

    pixels: np.ndarray  # int32 (n_env,), sorted, read-only
    sha256: str

    @property
    def size(self) -> int:
        return int(self.pixels.size)


@dataclass(frozen=True, slots=True)
class Rings:
    """Each envelope pixel's ring, or −1 for none, a bad pixel or a seam."""

    edges_nm: tuple[float, ...]  # empty when the rings came from a masks file
    label: np.ndarray  # int8 (n_env,), read-only
    n_rings: int
    sha256: str

    def centres_nm(self) -> np.ndarray:
        edges = np.asarray(self.edges_nm)
        return 0.5 * (edges[:-1] + edges[1:])


@dataclass(frozen=True, slots=True)
class PixelMap:
    """Every detector pixel's ring (−1: none) and, for geometry rings, its q."""

    label: np.ndarray  # int16 (NPIX,)
    n_rings: int
    q_nm: np.ndarray | None  # float64 (NPIX,); None for rings from a masks file
    operator_sha256: str | None  # provenance only: it depends on the node (pitfall 30)


def pixel_map(cfg: XpcsConfig, geometry: Any = None) -> PixelMap:
    """Label every detector pixel from the SAXS geometry or from ``ring_masks_file``.

    :param geometry: an AGIPD geometry object instead of ``cfg.saxs.geometry_file``
        (the tests' synthetic one).
    :raises ValueError: a masks file of the wrong shape, or one pixel in two rings.
    """
    if cfg.ring_masks_file is not None:
        masks = np.load(cfg.ring_masks_file)
        if masks.ndim != 4 or masks.shape[1:] != RING_MASK_SHAPE:
            raise ValueError(
                f"{cfg.ring_masks_file} has shape {masks.shape}, expected "
                f"(R, 16, 512, 128)"
            )
        flat = masks.reshape(masks.shape[0], NPIX) != 0
        if (flat.sum(axis=0) > 1).any():
            raise ValueError(f"{cfg.ring_masks_file}: a pixel belongs to two rings")
        label = np.where(flat.any(axis=0), flat.argmax(axis=0), -1).astype(np.int16)
        return PixelMap(label, int(flat.shape[0]), None, None)

    from analysis.saxs.operator import build_operator, geometry_from_config

    geom = geometry if geometry is not None else geometry_from_config(cfg.saxs)
    op, ai = build_operator(geom, cfg.saxs)
    edges = cfg.ring_edges_nm
    pairs = tuple(zip(edges[:-1], edges[1:], strict=True))
    label = ring_labels_from(ai, PhotonStatsConfig(saxs=cfg.saxs, rings_nm=pairs))
    q = np.asarray(ai.array_from_unit(SHAPE, "center", cfg.saxs.unit)).ravel()
    return PixelMap(label, len(pairs), q.astype(np.float64), op.sha256)


def static_bad_for(cfg: XpcsConfig) -> np.ndarray:
    """The static mask folded into the rings; none when the rings are a file's.

    Rings from ``ring_masks_file`` are authoritative, as extra-speckle's own are
    when the equivalence gate compares the two on identical pixels.
    """
    if cfg.ring_masks_file is not None:
        return np.zeros(NPIX, dtype=bool)
    from analysis.saxs.masks import build_static_bad

    return build_static_bad(cfg.saxs).bad


def _radius_mm(q_nm: np.ndarray | float, cfg: XpcsConfig) -> np.ndarray:
    """Detector-plane radius of q, for a flat detector facing the beam."""
    sin_theta = np.asarray(q_nm) * cfg.saxs.wavelength_m * 1e9 / (4 * np.pi)
    return cfg.saxs.sdd_m * 1e3 * np.tan(2 * np.arcsin(sin_theta))


def build_envelope(pixels: PixelMap, cfg: XpcsConfig) -> Envelope:
    """The annulus spanned by the rings, dilated by ``cfg.envelope_margin_mm``.

    Rings from a masks file are used as they are: their union, undilated.
    """
    if pixels.q_nm is None:
        keep = pixels.label >= 0
    else:
        edges = cfg.ring_edges_nm
        low = float(_radius_mm(edges[0], cfg)) - cfg.envelope_margin_mm
        high = float(_radius_mm(edges[-1], cfg)) + cfg.envelope_margin_mm
        with np.errstate(invalid="ignore"):
            radius = _radius_mm(pixels.q_nm, cfg)
        keep = (radius >= low) & (radius <= high)
    chosen = np.flatnonzero(keep).astype(np.int32)
    if chosen.size == 0:
        raise ValueError("the rings select no detector pixel")
    chosen.flags.writeable = False
    return Envelope(chosen, sha256_of_array(chosen))


def build_rings(
    pixels: PixelMap, envelope: Envelope, static_bad: np.ndarray, cfg: XpcsConfig
) -> Rings:
    """Ring labels over the envelope, with the static mask folded in as −1."""
    label = pixels.label[envelope.pixels].astype(np.int8)
    label[np.asarray(static_bad, dtype=bool)[envelope.pixels]] = -1
    label.flags.writeable = False
    counts = np.bincount(label[label >= 0], minlength=pixels.n_rings)
    if (counts == 0).any():
        raise ValueError(
            f"ring(s) {np.flatnonzero(counts == 0).tolist()} hold no good pixel"
        )
    edges = () if cfg.ring_masks_file is not None else tuple(cfg.ring_edges_nm)
    digest = hashlib.sha256(repr(edges).encode())
    digest.update(sha256_of_array(label).encode())
    return Rings(edges, label, pixels.n_rings, digest.hexdigest())
