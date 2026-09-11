"""Sparse full-split integration operator (context file §6.1).

The operator is pyFAI's own ``("full", "csc", "cython")`` sparse matrix, lifted
out of the engine so that the per-frame integration can run over photon hits
only (``sparse.integrate_frame``) instead of over the dense detector.

Array names differ from the context file's pseudocode to avoid a collision with
``image.data`` and to match pyFAI's own attribute names:

===============  ==================  ===========================================
this module      context file §6.1   pyFAI ``engine.lut``
===============  ==================  ===========================================
``coef``         ``data``            ``lut[0]`` — split coefficient per entry
``bins``         ``rows``            ``lut[1]`` — CSC row index, i.e. the q bin
``indptr``       ``indptr``          ``lut[2]`` — column starts, one per pixel
===============  ==================  ===========================================

The matrix is in CSC layout with one column per flattened pixel, so
``len(indptr) == NPIX + 1``; this is asserted at build time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from extra_geom import AGIPD_1MGeometry
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from analysis.saxs.config import NPIX, SHAPE, FirstPassConfig

__all__ = [
    "SparseOperator",
    "build_operator",
    "geometry_from_config",
    "load_operator",
    "operator_sha256",
    "save_operator",
]


@dataclass(frozen=True, slots=True)
class SparseOperator:
    """An immutable CSC operator plus the solid-angle array and the q axis."""

    coef: np.ndarray  # float32 (nnz,)
    bins: np.ndarray  # int32   (nnz,)     q-bin index per entry
    indptr: np.ndarray  # int32   (NPIX+1,)  column (= pixel) starts
    omega: np.ndarray  # float64 (NPIX,)   relative solid angle
    q: np.ndarray  # float64 (npt,)    bin centres
    npt: int
    nnz: int
    shape: tuple[int, int]
    method: tuple[str, str, str]
    sdd_m: float
    wavelength_m: float
    sha256: str


def _readonly(array: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    """Return a contiguous, read-only copy, asserting the dtype is unchanged.

    The dtype is asserted rather than cast: a pyFAI change to the LUT layout
    must surface loudly instead of being silently converted.
    """
    if array.dtype != dtype:
        raise TypeError(f"expected dtype {np.dtype(dtype)}, got {array.dtype}")
    out = np.ascontiguousarray(array).copy()
    out.flags.writeable = False
    return out


def operator_sha256(
    coef: np.ndarray,
    bins: np.ndarray,
    indptr: np.ndarray,
    omega: np.ndarray,
    q: np.ndarray,
    *,
    npt: int,
    shape: tuple[int, int],
    method: tuple[str, str, str],
    sdd_m: float,
    wavelength_m: float,
) -> str:
    """sha256 over the operator arrays and the geometry scalars that set them.

    Stable across rebuilds of the same geometry; any change to the geometry,
    the distance, the wavelength, ``npt`` or the method changes it.
    """
    header = json.dumps(
        {
            "npt": npt,
            "shape": list(shape),
            "method": list(method),
            "sdd_m": sdd_m,
            "wavelength_m": wavelength_m,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(header.encode())
    for array in (coef, bins, indptr, omega, q):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def geometry_from_config(cfg: FirstPassConfig) -> AGIPD_1MGeometry:
    """Load the CrystFEL geometry named by ``cfg`` (context file §3 rule 4)."""
    if cfg.geometry_file is None:
        raise ValueError(
            "cfg.geometry_file is None; pass a geometry object to "
            "build_operator directly (P1 synthetic geometry)"
        )
    return AGIPD_1MGeometry.from_crystfel_geom(cfg.geometry_file)


def build_operator(
    geom: AGIPD_1MGeometry, cfg: FirstPassConfig
) -> tuple[SparseOperator, AzimuthalIntegrator]:
    """Build the sparse operator from a geometry object.

    The geometry object is the single source of detector positions: PONI is 0
    at the geometry origin, and there is no ``setFit2D``, no second
    ``set_pixel_corners`` and no PONI file (context file §3 rule 4).

    The integrator is returned alongside the operator because the self-test
    needs the live compiled engine, which the saved ``.npz`` cannot provide.
    """
    ai = AzimuthalIntegrator(
        detector=geom.to_pyfai_detector(),
        dist=cfg.sdd_m,
        wavelength=cfg.wavelength_m,
    )
    probe = ai.integrate1d(
        np.ones(SHAPE, dtype=np.float32),
        cfg.npt,
        method=cfg.method,
        unit=cfg.unit,
    )
    resolved = (
        probe.method.split_lower,
        probe.method.algo_lower,
        probe.method.impl_lower,
    )
    if resolved != tuple(cfg.method):
        raise RuntimeError(
            f"pyFAI resolved method {resolved}, requested {tuple(cfg.method)}; "
            "a method string or an unavailable engine has silently substituted "
            "another integrator (CLAUDE.md pitfall 1)"
        )

    engine = ai.engines[probe.method].engine
    raw_coef, raw_bins, raw_indptr = engine.lut
    coef = _readonly(raw_coef, np.float32)
    bins = _readonly(raw_bins, np.int32)
    indptr = _readonly(raw_indptr, np.int32)

    if indptr.size != NPIX + 1:
        raise RuntimeError(
            f"CSC indptr has {indptr.size} entries, expected NPIX+1 = {NPIX + 1}; "
            "the operator is not one column per pixel"
        )
    if coef.size != bins.size or coef.size != engine.nnz:
        raise RuntimeError(
            f"LUT size mismatch: coef {coef.size}, bins {bins.size}, "
            f"engine.nnz {engine.nnz}"
        )
    if bins.size and (bins.min() < 0 or bins.max() >= cfg.npt):
        raise RuntimeError(
            f"CSC bin indices out of range [0, {cfg.npt}): [{bins.min()}, {bins.max()}]"
        )

    omega = np.ascontiguousarray(ai.solidAngleArray(SHAPE), dtype=np.float64).ravel()
    omega.flags.writeable = False
    q = _readonly(
        np.ascontiguousarray(engine.bin_centers, dtype=np.float64), np.float64
    )
    if q.size != cfg.npt:
        raise RuntimeError(f"engine returned {q.size} bin centres, expected {cfg.npt}")

    operator = SparseOperator(
        coef=coef,
        bins=bins,
        indptr=indptr,
        omega=omega,
        q=q,
        npt=cfg.npt,
        nnz=int(engine.nnz),
        shape=SHAPE,
        method=tuple(cfg.method),
        sdd_m=cfg.sdd_m,
        wavelength_m=cfg.wavelength_m,
        sha256=operator_sha256(
            coef,
            bins,
            indptr,
            omega,
            q,
            npt=cfg.npt,
            shape=SHAPE,
            method=tuple(cfg.method),
            sdd_m=cfg.sdd_m,
            wavelength_m=cfg.wavelength_m,
        ),
    )
    return operator, ai


def save_operator(op: SparseOperator, path: str | Path) -> Path:
    """Write the operator to ``path`` as an ``.npz`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(
        {
            "npt": op.npt,
            "nnz": op.nnz,
            "shape": list(op.shape),
            "method": list(op.method),
            "sdd_m": op.sdd_m,
            "wavelength_m": op.wavelength_m,
            "sha256": op.sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    np.savez(
        path,
        coef=op.coef,
        bins=op.bins,
        indptr=op.indptr,
        omega=op.omega,
        q=op.q,
        meta=np.array(meta),
    )
    return path


def load_operator(path: str | Path) -> SparseOperator:
    """Load an operator written by :func:`save_operator`, verifying its hash."""
    with np.load(path) as handle:
        meta = json.loads(str(handle["meta"]))
        coef = _readonly(handle["coef"], np.float32)
        bins = _readonly(handle["bins"], np.int32)
        indptr = _readonly(handle["indptr"], np.int32)
        omega = _readonly(handle["omega"], np.float64)
        q = _readonly(handle["q"], np.float64)

    op = SparseOperator(
        coef=coef,
        bins=bins,
        indptr=indptr,
        omega=omega,
        q=q,
        npt=int(meta["npt"]),
        nnz=int(meta["nnz"]),
        shape=(int(meta["shape"][0]), int(meta["shape"][1])),
        method=(meta["method"][0], meta["method"][1], meta["method"][2]),
        sdd_m=float(meta["sdd_m"]),
        wavelength_m=float(meta["wavelength_m"]),
        sha256=str(meta["sha256"]),
    )
    recomputed = operator_sha256(
        op.coef,
        op.bins,
        op.indptr,
        op.omega,
        op.q,
        npt=op.npt,
        shape=op.shape,
        method=op.method,
        sdd_m=op.sdd_m,
        wavelength_m=op.wavelength_m,
    )
    if recomputed != op.sha256:
        raise ValueError(
            f"operator at {path} is corrupt: stored sha256 {op.sha256}, "
            f"recomputed {recomputed}"
        )
    return op
