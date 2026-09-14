"""Geometry, the cached pyFAI engine and the q axis.

**Why there is no sparse kernel here.** The AGIPD pass lifts pyFAI's CSC matrix
out of the engine and integrates over photon hits, because its frames are
0.7–1.8 % non-zero. JUNGFRAU frames are dense — 0.005 % of pixels are exactly
zero and a fifth are negative — so this pass runs dense pyFAI per frame.

**Why the dynamic mask is a NaN, not a ``mask=`` argument.** pyFAI keys its
cached sparse matrix on a checksum of the mask, so a mask that changes per frame
rebuilds the full-split CSC matrix every frame, which its own docstring calls
"a very time consuming operation". Building the engine **once** with the ``.edf``
static mask and carrying the per-frame mask as NaN in the data and variance
arrays is several times faster and gives bit-identical sums: pyFAI's
preprocessing drops a non-finite pixel from the numerator *and* the
normalisation, which is the exact-denominator property AGIPD has to reconstruct
sparsely.

That option exists here and not there because JUNGFRAU frames are ``float32``.
AGIPD's ``int16`` cannot carry NaN, which is what forced the sparse correction.
:mod:`analysis.waxs.selftest` gates the equivalence on real frames so a change
in pyFAI's NaN handling cannot pass silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

from analysis.common.arrays import frozen_copy
from analysis.common.cpu import file_sha256
from analysis.common.pyfai import check_resolved_method
from analysis.waxs.config import MODULE_SHAPE, JungfrauWaxsConfig
from analysis.waxs.masks import build_static_bad

__all__ = [
    "WaxsOperator",
    "WavelengthMismatch",
    "build_operator",
    "operator_sha256",
]


class WavelengthMismatch(ValueError):
    """The PONI was refined at a different photon energy than the config names."""


@dataclass(frozen=True, slots=True)
class WaxsOperator:
    """The geometry, the static mask and the q axis they produce.

    Deliberately small: unlike AGIPD's ``SparseOperator`` this carries no
    matrix, because the matrix lives inside the pyFAI engine that
    :func:`build_operator` returns alongside it. The PONI text is carried
    verbatim so a worker — or a reader months later — can rebuild the identical
    geometry from the output file alone.
    """

    q: np.ndarray  # float64 (npt,)   bin centres, cfg.unit
    static_bad: np.ndarray  # bool    (npix,)  non-zero = excluded, read-only
    #: The same mask as ``static_bad``, 2-D and **writable**, for handing to
    #: pyFAI. Built once and reused, so the hot loop neither copies per frame
    #: nor passes pyFAI a read-only buffer.
    static_mask_2d: np.ndarray
    omega: np.ndarray  # float64 (npix,)  relative solid angle
    poni_text: str
    shape: tuple[int, int]
    npt: int
    method: tuple[str, str, str]
    unit: str
    dist_m: float
    wavelength_m: float
    poni_sha256: str | None
    static_sha256: str
    sha256: str

    @property
    def q_populated(self) -> tuple[float, float]:
        """The q range the kept pixels actually cover."""
        return float(self.q.min()), float(self.q.max())


def operator_sha256(
    q: np.ndarray,
    omega: np.ndarray,
    static_bad: np.ndarray,
    *,
    npt: int,
    shape: tuple[int, int],
    method: tuple[str, str, str],
    unit: str,
    dist_m: float,
    wavelength_m: float,
) -> str:
    """sha256 over the q axis, the solid angle, the mask and the scalars.

    Stable across rebuilds of the same PONI; any change to the geometry, the
    distance, the wavelength, the mask, ``npt``, the unit or the method changes
    it.
    """
    header = json.dumps(
        {
            "npt": npt,
            "shape": list(shape),
            "method": list(method),
            "unit": unit,
            "dist_m": dist_m,
            "wavelength_m": wavelength_m,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(header.encode())
    for array in (q, omega, np.packbits(static_bad)):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def build_operator(
    cfg: JungfrauWaxsConfig, poni_file: str | Path | None = None
) -> tuple[WaxsOperator, AzimuthalIntegrator]:
    """Load the PONI, build the cached engine and record the q axis.

    :param poni_file: overrides ``cfg.poni_file``; the tests pass a synthetic
        PONI written to a temporary directory.

    :raises WavelengthMismatch: the PONI's wavelength and
        ``cfg.photon_energy_kev`` disagree by more than float32 precision. The
        PONI carries its own wavelength and a mismatch means it was refined at
        a different energy, so neither value may silently win: q scales with it.
    """
    path = Path(poni_file if poni_file is not None else cfg.poni_file or "")
    if not path.name:
        raise ValueError(
            "no PONI file: set cfg.poni_file, or pass poni_file= directly "
            "(which is what the unit tests do with a synthetic geometry)"
        )
    ai = AzimuthalIntegrator.sload(path)

    if tuple(ai.detector.shape) != MODULE_SHAPE:
        raise ValueError(
            f"{path} describes a detector of shape {tuple(ai.detector.shape)}, "
            f"expected a JUNGFRAU-500K module {MODULE_SHAPE}"
        )
    relative = abs(ai.wavelength - cfg.wavelength_m) / cfg.wavelength_m
    if relative > float(np.finfo(np.float32).eps):
        raise WavelengthMismatch(
            f"{path} was refined at wavelength {ai.wavelength:.6e} m, but "
            f"cfg.photon_energy_kev {cfg.photon_energy_kev} keV means "
            f"{cfg.wavelength_m:.6e} m ({100 * relative:.3f} % apart); q scales "
            "with it, so settle which is authoritative rather than overriding"
        )

    static = build_static_bad(cfg)
    # Writable and C-contiguous, because it is handed to pyFAI on every frame.
    mask_2d = np.array(static.bad.reshape(MODULE_SHAPE), dtype=bool, order="C")

    # A zero frame and a zero variance: this call exists only to build and
    # cache the sparse matrix and to read the bin centres off it. Two separate
    # buffers, not one aliased twice, so pyFAI is never handed the same array
    # as both signal and variance.
    probe = ai.integrate1d(
        np.zeros(MODULE_SHAPE, dtype=np.float32),
        cfg.npt,
        method=cfg.method,
        unit=cfg.unit,
        mask=mask_2d,
        variance=np.zeros(MODULE_SHAPE, dtype=np.float32),
    )
    check_resolved_method(probe, cfg.method)

    q = frozen_copy(probe.radial, np.float64, cast=True)
    if q.size != cfg.npt:
        raise RuntimeError(f"pyFAI returned {q.size} bin centres, expected {cfg.npt}")
    omega = frozen_copy(
        np.asarray(ai.solidAngleArray(MODULE_SHAPE)).reshape(-1), np.float64, cast=True
    )

    operator = WaxsOperator(
        q=q,
        static_bad=static.bad,
        static_mask_2d=mask_2d,
        omega=omega,
        poni_text=path.read_text(),
        shape=MODULE_SHAPE,
        npt=cfg.npt,
        method=tuple(cfg.method),
        unit=cfg.unit,
        dist_m=float(ai.dist),
        wavelength_m=float(ai.wavelength),
        poni_sha256=file_sha256(path),
        static_sha256=static.sha256,
        sha256=operator_sha256(
            q,
            omega,
            static.bad,
            npt=cfg.npt,
            shape=MODULE_SHAPE,
            method=tuple(cfg.method),
            unit=cfg.unit,
            dist_m=float(ai.dist),
            wavelength_m=float(ai.wavelength),
        ),
    )
    return operator, ai
