from __future__ import annotations

import functools

import numpy as np
from extra_data import open_run
from extra_data.components import AGIPD1M
from extra_geom import AGIPD_1MGeometry
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

GEOM_PATH = "/gpfs/exfel/exp/MID/202601/p010400/usr/geometry/geom_latest.geom"
MASK_PATH = (
    "/gpfs/exfel/exp/MID/202601/p010400/usr/masks/mask_2026-05-11_AGIPD_updated.npy"
)
SDD_MM = 7531.5962
WAVELENGTH_M = 1.3715e-10

NPT = 1000
METHOD = ("no", "csr", "cython")
UNIT = "q_nm^-1"


@functools.lru_cache(maxsize=1)
def _mask() -> np.ndarray:
    m = np.load(MASK_PATH).reshape(16 * 512, 128)
    assert m.shape == (8192, 128), m.shape
    return np.ascontiguousarray(m, dtype=np.int8)


CENTRE_XY_PX: tuple[float, float] = (607.4598195630211, 672.076693118667)
CENTRE_PROVENANCE: str = (
    "ring fit on assembled images, IA + <beamline scientist>, 2026-08"
)


def _beam_offset_m(geom: AGIPD_1MGeometry) -> tuple[float, float]:
    """Lab-frame shift (x, y) in metres that moves the geometry origin onto
    CENTRE_XY_PX."""
    probe = 10e-3  # large enough that snapping quantisation doesn't dominate

    c0 = np.asarray(geom._snapped().centre, dtype=float)  # (y, x) px
    jx = (np.asarray(geom.offset((probe, 0))._snapped().centre) - c0) / probe
    jy = (np.asarray(geom.offset((0, probe))._snapped().centre) - c0) / probe
    jac = np.column_stack([jx, jy])  # d(centre_px)/d(shift_m)

    cx_target, cy_target = CENTRE_XY_PX
    delta = np.array([cy_target, cx_target]) - c0  # (y, x) px
    shift = np.linalg.solve(jac, delta)
    return float(shift[0]), float(shift[1])


@functools.lru_cache(maxsize=1)
def _ai() -> tuple[AzimuthalIntegrator, np.ndarray, tuple[float, float]]:
    raw = AGIPD_1MGeometry.from_crystfel_geom(GEOM_PATH)
    shift = _beam_offset_m(raw)
    geom = raw.offset(shift)

    ai = AzimuthalIntegrator(
        detector=geom.to_pyfai_detector(),
        dist=SDD_MM * 1e-3,
        wavelength=WAVELENGTH_M,
        poni1=0.0,
        poni2=0.0,
    )

    res = ai.integrate1d(
        np.zeros((8192, 128), dtype=np.float32),
        NPT,
        mask=_mask(),
        unit=UNIT,
        method=METHOD,
    )
    return ai, np.asarray(res.radial, dtype=np.float32), shift


def integrate_part(
    proposal: int, run_no: int, sl: slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ai, _, _ = _ai()
    mask = _mask()

    det = AGIPD1M(open_run(proposal, run_no, data="proc"), min_modules=16)
    xarr = det.select_trains(sl).get_array("image.data")
    assert xarr.dims == ("module", "train", "pulse", "slow_scan", "fast_scan"), (
        xarr.dims
    )

    data = xarr.values  # (16, n_train, n_pulse, 512, 128)
    n_train, n_pulse = data.shape[1], data.shape[2]

    train_id = np.repeat(xarr.train.values, n_pulse)  # C order: train-major
    pulse_id = np.tile(xarr.pulse.values, n_train)

    out = np.empty((n_train * n_pulse, NPT), dtype=np.float32)
    for k, (t, p) in enumerate(np.ndindex(n_train, n_pulse)):
        res = ai.integrate1d(
            np.ascontiguousarray(data[:, t, p]).reshape(-1, data.shape[-1]),
            NPT,
            mask=mask,
            unit=UNIT,
            method=METHOD,
        )
        out[k] = res.intensity

    return train_id, pulse_id, out
