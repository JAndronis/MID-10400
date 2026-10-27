"""Geometry loading, the method assertion and the operator hash."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("pyFAI")
pytest.importorskip("fabio")

from waxs_mockrun import write_synthetic_poni  # noqa: E402

from analysis.waxs.config import MODULE_SHAPE  # noqa: E402
from analysis.waxs.masks import build_static_bad, load_static_mask  # noqa: E402
from analysis.waxs.operator import (  # noqa: E402
    WavelengthMismatch,
    build_operator,
)


def test_a_poni_refined_at_another_wavelength_is_refused(cfg, tmp_path):
    """The PONI carries its own wavelength, and q scales with it."""
    wrong = write_synthetic_poni(tmp_path / "wrong.poni", photon_energy_kev=9.000)
    with pytest.raises(WavelengthMismatch, match="q scales"):
        build_operator(dataclasses.replace(cfg, poni_file=str(wrong)))
    # ...while one at the configured energy passes at float32 precision.
    close = write_synthetic_poni(tmp_path / "close.poni", photon_energy_kev=9.04)
    op, _ = build_operator(dataclasses.replace(cfg, poni_file=str(close)))
    assert op.wavelength_m == pytest.approx(cfg.wavelength_m, rel=1e-7)


def test_a_detector_of_the_wrong_shape_is_refused(cfg, tmp_path):
    from pyFAI.detectors import Pilatus1M
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    ai = AzimuthalIntegrator(
        detector=Pilatus1M(), dist=0.232, wavelength=cfg.wavelength_m
    )
    path = tmp_path / "pilatus.poni"
    ai.save(str(path))
    with pytest.raises(ValueError, match="JUNGFRAU-500K module"):
        build_operator(dataclasses.replace(cfg, poni_file=str(path)))


def test_the_hash_is_stable_and_moves_with_npt(cfg):
    first, _ = build_operator(cfg)
    again, _ = build_operator(cfg)
    assert first.sha256 == again.sha256
    coarser, _ = build_operator(dataclasses.replace(cfg, npt=cfg.npt // 2))
    assert coarser.sha256 != first.sha256


# ── the static mask ──────────────────────────────────────────────────────────
def test_the_edf_is_used_as_is_with_no_inversion(cfg, geometry_files):
    """The ``.edf`` files are native pyFAI masks, non-zero = excluded."""
    import fabio

    with fabio.open(str(geometry_files.mask)) as image:
        raw = np.asarray(image.data)
    loaded = load_static_mask(geometry_files.mask)
    assert np.array_equal(loaded, raw.reshape(-1) != 0)

    static = build_static_bad(cfg)
    assert static.n_excluded == int((raw != 0).sum())
    assert [s.name for s in static.sources] == ["static_mask"]
    assert static.sources[0].sha256


def test_a_mask_excluding_everything_is_refused(tmp_path):
    from waxs_mockrun import MODULE_SHAPE as SHAPE
    from waxs_mockrun import write_static_mask

    path = write_static_mask(tmp_path / "all.edf", np.ones(SHAPE, dtype=np.uint8))
    with pytest.raises(ValueError, match="excludes every pixel"):
        load_static_mask(path)


# ── nothing we hand pyFAI, or freeze, may be an array pyFAI owns ─────────────
def test_pyfai_keeps_writable_buffers_and_the_operator_frozen_copies(operator, cfg):
    """The bug behind ``ValueError: buffer source array is read-only``.

    ``np.ascontiguousarray(x, dtype)`` returns ``x`` itself when it is already
    contiguous and of that dtype, so freezing the result freezes an array pyFAI
    still owns (its ``_dssa`` solid-angle cache, or an engine's bin centres).
    pyFAI's Cython kernels acquire writable buffers, so the damage surfaces
    later, in an unrelated call, with a message that names neither. Whether
    pyFAI accepts a read-only mask depends on the platform, so the mask handed
    to it every frame is writable.
    """
    op, ai = operator

    assert ai._dssa.flags.writeable
    for engine in ai.engines.values():
        for attribute in ("bin_centers", "bin_centers0", "bin_centers1"):
            array = getattr(engine.engine, attribute, None)
            if isinstance(array, np.ndarray):
                assert array.flags.writeable, attribute
                assert array is not op.q
    # The pass's own outputs are frozen copies that own their memory.
    for frozen in (op.q, op.omega, op.static_bad):
        assert not frozen.flags.writeable
    assert op.q.base is None and op.omega.base is None
    assert op.static_mask_2d.flags.writeable
    assert op.static_mask_2d.flags.c_contiguous
    assert np.array_equal(op.static_mask_2d.reshape(-1), op.static_bad)

    frame = np.ones(MODULE_SHAPE, dtype=np.float32)
    for _ in range(3):
        result = ai.integrate1d(
            frame.copy(),
            cfg.npt,
            method=op.method,
            unit=op.unit,
            mask=op.static_mask_2d,
            variance=frame.copy(),
        )
        assert np.isfinite(result.sum_signal).all()
    assert len(ai.engines) == 1
