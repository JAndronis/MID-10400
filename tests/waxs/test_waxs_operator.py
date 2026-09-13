"""Geometry loading, the method assertion and the operator hash (W2)."""

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


def test_operator_carries_a_populated_q_axis(operator, cfg):
    op, ai = operator
    assert op.q.shape == (cfg.npt,)
    assert np.all(np.diff(op.q) > 0)
    assert op.unit == "q_nm^-1"
    assert op.shape == MODULE_SHAPE
    assert op.omega.shape == (MODULE_SHAPE[0] * MODULE_SHAPE[1],)


def test_the_engine_is_the_requested_one(operator):
    """CLAUDE.md pitfall 1: a method string can silently resolve elsewhere."""
    op, ai = operator
    assert op.method == ("full", "csc", "cython")
    assert len(ai.engines) == 1


def test_the_wavelength_assertion_fires_on_a_mismatched_poni(cfg, tmp_path):
    """§5 R4: the PONI carries its own wavelength, and q scales with it."""
    wrong = write_synthetic_poni(tmp_path / "wrong.poni", photon_energy_kev=9.000)
    with pytest.raises(WavelengthMismatch, match="q scales"):
        build_operator(dataclasses.replace(cfg, poni_file=str(wrong)))


def test_a_matching_poni_is_accepted_at_float32_precision(cfg, tmp_path):
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


def test_the_poni_text_round_trips(operator):
    """The file is stored verbatim so the geometry survives the file moving."""
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    op, ai = operator
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "again.poni"
        path.write_text(op.poni_text)
        rebuilt = AzimuthalIntegrator.sload(path)
    assert rebuilt.dist == pytest.approx(ai.dist)
    assert rebuilt.wavelength == pytest.approx(ai.wavelength)


# ── the static mask ──────────────────────────────────────────────────────────
def test_the_edf_is_used_as_is_with_no_inversion(cfg, geometry_files):
    """§6 O2: native pyFAI masks, non-zero = excluded."""
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


def test_a_mask_of_the_wrong_shape_is_refused(tmp_path):
    from waxs_mockrun import write_static_mask

    path = write_static_mask(tmp_path / "small.edf", np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="expected"):
        load_static_mask(path)


def test_masking_reduces_the_denominator(cfg, operator):
    """The polarity test from §6 O2: masking half the module halves N."""
    from analysis.waxs.config import MODULE_SHAPE as SHAPE

    op, ai = operator
    frame = np.ones(SHAPE, dtype=np.float32)
    unmasked = ai.integrate1d(
        frame, cfg.npt, method=op.method, unit=op.unit, variance=frame
    )
    half = np.zeros(SHAPE, dtype=np.uint8)
    half[: SHAPE[0] // 2] = 1
    masked = ai.integrate1d(
        frame, cfg.npt, method=op.method, unit=op.unit, mask=half, variance=frame
    )
    ratio = masked.sum_normalization.sum() / unmasked.sum_normalization.sum()
    assert 0.4 < ratio < 0.6


# ── nothing we hand pyFAI, or freeze, may be an array pyFAI owns ─────────────
def test_building_the_operator_leaves_pyfais_own_arrays_writable(operator):
    """The bug behind ``ValueError: buffer source array is read-only``.

    ``np.ascontiguousarray(x, dtype)`` returns ``x`` itself when it is already
    contiguous and of that dtype, so freezing the result freezes an array pyFAI
    still owns — its ``_dssa`` solid-angle cache, or an engine's bin centres.
    pyFAI's Cython kernels acquire writable buffers, so the damage surfaces
    later, in an unrelated call, with a message that names neither.
    """
    op, ai = operator

    assert ai._dssa.flags.writeable
    for engine in ai.engines.values():
        for attribute in ("bin_centers", "bin_centers0", "bin_centers1"):
            array = getattr(engine.engine, attribute, None)
            if isinstance(array, np.ndarray):
                assert array.flags.writeable, attribute
                assert array is not op.q


def test_the_frozen_outputs_are_copies_not_views(operator):
    op, ai = operator
    assert not op.q.flags.writeable
    assert not op.omega.flags.writeable
    # A copy owns its memory, so it has no base to have been carved from.
    assert op.q.base is None
    assert op.omega.base is None


def test_the_mask_handed_to_pyfai_is_writable(operator):
    """pyFAI tolerates a read-only mask today; relying on that is not a plan."""
    op, ai = operator
    assert op.static_mask_2d.flags.writeable
    assert op.static_mask_2d.flags.c_contiguous
    assert op.static_mask_2d.shape == MODULE_SHAPE
    assert np.array_equal(op.static_mask_2d.reshape(-1), op.static_bad)
    # ...while the flat one stays frozen, because the pass treats it as a fact.
    assert not op.static_bad.flags.writeable


def test_the_operator_survives_repeated_use(operator, cfg):
    """A second integration on the same integrator must still work."""
    op, ai = operator
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


def test_a_read_only_signal_is_what_pyfai_actually_rejects(operator, cfg):
    """Pins the failure mode, so the guard above has a stated reason."""
    op, ai = operator
    frozen = np.zeros(MODULE_SHAPE, dtype=np.float32)
    frozen.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        ai.integrate1d(
            frozen,
            cfg.npt,
            method=op.method,
            unit=op.unit,
            mask=op.static_mask_2d,
            variance=np.zeros(MODULE_SHAPE, dtype=np.float32),
        )
