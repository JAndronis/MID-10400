"""Operator build, hashing and round-trip."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from analysis.saxs.config import METHOD, NPIX, AgipdSaxsConfig
from analysis.saxs.operator import build_operator, load_operator, save_operator

extra_geom = pytest.importorskip("extra_geom")


def test_method_resolves_to_full_csc_cython(op, engine):
    """The requested method tuple must be the one that ran.

    A pyFAI method *string* can silently resolve to a no-split NumPy
    histogram, so the build asserts the resolved tuple, and the engine that
    holds the LUT must be the full-split CSC one.
    """
    assert op.method == METHOD
    assert type(engine).__module__ == "pyFAI.ext.splitPixelFullCSC"


def test_operator_is_a_csc_matrix_with_one_column_per_pixel(op, cfg):
    assert op.indptr.shape == (NPIX + 1,)
    assert op.indptr[0] == 0
    assert op.indptr[-1] == op.nnz
    assert np.all(np.diff(op.indptr) >= 0)
    assert op.coef.shape == op.bins.shape == (op.nnz,)
    assert op.bins.min() >= 0
    assert op.bins.max() < cfg.npt
    assert op.omega.shape == (NPIX,)
    assert op.q.shape == (cfg.npt,)
    assert np.all(np.diff(op.q) > 0)


def test_operator_arrays_are_read_only(op):
    """The operator is shared across workers; nothing may mutate it in place."""
    for array in (op.coef, op.bins, op.indptr, op.omega, op.q):
        assert not array.flags.writeable


def test_operator_hash_is_stable_and_moves_with_the_geometry(quad_pos, cfg, op):
    """A rebuild reproduces every array; one quadrant moved by one pixel, the
    smallest change the constructor can express, changes the hash."""
    rebuilt, _ = build_operator(
        extra_geom.AGIPD_1MGeometry.from_quad_positions(quad_pos=quad_pos), cfg
    )
    assert rebuilt.sha256 == op.sha256
    for name in ("coef", "bins", "indptr", "omega", "q"):
        assert np.array_equal(getattr(rebuilt, name), getattr(op, name)), name
    moved = [*quad_pos[:3], (quad_pos[3][0], quad_pos[3][1] + 1)]
    shifted, _ = build_operator(
        extra_geom.AGIPD_1MGeometry.from_quad_positions(quad_pos=moved), cfg
    )
    assert shifted.sha256 != op.sha256


def test_load_rejects_a_corrupt_operator(op, tmp_path):
    """A tampered file must raise, never load silently."""
    path = save_operator(op, tmp_path / "operator.npz")
    with np.load(path) as handle:
        arrays = {name: handle[name] for name in handle.files}
    arrays["coef"] = arrays["coef"].copy()
    arrays["coef"][0] += np.float32(1.0)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="corrupt"):
        load_operator(path)


def test_config_rejects_a_non_full_split_method():
    with pytest.raises(ValueError, match="method must be"):
        AgipdSaxsConfig(proposal=10400, run=423, method=("no", "csr", "cython"))


# ── beam centre ───────────────────────────────────────────────────────────────
# `cfg.beam_center` is expressed in the `to_distortion_array()` frame, not the
# `to_pyfai_detector()` one PONI = 0 lives in, so `build_operator` has to
# install that corner array before calling setFit2D. These pin that pairing:
# dropping either half moves the beam by the offset between the two origins
# without raising anything.


@pytest.fixture(scope="module")
def centred_cfg(cfg):
    """``cfg`` with the beamtime's agreed beam centre."""
    from analysis.saxs.config import DEFAULT_BEAM_CENTER_PX, DEFAULT_BEAM_CENTER_PY

    return replace(
        cfg,
        beam_center_px=DEFAULT_BEAM_CENTER_PX,
        beam_center_py=DEFAULT_BEAM_CENTER_PY,
    )


def test_half_a_beam_centre_is_rejected():
    """One axis set and the other None would silently mean PONI = 0 in it."""
    for half in ({"beam_center_px": 607.46}, {"beam_center_py": 672.08}):
        other = "beam_center_py" if "px" in next(iter(half)) else "beam_center_px"
        with pytest.raises(ValueError, match="together"):
            AgipdSaxsConfig(
                proposal=10400,
                run=423,
                geometry_file=None,
                pixel_mask_file=None,
                **{other: None},
                **half,
            )


def test_save_load_round_trips_a_centred_operator(geom, centred_cfg, tmp_path):
    centred, _ = build_operator(geom, centred_cfg)
    loaded = load_operator(save_operator(centred, tmp_path / "centred.npz"))
    assert loaded.sha256 == centred.sha256
    assert loaded.beam_center == centred.beam_center
    for name in ("npt", "nnz", "shape", "method", "sdd_m", "wavelength_m"):
        assert getattr(loaded, name) == getattr(centred, name), name
    for name in ("coef", "bins", "indptr", "omega", "q"):
        assert np.array_equal(getattr(loaded, name), getattr(centred, name)), name


def test_build_operator_reproduces_extra_speckle_configsaxs(geom, centred_cfg):
    """Cross-check against the implementation the centre was derived against.

    ``extra_speckle.setup.configuration.ConfigSAXS`` is what produced the
    numbers in ``config.py``. If the two constructions diverge (a changed
    corner-array origin, a Fit2D argument order, a millimetre), the q each
    pixel is assigned to moves, and this test catches it.
    """
    configuration = pytest.importorskip(
        "extra_speckle.setup.configuration",
        reason="cross-check needs extra-speckle",
    )
    from types import SimpleNamespace

    ours, ai_ours = build_operator(geom, centred_cfg)
    theirs = configuration.ConfigSAXS(
        # ConfigSAXS reads only `.detector` and `.poni` off this.
        detector=SimpleNamespace(detector=geom.to_pyfai_detector(), poni=None),
        sdd=centred_cfg.sdd_m * 1e3,
        wavelength=centred_cfg.wavelength_m,
        px=centred_cfg.beam_center_px,
        py=centred_cfg.beam_center_py,
    )
    # DetectorAGIPD1M installs the distortion array before ConfigSAXS runs;
    # the stand-in above skips that, so do it here for the same reason.
    theirs.ai.detector.set_pixel_corners(geom.to_distortion_array())

    np.testing.assert_allclose(
        ai_ours.array_from_unit(unit=centred_cfg.unit),
        theirs.ai.array_from_unit(unit=centred_cfg.unit),
        rtol=0,
        atol=0,
    )
    assert ai_ours.dist == pytest.approx(theirs.ai.dist)
    assert (ai_ours.poni1, ai_ours.poni2) == pytest.approx(
        (theirs.ai.poni1, theirs.ai.poni2)
    )
    assert ours.q.shape == (centred_cfg.npt,)


def test_each_pixel_sits_at_the_q_of_its_distance_from_the_beam_centre(
    geom, centred_cfg
):
    """An analytic oracle for the beam-centre convention, with no pyFAI in it.

    ``cfg.beam_center`` is in pixels of the ``to_distortion_array()`` frame,
    whose origin is the corner of the assembled bounding box. Measured from
    there, a pixel at in-plane distance r from the beam scatters at
    2θ = atan(r / sdd), and its operator column must put its weight at
    q = 4π sin θ / λ, to within a bin.
    """
    op, _ = build_operator(geom, centred_cfg)
    corners = geom.to_distortion_array().reshape(NPIX, 4, 3)
    centres = corners[:, :, 1:].mean(axis=1)  # (y, x) in metres
    pixel = geom.to_pyfai_detector().pixel1
    beam = np.array(
        [centred_cfg.beam_center_py * pixel, centred_cfg.beam_center_px * pixel]
    )
    r = np.hypot(*(centres - beam).T)
    theta = np.arctan2(r, centred_cfg.sdd_m) / 2
    q_nm = 4 * np.pi * np.sin(theta) / (centred_cfg.wavelength_m * 1e9)

    bin_width = np.diff(op.q).mean()
    sample = np.random.default_rng(0).choice(NPIX, 200, replace=False)
    sample = sample[op.indptr[sample + 1] > op.indptr[sample]]
    assert sample.size > 150
    for p in sample:
        cols = slice(op.indptr[p], op.indptr[p + 1])
        q_op = np.average(op.q[op.bins[cols]], weights=op.coef[cols])
        assert q_op == pytest.approx(q_nm[p], abs=bin_width), p
