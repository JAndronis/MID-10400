"""Operator build, hashing and round-trip (context file §6.1)."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from analysis.saxs.config import METHOD, NPIX, SHAPE, AgipdSaxsConfig
from analysis.saxs.operator import build_operator, load_operator, save_operator

extra_geom = pytest.importorskip("extra_geom")


def test_method_resolves_to_full_csc_cython(op, engine):
    """§3 rule 1: the requested tuple must be the one that actually ran.

    A method *string* silently resolves to a no-split NumPy histogram
    (CLAUDE.md pitfall 1), so the build asserts the resolved tuple and the
    engine that carries the LUT is the full-split CSC one.
    """
    assert op.method == METHOD
    assert type(engine).__module__ == "pyFAI.ext.splitPixelFullCSC"


def test_operator_shapes_and_dtypes(op, cfg):
    assert op.shape == SHAPE
    assert op.indptr.shape == (NPIX + 1,)  # CSC column per pixel
    assert op.coef.shape == op.bins.shape == (op.nnz,)
    assert op.omega.shape == (NPIX,)
    assert op.q.shape == (cfg.npt,)
    assert (op.coef.dtype, op.bins.dtype) == (np.float32, np.int32)
    assert (op.indptr.dtype, op.omega.dtype, op.q.dtype) == (
        np.int32,
        np.float64,
        np.float64,
    )
    assert op.bins.min() >= 0
    assert op.bins.max() < cfg.npt
    assert np.all(np.diff(op.indptr) >= 0)
    assert op.indptr[0] == 0
    assert op.indptr[-1] == op.nnz
    assert np.all(np.diff(op.q) > 0)


def test_operator_arrays_are_read_only(op):
    """The operator is shared across workers; nothing may mutate it in place."""
    for array in (op.coef, op.bins, op.indptr, op.omega, op.q):
        assert not array.flags.writeable


def test_operator_hash_stable_across_rebuilds(quad_pos, cfg, op):
    """P1 acceptance: the hash is stable across rebuilds."""
    rebuilt, _ = build_operator(
        extra_geom.AGIPD_1MGeometry.from_quad_positions(quad_pos=quad_pos), cfg
    )
    assert rebuilt.sha256 == op.sha256
    assert np.array_equal(rebuilt.coef, op.coef)
    assert np.array_equal(rebuilt.bins, op.bins)
    assert np.array_equal(rebuilt.indptr, op.indptr)
    assert np.array_equal(rebuilt.omega, op.omega)


def test_operator_hash_changes_with_geometry(quad_pos, cfg, op):
    """P1 acceptance: a changed geometry changes the hash.

    One quadrant moves by a single pixel, the smallest change the constructor
    can express.
    """
    moved = [*quad_pos[:3], (quad_pos[3][0], quad_pos[3][1] + 1)]
    shifted, _ = build_operator(
        extra_geom.AGIPD_1MGeometry.from_quad_positions(quad_pos=moved), cfg
    )
    assert shifted.sha256 != op.sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [("sdd_m", 7.6), ("photon_energy_kev", 9.1), ("npt", 400)],
)
def test_operator_hash_changes_with_config(geom, cfg, op, field, value):
    """Distance, wavelength and binning all change the operator."""
    changed, _ = build_operator(geom, replace(cfg, **{field: value}))
    assert changed.sha256 != op.sha256


def test_operator_carries_the_geometry_scalars(op, cfg):
    assert op.sdd_m == cfg.sdd_m
    assert op.wavelength_m == pytest.approx(cfg.wavelength_m)


def test_save_load_roundtrip_preserves_hash(op, tmp_path):
    path = save_operator(op, tmp_path / "operator.npz")
    loaded = load_operator(path)
    assert loaded.sha256 == op.sha256
    assert loaded.npt == op.npt
    assert loaded.nnz == op.nnz
    assert loaded.shape == op.shape
    assert loaded.method == op.method
    for name in ("coef", "bins", "indptr", "omega", "q"):
        assert np.array_equal(getattr(loaded, name), getattr(op, name))


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
    with pytest.raises(ValueError, match="rule 1"):
        AgipdSaxsConfig(proposal=10400, run=423, method=("no", "csr", "cython"))


def test_config_hash_covers_every_field(cfg):
    baseline = cfg.config_hash()
    assert baseline == AgipdSaxsConfig(**asdict(cfg)).config_hash()
    for field, value in (("npt", 400), ("sdd_m", 7.6), ("photon_energy_kev", 9.1)):
        assert replace(cfg, **{field: value}).config_hash() != baseline
    # The centre has to move it too: the writer checks this hash to decide
    # whether a run may resume into an existing output file, and re-centred
    # sums must never be appended to PONI = 0 ones.
    centred = replace(cfg, beam_center_px=607.46, beam_center_py=672.08)
    assert centred.config_hash() != baseline
    assert (
        replace(centred, beam_center_px=608.46).config_hash() != centred.config_hash()
    )


def test_geometry_from_config_requires_a_file(cfg):
    from analysis.saxs.operator import geometry_from_config

    with pytest.raises(ValueError, match="geometry_file"):
        geometry_from_config(cfg)


# ── beam centre (open task 4) ─────────────────────────────────────────────────
# `cfg.beam_center` is expressed in the `to_distortion_array()` frame, not the
# `to_pyfai_detector()` one PONI = 0 lives in, so `build_operator` has to
# install that corner array before calling setFit2D. These pin that pairing:
# dropping either half moves the beam by the offset between the two origins
# (~20 px in y on this beamtime's geometry) without raising anything.


@pytest.fixture(scope="module")
def centred_cfg(cfg):
    """``cfg`` with the beamtime's agreed beam centre."""
    from analysis.saxs.config import DEFAULT_BEAM_CENTER_PX, DEFAULT_BEAM_CENTER_PY

    return replace(
        cfg,
        beam_center_px=DEFAULT_BEAM_CENTER_PX,
        beam_center_py=DEFAULT_BEAM_CENTER_PY,
    )


def test_the_shipped_default_is_the_agreed_beam_centre():
    """The values agreed with the beamline scientist, not PONI = 0.

    A real run takes them from the dataclass default, so this is the only
    place the pairing of geometry and centre is asserted for the shipped
    configuration.
    """
    cfg = AgipdSaxsConfig(proposal=10400, run=423)
    assert cfg.beam_center == (607.4598195630211, 672.076693118667)


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


def test_beam_centre_moves_the_q_axis(geom, cfg, centred_cfg):
    """Not a no-op: the centre is what q is measured from."""
    poni_zero, _ = build_operator(geom, cfg)
    centred, _ = build_operator(geom, centred_cfg)
    assert not np.allclose(centred.q, poni_zero.q)
    assert centred.beam_center == centred_cfg.beam_center
    assert poni_zero.beam_center is None


def test_beam_centre_enters_the_operator_hash(geom, centred_cfg):
    """Two operators that differ only in the centre must not share a hash.

    They would otherwise resume into each other's output file, since the
    operator hash is what the writer checks.
    """
    centred, _ = build_operator(geom, centred_cfg)
    moved, _ = build_operator(
        geom, replace(centred_cfg, beam_center_px=centred_cfg.beam_center_px + 1.0)
    )
    assert moved.sha256 != centred.sha256


def test_beam_centre_survives_the_npz_round_trip(geom, centred_cfg, tmp_path):
    centred, _ = build_operator(geom, centred_cfg)
    loaded = load_operator(save_operator(centred, tmp_path / "centred.npz"))
    assert loaded.beam_center == centred.beam_center
    assert loaded.sha256 == centred.sha256


def test_build_operator_reproduces_extra_speckle_configsaxs(geom, centred_cfg):
    """Cross-check against the implementation the centre was derived against.

    ``extra_speckle.setup.configuration.ConfigSAXS`` is what produced the
    numbers in ``config.py``. If our construction and its construction ever
    diverge -- a changed corner-array origin, a Fit2D argument order, a
    millimetre -- the q each pixel is assigned to moves, and this is what
    catches it.
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
