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


def test_geometry_from_config_requires_a_file(cfg):
    from analysis.saxs.operator import geometry_from_config

    with pytest.raises(ValueError, match="geometry_file"):
        geometry_from_config(cfg)
