"""Invariants a refactor must not move, pinned as literals.

``config_hash`` gates resume and gates reopening a stored file, and
``operator_sha256`` gates whether a worker will use one. A change to either
refuses every stored ``.h5``, since stored files are compared, not recomputed.
A red assertion here means a change moved something it claimed only to
rearrange; a deliberate change updates the literal in the same commit.

The configs are built with no input files, so every value below is a function of
the field names and their defaults alone, with nothing read off disk.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyFAI")

from analysis.saxs import operator as saxs_operator  # noqa: E402
from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.waxs import operator as waxs_operator  # noqa: E402
from analysis.waxs.config import JungfrauWaxsConfig  # noqa: E402

SAXS_CONFIG_HASH = "9104a057416d3791846400d1431923ee8f5d27cf46cbb13e9c32252f59dc58be"
WAXS_CONFIG_HASH = "316265309d201fe945d3fc0bbc911f39752e9fa1a9ff44ebb9181107a472417d"


def test_saxs_config_hash_is_unchanged():
    cfg = AgipdSaxsConfig(
        proposal=10400, run=423, geometry_file=None, pixel_mask_file=None
    )
    assert cfg.config_hash() == SAXS_CONFIG_HASH


def test_waxs_config_hash_is_unchanged():
    cfg = JungfrauWaxsConfig(
        proposal=10400, run=423, detector="jf1", poni_file=None, static_mask_file=None
    )
    assert cfg.config_hash() == WAXS_CONFIG_HASH


# ── operator digests ──────────────────────────────────────────────────────────
# The two passes hash different payloads under different key names: saxs carries
# ``sdd_m`` and a beam centre, waxs ``dist_m``, ``unit`` and the static mask. A
# shared "normalisation" of the two would move both digests, which is why each is
# pinned separately.
SAXS_OPERATOR_SHA = "bd34a56a3545b494cf0a229d92ee770e4200278a8fe0fbda49093bf06fffbf80"
SAXS_OPERATOR_SHA_PONI_ZERO = (
    "bc899ee0ea7eed94285f10f3ee6a94c362f7e1e524a6de3954e7458d546d3119"
)
WAXS_OPERATOR_SHA = "c99d4bb166fafc47bc64ec091ac4d8b9ecd5dacf4942e9167f32e45c1f8473ba"


@pytest.fixture
def saxs_arrays():
    rng = np.random.default_rng(20260914)
    return {
        "coef": rng.random(64).astype(np.float32),
        "bins": np.arange(64, dtype=np.int32),
        "indptr": np.arange(0, 65, dtype=np.int32),
        "omega": rng.random(32).astype(np.float64),
        "q": np.linspace(0.1, 1.0, 8, dtype=np.float64),
    }


def test_saxs_operator_sha256_is_unchanged(saxs_arrays):
    got = saxs_operator.operator_sha256(
        saxs_arrays["coef"],
        saxs_arrays["bins"],
        saxs_arrays["indptr"],
        saxs_arrays["omega"],
        saxs_arrays["q"],
        npt=8,
        shape=(16, 2),
        method=("full", "csc", "cython"),
        sdd_m=7.531,
        wavelength_m=1.3715e-10,
        beam_center=(607.46, 672.08),
    )
    assert got == SAXS_OPERATOR_SHA


def test_saxs_operator_sha256_without_a_beam_centre_is_unchanged(saxs_arrays):
    """A PONI-zero operator hashes differently, and that difference is the point."""
    got = saxs_operator.operator_sha256(
        saxs_arrays["coef"],
        saxs_arrays["bins"],
        saxs_arrays["indptr"],
        saxs_arrays["omega"],
        saxs_arrays["q"],
        npt=8,
        shape=(16, 2),
        method=("full", "csc", "cython"),
        sdd_m=7.531,
        wavelength_m=1.3715e-10,
        beam_center=None,
    )
    assert got == SAXS_OPERATOR_SHA_PONI_ZERO
    assert got != SAXS_OPERATOR_SHA


def test_waxs_operator_sha256_is_unchanged():
    q = np.linspace(9.8, 23.7, 16, dtype=np.float64)
    omega = np.random.default_rng(20260914).random(16).astype(np.float64)
    bad = np.zeros(16, dtype=bool)
    bad[::3] = True
    got = waxs_operator.operator_sha256(
        q,
        omega,
        bad,
        npt=16,
        shape=(4, 4),
        method=("full", "csc", "cython"),
        unit="q_nm^-1",
        dist_m=0.232,
        wavelength_m=1.3715e-10,
    )
    assert got == WAXS_OPERATOR_SHA


# ── the output digest is sensitive ────────────────────────────────────────────
def test_the_output_digest_notices_one_changed_value(tmp_path, h5_digest):
    """A pin is worth nothing unless a single moved number breaks it."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "probe.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("frames")
        group.create_dataset("signal", data=np.arange(64, dtype=np.float64))
        group.attrs["npt"] = 500
        handle.create_group("provenance").attrs["wall_s"] = 1.0

    before = h5_digest(path)
    with h5py.File(path, "r+") as handle:
        handle["frames/signal"][7] += 1e-9
    assert h5_digest(path) != before


def test_the_output_digest_ignores_volatile_attributes(tmp_path, h5_digest):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "probe.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("frames/signal", data=np.arange(8, dtype=np.float64))
        handle.create_group("provenance").attrs["wall_s"] = 1.0

    before = h5_digest(path)
    with h5py.File(path, "r+") as handle:
        handle["provenance"].attrs["wall_s"] = 999.0
    assert h5_digest(path) == before
