"""The AGIPD pass's output is a function of its input, pinned by digest.

A refactor of this pass must not move a single stored number; if the digest
moves, the change altered an output. Pinned per platform: bit-exact floats are a
property of the wheels a platform installs, so a digest is only compared where
it was captured.
"""

from __future__ import annotations

import platform

import pytest

pytest.importorskip("extra_data")

from test_run import InlinePool, run_on_mock  # noqa: E402  (tests/saxs is on sys.path)

#: Digest of a whole mock run's output, volatile provenance excluded, keyed by
#: (platform.system(), platform.machine()). On Linux x86_64 it is stable across
#: SIMD level, OpenMP threads and BLAS kernel.
#:
#: A new config field is recorded in provenance, so it moves the digest without
#: moving any dataset. Before re-pinning, check that the datasets digest the
#: same with the new attributes left out.
EXPECTED_DIGEST = {
    ("Linux", "x86_64"): (
        "f82b98f3bf8a9b4f20e0d0f417097268488b775b15515b9561e5460daa77957d"
    ),
}


@pytest.fixture
def pipeline(mock_pipeline, geom):
    mock_pipeline.geometry = geom
    return mock_pipeline


def test_output_digest_is_pinned(pipeline, tmp_path, h5_digest):
    key = (platform.system(), platform.machine())
    if key not in EXPECTED_DIGEST:
        pytest.skip(f"no output digest pinned for {key}")
    run_on_mock(pipeline, tmp_path, pool_factory=InlinePool)
    assert h5_digest(tmp_path / "out.h5") == EXPECTED_DIGEST[key]


def test_output_digest_is_stable_across_runs(pipeline, tmp_path, h5_digest):
    """Two runs of the same pass on the same input agree, so the digest is usable."""
    run_on_mock(pipeline, tmp_path, pool_factory=InlinePool)
    first = h5_digest(tmp_path / "out.h5")
    (tmp_path / "out.h5").unlink()
    run_on_mock(pipeline, tmp_path, pool_factory=InlinePool)
    assert h5_digest(tmp_path / "out.h5") == first
