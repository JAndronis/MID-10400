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
#: (platform.system(), platform.machine()). Linux x86_64 is the locked
#: environment on Maxwell, where it is stable across SIMD level, OpenMP threads
#: and BLAS kernel; the pre-cleanup tree produces the same value.
EXPECTED_DIGEST = {
    ("Linux", "x86_64"): (
        "5a818eaaffabfdd917b7edc761d7d43b0945bc2d981e3a194b64b5a9efb7b21f"
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
    """Two runs of the same pass on the same input agree — the digest is usable."""
    run_on_mock(pipeline, tmp_path, pool_factory=InlinePool)
    first = h5_digest(tmp_path / "out.h5")
    (tmp_path / "out.h5").unlink()
    run_on_mock(pipeline, tmp_path, pool_factory=InlinePool)
    assert h5_digest(tmp_path / "out.h5") == first
