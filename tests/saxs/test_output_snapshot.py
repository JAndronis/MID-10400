"""The AGIPD pass's output is a function of its input, pinned by digest.

A refactor of this pass must not move a single stored number. The digest below
was captured before that refactor began; if it moves, the change altered an
output and is not the pure refactor it claims to be.
"""

from __future__ import annotations

import pytest

pytest.importorskip("extra_data")

from test_run import InlinePool, run_on_mock  # noqa: E402  (tests/saxs is on sys.path)

#: Digest of a whole mock run's output, volatile provenance excluded.
EXPECTED_DIGEST = "e36d3807eff80453dcd524d25bc2a9e293a8a12217dc76d992ab37dcb28e78ba"


@pytest.fixture
def pipeline(mock_pipeline, geom):
    mock_pipeline.geometry = geom
    return mock_pipeline


def test_output_digest_is_pinned(pipeline, tmp_path, h5_digest):
    run_on_mock(pipeline, tmp_path, pool_factory=InlinePool)
    assert h5_digest(tmp_path / "out.h5") == EXPECTED_DIGEST


def test_output_digest_is_stable_across_runs(pipeline, tmp_path, h5_digest):
    """Two runs of the same pass on the same input agree — the digest is usable."""
    run_on_mock(pipeline, tmp_path, pool_factory=InlinePool)
    first = h5_digest(tmp_path / "out.h5")
    (tmp_path / "out.h5").unlink()
    run_on_mock(pipeline, tmp_path, pool_factory=InlinePool)
    assert h5_digest(tmp_path / "out.h5") == first
