"""The JUNGFRAU pass's output is a function of its input, pinned by digest.

A refactor of this pass must not move a single stored number. The digest below
was captured before that refactor began; if it moves, the change altered an
output and is not the pure refactor it claims to be.
"""

from __future__ import annotations

import pytest

pytest.importorskip("extra_data")

from test_waxs_run import InlinePool, run  # noqa: E402  (tests/waxs is on sys.path)

#: Digest of a whole mock run's output, volatile provenance excluded.
EXPECTED_DIGEST = "ce911c5bf8ce5d641ea3845061c01b014e06c8578a012cfd507f917326cbc7b6"


def test_output_digest_is_pinned(cfg, mock_run_factory, tmp_path, h5_digest):
    mock, dc = mock_run_factory()
    run(cfg, mock, dc, output_path=tmp_path / "out.h5", pool_factory=InlinePool)
    assert h5_digest(tmp_path / "out.h5") == EXPECTED_DIGEST


def test_output_digest_is_stable_across_runs(
    cfg, mock_run_factory, tmp_path, h5_digest
):
    """Two runs of the same pass on the same input agree — the digest is usable."""
    mock, dc = mock_run_factory()
    run(cfg, mock, dc, output_path=tmp_path / "a.h5", pool_factory=InlinePool)
    run(cfg, mock, dc, output_path=tmp_path / "b.h5", pool_factory=InlinePool)
    assert h5_digest(tmp_path / "a.h5") == h5_digest(tmp_path / "b.h5")
