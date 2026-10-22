"""The JUNGFRAU pass's output is a function of its input, pinned by digest.

A refactor of this pass must not move a single stored number; if the digest
moves, the change altered an output. Pinned per platform: bit-exact floats are a
property of the wheels a platform installs, so a digest is only compared where
it was captured.
"""

from __future__ import annotations

import platform

import pytest

pytest.importorskip("extra_data")

from test_waxs_run import InlinePool, run  # noqa: E402  (tests/waxs is on sys.path)

#: Digest of a whole mock run's output, volatile provenance excluded, keyed by
#: (platform.system(), platform.machine()). Linux x86_64 is the locked
#: environment on Maxwell, where it is stable across SIMD level, OpenMP threads
#: and BLAS kernel.
#:
#: A pyFAI upgrade can move it without changing real results: the fixture's jf1
#: PONI has detector orientation 4, whose full-split coefficients pyFAI may round
#: differently between versions, while the real detectors have orientation 3.
EXPECTED_DIGEST = {
    ("Linux", "x86_64"): (
        "648d593f156f52b2c2c4ad74f9d97bb481f2a14417167a24fb7bc08554bdd9a3"
    ),
}


def test_output_digest_is_pinned(cfg, mock_run_factory, tmp_path, h5_digest):
    key = (platform.system(), platform.machine())
    if key not in EXPECTED_DIGEST:
        pytest.skip(f"no output digest pinned for {key}")
    mock, dc = mock_run_factory()
    run(cfg, mock, dc, output_path=tmp_path / "out.h5", pool_factory=InlinePool)
    assert h5_digest(tmp_path / "out.h5") == EXPECTED_DIGEST[key]


def test_output_digest_is_stable_across_runs(
    cfg, mock_run_factory, tmp_path, h5_digest
):
    """Two runs of the same pass on the same input agree, so the digest is usable."""
    mock, dc = mock_run_factory()
    run(cfg, mock, dc, output_path=tmp_path / "a.h5", pool_factory=InlinePool)
    run(cfg, mock, dc, output_path=tmp_path / "b.h5", pool_factory=InlinePool)
    assert h5_digest(tmp_path / "a.h5") == h5_digest(tmp_path / "b.h5")
