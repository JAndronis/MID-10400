"""Fixtures for the hit-train XPCS tests.

The geometry is EXtra-geom's idealised AGIPD-1M, as in ``tests/saxs``, so
nothing here needs Maxwell, a CrystFEL file or data. The proc-like mock run is
``tests/saxs/mockrun.py``'s, reached by path because test directories here are
not packages.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("extra_geom", reason="the XPCS pass needs EXtra-geom")
pytest.importorskip("pyFAI", reason="the XPCS pass needs pyFAI")

sys.path.insert(0, str(Path(__file__).parents[1] / "saxs"))

from extra_geom import AGIPD_1MGeometry  # noqa: E402
from xpcs_fakes import EMPTY, FRAMES, SHORT, TRAIN_IDS  # noqa: E402

from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.xpcs.config import XpcsConfig  # noqa: E402

QUAD_POS = [(-525, 625), (-550, -10), (520, -160), (542.5, 475)]

#: Two rings, 77 767 and 182 594 pixels of the idealised geometry.
TEST_EDGES_NM = (0.2, 0.3, 0.45)


@pytest.fixture(scope="session")
def geom() -> AGIPD_1MGeometry:
    return AGIPD_1MGeometry.from_quad_positions(quad_pos=QUAD_POS)


@pytest.fixture(scope="session")
def saxs_cfg() -> AgipdSaxsConfig:
    """No input files, so ``config_hash`` has nothing to digest on disk."""
    return AgipdSaxsConfig(
        proposal=10400,
        run=423,
        geometry_file=None,
        pixel_mask_file=None,
        beam_center_px=None,
        beam_center_py=None,
        photon_stats_root="/nonexistent/agipd_photon_stats",
    )


@pytest.fixture(scope="session")
def cfg(saxs_cfg) -> XpcsConfig:
    return XpcsConfig(
        saxs=saxs_cfg,
        ring_edges_nm=TEST_EDGES_NM,
        envelope_margin_mm=5.0,
        n_workers=2,
        trains_per_block=2,
        output_root="/nonexistent/xpcs",
    )


@pytest.fixture
def with_config(cfg):
    """``cfg`` with some fields replaced."""

    def make(**changes) -> XpcsConfig:
        return replace(cfg, **changes)

    return make


@pytest.fixture(scope="session")
def mock_run_factory(memoised_run_factory):
    """Mock AGIPD runs, cached by resolved signature."""
    from mockrun import write_mock_run

    return memoised_run_factory(write_mock_run, "xpcs-mockrun")


@pytest.fixture(scope="session")
def mock_run(mock_run_factory):
    """``(MockRun, DataCollection)``: 12 frames a train, two awkward trains."""
    return mock_run_factory(
        train_ids=TRAIN_IDS,
        frames_per_train=FRAMES,
        short_module_trains=(SHORT,),
        zero_frame_trains=(EMPTY,),
    )


@pytest.fixture
def pipeline(cfg, geom, mock_run, tmp_path):
    """Run :func:`run_xpcs_hits` on the mock run into ``tmp_path``."""
    from types import SimpleNamespace

    from xpcs_fakes import InlinePool

    from analysis.xpcs.hits import HitSet
    from analysis.xpcs.run import run_xpcs_hits

    run, dc = mock_run

    def go(hits, *, pool=InlinePool, **changes):
        config = replace(cfg, output_root=str(tmp_path), **changes)
        return run_xpcs_hits(
            config,
            hits if isinstance(hits, HitSet) else HitSet.of(hits),
            dc=dc,
            geometry=geom,
            run_dir=run.path,
            pool_factory=pool,
        )

    def configured(**changes):
        return replace(cfg, output_root=str(tmp_path), **changes)

    return SimpleNamespace(run=go, cfg=configured, mock=run, dc=dc, geom=geom)
