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
# The SNR tests share the triage's synthetic results files (`triage_fakes`).
sys.path.append(str(Path(__file__).parents[1] / "triage"))

from extra_geom import AGIPD_1MGeometry  # noqa: E402
from xpcs_fakes import EMPTY, FRAMES, GOOD, SHORT, TRAIN_IDS, InlinePool  # noqa: E402

from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.xpcs.config import XpcsConfig  # noqa: E402

QUAD_POS = [(-525, 625), (-550, -10), (520, -160), (542.5, 475)]

#: Two rings, 77 767 and 182 594 pixels of the idealised geometry.
TEST_RINGS_NM = ((0.2, 0.3), (0.3, 0.45))


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
        rings_nm=TEST_RINGS_NM,
        # The mock hits sit two ids apart; the production default is 1.
        max_partner_gap=10,
        # The kernel tests compare against rings built without it; the outlier
        # tests switch it on (test_xpcs_outliers.py).
        outlier_thresholds=None,
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
    """``(MockRun, DataCollection)``, 12 frames a train.

    One train lacks a module's frames and one has no frames at all.
    """
    return mock_run_factory(
        train_ids=TRAIN_IDS,
        frames_per_train=FRAMES,
        short_module_trains=(SHORT,),
        zero_frame_trains=(EMPTY,),
    )


def _memoised(function, key):
    """``function`` with its results kept per ``key(*args, **kwargs)``."""
    cache = {}

    def wrapper(*args, **kwargs):
        k = key(*args, **kwargs)
        if k not in cache:
            cache[k] = function(*args, **kwargs)
        return cache[k]

    return wrapper


@pytest.fixture(scope="session", autouse=True)
def _built_once():
    """Build the pyFAI operator and the run plan once per geometry and run.

    Both are pure functions of their inputs, and every pass on the mock run
    asks for the same ones; building them dominates a pass in this suite. A
    spawned worker is a fresh interpreter and builds its own.
    """
    import analysis.saxs.operator as operator
    import analysis.saxs.plan as plan

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            operator,
            "build_operator",
            _memoised(operator.build_operator, lambda geom, saxs: (id(geom), saxs)),
        )
        patch.setattr(
            plan,
            "build_plan",
            _memoised(
                plan.build_plan,
                lambda saxs, dc=None, control_dc=None: (saxs, id(dc), id(control_dc)),
            ),
        )
        yield


def _run(config, hits, mock_run, geom, pool):
    from analysis.xpcs.hits import HitSet
    from analysis.xpcs.run import run_xpcs_hits

    run, dc = mock_run
    return run_xpcs_hits(
        config,
        hits if isinstance(hits, HitSet) else HitSet.of(hits),
        dc=dc,
        geometry=geom,
        run_dir=run.path,
        pool_factory=pool,
    )


@pytest.fixture(scope="session")
def finished_template(cfg, geom, mock_run, tmp_path_factory, _built_once):
    """An output root holding the finished pass over ``GOOD``; copy, never write."""
    root = tmp_path_factory.mktemp("xpcs-finished")
    _run(replace(cfg, output_root=str(root)), GOOD, mock_run, geom, InlinePool)
    return root


@pytest.fixture
def pipeline(cfg, geom, mock_run, tmp_path, finished_template):
    """Run :func:`run_xpcs_hits` on the mock run into ``tmp_path``.

    ``finished()`` copies the finished pass over ``GOOD`` into ``tmp_path``
    instead, as if ``run(GOOD)`` had just returned.
    """
    import shutil
    from types import SimpleNamespace

    run, dc = mock_run

    def configured(**changes):
        return replace(cfg, output_root=str(tmp_path), **changes)

    def go(hits, *, pool=InlinePool, **changes):
        return _run(configured(**changes), hits, mock_run, geom, pool)

    def finished():
        shutil.copytree(finished_template, tmp_path, dirs_exist_ok=True)
        return configured().output_file

    return SimpleNamespace(
        run=go, finished=finished, cfg=configured, mock=run, dc=dc, geom=geom
    )
