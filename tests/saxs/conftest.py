"""Fixtures for the P1 integrator tests.

The geometry is EXtra-geom's idealised AGIPD-1M built from the quad positions
used in its own tests, so these run anywhere: no CrystFEL file, no Maxwell, no
data. Building the full-split CSC operator over 1 048 576 pixels costs ~0.3 s
and ~20 MB, so it is session-scoped and built once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("extra_geom", reason="P1 needs EXtra-geom")
pytest.importorskip("pyFAI", reason="P1 needs pyFAI")

from extra_geom import AGIPD_1MGeometry  # noqa: E402

from analysis.common.masks import MaskSource, StaticMask  # noqa: E402
from analysis.saxs.config import NPIX, AgipdSaxsConfig  # noqa: E402
from analysis.saxs.operator import build_operator  # noqa: E402

#: EXtra-geom's test quad positions, in pixel units (the constructor's default
#: ``unit`` is ``pixel_size``).
QUAD_POS = [(-525, 625), (-550, -10), (520, -160), (542.5, 475)]


@pytest.fixture(scope="session")
def quad_pos() -> list[tuple[float, float]]:
    return list(QUAD_POS)


@pytest.fixture(scope="session")
def cfg() -> AgipdSaxsConfig:
    """Config for the synthetic geometry.

    The real defaults are GPFS paths, so every input file is set to ``None``
    here: the tests build their geometry and masks in-process, and
    ``config_hash`` must not try to sha256 a file that is not mounted.

    The beam centre goes the same way. It was refined against the beamtime's
    own geometry, and these quad positions are EXtra-geom's test ones, whose
    central hole sits elsewhere — applying it here would put the beam
    0.0008 nm⁻¹ from a pixel rather than in the hole. The beam-centre path has
    its own tests in ``test_operator.py``.
    """
    return AgipdSaxsConfig(
        proposal=10400,
        run=423,
        npt=500,
        geometry_file=None,
        pixel_mask_file=None,
        beam_center_px=None,
        beam_center_py=None,
    )


@pytest.fixture(scope="session")
def geom() -> AGIPD_1MGeometry:
    return AGIPD_1MGeometry.from_quad_positions(quad_pos=QUAD_POS)


@pytest.fixture(scope="session")
def operator_and_engine(geom, cfg):
    """``(SparseOperator, pyFAI engine)`` from the synthetic geometry."""
    op, ai = build_operator(geom, cfg)
    probe_method = next(iter(ai.engines))
    return op, ai.engines[probe_method].engine


@pytest.fixture(scope="session")
def op(operator_and_engine):
    return operator_and_engine[0]


@pytest.fixture(scope="session")
def engine(operator_and_engine):
    return operator_and_engine[1]


def _make_frame(rng: np.random.Generator, occupancy: float) -> np.ndarray:
    """A synthetic int16 photon frame with the given nonzero fraction.

    Counts are 1-3, matching CLAUDE.md's r0423/r0426 distribution (98.9 % zero,
    1.1-1.3 % ones, 2.5e-4 twos, max < 20).
    """
    x = np.zeros(NPIX, dtype=np.int16)
    hits = rng.choice(NPIX, int(occupancy * NPIX), replace=False)
    x[hits] = rng.integers(1, 4, hits.size).astype(np.int16)
    return x


def _make_masks(
    rng: np.random.Generator,
    *,
    static_fraction: float = 0.04,
    disagree: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bad, base_bad)``.

    ``base_bad`` is the cell's base mask. ``bad`` is this frame's mask, which
    differs from it at ``disagree`` pixels flipped in *both* directions: half
    flagged by the frame but not the cell, half by the cell but not the frame.
    """
    base_bad = rng.random(NPIX) < static_fraction
    bad = base_bad.copy()
    if disagree:
        set_in_cell = np.flatnonzero(base_bad)
        clear_in_cell = np.flatnonzero(~base_bad)
        half = disagree // 2
        # cell says bad, frame says good
        bad[rng.choice(set_in_cell, half, replace=False)] = False
        # cell says good, frame says bad
        bad[rng.choice(clear_in_cell, disagree - half, replace=False)] = True
    return bad, base_bad


@pytest.fixture(scope="session")
def make_frame() -> Callable[[np.random.Generator, float], np.ndarray]:
    """Factory for synthetic int16 photon frames."""
    return _make_frame


@pytest.fixture(scope="session")
def make_masks() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Factory for ``(bad, base_bad)`` mask pairs."""
    return _make_masks


def _make_mask_train(
    rng: np.random.Generator,
    cell_ids: np.ndarray,
    *,
    static_bits: int = 1 << 0,
    static_fraction: float = 0.02,
    dynamic_bits: int = 1 << 12,
    dynamic_per_frame: int = 40,
    extra_bits: int = 0,
) -> np.ndarray:
    """Build one train of synthetic ``image.mask``, ``(16, n, 512, 128)`` uint32.

    Every memory cell gets its own fixed set of statically flagged pixels, so
    the majority vote has real per-cell structure; ``dynamic_per_frame`` extra
    pixels are flagged in one frame only, standing in for the per-frame bits 12
    and 13 of the real data.
    """
    n_frames = cell_ids.size
    mask = np.zeros((16, n_frames, 512, 128), dtype=np.uint32)
    flat = mask.reshape(16, n_frames, -1)
    n_module_px = 512 * 128
    for frame, cell in enumerate(cell_ids.tolist()):
        cell_rng = np.random.default_rng(1000 + cell)
        for module in range(16):
            static = cell_rng.choice(
                n_module_px, int(static_fraction * n_module_px), replace=False
            )
            flat[module, frame, static] |= np.uint32(static_bits)
        dynamic = rng.choice(n_module_px, dynamic_per_frame, replace=False)
        flat[0, frame, dynamic] |= np.uint32(dynamic_bits)
        if extra_bits:
            flat[0, frame, rng.integers(0, n_module_px)] |= np.uint32(extra_bits)
    return mask


@pytest.fixture
def make_mask_train() -> Callable[..., np.ndarray]:
    """Factory for synthetic ``image.mask`` trains."""
    return _make_mask_train


@pytest.fixture(scope="session")
def static_mask() -> StaticMask:
    """A small random static mask, built without touching the filesystem."""
    rng = np.random.default_rng(4242)
    bad = rng.random(NPIX) < 0.01
    bad.flags.writeable = False
    return StaticMask(
        bad=bad,
        sha256=hashlib.sha256(np.packbits(bad).tobytes()).hexdigest(),
        sources=(MaskSource("synthetic", None, None, int(bad.sum())),),
    )


@pytest.fixture(scope="session")
def run_cfg(cfg) -> AgipdSaxsConfig:
    """Config sized for the mock run: small blocks, two workers."""
    from dataclasses import replace

    return replace(cfg, trains_per_block=2, n_workers=2, selftest_frames=2)


@pytest.fixture(scope="session")
def mock_run_factory(tmp_path_factory):
    """Build (and cache) mock runs by keyword signature.

    Writing sixteen gzip+shuffle module files is the slowest thing in this
    suite, so runs are memoised: most tests ask for the same default run.
    """
    from extra_data import RunDirectory
    from mockrun import write_mock_run

    cache: dict[tuple, tuple] = {}

    def factory(**kwargs):
        key = tuple(
            sorted(
                (k, tuple(v) if isinstance(v, (list, tuple)) else v)
                for k, v in kwargs.items()
            )
        )
        if key not in cache:
            root = tmp_path_factory.mktemp("mockrun")
            run = write_mock_run(root, **kwargs)
            cache[key] = (run, RunDirectory(str(root)))
        return cache[key]

    return factory


@pytest.fixture(scope="session")
def mock_pipeline(run_cfg, geom, mock_run_factory):
    """Everything the worker and writer need for the default mock run."""
    from types import SimpleNamespace

    from extra_data import by_id
    from extra_data.components import AGIPD1M

    from analysis.common.status import FrameStatus
    from analysis.saxs.masks import BaseMaskAccumulator, build_static_bad
    from analysis.saxs.operator import build_operator
    from analysis.saxs.plan import build_plan

    run, dc = mock_run_factory()
    plan = build_plan(run_cfg, dc=dc)
    op, ai = build_operator(geom, run_cfg)
    static = build_static_bad(run_cfg)

    det = AGIPD1M(dc, min_modules=run_cfg.min_modules)
    accumulator = BaseMaskAccumulator(
        op, static, mask_bits=run_cfg.mask_bits, expected_bits=run_cfg.expected_bits
    )
    for train in plan.trains:
        if train.status is not FrameStatus.OK:
            continue
        key = det.select_trains(by_id[[train.train_id]])["image.mask"]
        accumulator.update(key.ndarray(decompress_threads=1), key.cell_id_coordinates())
    masks = accumulator.finalise()

    return SimpleNamespace(
        cfg=run_cfg,
        run=run,
        dc=dc,
        plan=plan,
        op=op,
        ai=ai,
        static=static,
        masks=masks,
        detector=det,
    )


@pytest.fixture
def worker_ready(mock_pipeline):
    """Initialise the worker in-process and tear it down afterwards."""
    from analysis.saxs import worker

    worker.init_from_detector(
        mock_pipeline.cfg,
        mock_pipeline.op,
        mock_pipeline.masks,
        mock_pipeline.detector,
    )
    yield mock_pipeline
    worker._STATE = None
