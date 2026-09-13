"""Fixtures for the JUNGFRAU WAXS pass.

Everything here runs without Maxwell and without the real PONI or ``.edf``: the
geometry is a synthetic PONI written to a temporary directory, the way the SAXS
fixtures use a synthetic ``AGIPD_1MGeometry``. The real-data gate that needs
``data/`` is in ``test_real_frames.py`` and skips when the files are absent.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyFAI")
pytest.importorskip("fabio")
pytest.importorskip("extra_data")

from waxs_mockrun import (  # noqa: E402
    MODULE_SHAPE,
    PHOTON_KEV,
    write_mock_run,
    write_static_mask,
    write_synthetic_poni,
)

from analysis.waxs.config import JungfrauWaxsConfig  # noqa: E402

#: Small enough to keep the mock run's arrays cheap, large enough that bins are
#: populated across the whole radial range.
NPT = 64


@pytest.fixture(scope="session")
def geometry_files(tmp_path_factory):
    """A synthetic PONI and static ``.edf`` mask **per detector**.

    Per detector rather than shared, because the config refuses a ``jf1.poni``
    handed to a jf2 run — integrating one detector's frames through the other's
    geometry is the mix-up that produces a plausible I(q) and no error.
    """
    directory = tmp_path_factory.mktemp("waxs-geometry")
    files = {}
    for index, detector in enumerate(("jf1", "jf2")):
        poni = write_synthetic_poni(
            directory / f"{detector}.poni", orientation=4 - index
        )
        rng = np.random.default_rng(11 + index)
        mask = np.zeros(MODULE_SHAPE, dtype=np.uint8).reshape(-1)
        mask[rng.choice(mask.size, int(0.4 * mask.size), replace=False)] = 1
        edf = write_static_mask(
            directory / f"{detector}.edf", mask.reshape(MODULE_SHAPE)
        )
        files[detector] = SimpleNamespace(poni=poni, mask=edf)
    return SimpleNamespace(poni=files["jf1"].poni, mask=files["jf1"].mask, **files)


@pytest.fixture
def config_for_detector(geometry_files, tmp_path):
    """Build a config for either detector, pointed at its own synthetic files."""
    import dataclasses

    from analysis.waxs.config import config_for

    def build(detector, **overrides):
        files = getattr(geometry_files, detector)
        defaults = {
            "poni_file": str(files.poni),
            "static_mask_file": str(files.mask),
            "npt": NPT,
            "trains_per_block": 2,
            "n_workers": 2,
            "cell_sample_trains": 2,
            "selftest_frames": 2,
            "output_root": str(tmp_path / "out"),
        }
        return dataclasses.replace(
            config_for(10400, 423, detector), **(defaults | overrides)
        )

    return build


@pytest.fixture
def cfg(geometry_files, tmp_path):
    """A config pointing at the synthetic geometry and a temporary output root."""
    return JungfrauWaxsConfig(
        proposal=10400,
        run=423,
        detector="jf1",
        poni_file=str(geometry_files.poni),
        static_mask_file=str(geometry_files.mask),
        npt=NPT,
        trains_per_block=2,
        n_workers=2,
        cell_sample_trains=2,
        selftest_frames=2,
        output_root=str(tmp_path / "out"),
    )


@pytest.fixture(scope="session")
def mock_run_factory(tmp_path_factory):
    """Write a mock run, memoised by its keyword signature.

    Writing 16 cells of 512x1024 through gzip is the slowest thing in this
    suite, so runs with identical arguments are written once per session.
    """
    cache: dict[tuple, tuple] = {}

    def factory(**kwargs):
        from extra_data import RunDirectory

        key = tuple(sorted(kwargs.items()))
        if key not in cache:
            root = tmp_path_factory.mktemp("waxs-mockrun")
            run = write_mock_run(root, **kwargs)
            cache[key] = (run, RunDirectory(str(root)))
        return cache[key]

    return factory


@pytest.fixture
def operator(cfg):
    from analysis.waxs.operator import build_operator

    return build_operator(cfg)


@pytest.fixture
def model():
    from analysis.waxs.integrate import ErrorModel

    return ErrorModel(0.32, PHOTON_KEV, "configured")


@pytest.fixture
def pipeline(cfg, mock_run_factory, operator):
    """A mock run with its plan, operator and cell classification built."""
    from analysis.waxs.cells import CellAccumulator
    from analysis.waxs.integrate import ErrorModel
    from analysis.waxs.plan import build_plan, open_detector

    run, dc = mock_run_factory()
    op, ai = operator
    det = open_detector(cfg, dc)

    accumulator = CellAccumulator(cfg, op.static_bad)
    from extra_data import by_id

    for train_id in run.detector_trains[:2]:
        selected = det.select_trains(by_id[[train_id]])
        accumulator.update(
            np.asarray(selected["data.adc"].ndarray())[0, 0],
            np.asarray(selected["data.mask"].ndarray())[0, 0],
            np.asarray(selected["data.memoryCell"].ndarray()).reshape(-1),
        )
    classification = accumulator.finalise()
    model = ErrorModel(classification.read_noise_kev, cfg.photon_energy_kev, "measured")
    plan = build_plan(cfg, classification.lit, dc=dc, control_dc=dc, det=det)
    return SimpleNamespace(
        cfg=cfg,
        run=run,
        dc=dc,
        det=det,
        op=op,
        ai=ai,
        model=model,
        classification=classification,
        plan=plan,
    )


@pytest.fixture
def worker_ready(pipeline):
    """The in-process worker, torn down so no state leaks between tests."""
    from analysis.waxs import worker

    worker.init_from_detector(
        pipeline.cfg,
        pipeline.op,
        pipeline.ai,
        pipeline.model,
        pipeline.classification.lit,
        pipeline.det,
    )
    yield pipeline
    worker._STATE = None
