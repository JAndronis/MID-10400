"""Fixtures for the JUNGFRAU WAXS pass.

Everything here runs without Maxwell and without the real PONI or ``.edf``.
The geometry is a synthetic PONI written to a temporary directory, as the SAXS
fixtures use a synthetic ``AGIPD_1MGeometry``. The real-data gate that reads
proc is in ``test_waxs_real_frames.py`` and skips off Maxwell.

A whole pass over a mock run costs seconds, so the finished files are built
once per session (:func:`finished`) and shared. A test that changes one copies
it to its own ``tmp_path`` first.
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
    """A synthetic PONI and static ``.edf`` mask per detector.

    One set per detector because the config refuses a ``jf1.poni`` handed to a
    jf2 run. Integrating one detector's frames through the other's geometry
    gives a plausible I(q) and no error.
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


def _config(geometry_files, output_root, detector="jf1", **overrides):
    """A small config for one detector, pointed at its own synthetic files."""
    files = getattr(geometry_files, detector)
    settings = {
        "poni_file": str(files.poni),
        "static_mask_file": str(files.mask),
        "npt": NPT,
        "trains_per_block": 2,
        "n_workers": 2,
        "cell_sample_trains": 2,
        "selftest_frames": 2,
        "output_root": str(output_root),
    }
    return JungfrauWaxsConfig(
        proposal=10400, run=423, detector=detector, **(settings | overrides)
    )


@pytest.fixture
def config_for_detector(geometry_files, tmp_path):
    """Build a config for either detector, pointed at its own synthetic files."""

    def build(detector, **overrides):
        return _config(geometry_files, tmp_path / "out", detector, **overrides)

    return build


@pytest.fixture
def cfg(geometry_files, tmp_path):
    """A config pointing at the synthetic geometry and a temporary output root."""
    return _config(geometry_files, tmp_path / "out")


@pytest.fixture(scope="session")
def finished(geometry_files, mock_run_factory, tmp_path_factory):
    """Finished pass outputs, one per (detector, mock signature), shared read-only.

    ``finished("jf1", **mock_kwargs)`` returns the config, the mock, its
    ``DataCollection``, the file and the per-cell grid the pass returned. The
    config's ``output_root`` is the session's, which ``config_hash`` leaves out,
    so the file is complete for any test's ``cfg``. Copy the file before
    changing it.
    """
    from waxs_harness import run

    from analysis.waxs.config import DETECTOR_MODNOS, DETECTOR_NAMES

    cache = {}

    def build(detector="jf1", **mock_kwargs):
        key = (detector, tuple(sorted(mock_kwargs.items())))
        if key not in cache:
            root = tmp_path_factory.mktemp(f"waxs-finished-{detector}")
            cfg = _config(geometry_files, root, detector)
            if detector != "jf1":
                mock_kwargs = {
                    "detector_name": DETECTOR_NAMES[detector],
                    "module": DETECTOR_MODNOS[detector],
                } | mock_kwargs
            mock, dc = mock_run_factory(**mock_kwargs)
            grid = run(cfg, mock, dc, output_path=cfg.output_file)
            cache[key] = SimpleNamespace(
                cfg=cfg, mock=mock, dc=dc, path=cfg.output_file, grid=grid
            )
        return cache[key]

    return build


@pytest.fixture(scope="session")
def mock_run_factory(memoised_run_factory):
    """Mock JUNGFRAU runs, cached by resolved signature.

    A filled default run is about 135 MB of dense float32 that lives until the
    session ends, so the cache keeps the suite's disk use bounded.
    """
    return memoised_run_factory(write_mock_run, "waxs-mockrun")


@pytest.fixture
def operator(cfg):
    from analysis.waxs.operator import build_operator

    return build_operator(cfg)


@pytest.fixture
def model():
    """The readout-noise record a file stores, a diagnostic and not the variance."""
    from analysis.waxs.integrate import ErrorModel

    return ErrorModel(0.32, PHOTON_KEV, "configured")


@pytest.fixture
def counting(cfg):
    """The photon-counting settings the default config names."""
    from analysis.waxs.photons import PhotonCounting

    return PhotonCounting.from_config(cfg)


@pytest.fixture(scope="session")
def pipeline(geometry_files, mock_run_factory, tmp_path_factory):
    """The default mock run with its plan, operator and cell classification built.

    Session-scoped, so its config writes nowhere a test would look; tests pass
    an explicit output path.
    """
    from analysis.waxs.cells import CellAccumulator
    from analysis.waxs.integrate import ErrorModel
    from analysis.waxs.operator import build_operator
    from analysis.waxs.plan import build_plan, open_detector

    cfg = _config(geometry_files, tmp_path_factory.mktemp("waxs-pipeline"))
    run, dc = mock_run_factory()
    op, ai = build_operator(cfg)
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
        pipeline.classification.lit,
        pipeline.det,
    )
    yield pipeline
    worker._STATE = None
