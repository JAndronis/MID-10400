"""The DAMNIT surface: what the variables return and what the overview draws."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from analysis.waxs import damnit  # noqa: E402
from analysis.waxs.config import DETECTORS, default_poni_file  # noqa: E402


@pytest.fixture
def grid(cfg, mock_run_factory, tmp_path):
    """A finished pass over the mock run, as the variable would return it."""
    from analysis.waxs.run import run_jungfrau_waxs

    mock, dc = mock_run_factory()
    return run_jungfrau_waxs(
        dataclasses.replace(cfg, n_workers=1),
        dc=dc,
        run_dir=mock.path,
        output_path=tmp_path / "damnit.h5",
        reduce="per_cell",
    )


def test_config_for_is_reachable_and_per_detector():
    for detector in DETECTORS:
        cfg = damnit.config_for(10400, 423, detector)
        assert cfg.detector == detector
        assert cfg.poni_file == default_poni_file(detector)


def test_config_for_takes_overrides():
    cfg = damnit.config_for(10400, 423, "jf1", npt=128, allow_incomplete=True)
    assert cfg.npt == 128 and cfg.allow_incomplete


def test_the_grid_is_what_damnit_stores(grid, cfg):
    assert grid.dims == ("trainId", "cellId", "q")
    assert grid.name == "intensity"
    assert grid.dtype == np.float32
    assert "n_frames" in grid.coords
    # Small enough that the per-frame grid is the obvious thing to store:
    # 3000 trains x 8 cells x npt 500 is 48 MB, against 0.93 GB for AGIPD.
    assert grid.nbytes < 10_000_000


def test_the_grid_survives_damnits_netcdf_round_trip(grid, tmp_path):
    """DAMNIT stores a DataArray through netCDF; the coords have to survive."""
    xr = pytest.importorskip("xarray")

    path = tmp_path / "roundtrip.nc"
    grid.to_netcdf(path)
    back = xr.open_dataarray(path)
    assert back.dims == grid.dims
    assert np.array_equal(back["n_frames"].values, grid["n_frames"].values)
    assert back.values == pytest.approx(grid.values)


def test_jungfrau_waxs_does_not_swallow_an_incomplete_run(monkeypatch):
    """Failing loudly needs no code here — only that nothing catches it."""
    from analysis.waxs.writer import IncompleteRun

    def boom(cfg, **kwargs):
        assert kwargs["reduce"] == "per_cell"
        raise IncompleteRun("6 frames did not reach OK")

    monkeypatch.setattr("analysis.waxs.run.run_jungfrau_waxs", boom)
    with pytest.raises(IncompleteRun, match="did not reach OK"):
        damnit.jungfrau_waxs(10400, 423, "jf1")


# ── the overview ─────────────────────────────────────────────────────────────
def test_the_overview_draws_both_detectors_as_separate_traces(grid):
    """§6 O5: two traces, never a concatenation."""
    figure = damnit.overview_figure(grid, grid)
    curve_axes = figure.axes[0]
    assert len(curve_axes.lines) == 2
    labels = [line.get_label() for line in curve_axes.lines]
    # jf1 is the reference and keeps its label; jf2 carries the factor applied.
    assert labels[0] == "jf1"
    assert labels[1].startswith("jf2 x")
    assert curve_axes.get_yscale() == "log"


def test_the_overview_marks_the_overlap(grid):
    figure = damnit.overview_figure(grid, grid)
    texts = [t.get_text() for t in figure.axes[0].texts]
    assert any("overlap" in t for t in texts)


def test_the_overview_shows_the_fitted_scale(grid):
    """§6 O5: jf2 is drawn on jf1's scale, and by how much is on the figure."""
    figure = damnit.overview_figure(grid, grid)
    note = " ".join(t.get_text() for t in figure.axes[0].texts)
    assert "scale" in note and "overlap" in note
    # The same grid twice, so the factor is 1 and the fit is perfect.
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert labels[0] == "jf1"
    assert labels[1].startswith("jf2 x")


def test_the_overview_does_not_move_the_reference(grid):
    figure = damnit.overview_figure(grid, grid)
    reference, other = figure.axes[0].lines
    import numpy as np

    assert np.allclose(reference.get_ydata(), other.get_ydata())


def test_the_overview_draws_when_only_one_detector_ran(grid):
    figure = damnit.overview_figure(grid, None)
    assert len(figure.axes[0].lines) == 1
    not_processed = [t.get_text() for ax in figure.axes[1:] for t in ax.texts]
    assert any("not processed" in t for t in not_processed)


def test_the_overview_survives_a_run_with_nothing_integrated(grid):
    empty = grid.copy()
    empty.values[:] = 0.0
    empty["n_frames"].values[:] = 0
    figure = damnit.overview_figure(empty, empty)
    messages = [t.get_text() for ax in figure.axes[1:] for t in ax.texts]
    assert any("no integrated frames" in t for t in messages)


def test_the_overview_labels_the_axes_in_kev(grid):
    figure = damnit.overview_figure(grid, grid)
    assert "keV" in figure.axes[0].get_ylabel()
    assert "q [nm" in figure.axes[0].get_xlabel()


# ── the file readers ─────────────────────────────────────────────────────────
def test_the_file_readers_return_the_same_thing_as_the_pass(
    cfg, mock_run_factory, tmp_path
):
    from analysis.waxs.run import run_jungfrau_waxs

    mock, dc = mock_run_factory()
    path = tmp_path / "readers.h5"
    returned = run_jungfrau_waxs(
        dataclasses.replace(cfg, n_workers=1),
        dc=dc,
        run_dir=mock.path,
        output_path=path,
        reduce="per_cell",
    )
    from_file = damnit.per_cell_from_file(path)
    assert from_file.values == pytest.approx(returned.values)

    pooled = damnit.pooled_from_file(path)
    assert set(pooled.data_vars) == {"intensity", "sigma", "n_frames"}
