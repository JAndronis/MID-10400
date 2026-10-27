"""The DAMNIT surface: what the variables return and what the overview draws."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from analysis.waxs import damnit  # noqa: E402


@pytest.fixture
def grid(finished):
    """A finished pass over the mock run, as the variable would return it."""
    return finished().grid.copy(deep=True)


def test_the_grid_survives_damnits_netcdf_round_trip(grid, tmp_path):
    """DAMNIT stores a DataArray through netCDF; the coords have to survive."""
    xr = pytest.importorskip("xarray")

    path = tmp_path / "roundtrip.nc"
    grid.to_netcdf(path)
    back = xr.open_dataarray(path)
    assert back.dims == grid.dims
    assert np.array_equal(back["n_frames"].values, grid["n_frames"].values)
    assert back.values == pytest.approx(grid.values)


def test_the_damnit_variable_runs_at_the_stored_npt(monkeypatch):
    """The stored files are at npt 200, so the variable must build that config."""
    seen = []
    monkeypatch.setattr(
        "analysis.waxs.run.run_jungfrau_waxs", lambda cfg, **kwargs: seen.append(cfg)
    )
    damnit.jungfrau_waxs(10400, 423, "jf2")
    damnit.jungfrau_waxs(10400, 423, "jf2", npt=500)
    assert [cfg.npt for cfg in seen] == [200, 500]
    assert seen[0] == damnit.damnit_config(10400, 423, "jf2")
    assert damnit.config_for(10400, 423, "jf2").npt == 500


# ── the overview ─────────────────────────────────────────────────────────────
def test_the_overview_draws_two_one_or_no_detectors(grid):
    """Two traces, jf2 scaled onto jf1 and never concatenated; one; or none.

    The same grid twice must fit a factor of 1, so the two traces coincide.
    """
    figure = damnit.overview_figure(grid, grid)
    reference, other = figure.axes[0].lines
    assert np.allclose(reference.get_ydata(), other.get_ydata())
    assert "photons" in figure.axes[0].get_ylabel()

    assert len(damnit.overview_figure(grid, None).axes[0].lines) == 1

    empty = grid.copy()
    empty.values[:] = 0.0
    empty["n_frames"].values[:] = 0
    assert len(damnit.overview_figure(empty, empty).axes[0].lines) == 0


def test_keV_and_photon_grids_are_never_combined(grid):
    """A fitted factor would absorb the unit change as a detector mismatch."""
    from analysis.waxs.combine import UnitsMismatch

    old = grid.copy()
    old.attrs["units"] = "keV"
    with pytest.raises(UnitsMismatch, match="keV"):
        damnit.combined_curve(grid, old)
    with pytest.raises(UnitsMismatch):
        damnit.overview_figure(old, grid)
    assert damnit.combined_curve(grid, grid).attrs["units"] == "photons"


# ── jungfrau_preprocessing and the per-detector readers ─────────────────────
def _colour(cell) -> str:
    return "#" + bytes(cell.background).hex()


def test_preprocess_reports_each_detector(monkeypatch, tmp_path):
    pytest.importorskip("damnit")
    from analysis.diagnostics.damnit import COLORS

    def fake(cfg, **kwargs):
        if cfg.detector == "jf2":
            raise RuntimeError("no PONI")
        return None  # jf1: no lit cell, so no file

    monkeypatch.setattr("analysis.waxs.run.run_jungfrau_waxs", fake)
    cell = damnit.preprocess(10400, 423, output_root=str(tmp_path))
    assert cell.summary_value == "JF1 dark · JF2 failed"
    assert _colour(cell) == COLORS["error"]
    dark = damnit.preprocess(10400, 423, ("jf1",), output_root=str(tmp_path))
    assert dark.summary_value == "JF1 dark"
    assert _colour(dark) == COLORS["good"]


def test_the_reader_returns_the_grid_and_skips_a_missing_file(
    finished, grid, cfg, tmp_path
):
    import shutil

    import xarray as xr

    damnit_ctx = pytest.importorskip("damnit.context")
    root = tmp_path / "root"
    target = damnit.damnit_config(
        cfg.proposal, cfg.run, cfg.detector, output_root=str(root)
    ).output_file
    target.parent.mkdir(parents=True)
    shutil.copy2(finished().path, target)
    got = damnit.jungfrau_waxs_from_file(
        cfg.proposal, cfg.run, cfg.detector, output_root=str(root)
    )
    xr.testing.assert_identical(got, grid)
    other = "jf2" if cfg.detector == "jf1" else "jf1"
    with pytest.raises(damnit_ctx.Skip):
        damnit.jungfrau_waxs_from_file(
            cfg.proposal, cfg.run, other, output_root=str(root)
        )


@pytest.mark.parametrize("allow", [True, False])
def test_a_refused_frame_is_tolerated_everywhere_or_nowhere(
    cfg, finished, tmp_path, monkeypatch, allow
):
    """The pass, the per-detector reader and the preprocessing colour agree.

    One finished file holds one ``DATA_CHECK_FAILED`` frame. Whatever
    ``allow_data_check_failures`` says, all three readers of it must say the
    same, or DAMNIT shows a green cell over a column that raises.
    """
    pytest.importorskip("damnit")
    import shutil

    from waxs_harness import run

    from analysis.common.writer import IncompleteRun
    from analysis.diagnostics.damnit import COLORS

    done = finished(train_ids=(10000, 10001), dead_frame_train=10001)
    mock, dc = done.mock, done.dc
    path = tmp_path / "dead.h5"
    shutil.copy2(done.path, path)

    policy = dataclasses.replace(cfg, allow_data_check_failures=allow)
    root = tmp_path / "root"
    target = damnit.damnit_config(
        cfg.proposal, cfg.run, cfg.detector, output_root=str(root)
    ).output_file
    target.parent.mkdir(parents=True)
    shutil.copy2(path, target)
    readers = {
        "pass": lambda: run(policy, mock, dc, output_path=path),
        "reader": lambda: damnit.jungfrau_waxs_from_file(
            cfg.proposal,
            cfg.run,
            cfg.detector,
            output_root=str(root),
            allow_data_check_failures=allow,
        ),
    }
    for name, read in readers.items():
        if allow:
            assert read() is not None, name
        else:
            with pytest.raises(IncompleteRun, match="DATA_CHECK_FAILED"):
                read()

    # The preprocessing cell colours the file already there.
    monkeypatch.setattr(
        "analysis.waxs.run.run_jungfrau_waxs", lambda cfg, **kwargs: None
    )
    cell = damnit.preprocess(
        cfg.proposal,
        cfg.run,
        (cfg.detector,),
        output_root=str(root),
        allow_data_check_failures=allow,
    )
    assert _colour(cell) == COLORS["good" if allow else "error"]
