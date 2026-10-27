"""The hit-recognition cell and figure, over DAMNIT values in memory."""

from __future__ import annotations

import sys
import types

import matplotlib
import numpy as np
import pytest
from triage_fakes import in_memory

matplotlib.use("Agg")

from analysis.diagnostics.damnit import COLORS
from analysis.triage.damnit import (
    NO_HITS_COLOR,
    hit_geometry_figure,
    hit_summary_cell,
)


class Cell:
    def __init__(self, data=None, summary_value=None, background=None):
        self.data, self.summary_value, self.background = data, summary_value, background


@pytest.fixture(autouse=True)
def fake_damnit(monkeypatch):
    module = types.ModuleType("damnit.context")
    module.Cell = Cell
    monkeypatch.setitem(sys.modules, "damnit", types.ModuleType("damnit"))
    monkeypatch.setitem(sys.modules, "damnit.context", module)


def memory(root, run):
    saxs, xgm, trans, _ = in_memory(root / "extracted_data" / f"p10400_r{run}.h5")
    return saxs, xgm, trans


def test_mesh_cell_counts_classes_and_is_green(mesh_store):
    cell = hit_summary_cell(
        10400, 121, *memory(mesh_store, 121), damnit_root=mesh_store, cache_dir=None
    )
    assert cell.summary_value.endswith("of 400")
    assert " hits / " in cell.summary_value and " edge / " in cell.summary_value
    assert cell.background == COLORS["good"]


def test_static_cell_counts_hits_and_off(mesh_store):
    cell = hit_summary_cell(
        10400, 423, *memory(mesh_store, 423), damnit_root=mesh_store, cache_dir=None
    )
    assert cell.summary_value == "600 hits / 0 edge / 20 off of 620"


def test_a_run_without_hits_is_grey(monkeypatch):
    import xarray as xr

    import analysis.triage.damnit as module

    tids = np.arange(5, dtype=np.uint64)
    found = {
        "kind": "static",
        "block": None,
        "classes": xr.DataArray(np.array(["off"] * 5), dims="trainId",
                                coords={"trainId": tids}),
        "summary": {"ambiguous": False},
    }  # fmt: skip
    monkeypatch.setattr(module, "classify_in_memory", lambda *a, **k: found)
    cell = hit_summary_cell(10400, 1, None, None, None)
    assert cell.summary_value == "0 hits / 0 edge / 5 off of 5"
    assert cell.background == NO_HITS_COLOR


def test_mesh_figure_has_map_when_positions_are_known(mesh_store):
    class Run:
        all_sources = {"MID_EXP_SAM/HEXAPOD/SSHEX"}

        def __getitem__(self, key):
            import xarray as xr

            tids = memory(mesh_store, 121)[0].trainId.values
            rng = np.random.default_rng(1)
            return types.SimpleNamespace(
                xarray=lambda: xr.DataArray(rng.random(tids.size), dims="trainId",
                                            coords={"trainId": tids})
            )  # fmt: skip

    args = memory(mesh_store, 121)
    with_map = hit_geometry_figure(
        10400, 121, *args, run=Run(), damnit_root=mesh_store, cache_dir=None
    )
    without = hit_geometry_figure(
        10400, 121, *args, damnit_root=mesh_store, cache_dir=None
    )
    assert len(with_map.axes) == 3 and len(without.axes) == 2


def test_static_figure_is_timeline_and_counts(mesh_store):
    fig = hit_geometry_figure(
        10400, 423, *memory(mesh_store, 423), damnit_root=mesh_store, cache_dir=None
    )
    assert len(fig.axes) == 2


def test_the_scan_map_puts_each_train_at_its_position_in_its_class_colour():
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.colors import to_rgba

    from analysis.triage.geometry import CLASS_COLORS
    from analysis.triage.geometry import hit_geometry_figure as figure

    classes = np.array(["off", "on", "edge", "on", "off", "on"])
    tids = np.arange(100, 106, dtype=np.uint64)
    found = {
        "run": 121, "kind": "mesh", "block": "b",
        "classes": xr.DataArray(classes, dims="trainId", coords={"trainId": tids}),
        "overlap": xr.DataArray(np.linspace(0, 1, 6), dims="trainId",
                                coords={"trainId": tids}),
    }  # fmt: skip
    x, z = np.arange(6.0), 10.0 + np.arange(6.0) ** 2
    fig = figure(found, (x, z))
    scan_map = fig.axes[0]
    drawn = {}
    for points in scan_map.collections:
        colour = tuple(points.get_facecolor()[0])
        for xy in points.get_offsets():
            drawn[tuple(xy)] = colour
    plt.close(fig)
    assert drawn == {
        (x[i], z[i]): to_rgba(CLASS_COLORS[c]) for i, c in enumerate(classes)
    }
