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
from analysis.triage.damnit import NO_HITS_COLOR, hit_summary_cell


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


def test_static_cell_counts_hits_and_off(mesh_store):
    cell = hit_summary_cell(
        10400, 423, *memory(mesh_store, 423), damnit_root=mesh_store, cache_dir=None
    )
    assert cell.summary_value == "600 hits / 0 edge / 20 off of 620"
    assert cell.background == COLORS["good"]


@pytest.mark.parametrize(
    ("classes", "ambiguous", "text", "colour"),
    [
        (["on", "edge", "off", "on"], False, "2 hits / 1 edge / 1 off of 4", "good"),
        (["off"] * 5, False, "0 hits / 0 edge / 5 off of 5", None),
        (["none"] * 3, True, "0 hits / 0 edge / 0 off of 3 (ambiguous)", "warning"),
    ],
)
def test_the_cell_counts_the_classes_and_colours_the_verdict(
    monkeypatch, classes, ambiguous, text, colour
):
    import xarray as xr

    import analysis.triage.damnit as module

    tids = np.arange(len(classes), dtype=np.uint64)
    found = {
        "kind": "mesh", "block": None,
        "classes": xr.DataArray(np.array(classes), dims="trainId",
                                coords={"trainId": tids}),
        "summary": {"ambiguous": ambiguous},
    }  # fmt: skip
    monkeypatch.setattr(module, "classify_in_memory", lambda *a, **k: found)
    cell = hit_summary_cell(10400, 1, None, None, None)
    assert cell.summary_value == text
    assert cell.background == (NO_HITS_COLOR if colour is None else COLORS[colour])


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


def test_hexapod_positions_follow_the_train_ids_and_need_the_hexapod():
    import xarray as xr

    from analysis.triage.damnit import HEXAPOD, hexapod_positions

    class Run:
        all_sources = {HEXAPOD}

        def __getitem__(self, key):
            axis = 1.0 if key[1].startswith("X") else 2.0
            values = axis * np.array([10.0, 11.0, 12.0])
            values = xr.DataArray(values, dims="trainId", coords={"trainId": [5, 6, 7]})
            return types.SimpleNamespace(xarray=lambda: values)

    x, z = hexapod_positions(Run(), np.array([7, 5, 9]))
    np.testing.assert_array_equal(x, [12.0, 10.0, np.nan])
    np.testing.assert_array_equal(z, [24.0, 20.0, np.nan])
    Run.all_sources = set()
    assert hexapod_positions(Run(), np.array([5])) is None
    assert hexapod_positions(None, np.array([5])) is None
