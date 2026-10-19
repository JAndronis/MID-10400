"""The hit-recognition cell and figure, over DAMNIT values in memory."""

from __future__ import annotations

import sys
import types

import matplotlib
import numpy as np
import pytest
from test_classify import in_memory, mesh_run, write_damnit
from triage_fakes import run_row, write_sqlite

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


@pytest.fixture
def store(damnit_root):
    rng = np.random.default_rng(6)
    for r in (120, 121, 122):
        x, _ = mesh_run(rng, -4.3, 0.2, n=400)
        saxs = np.broadcast_to((10.0**x)[:, None, None], (400, 2, 30)).copy()
        write_damnit(damnit_root / "extracted_data" / f"p10400_r{r}.h5", saxs,
                     xgm=np.full(400, 10.0))  # fmt: skip
    rows = [
        run_row(r, scan_type="dmesh SSHEX", total_transmission=0.0123, pulses=350.0)
        for r in (120, 121, 122)
    ]
    level = 10.0 ** np.r_[np.full(300, -3.0), np.full(20, -4.3), np.full(300, -3.0)]
    write_damnit(damnit_root / "extracted_data" / "p10400_r423.h5",
                 np.broadcast_to(level[:, None, None], (620, 2, 30)).copy(),
                 xgm=np.full(620, 10.0))  # fmt: skip
    write_sqlite(damnit_root / "runs.sqlite", rows + [run_row(423)])
    return damnit_root


def memory(root, run):
    saxs, xgm, trans, _ = in_memory(root / "extracted_data" / f"p10400_r{run}.h5")
    return saxs, xgm, trans


def test_mesh_cell_counts_classes_and_is_green(store):
    cell = hit_summary_cell(10400, 121, *memory(store, 121), damnit_root=store,
                            cache_dir=None)  # fmt: skip
    assert cell.summary_value.endswith("of 400")
    assert " hits / " in cell.summary_value and " edge / " in cell.summary_value
    assert cell.background == COLORS["good"]


def test_static_cell_counts_hits_and_off(store):
    cell = hit_summary_cell(10400, 423, *memory(store, 423), damnit_root=store,
                            cache_dir=None)  # fmt: skip
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


def test_mesh_figure_has_map_when_positions_are_known(store):
    class Run:
        all_sources = {"MID_EXP_SAM/HEXAPOD/SSHEX"}

        def __getitem__(self, key):
            import xarray as xr

            tids = memory(store, 121)[0].trainId.values
            rng = np.random.default_rng(1)
            return types.SimpleNamespace(
                xarray=lambda: xr.DataArray(rng.random(tids.size), dims="trainId",
                                            coords={"trainId": tids})
            )  # fmt: skip

    args = memory(store, 121)
    with_map = hit_geometry_figure(10400, 121, *args, run=Run(), damnit_root=store,
                                   cache_dir=None)  # fmt: skip
    without = hit_geometry_figure(10400, 121, *args, damnit_root=store, cache_dir=None)
    assert len(with_map.axes) == 3 and len(without.axes) == 2


def test_static_figure_is_timeline_and_counts(store):
    fig = hit_geometry_figure(10400, 423, *memory(store, 423), damnit_root=store,
                              cache_dir=None)  # fmt: skip
    assert len(fig.axes) == 2
