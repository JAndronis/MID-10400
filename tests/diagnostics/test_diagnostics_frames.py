"""Frame accounting, per-frame scalars and units from frame tables of both passes."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from diagnostics_fakes import FIRST_TRAIN, write_frame_table

from analysis.common.status import FrameStatus
from analysis.diagnostics.frames import (
    default_scalar_column,
    detector_kind,
    frame_accounting,
    frame_scalar,
    per_train,
    units,
)

JF_CELLS = (0, 1, 2, 3, 4, 5, 6, 15)


def signal(n_trains=6, n_labels=4, npt=5, seed=0):
    return (
        np.random.default_rng(seed)
        .poisson(5.0, (n_trains, n_labels, npt))
        .astype(np.float32)
    )


def test_agipd_accounting_counts_every_frame_once(tmp_path):
    path = write_frame_table(tmp_path / "a.h5", "agipd", signal(), empty_trains=2)
    table = frame_accounting(path)
    assert list(table.index) == ["OK"]
    assert list(table.columns) == ["frames"]  # no 155-column pulse table by default
    assert table.loc["OK", "frames"] == 24
    assert table.attrs["detector"] == "agipd"
    assert table.attrs["n_frames"] == 24
    # a NO_FRAMES train owns no rows, so it shows only in the plan's train counts
    assert table.attrs["trains"] == {"OK": 6, "NO_FRAMES": 2}


def test_jungfrau_accounting_breaks_down_by_cell(tmp_path):
    status = np.zeros((6, 8), np.uint8)
    status[1, 2] = FrameStatus.DATA_CHECK_FAILED  # written by the worker: labelled
    status[3, 7] = FrameStatus.LABEL_MISMATCH  # labelled too
    status[5, :] = FrameStatus.WORKER_ERROR  # marked: status only, no labels
    path = write_frame_table(
        tmp_path / "jf.h5", "jungfrau", signal(n_labels=8), labels=JF_CELLS,
        status=status,
    )  # fmt: skip
    table = frame_accounting(path)
    assert list(table.index) == [
        "OK", "LABEL_MISMATCH", "DATA_CHECK_FAILED", "WORKER_ERROR",
    ]  # fmt: skip
    assert table.loc["OK", "frames"] == 38
    assert table.loc["DATA_CHECK_FAILED", 2] == 1
    assert table.loc["LABEL_MISMATCH", 15] == 1
    # the marked rows carry cellId 0 as fill; they must not be booked to cell 0
    assert table.loc["WORKER_ERROR", "unlabelled"] == 8
    assert table.loc["WORKER_ERROR", 0] == 0
    assert table.loc["OK", 0] == 5
    per_label = table.drop(columns="frames").sum(axis=1)
    assert (per_label == table["frames"]).all()


def test_accounting_by_pulse_on_request(tmp_path):
    path = write_frame_table(tmp_path / "a.h5", "agipd", signal())
    table = frame_accounting(path, by="reader_pulseId")
    assert list(table.columns) == ["frames", 0, 1, 2, 3]
    assert frame_accounting(path, by=None).shape == (1, 1)


def test_units_default_and_recorded(tmp_path):
    agipd = write_frame_table(tmp_path / "a.h5", "agipd", signal())
    kev = write_frame_table(tmp_path / "k.h5", "jungfrau", signal())
    photons = write_frame_table(
        tmp_path / "p.h5", "jungfrau", signal(), units="photons"
    )
    root = write_frame_table(
        tmp_path / "r.h5", "jungfrau", signal(), units="photons", units_on_root=True
    )
    assert units(agipd) == "photons"
    assert units(kev) == "keV"
    assert units(photons) == "photons"
    assert units(root) == "photons"
    assert per_train(kev).attrs["units_recorded"] is False
    assert per_train(photons).attrs["units_recorded"] is True
    expected = {"units": "photons", "detector": "agipd"}
    assert expected.items() <= per_train(agipd).attrs.items()
    with h5py.File(tmp_path / "x.h5", "w") as f:
        f.create_group("frames")
        with pytest.raises(ValueError, match="neither"):
            detector_kind(f)


def test_default_scalar_prefers_photons_on_jungfrau(tmp_path):
    s = signal()
    plain = write_frame_table(tmp_path / "k.h5", "jungfrau", s)
    photonised = write_frame_table(
        tmp_path / "p.h5", "jungfrau", s,
        extra_columns={"n_photons": s.sum(-1).astype(np.float32)},
    )  # fmt: skip
    assert default_scalar_column(plain) == "energy_valid"
    assert default_scalar_column(photonised) == "n_photons"
    assert default_scalar_column(write_frame_table(tmp_path / "a.h5", "agipd", s)) == (
        "photons_valid"
    )


def test_frame_scalar_places_frames_by_label(tmp_path):
    """Cell 15 is acquired first on r0423 but stored last; position must not matter."""
    s = signal(n_trains=3, n_labels=8)
    scalar = np.arange(24, dtype=np.float32).reshape(3, 8)
    status = np.zeros((3, 8), np.uint8)
    status[1, 0] = FrameStatus.DATA_CHECK_FAILED
    cells = (15, 0, 1, 2, 3, 4, 5, 6)
    path = write_frame_table(
        tmp_path / "jf.h5", "jungfrau", s, labels=cells, scalar=scalar, status=status
    )
    grid = frame_scalar(path)
    assert grid.dims == ("trainId", "cellId")
    assert list(grid.cellId.values) == sorted(cells)
    assert grid.sel(trainId=FIRST_TRAIN, cellId=15) == 0
    assert grid.sel(trainId=FIRST_TRAIN, cellId=6) == 7
    assert np.isnan(grid.sel(trainId=FIRST_TRAIN + 1, cellId=15))  # not OK
    assert grid.attrs["column"] == "energy_valid"


def test_frame_scalar_refuses_shared_slots(tmp_path):
    path = write_frame_table(
        tmp_path / "jf.h5", "jungfrau", signal(), labels=(0, 0, 1, 2)
    )
    with pytest.raises(ValueError, match="share"):
        frame_scalar(path)
