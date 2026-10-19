"""The ``*_frames_used`` cells (phase D): colour by status, Skip without a file.

The cells read frame tables written by the shared writer (`diagnostics_fakes`),
and DAMNIT's own ``Cell``, so what is checked is what ctxrunner would store.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("damnit")

from damnit.context import Cell, Skip  # noqa: E402
from diagnostics_fakes import write_frame_table  # noqa: E402

from analysis.common.status import FrameStatus  # noqa: E402
from analysis.diagnostics.damnit import (  # noqa: E402
    COLORS,
    droplet_first_image,
    droplet_video,
    frames_used,
    frames_used_cell,
    output_file,
)


def colour(cell) -> str:
    """DAMNIT stores the background as RGB bytes."""
    return "#" + bytes(cell.background).hex()


def table(tmp_path, status, kind="agipd"):
    n_trains, n_labels = status.shape
    signal = np.ones((n_trains, n_labels, 5))
    return write_frame_table(tmp_path / "t.h5", kind, signal, status=status)


def test_every_frame_ok_is_green(tmp_path):
    cell = frames_used_cell(table(tmp_path, np.zeros((4, 3), np.uint8)))
    assert isinstance(cell, Cell)
    assert cell.summary_value == "12 / 12 (100.0%)"
    assert colour(cell) == COLORS["good"]
    assert int(cell.data) == 12


def test_a_worker_error_is_red_and_named(tmp_path):
    status = np.zeros((4, 3), np.uint8)
    status[0, :] = FrameStatus.NOT_PROCESSED  # commoner, but not a failure
    status[1, 0] = FrameStatus.WORKER_ERROR
    cell = frames_used_cell(table(tmp_path, status))
    assert cell.summary_value == "8 / 12 (66.7%) · WORKER_ERROR 1"
    assert colour(cell) == COLORS["error"]


@pytest.mark.parametrize("kind", ["agipd", "jungfrau"])
def test_a_label_mismatch_is_red(tmp_path, kind):
    status = np.zeros((2, 2), np.uint8)
    status[1, 1] = FrameStatus.LABEL_MISMATCH
    cell = frames_used_cell(table(tmp_path, status, kind))
    assert colour(cell) == COLORS["error"]


def test_an_incomplete_run_is_amber(tmp_path):
    status = np.zeros((4, 3), np.uint8)
    status[2:] = FrameStatus.NOT_PROCESSED
    status[0, 0] = FrameStatus.MISSING_MODULES
    cell = frames_used_cell(table(tmp_path, status))
    assert cell.summary_value == "5 / 12 (41.7%) · NOT_PROCESSED 6"
    assert colour(cell) == COLORS["warning"]


def test_no_file_is_skipped(tmp_path):
    with pytest.raises(Skip):
        frames_used_cell(tmp_path / "absent.h5")


def test_a_file_another_process_writes_is_busy(tmp_path, monkeypatch):
    # `import extra_speckle` turns HDF5 file locking off for the whole process
    # (and DAMNIT's context imports it); HDF5 reads the variable at every open,
    # so turn it back on for this reader and its writer.
    monkeypatch.setenv("HDF5_USE_FILE_LOCKING", "TRUE")
    path = table(tmp_path, np.zeros((2, 2), np.uint8))
    hold = (
        "import sys, h5py; f = h5py.File(sys.argv[1], 'r+'); "
        "print('open', flush=True); sys.stdin.read()"
    )
    writer = subprocess.Popen(
        [sys.executable, "-c", hold, str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        env={**os.environ, "HDF5_USE_FILE_LOCKING": "TRUE"},
    )  # fmt: skip
    try:
        assert writer.stdout.readline().strip() == "open"
        cell = frames_used_cell(path)
    finally:
        writer.communicate("", timeout=30)
    assert (cell.summary_value, colour(cell)) == ("busy", COLORS["warning"])


def test_a_damaged_file_is_red(tmp_path):
    path = tmp_path / "t.h5"
    path.write_bytes(b"not an hdf5 file")
    cell = frames_used_cell(path)
    assert (cell.summary_value, colour(cell)) == ("unreadable", COLORS["error"])


def test_the_path_is_the_pass_config_output(tmp_path):
    agipd = output_file(10400, 423, "agipd", output_root=str(tmp_path))
    jf2 = output_file(10400, 423, "jf2", output_root=str(tmp_path))
    assert agipd == tmp_path / "r0423" / "agipd_saxs.h5"
    assert jf2 == tmp_path / "r0423" / "jungfrau_waxs_jf2.h5"
    with pytest.raises(ValueError):
        output_file(10400, 423, "jf3")
    with pytest.raises(Skip):
        frames_used(10400, 423, "jf1", output_root=str(tmp_path))


@pytest.fixture
def camera_run(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from camera_fakes import mock_camera

    return mock_camera(tmp_path / "raw.h5", n_trains=12, n_images=9)


def test_the_droplet_video_cell_counts_decoded_frames(camera_run, tmp_path):
    cell = droplet_video(camera_run, 7, root=tmp_path / "videos", workers=2)
    assert isinstance(cell, Cell)
    assert cell.data == str(tmp_path / "videos" / "r0007_cam5.mp4")
    assert cell.summary_value.startswith("9 frames, ") and "gaps" in cell.summary_value


def test_the_first_image_cell_is_one_image_with_its_train(camera_run):
    image = droplet_first_image(camera_run)
    assert image.dims == ("y", "x") and image.shape == (60, 81)
    first = camera_run["MID_EXP_SAM/CAM/CAM5:daqOutput", "data.image.pixels"]
    assert image.attrs["trainId"] == first.train_id_coordinates()[0]


def test_a_run_without_the_camera_is_skipped(tmp_path):
    from extra_data import H5File
    from extra_data.tests.mockdata import write_file
    from extra_data.tests.mockdata.xgm import XGM

    write_file(str(tmp_path / "raw.h5"), [XGM("SA2_XTD1_XGM/DOOCS/MAIN")], 5)
    run = H5File(str(tmp_path / "raw.h5"))
    with pytest.raises(Skip):
        droplet_first_image(run)
    with pytest.raises(Skip):
        droplet_video(run, 1, root=tmp_path)


# ── photon statistics, XPCS score, overview (user, 2026-10-14) ────────────────
def test_the_statistics_overview_draws_three_panels(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from test_diagnostics_photon_stats import nb_frames

    from analysis.diagnostics.damnit import photon_statistics_overview
    from analysis.diagnostics.photon_stats import summary_of

    ds, weights, _ = nb_frames(np.random.default_rng(9), 300)
    s = summary_of(ds, weights)
    s.attrs.update(kind="static", n_hits=30)
    fig = photon_statistics_overview(s)
    assert len(fig.axes) == 3


def test_photon_statistics_skip_without_a_file(tmp_path):
    from analysis.diagnostics.damnit import photon_statistics

    with pytest.raises(Skip, match="no photon statistics"):
        photon_statistics(10400, 9999, None, None, None)


@pytest.mark.parametrize(
    ("ok", "plain", "extractable", "best", "expected"),
    [
        (7, 4, True, 10.04, "good"),
        (7, 4, False, 1.7, "error"),
        (7, 4, None, float("nan"), "warning"),
        (6, 4, True, 10.04, "warning"),
        (7, 3, False, 1.7, "warning"),
        (None, None, True, 10.04, "warning"),
    ],
)
def test_the_xpcs_score_cell(monkeypatch, ok, plain, extractable, best, expected):
    from analysis.diagnostics.damnit import xpcs_score_cell

    score = {
        "run": 318, "n_hits": 259,
        "hits": {}, "kbar": None, "n_rois_kbar_ok": ok, "n_plain_rois_kbar_ok": plain,
        "snr": {"best_snr": best, "extractable": extractable, "slices": []},
    }  # fmt: skip
    monkeypatch.setattr(
        "analysis.diagnostics.xpcs.xpcs_score", lambda run, proposal: score
    )
    cell = xpcs_score_cell(10400, 318)
    snr = "SNR —" if extractable is None else f"SNR {best:.1f}"
    kbar = f" · k̄ ok {ok}/7 ({plain}/4 plain)" if ok is not None else ""
    assert cell.summary_value == f"259 hits{kbar} · {snr}"
    assert colour(cell) == COLORS[expected]


def test_the_xpcs_score_skips_a_run_without_results(monkeypatch):
    from analysis.diagnostics.damnit import xpcs_score_cell

    def missing(run, proposal):
        raise LookupError("r5: no XPCS results file under /x")

    monkeypatch.setattr("analysis.diagnostics.xpcs.xpcs_score", missing)
    with pytest.raises(Skip, match="no XPCS results file"):
        xpcs_score_cell(10400, 5)
