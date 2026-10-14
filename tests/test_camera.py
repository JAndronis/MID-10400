"""Droplet videos (``analysis.camera``) on a mock camera written by EXtra-data.

The mock is EXtra-data's Basler camera under CAM5's source name, so the reads go
through the same ``KeyData`` calls as on a real run. The video is decoded again
with the same ffmpeg: one frame per image, even sides, the label drawn, the
display choices in the comment, and no partial file left by a failed write.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

pytest.importorskip("extra_data")
imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
pytest.importorskip("PIL")

from analysis.camera import (  # noqa: E402
    CAMERA_SOURCE,
    IMAGE_KEY,
    camera_facts,
    count_frames,
    first_image,
    gaps,
    to_8bit,
    write_camera_video,
)
from camera_fakes import FIRST_TRAIN, mock_camera  # noqa: E402


def comment(path) -> str:
    """The MP4 comment, as ffmpeg prints the input's metadata."""
    out = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True,
    ).stderr  # fmt: skip
    lines = [line for line in out.splitlines() if line.strip().startswith("comment")]
    return lines[0].split(":", 1)[1].strip() if lines else ""


def test_one_labelled_frame_per_image(tmp_path):
    run = mock_camera(tmp_path / "raw.h5")
    out = tmp_path / "videos" / "r0001_cam5.mp4"
    info = write_camera_video(run, out, label="r0001", fps=30, workers=3)
    assert info["frames"] == 24 and info["gaps"] == 0
    assert count_frames(out) == 24
    assert info["size"] == [80, 60]  # even sides
    reader = imageio_ffmpeg.read_frames(str(out), pix_fmt="gray", bpp=1)
    meta = next(reader)
    assert tuple(meta["size"]) == (80, 60)
    frame = np.frombuffer(next(reader), np.uint8).reshape(60, 80)
    # the label is white on black in the top-left corner of a dark-left ramp
    assert frame[:12, :40].max() > 200 and frame[:12, :40].min() < 40
    text = comment(out)
    assert "r0001" in text and CAMERA_SOURCE in text and "fps 30" in text
    assert f"trains {FIRST_TRAIN}-{FIRST_TRAIN + 23}" in text
    assert not list(out.parent.glob(".*partial*"))


def test_gaps_are_counted_and_every_image_is_a_frame(tmp_path):
    run = mock_camera(tmp_path / "raw.h5", n_trains=30, n_images=20)
    facts = camera_facts(run)
    assert facts["n_images"] == 20 and facts["gaps"] > 0
    out = tmp_path / "v.mp4"
    info = write_camera_video(run, out, workers=2)
    assert info["frames"] == count_frames(out) == 20
    assert info["gaps"] == facts["gaps"]


def test_a_failed_write_keeps_the_old_video_and_leaves_no_partial(tmp_path):
    run = mock_camera(tmp_path / "raw.h5", n_trains=8)
    out = tmp_path / "v.mp4"
    write_camera_video(run, out, workers=2)
    before = out.read_bytes()
    with pytest.raises(RuntimeError, match="ffmpeg exited"):
        write_camera_video(run, out, crf=-7, workers=2)  # libx264 refuses
    assert out.read_bytes() == before
    assert [p.name for p in tmp_path.glob("*.mp4*")] == ["v.mp4"]
    assert not list(tmp_path.glob(".*"))


def test_first_image_reads_the_first_train(tmp_path):
    run = mock_camera(tmp_path / "raw.h5", n_trains=10, n_images=5)
    train_id, image = first_image(run)
    tids = run[CAMERA_SOURCE, IMAGE_KEY].train_id_coordinates()
    assert train_id == tids[0]
    np.testing.assert_array_equal(image, run[CAMERA_SOURCE, IMAGE_KEY].ndarray()[0])


def test_a_run_without_images_raises(tmp_path):
    run = mock_camera(tmp_path / "raw.h5", n_trains=5, n_images=0)
    assert camera_facts(run)["n_images"] == 0
    with pytest.raises(ValueError, match="no images"):
        first_image(run)
    with pytest.raises(ValueError, match="no images"):
        write_camera_video(run, tmp_path / "v.mp4")


def test_to_8bit_averages_bins_and_crops_to_even_sides():
    image = np.arange(5 * 7, dtype=np.uint16).reshape(5, 7)
    out = to_8bit(image, (0.0, 255.0), binning=2)
    assert out.shape == (2, 2) and out.dtype == np.uint8
    assert out[0, 0] == int(image[:2, :2].mean())
    assert to_8bit(image, (0.0, 255.0), binning=1).shape == (4, 6)
    assert to_8bit(np.full((4, 4), 9000), (0.0, 4000.0), 1).max() == 255


def test_gaps_use_the_usual_step():
    assert gaps([1, 2, 3, 4]) == (1, 0)
    assert gaps([3, 6, 9, 15, 18]) == (3, 1)
    assert gaps([7]) == (0, 0)
