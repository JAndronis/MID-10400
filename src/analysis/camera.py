"""Droplet camera videos: every image of a run as a labelled H.264 file.

The camera images go to tape with raw, so each run keeps a video in
``usr/cached_files/droplet_videos/``. Every frame is labelled with the run, the
train ID and the seconds since the first image, because the video plays at a
constant rate and a DAQ gap shows only as a jump. The MP4 ``comment`` records
the display choices; the raw data are untouched.

ffmpeg is ``imageio-ffmpeg``'s static build (with libx264), because there is
none on PATH and the EuXFEL env's is redeployed nightly. It has no ffprobe, so
:func:`count_frames` decodes the file with the same ffmpeg.

Nothing here imports DAMNIT; the cells are in ``analysis.diagnostics.damnit``.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CAMERA_SOURCE",
    "VIDEO_ROOT",
    "camera_facts",
    "count_frames",
    "first_image",
    "gaps",
    "label_frame",
    "to_8bit",
    "video_path",
    "write_camera_video",
]

#: The droplet camera: (494, 659) uint16, one image per train.
CAMERA_SOURCE = "MID_EXP_SAM/CAM/CAM5:daqOutput"
IMAGE_KEY = "data.image.pixels"
VIDEO_ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400/usr/cached_files/droplet_videos")

#: Trains per second, for the seconds in the label.
TRAIN_RATE_HZ = 10
#: 3x real time for a camera that records every train.
DEFAULT_FPS = 30
DEFAULT_BINNING = 1
DEFAULT_CRF = 23
PRESET = "veryfast"
#: Images per task in the thread pool, and images sampled for the grey levels.
BLOCK = 4
LIMIT_SAMPLES = 8
PERCENTILES = (0.5, 99.5)
#: Bumped when the frames change for the same inputs; written into the comment.
VIDEO_VERSION = 1


def video_path(
    run_no: int, camera: str = "cam5", root: str | Path = VIDEO_ROOT
) -> Path:
    """``{root}/r{run:04d}_{camera}.mp4``."""
    return Path(root) / f"r{run_no:04d}_{camera}.mp4"


def gaps(train_ids) -> tuple[int, int]:
    """The camera's usual train step and the number of longer steps (DAQ drops)."""
    steps = np.diff(np.asarray(train_ids, dtype=np.int64))
    if not steps.size:
        return 0, 0
    step = int(np.bincount(steps).argmax())
    return step, int((steps > step).sum())


def camera_facts(run, source: str = CAMERA_SOURCE) -> dict[str, Any]:
    """Shape, dtype and train coverage of a camera in ``run``, reading no image."""
    if source not in run.all_sources:
        return {"source": source, "present": False, "n_images": 0}
    kd = run[source, IMAGE_KEY]
    tids = np.asarray(kd.train_id_coordinates(), dtype=np.uint64)
    step, n_gaps = gaps(tids)
    return {
        "source": source,
        "present": True,
        "shape": tuple(int(n) for n in kd.entry_shape),
        "dtype": str(kd.dtype),
        "n_images": int(tids.size),
        "n_trains": len(run.train_ids),
        "first_train": int(tids[0]) if tids.size else None,
        "last_train": int(tids[-1]) if tids.size else None,
        "step": step,
        "gaps": n_gaps,
    }


def first_image(run, source: str = CAMERA_SOURCE) -> tuple[int, np.ndarray]:
    """The run's first image of ``source`` and its train ID, reading only it.

    ``kd.ndarray()[0]`` would load every image, enough to run out of memory
    inside DAMNIT.

    :raises ValueError: the camera recorded no image in this run.
    """
    from extra_data import by_id

    kd = run[source, IMAGE_KEY]
    tids = kd.train_id_coordinates()
    if not len(tids):
        raise ValueError(f"no images of {source} in this run")
    one = kd.select_trains(by_id[[int(tids[0])]])
    return int(one.train_id_coordinates()[0]), one.ndarray()[0]


def to_8bit(image: np.ndarray, limits: tuple[float, float], binning: int) -> np.ndarray:
    """Average ``binning`` x ``binning`` pixels, crop to even sides, scale to uint8.

    yuv420p needs even sides.
    """
    img = np.asarray(image, dtype=np.float32)
    if binning > 1:
        h, w = (np.array(img.shape) // binning) * binning
        img = img[:h, :w].reshape(h // binning, binning, w // binning, binning)
        img = img.mean(axis=(1, 3))
    h, w = (np.array(img.shape) // 2) * 2
    lo, hi = limits
    scaled = (img[:h, :w] - lo) * (255.0 / max(hi - lo, 1.0))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def label_frame(frame: np.ndarray, text: str, font) -> np.ndarray:
    """Draw ``text`` in the top-left corner, white with a black outline."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    size = font.size
    ImageDraw.Draw(img).text(
        (max(2, frame.shape[1] // 60), max(2, frame.shape[0] // 40)),
        text,
        fill=255,
        font=font,
        stroke_width=max(1, size // 8),
        stroke_fill=0,
    )
    return np.asarray(img)


def write_camera_video(
    run,
    path: str | Path,
    *,
    source: str = CAMERA_SOURCE,
    label: str = "",
    binning: int = DEFAULT_BINNING,
    fps: int = DEFAULT_FPS,
    crf: int = DEFAULT_CRF,
    limits: tuple[float, float] | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    """Write every image of ``source`` in ``run`` to ``path`` (MP4, replaced).

    The file is written beside ``path`` under a temporary name and renamed into
    place, so a second job writing the same run (DAMNIT can launch a variable
    twice) never leaves a truncated video behind, and a failed write leaves the
    old one.

    :param limits: grey levels mapped to 0 and 255; default the
        ``PERCENTILES`` of ``LIMIT_SAMPLES`` images spread over the run.
    :param workers: threads preparing frames. Explicit, never the CPU count,
        which counts hyperthreads; libx264 threads itself.
    :returns: frames written, first and last train, limits, the usual step, the
        number of gaps, the saturated fraction of the sample, the output size
        and the wall time.
    :raises ValueError: the camera recorded no image in this run.
    """
    import imageio_ffmpeg
    from extra_data import by_id
    from PIL import ImageFont

    started = time.perf_counter()
    path = Path(path)
    kd = run[source, IMAGE_KEY]
    tids = np.asarray(kd.train_id_coordinates(), dtype=np.uint64)
    if not tids.size:
        raise ValueError(f"no images of {source} in this run")
    step, n_gaps = gaps(tids)
    picks = tids[
        np.linspace(0, tids.size - 1, min(LIMIT_SAMPLES, tids.size)).astype(int)
    ]
    sample = kd.select_trains(by_id[[int(t) for t in picks]]).ndarray()
    if limits is None:
        limits = tuple(float(v) for v in np.percentile(sample, PERCENTILES))
    lo, hi = limits
    top = float(np.iinfo(sample.dtype).max) if sample.dtype.kind in "iu" else np.inf
    saturated = float((sample >= top).mean())

    h, w = (np.array(kd.entry_shape) // binning) // 2 * 2
    font = ImageFont.load_default(size=max(10, int(h) // 28))
    t0 = int(tids[0])

    def prepare(block: np.ndarray) -> bytes:
        sel = kd.select_trains(by_id[[int(t) for t in block]])
        images = sel.ndarray()
        got = np.asarray(sel.train_id_coordinates(), dtype=np.uint64)
        if not np.array_equal(got, block):  # identity from the reader, not position
            raise RuntimeError(f"asked for trains {block}, got {got}")
        frames = [
            label_frame(
                to_8bit(img, limits, binning),
                f"{label}  train {int(tid)}  {(int(tid) - t0) / TRAIN_RATE_HZ:7.1f} s",
                font,
            )
            for tid, img in zip(got, images, strict=True)
        ]
        return np.stack(frames).tobytes()

    comment = (
        f"{label} {source} trains {t0}-{int(tids[-1])} ({tids.size} images, "
        f"step {step}, {n_gaps} gaps) limits {lo:.0f}-{hi:.0f} "
        f"(percentiles {PERCENTILES[0]}/{PERCENTILES[1]} of {picks.size} images) "
        f"bin {binning} fps {fps} crf {crf} {PRESET} analysis.camera v{VIDEO_VERSION}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.{os.getpid()}.partial.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-preset", PRESET, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-metadata", f"comment={comment}", str(partial),
    ]  # fmt: skip
    with tempfile.TemporaryFile() as errors:
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            broken = False
            try:
                with ThreadPoolExecutor(workers) as pool:
                    pending: deque = deque()
                    for block in np.array_split(tids, max(1, tids.size // BLOCK)):
                        pending.append(pool.submit(prepare, block))
                        if len(pending) > 2 * workers:  # bounds the frames in memory
                            ffmpeg.stdin.write(pending.popleft().result())
                    while pending:
                        ffmpeg.stdin.write(pending.popleft().result())
                ffmpeg.stdin.close()
            except BrokenPipeError:
                broken = True  # ffmpeg stopped reading; its own message says why
            code = ffmpeg.wait()
            if code or broken:
                errors.seek(0)
                message = errors.read().decode(errors="replace")[-2000:]
                raise RuntimeError(f"ffmpeg exited with {code}: {message}")
            os.replace(partial, path)
        except BaseException:
            if ffmpeg.poll() is None:
                ffmpeg.kill()
            ffmpeg.wait()
            partial.unlink(missing_ok=True)
            raise
    return {
        "path": str(path),
        "source": source,
        "frames": int(tids.size),
        "first_train": t0,
        "last_train": int(tids[-1]),
        "step": step,
        "gaps": n_gaps,
        "limits": [lo, hi],
        "saturated_fraction": saturated,
        "size": [int(w), int(h)],
        "binning": binning,
        "fps": fps,
        "crf": crf,
        "bytes": path.stat().st_size,
        "wall_s": round(time.perf_counter() - started, 2),
        "comment": comment,
    }


def count_frames(path: str | Path) -> int:
    """Frames in a video, by decoding every one with the same ffmpeg."""
    import imageio_ffmpeg

    return int(imageio_ffmpeg.count_frames_and_secs(str(path))[0])
