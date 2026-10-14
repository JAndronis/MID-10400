"""The DAMNIT cells of the diagnostics: frames used (phase D), droplet videos (E).

Each ``*_frames_used`` variable reads a pass's frame table straight from the
path its config names. It never depends on the pass's DAMNIT variable, so it
cannot relaunch the cluster job (CLAUDE.md pitfall 26). The cost is that a cell
shows the file as it was when the cell ran. After a reprocess, refresh it with
``damnit reprocess --match "frames used"`` (``--match`` compares titles, not
names, in damnit 0.4.0).

The context-file wrappers are an import plus a call, like the integrators'.
DAMNIT's ``Cell`` and ``Skip`` are imported inside the function, so this module
imports without DAMNIT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.common.status import FrameStatus

__all__ = [
    "COLORS",
    "DETECTORS",
    "droplet_first_image",
    "droplet_video",
    "frames_used",
    "frames_used_cell",
    "output_file",
]

#: Cell backgrounds (p010646's context, transfer_context/templates).
COLORS = {"good": "#33e32a", "warning": "#f2d40f", "error": "#ec0839"}

#: The passes with a frame table, by the detector name the wrappers use.
DETECTORS = ("agipd", "jf1", "jf2")

#: A frame with one of these codes means the pass went wrong, not the data.
FAILED = (FrameStatus.WORKER_ERROR, FrameStatus.LABEL_MISMATCH)

#: HDF5's words for a file another process holds open for writing.
_LOCKED = ("unable to lock file", "already open", "resource temporarily unavailable")


def output_file(proposal: int, run_no: int, detector: str, **overrides: Any) -> Path:
    """The frame table the pass writes for this run, from its own config.

    :param detector: ``agipd``, ``jf1`` or ``jf2``.
    :param overrides: config fields, e.g. ``output_root``.
    """
    if detector == "agipd":
        from analysis.saxs.damnit import config_for

        return config_for(proposal, run_no, **overrides).output_file
    if detector not in DETECTORS:
        raise ValueError(f"detector must be one of {DETECTORS}, got {detector!r}")
    from analysis.waxs.config import config_for

    return config_for(proposal, run_no, detector, **overrides).output_file


def frames_used(proposal: int, run_no: int, detector: str, **overrides: Any) -> Any:
    """The ``Cell`` of a ``*_frames_used`` variable: OK frames over all rows."""
    return frames_used_cell(output_file(proposal, run_no, detector, **overrides))


def frames_used_cell(path: str | Path) -> Any:
    """``used / total (%)`` of a frame table, coloured by its status codes.

    - green: every frame OK;
    - red: any WORKER_ERROR or LABEL_MISMATCH, or a file HDF5 cannot read;
    - amber: anything else (NOT_PROCESSED, i.e. incomplete; MISSING_MODULES;
      NO_FRAMES; DATA_CHECK_FAILED), or a file another process holds open.

    When not green, the count is followed by the code that decided the colour
    (the commonest failure on red, the commonest non-OK code on amber).

    "busy" needs HDF5 file locking, and inside DAMNIT it is off: the context
    imports ``extra_speckle``, whose ``xcca.tools`` sets
    ``HDF5_USE_FILE_LOCKING=FALSE`` for the whole process. There a file being
    written reads as it stands (amber, NOT_PROCESSED rows), or red if HDF5 trips
    over it mid-write; refresh the cell once the pass has finished.

    :raises Skip: no file, i.e. the pass has not written this run.
    """
    from damnit.context import Cell, Skip

    path = Path(path)
    if not path.exists():
        raise Skip(f"no frame table at {path}")
    try:
        with h5py.File(path, "r") as f:
            status = f["frames/status"][:]
    except OSError as error:
        text = str(error)
        if any(word in text.lower() for word in _LOCKED):
            return Cell(None, summary_value="busy", background=COLORS["warning"])
        return Cell(text, summary_value="unreadable", background=COLORS["error"])

    total = int(status.size)
    used = int((status == FrameStatus.OK).sum())
    pct = 100.0 * used / total if total else 0.0
    summary = f"{used} / {total} ({pct:.1f}%)"
    if used == total and total:
        return Cell(used, summary_value=summary, background=COLORS["good"])
    failed = np.isin(status, np.asarray(FAILED, dtype=status.dtype))
    shown = status[failed] if failed.any() else status[status != FrameStatus.OK]
    if shown.size:
        codes, counts = np.unique(shown, return_counts=True)
        summary += f" · {_name(codes[counts.argmax()])} {int(counts.max())}"
    return Cell(
        used,
        summary_value=summary,
        background=COLORS["error"] if failed.any() else COLORS["warning"],
    )


def _name(code) -> str:
    try:
        return FrameStatus(int(code)).name
    except ValueError:
        return f"code {int(code)}"


#: Threads preparing video frames in the DAMNIT job (explicit, pitfall 10).
VIDEO_WORKERS = 16


def droplet_video(run, run_no: int, *, root: str | Path | None = None,
                  workers: int = VIDEO_WORKERS) -> Any:  # fmt: skip
    """The ``droplet_video`` cell: write the run's CAM5 video and check it.

    The video is decoded again after writing; a frame count that differs from
    the number of trains with an image raises.

    :param run: DAMNIT's raw run (the camera is a raw source).
    :returns: a ``Cell`` with the path and "N frames, G gaps".
    :raises Skip: the run has no image of the camera.
    """
    from damnit.context import Cell, Skip

    from analysis import camera

    facts = camera.camera_facts(run)
    if not facts["n_images"]:
        raise Skip(f"no images of {camera.CAMERA_SOURCE}")
    path = camera.video_path(run_no, root=root or camera.VIDEO_ROOT)
    info = camera.write_camera_video(run, path, label=f"r{run_no:04d}", workers=workers)
    n = camera.count_frames(path)
    if n != info["frames"]:
        raise RuntimeError(f"{path} decodes to {n} frames, {info['frames']} written")
    return Cell(str(path), summary_value=f"{n} frames, {info['gaps']} gaps")


def droplet_first_image(run) -> Any:
    """The run's first CAM5 image, reading only that image.

    :returns: a (y, x) DataArray with its ``trainId`` and ``source`` in attrs,
        which DAMNIT shows as a thumbnail.
    :raises Skip: the run has no image of the camera.
    """
    import xarray as xr
    from damnit.context import Skip

    from analysis import camera

    if camera.CAMERA_SOURCE not in run.all_sources:
        raise Skip(f"no {camera.CAMERA_SOURCE} in this run")
    try:
        train_id, image = camera.first_image(run)
    except ValueError as error:
        raise Skip(str(error)) from None
    return xr.DataArray(
        image,
        dims=("y", "x"),
        attrs={"trainId": train_id, "source": camera.CAMERA_SOURCE},
    )
