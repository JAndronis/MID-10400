"""What a pass's frame table holds: status accounting, per-frame scalars, units.

Works on both passes' output files (``agipd_saxs.h5``, ``jungfrau_waxs_jf{1,2}.h5``)
through the detector-agnostic layout of `analysis.common.writer`: ``/frames`` (one
row per frame), ``/trains`` (the plan's row map) and ``/provenance``.

Labels are trusted only where the writer stored them. A block marked by
``FrameTableWriter.mark`` (a worker error, or rows never processed) gets a status
and nothing else, so its ``trainId`` and ``cellId`` are the zero fill value; such
rows are counted as ``unlabelled``, never under cell 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.common.status import FrameStatus
from analysis.common.writer import as_handle, pooled_per_train

__all__ = [
    "DEFAULT_UNITS",
    "default_scalar_column",
    "detector_kind",
    "frame_accounting",
    "frame_scalar",
    "per_train",
    "units",
]

#: What ``signal``/``variance`` hold in a file that records no ``units`` attribute:
#: AGIPD sums integer photon counts; JUNGFRAU files written before photonising
#: (CLAUDE.md open task 18) sum keV. S2 writes ``units`` to ``/provenance`` from then
#: on, so a recorded value always wins.
DEFAULT_UNITS = {"agipd": "photons", "jungfrau": "keV"}
#: The per-frame total of each detector's valid pixels, by preference. JUNGFRAU's
#: ``energy_valid`` stays raw keV even in a photonised file; ``n_photons`` is its
#: photon equivalent where it exists (S2, 2026-10-13).
SCALAR_COLUMNS = {
    "agipd": ("photons_valid",),
    "jungfrau": ("n_photons", "energy_valid"),
}
#: The second frame label of each detector and the name its axis gets.
LABELS = {"agipd": ("reader_pulseId", "pulseId"), "jungfrau": ("cellId", "cellId")}


def detector_kind(handle) -> str:
    """``"agipd"`` or ``"jungfrau"``, from the file's own schema."""
    if "frames/reader_pulseId" in handle:
        return "agipd"
    if "cells" in handle:
        return "jungfrau"
    raise ValueError(
        f"{handle.filename} is neither an AGIPD nor a JUNGFRAU frame table"
    )


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _units(handle) -> tuple[str, bool]:
    for attrs in (handle["provenance"].attrs, handle.attrs):
        if "units" in attrs:
            return _text(attrs["units"]), True
    return DEFAULT_UNITS[detector_kind(handle)], False


def units(source) -> str:
    """The unit of ``signal``/``variance``: the stored attribute, else the default.

    The attribute is looked for on ``/provenance``, then on the root. Without one,
    the detector's `DEFAULT_UNITS` applies; `per_train` records which it was.
    """
    with as_handle(source) as handle:
        return _units(handle)[0]


def default_scalar_column(source) -> str:
    """The per-frame valid-pixel total to correlate with (`SCALAR_COLUMNS`)."""
    with as_handle(source) as handle:
        for column in SCALAR_COLUMNS[detector_kind(handle)]:
            if f"frames/{column}" in handle:
                return column
    raise KeyError(f"{source}: none of {SCALAR_COLUMNS} in /frames")


def frame_accounting(source, by: str | None = "auto"):
    """Frames per status code, overall and per label.

    :param by: a ``/frames`` label column to break the counts down by; ``"auto"``
        is ``cellId`` for JUNGFRAU and nothing for AGIPD (155 pulse columns are a
        table nobody reads; pass ``"reader_pulseId"`` to get it).
    :returns: a DataFrame indexed by status name, ``OK`` always first and every
        other code only where it occurs. Column ``frames`` is the total; one column
        per label value follows, and ``unlabelled`` when some rows carry no label.
        ``attrs`` hold the detector, the totals, and the plan's train statuses —
        a train the plan gave no rows (``NO_FRAMES``) is invisible in the frame
        counts and appears only there.
    """
    import pandas as pd

    with as_handle(source) as handle:
        kind = detector_kind(handle)
        frames = handle["frames"]
        status = frames["status"][:]
        column = LABELS[kind][0] if by == "auto" and kind == "jungfrau" else by
        column = None if column == "auto" else column
        if column is not None:
            labels = frames[column][:]
            labelled = frames["trainId"][:] != 0
        plan_status = handle["trains/status"][:]
        n_trains = int(plan_status.size)

    codes = [
        code for code in FrameStatus if code is FrameStatus.OK or (status == code).any()
    ]
    table = pd.DataFrame(
        {"frames": [int((status == code).sum()) for code in codes]},
        index=pd.Index([code.name for code in codes], name="status"),
    )
    if column is not None:
        for value in np.unique(labels[labelled]):
            here = labelled & (labels == value)
            table[int(value)] = [int((status[here] == code).sum()) for code in codes]
        if (~labelled).any():
            table["unlabelled"] = [
                int((status[~labelled] == code).sum()) for code in codes
            ]
    table.attrs.update(
        detector=kind,
        label=column,
        n_frames=int(status.size),
        n_trains=n_trains,
        trains={
            FrameStatus(int(code)).name: int((plan_status == code).sum())
            for code in np.unique(plan_status)
        },
    )
    return table


def frame_scalar(source, column: str | None = None):
    """A per-frame column on a ``(trainId, label)`` grid, ``OK`` frames only.

    Each frame is placed by its stored trainId and label (CLAUDE.md pitfall 4);
    a slot without an ``OK`` frame is NaN. The grid is an in-memory view, not a
    stored product, so NaN marks absence here.

    :param column: a ``/frames`` column; default `default_scalar_column`.
    :raises ValueError: two ``OK`` frames share a (train, label) slot.
    """
    import xarray as xr

    with as_handle(source) as handle:
        kind = detector_kind(handle)
        column = column or default_scalar_column(handle)
        label_column, label_dim = LABELS[kind]
        frames = handle["frames"]
        ok = frames["status"][:] == FrameStatus.OK
        trains = frames["trainId"][:][ok]
        labels = frames[label_column][:][ok]
        values = frames[column][:][ok].astype(np.float64)
        unit, _ = _units(handle)

    train_ids, ti = np.unique(trains, return_inverse=True)
    label_ids, li = np.unique(labels, return_inverse=True)
    if np.unique(ti * label_ids.size + li).size != ti.size:
        raise ValueError(f"two OK frames share a (train, {label_dim}) slot")
    grid = np.full((train_ids.size, label_ids.size), np.nan)
    grid[ti, li] = values
    return xr.DataArray(
        grid,
        dims=("trainId", label_dim),
        coords={"trainId": train_ids, label_dim: label_ids},
        name=column,
        attrs={"column": column, "detector": kind, "signal_units": unit},
    )


def per_train(source):
    """`analysis.common.writer.pooled_per_train`, with the units on the result.

    ``attrs``: ``units`` of ``intensity``/``sigma`` (per pixel or solid angle, per
    frame), ``units_recorded`` (False when `DEFAULT_UNITS` supplied it) and
    ``detector``.
    """
    with as_handle(source) as handle:
        kind = detector_kind(handle)
        unit, recorded = _units(handle)
        result = pooled_per_train(handle)
    result.attrs.update(units=unit, units_recorded=recorded, detector=kind)
    return result
