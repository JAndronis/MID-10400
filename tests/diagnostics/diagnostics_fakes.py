"""Fakes for the diagnostics: frame tables in the passes' own layout, DAMNIT flux.

The frame table is laid out by the real writer class (``_create_layout``), so the
schema the diagnostics read is the one the passes write, columns added later
included. Rows are then filled directly, the way ``write_block`` (labels and data)
and ``mark`` (status only, labels left at the zero fill) leave them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from analysis.common.plan import RunPlan, TrainRecord
from analysis.common.status import FrameStatus

FIRST_TRAIN = 2637695397
#: What `FrameTableWriter.mark` leaves: a status and no labels.
MARKED = (FrameStatus.WORKER_ERROR, FrameStatus.NOT_PROCESSED)


@dataclass(frozen=True)
class StubConfig:
    """The `PassConfig` surface `_create_layout` reads."""

    npt: int
    overwrite: bool = False

    @property
    def output_file(self) -> Path:
        return Path("unused.h5")

    @property
    def operational_fields(self) -> frozenset[str]:
        return frozenset()

    def config_hash(self) -> str:
        return "stub"


#: JUNGFRAU frame-table columns a keV-era file (before photon counting) lacks.
KEV_ERA_ABSENT = ("n_photons", "n_level_missing", "count_energy_kev", "asic_level")


def writer_class(kind: str):
    if kind == "agipd":
        from analysis.saxs.writer import AgipdSaxsWriter

        return AgipdSaxsWriter
    from analysis.waxs.writer import JungfrauWaxsWriter

    return JungfrauWaxsWriter


def write_frame_table(
    path: Path,
    kind: str,
    signal: np.ndarray,
    *,
    labels=None,
    normalization: np.ndarray | None = None,
    variance: np.ndarray | None = None,
    status: np.ndarray | None = None,
    scalar: np.ndarray | None = None,
    q: np.ndarray | None = None,
    units: str | None = None,
    units_on_root: bool = False,
    extra_columns: dict[str, np.ndarray] | None = None,
    empty_trains: int = 0,
    first_train: int = FIRST_TRAIN,
) -> Path:
    """One run's output file with ``signal`` of shape (trains, labels, npt).

    :param labels: the second label per slot, the same for every train.
    :param status: (trains, labels) codes; default all OK.
    :param scalar: (trains, labels) per-frame total; default ``signal.sum(-1)``.
    :param empty_trains: trains the plan gives no rows (``NO_FRAMES``), appended.
    """
    n_trains, n_labels, npt = signal.shape
    labels = np.arange(n_labels) if labels is None else np.asarray(labels)
    status = np.zeros((n_trains, n_labels), np.uint8) if status is None else status
    normalization = np.ones_like(signal) if normalization is None else normalization
    variance = signal.copy() if variance is None else variance
    scalar = signal.sum(axis=-1) if scalar is None else scalar
    tids = first_train + np.arange(n_trains + empty_trains, dtype=np.uint64)

    records = [
        TrainRecord(int(t), n_labels, i * n_labels, FrameStatus.OK)
        for i, t in enumerate(tids[:n_trains])
    ] + [
        TrainRecord(int(t), 0, n_trains * n_labels, FrameStatus.NO_FRAMES)
        for t in tids[n_trains:]
    ]
    plan = RunPlan(tuple(records), (), n_trains * n_labels, f"FAKE_{kind.upper()}")
    cls = writer_class(kind)
    with h5py.File(path, "w") as handle:
        cls._create_layout(handle, StubConfig(npt), plan, "stub")
        frames = handle["frames"]
        if kind == "jungfrau" and units is None:
            # ``units=None`` means a keV file written without photon counting: the
            # current writer stamps ``units = "photons"`` and lays out the
            # counting columns, which such a file never had.
            handle["provenance"].attrs.pop("units", None)
            for name in KEV_ERA_ABSENT:
                if name in frames:
                    del frames[name]
        flat_status = status.reshape(-1)
        marked = np.isin(flat_status, [int(c) for c in MARKED])
        train_col = np.repeat(tids[:n_trains], n_labels)
        label_col = np.tile(labels, n_trains)
        train_col[marked], label_col[marked] = 0, 0
        frames["trainId"][:] = train_col
        frames["reader_pulseId" if kind == "agipd" else "cellId"][:] = label_col
        frames["status"][:] = flat_status
        frames["signal"][:] = signal.reshape(-1, npt)
        frames["normalization"][:] = normalization.reshape(-1, npt)
        frames["variance"][:] = variance.reshape(-1, npt)
        column = "photons_valid" if kind == "agipd" else "energy_valid"
        frames[column][:] = scalar.reshape(-1)
        for name, values in (extra_columns or {}).items():
            # a column the writer lays out itself (n_photons) is filled, any
            # other is created
            if name in frames:
                frames[name][:] = np.asarray(values).reshape(-1)
            else:
                frames.create_dataset(name, data=np.asarray(values).reshape(-1))
        centers = handle.create_group("q").create_dataset(
            "centers", data=np.linspace(0.1, 1.0, npt) if q is None else q
        )
        centers.attrs["unit"] = "nm^-1"
        if units is not None:
            (handle if units_on_root else handle["provenance"]).attrs["units"] = units
    return path


def write_damnit_flux(path: Path, xgm: dict[int, float], trans: dict[int, float]):
    """DAMNIT's ``xgm_intensity`` and ``total_transmission`` groups, as on p10400."""
    import xarray as xr

    for group, name, values in (
        ("xgm_intensity", "Energy", xgm),
        ("total_transmission", "__xarray_dataarray_variable__", trans),
    ):
        tids = np.array(sorted(values), dtype=np.uint64)
        da = xr.DataArray(
            np.array([values[int(t)] for t in tids]),
            dims="trainId",
            coords={"trainId": tids},
            name=name,
        )
        mode = "a" if Path(path).exists() else "w"
        da.to_dataset().to_netcdf(path, group=group, mode=mode, engine="h5netcdf")
    return path
