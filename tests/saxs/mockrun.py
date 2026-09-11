"""Build a proc-like AGIPD run on disk (context file §10, phase P3).

EXtra-data's own mock writer produces ``image/data`` as float32 and does not
chunk it the way corrected AGIPD files are chunked, so each module file is
reopened afterwards and the two image datasets are replaced with int16 photon
counts and a uint32 ``BadPixels`` field, both chunked ``(1, 512, 128)`` with
shuffle + gzip — the layout CLAUDE.md records for r0423 and r0426.

The three awkward-train cases are produced the way EXtra-data actually decides
them (``components.py`` ``_select_trains``):

* **zero-frame train** — ``frames_per_train`` accepts an array, so a train can
  simply be written with zero frames;
* **train with < 16 modules** — module presence is ``(counts > 0)``, so zeroing
  one module's ``INDEX/.../image/count`` for a train drops it below
  ``min_modules``;
* **dropped train** — ``INDEX/trainId`` and ``INSTRUMENT/.../image/trainId`` are
  renumbered to a non-contiguous sequence, so the run itself has a gap.

This leans on ``extra_data.tests.mockdata``, an internal test package of
EXtra-data with no API stability promise. It is what the context file
prescribes, but it is a thing to check when EXtra-data is upgraded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from extra_data.tests.mockdata import write_file
from extra_data.tests.mockdata.detectors import AGIPDModule

DETECTOR = "MID_DET_AGIPD1M-1"
MODULE_SHAPE = (512, 128)
N_MODULE_PX = MODULE_SHAPE[0] * MODULE_SHAPE[1]
STATIC_BIT = np.uint32(1 << 0)  # OFFSET_OUT_OF_THRESHOLD
DYNAMIC_BIT = np.uint32(1 << 12)  # VALUE_OUT_OF_RANGE


@dataclass(frozen=True)
class MockRun:
    """A written run and the facts a test asserts against it."""

    path: Path
    train_ids: tuple[int, ...]
    frames_per_train: tuple[int, ...]
    n_modules: int
    short_module_trains: tuple[int, ...] = ()
    zero_frame_trains: tuple[int, ...] = ()
    frames_by_train: dict[int, int] = field(default_factory=dict)

    @property
    def detector_trains(self) -> tuple[int, ...]:
        """Trains that own rows: enough modules *and* at least one frame."""
        return tuple(
            t
            for t in self.train_ids
            if t not in self.short_module_trains and self.frames_by_train[t]
        )

    @property
    def n_frames(self) -> int:
        return sum(self.frames_by_train[t] for t in self.detector_trains)

    def first_row(self, train_id: int) -> int:
        row = 0
        for tid in self.detector_trains:
            if tid == train_id:
                return row
            row += self.frames_by_train[tid]
        raise KeyError(train_id)


def device_id(module: int) -> str:
    """Karabo device id. ``write_instrument`` appends ``:xtdf`` itself."""
    return f"{DETECTOR}/DET/{module}CH0"


def source(module: int) -> str:
    """The INSTRUMENT source name, device id plus output channel."""
    return f"{device_id(module)}:xtdf"


def write_mock_run(
    root: Path,
    *,
    train_ids: tuple[int, ...] = (10000, 10001, 10002, 10003, 10004, 10005),
    frames_per_train: int | tuple[int, ...] = 3,
    n_modules: int = 16,
    occupancy: float = 0.02,
    short_module_trains: tuple[int, ...] = (),
    zero_frame_trains: tuple[int, ...] = (),
    float_data_module: int | None = None,
    negative_counts_module: int | None = None,
) -> MockRun:
    """Write an ``n_modules``-module proc-like run under ``root``.

    :param train_ids: the run's train ids. Pass a non-contiguous sequence to
        simulate a train the DAQ dropped.
    :param short_module_trains: trains that module 0 will report zero frames
        for, so the detector selection drops them for ``min_modules``.
    :param zero_frame_trains: trains written with no frames in any module.
    :param float_data_module: give this module float ``image/data``, to drive
        ``DATA_CHECK_FAILED``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train_ids = tuple(int(t) for t in train_ids)

    if isinstance(frames_per_train, int):
        counts = [
            0 if tid in zero_frame_trains else frames_per_train for tid in train_ids
        ]
    else:
        counts = list(frames_per_train)
    frames_by_train = dict(zip(train_ids, counts, strict=True))

    for module in range(n_modules):
        path = root / f"CORR-R0423-AGIPD{module:02d}-S00000.h5"
        write_file(
            str(path),
            [
                AGIPDModule(
                    device_id(module),
                    frames_per_train=np.array(counts, dtype=np.uint64),
                    raw=False,
                )
            ],
            ntrains=len(train_ids),
            firsttrain=train_ids[0],
            chunksize=32,
        )
        _renumber_trains(path, module, train_ids, counts)
        if module == 0 and short_module_trains:
            _zero_module_counts(path, module, train_ids, short_module_trains)
        _rewrite_images(
            path,
            module,
            occupancy=occupancy,
            seed=module,
            float_data=(module == float_data_module),
            negative_counts=(module == negative_counts_module),
        )

    return MockRun(
        path=root,
        train_ids=train_ids,
        frames_per_train=tuple(counts),
        n_modules=n_modules,
        short_module_trains=tuple(short_module_trains),
        zero_frame_trains=tuple(zero_frame_trains),
        frames_by_train=frames_by_train,
    )


def _renumber_trains(
    path: Path, module: int, train_ids: tuple[int, ...], counts: list[int]
) -> None:
    """Replace the writer's contiguous train range with ``train_ids``."""
    wanted = np.array(train_ids, dtype=np.uint64)
    if np.array_equal(
        wanted, np.arange(train_ids[0], train_ids[0] + len(train_ids), dtype=np.uint64)
    ):
        return
    with h5py.File(path, "r+") as handle:
        handle["INDEX/trainId"][: wanted.size] = wanted
        per_frame = np.repeat(wanted, np.asarray(counts, dtype=np.intp))
        handle[f"INSTRUMENT/{source(module)}/image/trainId"][:] = per_frame


def _zero_module_counts(
    path: Path, module: int, train_ids: tuple[int, ...], targets: tuple[int, ...]
) -> None:
    """Make this module report no frames for ``targets``.

    Module presence is ``count > 0``, so this is what takes a train below
    ``min_modules`` without disturbing any other train's rows.
    """
    with h5py.File(path, "r+") as handle:
        index = handle[f"INDEX/{source(module)}/image"]
        counts = index["count"][:]
        for target in targets:
            counts[train_ids.index(target)] = 0
        index["count"][:] = counts


def _rewrite_images(
    path: Path,
    module: int,
    *,
    occupancy: float,
    seed: int,
    float_data: bool = False,
    negative_counts: bool = False,
) -> None:
    """Replace ``image/data`` and ``image/mask`` with proc-like content."""
    with h5py.File(path, "r+") as handle:
        group = handle[f"INSTRUMENT/{source(module)}/image"]
        n_frames = group["data"].shape[0]
        cell_ids = np.asarray(group["cellId"][:]).reshape(-1)
        for name in ("data", "mask", "gain"):
            if name in group:
                del group[name]

        rng = np.random.default_rng(seed)
        dtype = np.float32 if float_data else np.int16
        data = np.zeros((n_frames, *MODULE_SHAPE), dtype=dtype)
        mask = np.zeros((n_frames, *MODULE_SHAPE), dtype=np.uint32)
        flat_data = data.reshape(n_frames, -1)
        flat_mask = mask.reshape(n_frames, -1)

        for frame in range(n_frames):
            hits = rng.choice(
                N_MODULE_PX, max(int(occupancy * N_MODULE_PX), 1), replace=False
            )
            flat_data[frame, hits] = rng.integers(1, 4, hits.size).astype(dtype)
            # per memory cell: the same pixels every time the cell appears
            cell_rng = np.random.default_rng(5000 + int(cell_ids[frame]) + 100 * module)
            flat_mask[frame, cell_rng.choice(N_MODULE_PX, 40, replace=False)] |= (
                STATIC_BIT
            )
            # per frame: a few pixels, as bits 12/13 behave in the real data
            flat_mask[frame, rng.choice(N_MODULE_PX, 5, replace=False)] |= DYNAMIC_BIT

        if negative_counts and n_frames:
            flat_data[0, 0] = -1

        for name, values in (("data", data), ("mask", mask)):
            group.create_dataset(
                name,
                data=values,
                chunks=(1, *MODULE_SHAPE),
                shuffle=True,
                compression="gzip",
                compression_opts=1,
            )
