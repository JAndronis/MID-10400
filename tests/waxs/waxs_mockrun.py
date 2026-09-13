"""Build a proc-like JUNGFRAU run on disk (WAXS context file §7, phase W3).

EXtra-data's own ``JUNGFRAUModule`` already writes the right dtypes and shapes —
``data/adc`` float32 ``(entries, 16, 512, 1024)``, ``data/mask`` uint32,
``data/memoryCell`` uint8 — so unlike the AGIPD mock nothing has to be recast.
What it does not write is *content*: the datasets are created unallocated, so
each file is reopened and filled with lit and dark cells, a static-looking mask
pattern per cell and a few per-frame bits.

The awkward-train cases are produced the way EXtra-data actually decides them:

* **zero-entry train** — ``INDEX/.../data/count`` zeroed for that train, which
  is what takes it out of ``JUNGFRAU.frame_counts`` entirely;
* **dropped train** — ``INDEX/trainId`` and the per-entry ``trainId`` are
  renumbered to a non-contiguous sequence, so the run itself has a gap.

There is no missing-module case: each of this experiment's detectors is a single
JUNGFRAU-500K module with its own PONI and mask, and ``plan.open_detector``
refuses a multi-module selection outright, so a train can only have its one
module or none.

This leans on ``extra_data.tests.mockdata``, an internal test package of
EXtra-data with no API stability promise — the same dependency the AGIPD mock
carries, and the same thing to check on an EXtra-data upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from extra_data.tests.mockdata import write_file
from extra_data.tests.mockdata.jungfrau import JUNGFRAUModule

MODULE_SHAPE = (512, 1024)
N_MODULE_PX = MODULE_SHAPE[0] * MODULE_SHAPE[1]
CELLS = 16
#: The real pattern, measured on the cluster: 0-6 and 15, which is deliberately
#: **not** contiguous. A mock lighting 0-7 would never exercise the code paths
#: that care - the ROI window, and any assumption that the lit set is a range.
LIT_CELLS = (0, 1, 2, 3, 4, 5, 6, 15)
STATIC_BIT = np.uint32(1 << 0)  # OFFSET_OUT_OF_THRESHOLD
DYNAMIC_BIT = np.uint32(1 << 21)  # WRONG_GAIN_VALUE
SEAM_BIT = np.uint32(1 << 22)  # NON_STANDARD_SIZE — set here, unlike AGIPD

#: Enough photons for the lit-cell fraction to clear ``lit_fraction_min`` by a
#: wide margin, as the real data does (42–53 % against ≤ 0.08 %).
LIT_RATE = 0.6
PHOTON_KEV = 9.04
READ_NOISE_KEV = 0.32


@dataclass(frozen=True)
class MockRun:
    """A written run and the facts a test asserts against it."""

    path: Path
    train_ids: tuple[int, ...]
    detector_name: str
    module: int = 1
    lit_cells: tuple[int, ...] = LIT_CELLS
    zero_entry_trains: tuple[int, ...] = ()
    static_mask: np.ndarray | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def detector_trains(self) -> tuple[int, ...]:
        """Trains that own rows: the module wrote an entry for them."""
        return tuple(t for t in self.train_ids if t not in self.zero_entry_trains)

    @property
    def rows_per_train(self) -> int:
        return len(self.lit_cells)

    @property
    def n_frames(self) -> int:
        return len(self.detector_trains) * self.rows_per_train

    def first_row(self, train_id: int) -> int:
        row = 0
        for tid in self.detector_trains:
            if tid == train_id:
                return row
            row += self.rows_per_train
        raise KeyError(train_id)


def device_id(detector_name: str, module: int = 1) -> str:
    """Karabo device id. ``write_instrument`` appends ``:daqOutput`` itself.

    Corrected data has used ``/CORR/`` in its source names since 2026/1, which
    is what ``lsxfel`` reports for r0423:
    ``MID_EXP_JF500K1/CORR/JNGFR01:daqOutput``. ``JUNGFRAU._source_corr_pat``
    matches only that form, so a mock written with the old ``/DET/`` names would
    silently exercise the raw-name fallback path instead of the one the real
    files take.
    """
    return f"{detector_name}/CORR/JNGFR{module:02d}"


def legacy_device_id(detector_name: str, module: int = 1) -> str:
    """The pre-2026/1 ``/DET/`` name, kept in the files as a soft link."""
    return f"{detector_name}/DET/JNGFR{module:02d}"


def source(detector_name: str, module: int = 1) -> str:
    return f"{device_id(detector_name, module)}:daqOutput"


def legacy_source(detector_name: str, module: int = 1) -> str:
    return f"{legacy_device_id(detector_name, module)}:daqOutput"


def write_mock_run(
    root: Path,
    *,
    detector_name: str = "MID_EXP_JF500K1",
    module: int = 1,
    legacy_alias: bool = True,
    train_ids: tuple[int, ...] = (10000, 10001, 10002, 10003),
    lit_cells: tuple[int, ...] = LIT_CELLS,
    zero_entry_trains: tuple[int, ...] = (),
    static_fraction: float = 0.4,
    nonfinite_train: int | None = None,
    extreme_train: int | None = None,
    shuffled_cells_train: int | None = None,
    seed: int = 0,
) -> MockRun:
    """Write a one-module proc-like JUNGFRAU run under ``root``.

    :param train_ids: the run's train ids. Pass a non-contiguous sequence to
        simulate a train the DAQ dropped.
    :param zero_entry_trains: trains the module wrote no entry for, which the
        plan records as ``NO_FRAMES``.
    :param static_fraction: fraction of pixels the returned static mask
        excludes, mimicking the real ``.edf`` files' 75.9 % / 84.4 %.
    :param nonfinite_train: put a NaN on a kept pixel of this train's first lit
        cell, to drive ``DATA_CHECK_FAILED``.
    :param extreme_train: put a 1.8e5 keV pixel — the magnitude jf2 actually
        carries, unflagged — on a kept pixel of this train's first lit cell.
    :param shuffled_cells_train: give this train a repeated ``memoryCell``
        entry, so its rows cannot be filled by label.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train_ids = tuple(int(t) for t in train_ids)
    rng = np.random.default_rng(seed)

    static = np.zeros(N_MODULE_PX, dtype=bool)
    static[
        rng.choice(N_MODULE_PX, int(static_fraction * N_MODULE_PX), replace=False)
    ] = True
    static = static.reshape(MODULE_SHAPE)

    path = root / f"CORR-R0423-JNGFR{module:02d}-S00000.h5"
    write_file(
        str(path),
        [JUNGFRAUModule(device_id(detector_name, module), raw=False)],
        ntrains=len(train_ids),
        firsttrain=train_ids[0],
        chunksize=1,
    )
    _renumber_trains(path, detector_name, train_ids, module)
    if zero_entry_trains:
        _zero_entry_counts(path, detector_name, train_ids, zero_entry_trains, module)
    if legacy_alias:
        _add_legacy_alias(path, detector_name, module)
    _fill_frames(
        path,
        detector_name,
        module,
        train_ids,
        lit_cells=lit_cells,
        static=static,
        seed=seed,
        nonfinite_train=nonfinite_train,
        extreme_train=extreme_train,
        shuffled_cells_train=shuffled_cells_train,
    )

    return MockRun(
        path=root,
        train_ids=train_ids,
        detector_name=detector_name,
        module=module,
        lit_cells=tuple(lit_cells),
        zero_entry_trains=tuple(zero_entry_trains),
        static_mask=static,
    )


def write_synthetic_poni(
    path, *, photon_energy_kev: float = PHOTON_KEV, orientation: int = 3
):
    """A PONI on a real ``Jungfrau`` detector, centred off one corner."""
    from pyFAI.detectors import Jungfrau
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    from pyFAI.units import hc

    detector = Jungfrau(orientation=orientation)
    ai = AzimuthalIntegrator(
        detector=detector,
        dist=0.232,
        poni1=MODULE_SHAPE[0] // 2 * 75e-6,
        poni2=MODULE_SHAPE[1] // 2 * 75e-6,
        wavelength=hc / photon_energy_kev * 1e-10,
    )
    ai.save(str(path))
    return path


def write_static_mask(path: Path, mask: np.ndarray) -> Path:
    """Write a native pyFAI ``.edf`` static mask, non-zero = excluded."""
    import fabio

    fabio.edfimage.EdfImage(data=mask.astype(np.uint8)).write(str(path))
    return path


def _renumber_trains(
    path: Path, detector_name: str, train_ids: tuple[int, ...], module: int = 1
) -> None:
    """Replace the writer's contiguous train range with ``train_ids``."""
    wanted = np.array(train_ids, dtype=np.uint64)
    contiguous = np.arange(train_ids[0], train_ids[0] + len(train_ids), dtype=np.uint64)
    if np.array_equal(wanted, contiguous):
        return
    with h5py.File(path, "r+") as handle:
        handle["INDEX/trainId"][: wanted.size] = wanted
        handle[f"INSTRUMENT/{source(detector_name, module)}/data/trainId"][:] = wanted


def _add_legacy_alias(path: Path, detector_name: str, module: int = 1) -> None:
    """Add the pre-2026/1 ``/DET/`` source name as a soft link, as proc does.

    ``lsxfel`` reports one of these per JUNGFRAU file in r0423. EXtra-data
    detects them by looking for an ``h5py.SoftLink`` under ``INSTRUMENT/`` whose
    id is listed in ``METADATA/dataSourceId`` (``file_access._read_data_sources``),
    and ``DataCollection.instrument_sources`` **includes** them — only
    ``detector_sources`` subtracts them. So the alias really is visible to
    ``MultimodDetectorBase._source_matches``, and the reason it does not become
    a second module is that ``_source_corr_pat`` matches ``/CORR/`` alone.
    That is worth a mock rather than an argument, which is why this exists.
    """
    canonical = source(detector_name, module)
    legacy = legacy_source(detector_name, module)
    with h5py.File(path, "r+") as handle:
        handle[f"INSTRUMENT/{legacy}"] = h5py.SoftLink(f"/INSTRUMENT/{canonical}")
        handle[f"INDEX/{legacy}"] = h5py.SoftLink(f"/INDEX/{canonical}")

        group = handle["METADATA"]
        entry = f"INSTRUMENT/{legacy}/data".encode()
        ids = group["dataSourceId"]
        free = [i for i, value in enumerate(ids[:]) if not value]
        if free:
            index = free[0]
        else:
            index = ids.shape[0]
            for name in ("dataSourceId", "root", "deviceId"):
                group[name].resize((index + 1,))
        ids[index] = entry
        group["root"][index] = b"INSTRUMENT"
        group["deviceId"][index] = f"{legacy}/data".encode()


def _zero_entry_counts(
    path: Path,
    detector_name: str,
    train_ids: tuple[int, ...],
    targets: tuple[int, ...],
    module: int = 1,
) -> None:
    """Make the module report no entry for ``targets``."""
    with h5py.File(path, "r+") as handle:
        index = handle[f"INDEX/{source(detector_name, module)}/data"]
        counts = index["count"][:]
        for target in targets:
            counts[train_ids.index(target)] = 0
        index["count"][:] = counts


def _fill_frames(
    path: Path,
    detector_name: str,
    module: int,
    train_ids: tuple[int, ...],
    *,
    lit_cells: tuple[int, ...],
    static: np.ndarray,
    seed: int,
    nonfinite_train: int | None,
    extreme_train: int | None,
    shuffled_cells_train: int | None,
) -> None:
    """Fill ``data/adc``, ``data/mask`` and ``data/memoryCell`` with content.

    Lit cells get Poisson photons at ``PHOTON_KEV`` plus readout noise; dark
    cells get the noise alone, so the dark-cell spread a run measures is
    ``READ_NOISE_KEV`` and the lit fraction is far above any sane threshold —
    the same separation the real detectors show.
    """
    keep = ~static.reshape(-1)
    first_kept = int(np.flatnonzero(keep)[0])

    with h5py.File(path, "r+") as handle:
        group = handle[f"INSTRUMENT/{source(detector_name, module)}/data"]
        adc, mask, memory = group["adc"], group["mask"], group["memoryCell"]
        n_entries = adc.shape[0]

        for entry in range(n_entries):
            rng = np.random.default_rng(seed * 1000 + entry)
            values = rng.normal(0.0, READ_NOISE_KEV, (CELLS, *MODULE_SHAPE))
            for cell in lit_cells:
                values[cell] += PHOTON_KEV * rng.poisson(LIT_RATE, MODULE_SHAPE)
            frame_mask = np.zeros((CELLS, *MODULE_SHAPE), dtype=np.uint32)
            flat_mask = frame_mask.reshape(CELLS, -1)
            for cell in range(CELLS):
                # per memory cell: the same pixels every time the cell appears
                cell_rng = np.random.default_rng(7000 + cell)
                flat_mask[cell, cell_rng.choice(N_MODULE_PX, 400, replace=False)] |= (
                    STATIC_BIT
                )
                flat_mask[cell, cell_rng.choice(N_MODULE_PX, 200, replace=False)] |= (
                    SEAM_BIT
                )
                # per frame: a handful, as bit 21 behaves in the real data
                flat_mask[cell, rng.choice(N_MODULE_PX, 5, replace=False)] |= (
                    DYNAMIC_BIT
                )

            train_id = train_ids[entry] if entry < len(train_ids) else None
            flat_values = values.reshape(CELLS, -1)
            if train_id is not None and train_id == nonfinite_train:
                flat_values[lit_cells[0], first_kept] = np.nan
            if train_id is not None and train_id == extreme_train:
                flat_values[lit_cells[0], first_kept] = 1.8e5

            cells = np.arange(CELLS, dtype=np.uint8)
            if train_id is not None and train_id == shuffled_cells_train:
                cells[1] = cells[0]

            adc[entry] = values.astype(np.float32)
            mask[entry] = frame_mask
            memory[entry] = cells
