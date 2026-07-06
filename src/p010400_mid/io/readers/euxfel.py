"""EuXFEL MID raw-data reader plugin (proposal 10400).

Tier-1 reader for the ferritin-crystallization beamtime at the MID instrument of
the European XFEL. :class:`EuXFELMIDRawReader` turns a run *directory* into a
lazy, dask-backed :class:`xarray.Dataset` via EXtra-data (per-module assembly of
the AGIPD-1M is left on demand to EXtra-geom, outside the reader).

This module is written to drop into pyBeamtime as ``io/readers/euxfel.py``: it
imports pyBeamtime by absolute path and self-registers with ``ReaderRegistry``
at import time. Upstreaming is a file move plus a ``from . import euxfel`` line
in ``pyBeamtime/io/readers/__init__.py`` — no code change to this file.

Source and key names are verified against run r0500 of p010400 (see the
module-level constants; CLAUDE.md 5.3). EXtra-data / EXtra-geom are imported
lazily inside :meth:`EuXFELMIDRawReader.load_run` so the pure path helpers,
``can_read``, and ``list_runs`` import and unit-test without them installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import xarray as xr
from pyBeamtime.core.run import RunMetadata
from pyBeamtime.io.elog import load_elog_csv
from pyBeamtime.io.readers import ReaderRegistry
from pyBeamtime.io.readers.base import BaseRawReader

# ── Verified sources / keys (run r0500, proposal p010400) ──────────────────────
# Confirmed via `lsxfel` + `run[source].keys()` on r0500 (CLAUDE.md 5.3). Recorded
# as named constants so a future run that renames a source is a one-line change.
AGIPD_DETECTOR_NAME = "MID_DET_AGIPD1M-1"  # 16 modules "{det}/DET/{0..15}CH0:xtdf"
JF500K1_SOURCE = "MID_EXP_JF500K1/DET/JNGFR01:daqOutput"
JF500K2_SOURCE = "MID_EXP_JF500K2/DET/JNGFR02:daqOutput"
XGM_SOURCE = "SA2_XTD1_XGM/XGM/DOOCS:output"
LITFRM_SOURCE = "MID_EXP_AGIPD1M1/REDU/LITFRM:output"
TIMESERVER_SOURCE = "MID_RR_UTC/TSYS/TIMESERVER:outputBunchPattern"
ATTENUATOR_SOURCE = "SA2_XTD1_ATT/MDL/MAIN"

AGIPD_IMAGE_KEY = "image.data"
AGIPD_CELLID_KEY = "image.cellId"  # for the (deferred) canonical cell_id coord
AGIPD_PULSEID_KEY = "image.pulseId"  # for the (deferred) canonical pulse_id coord
JF_ADC_KEY = "data.adc"
XGM_FLUX_KEY = "data.intensityTD"
LITFRM_PATTERN_KEY = "data.dataFramePattern"
LITFRM_XGM_PULSE_KEY = "data.xgmPulseId"  # XGM↔AGIPD pulse map (deferred use)
ATTENUATOR_TRANSMISSION_KEY = "actual.transmission"

# ── As-run experimental constants (CLAUDE.md §2) ───────────────────────────────
PHOTON_ENERGY_EV = 9040.0
DETECTOR_DISTANCE_M = 7.531

# ── Filesystem conventions ─────────────────────────────────────────────────────
# EuXFEL run directories are `r####` (zero-padded); each holds many per-module
# `RAW-*.h5` files. Verify `RAW-*.h5` against the real p010400 tree (CLAUDE.md 5.2).
_RUN_DIR_RE = re.compile(r"^r(\d+)$")
_RAW_SENTINEL_GLOB = "raw/r*/RAW-*.h5"

# Planck constant × speed of light in eV·Å (CODATA), for λ = hc / E.
_HC_EV_ANGSTROM = 12398.419843320026


# ── Module-level pure helpers (no EXtra-data; unit-tested) ──────────────────────
def parse_run_dir(name: str) -> int | None:
    """Parse a EuXFEL run-directory name into its run number.

    ``"r0042" -> 42``; anything not of the form ``r<digits>`` returns ``None``.
    """
    match = _RUN_DIR_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def proposal_number_from_root(root_path: Path) -> int:
    """Return the integer proposal number parsed from the root basename.

    ``.../p010400 -> 10400``. Raises :exc:`ValueError` if the basename is not of
    the form ``p<digits>``. Needed only for Dataset provenance and the
    EXtra-speckle ``Setup`` — not for path resolution.
    """
    name = Path(root_path).name
    match = re.fullmatch(r"p(\d+)", name)
    if match is None:
        raise ValueError(
            f"Root path basename {name!r} is not of the form 'p<number>' "
            f"(expected e.g. 'p010400'): {root_path}"
        )
    return int(match.group(1))


def list_run_dirs(raw_path: Path) -> list[tuple[int, Path]]:
    """Enumerate ``r####`` run directories under *raw_path*, sorted by run id.

    Filesystem only — opens no files. Non-matching entries and non-directories
    are ignored. Returns ``[]`` if *raw_path* is not a directory.
    """
    raw_path = Path(raw_path)
    if not raw_path.is_dir():
        return []
    runs: list[tuple[int, Path]] = []
    for entry in raw_path.iterdir():
        if not entry.is_dir():
            continue
        run_id = parse_run_dir(entry.name)
        if run_id is None:
            continue
        runs.append((run_id, entry))
    runs.sort(key=lambda pair: pair[0])
    return runs


def wavelength_angstrom_from_energy_ev(energy_ev: float) -> float:
    """Photon wavelength in ångström from energy in eV (λ = hc / E).

    A constant energy↔wavelength conversion for the ``wavelength_A`` provenance
    attribute — not detector geometry (which EXtra-geom owns).
    """
    return _HC_EV_ANGSTROM / energy_ev


def _isolate_source(array: xr.DataArray, prefix: str) -> xr.DataArray:
    """Prefix every dim/coord of *array* and drop its pandas indexes.

    EXtra-data hands back arrays with generic, colliding names:
    ``AGIPD1M.get_dask_array`` yields a ``train_pulse`` MultiIndex (levels
    ``trainId``/``pulseId``) plus ``dim_0``/``dim_1``/…, while
    ``get_dask_array(labelled=True)`` yields a plain ``trainId`` index plus its
    own ``dim_0``/…. Dropping these into one Dataset makes xarray try to *align*
    the shared names — and a ``trainId`` MultiIndex level cannot align with a
    plain ``trainId`` index, so construction raises ``AlignmentError`` (the
    repeated unindexed ``dim_0`` of differing sizes would clash next).

    Resetting the indexes (their values survive as plain coords) and giving
    every dim/coord a per-source prefix removes all shared names, so sources sit
    side by side with no cross-alignment and nothing is materialized. Collapsing
    to the canonical unified ``(train, pulse, module, ss, fs)`` schema (5.3) is
    the deferred next step (CLAUDE.md §8).
    """
    indexed = [dim for dim in array.dims if dim in array.indexes]
    if indexed:
        array = array.reset_index(indexed)
    renames = {name: f"{prefix}_{name}" for name in (*array.dims, *array.coords)}
    return array.rename(renames)


class EuXFELMIDRawReader(BaseRawReader):
    """Raw reader for the MID instrument at the European XFEL (proposal 10400).

    Directory layout under ``root_path``::

        raw/
            r0001/
                RAW-R0001-AGIPD00-S00000.h5
                RAW-R0001-JNGFR01-S00000.h5
                ...            # many per-module / per-source files
            r0002/
                ...

    A run is a *directory* of many per-module files (not a single master file),
    read via EXtra-data. Sample names come from ``root_path/elog.csv`` (see
    :mod:`pyBeamtime.io.elog`); runs absent from the elog get ``sample_name=None``.
    """

    slug = "mid"
    beamline = "MID"
    facility = "European XFEL"
    priority = 0
    paired_facility_reader = None

    _REQUIRED_DIRS = ("raw",)
    _SENTINEL_GLOB = _RAW_SENTINEL_GLOB

    @classmethod
    def can_read(cls, root_path: Path) -> bool:
        """True if *root_path* has a ``raw/`` dir containing ``r*/RAW-*.h5``.

        Two-layer sentinel: the ``raw`` directory must exist and at least one
        per-module ``RAW-*.h5`` file must live under an ``r*`` run directory.
        Used by ``beamtime init`` / ``validate`` and tests only (decision 008),
        never on the data-loading path.
        """
        root_path = Path(root_path)
        if not all(
            (root_path / directory).is_dir() for directory in cls._REQUIRED_DIRS
        ):
            return False
        return any(root_path.glob(cls._SENTINEL_GLOB))

    def list_runs(self, root_path: Path) -> list[RunMetadata]:
        """Return one :class:`RunMetadata` per ``r####`` run directory on disk.

        Filesystem only — no run is opened here. Fields that require reading the
        run with EXtra-data are left at their "unknown" values and are surfaced
        instead as :meth:`load_run` Dataset attributes:

        - ``exposure_time`` (per-pulse width): ``0.0`` placeholder — the field is
          non-optional (``float``); ``0.0`` matches the CoSAXS "unknown" convention.
        - ``n_frames`` (pulses per train): ``None``.
        - ``start_time`` / ``end_time``: ``None``.
        - ``extra["n_trains"]``: not set here.

        Lifting these into ``RunMetadata`` needs a per-run EXtra-data open with
        as-yet-unverified accessors (CLAUDE.md 5.2.1) and is deferred. Runs absent
        from ``elog.csv`` get ``sample_name=None`` (a first-class state, decision
        015); non-``sample`` elog columns pass through into ``extra``.
        """
        root_path = Path(root_path)
        elog = load_elog_csv(root_path)
        runs: list[RunMetadata] = []
        for run_id, _run_dir in list_run_dirs(root_path / "raw"):
            elog_row = elog.get(run_id, {})
            sample_name = elog_row.get("sample") or None
            extra = {key: value for key, value in elog_row.items() if key != "sample"}
            runs.append(
                RunMetadata(
                    scan_id=run_id,
                    exposure_time=0.0,
                    start_time=None,
                    end_time=None,
                    sample_name=sample_name,
                    n_frames=None,
                    extra=extra,
                )
            )
        return runs

    def get_run_path(self, run_id: int | str, root_path: Path) -> Path:
        """Return the run *directory* ``root_path/raw/r{run_id:04d}``.

        Deliberately deviates from the ABC's "primary HDF5 file" wording: a
        EuXFEL run is a directory of many per-module files, opened by EXtra-data's
        ``RunDirectory``.

        CONSEQUENCE: the generic ``Beamtime.enrich_metadata`` — which does
        ``h5py.File(get_run_path(...))`` — does NOT work for EuXFEL; it would try
        to open a directory. EuXFEL metadata enrichment must go through EXtra-data.
        This is a ratified deviation (CLAUDE.md 5.2), not an oversight; do not
        "fix" it by returning an arbitrary aggregator file.

        Raises :exc:`FileNotFoundError` if the run directory does not exist.
        """
        run_dir = Path(root_path) / "raw" / f"r{int(run_id):04d}"
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"No run directory for run_id={run_id!r}: {run_dir}"
            )
        return run_dir

    def load_run(
        self, run_id: int | str, root_path: Path, *, trains: slice | None = None
    ) -> xr.Dataset:
        """Build the Tier-1 lazy Dataset for one MID run (CLAUDE.md 5.3).

        Opens the run directory with EXtra-data and returns a dask-backed
        :class:`xarray.Dataset`. Detector arrays (AGIPD, Jungfrau) stay lazy —
        never materialized here. EXtra-data is imported lazily so the rest of this
        module imports without it.

        Parameters
        ----------
        trains:
            Optional positional slice selecting a subset of trains (e.g.
            ``slice(0, 20)``), forwarded to ``DataCollection.select_trains``.
            Opening and building the dask graph over a full MHz run (thousands of
            trains × 352 pulses) is expensive; slice for quick inspection.

        Schema status
        -------------
        Source/key names (r0500) and the EXtra-data API here are verified against
        the installed package. Each source is stored as a *separate, isolated*
        data variable via :func:`_isolate_source`: dims/coords are
        per-source–prefixed (``agipd_*``, ``jf500k1_*``, ``xgm_flux_*``,
        ``litframe_*``) and pandas indexes are reset. This is required because
        EXtra-data hands back colliding generic names (an ``AGIPD1M``
        ``train_pulse`` MultiIndex vs. plain ``trainId`` indexes, plus repeated
        ``dim_0``/…) that xarray would otherwise try — and fail — to align.

        Consequence: this is NOT yet the canonical unified schema
        ``(train, pulse, module, ss, fs)`` of 5.3. Splitting the raw AGIPD
        data/gain axis, aligning sources on a common train axis, and the
        XGM↔AGIPD pulse mapping remain deferred (CLAUDE.md §8).
        """
        import extra_data
        import numpy as np
        from extra_data.components import AGIPD1M
        from extra_data.exceptions import (
            NoDataError,
            PropertyNameError,
            SourceNameError,
        )

        root_path = Path(root_path)
        run_dir = self.get_run_path(run_id, root_path)
        run = extra_data.RunDirectory(run_dir)
        if trains is not None:
            run = run.select_trains(trains)  # by position; keeps everything lazy

        absent = (SourceNameError, PropertyNameError, NoDataError)

        def optional_dask(source: str, key: str) -> xr.DataArray | None:
            """Lazy labelled DataArray for *source*/*key*, or None if absent."""
            try:
                return run.get_dask_array(source, key, labelled=True)
            except absent:
                return None

        def run_value(source: str, key: str) -> Any | None:
            try:
                return run.get_run_value(source, key)
            except absent:
                return None

        data_vars: dict[str, xr.DataArray] = {}

        # AGIPD-1M (SAXS/XPCS/XCCA), lazy. Isolated so its train_pulse MultiIndex
        # neither collides nor aligns with the train-indexed sources below.
        agipd = AGIPD1M(run)
        data_vars["agipd"] = _isolate_source(
            agipd.get_dask_array(AGIPD_IMAGE_KEY), "agipd"
        )

        # Jungfrau-500K WAXS units (raw ADU), lazy; skip a unit if not recorded.
        for name, source in (("jf500k1", JF500K1_SOURCE), ("jf500k2", JF500K2_SOURCE)):
            array = optional_dask(source, JF_ADC_KEY)
            if array is not None:
                data_vars[name] = _isolate_source(array, name)

        # Per-pulse incident flux and lit-frame pattern.
        for name, source, key in (
            ("xgm_flux", XGM_SOURCE, XGM_FLUX_KEY),
            ("litframe", LITFRM_SOURCE, LITFRM_PATTERN_KEY),
        ):
            array = optional_dask(source, key)
            if array is not None:
                data_vars[name] = _isolate_source(array, name)

        # Per-train coords on a dedicated 'train_index' dim (kept separate from the
        # detectors' native train axes until canonical alignment is implemented).
        coords: dict[str, Any] = {"train_id": ("train_index", list(run.train_ids))}
        timestamps = run.train_timestamps()  # datetime64, aligned to run.train_ids
        if timestamps is not None:
            coords["timestamp"] = ("train_index", timestamps)

        proposal = proposal_number_from_root(root_path)
        sample_name = (
            load_elog_csv(root_path).get(int(run_id), {}).get("sample") or None
        )
        transmission = run_value(ATTENUATOR_SOURCE, ATTENUATOR_TRANSMISSION_KEY)

        attrs: dict[str, Any] = {
            "proposal": proposal,
            "run": int(run_id),
            "facility": self.facility,
            "beamline": self.beamline,
            "photon_energy_ev": PHOTON_ENERGY_EV,
            "wavelength_A": wavelength_angstrom_from_energy_ev(PHOTON_ENERGY_EV),
            "detector_distance_m": DETECTOR_DISTANCE_M,
            "sample_name": sample_name,
            "n_trains": len(run.train_ids),
        }
        # frames_per_train exists on the component; return type varies, so guard.
        try:
            attrs["n_pulses_per_train"] = int(np.max(agipd.frames_per_train()))
        except Exception:
            pass
        if transmission is not None:
            attrs["transmission"] = transmission

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)


# ── Self-register on import (matches maxiv.py / mock.py idiom) ──────────────────
ReaderRegistry.register(EuXFELMIDRawReader)
