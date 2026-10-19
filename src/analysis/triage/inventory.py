"""One row per run: what it is, how big, and which DAMNIT products exist (§3, §4.1).

Reads DAMNIT's sqlite and the per-run HDF5 files, both read-only. The sqlite is
opened through a ``mode=ro`` URI, so not even a journal is written beside it. A
product is ``present`` only when its main dataset has data. DAMNIT stores a variable
that returned nothing as a 0-d value, and that is recorded as ``empty``, not as data.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analysis.damnit_names import stored
from analysis.triage.samples import REFERENCE_SAMPLES, effective_sample

__all__ = [
    "COMMISSIONING_LAST_RUN",
    "DAMNIT_ROOT",
    "RUN_COLUMNS",
    "TRACKED",
    "build_inventory",
    "classify",
    "damnit_file",
    "file_facts",
    "jf_block",
    "parse_run_length",
    "read_run_table",
    "stray_light_era",
]

DAMNIT_ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400/usr/Shared/amore")
#: r1–42 were commissioning (user, 2026-10-01): inventoried, never analysed.
COMMISSIONING_LAST_RUN = 42
#: The stray-light geometry changed between these two runs (memory,
#: saxs-background-subtraction); runs strictly between them are undetermined.
STRAY_LIGHT_LAST_BEFORE, STRAY_LIGHT_FIRST_AFTER = 433, 443
#: The sqlite columns the inventory reads. Everything else in ``runs`` is a preview.
RUN_COLUMNS = (
    "run", "start_time", "run_length", "run_type", "sample", "scan_type",
    "n_trains", "pulses", "rep_rate", "raw_size", "proc_size",
    "total_transmission", "xgm_intensity",
)  # fmt: skip
#: DAMNIT product -> its main dataset, the one that must hold data for "present".
#: Keyed by the flat name `facts` reports under; `damnit_names.stored` finds the
#: grouped one in a file.
TRACKED: dict[str, str] = {
    "agipd_saxs": "intensity",
    "jungfrau_waxs_jf1": "intensity",
    "jungfrau_waxs_jf2": "intensity",
    "xpcs_ttcf": "__xarray_dataarray_variable__",
    "xpcs_mean_dataset": "g2",
    "droplet_fit": "volume",
    "xgm_intensity": "Energy",
    "total_transmission": "__xarray_dataarray_variable__",
}


def damnit_file(run: int, root: Path = DAMNIT_ROOT, proposal: int = 10400) -> Path:
    return Path(root) / "extracted_data" / f"p{proposal}_r{run}.h5"


def read_run_table(sqlite_path: Path) -> pd.DataFrame:
    """DAMNIT's ``runs`` table, `RUN_COLUMNS` only, sorted by run, opened read-only."""
    uri = f"file:{Path(sqlite_path)}?mode=ro"
    query = f"SELECT {', '.join(RUN_COLUMNS)} FROM runs ORDER BY run"
    with closing(sqlite3.connect(uri, uri=True)) as con:
        return pd.read_sql(query, con)


def parse_run_length(text: str | None) -> float:
    """DAMNIT's ``H:MM:SS`` run length in seconds; NaN when absent.

    Absent arrives as None or, through pandas, as a float NaN (r332, whose
    ``run_length`` variable errored).
    """
    if text is None or (isinstance(text, float) and np.isnan(text)) or text == "":
        return float("nan")
    h, m, s = (int(p) for p in str(text).split(":"))
    return float(3600 * h + 60 * m + s)


def classify(run: int, run_type: str | None, scan_type: str | None, sample: str) -> str:
    """The run's class (§3.1). The first rule that matches wins."""
    run_type, scan_type = str(run_type or ""), str(scan_type or "")
    if run_type.startswith(("Calibration", "Dark")):
        return "calib_dark"
    if run_type.startswith("Test"):
        return "test"
    if run <= COMMISSIONING_LAST_RUN:
        return "commissioning"
    if "SSHEX" in scan_type:
        return "mesh_sshex"
    if sample in REFERENCE_SAMPLES:
        return "reference"
    return "static_droplet"


def stray_light_era(run: int) -> str:
    """Which stray-light geometry the run was taken under."""
    if run <= STRAY_LIGHT_LAST_BEFORE:
        return f"le_r{STRAY_LIGHT_LAST_BEFORE}"
    if run >= STRAY_LIGHT_FIRST_AFTER:
        return f"ge_r{STRAY_LIGHT_FIRST_AFTER}"
    return "undetermined"


def jf_block(cells) -> str | None:
    """The JUNGFRAU readout block from the lit cells DAMNIT kept (§4.1)."""
    if cells is None:
        return None
    cells = sorted({int(c) for c in np.asarray(cells).ravel()})
    if cells == list(range(16)):
        return "all16"
    return "cells" + ",".join(str(c) for c in cells)


def _first_line(value) -> str:
    text = value.decode(errors="replace") if isinstance(value, bytes) else str(value)
    return text.strip().splitlines()[0] if text.strip() else ""


def _status(f: h5py.File, name: str, main: str) -> tuple[str, str | None]:
    flat = name
    name = stored(f, name)
    if name in f:
        group = f[name]
        if main not in group or group[main].ndim == 0:
            return "empty", None
        if name == "droplet_fit" and not np.isfinite(group[main][:]).any():
            return "empty", None
        return "present", None
    if ".errors" in f:
        for key in (name, flat):
            if key in f[".errors"]:
                return "errored", _first_line(f[".errors"][key][()])
    return "absent", None


def file_facts(path: Path) -> dict:
    """Product status, JF readout block and XGM statistics from one DAMNIT file."""
    facts: dict = {"damnit_file": str(path)}
    if not Path(path).exists():
        facts.update({f"var_{n}": "no_file" for n in TRACKED}, errors={})
        return facts
    errors = {}
    with h5py.File(path, "r") as f:
        for name, main in TRACKED.items():
            status, err = _status(f, name, main)
            facts[f"var_{name}"] = status
            if err is not None:
                errors[name] = err
        jf1 = stored(f, "jungfrau_waxs_jf1")
        jf = f[f"{jf1}/cellId"][:] if f"{jf1}/cellId" in f else None
        xgm = (
            f["xgm_intensity/Energy"][:]
            if facts["var_xgm_intensity"] == "present"
            else None
        )
    facts["errors"] = errors
    facts["jf_block"] = jf_block(jf)
    if xgm is not None and np.isfinite(xgm).any():
        mean = float(np.nanmean(xgm))
        facts["xgm_mean_uJ"] = mean
        facts["xgm_cv"] = float(np.nanstd(xgm) / mean) if mean else float("nan")
    return facts


def build_inventory(
    runs: pd.DataFrame, damnit_root: Path = DAMNIT_ROOT, proposal: int = 10400
) -> pd.DataFrame:
    """The §4.1 inventory, one row per run of `runs` (as `read_run_table` returns)."""
    rows = []
    for rec in runs.itertuples(index=False):
        run = int(rec.run)
        sample = effective_sample(run, rec.sample)
        start = (
            datetime.fromtimestamp(float(rec.start_time), tz=UTC)
            if pd.notna(rec.start_time)
            else None
        )
        row = {
            "run": run,
            "start_utc": start,
            "run_length_s": parse_run_length(rec.run_length),
            "run_type": rec.run_type,
            "label": rec.sample,
            "sample": sample.name,
            "sample_source": sample.source,
            "background_role": sample.background_role,
            "exclude": sample.exclude,
            "cls": classify(run, rec.run_type, rec.scan_type, sample.name),
            "scan_type": rec.scan_type,
            "n_trains": rec.n_trains,
            "pulses": rec.pulses,
            "rep_rate_mhz": rec.rep_rate,
            "attenuator": rec.total_transmission,
            "raw_tb": rec.raw_size,
            "proc_tb": rec.proc_size,
            "stray_light_era": stray_light_era(run),
        }
        row.update(file_facts(damnit_file(run, damnit_root, proposal)))
        rows.append(row)
    return pd.DataFrame(rows)
