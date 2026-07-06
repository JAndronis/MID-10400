"""Unit tests for the EuXFEL MID raw reader.

Covers the module-level pure helpers, ``can_read``, ``list_runs``,
``get_run_path``, and self-registration. ``load_run`` needs EXtra-data and real
data, so its smoke test is marked ``integration`` and skipped when either is
absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyBeamtime.core.run import RunMetadata
from pyBeamtime.io.readers import ReaderRegistry

from p010400_mid.io.readers.euxfel import (
    EuXFELMIDRawReader,
    _isolate_source,
    list_run_dirs,
    parse_run_dir,
    proposal_number_from_root,
    wavelength_angstrom_from_energy_ev,
)


# ── module-level pure helpers ──────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("r0042", 42),
        ("r0500", 500),
        ("r0000", 0),
        ("r12345", 12345),
        ("raw", None),
        ("r", None),
        ("r0042x", None),
        ("xr0042", None),
        ("R0042", None),  # case-sensitive
        ("r0003.txt", None),
    ],
)
def test_parse_run_dir(name: str, expected: int | None) -> None:
    assert parse_run_dir(name) == expected


def test_proposal_number_from_root() -> None:
    root = Path("/gpfs/exfel/exp/MID/202601/p010400")
    assert proposal_number_from_root(root) == 10400
    assert proposal_number_from_root(Path("p000700")) == 700


def test_proposal_number_from_root_rejects_bad_basename() -> None:
    with pytest.raises(ValueError):
        proposal_number_from_root(Path("/data/mid_202601"))


def test_list_run_dirs_sorted_and_filtered(raw_root: Path) -> None:
    result = list_run_dirs(raw_root / "raw")
    assert [run_id for run_id, _ in result] == [1, 7, 10]
    assert all(path.is_dir() for _, path in result)
    assert all(path.name == f"r{run_id:04d}" for run_id, path in result)


def test_list_run_dirs_missing_raw(tmp_path: Path) -> None:
    assert list_run_dirs(tmp_path / "does_not_exist") == []


def test_wavelength_from_energy() -> None:
    # 9.04 keV → ~1.371 Å (CLAUDE.md §2).
    assert wavelength_angstrom_from_energy_ev(9040.0) == pytest.approx(1.3715, abs=1e-3)


# ── can_read (two-layer sentinel) ──────────────────────────────────────────────
def test_can_read_true(raw_root: Path) -> None:
    assert EuXFELMIDRawReader.can_read(raw_root) is True


def test_can_read_false_no_raw(tmp_path: Path) -> None:
    assert EuXFELMIDRawReader.can_read(tmp_path) is False


def test_can_read_false_no_sentinel(tmp_path: Path, make_raw_tree) -> None:
    root = make_raw_tree(tmp_path / "p010400", [1, 2], sentinel=False)
    assert (root / "raw").is_dir()
    assert EuXFELMIDRawReader.can_read(root) is False


# ── list_runs ──────────────────────────────────────────────────────────────────
def test_list_runs_merges_elog(raw_root: Path, write_elog) -> None:
    write_elog(
        raw_root,
        [
            {"scan_id": 1, "sample": "ferritin_peg6000", "peg": "6000"},
            {"scan_id": 10, "sample": "ferritin_ref", "peg": ""},
        ],
    )
    runs = {m.scan_id: m for m in EuXFELMIDRawReader().list_runs(raw_root)}

    assert set(runs) == {1, 7, 10}
    assert runs[1].sample_name == "ferritin_peg6000"
    assert runs[1].extra == {"peg": "6000"}  # non-sample columns pass through
    assert runs[7].sample_name is None  # on disk but not in elog (decision 015)
    assert runs[10].sample_name == "ferritin_ref"

    # RunMetadata semantics from CLAUDE.md 5.2.1 (data-derived fields deferred).
    assert runs[1].exposure_time == 0.0
    assert runs[1].n_frames is None
    assert runs[1].start_time is None
    assert runs[1].end_time is None


def test_list_runs_no_elog(raw_root: Path) -> None:
    runs = EuXFELMIDRawReader().list_runs(raw_root)
    assert [m.scan_id for m in runs] == [1, 7, 10]
    assert all(m.sample_name is None for m in runs)
    assert all(isinstance(m, RunMetadata) for m in runs)


# ── get_run_path (returns the run directory) ───────────────────────────────────
def test_get_run_path_returns_directory(raw_root: Path) -> None:
    path = EuXFELMIDRawReader().get_run_path(7, raw_root)
    assert path == raw_root / "raw" / "r0007"
    assert path.is_dir()


def test_get_run_path_zero_pads_and_accepts_str(raw_root: Path) -> None:
    assert EuXFELMIDRawReader().get_run_path("1", raw_root).name == "r0001"


def test_get_run_path_missing_raises(raw_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EuXFELMIDRawReader().get_run_path(999, raw_root)


# ── self-registration ──────────────────────────────────────────────────────────
def test_registered_by_class_name() -> None:
    assert ReaderRegistry.get("EuXFELMIDRawReader") is EuXFELMIDRawReader


def test_registered_by_slug() -> None:
    assert ReaderRegistry.get_by_slug("mid") is EuXFELMIDRawReader


def test_reader_class_attrs() -> None:
    assert EuXFELMIDRawReader.slug == "mid"
    assert EuXFELMIDRawReader.beamline == "MID"
    assert EuXFELMIDRawReader.facility == "European XFEL"
    assert EuXFELMIDRawReader.priority == 0
    assert EuXFELMIDRawReader.paired_facility_reader is None


# ── _isolate_source (regression for the multi-source alignment crash) ──────────
def _agipd_like() -> xr.DataArray:
    """Mimic AGIPD1M.get_dask_array output: a stacked (trainId, pulseId) index."""
    trains = np.arange(1000, 1005, dtype="uint64")
    pulses = np.arange(0, 6, 2)
    frames = pd.MultiIndex.from_product([trains, pulses], names=["trainId", "pulseId"])
    return xr.DataArray(
        np.zeros((16, len(frames), 2, 4, 3)),
        dims=["module", "train_pulse", "dim_0", "dim_1", "dim_2"],
        coords={"train_pulse": frames, "module": np.arange(16)},
    )


def _train_indexed_like(n_trains: int = 3) -> xr.DataArray:
    """Mimic get_dask_array(labelled=True): a plain trainId index + dim_0/1/2."""
    trains = np.arange(1000, 1000 + n_trains, dtype="uint64")
    return xr.DataArray(
        np.zeros((n_trains, 8, 4, 3)),
        dims=["trainId", "dim_0", "dim_1", "dim_2"],
        coords={"trainId": trains},
    )


def test_raw_sources_collide_without_isolation() -> None:
    # Reproduces the load_run crash: a trainId MultiIndex level cannot align with
    # a plain trainId index of a different length.
    with pytest.raises(xr.AlignmentError):
        xr.Dataset({"agipd": _agipd_like(), "jf": _train_indexed_like()})


def test_isolate_source_prevents_collision_and_prefixes() -> None:
    ds = xr.Dataset(
        {
            "agipd": _isolate_source(_agipd_like(), "agipd"),
            "jf": _isolate_source(_train_indexed_like(), "jf"),
        }
    )
    assert set(ds.data_vars) == {"agipd", "jf"}
    # every dim/coord is per-source prefixed, so nothing aligns or collides
    assert "agipd_train_pulse" in ds.dims
    assert {"agipd_trainId", "agipd_pulseId", "jf_trainId"} <= set(ds.coords)
    assert not any(str(dim).startswith("dim_") for dim in ds.dims)
    # id values survive the reset_index
    assert int(ds["agipd_trainId"].values[0]) == 1000
    assert int(ds["jf_trainId"].values[-1]) == 1002


# ── integration: load_run (needs EXtra-data + real data) ───────────────────────
_DATA_ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400")


def _extra_data_available() -> bool:
    return importlib.util.find_spec("extra_data") is not None


@pytest.mark.integration
@pytest.mark.skipif(
    not _extra_data_available() or not _DATA_ROOT.exists(),
    reason="requires extra_data and the p010400 data tree (Maxwell)",
)
def test_load_run_smoke() -> None:
    ds = EuXFELMIDRawReader().load_run(500, _DATA_ROOT)
    assert "agipd" in ds.data_vars
    assert ds.attrs["proposal"] == 10400
    assert ds.attrs["run"] == 500
    assert ds.attrs["photon_energy_ev"] == 9040.0
