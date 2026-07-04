# EXtra-data — Codebase Summary

**Repo:** [European-XFEL/EXtra-data](https://github.com/European-XFEL/EXtra-data)
**Language:** Python 3 (h5py, numpy, pandas, xarray, dask-compatible)
**License:** BSD-3-Clause
**PyPI:** `pip install extra_data`

## What it does

EXtra-data is European XFEL's Python library for reading the facility's HDF5 experiment files (the standard "EuXFEL data format" written per-train, per-source at instruments like MID, SPB, FXE, etc.). It gives users a uniform way to open a single file or a whole run directory and pull out data by *source* (a device/detector) and *key* (a specific property of that source), without needing to know the underlying HDF5 layout, file-splitting scheme, or which files on disk hold which trains.

Core responsibilities:

- **Discovery & opening**: locate and open the (potentially many) HDF5 files that make up a run, transparently merging them into one logical dataset.
- **Indexing by train**: EuXFEL data is organized around "trains" (pulse trains, the facility's fundamental time unit); the library exposes selection, iteration, and alignment by train ID.
- **Lazy, chunked access**: large detector data (e.g. multi-megapixel AGIPD/LPD/DSSC/JUNGFRAU frames) is not loaded until requested, and can be pulled as NumPy arrays, labelled Xarray `DataArray`s, Dask arrays (for out-of-core/parallel processing), or pandas Series/DataFrames.
- **Multi-module detector assembly**: convenience classes stitch together the many per-module sources that make up a single physical detector into one array-like object.
- **Streaming & export**: re-serve file or run data live over ZeroMQ in the Karabo Bridge protocol (useful for testing online/streaming analysis code offline).
- **Validation & tooling**: CLI utilities to validate run structure, inspect what's in a run/proposal, check readability, check dCache locality, and build virtual CXI files for detector data.

This maps closely to Iason's `pyBeamtime` work — EXtra-data is exactly the "EXtra library ecosystem" layer that framework is designed to wrap (alongside EXtra-geom / EXtra-metro), providing the raw source/key/train access that a higher-level xarray+dask abstraction would sit on top of.

## Exposed API layer

The public surface is deliberately small and is re-exported at the top level in `extra_data/__init__.py`. Three classes form the core object model (per `docs/architecture.rst`), plus opening functions, selection helpers, and detector components.

### 1. Entry points (opening data)

| Function | Purpose |
|---|---|
| `H5File(path)` | Open a single EuXFEL HDF5 file → `DataCollection` |
| `RunDirectory(path)` | Open a whole run directory (many files) → `DataCollection` |
| `open_run(proposal, run, ...)` | Look up and open a run by proposal/run number (e.g. on Maxwell) → `DataCollection` |

### 2. `DataCollection` — the run/file-level object

Represents data for many sources across a range of trains. Returned by the functions above. Key methods include:

- **Selection/combination**: `select()`, `deselect()`, `select_trains()`, `union()`, `split_trains()`, `with_aliases()` / `only_aliases()` / `drop_aliases()`
- **Iteration**: `trains()`, `train_from_id()`, `train_from_index()`
- **Bulk access**: `get_array()`, `get_dask_array()`, `get_dataframe()`, `get_series()`
- **Metadata/introspection**: `all_sources`, `detector_sources`, `keys_for_source()`, `get_entry_shape()`, `get_dtype()`, `info()`, `train_info()`, `detector_info()`, `plot_missing_data()`
- **Run/control values**: `get_run_value()`, `get_run_values()`, `run_metadata`
- **Output**: `write()`, `write_virtual()`, `get_virtual_dataset()`

Item access (`run[source]` / `run[source, key]`) is the idiomatic way in, returning the two classes below.

### 3. `SourceData` — one source (device/detector module)

Returned by `run[source]`. Represents everything about a single named source (e.g. a motor, camera, or one detector module) across the selected trains: `keys()`, `select_keys()`, `select_trains()`, `data_counts()`, `run_value()` / `run_values()`, `device_class`, plus `is_control` / `is_instrument` / `is_legacy` type flags.

### 4. `KeyData` — one source + one property

Returned by `run[source, key]`. Represents a single array-like quantity (dtype/shape known without loading data). Loading/conversion methods: `ndarray()`, `xarray()`, `dask_array()`, `series()`, plus `select_trains()`, `train_from_id()`, `trains()`, `data_counts()`, `as_single_value()` (for scalar control values), `units`.

### 5. Supporting top-level exports

- `by_id`, `by_index` — helpers to specify train/pulse selections by ID vs. positional index.
- `SourceNameError`, `PropertyNameError`, plus other exceptions (`TrainIDError`, `AliasError`, `MultiRunError`) from `extra_data.exceptions`.
- `AliasIndexer` (`extra_data.aliases`) — lets users refer to sources/keys by short human-friendly aliases instead of full device names.
- Stacking helpers (`extra_data.stacking`): `stack_data()`, `stack_detector_data()`, `StackView` — for combining per-module arrays into one array without copying.

### 6. Detector components (`extra_data.components`)

Higher-level, detector-specific wrappers built on `DataCollection`/`SourceData` for European XFEL's segmented, multi-module 2D detectors, so users don't have to manually stitch together dozens of per-module sources:

- `AGIPD1M`, `AGIPD500K`
- `DSSC1M`
- `LPD1M`, `LPDSolo`
- `JUNGFRAU`

These share common base classes (`MultimodDetectorBase`, `XtdfDetectorBase`) and expose assembled-image access, pulse/frame selection, masking, and CXI export (`write_virtual_cxi()`). (Note: newer/more detector types now live in the separate `EXtra` package, which builds on this one.)

### 7. Command-line tools (installed as console scripts)

| Command | Backing module | Purpose |
|---|---|---|
| `lsxfel` | `cli/lsxfel.py` | List proposals/runs/sources in a directory |
| `extra-data-validate` | `validation.py` | Check a run/file matches the expected EuXFEL HDF5 structure |
| `extra-data-make-virtual-cxi` | `cli/make_virtual_cxi.py` | Build a virtual-dataset CXI file for detector data |
| `extra-data-locality` | `locality.py` | Check whether files are on disk or migrated to tape (dCache) |
| `extra-data-readable` | `cli/check_readable.py` | Check files can be opened/read |
| `karabo-bridge-serve-files` / `karabo-bridge-serve-run` | `cli/serve_files.py`, `cli/serve_run.py` | Re-stream file/run data over ZeroMQ in Karabo Bridge format |

### 8. Internal/lower-level (not part of the stable public API, but relevant architecturally)

- `FileAccess` (`file_access.py`) — one instance per physical HDF5 file on disk; manages the underlying `h5py.File` handle and cached index information. `DataCollection`, `SourceData`, and `KeyData` all reference shared `FileAccess` objects rather than duplicating file handles.
- `read_machinery.py` — shared low-level helpers (train-ID/index selection logic, proposal path discovery).
- `run_files_map.py` — caches per-run file metadata in JSON to speed up repeated opens of large runs.
- `writer.py` / `write_cxi.py` — implementation behind `DataCollection.write()`/`write_virtual()` and the virtual-CXI export.

## Typical usage pattern

```python
from extra_data import open_run

run = open_run(proposal=700000, run=1)          # -> DataCollection
run.all_sources                                  # discover available sources
src = run["SA3_XTD10_PES/ADC/1:network"]          # -> SourceData
key = src["digitizers.channel_4_A.raw.samples"]   # -> KeyData
arr = key.ndarray()                               # or .xarray() / .dask_array()

# Per-train iteration for data too large to hold in memory:
for train_id, data in run.select("*/DET/*", "image.data").trains():
    ...
```

## Relevance to your work

This is the library your `pyBeamtime` design already targets as the base I/O layer (alongside EXtra-geom for detector geometry and EXtra-metro). The `DataCollection → SourceData → KeyData` three-tier object model, and the fact that `KeyData` already exposes `.xarray()` and `.dask_array()` natively, lines up directly with your stated architectural preference for xarray `DataArray` output with dask-backed lazy loading — you'd mainly be wrapping/adapting this existing hierarchy behind your own abstract reader interface rather than reimplementing lazy access from scratch.
