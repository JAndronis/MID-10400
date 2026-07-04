# Architecture Reference

## Directory Layout

```
src/
    pyBeamtime/
        __init__.py
        __version__.py
        beamtime.py            # Beamtime class
        cli/
            __init__.py        # click group: `beamtime`
            init.py            # `beamtime init` command
            info.py            # `beamtime info` command
            validate.py        # `beamtime validate` command
            import_elog.py     # `beamtime import-elog` command
        core/
            __init__.py
            run_group.py       # RunGroup class
            run.py             # Run class (thin delegate to RunGroup)
        io/
            __init__.py
            zarr_store.py       # UserProcessedReader + UserProcessedWriter + schema constants
            reduction_store.py  # ReductionWriter/Reader + REDUCTION_SCHEMA_VERSION  [to add]
            elog.py             # load_elog_csv() utility + canonical schema constants
            readers/
                __init__.py     # ReaderRegistry singleton
                base.py         # BaseRawReader, BaseFacilityProcessedReader ABCs + RunMetadata
                mock.py         # Mock reader for tests
                maxiv.py        # MAX IV reader plugins (CoSAXSRawReader, CoSAXSProcessedAzintReader)
                esrf.py         # ESRF reader plugins  [to add]
                euxfel.py       # EuXFEL reader plugins  [to add]
                desy.py         # DESY reader plugins (P10, ...)  [to add]
        analysis/
            __init__.py
            analysis_result.py # AnalysisResult + AzimuthalIntegration + MultiDetectorAzimuthalIntegration
            cormap.py          # pure CorMap statistics & pairwise matrix  [to add]
            frame_selection.py # CorMap + KDE frame-selection function     [to add]
            merge.py           # SAXS/WAXS merge + scale-factor helpers    [to add]
            filters.py         # BaseFrameFilter ABC + CorMapKDEFilter     [to add]
            integrate.py       # SAXS/WAXS azimuthal integration           [to add]
            correlate.py       # XPCS correlation functions                [to add]
            fit.py             # physical model fitting                    [to add]
            plot.py            # visualisation                             [to add]
```

## Beamtime Path Model

```
root_path/                          # one beamtime, one facility, one beamline
    beamtime.json                   # config written by `beamtime init`
    elog.csv                        # canonical elog sidecar (written by `beamtime import-elog`)
    <raw data dirs>/                # reader resolves internally; not encoded in Beamtime
    user_processed/                 # default user_processed_path
    ...
facility_processed_path/            # separate; nullable; facility-specific location
```

## beamtime.json Schema

Written by `beamtime init`. Read by `Beamtime.from_config()`.

```json
{
  "schema_version": 1,
  "beamline": "CoSAXS",
  "facility": "MAX IV",
  "root_path": "/data/2024/beamtime_12345",
  "facility_processed_path": null,
  "user_processed_path": null,
  "raw_reader": "CoSAXSRawReader",
  "facility_processed_reader": null
}
```

`schema_version` is validated by `from_config`; raise `ConfigVersionError` on mismatch.
`beamline` / `facility` are display-only. All logic is driven by reader class names.
`user_processed_path` null → resolved to `root_path / "user_processed"` at load time.
`root_path` stored absolute; validated to exist at `from_config` time.

## RunMetadata

Defined in `io/readers/base.py`. Returned by `BaseRawReader.list_runs()`.

```python
@dataclass
class RunMetadata:
    scan_id: int                        # from filename; authority is the filesystem
    start_time: datetime | None         # from HDF5 /entry/start_time
    end_time: datetime | None           # from HDF5 /entry/end_time
    sample_name: str | None             # from elog.csv; None = not logged = unlabelled
    n_frames: int | None                # total detector frames in the scan
    exposure_time: float | None         # from HDF5 /entry/instrument/eiger/count_time
    extra: dict[str, Any] = field(default_factory=dict)  # elog extra cols + anything else
```

**Invariant:** `scan_id` is always present (parsed from filename). All other fields are
best-effort and may be `None`. `sample_name = None` is a first-class state meaning "not
in elog" — not an error, not an empty string.

`select_runs()` with any sample filter silently skips runs where `sample_name is None`.
`select_runs()` also supports filtering on arbitrary `extra` fields.
Unlabelled runs remain accessible by scan number via `beamtime[scan_id]`.

## Elog Sidecar

### Canonical `elog.csv` schema

Lives at `root_path / elog.csv`. Written by `beamtime import-elog`.

Required columns:

| Column | Type | Notes |
|---|---|---|
| `scan_id` | int | must match filenames on disk |
| `sample` | str | sample name; must not be empty |

All additional columns from the source file are preserved verbatim and stored in
`RunMetadata.extra` as strings.

`beamtime import-elog` is the only writer. The file should not be hand-edited.
If `elog.csv` is absent, `list_runs()` returns all scans with `sample_name = None`.

### `io/elog.py`

```python
ELOG_FILENAME = "elog.csv"
ELOG_SCAN_COL = "scan_id"
ELOG_SAMPLE_COL = "sample"

def load_elog_csv(root_path: Path) -> dict[int, dict]:
    """
    Read root_path/elog.csv and return {scan_id: {"sample": ..., ...extra}} mapping.
    Returns empty dict if file absent.
    Raises ElogSchemaError if required columns are missing or scan_id not parseable as int.
    """
```

Used by reader plugins (e.g. `CoSAXSRawReader.list_runs()`) to merge sample names into
`RunMetadata`. Not called by `Beamtime` directly.

## CLI Design

Library: `click`.

```
beamtime init         --beamline SLUG [--root-path PATH]
                      [--facility-processed-path PATH]
                      [--user-processed-path PATH]
                      [--force]

beamtime info         [--root-path PATH]
                      # Reads beamtime.json, prints summary

beamtime validate     [--root-path PATH]
                      # Runs raw_reader.can_read(root_path), reports result

beamtime import-elog  <source-csv>
                      [--root-path PATH]
                      [--scan-col COLNAME]     # default: auto-detect
                      [--sample-col COLNAME]   # default: auto-detect
                      [--force]
                      # Normalises source CSV -> canonical elog.csv in root_path
```

`beamtime init` flow:
1. Resolve `root_path` (arg or cwd).
2. Guard: `beamtime.json` must not exist unless `--force`.
3. Resolve slug → class name via `ReaderRegistry.get_by_slug(slug)`.
4. `raw_reader.can_read(root_path)` → warn if False, but continue.
5. Write `beamtime.json`.

`beamtime import-elog` flow:
1. Resolve `root_path` (arg or cwd); validate `beamtime.json` exists.
2. Read source CSV; auto-detect or use explicit `--scan-col` / `--sample-col`.
3. Validate: scan col parseable as int, sample col non-empty, no duplicate scan IDs.
4. Rename to canonical column names; preserve all other columns verbatim.
5. Guard: `elog.csv` must not exist unless `--force`.
6. Write `root_path/elog.csv`.

Auto-detection for column names: case-insensitive, whitespace-collapsed match against
common variants (`scan id`, `scan_id`, `scan`, `#` for scan; `sample`, `sample name`,
`sample_name` for sample). Fails with a clear error if ambiguous or not found.

## Reader Plugin Protocol

```python
class BaseRawReader(ABC):
    paired_facility_reader: type[BaseFacilityProcessedReader] | None = None
    slug: str           # unique human slug, e.g. "cosaxs"; used by CLI
    priority: int = 0   # tiebreaker for ReaderRegistry.detect()

    @classmethod
    @abstractmethod
    def can_read(cls, root_path: Path) -> bool:
        """Sentinel-file fingerprinting. Used by init/validate/tests only."""

    @abstractmethod
    def list_runs(self, root_path: Path) -> list[RunMetadata]:
        """
        Return metadata for every run discoverable under root_path.
        All scans on disk are returned. Runs not in elog.csv get sample_name=None.
        """

    @abstractmethod
    def get_run_path(self, run_id: int | str, root_path: Path) -> Path:
        """Return the path to the master file for a given run ID."""

    @abstractmethod
    def load_run(self, run_id: int | str, root_path: Path) -> xr.Dataset:
        """Return a lazy, dask-backed xarray Dataset for one run."""


class BaseFacilityProcessedReader(ABC):
    slug: str
    priority: int = 0
    result_class: type[AnalysisResult] = AnalysisResult  # subclasses override this

    @classmethod
    @abstractmethod
    def can_read(cls, facility_processed_path: Path) -> bool: ...

    @abstractmethod
    def load_run(self, run_id: int | str, facility_processed_path: Path) -> "AnalysisResult": ...
```

### can_read Pattern

```python
class CoSAXSRawReader(BaseRawReader):
    slug = "cosaxs"
    _REQUIRED_DIRS = ("raw",)
    _SENTINEL_GLOB = "raw/scan-*.h5"

    @classmethod
    def can_read(cls, root_path: Path) -> bool:
        if not all((root_path / d).is_dir() for d in cls._REQUIRED_DIRS):
            return False
        return any(root_path.glob(cls._SENTINEL_GLOB))
```

One reader per beamline (not per facility) to keep `can_read` unambiguous without
needing to peek inside files.

## CoSAXS HDF5 Structure

Raw data lives in `root_path/raw/`. Each scan produces a master file and one or more
detector sidecar files, all in the same directory:

```
raw/
    scan-####.h5           # master file
    scan-####_eiger.h5     # Eiger SAXS detector frames (external link from master)
    scan-####_mythen.h5    # Mythen WAXS detector (external link from master)
    scan-####_pcap.h5      # PandABox timing/flux data (external link from master)
```

The scan number (`####`) is the run ID. It is **only** in the filename — not stored
inside the HDF5 content. `list_runs()` parses it from `Path("scan-####.h5").stem`.

External links in the master file are **relative paths**. h5py resolves them relative
to the directory of the file being opened, so the master must be opened from `raw/`
(i.e. pass the full absolute path). Do not rely on transparent link following across
filesystems or when the CWD differs from `raw/`.

### Key HDF5 paths in the master file

| Field | Path | Type | Notes |
|---|---|---|---|
| Start time | `/entry/start_time` | str (ISO 8601) | |
| End time | `/entry/end_time` | str (ISO 8601) | |
| Scan command | `/entry/title` | str | e.g. `"dscan"`, `"t-scan"` |
| Eiger frames | `/entry/instrument/eiger/data` | `(n, 2162, 2068)` uint32 | via external link to `_eiger.h5` |
| Mythen frames | `/entry/instrument/mythen_tr/data` | `(n, 1280)` int32 | via external link to `_mythen.h5` |
| Per-frame i₀ | `/entry/instrument/pcap/data/i_0` | `(n,)` float64 | via external link to `_pcap.h5` |
| Per-frame i_t | `/entry/instrument/pcap/data/i_t` | `(n,)` float64 | transmitted intensity |
| Per-frame timestamp | `/entry/instrument/pcap/data/timestamp` | `(n,)` float64 | hardware clock |
| Ring current | `/entry/instrument/ring_current/data` | `(n,)` float64 | per frame |
| Photon energy | `/entry/instrument/eiger/photon_energy` | float64 scalar | eV |
| Exposure time | `/entry/instrument/eiger/count_time` | float64 scalar | seconds per frame |
| Detector distance | `/entry/instrument/start_positioners/detector_carriage_z_position` | float64 scalar | mm |
| Sample description | `/entry/instrument/start_positioners/sample_description` | str | always empty; do not use |
| Sample x | `/entry/instrument/start_positioners/sample_table_x_position` | float64 scalar | |
| Sample y | `/entry/instrument/start_positioners/sample_table_y_position` | float64 scalar | |

Full instrument snapshot (~80 positioner/sensor fields) is in
`/entry/instrument/start_positioners/`. All fields are scalars (values at scan start).
These are stored wholesale into `xr.Dataset.attrs`.

### load_run() xarray Dataset conventions

`load_run()` returns a lazy dask-backed `xr.Dataset`. Dimension and variable names are
fixed for all CoSAXS data:

```
Dimensions:
    frame       — scan step index (0-based)
    pix_x       — Eiger pixel x (2068)
    pix_y       — Eiger pixel y (2162)
    q_mythen    — Mythen channel index (1280)

Data variables:
    eiger        (frame, pix_y, pix_x)   uint32    raw Eiger frames, lazy
    mythen       (frame, q_mythen)        int32     raw Mythen frames, lazy
    i0           (frame,)                 float64   incident flux
    i_t          (frame,)                 float64   transmitted flux
    ring_current (frame,)                 float64

Coordinates:
    timestamp    (frame,)   float64   PandABox hardware timestamp

Attributes (xr.Dataset.attrs):
    scan_id                 int
    start_time              str     ISO 8601
    end_time                str     ISO 8601
    scan_command            str     e.g. "dscan"
    sample_name             str | None
    photon_energy_ev        float
    detector_distance_mm    float
    beamline                "CoSAXS"
    facility                "MAX IV"
    <positioner_key>        float   one attr per start_positioners field
```

## ReaderRegistry

Class-level singleton in `io/readers/__init__.py`.

```python
class ReaderRegistry:
    # Registration — called at module import of each reader file
    @classmethod
    def register(cls, reader_cls): ...           # uses reader_cls.slug

    # Used by Beamtime.from_config — resolves class name string from JSON
    @classmethod
    def get(cls, class_name: str): ...

    # Used by `beamtime init` — resolves human slug to class
    @classmethod
    def get_by_slug(cls, slug: str): ...

    # Used by `beamtime validate` — finds all matching readers
    @classmethod
    def detect(cls, root_path: Path): ...
    # Returns (raw_cls, facility_processed_cls | None)
    # Collects all matching, sorts by priority desc, warns on ties
```

## AnalysisResult Hierarchy

Defined in `analysis/analysis_result.py`. Returned by `BaseFacilityProcessedReader.load_run()`
and accessible via `Run.load_processed()`.

```python
class AnalysisResult:
    """Base class for all typed facility-processed results."""

class AzimuthalIntegration(AnalysisResult):
    """1D SAXS/WAXS azimuthal integration result."""
    q: xr.DataArray          # scattering vector
    intensity: xr.DataArray  # integrated intensity
    error: xr.DataArray      # uncertainty

class MultiDetectorAzimuthalIntegration(AnalysisResult):
    """Two-detector (SAXS + WAXS) azimuthal integration result.

    Holds one AzimuthalIntegration per detector slot. Either slot may
    be None if the corresponding detector file is absent.

    Does NOT wrap a single xr.Dataset. `.data` raises NotImplementedError;
    access `.saxs` or `.waxs` directly.
    """
    saxs: AzimuthalIntegration | None
    waxs: AzimuthalIntegration | None
```

`BaseFacilityProcessedReader.result_class` declares statically what type a reader produces.
`CoSAXSProcessedAzintReader.result_class = MultiDetectorAzimuthalIntegration`.

## Run API

`Run` is a frozen dataclass holding a `RunMetadata`, two optional injected
loader callables (set by `Beamtime._discover_runs()` at construction time),
and an optional `frame_mask`. Callers never pass reader/path arguments.

```python
run.metadata                                         # RunMetadata
run.frame_mask        # bool xr.DataArray | None     # analysis mask; opt-in
run.load_raw()        -> xr.Dataset                  # calls raw_loader(run_id, metadata)
run.load_processed()  -> AnalysisResult              # calls processed_loader(run_id)
run.has_processed     -> bool                        # True iff processed_loader is set
run.with_mask(mask)   -> Run                         # returns new Run with mask set
```

Loader injection signature: `raw_loader: Callable[[int | str, RunMetadata], xr.Dataset]`.
The `(run_id, metadata)` signature allows callers to override `scan_id` at load time.

**Immutability:** `Run` is declared `@dataclass(frozen=True, eq=False)`.
`eq=False` suppresses auto-generated `__hash__` (xarray DataArrays aren't
hashable). Modifications go through `dataclasses.replace` or the
`run.with_mask()` convenience wrapper.

**Opt-in frame mask:** `run.load_processed()` returns full, unmasked data
regardless of `run.frame_mask`. Consumers that wish to respect the mask must
read `run.frame_mask` explicitly. `RunGroup.average_frames()` and
`RunGroup.merge_detectors()` do this internally; new consumers must decide
consciously.

## RunGroup API

`RunGroup` is the analysis entry point. Created by `Beamtime.select_runs()`
or by combining existing RunGroups. Holds an ordered collection of `Run`
objects and a back-reference to the parent `Beamtime` for calibration
inheritance.

### Set operations

All three operators return a new RunGroup (inputs unchanged):

```python
rg1 | rg2       # union on scan_ids
rg1 & rg2       # intersection on scan_ids
rg1 + rg2       # union on scan_ids (same as |, NOT concatenation)
```

For scan_ids present in both operands, `frame_mask` values must match:

- both `None`: OK, either Run carried through.
- both non-None and equal (`xr.DataArray.equals`): OK, either carried through.
- otherwise: raise `RunGroupMaskConflictError`, naming the conflicting scan_id.

`+` deviates from MDAnalysis `AtomGroup +`: RunGroup disallows duplicates
because Runs carry derived analysis state.

### Frame-mask application

```python
rg.with_frame_masks(masks: dict[int, xr.DataArray]) -> RunGroup
```

Returns a new RunGroup with masks applied to the matching Runs. scan_ids
in `masks` not in `rg` raise `KeyError`. Existing masks on source Runs are
replaced (not AND-ed) — filter chaining is the base-class `BaseFrameFilter`
responsibility, not RunGroup's.

### Reduction methods

```python
rg.average_frames()                        -> dict[int, MultiDetectorAzimuthalIntegration]
rg.merge_detectors(results, scale_factor)  -> dict[int, AzimuthalIntegration]
```

Both iterate per-run, respect `run.frame_mask` (opt-in), and return new
dict-keyed-by-scan_id results. See `reduction_module_spec.md` for detail.

### Analysis via external filters

Frame-level filtering does NOT live on RunGroup. Users instantiate a
`BaseFrameFilter` subclass and call it:

```python
filtered = CorMapKDEFilter(alpha=0.01)(rg)    # new RunGroup, masks populated
```

See `filter_module_spec.md`.

### Filter provenance

RunGroup tracks the filters applied to it:

```python
rg.filter_history   # tuple[BaseFrameFilter, ...], appended in order

rg.with_filter_record(f) -> RunGroup
    # Returns a copy with f appended to filter_history. Used by
    # BaseFrameFilter.__call__; user code should not call directly.
```

Set-op semantics: equal histories carry through; divergent histories
drop to `()` with a `UserWarning` (weaker than the mask-conflict rule,
which raises — divergent provenance is an annotation, divergent masks
are a correctness bug).

The `__repr__` surfaces the history for debugging:
`RunGroup(n_runs=42, filters=[CorMapKDEFilter(alpha=0.01)])`

### Reduction persistence

```python
rg.save_reduction(name: str, *, force: bool = False) -> None
```

Writes masks + filter_history to
`self._parent.user_processed_path / "reductions.zarr"` under
`sessions/<name>/`. Averaged/merged data is not saved; it's recomputed
from loaded masks on demand. See `persistence_module_spec.md`.

## Beamtime Convenience Properties

```python
beamtime.run_ids    # list[int]
beamtime.samples    # list[str | None]
beamtime.start_time # datetime | None   (earliest across all runs)
beamtime.end_time   # datetime | None   (latest across all runs)
beamtime.facility   # str
beamtime.beamline   # str
```

`beamtime.enrich_metadata()` — triggers HDF5-based metadata extraction for all runs,
populating fields not available at `list_runs()` time. Details TBD.

### Reduction persistence

```python
beamtime.load_reduction(name: str)   -> RunGroup
beamtime.list_reductions()            -> list[str]
beamtime.delete_reduction(name: str)  -> None
```

Backed by `user_processed_path / "reductions.zarr"` via
`io/reduction_store.py`. See `persistence_module_spec.md` for the store
schema and round-trip semantics.

## zarr Schema

`zarr_store.py` defines:

```python
SCHEMA_VERSION = 1
ZARR_ATTR_VERSION = "xpcs_framework_schema_version"
```

`UserProcessedWriter` writes `SCHEMA_VERSION` as a zarr attribute on every store root.
`UserProcessedReader` validates it; raises `SchemaVersionError` on mismatch.

## Reduction Schema

`io/reduction_store.py` defines a separate schema for reduction sessions
(frame masks + filter history):

```python
REDUCTION_SCHEMA_VERSION = 1
REDUCTION_ZARR_ATTR_VERSION = "pyBeamtime_reduction_schema_version"
REDUCTION_STORE_FILENAME = "reductions.zarr"
```

Lives at `user_processed_path / reductions.zarr`. Versioned independently
of `SCHEMA_VERSION`. See `persistence_module_spec.md` for the full
session layout.

## Data Flow

```
root_path/raw/scan-####.h5  (+ _eiger.h5, _mythen.h5, _pcap.h5)
        │
        ▼ run.load_raw()  [CoSAXSRawReader.load_run() injected at construction]
   xr.Dataset (lazy, dask-backed)
        │
        ▼ RunGroup.integrate() / .correlate() / etc.
   xr.Dataset (processed)
        │
        ▼ UserProcessedWriter.save()
   zarr store (user_processed_path / run_id / ...)   [default: root_path/user_processed/]
        │
        ▼ UserProcessedReader.load()   (future sessions)
   xr.Dataset (processed, lazy)

facility_processed_path/
        │
        ▼ run.load_processed()  [CoSAXSProcessedAzintReader.load_run() injected at construction]
   AzimuthalIntegration  [or other AnalysisResult subclass]
```
