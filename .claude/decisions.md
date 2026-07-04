# Design Decisions Log

Rationale for non-obvious choices. Useful context before proposing changes.

---

## 001 — No `raw_data_path` on `Beamtime`

**Decision:** `Beamtime` stores only `root_path`. There is no `raw_data_path` attribute.

**Rationale:** Raw data at synchrotron facilities is rarely a flat directory. Detector frames,
metadata/nexus files, and motor logs often live in different subdirectories (e.g. MAX IV's
`hdf5`/`master` split, ESRF's `RAW_DATA` layout). A single `raw_data_path` would be a fiction
immediately papered over. The raw reader plugin already encodes the facility's directory
conventions — letting it resolve paths from `root_path` avoids duplicating that knowledge
on `Beamtime`.

---

## 002 — Two processed data paths, not one

**Decision:** `facility_processed_path` and `user_processed_path` are separate attributes,
not a single `processed_data_path`.

**Rationale:** Two fundamentally different things:
- Facility-processed data is produced by the facility's own pipeline, lives in a
  facility-specific location, and has a facility-specific schema. At EuXFEL, correlation
  functions are expensive to recompute so users may prefer to load facility output. Nullable.
- User-processed data is produced by this framework, always zarr, always the same schema.
  Lives where the user wants, defaulting to `root_path / "processed"`.

---

## 003 — Three reader types, not one

**Decision:** `BaseRawReader` (plugin), `BaseFacilityProcessedReader` (plugin, nullable),
`UserProcessedReader` (internal, fixed).

**Rationale:** Raw and facility-processed readers vary per facility — plugin architecture
appropriate. User-processed data always uses the same zarr schema written by this framework
— a plugin adds complexity with no benefit.

---

## 004 — Reader responsibility boundary

**Decision:** Readers only read. No knowledge of downstream analysis, output paths, or
processed data locations.

**Rationale:** Coupling readers to the analysis pipeline makes them harder to test and
replace. `Beamtime` owns path decisions; analysis modules own output decisions.

---

## 005 — zarr read/write co-located in `zarr_store.py`

**Decision:** `UserProcessedReader` and `UserProcessedWriter` live in the same module,
alongside schema constants.

**Rationale:** Standard Python ecosystem practice (cf. `xarray/backends/zarr.py`). Schema
defined once; both reader and writer import from the same constants. Schema version bumps
touch exactly one file.

---

## 006 — Analysis logic lives in `RunGroup`, not `Run`

**Decision:** `Run.method()` always delegates to `RunGroup([self]).method()`. No analysis
logic duplicated on `Run`.

**Rationale:** Avoids maintaining two implementations of every analysis method.

---

## 007 — Beamtime represents exactly one visit

**Decision:** One `Beamtime` = one facility, one beamline, one proposal/visit.

**Rationale:** Makes `root_path` well-defined as a single value. Multi-visit comparisons
handled by combining `RunGroup` objects from separate `Beamtime` instances via set operations.

---

## 008 — Config-primary, no runtime auto-detection

**Decision:** `Beamtime.from_config()` is the real constructor. `from_path(root)` is a
convenience wrapper that looks for `root / beamtime.json`. There is no runtime
`ReaderRegistry.detect()` call during data loading. The config file is the sole source of
truth for which readers to use.

**Rationale:** Auto-detection (inspecting directory layout at runtime) is fragile — two
beamlines at the same facility could have similar structures, and detection would run on
every `Beamtime` construction. The `beamtime init` CLI makes detection a one-time explicit
step. `can_read` is retained on reader ABCs for use by `init` (validation warning) and
tests, but is not called on the data-loading path.

**Alternative considered:** Config-primary with detection fallback. Rejected because the
fallback path adds complexity and the one-time cost of running `beamtime init` is trivially
low.

---

## 009 — One reader per beamline, not one per facility

**Decision:** Reader plugins are scoped to individual beamlines (e.g. `CoSAXSRawReader`,
`P10RawReader`), not facilities (e.g. `MAXIVRawReader`).

**Rationale:** Beamline layouts differ significantly even within a facility — different
detector types, scan metadata schemas, file naming conventions. A per-facility reader would
need internal branching to handle each beamline, which is equivalent complexity without the
clean separation. For the scattering/XPCS use case only a small number of beamlines are
relevant, so the number of reader classes stays manageable. Per-beamline scope also makes
`can_read` unambiguous: two readers for the same facility won't both match the same
`root_path`, avoiding the need to peek inside files in most cases.

---

## 010 — `beamtime.json` is the config filename, stored at `root_path`

**Decision:** Config file is named `beamtime.json` (visible, not dot-prefixed), lives at
`root_path / beamtime.json`.

**Rationale:** Dot-prefixed hidden files are conventional for per-user tool config (`.git`,
`.env`) — not for project-level scientific metadata that collaborators need to see and
version-control. Placing it at `root_path` makes `from_path(root)` trivially simple.

---

## 011 — `root_path` stored absolute in config; known limitation if directory moves

**Decision:** `root_path` is written as an absolute path into `beamtime.json`.

**Rationale:** Allows `from_config(config_path)` to validate the root independently of
where the config file itself lives. Simpler than resolving from `config_path.parent`.

**Known limitation:** If the beamtime directory is archived to a different mount, the user
must edit `root_path` in `beamtime.json` manually. Acceptable for a solo PhD project;
document this explicitly in user-facing docs.

---

## 012 — Slug on reader class; class name in config

**Decision:** Each reader declares a `slug` class attribute (e.g. `slug = "cosaxs"`).
`beamtime init --beamline cosaxs` resolves this to the class name via the registry, and
writes the **class name** (e.g. `"CoSAXSRawReader"`) into `beamtime.json`.

**Rationale:** The slug is the human-facing identifier for the CLI. The class name in the
config decouples the stored config from the CLI interface — the slug can change without
breaking existing config files, and the class name is unambiguous for `ReaderRegistry.get()`.
Storing the class name (not the slug) in the config means `from_config` doesn't need to
know anything about slugs.

---

## 013 — `can_read` returns False during init → warning, not error

**Decision:** If `raw_reader.can_read(root_path)` returns False during `beamtime init`,
emit a warning but still write the config.

**Rationale:** Data may not have arrived yet at the time the user initialises the beamtime.
Running `init` before data transfer is a valid workflow. The user explicitly chose the
beamline slug; they know what they're doing. A hard error here would be actively unhelpful.

---

## 014 — Sample names come from an elog sidecar, not from HDF5

**Decision:** `sample_name` in `RunMetadata` comes from `root_path/elog.csv`, not from
the raw HDF5 file. The HDF5 field `/entry/instrument/start_positioners/sample_description`
exists but is never populated by the CoSAXS acquisition system and must not be used.

**Rationale:** CoSAXS (and many other beamlines) do not write sample identity into the
data files. The beamline's own recommendation is to use an external logbook (Elogy). A
local CSV sidecar is the practical solution: it works offline, is version-controllable,
and is independent of network access to facility systems.

**Alternative considered:** Reading directly from the Elogy REST API at runtime. Rejected
because Elogy is only accessible on the MAX IV internal network, making offline
post-beamtime analysis impossible. A `beamtime import-elog` CLI command that normalises
the elog once and writes `elog.csv` gives the same data with no runtime network dependency.

---

## 015 — `list_runs()` returns all scans on disk; `sample_name=None` for unlabelled runs

**Decision:** `list_runs()` discovers all `scan-####.h5` master files and returns a
`RunMetadata` for each. Scans absent from `elog.csv` get `sample_name=None` — they are
not filtered out at the reader level.

**Rationale:** Two competing concerns:
1. Scans without an elog entry are typically junk (aborted scans, accidental triggers).
2. Silently hiding scans makes debugging confusing and loses discoverability.

The resolution is a two-tier approach: `list_runs()` returns everything; filtering happens
at the `select_runs()` level. Any `select_runs()` call that filters by sample name silently
skips runs with `sample_name=None`. Unlabelled runs remain accessible directly by scan ID
(`beamtime[48420]`). This is a strict superset of the alternative (filtering in
`list_runs()`) with no additional complexity at the reader level.

**`sample_name=None` is a first-class state**, not an error condition. It must be `None`,
not an empty string or a sentinel like `"unknown"`.

---

## 016 — Elog CSV has a canonical schema; `beamtime import-elog` is the only writer

**Decision:** `elog.csv` always has columns `scan_id` (int) and `sample` (str), plus
any additional columns from the source file preserved verbatim. The file is written only
by `beamtime import-elog`, never edited by hand.

**Rationale:** The source elog (Google Sheet export, Elogy export, etc.) has unstable
column names (e.g. `"scan id"` with a space, a datestamped export column from Google
Sheets). Normalisation happens once at import time, keeping the reader simple. The reader
calls `load_elog_csv()` from `io/elog.py` which assumes the canonical schema and raises
`ElogSchemaError` if it is violated — it never does fuzzy column matching at load time.

**Minimum required fields:** every row must have a non-null `scan_id` and a non-empty
`sample`. These are the minimum fields for any intentional scan, including calibrants.

---

## 017 — CoSAXS scan ID is parsed from the filename, not read from HDF5

**Decision:** The run ID for a CoSAXS scan is the integer parsed from the master filename
(`scan-48363.h5` → `48363`). No HDF5 field stores the scan number.

**Rationale:** The HDF5 content has no scan number field. The filename is the only
authoritative source. This is documented explicitly to prevent future implementors from
searching for an HDF5 equivalent.

---

## 018 — All CoSAXS start_positioners fields stored as Dataset attrs

**Decision:** All ~80 scalar fields from `/entry/instrument/start_positioners/` are stored
as attributes on the returned `xr.Dataset` using their original HDF5 key names, in addition
to the explicitly named convenience attrs (`photon_energy_ev`, `detector_distance_mm`, etc.).

**Rationale:** These fields are instrument state snapshots, not data. Storing them as attrs
(rather than data variables or coordinates) keeps the dataset dimensions clean. Preserving
all fields wholesale, rather than cherry-picking, avoids having to decide upfront which
positioners matter — different experiments care about different motors. The explicitly named
attrs are convenience aliases with normalised names for fields that analysis code will
commonly access programmatically.

---

## 019 — Duplicate scan IDs in import-elog: warn, keep first

**Decision:** If the source elog CSV contains duplicate `scan_id` values after expansion,
`beamtime import-elog` keeps the first occurrence, emits a warning to stderr, and continues.
It does not raise an error.

**Rationale:** During beamtime it is common to log a scan range (e.g. `48629-31`) and then
later add an individual entry for one of those scans with a corrected sample name. A hard
error would block import entirely. Keeping the first occurrence preserves the most explicit
entry (individual rows typically appear after ranges). The user sees the warning and can
re-import with `--force` after manually resolving the conflict.

---

## 020 — Scan-ID range cells are expanded by import-elog

**Decision:** `beamtime import-elog` expands range cells in the scan column before writing
`elog.csv`. Supported formats: `"48629-31"` (shared prefix, suffix increments) and
`"48639-48654"` (full range). Each expanded ID is written as a separate row with the same
sample name.

**Rationale:** Range notation is the natural way to log a batch of identical-sample scans
during beamtime. Rejecting these as non-integer values would force users to manually expand
ranges before import, which is error-prone. Expansion at import time means `elog.csv`
always has one row per scan ID and the reader never needs to handle ranges.

---

## 021 — Run loader injection via Beamtime._discover_runs()

**Decision:** `Run` objects do not hold a reference to their reader or root path. Instead,
`Beamtime._discover_runs()` creates a `raw_loader: Callable[[int | str, RunMetadata], xr.Dataset]`
closure per run and injects it at construction time. `run.load_raw()` takes no arguments.
The same pattern applies to `run.load_processed()` for facility-processed data.

**Rationale:** Keeps `Run` decoupled from reader internals. The loader signature includes
`metadata` so readers can use it (e.g. for path resolution) without needing a separate
`get_run_path` call. Symmetry between `load_raw()` and `load_processed()` makes the API
predictable.

---

## 022 — AnalysisResult as typed return from facility processed readers

**Decision:** `BaseFacilityProcessedReader.load_run()` returns an `AnalysisResult` subclass,
not a raw `xr.Dataset`. Each reader declares `result_class` as a class attribute.
`CoSAXSProcessedAzintReader.result_class = AzimuthalIntegration`.

**Rationale:** Facility-processed outputs have known, stable schemas (e.g. CoSAXS azint
always produces q/intensity/error). Typed accessors (`.q`, `.intensity`, `.error`) are
safer and more discoverable than string-keyed xarray variables. The base `AnalysisResult`
class exists as a fallback for readers where the schema is not yet modelled.

**`CoSAXSProcessedAzintReader` directory/file structure:** TBD — subject to change.

---

## 023 — Run is immutable; analysis state lives as optional fields on Run

**Decision:** `Run` is `@dataclass(frozen=True, eq=False)`. Derived analysis
state (currently `frame_mask`) lives as optional fields on `Run`. Analysis
operations that produce masked Runs return new instances via
`dataclasses.replace(...)` or the `run.with_mask(...)` helper. The source
Run is never mutated.

**Rationale:** Frame masks are Run-scoped; storing them on Run means they
travel through RunGroup set operations without a separate mapping.
Immutability prevents action-at-a-distance when the same Run is reached
via `beamtime[scan_id]` and via a filtered RunGroup. `eq=False` suppresses
auto-generated `__hash__`, which would clash with xarray DataArrays.

**Known tension:** Run was originally a thin delegate with only metadata
and loader closures. `frame_mask` is the first derived-state field on Run.
Further analysis-state fields should be scrutinized — not every derived
quantity belongs on Run.

---

## 024 — Frame filters are an MDAnalysis-style ABC, not RunGroup methods

**Decision:** Frame-level filtering is NOT a RunGroup method. It lives in
`pyBeamtime.analysis.filters` as a `BaseFrameFilter` ABC. Users instantiate
concrete filter classes (e.g. `CorMapKDEFilter(alpha=0.01)`) and call them
on a RunGroup. Filters return new RunGroups carrying frame masks on their
member Runs.

**Rationale:** Mirrors MDAnalysis's `AnalysisBase` pattern — group members
who want to add new analysis types subclass an obvious skeleton without
touching RunGroup internals. Keeps RunGroup lean; keeps filters individually
testable and composable. No plugin registry or auto-discovery — users
instantiate explicitly, as in MDAnalysis.

**Chaining:** Filters chain via AND of masks. A filter's `_compute_mask`
sees full (unmasked) data; the base class's `__call__` AND-s the returned
mask with any pre-existing `run.frame_mask` before packing into the output
RunGroup.

---

## 025 — RunGroup set ops deduplicate; `+` is NOT concatenation

**Decision:** RunGroup's `+`, `|`, `&` all perform set operations on scan_ids
with deduplication. For scan_ids present in both operands with different
`frame_mask` values, the operation raises `RunGroupMaskConflictError`. `+`
does NOT allow duplicates.

**Rationale:** MDAnalysis `AtomGroup +` concatenates-with-duplicates because
AtomGroup carries no derived state. Runs do (currently `frame_mask`,
potentially more in future). Duplicate scan_ids with divergent state is
nonsensical and almost always a bug — silent merging would produce wrong
analysis with no visible error. Raising is the safe default; explicit dedup
in user code handles the rare legitimate case.

**Deviation from MDAnalysis is documented explicitly** in `claude.md` and
`architecture.md` so users familiar with MDAnalysis aren't surprised.

---

## 026 — Frame-mask application is opt-in

**Decision:** `run.load_processed()` returns full, unmasked data regardless
of `run.frame_mask` state. Consumers that want to respect the mask must
read `run.frame_mask` explicitly and apply it. `RunGroup.average_frames()`
and `RunGroup.merge_detectors()` do this internally.

**Rationale:** Hidden masking inside a load method is the class of
action-at-a-distance bug that costs hours to debug. Opt-in keeps the mask
a visible, inspectable object that callers can AND, OR, invert, or
override without needing an "unmask" escape hatch.

---

## 027 — RunGroup tracks filter provenance

**Decision:** `RunGroup` carries an immutable
`filter_history: tuple[BaseFrameFilter, ...]` attribute. When a filter's
`__call__` produces a new RunGroup, it appends itself via
`RunGroup.with_filter_record(self)`. Filter instances are stored, not
serialized parameter dicts.

Set-op behavior:
- Equal histories on both operands → carry through unchanged.
- Divergent histories → resulting RunGroup has `filter_history = ()`
  and a `UserWarning` is emitted.

**Rationale:** Three weeks after a beamtime the first debugging question
is "what filters did this RunGroup go through?". Storing the instances is
~5 lines and makes that question trivial to answer. The conflict error for
divergent masks references this history, giving "you filtered the same
RunGroup two different ways" as a first-class explanation rather than
forcing the user to reconstruct it from notebook state.

Weaker set-op semantics than masks (drop-and-warn vs raise) because
divergent provenance on non-conflicting scan_ids is a legitimately
ambiguous annotation — not a correctness bug.

**Serialization deferred:** Filter instances live in memory only.
Persistence spec will address how to round-trip filter_history to/from
disk, which constrains the filter ABC (parameters must be serializable).

---

## 028 — Reduction persistence: masks-only, named sessions, dedicated zarr store

**Decision:** Reduction state is persisted to a single zarr store
(`user_processed_path / "reductions.zarr"`) containing named sessions.
Each session stores per-run `frame_mask` arrays and a serialized
`filter_history`. Averaged curves, merged curves, and other downstream
artifacts are NOT persisted — they are recomputed on demand.

API: `RunGroup.save_reduction(name)` / `Beamtime.load_reduction(name)` /
`Beamtime.list_reductions()` / `Beamtime.delete_reduction(name)`.

Independent `REDUCTION_SCHEMA_VERSION` constant, separate from
`SCHEMA_VERSION` in `zarr_store.py`.

**Rationale:** Masks are the expensive commitment (CorMap is O(n²·n_q)
per run; KDE peak selection is a human-reviewed choice). Averaging and merging
are cheap — recomputing from loaded masks is not a bottleneck. Saving
only masks keeps the persisted artifact small, easy to inspect, and
easy to version.

Named sessions (vs auto-derived from selection criteria) support "same
data, multiple filtering strategies" without collision and without
forcing the user to remember the exact `select_runs` criteria used
months earlier.

A dedicated store (not mixed with processed data) keeps schema evolution
independent. Processed-data schema changes and reduction-schema changes
happen on different cadences and for different reasons.

---

## 029 — Filters must be serializable for round-trip

**Decision:** `BaseFrameFilter` requires subclasses to either (a) be
`@dataclass` subclasses or (b) override `to_dict`, `from_dict`, and
`__eq__`. The base class provides default `to_dict` / `from_dict`
implementations that work for `@dataclass` subclasses.

Filter class identity is resolved on load via dotted import path
(`cls.__module__` + `__qualname__`). Filters defined in non-importable
contexts (ad-hoc in Jupyter) cannot round-trip — documented limitation.

**Rationale:** Round-trippable reduction state (decision 028) requires
reconstructing filter instances from disk. The `@dataclass` convention
gives `__eq__`, `__repr__`, and parameter serialization for free, at
the cost of a small stylistic constraint. Non-dataclass filters are
allowed but must do the work themselves.

`__eq__` matters for set-op "equal histories carry through" semantics
(decision 027) — without it, loaded filters won't compare equal to
freshly-constructed equivalents, silently breaking provenance checks
on combined RunGroups.

---

## 030 — `merge_saxs_waxs`: empirical scale factor + mask non-positive WAXS

**Decision:** Two bugs fixed in `analysis/merge.py`:

1. **`nanmean` included WAXS beamstop zeros.** `np.nanmean` only skips `NaN`.
   WAXS intensity at the low-q beamstop edge is zero (or very small positive),
   not `NaN`. Averaging SAXS (~5×10⁻⁴) with those zeros pulled the merged
   curve to roughly half the true value throughout the overlap region.
   Fix: set `<= 0` entries in the stack to `NaN` before `nanmean` so
   beamstop points are excluded and pure SAXS is used where WAXS is invalid.

2. **Solid-angle scale factor does not align intensities.** The geometric ratio
   `(pixel1·pixel2/dist²)_SAXS / (pixel1·pixel2/dist²)_WAXS` ignores detector
   efficiency, incident-flux normalisation, and filter/attenuator differences.
   A scale factor derived from geometry alone causes a step discontinuity at the
   overlap boundaries when the curves are not on the same absolute scale.
   Fix: replace with an empirical least-squares scale factor computed over valid
   (finite, positive) overlap points: `sf = Σ(I_saxs·I_waxs) / Σ(I_waxs²)`.
   The solid-angle ratio is retained as a fallback if the overlap contains no
   valid points.

**Rationale:** These bugs are consistent with the design direction already
captured in `reduction_module_spec.md` (R7: scale factor should not be a silent
geometric assumption; R8: prefer empirical/water-based normalisation; "What to
Cut": PONI arguments in `merge_saxs_waxs`). The empirical overlap fit is the
standard approach in SAXS/WAXS stitching software (BioXTAS RAW, ATSAS) and is
robust to the CoSAXS pipeline's variable solid-angle-correction state. The
current function signature (poni paths) is interim until the spec-compliant API
(`scale_factor` as an explicit required argument) is implemented.
