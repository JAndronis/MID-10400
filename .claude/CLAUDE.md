# CLAUDE.md -- Ferritin Crystallization at EuXFEL (proposal 10400)

This file initializes Claude Code for the analysis of a time-resolved X-ray
scattering beamtime on ferritin crystallization in acoustically levitated
droplets, measured at the MID instrument of the European XFEL in May 2026
(cycle 202601, proposal p010400). It summarizes the science, the as-run
experimental configuration, the software stack, and the target code
architecture, and it fixes the coding conventions to follow.

The end goal is a set of wrappers around the EuXFEL `extra-data`, `extra-geom`,
and `extra-speckle` libraries that (1) load, inspect, visualize, and manipulate
the data efficiently, and (2) expose processed data as numpy arrays or xarray
objects for arbitrary downstream processing and averaging. The wrappers must be
structured so they can eventually be merged into the separate `pyBeamtime`
framework as its EuXFEL reader plugin (see "Target architecture").

---

## 1. Scientific context

### System
Ferritin (not apo) (Sigma) at 50 mg/ml in 150 mM NaCl, plus polyethylene glycol (PEG)
as crystallizing agent, in acoustically levitated droplets of initial diameter
~1 mm. As the droplet dries, protein concentration rises, the solution
supersaturates, and crystals form and later dissolve.

The key polymer-size parameter is $\xi = R_g / a$ (PEG radius of gyration over
protein radius), which sets the phase-diagram topology (Ilett et al. 1995): for
$\xi \lesssim 0.25$ the polymer only widens fluid-crystal coexistence with no
stable dense-liquid phase; for $\xi \gtrsim 0.25$ a three-phase gas-liquid-crystal
region opens (LLPS possible). PEG 6000 gives $\xi \approx 0.40$ (LLPS possible);
PEG 1000 gives $\xi \approx 0.16$ (no LLPS).

### Central question
Can XPCS + XCCA resolve the dynamics and angular structure of the
aggregate-to-crystal transition in real time, and thereby confirm or refute the
continuous-order, desolvation-driven crystallization mechanism (Houben et al.
2020) under controlled droplet drying?

### Mechanistic scaffold
- Houben et al. 2020 (cryo-STEM tomography of ferritin): amorphous aggregates
  gradually order into fcc from interior to surface; no classical nuclei, no
  highly ordered small precursors; desolvation is the proposed driver.
- Sauter et al. 2015 (real-time SAXS of beta-lactoglobulin): two-step
  nucleation; a broad pre-Bragg SAXS peak (the intermediate) forms first and is
  consumed as Bragg peaks grow.
- Girelli et al. (in prep, CoSAXS/MAX IV; this group's immediate predecessor):
  kinetic phenomenology of the same levitated ferritin-PEG system; fcc Bragg
  peaks at $q \approx 0.6, 0.7, 1.0$ nm$^{-1}$; crystals form then dissolve near
  the drying turning point $t^*$; PEG molecular weight controls the pathway.

### Discriminating predictions (why XPCS/XCCA are needed)
- SAXS alone cannot separate phase separation from aggregation at low $q$. This
  ambiguity is the reason the experiment exists; do not let analysis code or
  narrative overclaim a mechanism from SAXS $S(q)$ alone.
- XCCA: a disordered-dense-fluid (two-step, Sauter-like) intermediate predicts a
  sudden onset of angular correlations; a continuously ordering (Houben)
  intermediate predicts a gradual rise. This is the key experimental
  discriminator.
- XPCS: transient high-density-fluid dynamics inside a spinodal would appear as
  slowing dynamics preceding crystallization.

---

## 2. As-run experimental configuration

Confirmed parameters for this beamtime (use these; do not infer from the
proposal draft, which differs):

- Facility / instrument: European XFEL, MID. Cycle 202601, proposal 10400.
- Photon energy: 9.04 keV (wavelength ~1.371 A).
- Geometry: transmission.
- SAXS/XPCS/XCCA detector: AGIPD-1M at 7.531 m from the sample.
- WAXS detector: two Jungfrau-500K modules placed close to the sample, above the
  beam and normal to it. A pyFAI PONI file for the Jungfraus already exists and
  is the authoritative WAXS geometry source.
- Sample delivery: acoustic levitator; microscope perpendicular to the beam for
  droplet sizing / concentration estimation.
- Samples of interest: ferritin 50 mg/ml + PEG 6000 (the PEG 1000 samples did
  not crystallize and are out of scope). No PEG 3000/3300 was measured.
- Reference: dilute ferritin at 10 mg/ml in water, in capillaries (for the
  protein form factor $P(q)$), collected as planned.

### Physics landmarks (for reduction and sanity checks)
- fcc protein Bragg peaks near $q \approx 0.6, 0.7, 1.0$ nm$^{-1}$.
- Aggregate / monomer XPCS split near $q \approx 0.1$ nm$^{-1}$ (aggregates
  below, single proteins above).
- WAXS: water peak near $q \approx 20$ nm$^{-1}$; PEG semi-crystalline peak near
  $q \approx 13.7$ nm$^{-1}$ (appears after protein-crystal melting, so it is a
  consequence, not a cause).
- Diagnostic for XPCS relaxation: $\Gamma \propto q^2$ indicates diffusion;
  $\Gamma \propto q$ indicates shear/flow (e.g. acoustic streaming across the
  finite beam). Do not assume diffusion; test the $q$ scaling.
- XCCA ROIs deliberately exclude the fcc Bragg positions and focus on pre- and
  during-crystallization angular order; Bragg-peak evolution and dissolution are
  handled by SAXS azimuthal integration and WAXS instead.

---

## 3. Data location and access

- Root: `/gpfs/exfel/exp/MID/202601/p010400`
- Work from with facility processed (dark and flatfield corrected) `proc` data.
  This directory just has calibrated and compressed data with reduced metadata per run,
  but it has to still be further reduced through analyses like SAXS or XPCS.
- Run table: a user-provided CSV maps runs to samples. Column names are
  `Run Number` (run id, integer) and `Sample Name`. This is imported into
  pyBeamtime's canonical `elog.csv` via `beamtime import-elog <csv>
  --scan-col "Run Number" --sample-col "Sample Name"`.
- Pulse pattern, repetition rate, sample-detector distance, and AGIPD quadrant
  geometry are read from run metadata at load time (extra-speckle's `Setup`
  auto-pulls these; `utils.geometry_from_encoders` reads AGIPD motor encoders
  from `MID_AGIPD_MOTION/MDL/DOWNSAMPLER`). Do not hard-code them.

Concrete source/key names (AGIPD source, Jungfrau sources, XGM, transmission
diodes, levitator, microscope triggers) are NOT assumed in this file. Resolve
them at implementation time from the actual run via `lsxfel` and
`run.all_sources` / `run.info()`, and record the resolved names in code as
named constants with a comment pointing to the run they were verified against.

---

## 4. Software stack and the "do not reinvent" mandate

Use tools that already exist in the EuXFEL ecosystem. The wrapper's job is
adaptation and orchestration, not reimplementation.

### extra-data (I/O layer)
`open_run(proposal, run, data='raw') -> DataCollection`; three-tier object model
`DataCollection -> SourceData -> KeyData`; `KeyData` exposes `.ndarray()`,
`.xarray()`, `.dask_array()` natively. Multi-module detector components
`extra_data.components.AGIPD1M`, `JUNGFRAU`. Train-based selection/iteration
(`select`, `select_trains`, `trains()`, `by_id`, `by_index`). This is the base
I/O layer; lazy dask-backed xarray output aligns with the target design.

### extra-geom (geometry / assembly)
`AGIPD_1MGeometry` (16 modules x 8 tiles; `from_crystfel_geom`,
`from_quad_positions`) and `JUNGFRAUGeometry`. Assembly via `position_modules*`;
`inspect()` / `plot_data()` for visualization; `to_pyfai_detector()` /
`to_distortion_array()` for pyFAI interop; `AGIPD_1MMotors` for motor-driven
geometry. `agipd_asic_seams()` for ASIC-boundary masks.

### extra-speckle (reduction: XPCS, XCCA, SAXS, XSVS)
Built for EuXFEL burst mode and AGIPD-1M at MID; reference: Leonau et al.,
arXiv:2506.08668. Central object is `Setup` (Detector + Configuration +
experimental metadata; builds `setup.config.ai`, the pyFAI integrator).
Unified `.get / .fit / .plot` per technique. Key entry points:
- `saxs.get(data, setup) -> xr.DataArray` (azimuthal integration).
- `xpcs.get(data, setup, data_off=...)` (in-memory TTCF for a chunk) and
  `xpcs.calc_xpcs_agipd(run, setup, ...)` (full-run, parallel, cached; sparsify
  threshold ~0.05, ~50 workers, RAM-limited; emits `TTCF_raw.nc`, `TTCF_off.nc`,
  `kbar.nc`). Kernels: `TTCFCalculator`, `TTCFSparseCalculator`.
- `xpcs.tools.reduce_ttcf`, `xpcs.fit(g2, model='exp', beta='local'|'global',
  dr=True)` (single exponential + Stokes-Einstein diffusion fit).
- `xcca.calc_xcca_agipd(run, setup, npt_azim, npt_rad, ...)` and
  `corr.get_angular_cc` (FFT / Wiener-Khinchin).
- `pipeline.xpcs_offline(...)` (alias `saxs_xpcs_offline`): the "run everything"
  orchestrator (~50 kwargs) returning a dict of intermediates and figures.

### KNOWN ISSUES in extra-speckle -- read before relying on any of this
- XCCA has NO public API: `xcca/__init__.py` has `.get/.fit/.plot` all commented
  out. Only the internal `calc_xcca_agipd` / `corr.get_angular_cc` /
  polar-mapping work. The GPU worker path looks experimental; sanity-check it
  before use. XCCA is a primary technique here, so the wrapper must reach into
  these internals deliberately and wrap them cleanly.
- `pipeline.xpcs_offline` is hard-restricted to `detector == 'AGIPD 1M'` (raises
  otherwise). WAXS on the two Jungfrau-500K must therefore go through the
  lower-level `saxs.get` with a separate PONI-based `Setup`, NOT the pipeline.
- KWW / stretched-exponential $g_2$ fitting is a no-op (`model='str'` just
  prints). Double-KWW $g_2$ analysis will need custom fitting code (extend
  `xpcs/fit.py` or implement separately as pure functions).
- `xsvs.get` is a placeholder stub; the XSVS branch of `xpcs_offline` has a
  wiring gap (undeclared kwargs) and will raise `NameError` if invoked. XSVS is
  not a primary technique here.
- No automated tests in the package. This raises, not lowers, the bar on our own
  unit-test discipline.
- `_deprecated` variants coexist with live functions
  (`calc_single_module_deprecated`, `reduce_ttcf_rolling_deprecated`); grep for
  `deprecated` to avoid the dead path.

---

## 5. Target architecture (fits pyBeamtime)

The wrappers drop into `pyBeamtime` as its EuXFEL reader plugin and analysis
adapters. The pyBeamtime `BaseRawReader` / `Beamtime` source has been read; the
ABC contract below is verified, not inferred. Still read `core/run.py` (for the
exact `RunMetadata` field order) and `io/readers/__init__.py` /
`io/readers/maxiv.py` (for the registration idiom) before implementing, and do
not implement any extra-speckle adapter without reading the relevant function.

### 5.1 Data-access model -- LOCKED: two-tier

extra-speckle's correlators do not consume a pre-assembled xarray Dataset; they
stream per-module from an extra-data run handle + `Setup`, which is what makes
MHz XPCS on AGIPD-1M memory-tractable. Forcing XPCS through a materialized
`(train, pulse, 16, 512, 128)` Dataset would fight that and blow up memory. The
model is therefore two-tier (agreed, not a default):

- Tier 1 -- loading / inspection / visualization / SAXS. The EuXFEL raw reader
  `load_run` returns a lazy, dask-backed `xr.Dataset` built from extra-data
  (+ extra-geom for assembly on demand). This is goal (1) and the first
  priority.
- Tier 2 -- XPCS / XCCA. Thin orchestrators in `analysis/` that take the same
  extra-data run handle and a `Setup` and delegate to extra-speckle
  (`calc_xpcs_agipd`, `calc_xcca_agipd`, `saxs.get`), returning results as
  xarray (goal 2). No reimplementation of correlators. These do NOT go through
  the reader (consistent with the "readers only read" decision), so `load_run`
  never needs to materialize the full detector.

### 5.2 EuXFEL reader plugin (implementation spec for Claude Code)

New file `io/readers/euxfel.py`, class `EuXFELMIDRawReader(BaseRawReader)`. This
is the spec to implement from; there is no separate draft module. Keep path
parsing in module-level pure functions so they unit-test without extra-data
installed.

Verified ABC contract (from the real `base.py` / `beamtime.py`):
- `BaseRawReader` has FOUR abstract methods: `can_read`, `list_runs`,
  `get_run_path`, `load_run`. `get_run_path` IS required (an earlier note that
  it had been removed was wrong; `architecture.md` is correct, the `claude.md`
  ABC summary omitted it, and `Beamtime.enrich_metadata` calls it).
- Class attrs to set: `slug = "mid"`, `beamline = "MID"`,
  `facility = "European XFEL"`, `priority = 0`, `paired_facility_reader = None`.
  `Beamtime.facility` / `.beamline` read these off the reader.
- `load_run(self, run_id, root_path)` receives `(run_id, root_path)` ONLY.
  `Beamtime._discover_runs` builds `raw_loader(run_id, metadata)` but calls
  `reader.load_run(run_id, root_path)` -- the reader never sees `metadata`
  (decision 021's "readers use metadata for path resolution" is not what the
  code does). Path resolution is `get_run_path`'s job, from `(run_id, root_path)`.

Module-level pure helpers:
- `parse_run_dir(name) -> int | None`: `"r0042" -> 42`, else `None`.
- `proposal_number_from_root(root_path) -> int`: parse `p010400 -> 10400` from
  the root basename; raise on mismatch. Needed only for Dataset provenance and
  the extra-speckle `Setup`, not for path resolution.
- `list_run_dirs(raw_path) -> list[tuple[int, Path]]`: enumerate `r####` dirs,
  sorted by run id; filesystem only, opens nothing.

Method logic:
- `can_read(root_path)`: two-layer sentinel -- `(root_path/"raw").is_dir()` and
  `any(root_path.glob("raw/r*/RAW-*.h5"))`. VERIFY the glob against the real
  p010400 tree before trusting it (`RAW-*.h5` is the standard per-module name).
- `get_run_path(run_id, root_path) -> Path`: return the run DIRECTORY
  `root_path/"raw"/f"r{int(run_id):04d}"`. A run is a directory of many
  per-module files, not one file, so this deliberately deviates from the ABC's
  "primary HDF5 file" wording. CONSEQUENCE: the generic `Beamtime.enrich_metadata`
  (raw `h5py.File` on this path) does NOT work for EuXFEL; EuXFEL metadata
  enrichment must go through extra-data. This is a ratified deviation, not an
  oversight -- do not "fix" it by returning an arbitrary aggregator file.
- `list_runs(root_path) -> list[RunMetadata]`: `load_elog_csv(root_path)` gives
  `{scan_id: {"sample": ..., **extra}}`; for each `(run_id, _)` from
  `list_run_dirs`, build a `RunMetadata` (see field semantics in 5.2.1). Runs
  absent from the elog get `sample_name=None` (first-class state; not `""`, not
  a run-level filter). Non-`sample` elog columns pass through into
  `RunMetadata.extra`.
- `load_run(run_id, root_path) -> xr.Dataset`: open via
  `extra_data.RunDirectory(get_run_path(run_id, root_path))` (uses `root_path`
  directly, avoids `find_proposal` ambiguity; `open_run(proposal, run_id,
  data="raw")` is the alternative). Parse the proposal for provenance attrs.
  Return the Tier-1 Dataset of 5.3. BLOCKED until 5.3 is ratified and the
  extra-data source/key names are verified from a real run -- until then this
  raises `NotImplementedError` naming those two blockers.

Registration: `ReaderRegistry.register(EuXFELMIDRawReader)` at module bottom, and
`io/readers/__init__.py` must import `euxfel` (as it imports `maxiv`) so
`from_config` can resolve `"EuXFELMIDRawReader"`. If `maxiv.py` registers via a
decorator, match that idiom instead.

Config (`beamtime.json`): `raw_reader = "EuXFELMIDRawReader"`,
`facility_processed_reader = null` (reprocessing from `raw`),
`facility_processed_path = null`. Note `beamtime init` resolves `--beamline mid`
to the class name via the slug.

Elog import: auto-detect will NOT match the user's columns (`Run Number`,
`Sample Name` are not in the recognised variant set), so the command must be
explicit: `beamtime import-elog <csv> --scan-col "Run Number" --sample-col
"Sample Name"`.

#### 5.2.1 RunMetadata field semantics (EuXFEL) -- RESOLVED

The `RunMetadata` fields are CoSAXS/Eiger-shaped; their EuXFEL meaning is fixed
as follows (construct via keyword args; confirm field order against
`core/run.py`):
- `scan_id` (int): the run number, parsed from the `r####` directory name (there
  is no HDF5 field for it).
- `sample_name` (str | None): from `elog.csv`; `None` if the run is unlisted.
- `n_frames` (int | None): the number of PULSES IN A TRAIN. (Not trains, not
  trains x pulses.)
- `exposure_time` (float | None): the individual PULSE WIDTH. Genuinely ambiguous
  for a charge-integrating detector like AGIPD, so the pulse width is the agreed
  convention.
- `start_time` / `end_time` (datetime | None): best-effort; left `None` for now
  (populating them needs a per-run extra-data open with as-yet-unverified
  `run.run_metadata` keys). May be filled later.
- `extra` (dict): elog pass-through columns PLUS an `n_trains` field (number of
  trains in the run). `n_trains` is the EuXFEL-specific count that has no place
  in the fixed fields.

Concrete anchor (run r0500, from `lsxfel`): `n_frames = 352` ("352 frames per
train"), `extra["n_trains"] = 3003`, duration ~5 min. `start_time` / `end_time`
are derivable here (the run reports first/last train IDs and a duration, and
`MID_RR_UTC/TSYS/TIMESERVER` is present), so they can be upgraded from `None`
once the extra-data timestamp accessor is confirmed.

### 5.3 Tier-1 Dataset schema (EuXFEL) -- SOURCES AND KEYS VERIFIED (r0500)

The EuXFEL analog of decision 018. SOURCE and KEY names below are both verified
from run r0500 (`lsxfel` + a `run[source].keys()` pass). These are the concrete
strings the reader uses; no guessing remains.

Verified sources and keys (run r0500):

```
AGIPD-1M (SAXS/XPCS/XCCA):  MID_DET_AGIPD1M-1/DET/{0..15}CH0:xtdf
    keys image.data / image.cellId / image.pulseId / image.trainId (+length,
    status). Access via extra_data.components.AGIPD1M; 16 x 512 x 128; 352
    frames/train.
Jungfrau-500K x2 (WAXS):    MID_EXP_JF500K1/DET/JNGFR01:daqOutput
                            MID_EXP_JF500K2/DET/JNGFR02:daqOutput
    keys data.adc (raw ADU), data.gain, data.memoryCell, data.frameNumber,
    data.bunchId, data.trainId. RAW: quantitative WAXS needs gain-aware JF
    calibration (pedestal/gain), not data.adc alone.
XGM (incident flux):        SA2_XTD1_XGM/XGM/DOOCS:output
    key data.intensityTD (per-pulse train-resolved flux); data.xTD/yTD beam
    position; data.intensitySigmaTD.
Lit-frame finder:           MID_EXP_AGIPD1M1/REDU/LITFRM:output
    keys data.dataFramePattern, data.detectorPulseId, data.xgmPulseId,
    data.nPulsePerFrame, data.energyPerFrame, data.nFrame. Gives lit-frame
    selection AND the XGM-pulse <-> AGIPD-frame mapping needed for per-pulse
    normalization -- do not attempt to align XGM to AGIPD without it.
Bunch pattern:              MID_RR_UTC/TSYS/TIMESERVER:outputBunchPattern
    key data.bunchPatternTable; decode with euxfel_bunch_pattern for X-ray
    pulse IDs and rep rate. Cross-check MID_RR_SYS/MDL/PULSE_PATTERN_DECODER.
Droplet camera (CONTROL):   MID_EXP_CAM/PROC/DROPLET_DOWNSTREAM
    control keys current_vol (fitted droplet volume), callibration_known_volume,
    current_pos_x/y, roi_*. This is the droplet path-length / concentration
    observable -- a control-rate value, NOT a per-pulse instrument stream, and
    NOT a transmission diode (none exists). Raw video: MID_EXP_SAM/CAM/CAM{1,5}
    :daqOutput (data.image.*).
AGIPD quadrant encoders:    MID_EXP_AGIPD1M/MOTOR/Q{1-4}M{1-2}
    (note: NOT the MID_AGIPD_MOTION/MDL/DOWNSAMPLER that extra-speckle's
    geometry_from_encoders expects -- see open decision on geometry).
Attenuator/transmission:    SA2_XTD1_ATT/MDL/MAIN key actual.transmission
    (realized transmission; target.transmission / desiredTransmission also
    present). Relevant to the 3-transmission scheme.
Beamstop:                   MID_EXP_SAM/MOTOR/BST{X,Y,Z} (mask the shadow).
Photon energy:              MID_OPT_MONO/MDL/ENERGY_CHANGER or
    MID_XTD1_UND/DOOCS/ENERGY (known: 9.04 keV).
```

Proposed Dataset (dims / vars / attrs):

```
Dimensions:
    train        - train index
    pulse        - AGIPD frame (pulse/cell) index within a train (352 for r0500)
    module       - AGIPD module (16)
    ss, fs       - AGIPD slow-/fast-scan pixel within a module (512, 128)
    jf_ss, jf_fs - Jungfrau pixel dims (512, 1024) per unit
    jf_cell      - Jungfrau storage-cell index (train-wise)

Data variables (lazy, dask-backed):
    agipd        (train, pulse, module, ss, fs)  AGIPD1M comp.; image.data
    jf500k1      (train, jf_cell, jf_ss, jf_fs)  JNGFR01 daqOutput data.adc
    jf500k2      (train, jf_cell, jf_ss, jf_fs)  JNGFR02 daqOutput data.adc
    xgm_flux     (train, pulse)                  XGM data.intensityTD
    litframe     (train, pulse)                  LITFRM data.dataFramePattern
    (no i_transmitted -- no diode; droplet volume MID_EXP_CAM/PROC/
     DROPLET_DOWNSTREAM.current_vol is a CONTROL value, carried in attrs/coords
     at control rate, not a per-pulse variable)

Coordinates:
    train_id     (train,)
    pulse_id     (pulse,)   from AGIPD image.pulseId
    cell_id      (pulse,)   from AGIPD image.cellId
    xgm_pulse_id (pulse,)   from LITFRM data.xgmPulseId (XGM<->AGIPD alignment)
    timestamp    (train,)   from TIMESERVER

Attributes (xr.Dataset.attrs):
    proposal              10400
    run                   int
    facility              "European XFEL"
    beamline              "MID"
    photon_energy_ev      9040.0
    wavelength_A          ~1.371
    detector_distance_m   7.531
    sample_name           str | None
    rep_rate_hz           from bunch pattern
    n_pulses_per_train    352 for r0500  (== RunMetadata.n_frames)
    n_trains              3003 for r0500 (== RunMetadata.extra["n_trains"])
    transmission          from attenuator source
    <selected control snapshot>  beamstop XYZ, quadrant motors, mono energy
```

Assembly to a physical image is on demand via extra-geom, not stored in the
Dataset. Keep raw per-module data lazy; never eagerly materialize full AGIPD
arrays. Tier-2 XPCS/XCCA do NOT consume this Dataset -- they take the
`RunDirectory` handle + `Setup` directly (5.1).

### 5.4 Analysis adapters (Tier 2)

Wrap extra-speckle, do not reimplement. Return xarray. Suggested homes
(consistent with the pyBeamtime `analysis/` layout, marked `[to add]` there):
- SAXS/WAXS azimuthal integration -> adapter over `saxs.get` (+ separate
  PONI-based `Setup` for the Jungfrau WAXS).
- XPCS correlation -> adapter over `xpcs.calc_xpcs_agipd` +
  `reduce_ttcf`; custom KWW/double-KWW $g_2$ fitting lives here (pure functions),
  since extra-speckle lacks it.
- XCCA -> adapter over `calc_xcca_agipd` / `corr.get_angular_cc`, providing the
  clean public surface extra-speckle is missing.

### 5.5 Masking and normalization (later, but design for it now)

- Masking: AGIPD tile gaps and bad pixels are always masked. Additional custom
  masks (beam shadows, unwanted features) will be drawn interactively (e.g.
  pyFAI-calib2) and loaded as boolean arrays. Design mask handling as
  composable boolean `xr.DataArray`s so hand-drawn masks AND with the automatic
  ones.
- Normalization: per-pulse XGM for incident intensity, plus droplet path-length
  correction (the levitated droplet evaporates, changing thickness). This is a
  later priority; keep the loader normalization-agnostic and expose the raw XGM
  and transmission quantities so normalization is a downstream pure function.

---

## 6. Coding standards and workflow

- Package manager: `uv`. Project is `pyproject.toml`-based.
- Formatting/linting: run `uvx ruff format` and `uvx ruff check` and keep code
  clean at all times.
- Python: full type annotations; follow PEP 8 / PEP 257. Target Python >=3.10
  (matches extra-geom).
- Structure: prefer small, modular, pure functions to enable unit testing. Side
  effects (I/O, caching, plotting) isolated at the edges.
- Testing: write unit tests for pure functions. Because extra-speckle ships no
  tests, treat the boundary with it as untrusted and test the adapters
  (input/output shapes, dims, coords, attrs) with mock or small real chunks.
- Comments: only for non-obvious algorithmic choices (e.g. why a sparsify
  threshold, why a particular $q$ scaling test). Do not narrate obvious code.
- Prose style for ALL code prose -- inline comments, module/file-top comments,
  and function/class docstrings alike. Write every one for a reader who has
  NEVER seen this `.claude` directory: the `.claude` files are working notes for
  the assistant, not user documentation. Two consequences:
  1. No conversational tone or meta-references to how the code came to be. Never
     mention that a choice was discussed, requested, agreed, ratified, that
     someone held an opinion, or that a point was raised -- state what the code
     does and why, impersonally. (E.g. write "Returns the run directory; a run
     is a directory of per-module files, so this is not a single HDF5 file."
     NOT "As we decided, this deliberately deviates from the ABC because you
     pointed out a run is a directory.")
  2. Be self-contained. Do not lean on shorthand defined only in these
     directives ("Tier 1/Tier 2", "decision 018", "the two-tier model", "as-run
     config", bare landmark numbers). Either spell the concept out in prose or
     omit it. A term is fine only if it is standard in the field (SAXS, XPCS,
     XCCA, AGIPD, train, pulse) or defined elsewhere in the same module's public
     API.
- Output types: xarray throughout (extra-data already exposes `.xarray()` /
  `.dask_array()`), so downstream averaging/processing stays in xarray.
- Do not hand-roll HDF5 paths, calibration, or geometry math that the `extra-*`
  packages already provide.

### Hard rule on unread code
Never guess or speculate about unread files or module internals. Before writing
or modifying a module, read the relevant existing source (pyBeamtime module,
extra-speckle function, notebook example). Request or open the real structure
first; do not implement against this summary alone.

---

## 7. First-priority scope

The immediate deliverable is Tier 1: the EuXFEL raw reader
(`io/readers/euxfel.py`) producing the lazy `xr.Dataset` above, plus
loading / inspection / visualization wrappers over extra-data and extra-geom
(open a run, list sources, assemble and plot a frame, select trains/pulses,
pull control channels). XPCS/XCCA orchestration (Tier 2), masking, and
normalization come after, in that rough order.

Initial parameter placeholder (to be revisited): XPCS/XCCA $q$-binning is not
yet fixed; start with 10 $q$-bins.

---

## 8. Open decisions (resolve explicitly; do not silently default)

Resolved (recorded here so they are not re-litigated): data-access model is
two-tier (5.1); reader slug is `mid`, no `paired_facility_reader` (reprocess from
`raw`); `RunMetadata` field semantics are fixed (5.2.1: `n_frames` = pulses per
train, `exposure_time` = pulse width, `n_trains` in `extra`); `get_run_path`
returns the run directory with the `enrich_metadata` deviation documented (5.2).

Still open:
1. AGIPD geometry source of truth. The encoder source extra-speckle expects
   (`MID_AGIPD_MOTION/MDL/DOWNSAMPLER`) is ABSENT in r0500; the quadrant motors
   are `MID_EXP_AGIPD1M/MOTOR/Q{1-4}M{1-2}`. So `geometry_from_encoders` likely
   will not work unmodified -- lean toward pinning a `.geom`/quad file, or verify
   whether the installed extra-speckle reads these motors. Jungfrau WAXS uses the
   existing PONI.
2. XPCS $q$-binning (start 10 bins), XCCA ROI list and $\Delta\phi$ sampling,
   Bragg-exclusion windows.
3. Custom $g_2$ model form (single vs double KWW) and where the fitting code
   lives (extra-speckle has no KWW).
4. Normalization recipe: per-pulse XGM (`data.intensityTD`) aligned to AGIPD
   frames via LITFRM `data.xgmPulseId`, times droplet path length from
   `DROPLET_DOWNSTREAM.current_vol`. No transmission diode exists. Plus the
   mask-composition convention (AGIPD gaps + beamstop shadow + hand-drawn).
5. Jungfrau WAXS calibration: raw `data.adc` needs gain-aware pedestal/gain
   correction before quantitative use; decide whether to apply EuXFEL JF
   calibration constants via extra-* or defer.

---

## 9. Do NOT

- Do not reimplement TTCF/g2, azimuthal integration, or angular
  cross-correlation; delegate to extra-speckle.
- Do not force XPCS/XCCA through a materialized full-detector Dataset.
- Do not hard-code pulse pattern, rep rate, energy, distance, geometry, or file
  paths; read them from metadata / `extra-*`.
- Do not use `pipeline.xpcs_offline` for Jungfrau WAXS (AGIPD-1M only).
- Do not use `model='str'` XPCS fitting or `xsvs.get` expecting real results.
- Do not overclaim mechanism from SAXS $S(q)$ alone; the phase-separation vs
  aggregation ambiguity is the scientific reason XPCS/XCCA are being measured.
- Do not implement any module against this file's summaries without first
  reading the real source.
