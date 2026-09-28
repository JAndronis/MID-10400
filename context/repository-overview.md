# MID p010400 analysis repository — structure and implemented analysis

A standalone description of this repository: what the code is, how the source tree is organised,
and what analysis it currently performs. Written to be pasted into a conversation as background,
so it assumes no access to the files themselves. It describes the code as it stands on
**2026-09-15**; it deliberately carries no open questions or pending work, only what exists.

---

## 1. What this repository is

Analysis code for **European XFEL proposal 10400** at the **MID** instrument (cycle 202601):
time-resolved SAXS/WAXS of **ferritin crystallisation in ultrasonically levitated droplets**
(~1 mm). Ferritin (50 mg/ml, 150 mM NaCl) crystallises in the presence of PEG 6000; the droplet
evaporates over the measurement, so the run is a kinetic series and each train is a time point.

The scientific question is which route crystallisation takes — continuous gradual ordering, or
two-step nucleation through a disordered dense intermediate. Answering it needs per-frame SAXS and
WAXS intensity curves for every run, then angular cross-correlation (XCCA) and X-ray photon
correlation spectroscopy (XPCS) on the runs those curves select.

**What this repository implements is the first of those: the two detector integration passes**, one
for SAXS on the AGIPD-1M and one for WAXS on the two JUNGFRAU-500Ks. XPCS is run through the
facility's own `extra-speckle` pipeline, driven from this repository's DAMNIT context file
(§3.5); XCCA uses the facility's toolbox and has no pass of its own here.

The repository is a **uv project** with a src layout. Everything in it is either one of those two
passes, the glue that runs them, a multi-facility data-reader plugin, or tests and acceptance
scripts for those. The glue is **DAMNIT**, the facility's per-run processing and summary-table
service, running on the Maxwell HPC cluster: it evaluates a set of declared "variables" for every
run and shows them as a table with previews.

### Instrument and data at a glance

| Item | Value |
|---|---|
| Facility path | `/gpfs/exfel/exp/MID/202601/p010400/` (`raw/`, `proc/`, `usr/`, `scratch/`) |
| Photon energy | 9.04 keV — set in both passes' configs, and what fixes the wavelength and so the q scale |
| SAXS detector | AGIPD-1M, 16 modules of 512×128, at 7.531 m; q ≈ 0.08–1.07 nm⁻¹ |
| WAXS detectors | two JUNGFRAU-500K ("jf1", "jf2"), each 512×1024, at 232 mm; after masking jf1 covers q ≈ 11.5–23.7 nm⁻¹ and jf2 ≈ 9.8–18.5 nm⁻¹ |
| Pulse structure | ~1.128 MHz; the pattern varies per run (155 and 350 pulses/train both occur) |
| A long run | ~3000 trains at 155 AGIPD frames per train ⇒ ~465 000 frames to integrate |
| AGIPD proc data | `int16` **photon counts**, ~99 % zeros (0.7–1.8 % of pixels non-zero per frame) |
| JUNGFRAU proc data | `float32` **keV**, dense; 16 memory cells per train, of which typically 8 are lit |

Both detectors' corrected ("proc") files carry a per-frame `mask` — a uint32 `BadPixels`
bitfield from the calibration pipeline — alongside the data.

---

## 2. Repository layout

```
CLAUDE.md               working notes for the repository (beamtime facts, environment, pitfalls)
README.md               setup, how to run the passes and the tests
pyproject.toml          uv/hatchling project; deps pinned in uv.lock
context/                design guides (this file; the two integrator guides; an EXtra toolkit guide)

src/analysis/           the analysis package — the only heavy code
    common/             detector-agnostic layer shared by both passes
        status.py       FrameStatus codes + DataCheckFailed
        masks.py        mask vocabulary (MaskSource, StaticMask, frame_bad, bit helpers)
        plan.py         row model: TrainRecord, Block, RunPlan, build_blocks, evenly_spaced
        config.py       config hashing, by-name pickling, shared config members
        writer.py       FrameTableWriter: HDF5 layout, ledger, resume, pooling reducers
        run.py          worker fan-out (fan_out) and the shared provenance record
        cpu.py          physical-core counting, spawned pool, sha256, timing phases
        arrays.py       frozen_copy, relative_difference
        pyfai.py        guards that check which pyFAI engine actually ran
    saxs/               AGIPD SAXS pass  (DAMNIT variable `agipd_saxs`)
    waxs/               JUNGFRAU WAXS pass (DAMNIT variables `jungfrau_waxs_*`)
    threadenv.py        thread pinning; a leaf module with no imports of its own

src/readers/            pyBeamtime reader plugin
    io/readers/euxfel.py   EuXFELMIDRawReader — a run directory to a lazy xarray Dataset

src/amore/              DAMNIT deployment directory (deliberately NOT packaged)
    context.py          the DAMNIT variable definitions
    analysis_helpers.py droplet ellipse fitting and volume tracking from camera images

scripts/                acceptance and probe scripts, run by hand on a cluster node
tests/                  pytest suite (~480 tests), split common/ saxs/ waxs/
notebooks/              exploratory notebooks (damnit, dask, waxs, extra-speckle masking)
data/                   local exported detector slices and mask files for offline work (untracked)
extern/pybeamtime/      local editable checkout of the pyBeamtime dependency
```

Only `src/analysis` and `src/readers` are packaged into the wheel. `src/amore` mirrors the DAMNIT
context directory on the cluster: DAMNIT `exec`s that context file into a dict, so functions
defined there cannot be pickled to spawned worker processes — which is why every DAMNIT variable
body is just an import plus a call into `analysis.*`.

---

## 3. The analysis that is implemented

Two integration passes, one per detector family. They share a design, a shared layer
(`analysis.common`) and an output-file format, and differ in the integration kernel, the masking
and the error model.

### 3.1 What both passes do

For one run, read every detector frame from the corrected (`proc`) data, integrate it azimuthally
into `npt` q bins (500 by default, and a config field like everything else), and store **per-q-bin sufficient statistics per frame** rather than an
averaged I(q):

```
signal        S(q) = Σ  c · x        (numerator)
normalization N(q) = Σ  c · Ω        (denominator, over the pixels actually kept)
variance      V(q)                   (per-frame variance in the same units)
```

so any later grouping — per train, per pulse, per cell, per time window, per bin of droplet
volume V/V₀ (§3.5) — pools
exactly:

```
I(q) = Σ_G S / Σ_G N          σ(q) = sqrt(Σ_G V) / Σ_G N
```

This is why nothing in the hot loop applies XGM normalisation, transmission or background
subtraction: those are post-hoc operations on the stored sums, and dividing frames in place before
integration would be irreversible.

The rest of the shared design:

- **Every row is addressed by label, never by array position.** A frame's train id and pulse/cell
  id come from the reader's own coordinate arrays. A train whose labels or frame count disagree
  with the plan is marked `LABEL_MISMATCH` and not integrated, rather than having its frames slide
  onto a neighbouring train.
- **A per-frame status ledger, with no NaN sentinels.** A frame that was not integrated carries a
  status code and zeros, so the code is the only thing distinguishing "integrated to zero" from
  "never integrated", and any pooling selects on it. The codes, as stored in `frames/status`:
  `OK = 0` (integrated), `MISSING_MODULES = 1` (the train did not write every detector module),
  `NO_FRAMES = 2` (the train has no detector frame at all), `LABEL_MISMATCH = 3` (the frames' own
  labels disagree with the plan, so they cannot be placed), `DATA_CHECK_FAILED = 4` (the frame's
  values are not what the pass is entitled to assume — a non-integer or negative count for AGIPD,
  a frame with every pixel excluded for JUNGFRAU), `WORKER_ERROR = 5` (the block raised; the run
  continued and recorded the exception), `NOT_PROCESSED = 255` (the fill value, still untouched —
  what a killed pool leaves behind).
- **A self-test gate before the worker pool starts.** Each pass compares its production kernel
  against an independent reference on real frames of the run being processed, and aborts the job
  if they disagree — so a geometry or masking mistake cannot produce a whole run of wrong sums.
- **Blocks, resume and a config hash.** Trains are grouped into blocks (the unit of scheduling,
  ledger and resume). The output file stores a `config_hash` covering every configuration field
  that can change a stored number, plus the sha256 of every input file (geometry, PONI, masks). A
  file whose hash differs is refused — the pass raises instead of opening it, leaving the file
  untouched — so blocks computed under different rules can never be merged silently. Passing
  `overwrite=True` replaces the file from scratch instead. The *layout* is checked separately and for the same reason:
  the hash covers the configuration, not the schema, so a file that lacks a per-frame column the
  pass now stores is refused as a schema mismatch instead of being resumed into and failing
  part-processed. Adding a column therefore means existing files are reprocessed, not resumed.
- **Provenance.** Host, platform, package versions, worker count, timings, the operator hash, the
  bits seen in the per-frame masks, the status summary and a set of run checks (pulse pattern, XGM
  photon energy, detector motion) are written into the file.
- **Parallelism.** One spawned, single-threaded worker process per *physical* core. Workers only
  read and return arrays; the parent process owns the output file and every write to it, so a
  failed block reaches the ledger instead of the file.

### 3.2 AGIPD SAXS pass (`analysis.saxs`)

Entry points: `run_agipd_saxs(cfg)`, and `analysis.saxs.damnit.agipd_saxs(proposal, run)`.

The defining constraint is that AGIPD proc frames are **integer photon counts that are ~99 %
zero**. So instead of integrating a dense 1M-pixel image per frame, the pass lifts pyFAI's own
full-split sparse matrix (CSC — compressed sparse column, one column per detector pixel) out of
the engine and sums over the photon hits only.

| Module | Role |
|---|---|
| `config.py` | `AgipdSaxsConfig`: a frozen dataclass — geometry file, detector distance, photon energy, beam centre, `npt`, method, mask sources, blocking and output paths |
| `operator.py` | `SparseOperator`: the CSC arrays (`coef`, `bins`, `indptr`), the solid-angle array Ω and the q axis, plus a sha256 over all of them. Built once per run, saved to disk for the workers to load |
| `sparse.py` | the pure kernels: `gather` (Σ c^p·w over CSC columns), `denominator`, `integrate_frame`. No I/O, no EXtra-data import |
| `masks.py` | the static mask (the double-width pixels at the readout-chip, "ASIC", boundaries ∪ the hand-maintained pixel mask) and the per-memory-cell **base masks** with precomputed denominators |
| `plan.py` | opens the run, builds the train/block row model, records the run checks |
| `worker.py` | per-block read and frame loop in a spawned worker |
| `writer.py` | the AGIPD schema on top of `FrameTableWriter`, plus the `(trainId, pulseId, q)` reducer |
| `selftest.py` | sparse vs dense-pyFAI agreement gate on real frames |
| `damnit.py` | the DAMNIT-facing functions and the three-panel overview figure |

Masking is exact, in three layers: a **static** mask that excludes the same pixels all run; a
**per-cell base mask** (the calibration pipeline's static bad pixels are per memory cell, so a
majority vote over a few sampled trains gives each cell its own mask and a precomputed
denominator `D = Σ c·Ω`); and a **per-frame correction** that adjusts `D` by only those pixels
where the frame's own mask disagrees with its cell's — adding pixels the cell calls bad and the
frame calls good, subtracting the reverse. The result is exact, not approximate. Integer data
cannot carry NaN, which is what forces this sparse denominator correction rather than the simpler
NaN trick the WAXS pass uses.

Variance is `Σ c²·x`, the Poisson variance of integer photon counts. pyFAI's own Poisson error
model is not used: it floors per-pixel variance at 1, which on frames this sparse inflates the
variance by nearly two orders of magnitude.

Measured behaviour: a full run (~465 000 frames) integrates in about 6 minutes on 36 workers, and
the pooled I(q) matches a dense pyFAI reference to ~5e-8 relative.

### 3.3 JUNGFRAU WAXS pass (`analysis.waxs`)

Entry points: `run_jungfrau_waxs(cfg)`, `config_for(proposal, run, detector)`, and
`analysis.waxs.damnit.jungfrau_waxs(proposal, run, detector)`.

**One config, one PONI, one static mask and one output file per detector** — a PONI is pyFAI's
geometry file, naming the detector type, the sample-detector distance, the point of normal
incidence and the wavelength the geometry was refined at. The two
JUNGFRAU-500Ks have different geometries, masks and q ranges; they meet only in
`analysis.waxs.combine`.

JUNGFRAU frames are `float32` keV and **dense**, so this pass runs dense pyFAI per frame. Two
consequences shape the module list:

- **The per-frame mask is carried as NaN in the data and variance arrays**, not passed as
  `integrate1d(mask=)`. pyFAI caches its sparse matrix keyed on a checksum of the mask, so a mask
  that changes per frame would rebuild the full-split matrix every frame (measured ~8× slower).
  pyFAI's preprocessing drops a non-finite pixel from the numerator *and* the normalisation, so
  the NaN path is bit-identical to the reference — and `selftest.py` exists precisely to keep
  proving that on real frames.
- **No per-cell base mask and no seam mask.** The static mask is the `.edf` file alone: the
  JUNGFRAU `data.mask` does set the `NON_STANDARD_SIZE` bit that AGIPD's does not, so there is
  nothing for a seam mask to add.

| Module | Role |
|---|---|
| `config.py` | `JungfrauWaxsConfig`, per-detector file defaults, `config_for`, and the storage-cell-sequence check |
| `cells.py` | measures **which memory cells are lit** and the **readout noise**, from the same sampled trains |
| `operator.py` | `WaxsOperator`: the q axis, solid angle, static mask and the PONI text verbatim, plus the cached pyFAI engine |
| `integrate.py` | `ErrorModel` and the pure `integrate_frame`; `extreme_pixels`, `frame_maxima` |
| `masks.py` | loads the `.edf` static mask; builds the per-frame union |
| `plan.py`, `worker.py`, `writer.py`, `selftest.py` | as on the AGIPD side |
| `combine.py` | fits one multiplicative factor putting jf2 onto jf1 over their q overlap, and merges them into one curve |
| `damnit.py` | the DAMNIT-facing functions and the two-trace overview |

**Lit cells.** A JUNGFRAU train holds 16 memory cells, of which only some see beam; the proposal
used several readout patterns across its runs, so the lit set is *measured per run* rather than
assumed. A cell counts as lit when enough of its kept pixels hold at least half a photon — by
default, more than 4.8e-4 of them clear 4.5 keV, half the ~9 keV single-photon energy — and a large multiplicative step between cells
splits them again, so an attenuated run separates on the step rather than on the absolute level. The pass then gates on the *shape* of the result — every real readout pattern is a run of
consecutive cells modulo 16 — instead of against one fixed set. The cell ids come from
`data.memoryCell`, never from a frame's position in the array, which is a different sequence. A
run with no lit cell at all returns `None`: a run that saw no beam is a result, not a failure.

**Error model.** `Var(x) = σ_read² + E·x`, in keV². `Σc²·x` is the Poisson variance of integer
counts and means nothing on keV-valued data with a negative noise tail; for a pixel holding `x` keV
deposited by `E`-keV photons the count is `x/E`, whose Poisson variance in keV² is `E·x`, and the
readout term adds `σ_read²`. The result is stored **unclamped** — clamping at zero would bias the
noise floor upward on every negative-noise pixel and that bias would not cancel, whereas the
unbiased form's negatives do once a bin pools many frames. Negative per-frame bins are counted
into the ledger. `σ_read` is measured from the run's own dark cells where it has any, and falls
back to a per-detector value when the run reads all 16 cells.

**Wild pixels.** A pixel whose value is non-finite or outside `±max_abs_kev` (1000 keV by
default, against a single-photon peak near 9 keV) is excluded from that frame and the frame is still integrated, with the count recorded in `frames/n_extreme_pixels`.
(These are real Bragg spots from NaCl crystallising out of the drying droplet, which land in jf1's
q range and saturate.) An affected frame's ring bin therefore reads slightly low, so a reader
filters on that column before treating that bin quantitatively. `DATA_CHECK_FAILED` survives only
for a frame with every pixel excluded.

**Detector cross-check.** `combine_files` fits jf2 onto jf1 over their overlap and reports the
factor, its error, the reduced χ² of the residual and that residual's slope across the overlap.
The factor absorbs any relative normalisation difference by construction; what it cannot absorb is
a difference in *shape*, which is what the χ² and the slope are there to expose. One measured
example: on a run without Bragg peaks the factor came out at ≈0.988, so two independently refined
PONIs and masks agreed on absolute I(q) to about 1 %. Note that the cross-check is only sharp when
the overlap carries a feature — over a featureless power law a small error in one q axis is
degenerate with a scale factor, and the fit absorbs it.

### 3.4 Output file format (both passes)

One HDF5 file per run (per detector, for WAXS), under
`scratch/agipd_saxs/r{run:04d}/agipd_saxs.h5` and
`scratch/jungfrau_waxs/r{run:04d}/jungfrau_waxs_{jf1,jf2}.h5`.

```
/frames/            one row per frame, in train order
    trainId, status
    signal, normalization, variance        (n, npt) float32
    AGIPD:    reader_pulseId  the pulse id the reader gave this frame
              cellId          the memory cell it was stored in
              photons_valid   total photons integrated (bad pixels excluded)
              n_bad_pixels    pixels excluded from this frame
              n_frame_specific  pixels where this frame's mask differs from its cell's
              max_count       largest photon count in the frame
    JUNGFRAU: cellId          the memory cell, from data.memoryCell
              energy_valid    total keV integrated (bad pixels excluded)
              n_bad_pixels    pixels excluded from this frame
              n_negative_variance_bins  q bins whose unclamped variance came out below zero
              max_kev         largest value surviving the union mask
              max_kev_static  largest value surviving the static mask alone
              n_extreme_pixels  pixels the value check dropped
/trains/            trainId, first (row offset), count, status
/q/centers          the q axis, with its unit as an attribute
/operator/          AGIPD: the CSC arrays, Ω, q, hashes
                    JUNGFRAU: q, Ω, the packed static mask, the PONI text verbatim
/masks/  (AGIPD)    per-cell base masks (packed), their denominators, the static mask, sampled trains
/cells/  (JUNGFRAU) the lit/dark split, each cell's lit fraction, the error-model parameters
/provenance         attributes: config, config_hash, which fields the hash covers, run checks,
                    package versions, timings, status summary, host
```

Readers in `analysis.common.writer` turn that back into xarray objects: `pooled_per_train` (per
train I(q) and σ(q)), and `per_label` — exposed as `per_pulse` for AGIPD, giving a
`(trainId, pulseId, q)` grid, and `per_cell` for JUNGFRAU, giving `(trainId, cellId, q)`. The grid
axis is named for the physical quantity (`pulseId`) while the stored column keeps the name of where
the label came from (`reader_pulseId`), a reminder that it is the reader's label and not a
position. Only `OK` frames are placed on those grids, one frame per slot, so the `n_frames`
coordinate is 1 where a frame landed and 0 where none did; an empty slot holds zeros, never NaN,
and the count of frames that could not be placed rides along in the result's attributes.

Reading a finished run back:

```python
from analysis.saxs.damnit import pooled_from_file, per_pulse_from_file

pooled = pooled_from_file(".../r0423/agipd_saxs.h5")   # Dataset: intensity, sigma, n_frames
curve  = pooled.intensity.where(pooled.n_frames > 0)   # select on n_frames, never on isnan
grid   = per_pulse_from_file(".../r0423/agipd_saxs.h5")
```

`analysis.waxs.damnit` has the matching `pooled_from_file` and `per_cell_from_file`. Note the two
`n_frames` differ: on the pooled dataset it is **how many `OK` frames that train pooled** (0 for a
train with none), while on the per-pulse and per-cell grids it is 0 or 1, since each slot holds at
most one frame. For anything finer, read `/frames` directly and select on `status == 0`.

### 3.5 DAMNIT integration (`src/amore/context.py`)

DAMNIT is the facility's per-run processing and summary table. The context file declares the
variables; the heavy ones run as one Slurm job per run on a whole node. Groups of variables:

- **Metadata**: train count, run length, raw/proc size, run type, sample name, scan type.
- **Beam**: XGM pulse energy and overview plot, pulse count, repetition rate, the three
  transmission stages and their product.
- **Droplet tracking**: fits an ellipse to the levitated droplet in the camera images and returns
  its centre, radii and volume over the run — the time axis every kinetic interpretation rests on.
  Implemented in `src/amore/analysis_helpers.py`.
- **XPCS**: drives the facility's `extra-speckle` offline pipeline and exposes its figures — the
  two-time correlation function, the intensity autocorrelation g₂ and its fit, a SAXS overview,
  the q rings used, the beam centre, outlier diagnostics, the mean photon count per pixel, and the
  corrections applied. The pipeline is the facility's, not this repository's; what lives here is
  the parameter choice and the plumbing.
- **The two integration passes**: `agipd_saxs` and `agipd_iq_overview`; `jungfrau_waxs_jf1`,
  `jungfrau_waxs_jf2`, `jungfrau_waxs_overview` and `jungfrau_waxs_combined`. Each body is an
  import plus a call.

### 3.6 pyBeamtime reader plugin (`src/readers`)

`EuXFELMIDRawReader` implements pyBeamtime's raw-reader interface for this beamtime: it maps a
proposal root to run directories, reports run metadata from the filesystem alone, and turns one
run directory into a lazy, dask-backed `xarray.Dataset` via EXtra-data, with each source isolated
under its own prefixed dimensions. Importing the `readers` package registers it with pyBeamtime's
`ReaderRegistry` under the slug `"mid"`. It is deliberately self-contained — absolute
`pyBeamtime.*` imports, self-registration at import time, and the EuXFEL stack imported lazily
inside `load_run` — so the same file works as a plugin here or as a module inside pyBeamtime
itself, and so its pure path helpers stay unit-testable without the EuXFEL stack installed.

---

## 4. Conventions that hold across the codebase

These are invariants, not style preferences — most were introduced after something went wrong
silently.

1. **Masks: non-zero means excluded**, everywhere, which is also pyFAI's convention. Nothing in
   either pass inverts a mask.
2. **No NaN sentinels in stored output.** Status codes and `n_frames` counters carry that
   information instead.
3. **Identity comes from the reader, never from array position** — train ids, pulse ids and memory
   cell ids alike.
4. **pyFAI integration methods are given as tuples and the resolved method is checked.** A method
   given as a string can silently resolve to a different integrator, and an unavailable engine
   falls back without an error.
5. **Never hand pyFAI a read-only array, and never freeze an array pyFAI might own.**
   `np.ascontiguousarray` returns the input itself when it already matches, so freezing the result
   can freeze pyFAI's own cache; and its Cython kernels require writable buffers on some platforms
   but not others. `analysis.common.arrays.frozen_copy` exists for this.
6. **Every numerical library is pinned to one thread** before anything imports numpy, and worker
   pools are **spawned, never forked**.
7. **Configs are frozen slots dataclasses that pickle by field *name*.** The dataclass default
   pickles by position, so adding a field would silently shift every later value when a stale
   import sends state to a fresh worker.
8. **The config hash covers only what can change a stored number.** Worker count, block size,
   output root and similar are excluded and recorded separately, so the same run processed at a
   different worker count still resumes.
9. **Read every HDF5 attribute inside its `with` block.** An `h5py` `AttributeManager` kept past
   the block returns defaults instead of raising, so a provenance value silently reads as absent.

---

## 5. Environment, tests and scripts

**Stack** (versions pinned in `uv.lock`; these are the ones every measurement was taken against):
Python 3.12, numpy 2.4, h5py 3.16, EXtra-data 1.24, euxfel-EXtra 2026.1.1, EXtra-geom 1.16,
pyFAI 2026.5, fabio, xarray, plus `extra-speckle` and `damnit` from the facility's git servers and
`pyBeamtime` as a local editable path dependency.

```bash
uv sync --extra euxfel    # the full stack; plain `uv sync` gives only the core (pyBeamtime + pytest)
uv run pytest             # ~480 unit tests
uvx ruff format && uvx ruff check
```

The test suite needs **no cluster and no facility data**, but it does need the `euxfel` extra: the
two integrator suites import pyFAI and EXtra-geom, and skip themselves without them. Everything
they integrate is synthetic.

A pass is invoked with a config, from a notebook, a script or a DAMNIT variable:

```python
from analysis.saxs.damnit import agipd_saxs          # -> (trainId, pulseId, q)
from analysis.waxs.damnit import jungfrau_waxs       # -> (trainId, cellId, q)

grid = agipd_saxs(10400, 423)                        # proposal, run
jf1  = jungfrau_waxs(10400, 423, "jf1")              # None if the run had no lit cell

# any config field can be overridden through the same call
grid = agipd_saxs(10400, 423, npt=1000, allow_incomplete=True)
```

Both open the run themselves — proc for the frames, raw for the control sources the run checks
need — write their per-frame sums under `scratch/`, and return the grid.

**The ledger and the refusal to return a partial result are two different things, and both hold.**
Whatever happens to a frame is recorded in the file: the run does not abort on a failed block, and
the file's ledger accounts for every row. But at the end of the pass, a run in which some frame did
not reach `OK` raises `IncompleteRun` rather than handing back a grid with holes in it, because a
partial I(q) that looks whole is worse than a failed variable. Setting `allow_incomplete=True` in
the config turns that refusal off, and then the holes are visible in the grid's `n_frames`
coordinate and in the file's status summary. Either way the sums that were computed are on disk and
the run can be resumed.

The lower-level entry points (`run_agipd_saxs`, `run_jungfrau_waxs`, `config_for`) take an explicit
config and accept an already-open run, which is how the tests and the acceptance scripts drive
them. Every default named in this document — `npt`, the lit-cell thresholds, `max_abs_kev`, the
file paths — lives in `analysis/saxs/config.py` or `analysis/waxs/config.py` and is overridable
per run; overriding one that can change a stored number simply means the run will not resume into a
file written without it.

**Tests** cover `src/` only. Both integrator suites build **mock runs of real shape and dtype**, so
they run on a laptop in a few minutes. Two kinds of test are worth knowing about because they are
what catches an accidental change to a stored number:

- `tests/common/test_pinned_digests.py` pins both configs' field lists, their config hashes, both
  operator hashes and every module's export surface.
- `test_output_snapshot.py` in each suite pins a content digest of a whole mock run's output.

`tests/test_context.py` loads the DAMNIT context file the way DAMNIT does and asserts by AST that
every integration variable's body is an import plus a call.

**Scripts** in `scripts/` are run by hand on a cluster node against real data and each writes its
verdict as JSON beside itself. They are deliberately not unit-tested, so a change to one is
verified by running it.

| Script | What it does |
|---|---|
| `p4_acceptance.py` | runs the AGIPD pass, then gates it on configuration, the self-test, an independent dense reference, timing and the ledger |
| `w4_acceptance.py` | the same for one JUNGFRAU detector |
| `w1_facts.py` | per run and detector: files and sha256s, source names, the lit-cell split and readout noise, whether an ROI would save I/O |
| `w6_data_check.py` | why a run's frames hit `DATA_CHECK_FAILED`, by re-reading the offending frames against both masks |

---

## 6. Related files in the repository

- `CLAUDE.md` — the working notes: beamtime facts, measured data properties, environment details
  and a catalogue of pitfalls.
- `context/agipd-saxs-integrator.md` — the AGIPD pass design guide, with the measurements behind
  each decision.
- `context/jungfrau-waxs-integrator.md` — the JUNGFRAU pass design guide, written as a delta
  against the AGIPD one.
- `context/extra-toolkit-context.md` — what the facility's EXtra component library provides
  (pulse patterns, XGM, calibration constants, the XCCA toolbox).
- `README.md` — setup and usage.
