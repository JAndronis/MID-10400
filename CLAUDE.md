# CLAUDE.md — MID p010400: ferritin crystallisation in levitated droplets

Repository guide for Claude Code. Read this file first, then the context file for the task at hand.

`<pkg>` = `analysis` (`src/analysis/`), this repository's src-layout analysis package. It holds
`analysis.common` (detector-agnostic: status codes, the train/block row model, the mask vocabulary,
CPU and pool helpers, the frame-table writer), `analysis.saxs` (AGIPD) and `analysis.waxs`
(JUNGFRAU). The pyBeamtime reader plugin (`src/readers/`) and the DAMNIT context files
(`src/amore/`, not packaged) are separate.

## Context files

| File | Read when |
|---|---|
| `context/agipd-saxs-integrator.md` | Working on the AGIPD SAXS loader/integrator (`agipd_saxs`, `<pkg>.saxs`) |
| `context/jungfrau-waxs-integrator.md` | Working on the JUNGFRAU WAXS integrator (`<pkg>.waxs`). Written as a delta against the AGIPD file — read that one first |
| `context/extra-toolkit-context.md` | Using EXtra components: XCCA toolbox, `XrayPulses`, `XGM`, calibration, quadrant motors |

Adjust the paths above if the context files live elsewhere.

## Working rules

1. **Plan first.** Present a plan before creating or editing files. Implement one phase at a time
   and stop at its acceptance criteria.
2. **Verify, don't guess.**
   - Check external API signatures against the installed source.
   - Check EuXFEL source and key names against `lsxfel` output for the run in question. Never
     guess them.
   - Never state a numerical threshold without a traceable source.
3. **Specs are markdown.** Write them as prose plus pseudocode (this file, `context/`). No
   standalone `.py` files as specs.
4. **Benchmark before deciding.** Performance decisions need a measurement on a Maxwell node from
   the DAMNIT partition. Keep benchmark JSONs next to the benchmark script.
5. **Priorities.** For analysis code: data quality > completeness guarantees > speed.
6. **Style.** `uvx ruff format`, `uvx ruff check`.

---

## Scientific context

**Experiment.** European XFEL proposal 10400 at MID. Time-resolved SAXS/WAXS of ferritin
crystallisation in ~1 mm ultrasonically levitated droplets.
- Sample: ferritin (Sigma F4503), 50 mg/ml, 150 mM NaCl.
- Primary crystallising condition: PEG 6K, 5 w/v %.
- At EuXFEL only PEG 6K samples crystallised; PEG 1K did not, unlike the CoSAXS predecessor. The
  mechanistic interpretation is therefore limited to PEG 6K.
- Team: Andronis (organiser), Girelli, Karina, Bag, Bergstrom, Plivelic, Roosen-Rugen, Perakis.

**Central question.** Does crystallisation proceed by continuous-order desolvation, or by two-step
nucleation?
- *Continuous-order desolvation* (Houben et al. 2020, *Nature* 579:540): order and density rise
  gradually and concurrently; no classical nuclei.
- *Two-step nucleation* via a disordered dense intermediate (Sauter et al. 2015, *JACS* 137:1485):
  a broad intermediate SAXS peak at q ≈ 0.7–1.4 nm⁻¹ precedes the Bragg peaks and decays as they
  grow.
- *XCCA is the main discriminator.* Continuous order predicts a gradual pre-Bragg rise in angular
  correlations; two-step nucleation predicts no angular pre-order.
- *Possible addition:* Xu et al. MD simulations (transient high-density fluid inside the spinodal,
  Ostwald's rule of stages).

**Predecessor results** (Girelli/Andronis, in prep., CoSAXS/MAX IV):
- fcc Bragg peaks at q ≈ 0.6, 0.7, 1.0 nm⁻¹.
- Droplet volume follows a D² law until a turning point t*. Crystals form, then dissolve near t*
  (dehydration-driven): the unit cell contracts before melting, and the crystals are 30–70 % water.
- PEG 1000 suppresses melting (surface hydration); PEG ≥ 3300 does not.
- PEG molecular weight sets the mechanism through the depletion parameter ξ = r_g/a:
  - PEG 6000, ξ = 0.40: enthalpic, S(q=0.1) > 1.
  - PEG 1000, ξ = 0.16: entropic, S(q=0.1) < 1, with nonclassical aggregate precursors.
- LLPS threshold ξ > 0.25 comes from Ilett et al. 1995, *PRE* 51:1344 — not from Tanaka & Ataka,
  which uses a different parameter.

**Analysis passes:**
1. SAXS/WAXS on all runs (`agipd_saxs`).
2. XCCA on runs selected from pass 1.
3. XPCS.

**Firm epistemic boundaries:**
1. **Low-q enhancement.** A low-q rise before crystallisation in PEG 6K cannot be interpreted
   mechanistically from SAXS alone: phase separation and aggregation are indistinguishable.
2. **XCCA ROIs** must exclude Bragg q-ranges by design.
3. **XPCS lag window.**
   - The minimum accessible lag is ~1 µs (222 ns nominal).
   - Shear decorrelation from the velocity gradient across the beam width is a real concern.
     Diagnostic: Γ ∝ q² means diffusion, Γ ∝ q means shear.
   - Advective decorrelation on ns scales falls below the window; it lowers contrast without
     appearing as explicit dynamics.
4. **XFEL heating.** Local heating (~10–20 K on µs) exceeds evaporative cooling (~0.2–0.5 mK) by
   4–5 orders of magnitude within a train.
5. **Beam damage.**
   - Characterise it in capillaries, not droplets: droplets conflate evaporation with damage.
   - PEG 6K is more radiation-vulnerable than PEG 1K. Chain scission could shift ξ towards the
     LLPS threshold.
   - High PEG concentration may be radioprotective through OH-radical scavenging.

**Method references:**
- XCCA: Moller et al. (krypton, serial regime, FCC vs HCP via C(q₁,q₂,φ)). Our case is time
  resolved, which adds normalisation problems (changing scatterer population) and a possible
  flow-aligned-aggregate confound.
- MID XPCS pipeline: Leonau et al., *J. Synchrotron Rad.* 33, 725 (2026).

---

## Instrument and runs

| Item | Value |
|---|---|
| Facility / proposal path | European XFEL MID, `/gpfs/exfel/exp/MID/202601/p010400/` |
| Beamtime | May 2026 |
| Photon energy | 9.04 keV — **disputed**: the XGM's `pulseEnergy.wavelengthUsed` gives 9.000 keV nominal on r0423, 0.44 % lower. q scales with it (open task 15) |
| SAXS detector | AGIPD-1M at 7.531 m; q ≈ 0.076–1.07 nm⁻¹ with `geom_latest.geom` |
| WAXS detectors | two JUNGFRAU-500K |
| Pulse structure | ~1.128 MHz; pattern varies between runs (350 and 155 pulses/train seen). r0423, r0426: 155 AGIPD frames/train |
| Transmission | no downstream diode; droplet path length from droplet imaging (see Data files) |
| Run length | r0423, r0426: ~3000 trains each (~465 k AGIPD frames) |

| Run | What is known |
|---|---|
| r0423 | No Bragg reflections in sampled trains. Full benchmark set (`agipd_stage_rates.py`). Stationary low-q anisotropic lobe (open task 8). Sample/background pairing with r0464 must be matched by V/V₀, not train index. |
| r0426 | Crystallised: excess intensity at 0.59–0.64 and 1.00–1.03 nm⁻¹ relative to r0423. |
| r0500 | Quadrant-motor encoder source expected by extra-speckle is absent. |

**Corrections: physics caveats**
- **Optically thick droplet.** At 9.04 keV, µt ≈ 3.4–3.7 (transmission 2.6–3.3 %), far above
  optimal.
- **Iron fluorescence** (6.4 keV) grows as ferritin concentrates, giving a train-correlated
  incoherent background.
- **Low-q rise.** It may scale as r² (interfacial/refraction) rather than r¹ (bulk), which a simple
  T·t correction cannot remove.
- **`droplet_transmission.py`** implements the correction chain: attenuation, spheroid chord,
  V/V₀ matching, displaced-solvent factor, rⁿ diagnostic. The ferritin iron mass fraction (nominal
  20 wt %) is the dominant uncertainty.
- **XGM.** Proc data are not XGM-normalised; calibration applies detector corrections only. For
  XPCS/XCCA, prefer the ROI-averaged detector intensity per frame (TTCF denominator) over upstream
  XGM division, because droplet transmission varies independently of the XGM signal.

---

## Environment, dependencies, tools

**Cluster and DAMNIT**
- Maxwell HPC. DAMNIT runs `cluster=True` variables in one Slurm job per run on a whole node.
  The default `slurm_time` is 2 h. DAMNIT passes no `--gres`/`--constraint`, so a GPU node cannot
  be requested.
- Node as measured on the DAMNIT partition: 2 sockets × 18 cores, 36 physical / 72 logical cores,
  one L3 cache domain per socket.
- Maxwell's `exfel-python` module is redeployed nightly from EXtra master. Don't use it for results
  that must be reproducible.

**Project**
- uv project at `/home/andronis/MID-10400`: src layout, hatchling backend, dependencies pinned in
  `uv.lock`.
- DAMNIT uses it via `damnit db-config context_python /home/andronis/MID-10400/.venv/bin/python`.
- **Heavy code lives in `<pkg>`, not in the DAMNIT context file.** DAMNIT `exec`s the context file
  into a dict, so functions defined there cannot be pickled to spawned worker processes.
- Any process reading AGIPD data sets `EXTRA_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.

**Measured stack** (versions used for all benchmarks):

| Package | Version | Role |
|---|---|---|
| numpy | 2.4.6 | |
| h5py | 3.16.0 | |
| EXtra-data | 1.24.0 | run access, detector reads |
| euxfel-EXtra | 2026.1.1 | components (`XrayPulses`, `XGM`, calibration, XCCA toolbox) |
| EXtra-geom | 1.16.0 | AGIPD geometry, `to_pyfai_detector()`, `agipd_asic_seams()` |
| pyFAI | 2026.5.0 | sparse operator construction, reference integration |
| zlib_into 0.4, isal 1.8.0 | dev group only | benchmark stage 6. **zlib_into switches EXtra-data to its slower threaded path** (see Pitfalls) |

**Tools**

| Tool | Use | Notes |
|---|---|---|
| EXtra-data | proc run access | `AGIPD1M(dc, min_modules=16)["image.*"].ndarray(decompress_threads=1)`; identity from `train_id_coordinates()`, `pulse_id_coordinates()`, `cell_id_coordinates()` |
| EXtra-geom | geometry | `AGIPD_1MGeometry.from_crystfel_geom`, `to_pyfai_detector()` (PONI = 0 at geometry origin), `agipd_asic_seams()` |
| EXtra | components | `XrayPulses`, `XGM`, `CalibrationData.from_correction`, `AGIPD1MQuadrantMotors`, `extra.applications.xcca` — see `context/extra-toolkit-context.md` |
| pyFAI | operators, reference | method tuples only; no Poisson error model on photon-count data |
| DAMNIT | per-run orchestration, summaries | Context variables are thin wrappers over `<pkg>`: `analysis.saxs.damnit` for `agipd_saxs`, `analysis.waxs.damnit` for `jungfrau_waxs_jf1`/`jf2`/`_overview`. `tests/test_context.py` loads the context file the way DAMNIT does and asserts, by AST, that every wrapper body is an import plus a call |
| extra-speckle | XPCS, Tier-2 data access | |
| pyBeamtime (own) | multi-facility readers | EuXFEL reader plugin contract: `load_run(self, run_id, root_path)`. `get_run_path` is required in the ABC even though the docs omit it |
| `scripts/p4_acceptance.py` | P4 acceptance for `agipd_saxs` | runs the pass, then the four §10 gates; writes its verdict as JSON beside itself. Needs a node, the real geometry/mask files and r0423 |
| `scripts/w1_facts.py` | W1 facts for `analysis.waxs` | per run and detector: file existence + sha256, source names, lit-cell split and readout noise (**open task 2's O4**), whether `data.mask` is train-invariant, whether an ROI over the lit cells saves I/O, extreme-pixel counts. `--runs 423 426 --detectors jf1 jf2`; JSON beside itself |
| `scripts/w4_acceptance.py` | W4 acceptance for `analysis.waxs` | one detector at a time: configuration, NaN-equivalence self-test, an independent per-frame-mask reference against the stored sums, timing, ledger. `--run 423 --detector jf1 --workers 36`; JSON beside itself |
| `agipd_stage_rates.py` | 9-stage benchmark | run from the uv environment; writes JSON after each stage; `--train-offset` avoids page-cached trains |
| pasha | legacy parallelism in `analysis_helpers.py` | fork-only; do not use in new code |
| PyMuPDF | reading reference PDFs | rasterise at 2× (`fitz.Matrix(2, 2)`) before extraction |

**Two-tier data access**
- Tier 1: lazy xarray via EXtra-data. Implemented.
- Tier 2: extra-speckle orchestration for XPCS/XCCA without materialising the full detector
  array. Not yet implemented.

---

## Data files and formats found

### Paths

| What | Path |
|---|---|
| Control sources (XGM, timeserver, motors) | raw only. `open_run(..., data="proc")` opens **one** location and proc holds corrected detector files alone, so `XrayPulses`, `XGM` and `AGIPD1MQuadrantMotors` all raise against it. Use `data="raw"` (or `"all"`) for those |
| Proc (corrected) data | `/gpfs/exfel/exp/MID/202601/p010400/proc/r{run:04d}/CORR-R{run:04d}-AGIPD{module:02d}-S{seq:05d}.h5` (one file per module per sequence). JUNGFRAU: `CORR-R{run:04d}-JNGFR{modno:02d}-S{seq:05d}.h5`, modno 01 = jf1 and 02 = jf2, 6 × 500 trains each on r0423. Same tree as `/gpfs/exfel/d/proc/MID/202601/p010400/r{run:04d}` |
| Scratch | `/gpfs/exfel/exp/MID/202601/p010400/scratch/` |
| AGIPD geometry in use | `/gpfs/exfel/exp/MID/202601/p010400/usr/geometry/geom_latest.geom` |
| JUNGFRAU geometry and masks | `usr/geometry/jf{1,2}.poni` and `usr/masks/jf{1,2}.edf` (pyFAI PONI files and native pyFAI masks — non-zero = excluded, no inversion). `usr/geometry` and `usr/masks` are the same directories as `/gpfs/exfel/u/usr/MID/202601/p010400/{geometry,masks}` |
| AGIPD pixel mask | `/gpfs/exfel/exp/MID/202601/p010400/usr/masks/mask_2026-09-08_AGIPD_SAXS.npy` — the **only** mask file. Non-zero = excluded; carries the bad pixels *and* the low-q lobe (integrator I4, option (a)); OR'd into `static_bad` with the ASIC seams. `usr/masks` is the same directory as `/gpfs/exfel/u/usr/MID/202601/p010400/masks` |
| Superseded | `usr/Shared/IA/custom_agipd_mask.npy` and `usr/masks/mask_2026-05-11_AGIPD_updated.npy`. Do **not** apply either alongside the mask above |

### File structure (EuXFEL format, per module file)

- **Index.** `INDEX/trainId` plus `INDEX/<source>/image/first` and `count` map trains to rows.
  `count` is present in r0423 CORR files.
- **Frame datasets.** `INSTRUMENT/<source>/image/<key>`, one row per frame.
  - Confirmed on r0423: `data`, `mask`, `cellId`.
  - Per the EuXFEL docs, the gain stage is also stored, compressed; it is not used or inspected
    here.
- **Access.** Read through EXtra-data. Take `<source>` names from `lsxfel`; do not hardcode them.

### AGIPD `image.data` (r0423, r0426)

| Property | Value |
|---|---|
| dtype | int16 |
| Chunks / filters | (1, 512, 128), shuffle + deflate; one chunk per module frame |
| Stored / logical | 0.8 % (module 0, S00000); ~1.9 % averaged over all modules |
| Content | **photon counts**: 98.9 % zeros, 1.1–1.3 % ones, 2.5e-4 twos; no negatives; max < 20 per pixel per pulse even in crystallised r0426 |
| Nonzero fraction per frame | 0.7–1.8 %; module 3 densest (4.2–4.6 %), outer modules ~0.2–0.5 % |
| Mean counts per pixel per pulse | 0.15 at 0.08 nm⁻¹ → ~0.007 at 0.6 nm⁻¹ → 7e-4 at 1.05 nm⁻¹ (r0423) |
| Frames / memory cells per train | 155 / 155 (r0423, r0426) — read per run, never hardcode |

**Consequence of photon-count data.** The corrected data are already thresholded to integer
photons, so the correction's photon threshold and recast settings directly set the counting
statistics (open task 6). Integer data cannot carry NaN, so bad pixels must come from
`image.mask`.

### AGIPD `image.mask` (r0423, r0426)

| Property | Value |
|---|---|
| dtype / layout | uint32 bitfield (`BadPixels`), chunks (1, 512, 128), shuffle + deflate, stored ~2 % |
| Bits present | 0 OFFSET_OUT_OF_THRESHOLD, 1 NOISE_OUT_OF_THRESHOLD, 7 FF_GAIN_EVAL_ERROR, 8 FF_GAIN_DEVIATION, 9 FF_NO_ENTRIES, 12 VALUE_OUT_OF_RANGE, 13 GAIN_THRESHOLDING_ERROR |
| `NON_STANDARD_SIZE` (22) | never set, so the double-width ASIC-edge pixels are **not** flagged; apply `agipd_asic_seams()` separately |
| Static bits | per memory cell: 4.05 % always flagged, 0.022 % vary between trains; per-q-band coverage identical in r0423 and r0426 |
| Bit 12 | ~31 px/frame outside static-flagged pixels (~80–130 including them); always zero-valued in `image.data`; flat in q; no enhancement at Bragg q; uncorrelated with frame intensity (r ≈ 0.06); spread over ~20 k pixels (73 % in the worst 10 k) — a pixel-level flag, not intensity censoring |
| Bit 13 | 0.03 px/frame, ~20 distinct pixels |

### Geometry and static masks

- **Geometry source of truth is unresolved.** `geom_latest.geom` + 7.531 m gives a sensible q
  range. The quadrant-motor encoder source is absent in r0500.
- **Beam centre `(607.46, 672.08)`** was derived with the beamline scientist. It is expressed in
  the `geom.to_distortion_array()` frame (origin at the corner of the assembled bounding box, all
  coordinates positive), **not** the `to_pyfai_detector()` frame where PONI = 0 is the beam; the
  two origins are 138.4 mm / 121.6 mm apart. Using it therefore means replacing the corner array
  *and* calling `setFit2D`, together — `analysis.saxs.config.DEFAULT_BEAM_CENTER_PX/PY`, applied by
  `operator.build_operator`, cross-checked against `extra_speckle`'s `ConfigSAXS`.
- **The two conventions disagree by 19.9 px (4.0 mm), almost entirely in y** (x agrees to 0.5 px)
  on `geom_latest.geom`. Measured on r488 train 2637950809: q range 0.0772–1.0668 → 0.0827–1.0556
  nm⁻¹, and the fcc Bragg peaks at 0.584 and 0.633 nm⁻¹ **merge into one at 0.610**. This is not a
  settled question — see open task 4 — so the `beam_center` fields carry whichever convention is
  configured and both are reachable. Anything comparing against the old pipeline or against
  extra-speckle must state which one it used.
- **Seams and pixel mask.** ASIC seams (`agipd_asic_seams()`) and the one pixel mask are both
  needed in addition to `image.mask`. There is deliberately a single mask file: two overlapping
  ones would have to be kept in step with each other.

### Other sources

| Source | Status |
|---|---|
| XGM | `XGM(run).pulse_energy()` dims `(trainId, pulseIndex)` — pulse *index*, not pulse ID. `wavelength()`/`photon_energy()` raise if not constant; use `*_by_train()`. `photon_energy_by_train()` already returns **keV** (it converts `wavelengthUsed` in nm), so do not divide by 1000. Control source: raw only, not proc |
| X-ray pulse pattern | `XrayPulses(run)`; replaces the fake `pulseId = np.arange(n_pulses)` of the old pipeline |
| Lit-frame finder (LITFRM) | intended for XGM alignment (`data.xgmPulseId`); source and keys unverified — get them from `lsxfel` |
| Droplet imaging | path length from `MID_EXP_CAM/PROC/DROPLET_DOWNSTREAM.current_vol`; verify per run with `lsxfel`. Ellipse-fit volume tracking lives in `analysis_helpers.py` |
| JUNGFRAU-500K (WAXS) | **Lit memory cells are `{0,1,2,3,4,5,6,15}`, not `{0…7}`** — measured on the cluster over r0423 and r0426, both detectors, and identical in all four. Cell 7 is dark; cell 15 is lit and runs ~5 % below cells 0–6, the usual JUNGFRAU first-storage-cell behaviour. The lit *array positions* are 0–7; reading the cell ids off those positions is CLAUDE.md pitfall 4 and is how `{0…7}` was first recorded. Readout noise measured per run: jf1 0.359 (r0423) / 0.337 (r0426), jf2 0.318 / 0.318. `data.mask` is **train-invariant** over sampled trains (a potential halving of the pass's I/O, unverified over a whole run), and `roi` cannot help: both datasets are chunked `(1, 16, 512, 1024)`, one chunk spanning all sixteen cells, and the lit set is not contiguous. **Sources settled** (`lsxfel`, r0423 proc): `MID_EXP_JF500K1/CORR/JNGFR01:daqOutput` and `MID_EXP_JF500K2/CORR/JNGFR02:daqOutput`, keys `data.adc` / `data.mask` / `data.memoryCell`. Each carries a legacy `…/DET/JNGFR0n:daqOutput` **soft link** to the CORR name; it appears in `instrument_sources` but not in `source_to_modno`, because `_source_corr_pat` matches `/CORR/` alone. Both names match `_det_name_pat`, so auto-detection raises "Multiple detectors found" against a whole run — pass `detector_name` (and `first_modno` 1 / 2). **Inspected, both detectors** (r0423 train 2637695397). `float32` in **keV**, single-photon peak 8.87 / 9.12 keV; `(train, cell, 512, 1024)` with **16 memory cells/train, only cells 0–7 lit on both** (dark cells ≤ 0.004 % of pixels above half a photon); readout σ 0.323 / 0.317 keV; **dense** — 0.002–0.004 % exactly zero. `data.mask` bits {0, 1, 21 `WRONG_GAIN_VALUE`, 22 `NON_STANDARD_SIZE`} — bit 22 **is** set, so no seam mask is needed. ~~jf2 carries 15 pixels at up to ±1.8e5 keV that `data.mask` does not flag~~ — **wrong, corrected 2026-09-13**: on the cluster every extreme pixel is flagged on both detectors (130 jf1 / 204 jf2 above 1000 keV, none unflagged). That claim came from comparing jf2's data against jf1's mask — `data/proc_mask_jf2_r423.nc` is a byte-identical copy of the jf1 export and needs re-exporting. Geometry from `jf1.poni` / `jf2.poni` (pyFAI `Jungfrau`, 75 µm, 232 mm, 9.04 keV); static masks `jf1_mask.edf` / `jf2_mask.edf` are **native pyFAI masks from silx view** — non-zero = excluded, pass straight to `mask=`, no inversion — covering 75.9 % / 84.4 %. q populated after masking: jf1 11.5–23.7, jf2 9.8–18.5 nm⁻¹. Full detail and open questions: `context/jungfrau-waxs-integrator.md`. **Integrator implemented** (`analysis.waxs`): dense pyFAI per frame with the per-frame mask carried as NaN (pitfall 16), `Var = σ_read² + E·x` in keV² stored unclamped, lit cells and σ_read measured per run |
| Quadrant motors | `AGIPD1MQuadrantMotors`; assert no movement within a run |

### Read performance (r0423, one core, HDF5 path unless stated)

| Operation | Cost |
|---|---|
| `image.data` via EXtra-data, `decompress_threads=1` | 4.6 ms/frame (461 MB/s logical), CPU-bound |
| `image.mask` via EXtra-data, `decompress_threads=1` | 8.2 ms/frame |
| `image.mask`, EXtra-data threaded path (zlib_into present) | 2.1× slower at 2 threads, 3.2× slower at 16 |
| Process scaling (data read) | 32 procs 88 % efficiency; 36 → 72 procs +29 % |
| In-memory decode floor (ISA-L inflate + zlib_into unshuffle) | 0.9 ms/frame data, 1.6 ms/frame mask |
| Per-chunk `get_chunk_info_by_coord` | cost grows with chunk position (9 µs → 1.1–3.0 ms at 40 k chunks, local test); `chunk_iter` returns all offsets in one pass |
| GPFS | not limiting: ~0.1–0.25 GB/s of stored bytes at 32 processes |

---

## Known pitfalls

1. **pyFAI method strings.** `method="csc"` silently resolves to no-split NumPy histogram. Use
   tuples and assert `res.method`. When OpenCL is unavailable, the fallback warning names a
   different engine than the one that runs (pyFAI 2026.5.0).
2. **pyFAI Poisson error model.** It floors per-pixel variance at 1, inflating variance ~87× on
   frames with 1 % occupancy. Pass `variance=x` or compute Σc²·x.
3. **EXtra-data threaded decompression.** It does a per-chunk `get_chunk_info_by_coord` under the
   h5py lock and is slower than the HDF5 path on these files. Pass `decompress_threads=1` or set
   `EXTRA_NUM_THREADS=1`.
4. **Frame identity.** Never infer train/frame identity from array position. A dropped train
   shifts frames onto the wrong trainId under `reshape(..., -1, ...)` plus offsets.
5. **pyFAI engine selection** can silently differ between runs. Record `res.method`. Changing
   integrator parameters invalidates pyFAI's cached sparse matrix, so enforce geometric consistency
   explicitly.
6. **Masking.** Blanket `mask > 0` is only correct where every present bit marks an unusable pixel
   (true for the bits listed above). Record the bit set per run and flag new bits.
7. **XGM division.**
   - `analysis_helpers.load_chunk` divides frames by I0 in place before integration, which is
     irreversible.
   - The `(trainId, pulseIndex)` broadcast there can hide an off-by-one.
   - Normalise the stored sums afterwards instead.
8. **Pooled vs mean-of-frames I(q).** They differ by up to ~2.7 % in a synthetic test (30 %
   frame-to-frame variation, 5 % dynamic bad pixels). Store sums and pool.
9. **XCCA toolbox.** `_CumulativeVarianceBase.from_dataset` updates twice per sample; use the
   `Averaged*` classes or `.update()`. The toolbox is not in any EXtra release and Maxwell tracks
   master nightly, so pin it.
10. **`os.sched_getaffinity` counts logical CPUs.** It reports 72 on the DAMNIT node, not 36.
    Defaulting a worker count to it silently opts into hyperthreading, which is the decision the
    SAXS integrator's P6 exists to make. `analysis.saxs.config.physical_cores` counts sysfs
    sibling groups instead.
11. **DAMNIT annotations break under `from __future__ import annotations`.** DAMNIT's
    dependency injection reads `func.__annotations__` and matches the `var#`/`meta#`/`mymdc#`
    prefix. Under PEP 563 an annotation becomes its own *source text*, so `"var#x"` arrives as
    `"'var#x'"` — quotes included — the prefix no longer matches and every dependency silently
    resolves to nothing. Never add that import to `src/amore/context.py`, and pass
    `dont_inherit=True` when compiling it (`compile()` inherits the caller's future flags).
12. **h5py attrs outlive their file.** An `AttributeManager` kept past its `with` block does not
    raise on `.get`; it returns the default, so a provenance value reads as absent. Read every
    attribute inside the block.
13. **Silent failures in the old pipeline.** pasha forks from a non-main thread; `ThreadPoolExecutor`
    futures are never checked; `mp.Queue.empty()` is racy. Failures end as NaN rows.
14. **A beam centre only means something in its own corner-array frame.**
    `geom.to_pyfai_detector()` and `geom.to_distortion_array()` place the same pixels at different
    coordinates — origins 138.4 mm / 121.6 mm apart — and `setFit2D` interprets its `centerX` /
    `centerY` in whichever array the detector currently holds. So `setFit2D(sdd, px, py)` on a bare
    `to_pyfai_detector()` silently puts the beam somewhere neither convention intends (1.0 nm⁻¹
    off, measured). Replace the corners and set the centre together, or do neither. Nothing raises
    either way: both give a plausible q range and a plausible-looking I(q).
15. **`extra_speckle.Setup(geom=None)` builds geometry from motor encoders.**
    `DetectorAGIPD1M._init_geom` falls through to `geometry_from_encoders(run)`, whose three
    nested bare `except:` clauses end in `"Encoder positions not found, setting all values to 0"`
    and hardcoded quad positions. A `Setup` that looks configured can be running nominal geometry.
    Read its printed output; pass `geom=` to mean a specific file.
16. **pyFAI rebuilds its sparse matrix whenever the mask changes.** `setup_sparse_integrator`
    keys the cached matrix on a checksum of the mask and its own docstring calls the rebuild "a
    very time consuming operation". So a *per-frame* mask passed to `integrate1d(mask=)` rebuilds
    the full-split matrix every frame — measured 16.8 vs 2.3–2.9 ms/frame on a JUNGFRAU module. On
    float data the fix is to build the engine once with the static mask and carry the per-frame
    mask as **NaN in the data and variance arrays**: pyFAI's preprocessing drops a non-finite pixel
    from the numerator *and* the normalisation, giving bit-identical sums (measured 0.0e+00 max
    relative difference on all three sums, both detectors, all 500 bins). Integer data cannot carry
    NaN, which is why AGIPD needs its sparse denominator correction instead.
17. **Never freeze an array pyFAI might own, and never hand it a read-only one.**
    `np.ascontiguousarray(x, dtype)` returns `x` *itself* when it is already contiguous and of
    that dtype, so `arr = np.ascontiguousarray(ai.solidAngleArray(shape), np.float64);
    arr.flags.writeable = False` freezes pyFAI's own `_dssa` cache. Separately, pyFAI's Cython
    kernels acquire *writable* buffers: passing a read-only `mask`, `data` or `variance` raises
    `ValueError: buffer source array is read-only`. **Whether the mask is tolerated is
    platform-dependent** — the macOS wheels accept a read-only mask and the Linux ones on Maxwell
    do not, so this passes every local test and fails on the cluster. Copy before freezing
    (`analysis.saxs.operator._readonly`, `analysis.waxs.operator._frozen`) and keep one writable
    mask on the operator for the hot loop.
18. **`extra_speckle.saxs.get` hard-wires npt and the split scheme.** `_apply_pyfai` passes
    `npt=300`, so `get(..., npt=500)` raises on the duplicate keyword rather than rebinning; and
    `get` consumes `method` for its own `"1d"`/`"2d"` switch, so pyFAI's method can never be passed
    through and it always runs the default `("bbox","csr","cython")`. Its default unit is `q_A^-1`.

---

## Open tasks

| # | Task | Next step / gate |
|---|---|---|
| 1 | AGIPD SAXS integrator (`agipd_saxs`) | **P1–P4 done.** P4 accepted on r0423 2026-09-11: 465 000/465 000 frames OK, pooled I(q) within 4.9e-8 of a dense pyFAI reference, 6.29 min on 36 workers against > 1 h for `analysis_helpers.integrate_run`. **P5 implemented**: `agipd_saxs` / `agipd_iq_overview` in `src/amore/context.py` keep their names and columns, now backed by `analysis.saxs.damnit` instead of `analysis_helpers.integrate_run` — the column's contents change from Å⁻¹ I0-divided to nm⁻¹ undivided, so clear it for runs processed before this. **The beam centre moved on 2026-09-13** (open task 4), which moves the q axis again and changes every stored operator hash — anything integrated before that date must be reprocessed, not merged. Next: run it on r0423 and r0426, then P6 (36 vs 72 workers) |
| 2 | WAXS JUNGFRAU integrator (`<pkg>.waxs`) | **W0, W2, W3 and W5 done 2026-09-13; W1 and W4 need the cluster.** `analysis.common` extracted (SAXS suite passes unedited, `config_hash` byte-identical); `analysis.waxs` implemented and gated on the real r0423 train of both detectors — σ_read 0.3230/0.3175 keV, lit cells (0…7), and the D5′ NaN path **exactly** equal to the per-frame-mask reference on all 500 bins. 105 WAXS tests, 334 in the suite. Two spec decisions were amended by measurement (context file §3 D3 and D5′) and one general pyFAI trap recorded (pitfall 16). Three DAMNIT variables wired: `jungfrau_waxs_jf1`, `jungfrau_waxs_jf2`, `jungfrau_waxs_overview`. **O5 resolved and O6 closed 2026-09-13.** `analysis.waxs.combine` fits a scale factor for jf2 against jf1 over their overlap and merges the two into one curve (`damnit.combined_curve`, DAMNIT variable `jungfrau_waxs_combined`); its χ²ᵣ is a cross-check on the two PONIs **only when the overlap carries a feature** — a q error is degenerate with a scale factor on a featureless curve (measured: 5 % q shift gives χ²ᵣ 0.22 smooth, 776 with a peak), so it bites on r0426 and not on r0423. O6 is closed by decision: no combined SAXS+WAXS curve is planned. **W1 O1 done 2026-09-13:** source names, module numbers and the legacy-alias behaviour settled from `lsxfel` (see the JUNGFRAU row) and wired into `config.DETECTOR_NAMES`/`DETECTOR_MODNOS`; the mock now reproduces the real `/CORR/` layout with its soft-linked `/DET/` alias, and `config_for(proposal, run, detector)` needs no arguments beyond those. **W1 complete 2026-09-13** (`scripts/w1_facts.py` on max-exfl484, r0423 + r0426 × jf1 + jf2): files confirmed, sources confirmed, q ranges 11.53–23.68 / 9.80–18.46 nm⁻¹ from the real PONIs, O4 settled (lit set invariant, and `{0…6,15}` not `{0…7}` — `expected_lit_cells` corrected), `data.mask` train-invariant over sampled trains, `roi` ruled out by the chunk layout, and D6's premise corrected. The first cluster run also surfaced CLAUDE.md pitfall 17. **W4 accepted on r0423 jf1 2026-09-13** (max-exfl484, 36 workers): 24 000/24 000 frames OK, self-test exactly 0.0, dense reference 6.5e-08, **17.5 s** wall at 0.748 parallel efficiency. Per frame per core: read_data 11.6, read_mask 5.0, integrate 3.0 ms — `integrate` did **not** degrade under load, unlike every AGIPD stage, so that warning does not transfer. The read is dominated by `data.adc` (uncompressed) not `data.mask` (gzipped), so the mask train-invariance would save ~25 % of worker time, not half — 4 s on a 17.5 s run, and not worth the risk of a stale mask. **Next:** the same on jf2, then `analysis.waxs.combine` on the two real files |
| 3 | Lit-frame selection | LITFRM source/keys from `lsxfel`; compare with `XrayPulses` counts per train |
| 4 | AGIPD geometry source of truth | **Beam centre wired, not settled.** `agipd_saxs` now applies the agreed `(607.46, 672.08)` via `set_pixel_corners(to_distortion_array())` + `setFit2D`, paired with `geom_latest.geom`. That pairing merges the r488 fcc doublet (0.584 + 0.633 → 0.610 nm⁻¹), so it is the pairing that needs confirming, not the code. Note the old pipeline and `extra_speckle.Setup` agree with each other because they share this construction — and `Setup` with `geom=None` silently builds from motor encoders (`geometry_from_encoders`, bare `except:`, falls back to hardcoded quad positions), so it may not be the same geometry at all. Decisive test: `integrate2d` and check whether I(q,χ) on the 0.633 nm⁻¹ ring is flat in χ or sinusoidal — amplitude and phase give the displacement and its direction. Then resolve the encoder source (absent in r0500) and CrystFEL file vs `geom.offset()` |
| 5 | Polarisation correction | Confirm detector-frame ↔ lab-horizontal orientation and factor with MID; ≤ 5.4e-4 effect at q_max |
| 6 | Correction settings | Photon threshold / recast settings for r0423, r0426 from the correction reports |
| 7 | XGM normalisation recipe | Pulse-index alignment (LITFRM `data.xgmPulseId`), applied post hoc to stored sums |
| 8 | Anisotropic low-q lobe (r0423) | **In `agipd_saxs`: resolved.** Excluded by the one static pixel mask (see Paths), OR'd into `static_bad` — integrator I4, option (a). No φ-sector logic and no per-region sums, so the φ ≈ 278–330° values from the old integrator frame need no re-derivation. **Still open as a separate analysis:** lobe amplitude (decays ~16 % over 300 s in the lowest q band, independent of the isotropic drift) vs droplet volume. It needs its own pass, because the `agipd_saxs` sums no longer carry the lobe |
| 9 | Transmission correction integration | `droplet_transmission.py` on stored sums; V/V₀ pairing (r0423 ↔ r0464) |
| 10 | XPCS | Define q-binning; custom g2 model with KWW fitting (extra-speckle lacks it); shear vs diffusion diagnostic |
| 11 | XCCA second pass | ROI list excluding Bragg q; per-shot masks; `AveragedAngularCorrelationMasked`; bulk chunk reader (integrator I3) |
| 12 | Tier-2 data access | extra-speckle orchestration without materialising full detector arrays |
| 13 | pyBeamtime integration | EuXFEL reader plugin on top of the `<pkg>` reader layer |
| 14 | Upstream issues | EXtra-data per-chunk lookup; pyFAI method-string resolution and OpenCL fallback warning; XCCA `from_dataset` double update |
| 15 | Photon energy, 9.04 vs 9.000 keV | `cfg.photon_energy_kev` (9.04, from this file) sets the wavelength and so the q scale; `XGM.photon_energy_by_train()` reports 9.000 keV nominal on r0423, constant across all trains. The 0.44 % gap shifts every q by 0.44 % — 0.003 nm⁻¹ at the 0.6 nm⁻¹ Bragg peak, 0.005 at q_max. `plan.run_checks` records both and warns. Settle which is authoritative with MID; it is a config change plus a reintegration, not a code change |
