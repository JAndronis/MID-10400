# CLAUDE.md — MID p010400: ferritin crystallisation in levitated droplets

Repository guide for Claude Code. Read this file first, then the context file for the task at hand.

`<pkg>` = `analysis` (`src/analysis/`), this repository's src-layout analysis package.
The pyBeamtime reader plugin (`src/readers/`) and the DAMNIT context files (`src/amore/`,
not packaged) are separate.

## Context files

| File | Read when |
|---|---|
| `context/saxs-first-pass-integrator.md` | Working on the first-pass AGIPD SAXS loader/integrator (`<pkg>.saxs`) |
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
1. SAXS/WAXS first pass on all runs.
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
| Photon energy | 9.04 keV |
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
| DAMNIT | per-run orchestration, summaries | |
| extra-speckle | XPCS, Tier-2 data access | |
| pyBeamtime (own) | multi-facility readers | EuXFEL reader plugin contract: `load_run(self, run_id, root_path)`. `get_run_path` is required in the ABC even though the docs omit it |
| `scripts/p4_acceptance.py` | P4 acceptance for the SAXS first pass | runs the pass, then the four §10 gates; writes its verdict as JSON beside itself. Needs a node, the real geometry/mask files and r0423 |
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
| Proc (corrected) data | `/gpfs/exfel/exp/MID/202601/p010400/proc/r{run:04d}/CORR-R{run:04d}-AGIPD{module:02d}-S{seq:05d}.h5` (one file per module per sequence) |
| Scratch | `/gpfs/exfel/exp/MID/202601/p010400/scratch/` |
| AGIPD geometry in use | `/gpfs/exfel/exp/MID/202601/p010400/usr/geometry/geom_latest.geom` |
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

- **Geometry source of truth is unresolved.** `geom_latest.geom` + 7.531 m (PONI = 0 on
  `to_pyfai_detector()`) gives a sensible q range. The quadrant-motor encoder source is absent in
  r0500.
- **Beam centre `(607.46, 672.08)`** was derived with the beamline scientist for the old stacked
  detector + `setFit2D` construction. Do not reuse it with `to_pyfai_detector()`. Agreed
  alternative: `geom.offset()`.
- **Seams and pixel mask.** ASIC seams (`agipd_asic_seams()`) and the one pixel mask are both
  needed in addition to `image.mask`. There is deliberately a single mask file: two overlapping
  ones would have to be kept in step with each other.

### Other sources

| Source | Status |
|---|---|
| XGM | `XGM(run).pulse_energy()` dims `(trainId, pulseIndex)` — pulse *index*, not pulse ID. `wavelength()`/`photon_energy()` raise if not constant; use `*_by_train()` |
| X-ray pulse pattern | `XrayPulses(run)`; replaces the fake `pulseId = np.arange(n_pulses)` of the old pipeline |
| Lit-frame finder (LITFRM) | intended for XGM alignment (`data.xgmPulseId`); source and keys unverified — get them from `lsxfel` |
| Droplet imaging | path length from `MID_EXP_CAM/PROC/DROPLET_DOWNSTREAM.current_vol`; verify per run with `lsxfel`. Ellipse-fit volume tracking lives in `analysis_helpers.py` |
| JUNGFRAU-500K (WAXS) | not inspected: dtype, compression, frames per train, gain/mask bits and value distribution unknown |
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
10. **Silent failures in the old pipeline.** pasha forks from a non-main thread; `ThreadPoolExecutor`
    futures are never checked; `mp.Queue.empty()` is racy. Failures end as NaN rows.

---

## Open tasks

| # | Task | Next step / gate |
|---|---|---|
| 1 | First-pass AGIPD SAXS integrator | Phases P1–P6 in `context/saxs-first-pass-integrator.md`; P1 needs no Maxwell |
| 2 | WAXS JUNGFRAU inspection → integrator extension | Adapt benchmark stages 1 and 7 to JUNGFRAU; then gain-aware handling and spec |
| 3 | Lit-frame selection | LITFRM source/keys from `lsxfel`; compare with `XrayPulses` counts per train |
| 4 | AGIPD geometry source of truth | Resolve encoder/motor source (absent in r0500) and decide CrystFEL file vs `geom.offset()` |
| 5 | Polarisation correction | Confirm detector-frame ↔ lab-horizontal orientation and factor with MID; ≤ 5.4e-4 effect at q_max |
| 6 | Correction settings | Photon threshold / recast settings for r0423, r0426 from the correction reports |
| 7 | XGM normalisation recipe | Pulse-index alignment (LITFRM `data.xgmPulseId`), applied post hoc to stored sums |
| 8 | Anisotropic low-q lobe (r0423) | **First pass: resolved.** Excluded by the one static pixel mask (see Paths), OR'd into `static_bad` — integrator I4, option (a). No φ-sector logic and no per-region sums, so the φ ≈ 278–330° values from the old integrator frame need no re-derivation. **Still open as a separate analysis:** lobe amplitude (decays ~16 % over 300 s in the lowest q band, independent of the isotropic drift) vs droplet volume. It needs its own pass, because the first-pass sums no longer carry the lobe |
| 9 | Transmission correction integration | `droplet_transmission.py` on stored sums; V/V₀ pairing (r0423 ↔ r0464) |
| 10 | XPCS | Define q-binning; custom g2 model with KWW fitting (extra-speckle lacks it); shear vs diffusion diagnostic |
| 11 | XCCA second pass | ROI list excluding Bragg q; per-shot masks; `AveragedAngularCorrelationMasked`; bulk chunk reader (integrator I3) |
| 12 | Tier-2 data access | extra-speckle orchestration without materialising full detector arrays |
| 13 | pyBeamtime integration | EuXFEL reader plugin on top of the `<pkg>` reader layer |
| 14 | Upstream issues | EXtra-data per-chunk lookup; pyFAI method-string resolution and OpenCL fallback warning; XCCA `from_dataset` double update |
