# CLAUDE.md — MID p010400: ferritin crystallisation in levitated droplets

Repository guide for Claude Code. Read this file first, then the context file for the task at hand.

`<pkg>` = `analysis` (`src/analysis/`), this repository's src-layout analysis package. It holds
`analysis.common` (detector-agnostic: status codes, the train/block row model, the mask vocabulary,
CPU and pool helpers, the frame-table writer and its `FrameRows`, the pass skeleton
`common.run.guarded_pass`, the operator comparison `common.complete.operator_differences`, the
proc/raw opener `common.plan.open_proc`), `analysis.saxs` (AGIPD; its three output files are
`analysis.saxs.products`) and `analysis.waxs`
(JUNGFRAU). The droplet figures share `analysis.droplets` (the droplet catalogue, runs left out,
matched t/t*), `analysis.xpcs.kww` (the per-ring KWW fit) and `analysis.plotting` (style, grids,
colours); a new droplet or exclusion is an edit there, not in a script. The DAMNIT context files
(`src/amore/`, not packaged) are separate.

## Context files

| File | Read when |
|---|---|
| `context/agipd-saxs-integrator.md` | Working on the AGIPD SAXS loader/integrator (`agipd_saxs`, `<pkg>.saxs`) |
| `context/jungfrau-waxs-integrator.md` | Working on the JUNGFRAU WAXS integrator (`<pkg>.waxs`). Written as a delta against the AGIPD file — read that one first |
| `context/extra-toolkit-context.md` | Using EXtra components: XCCA toolbox, `XrayPulses`, `XGM`, calibration, quadrant motors |
| `context/run-triage.md` | Working on the run triage (`<pkg>.triage`): per-run quality metrics from DAMNIT, mesh-scan characterisation, red-box ranking for the move to tape |
| `context/damnit-integrity.md` | The output lock, `is_complete`, resume, the DAMNIT context file and its install (`scripts/install_context.py`), `scripts/reprocess_safe.py`; DAMNIT 0.4.0 and Slurm facts (open task 17). **§8: the `Group` columns, the clean-run order, the fresh reprocess and what went wrong on 2026-10-19** |
| `context/jungfrau-photonising.md` | Photon counting in the WAXS pass (`<pkg>.waxs.photons`): the W7 measurements behind it, the design, W4 keV vs photons (open task 18). A delta against `context/jungfrau-waxs-integrator.md` |
| `context/performance.md` | Running on the cluster (`allcpu`, submitting with `sbatch` by hand), pyFAI 2026.5 → 2026.9, virtual CXI files (open task 19) |
| `context/diagnostics.md` | `<pkg>.diagnostics` (frame accounting, flux, SVD, χ², XGM scatter, cell–XGM correlation), per-run XPCS SNR, droplet hits, `*_frames_used` cells, droplet videos, the DAMNIT layout (open task 20) |
| `context/xpcs-quality.md` | XPCS quality from photon statistics: per-frame photon histograms per ring (`<pkg>.saxs.photon_stats`), single-frame contrast, shape checks (open task 20) |
| `context/xpcs-hits.md` | The hit-train XPCS pass (`<pkg>.xpcs`): `HitSet`, the extract-once photon store, partner policy, the `XpcsHits` reader (`analysis.xpcs.reader`; the file layout in `results_file`, the estimators in `estimators`, the SNR and score in `snr`), the P0 facts (sequence boundaries, mask, pulse clock, bytes per train), P4 acceptance (§10, incl. the g2 float32 proof), P5 DAMNIT wiring (§11), what P6 still needs |
| `context/levitator-flow.md` | Levitator and flow in XPCS g2 (s443, r0443–452): recorrelation echoes at T = 9.84 µs (≈ 101.6 kHz drive) in r0445/446, Γ ∝ qⁿ per run (n ≈ 0.65–0.82 late, not diffusion), implied Δv, the 193 kHz AGIPD intensity line, two-time-map checks |
| `../transfer_context/` (outside the repo, read-only) | Lessons from p010646 (MID, 2026-10): JUNGFRAU photonising, DAMNIT context hazards and testing, SLURM, caching, camera videos, fxs/XCCA. Compared with this repository on 2026-10-13; what applies is merged below (pitfalls 23–31, open tasks 17–21). Its numbers are p010646 measurements, not ours |

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
   the DAMNIT partition. Scripts write their JSON beside themselves, and `*.json` / `*.csv` are
   gitignored: quote a number in `context/` with the script's commit and the JSON's path.
5. **Priorities.** For analysis code: data quality > completeness guarantees > speed.
6. **Style.** `uvx ruff format`, `uvx ruff check`.
7. **Tests cover `src/` only.** `scripts/` is deliberately not unit-tested (the three script test
   files were removed 2026-09-14). So a change to a script's logic is caught
   by **running it**, not by pytest — do that before trusting an edit to one. A test whose subject
   is `analysis.*` belongs under the module it tests, even when it is convenient to write it
   against a script's fixtures. Logic a figure reports as a number belongs in `analysis.*`.
8. **Comments and docstrings.** Code is read without this file or `context/` open, so a comment
   must stand alone.
   - *Self-contained.* State the fact in place. Never cite CLAUDE.md, a `context/` file, a
     section number, a pitfall or open-task number, or a phase or session label (P4, W7, S3).
     If a pointer carried a fact the code needs, write that fact in a sentence or two. If it
     carried history, measurements or the rationale of a past decision, leave it in git or
     `context/`.
   - *No history or volatile facts.* No "added on", "originally", "now", dates, run lists, version
     numbers or counts, unless the code depends on them. Describe what is true today.
   - *Why, not what.* Keep a comment for a reason, invariant, unit, hazard or constraint the code
     cannot show. Delete narration, section labels and commented-out code.
   - *Docstrings.* One summary sentence, then prose only for what a caller must know (contract,
     units, side effects, failure modes). Keep Sphinx `:param:` / `:returns:` / `:raises:` fields
     where they add a unit, shape, range or failure condition, and drop those that restate the
     name. A helper with an obvious name and body needs one line or none. A module docstring
     says what the module owns and does not; a test docstring says which behaviour it pins.
   - *Truth over tradition.* A comment that disagrees with the code is wrong: fix the comment.
     Do not state a number with no source.
   - *Prose.* Short declarative sentences, no em dashes, no hedging filler, no rhetorical
     "not X but Y". Keep pragma comments (`# noqa`, `# type:`) verbatim.
   - A comment-only change is checked by comparing the AST with `HEAD` after dropping bare string
     statements; the suite need not run.
9. **Tests first.** Use `pstack:tdd`. A new test must fail against a named mutant in
   `scripts/mutants/<slice>.toml` (or a real bug) before it counts; a deleted test must leave
   every table green (`python scripts/mutants.py -j 6`).

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
   - The minimum accessible lag is one pulse period: **444 ns on r0423** (2.25 MHz, pulse-ID
     step 2; checked 2026-10-13). 222 ns would need 4.5 MHz. Read it per run
     (`XrayPulses.pulse_periods()`), since the pattern varies between runs.
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
| Photon energy | **9.04 keV** — settled 2026-10-13 (user: the undulator spectrum and the beamline staff). The XGM's `pulseEnergy.wavelengthUsed` and the mono read 9.000 keV, which are **setpoints**, not measurements (open task 15) |
| SAXS detector | AGIPD-1M at 7.531 m; q ≈ 0.076–1.07 nm⁻¹ with `geom_latest.geom` |
| WAXS detectors | two JUNGFRAU-500K |
| Pulse structure | **2.25 MHz on r0423**: 155 pulses/train, pulse IDs 200, 202, …, constant pattern (`XrayPulses` on raw, checked 2026-10-13; the earlier "~1.128 MHz" here was wrong, and DAMNIT's `rep_rate` and `utils.py` already say 2.26). Pattern varies between runs (350 and 155 pulses/train seen), so read it per run. r0423, r0426: 155 AGIPD frames/train |
| Transmission | no downstream diode; droplet path length from droplet imaging (see Data files) |
| Run length | r0423, r0426: ~3000 trains each (~465 k AGIPD frames) |

| Run | What is known |
|---|---|
| r0423 | No Bragg reflections in sampled trains. Stationary low-q anisotropic lobe (open task 8). Sample/background pairing with r0464 must be matched by V/V₀, not train index. |
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
- **DAMNIT's cluster variables run on `allcpu`**, `--clusters maxwell`, whole nodes (`select/linear`),
  of **whatever class is free**: from 2 × 10 cores (E5-2640 v4) to 2 × 64 (EPYC 9554), all with 2
  threads per core (`context/performance.md` §1.1). The 2 × 18-core Gold node measured earlier
  (36 physical / 72 logical, one L3 domain per socket) is one of those classes, not "the" DAMNIT
  node. `physical_cores()` sizes the pool to the node, so DAMNIT timings are comparable only on
  the same class; provenance records the host. `allcpu` **preempts and requeues** jobs under the
  same job id (`PreemptMode=REQUEUE`).
- Maxwell's `exfel-python` module is redeployed nightly from EXtra master. Don't use it for results
  that must be reproducible.

**Project**
- uv project: src layout, hatchling backend, dependencies pinned in `uv.lock`. Development clone
  `usr/Shared/IA/MID-10400` (this one); **deploy clone `usr/Software/MID-10400`**, which mirrors
  the development clone's `main` and is never edited by hand.
- **The shared environment** (`scripts/shared_env.sh`, README "On Maxwell"; since 2026-10-06):
  the interpreter in `usr/Software/uv-python`, the deploy clone's `.venv` (editable, no dev group),
  and the `mid-10400` Jupyter kernelspec in `usr/Software/jupyter`. Git hooks in the development
  clone run `shared_env.sh update` after every commit, merge or rewrite on `main`, so committed
  code reaches DAMNIT and the kernel at once and uncommitted code never does. Each user runs
  `shared_env.sh link-kernel` once.
- DAMNIT uses it via `damnit db-config context_python
  /gpfs/exfel/exp/MID/202601/p010400/usr/Software/MID-10400/.venv/bin/python` (previously
  `usr/Software/mid-10400-post-env`, Python 3.11 over the facility `202502` env — not the measured
  stack).
- **pyBeamtime is gone from this repository.** Dropped as a dependency 2026-10-06; the reader
  plugin (`src/readers`) and its test removed 2026-10-13. Its plugins will live in a separate
  repository; the last version here is `git show b14df61:src/readers/io/readers/euxfel.py`.
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
| pyFAI | **2026.9.0** (from 2026.5.0, merged 2026-10-14) | sparse operator construction, reference integration. 2026.9 vs 2026.5: every compared array bit-identical, every timing within ±6 % (`context/performance.md` §2); merging it makes every stored output "not complete" (open task 17), by design |
| imageio-ffmpeg, Pillow | 0.6.0, — | `analysis.camera` droplet videos: the ffmpeg binary (none on PATH) and the frame labels |
| zlib_into 0.4, isal 1.8.0 | dev group only | benchmark stage 6. **zlib_into switches EXtra-data to its slower threaded path** (see Pitfalls) |

**Tools**

| Tool | Use | Notes |
|---|---|---|
| EXtra-data | proc run access | `AGIPD1M(dc, min_modules=16)["image.*"].ndarray(decompress_threads=1)`; identity from `train_id_coordinates()`, `pulse_id_coordinates()`, `cell_id_coordinates()` |
| EXtra-geom | geometry | `AGIPD_1MGeometry.from_crystfel_geom`, `to_pyfai_detector()` (PONI = 0 at geometry origin), `agipd_asic_seams()` |
| EXtra | components | `XrayPulses`, `XGM`, `CalibrationData.from_correction`, `AGIPD1MQuadrantMotors`, `extra.applications.xcca` — see `context/extra-toolkit-context.md` |
| pyFAI | operators, reference | method tuples only; no Poisson error model on photon-count data |
| DAMNIT | per-run orchestration, summaries | Context variables are thin wrappers over `<pkg>`. **Since 2026-10-19 the analysis columns are `Group` classes** (`agipd`, `jungfrau`, `hits`, `xpcs`, `camera`; DAMNIT names `<group>.<var>`, titles unchanged): section "DAMNIT columns" below, `context/damnit-integrity.md` §8. One preprocessing variable per detector kind reads proc (`agipd.preprocessing`, `jungfrau.preprocessing`); every other column reads their files. `tests/test_context.py` loads the context file through `ctxrunner` and asserts, by AST, that every wrapper body is an import plus a call and that no call passes `overwrite`. **The file DAMNIT reads is `usr/Shared/amore/context.py`** (pitfall 27): install with `scripts/install_context.py` |
| `scripts/install_context.py` | Install `src/amore/context.py` + `analysis_helpers.py` as DAMNIT's live context (**not unit-tested**) | dry run unless `--install`; refuses uncommitted `src/amore` or a live file changed since the last install (first time: `--accept-live <id>` after the diff); evaluates both with ctxrunner; backup + `.diff` + sha256 in `usr/Shared/amore/backups/` (`context/damnit-integrity.md` §6) |
| `scripts/reprocess_safe.py` | Reprocess runs through DAMNIT without `overwrite` (**not unit-tested**) | `status`, `start`, `check`, `cleanup`, `restore`, with `--dry-run`: renames outputs aside, refuses busy runs, leftover backups or a quota shortfall, then runs `damnit reprocess --match <title>`; `check` = `is_complete` + every dataset test-read + unchanged frame count; `start --fresh` also moves the window sums, photon histograms and XPCS photon store aside and deletes the hit-rule signal caches, for a clean slate (`context/damnit-integrity.md` §7, §8.3) |
| `scripts/prune_damnit_variables.py` | Remove variables the live context no longer defines from `runs.sqlite` (**not unit-tested**) | dry run unless `--delete`; backs up `runs.sqlite` first; leaves the per-run `.h5` untouched (`context/damnit-integrity.md` §8.3) |
| Removed scripts (2026-10-22) | The finished acceptance, fact and benchmark scripts: `p4_acceptance`, `w1_facts`, `w4_acceptance`, `w6_data_check`, `w7_photonize_check`, `w7_before_after`, `bench_pyfai`, `bench_io`, `bench_extract_blocks`, `bench_store_write`, `xpcs_hits_acceptance`, `xpcs_hits_context_check`, `xpcs_hits_facts`, `xpcs_g1_float32`, `photon_stats_acceptance`, `photon_stats_fused_acceptance`, `photon_stats_shape_check`, `xpcs_quality_probe` | Every number this file and `context/` quote from them was made by the last version: `git show be9dda4:scripts/<name>.py`. Their SLURM self-submission (`-w`, `--local`, `--dry-run`) left with them; `analysis.common.slurm` followed on 2026-10-27 (`git show 7edd1ed:src/analysis/common/slurm.py`) |
| `scripts/droplet_video_acceptance.py` | Acceptance of the droplet videos (**not unit-tested**) | open task 20 |
| extra-speckle | Reference for XPCS (the G1/G2 comparisons of `context/xpcs-hits.md`), Tier-2 data access | No longer imported by the DAMNIT context (2026-10-19): the hit-train pass replaced `xpcs_offline` |
| `scripts/pixel_sums_audit.py` | S3 audit of the per-pixel window sums (`context/agipd-saxs-integrator.md` §15.5) (**not unit-tested**) | read-only, login node: per proc run, `complete` / `missing` / `incomplete` / `mismatch` against the frame table / `busy` (a DAMNIT job queued or running) / `no_agipd`; lists what `run_pixel_sums` must backfill. ~19 min over 461 runs. JSON beside itself |
| `scripts/run_triage.py` | Run triage for the move to tape (`analysis.triage`; `context/run-triage.md`) (**not unit-tested**) | `--phase inventory` (T1): per-run class, effective sample, sizes, DAMNIT product status, droplet series with their flags, and the §9 checks. 39 s on a login node. JSON + CSV beside itself. `--phase xpcs` (SNR) and `--phase classify` (droplet hits per train, `analysis.triage.classify`; `context/diagnostics.md` §5). Ranking is to come |
| pasha | legacy parallelism in `analysis_helpers.py` | fork-only; do not use in new code |
| PyMuPDF | reading reference PDFs | rasterise at 2× (`fitz.Matrix(2, 2)`) before extraction |

**Two-tier data access**
- Tier 1: lazy xarray via EXtra-data. Implemented.
- Tier 2: extra-speckle orchestration for XPCS/XCCA without materialising the full detector
  array. Not yet implemented.

---

### DAMNIT columns

Names are `<group>.<var>`; titles are unchanged (`AGIPD I(q)`, `XPCS g2`, ...). Full table, the old-to-new
name map and the reprocess recipe: `context/damnit-integrity.md` §8.

| Group | Variables |
|---|---|
| `agipd` | `preprocessing`, `saxs`, `iq_overview`, `frames_used`, `diagnostics` |
| `jungfrau` | `preprocessing`, `jf1`, `jf2`, `iq_overview`, `combined`, `jf1_frames_used`, `jf2_frames_used` |
| `hits` | `summary`, `geometry` (hit recognition, mesh map) |
| `xpcs` | `preprocessing` (hit-train pass), `g2`, `ttcf`, `overview`, the old extra-speckle figures and data reproduced from the hit store, `photon_statistics`, `frames_used`, `score` |
| `camera` | `video`, `first_image` |

**Clean-run order:** preprocessing → SAXS/WAXS columns → hit rule (a barrier across the runs of a mesh
block) → photon statistics and the hit-train pass, both on hits only → score and `frames_used`. Reprocess it
with `reprocess_all.sh launch`, then `xpcs-hits`, then `refresh`, one DAMNIT array at a time (`--fresh` is
passed, so no cache of an earlier generation is read). `xpcs.score` and the `*_frames_used` cells have no
`var#` on the passes, so refresh them last.

## Data files and formats found

### Paths

| What | Path |
|---|---|
| Control sources (XGM, timeserver, motors) | raw only. `open_run(..., data="proc")` opens **one** location and proc holds corrected detector files alone, so `XrayPulses`, `XGM` and `AGIPD1MQuadrantMotors` all raise against it. Use `data="raw"` (or `"all"`) for those |
| Proc (corrected) data | `/gpfs/exfel/exp/MID/202601/p010400/proc/r{run:04d}/CORR-R{run:04d}-AGIPD{module:02d}-S{seq:05d}.h5` (one file per module per sequence). JUNGFRAU: `CORR-R{run:04d}-JNGFR{modno:02d}-S{seq:05d}.h5`, modno 01 = jf1 and 02 = jf2, 6 × 500 trains each on r0423. Same tree as `/gpfs/exfel/d/proc/MID/202601/p010400/r{run:04d}` |
| Scratch | `/gpfs/exfel/exp/MID/202601/p010400/scratch/` |
| AGIPD per-pixel window sums | `/gpfs/exfel/exp/MID/202601/p010400/usr/cached_files/agipd_pixel_sums/r{run:04d}/agipd_pixel_sums.h5`, written by `agipd_saxs` and `analysis.saxs.run.run_pixel_sums` (`context/agipd-saxs-integrator.md` §15). In `usr`, not scratch: scratch does not survive the move to tape, and these are the run's last 2D record once proc is taped. The directory does not exist yet (2026-10-07); the pass creates it, so it must be writable by whoever runs the pass, DAMNIT included |
| AGIPD geometry in use | `/gpfs/exfel/exp/MID/202601/p010400/usr/geometry/geom_latest.geom` |
| JUNGFRAU geometry and masks | `usr/geometry/jf{1,2}.poni` and `usr/masks/jf{1,2}.edf` (pyFAI PONI files and native pyFAI masks — non-zero = excluded, no inversion). `usr/geometry` and `usr/masks` are the same directories as `/gpfs/exfel/u/usr/MID/202601/p010400/{geometry,masks}` |
| AGIPD pixel mask | `/gpfs/exfel/exp/MID/202601/p010400/usr/masks/mask_2026-09-08_AGIPD_SAXS.npy` — the **only** mask file. Non-zero = excluded; carries the bad pixels *and* the low-q lobe (integrator I4, option (a)); OR'd into `static_bad` with the ASIC seams. `usr/masks` is the same directory as `/gpfs/exfel/u/usr/MID/202601/p010400/masks` |
| Superseded | `usr/Shared/IA/custom_agipd_mask.npy` and `usr/masks/mask_2026-05-11_AGIPD_updated.npy`. Do **not** apply either alongside the mask above |
| AGIPD photon histograms | `usr/cached_files/agipd_photon_stats/r{run:04d}/agipd_photon_stats.h5`: per frame, 17 rings, k = 0…15, written by the `agipd_saxs` pass (`context/xpcs-quality.md` §7). Written for r0423, r0450 and r318 only so far; the rest comes with the bulk reprocess, ≈ 22–36 GB for the proposal |
| Droplet videos | `usr/cached_files/droplet_videos/r{run:04d}_cam5.mp4`, from `MID_EXP_SAM/CAM/CAM5:daqOutput` `data.image.pixels` ((494, 659) uint16, one image per train); labelled, display choices in the MP4 `comment` (`context/diagnostics.md` §7) |
| extra-speckle's k̄ | `scratch/xpcs/r{run:04d}/kbar.nc`, from its last run (old, comparison only); agrees with our k̄ to 0.4–3 % |
| Output locks | `<output>.lock` beside each pass output while a job writes it (open task 17). A lock left by a dead or requeued job is taken over and the takeover recorded in provenance; never delete one by hand while its job may run |
| DAMNIT context backups | `usr/Shared/amore/backups/`: each install's previous file, `.diff` and sha256 |

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
| JUNGFRAU-500K (WAXS) | **The lit memory cells are not a constant: the proposal used four readout patterns.** Over 746 (run, detector) results spanning r0001–r0500 — `{0…6,15}` for r0379–r0500 (114), all 16 for r0054–r0378 (213), `{15}` for r0052/53/56, and none at all for 43 runs that saw no beam. Every one is a run of consecutive cells **mod 16**, a storage-cell sequence, and *that* is what the pass gates on (`is_storage_cell_sequence`); `EXPECTED_LIT_CELLS` `{0,1,2,3,4,5,6,15}` is the science-block expectation a reprocess may pin, never the default. Cell 7 is dark there; cell 15 is lit and runs ~5 % below cells 0–6, the usual JUNGFRAU first-storage-cell behaviour. The lit *array positions* are 0–7; reading the cell ids off those positions is CLAUDE.md pitfall 4 and is how `{0…7}` was first recorded. Readout noise measured per run where a dark cell exists (jf1 0.359 r0423 / 0.337 r0426, jf2 0.318 / 0.318); a run reading all 16 falls back to `DEFAULT_READ_NOISE_KEV` (jf1 0.3440, jf2 0.3195, the per-detector medians), and `ErrorModel.source` records which. A run with no lit cell returns `None` rather than raising — no beam is a result, not a failure. `data.mask` is **train-invariant** over sampled trains (a potential halving of the pass's I/O, unverified over a whole run), and `roi` cannot help: both datasets are chunked `(1, 16, 512, 1024)`, one chunk spanning all sixteen cells, and the lit set is not contiguous. **Sources settled** (`lsxfel`, r0423 proc): `MID_EXP_JF500K1/CORR/JNGFR01:daqOutput` and `MID_EXP_JF500K2/CORR/JNGFR02:daqOutput`, keys `data.adc` / `data.mask` / `data.memoryCell`. Before 2026-10-28 the WAXS pass looked for its run checks (XGM, timeserver) in proc, so every stored `jungfrau_waxs` file records them as unavailable; the pass now reads them from raw like the AGIPD one. Each carries a legacy `…/DET/JNGFR0n:daqOutput` **soft link** to the CORR name; it appears in `instrument_sources` but not in `source_to_modno`, because `_source_corr_pat` matches `/CORR/` alone. Both names match `_det_name_pat`, so auto-detection raises "Multiple detectors found" against a whole run — pass `detector_name` (and `first_modno` 1 / 2). **Inspected, both detectors** (r0423 train 2637695397). `float32` in **keV**, single-photon peak 8.87 / 9.12 keV; `(train, cell, 512, 1024)` with **16 memory cells/train, eight lit on both** — array *positions* 0–7, carrying cell ids `{0…6,15}` (dark cells ≤ 0.004 % of pixels above half a photon); readout σ 0.323 / 0.317 keV; **dense** — 0.002–0.004 % exactly zero. `data.mask` bits {0, 1, 21 `WRONG_GAIN_VALUE`, 22 `NON_STANDARD_SIZE`} — bit 22 **is** set, so no seam mask is needed. ~~jf2 carries 15 pixels at up to ±1.8e5 keV that `data.mask` does not flag~~ — **wrong, corrected 2026-09-13**: on the cluster every extreme pixel is flagged on both detectors (130 jf1 / 204 jf2 above 1000 keV, none unflagged). That claim came from comparing jf2's data against jf1's mask — `data/proc_mask_jf2_r423.nc` is a byte-identical copy of the jf1 export and needs re-exporting. Geometry from `jf1.poni` / `jf2.poni` (pyFAI `Jungfrau`, 75 µm, 232 mm, 9.04 keV); static masks `jf1_mask.edf` / `jf2_mask.edf` are **native pyFAI masks from silx view** — non-zero = excluded, pass straight to `mask=`, no inversion — covering 75.9 % / 84.4 %. q populated after masking: jf1 11.5–23.7, jf2 9.8–18.5 nm⁻¹. Full detail and open questions: `context/jungfrau-waxs-integrator.md`. **Integrator implemented** (`analysis.waxs`): dense pyFAI per frame with the per-frame mask carried as NaN (pitfall 16), `Var = σ_read² + E·x` in keV² stored unclamped, lit cells measured per run and gated on their shape, σ_read measured per run where a dark cell exists. **Detector settings** (raw `MID_EXP_JF500K{1,2}/DET/CONTROL` run values, r0423, checked 2026-10-13): `gainMode` **dynamic**, `settings` gain0, `storageCellStart` **15** on both — which is why cell 15 is acquired first (15, 0, 1, …). The pass stores rows in ascending cell id, so cell 15 comes last within a train. **The keV data still carry a zero level** that no correction removes (pitfall 23), so the integrated keV are to be **photonised** (open task 18). **Photonised since 2026-10-14** (`context/jungfrau-photonising.md`): per 256² ASIC (the proc layout, checked against bit 22) the zero level is subtracted per frame and the frame rounded to photons; the files store photons (`units` attribute), `asic_level (n, 8)` per frame and a Poisson variance, which matches the frame-to-frame scatter to 1.2 % (jf1) / 0.2 % (jf2). Measured on r0423 and r0370 (W7, 100 trains): jf1 has a **module-wide common mode of 0.13–0.16 keV rms from train to train**, jf2 none; lit cells sit +0.127 keV above dark ones in the same train on both; the pure per-pixel noise is 0.295 / 0.299 keV, so σ_read's jf1/jf2 difference was jf1's common mode. keV ÷ (E·photons) = 1.036 (charge sharing), rising slowly with q |
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
    SAXS integrator's P6 exists to make. `analysis.common.cpu.physical_cores` counts sysfs
    sibling groups instead. Inside a job `nproc` follows `OMP_NUM_THREADS`, so it prints 1 there;
    and DAMNIT lands on node classes of 20–128 logical CPUs (the Cluster section), so read the
    count on the node, never assume it.
11. **DAMNIT annotations break under `from __future__ import annotations`.** DAMNIT's
    dependency injection reads `func.__annotations__` and matches the `var#`/`meta#`/`mymdc#`
    prefix. Under PEP 563 an annotation becomes its own *source text*, so `"var#x"` arrives as
    `"'var#x'"` — quotes included — the prefix no longer matches and every dependency silently
    resolves to nothing. Never add that import to `src/amore/context.py`, and pass
    `dont_inherit=True` when compiling it (`compile()` inherits the caller's future flags).
12. **A config hash must cover only what changes a stored number.**
    `config_hash` gates resume and gates reopening a file. It originally covered `n_workers`,
    `trains_per_block`, `selftest_frames`, `output_root`, `allow_incomplete` and `overwrite`, none
    of which can change a value in `/frames` — so a pass run at 36 workers wrote a file the same
    pass at the default worker count refused, and a file written with `overwrite=True` could never
    match a later `overwrite=False` run and was refused every time. `analysis.common.config`
    excludes those six for **both** passes; everything else stays in, including
    `base_mask_trains` and `cell_sample_trains`, which look operational but choose which trains are
    sampled and so change the masks and the measured readout noise.
    **A pass may exclude more, and the WAXS one does:** `WAXS_OPERATIONAL_FIELDS` adds
    `expected_lit_cells`, which only *gates* — it refuses a run before the output file is opened,
    so it cannot change a stored number, and pinning it for a reprocess must not invalidate files
    written without it. `lit_fraction_min` and `lit_gap_ratio` are the opposite case and stay in:
    they decide which cells are integrated, and so which rows exist at all.
    The set is therefore asked of the config (`cfg.operational_fields`, on the `PassConfig`
    protocol) and stored in provenance as `config_operational_fields`; **read a file's set off
    that attribute, never off the shared constant**, which may have moved since. The operational
    values themselves are recorded in full, so a stored file explains its own compatibility rules.
    **Files written before 2026-09-13 carry the old hash and must be reprocessed**, not resumed —
    and for `jungfrau_waxs_*.h5`, before **2026-09-14** as well, when `lit_gap_ratio` and
    `read_noise_fallback_kev` joined the config and `lit_fraction_min`'s default moved.
13. **h5py attrs outlive their file.** An `AttributeManager` kept past its `with` block does not
    raise on `.get`; it returns the default, so a provenance value reads as absent. Read every
    attribute inside the block.
14. **Silent failures in the old pipeline.** pasha forks from a non-main thread; `ThreadPoolExecutor`
    futures are never checked; `mp.Queue.empty()` is racy. Failures end as NaN rows.
15. **A beam centre only means something in its own corner-array frame.**
    `geom.to_pyfai_detector()` and `geom.to_distortion_array()` place the same pixels at different
    coordinates — origins 138.4 mm / 121.6 mm apart — and `setFit2D` interprets its `centerX` /
    `centerY` in whichever array the detector currently holds. So `setFit2D(sdd, px, py)` on a bare
    `to_pyfai_detector()` silently puts the beam somewhere neither convention intends (1.0 nm⁻¹
    off, measured). Replace the corners and set the centre together, or do neither. Nothing raises
    either way: both give a plausible q range and a plausible-looking I(q).
16. **`extra_speckle.Setup(geom=None)` builds geometry from motor encoders.**
    `DetectorAGIPD1M._init_geom` falls through to `geometry_from_encoders(run)`, whose three
    nested bare `except:` clauses end in `"Encoder positions not found, setting all values to 0"`
    and hardcoded quad positions. A `Setup` that looks configured can be running nominal geometry.
    Read its printed output; pass `geom=` to mean a specific file.
17. **pyFAI rebuilds its sparse matrix whenever the mask changes.** `setup_sparse_integrator`
    keys the cached matrix on a checksum of the mask and its own docstring calls the rebuild "a
    very time consuming operation". So a *per-frame* mask passed to `integrate1d(mask=)` rebuilds
    the full-split matrix every frame — measured 16.8 vs 2.3–2.9 ms/frame on a JUNGFRAU module. On
    float data the fix is to build the engine once with the static mask and carry the per-frame
    mask as **NaN in the data and variance arrays**: pyFAI's preprocessing drops a non-finite pixel
    from the numerator *and* the normalisation, giving bit-identical sums (measured 0.0e+00 max
    relative difference on all three sums, both detectors, all 500 bins). Integer data cannot carry
    NaN, which is why AGIPD needs its sparse denominator correction instead.
18. **Never freeze an array pyFAI might own, and never hand it a read-only one.**
    `np.ascontiguousarray(x, dtype)` returns `x` *itself* when it is already contiguous and of
    that dtype, so `arr = np.ascontiguousarray(ai.solidAngleArray(shape), np.float64);
    arr.flags.writeable = False` freezes pyFAI's own `_dssa` cache. Separately, pyFAI's Cython
    kernels acquire *writable* buffers: passing a read-only `mask`, `data` or `variance` raises
    `ValueError: buffer source array is read-only`. **Whether the mask is tolerated is
    platform-dependent** — the macOS wheels accept a read-only mask and the Linux ones on Maxwell
    do not, so this passes every local test and fails on the cluster. Copy before freezing
    (`analysis.common.arrays.frozen_copy`, whose `cast=` says which of the two behaviours is wanted) and keep one writable
    mask on the operator for the hot loop.
19. **`extra_speckle.saxs.get` hard-wires npt and the split scheme.** `_apply_pyfai` passes
    `npt=300`, so `get(..., npt=500)` raises on the duplicate keyword rather than rebinning; and
    `get` consumes `method` for its own `"1d"`/`"2d"` switch, so pyFAI's method can never be passed
    through and it always runs the default `("bbox","csr","cython")`. Its default unit is `q_A^-1`.
20. **A frozen slots dataclass pickles by *position*.** Both configs are
    `@dataclass(frozen=True, slots=True)` and both are pickled into a spawned pool, and for
    frozen+slots `dataclasses` installs a `__getstate__` returning a bare list of field values
    and a `__setstate__` that zips it back onto `fields(self)`. So a field added to a config
    shifts every later field of a state pickled by an older import, **silently**. Adding
    `lit_gap_ratio` and `read_noise_fallback_kev` on 2026-09-14 did exactly that to a notebook
    kernel holding the previous import: 26 values arrived at 28 fields, and the workers got
    `read_noise_kev` 1000 keV, `max_abs_kev` 8 keV, `min_modules` `None` and `allow_incomplete`
    unset. Only the last raised — a `TypeError` from inside `extra_data`, naming nothing that
    leads back to the config — and the worker's `operator_sha256` check passed, because every
    field `build_operator` reads sits before the insertion point. Both configs now define
    `__getstate__`/`__setstate__` in their **own class body** (a mixin does not work:
    `dataclasses._add_slots` installs the positional pair unless `__getstate__` is in the class's
    own `__dict__`), delegating to `analysis.common.config.state_by_name` / `restore_by_name`,
    which key the state by name, refuse a state whose names are not the class's rather than
    filling a missing field from its default, and re-run `__post_init__` — which pickle otherwise
    bypasses entirely, so an unpickled config had never been validated.
21. **The value check drops pixels, not frames — and `DATA_CHECK_FAILED` now means something
    much narrower.** `analysis.waxs.integrate.extreme_pixels` returns the pixels the union mask
    does *not* already remove whose value is non-finite or outside `±max_abs_kev`;
    `integrate_frame` ORs them into a local copy of `bad` and integrates anyway, counting them into
    `frames/n_extreme_pixels`. `DATA_CHECK_FAILED` survives only for a frame with **every** pixel
    excluded. v1 failed the whole frame instead, and on r0480/jf1 that discarded 473 frames of
    126 230 good pixels each over a median of 3 bad ones — NaCl Bragg spots, so the loss tracked
    crystallisation. Two consequences when reading a file: a frame's ring bin reads **low** wherever
    `n_extreme_pixels > 0`, so filter on it before treating that bin quantitatively; and `max_kev`
    (over the union mask) beside `max_kev_static` (over the static mask) is what says which pixel
    went and what it held.
22. **A uv venv works only for whoever can read its interpreter.** uv installs Python into
    `~/.local/share/uv/python` by default, and a venv holds no interpreter: `bin/python` is a
    symlink and `pyvenv.cfg` `home=` points at its standard library. Maxwell home directories are
    `drwx------`, so for anyone else — colleagues, and DAMNIT's `xdamnprd` — the link dangles.
    `activate` still puts `.venv/bin` first on PATH, the shell skips the dead link, and `python`
    silently resolves to `/usr/bin/python` (3.9); for the builder everything works. uv 0.11.16
    reads the install location from `UV_PYTHON_INSTALL_DIR` only (a `python-install-dir` key in
    `uv.toml` is a parse error), so build shared venvs with `scripts/shared_env.sh`, which pins
    `--python` to `usr/Software/uv-python` and sets `umask 022` and `UV_LINK_MODE=copy`. Once a
    venv exists, `uv sync` keeps its interpreter; the trap is recreating one. Also: dropping a
    dependency can drop an **undeclared** import of another package — `extra_speckle.xcca`
    imports `numba` without declaring it, and losing pyBeamtime (which brought numba) broke the
    DAMNIT context file. Check it with ctxrunner on a scratch copy (pitfall 25), **not**
    `damnit read-context`, which was the advice here until 2026-10-13.
23. **Corrected JUNGFRAU data are keV with a zero level nobody removed.** Proc applies offset,
    relative gain and bad pixels, but no common mode. p010646 measured the zero-photon level jumping
    0.13–0.16 keV rms from train to train per module/ASIC, in dark and lit cells alike, plus static
    offsets per 64-column ADC group (±0.02 keV with a same-day dark, up to ±0.17 keV with an old one),
    and saw them as steps in I(q) (`../transfer_context/03`, `04`). **Measured here (W7,
    2026-10-13, r0423 and r0370, 100 trains):** jf1 has that common mode, 0.13–0.16 keV rms
    from train to train, 95 % of it shared by the whole module; jf2 has none. On r0423's
    ~5.5 keV/pixel/frame (~0.6 photon) it was 3.8 % rms of jf1's per-train intensity. Lit cells
    also sit +0.127 keV above the dark cells of the same train, on both detectors. The −0.16 keV
    "dark mean" first quoted here was one train of jf1's common mode; over 100 trains it is
    −0.04 / −0.00 keV. σ_read as measured (one std over pixel × dark cell × train around one
    global mean, `waxs/cells.py`) lumped the common mode in: 0.354 / 0.319 keV against a pure
    per-pixel 0.295 / 0.299. **The pass now photonises both detectors per ASIC** — the per-frame
    zero level subtracted, then `rint(x/E)` (open task 18, `context/jungfrau-photonising.md`).
    Rounding alone would not have removed it: a 0.1 keV level error moves ~1 % of the one-photon
    pixels, because charge sharing fills the valley at E/2.
24. **JUNGFRAU in dynamic gain: a pixel that switched gain carries no trustworthy value.** r0423
    ran `gainMode` dynamic (the JUNGFRAU row under Other sources). p010646 saw gain-switched
    pixels read ~530 "photons" at 15 keV (≈ 7.9 MeV) whatever the signal. Our D6 band, 5.6–8.2 MeV with 93 % piled
    near a hard edge (open task 16), is **consistent with** that, not proof. The pass already
    excludes those pixels (pitfall 21); never treat their values as intensities, and count them
    apart in any per-pixel analysis.
25. **`damnit read-context` is not a syntax check: it adds columns to the shared `runs.sqlite`.**
    To test a context change, copy the file to scratch and run the deploy venv's
    `damnit/ctxsupport/ctxrunner.py` there: `ctx context.py out.pkl` evaluates it with no DB, and
    `exec <proposal> <run> raw --var <name>` runs chosen variables, writing `extracted_data/` in
    the cwd only (`MPLBACKEND=Agg`). **Run it the way DAMNIT does** (damnit 0.4.0, checked
    2026-10-13): `python -m ctxrunner …` with `PYTHONPATH=<damnit>/ctxsupport` and the context's
    directory as the cwd, which is how `from analysis_helpers import …` resolves. **`ctx` exits 0
    even for a broken context**: the failure is in the pickled `error_info`, so read that.
    `exec` needs `--cluster-job` for `cluster=True` variables (skipped without it); `--match` is a
    case-insensitive title substring, and selecting a variable re-runs all its dependencies.
26. **A DAMNIT cluster variable can be launched twice for the same run.** Requesting a variable
    that depends on one (`agipd_iq_overview` → `agipd_saxs`) can relaunch it while another
    instance runs; two jobs then write one file, and the status can still look done while HDF5
    reports `addr overflow` or `truncated file` (p010646). Here the pass has no lock and
    rewrites `operator.npz`, `masks.npz`, `/operator`, `/masks` and provenance even when the run is
    complete; the **live** context also passes `overwrite=True` (pitfall 27), so every relaunch
    truncates. **Fixed 2026-10-14** (open task 17, `context/damnit-integrity.md`): each pass
    takes an `O_EXCL` lock file beside its output (both passes through
    `analysis.common.run.guarded_pass`); `is_complete` (read-only) runs before any
    setup and again under the lock, so a complete run is answered from the file with nothing
    written; reprocessing moves the output aside (`scripts/reprocess_safe.py`) instead of
    `overwrite=True`. `allcpu` requeues a preempted job under the **same job id**, so the lock
    records `SLURM_RESTART_COUNT` and a requeued attempt takes over the lock its first attempt
    left. Diagnose a suspected double write with `sacct` (two overlapping jobs for the run) and
    by test-reading every dataset.
27. **DAMNIT reads `usr/Shared/amore/context.py`, not `src/amore/context.py`.** Nothing copies one
    to the other. On 2026-10-13 they differed: the live file calls `agipd_saxs(..., overwrite=True)`
    and `jungfrau_waxs(..., npt=200, overwrite=True)`, and imports the old `analysis_helpers` v2
    from its own directory, so `tests/test_context.py` passes on a file DAMNIT never runs. Others
    edit the live file too. The user reconciled the two hunk by hunk on 2026-10-13
    (`context/damnit-integrity.md` §5); **the repo copy is authoritative**, and JF's `npt` 200
    lives in `analysis.waxs.damnit.DAMNIT_OVERRIDES`, not in the context. Install only through
    `scripts/install_context.py` (backup, diff, sha256 check against what was last installed).
28. **`srun` without `-n 1` on a whole-node partition starts one copy per CPU group**
    (`srun -p upex -c 24` started 128 // 24 = 5 copies writing the same files, p010646). Submit with
    `sbatch` and pass the worker count explicitly; tools that default to
    `os.cpu_count()` take every logical CPU (pitfall 10). Also: `squeue --clusters=all -j <id>`
    **exits 1 even for a running job** (the other clusters print "Invalid job id"), so read its
    stdout, never its return code; and a finished job leaves the listing after `MinJobAge`, 90 s.
29. **Importing `extra_speckle` switches off HDF5 file locking for the whole process** (the DAMNIT context itself no longer imports it since 2026-10-19; `analysis_helpers` or anything else that does still has this effect).
    `extra_speckle/xcca/tools.py:2` sets `HDF5_USE_FILE_LOCKING=FALSE`; while the DAMNIT context
    imported extra_speckle, HDF5 inside DAMNIT never refused a second writer. The output lock
    (pitfall 26) does not rely on HDF5 and is the only guard there.
30. **The AGIPD operator's sha256 depends on the node's instruction set.** AVX2 nodes (EPYC 75F3,
    E5-2640 v4, where the production files were written) build `2bec8d7c…`, AVX-512 nodes (EPYC
    9374F, Gold 6140) `a59609c8…`: `omega` differs by 1 ULP on 105 693 pixels; `coef`, `bins`,
    `indptr` and `q` are bit-identical (pyFAI 2026.5 and 2026.9). So never compare operators by
    hash across nodes: `is_complete` compares them array by array (`operator_differences`).
31. **A virtual CXI/VDS file fails silently once its sources are gone.** It points into proc;
    when proc is taped, a missing AGIPD module reads as fill values — zero photons — with no
    error. Check that every source file exists before trusting a value, and treat a VDS as a
    disposable index, never as a product (`context/performance.md` §3).

32. **Renaming a DAMNIT variable leaves the old column behind, under the same title.** A `Group`
    renames every variable (`agipd_saxs` → `agipd.saxs`). The old column and its data stay in
    `runs.sqlite` and the GUI, are never recomputed, and read like the new one (the r163 figure
    that "looked like the rerun" was `xpcs_kbar_plot`, all trains). Readers by name go through
    `analysis.damnit_names.stored`, whose fallback to the flat name is transitional and can hide a
    missing new value. Remove stale columns with `scripts/prune_damnit_variables.py`.
33. **Concurrent writers make healthy files look broken.** Array jobs, the listener (`xdamnprd`) and
    other people's jobs write `runs.sqlite` and the per-run `.h5` together. Seen on 2026-10-19:
    `database is locked` (DAMNIT's own `BEGIN IMMEDIATE`, and the hit rule's read), `bad object
    header` and `addr overflow` on a file read mid-write, 22 runs whose job died before writing, and
    `*.writing.h5` temp files left by killed jobs. The files were intact (every object read). Judge
    them only once nothing was modified for 8 minutes; use `--concurrent-jobs` 3–5; remember
    `reprocess_safe` refuses while any array is active, one's own included. A mesh block's hit rule
    reads its neighbours' files and now retries, then fails the run, instead of dropping a neighbour. On 2026-10-28 the retry still missed h5py's `RuntimeError` ("bad local heap signature") and h5netcdf's `AttributeError`, so r178–182 got no hit set, and an h5py `KeyError` would have dropped the neighbour silently. Since then every read failure but a missing file is retried, "no agipd_saxs" included, because a partial file can look like one.
34. **A reprocess reuses what it finds complete.** Window sums, photon histograms and the XPCS
    photon store were "never moved", so a "fresh" run read them. `reprocess_safe start --fresh` moves
    them aside with the output; stale locks go with a directory renamed aside. Check the new files'
    mtimes against the launch time before believing a run was clean (`context/damnit-integrity.md` §8.3). The converse: a change
    that only re-merges the XPCS sums (rings, thresholds, kernel) must **not** pass `--fresh`, which
    moves the photon store aside too, so every hit train is read from proc again (and `reprocess_all.sh
    xpcs-hits` passes it).
35. **`train_id_coordinates()` cannot check frame identity.** After `select_trains(by_id[[t]])`
    EXtra-data builds the coordinate from INDEX counts, so it always equals `t`, even when the
    frames' own `image.trainId` / `data.trainId` rows name another train. Until 2026-10-28 the
    SAXS, XPCS-extraction and WAXS workers (and the WAXS self-test and cell classifier) checked
    labels this way, so the check could never fail. Every worker now reads the frames' own
    trainId rows (about 16 ms per AGIPD train on r0423).

---

## Open tasks

| # | Task | Next step / gate |
|---|---|---|
| 1 | AGIPD SAXS integrator (`agipd_saxs`) | **P1–P4 done.** P4 accepted on r0423 2026-09-11: 465 000/465 000 frames OK, pooled I(q) within 4.9e-8 of a dense pyFAI reference, 6.29 min on 36 workers against > 1 h for `analysis_helpers.integrate_run`. **P5 implemented**: `agipd_saxs` / `agipd_iq_overview` in `src/amore/context.py` keep their names and columns, now backed by `analysis.saxs.damnit` instead of `analysis_helpers.integrate_run` — the column's contents change from Å⁻¹ I0-divided to nm⁻¹ undivided, so clear it for runs processed before this. **The beam centre moved on 2026-09-13** (open task 4), which moves the q axis again and changes every stored operator hash — anything integrated before that date must be reprocessed, not merged. Next: run it on r0423 and r0426, then P6 (36 vs 72 workers). **Per-pixel window sums (§15), S1 implemented 2026-10-07:** the pass also writes, per 10-train window (a train-id range), `counts` and `valid_frames` per pixel — `analysis.cache.detector_sums` exactly, checked window by window in `tests/saxs/test_pixel_sums.py` — to a separate file with its own hash, never truncated by `overwrite`. `run_pixel_sums` writes only that file, for runs whose 1D file exists; `analysis.saxs.pixel_sums.detector_sums` reads it as the cache does. **S2 done 2026-10-07** (max-cfel029, EPYC 9374F, 36 workers, r0423): sums cost **+11.5 % wall** (174.7 vs 156.7 s warm; 1.25 ms/frame/core in the workers), every window equals `detector_sums` exactly (gate E, 300 windows), the window-pooled frame table matches to 3.5e-8 (E2), and all P4 gates pass again under the 2026-09-13 beam centre. 280 MB for r0423. **S3 done 2026-10-07:** the reprocess of every run at `npt=500` wrote the window sums too; `scripts/pixel_sums_audit.py` finds all 459 AGIPD runs complete and in exact agreement with their frame tables, nothing to backfill — 62 111 windows, 53 GB. All 459 `scratch/agipd_saxs` files were rewritten at `npt=500` on 2026-10-07 (before that, `npt=2000`). **2026-10-14:** the pass now also writes per-frame photon histograms (open task 20), takes the output lock and returns early when complete (task 17). With pyFAI 2026.9 merged, every stored file reads "not complete" (pyFAI is in the version gate), so the **bulk reprocess through `scripts/reprocess_safe.py` is due**, with the user's go **2026-10-19:** the bulk reprocess is done, fresh (no cache reused) and checked: AGIPD 459 of 461 runs complete (r0356, r0382 have no output), window sums and photon histograms rewritten with the frame tables. Columns are now `agipd.*` (`context/damnit-integrity.md` §8). |
| 2 | WAXS JUNGFRAU integrator (`<pkg>.waxs`) | **W0–W4 done; W5 implemented but never run; both W4 output files invalidated by a 2026-09-14 config change.** `analysis.common` carries the detector-agnostic layer (status, row model, mask vocabulary, CPU/pool helpers, frame-table writer); `analysis.waxs` the pass; `analysis.waxs.combine` the two-detector scaling. **Four spec decisions were changed by measurement:** D5′ — the per-frame mask is a NaN, not an `integrate1d(mask=)` argument (8× faster, bit-identical, pitfall 17); D3 — the variance is stored unclamped, and σ_read now has three sources (configured → measured → per-detector fallback), because 213 runs read all 16 cells and have no dark one; D6 — its stated premise was wrong (see the JUNGFRAU row); **D4′ (2026-09-14) — the lit-cell gate is the *shape*, not a named set.** A sweep over 746 (run, detector) results found four readout patterns, so `expected_lit_cells` became an optional pin and the run-invariant check is `is_storage_cell_sequence`. The old `lit_fraction_min` of 0.10 was a fraction of scattered intensity and so moved with the beam; it produced 67 impossible sets such as `[1, 12]` and is now 4.805e-4, the centre of the one empty band in the pooled sample. **The sweep's JSON is not in the repo** (working rule 4) — re-run or locate it before quoting those numbers elsewhere. **W1** (r0423 + r0426 × jf1 + jf2): files, sources and q ranges confirmed; `data.mask` train-invariant; `roi` ruled out by the chunk layout. **W4** (max-exfl484, 36 workers, r0423, both detectors): 24 000/24 000 frames OK each, self-test exactly 0.0, independent reference 6.5e-08 / 8.0e-08, **17.5 / 20.1 s** wall — targets, not a current verdict, since the hash moved. **Detector cross-check:** `combine_files` fits jf2 onto jf1 at **0.98823 ± 0.00005**, so the two independent PONIs and masks agree on absolute I(q) to 1.2 % — χ²ᵣ 54 is a 0.74 % systematic against a 0.10 % claimed error, not a disagreement; leading suspect is polarisation (open task 5). **Next: rerun W4 on both detectors** (the files on scratch will be refused, not resumed — the 2026-09-14 `max_kev_static` column makes that a `SchemaMismatch` as well as a hash mismatch), then W5 — the four DAMNIT variables on r0423, then r0426, then in bulk. A bulk reprocess now survives a dark run: `jungfrau_waxs` and `combined_curve` return `None` rather than raising. Pre-flight `scripts/w1_facts.py` over the full run list to see which block each run belongs to. **2026-10-13/14:** the W4 rerun is done, in keV and in photons (r0423, max-exfl226, 36 workers; `context/jungfrau-photonising.md` §4); the pass now photonises (task 18), and DAMNIT runs it through `jungfrau_preprocessing` at `npt` 200 (`DAMNIT_OVERRIDES`; 836 of the 839 stored files are at 200). The bulk reprocess follows task 1's **2026-10-19:** reprocessed with the rest (JF 418 per detector complete, 43 dark); columns are `jungfrau.*`. |
| 3 | Lit-frame selection | LITFRM source/keys from `lsxfel`; compare with `XrayPulses` counts per train |
| 4 | AGIPD geometry source of truth | **Beam centre wired, not settled.** `agipd_saxs` now applies the agreed `(607.46, 672.08)` via `set_pixel_corners(to_distortion_array())` + `setFit2D`, paired with `geom_latest.geom`. That pairing merges the r488 fcc doublet (0.584 + 0.633 → 0.610 nm⁻¹), so it is the pairing that needs confirming, not the code. Note the old pipeline and `extra_speckle.Setup` agree with each other because they share this construction — and `Setup` with `geom=None` silently builds from motor encoders (`geometry_from_encoders`, bare `except:`, falls back to hardcoded quad positions), so it may not be the same geometry at all. Decisive test: `integrate2d` and check whether I(q,χ) on the 0.633 nm⁻¹ ring is flat in χ or sinusoidal — amplitude and phase give the displacement and its direction. Then resolve the encoder source (absent in r0500) and CrystFEL file vs `geom.offset()` |
| 5 | Polarisation correction | Confirm detector-frame ↔ lab-horizontal orientation and factor with MID. **The ≤ 5.4e-4 figure is the AGIPD one and does not transfer to WAXS**: pyFAI's azimuthal modulation amplitude is sin²(2θ)/2, which over the JUNGFRAU overlap runs 3.1 % at q = 11.5 nm⁻¹ to 7.8 % at 18.4 — two orders of magnitude larger, and the leading suspect for the 0.74 % shape difference between the two detectors on r0423 (`analysis.waxs.combine`). Applying it and watching that residual is now a concrete test. **A second suspect (2026-10-13):** the uncorrected JF zero level (pitfall 23) — −0.16 keV on jf1 against −0.00 on jf2 in the dark cells, an additive offset that becomes a shape difference (offset / I(q)) and a scale difference at once. Re-test the 0.74 % residual and the 0.98823 scale after photonising (open task 18), before or alongside polarisation. **Result 2026-10-13:** photonising moved the factor to 0.99076, χ²ᵣ 54.1 → 31.4 and the residual 0.74 → 0.64 %, but left the **tilt unchanged** (2.57e-3 → 2.56e-3 /nm), so the zero level is not the cause. **Polarisation is excluded too**, whichever way the detectors are mounted. The low-q deficit is a **droplet-dependent shadow on jf1's beam-side columns** (from ~820, down to −58 % at the outermost); masking columns ≥ 832 removes it. Write-up `context/jungfrau-overlap-tilt.md` on branch `claude/peaceful-gauss-05e78f`, **not merged**. Pending the user: a new jf1 static mask file (never overwrite `jf1_mask.edf`) |
| 6 | Correction settings | Photon threshold / recast settings for r0423, r0426 from the correction reports |
| 7 | XGM normalisation recipe | Pulse-index alignment (LITFRM `data.xgmPulseId`), applied post hoc to stored sums |
| 8 | Anisotropic low-q lobe (r0423) | **In `agipd_saxs`: resolved.** Excluded by the one static pixel mask (see Paths), OR'd into `static_bad` — integrator I4, option (a). No φ-sector logic and no per-region sums, so the φ ≈ 278–330° values from the old integrator frame need no re-derivation. **Still open as a separate analysis:** lobe amplitude (decays ~16 % over 300 s in the lowest q band, independent of the isotropic drift) vs droplet volume. It needs its own pass, because the `agipd_saxs` sums no longer carry the lobe |
| 9 | Transmission correction integration | `droplet_transmission.py` on stored sums; V/V₀ pairing (r0423 ↔ r0464) |
| 10 | XPCS | Define q-binning; custom g2 model with KWW fitting (extra-speckle lacks it); shear vs diffusion diagnostic |
| 11 | XCCA second pass | ROI list excluding Bragg q; per-shot masks; `AveragedAngularCorrelationMasked`; bulk chunk reader (integrator I3). Low priority (user, 2026-10-01). **Lessons from p010646's fxs** in `context/extra-toolkit-context.md` §2.4 — above all, a neighbour (adjacent-train) correction also removes real signal that persists across trains, which in a levitated droplet is exactly the pre-Bragg order this experiment looks for |
| 12 | Tier-2 data access | extra-speckle orchestration without materialising full detector arrays. Virtual CXI files were measured as a route and **not adopted** (pitfall 31, `context/performance.md` §3): 10–30 % cheaper read one frame at a time on AGIPD, 12× dearer a train at a time, slower on JF. The bigger AGIPD lever is still the bulk chunk reader (integrator I3) |
| 14 | Upstream issues | EXtra-data per-chunk lookup; pyFAI method-string resolution and OpenCL fallback warning; XCCA `from_dataset` double update |
| 16 | D6 value bound and r0480 | **Resolved 2026-09-14.** `scripts/w6_data_check.py` on 473/473 frames + 200 controls found none of the three predicted causes: the extreme pixels are unflagged and do reach the integrator, but they are a **separate population** — passing frames stop at 131 keV, failing ones run 5629–8234, a 43× empty band with the 1000 keV bound inside it, 93 % piled within 10 % of a hard edge. **97.2 % lie within 1 % of q = 22.33 nm⁻¹ and the rest at 19.3: NaCl (200) at 22.28 and (111) at 19.30 (a = 5.6402 Å), in the 1893:54 ratio rock salt gives**, and the rate rises through the run (deciles `[0,7,0,78,26,13,27,63,107,152]`) as the droplet evaporates. **Confirmed on jf2**, which spans q = 9.8–18.5 — below both reflections — and integrates r0480 completely. **Fix shipped:** the pixel is excluded and the frame kept (pitfall 21); the bound stays at 1000 keV, now with a traceable source. **Accepted cost, deliberately:** an affected frame's ring bin reads low, and the affected-q range is *not* stored per frame — this dataset is for kinetics, not XCCA/XPCS, so `frames/n_extreme_pixels` is the filter. **Left open, not blocking:** whether values above the band are clipped or real needs the **raw** gain stage (proc does not carry it); the same pixel gives 14 distinct values across 14 frames, so it is not a fixed per-pixel clip. **2026-10-13:** r0423 ran JUNGFRAU in **dynamic** gain, and p010646 found gain-switched pixels untrustworthy at ~7.9 MeV-equivalent whatever the signal (pitfall 24) — consistent with this band, which makes "clipped or real" lean towards "neither: gain-switched". The raw gain stage would still be the proof |
| 15 | Photon energy, 9.04 vs 9.000 keV | **Settled 2026-10-13: 9.04 keV is real** (user: the undulator spectrum and the beamline staff); the XGM's and the mono's 9.000 keV are setpoints. No reintegration needed. History: `cfg.photon_energy_kev` (9.04, from this file) sets the wavelength and so the q scale; `XGM.photon_energy_by_train()` reports 9.000 keV nominal on r0423, constant across all trains. The 0.44 % gap shifts every q by 0.44 % — 0.003 nm⁻¹ at the 0.6 nm⁻¹ Bragg peak, 0.005 at q_max. `plan.run_checks` records both and warns. Settle which is authoritative with MID; it is a config change plus a reintegration, not a code change. p010646's lesson (`../transfer_context/02`) is to read E from the run: there the proposal's 18 keV was 15 in the data. Photonising JF (open task 18) records the E it divides by per frame, so it does not settle this either way |
| 17 | DAMNIT and pass integrity (session S1, 2026-10-13) | From the p010646 comparison (pitfalls 26, 27). (a) `output_lock`: an `O_EXCL` lock file per output, with stale-lock takeover when the holder's job is gone; (b) `is_complete`: a read-only check (hash, schema, no `NOT_PROCESSED`/`WORKER_ERROR`, pyFAI version and operator sha256, window sums) at the top of `run_agipd_saxs`/`run_pixel_sums`/`run_jungfrau_waxs`, so a complete run writes nothing; (c) resume retries `WORKER_ERROR` (`block_complete` counts it as done today); (d) reconcile the live and repo contexts hunk by hunk with the user and drop `overwrite=`; (e) `scripts/install_context.py` (ctxrunner on a scratch copy, sha256 guard, backup + diff); (f) `scripts/reprocess_safe.py` (rename aside, refuse busy runs, `damnit reprocess --match`, check, cleanup). Acceptance: mock tests for each; on one real run, re-running `agipd_iq_overview` leaves `agipd_saxs.h5` untouched. **Done, merged 2026-10-14** (`context/damnit-integrity.md`): lock with requeue takeover (`SLURM_RESTART_COUNT`), `is_complete` comparing operators array by array (pitfall 30), `WORKER_ERROR` retried, the context reconciled with the user, both scripts. Real-run gate passed (Slurm 25293843, r0423: answered from the file, every file's inode, mtime and sha256 unchanged). **Left, with the user's OK:** `scripts/install_context.py --install` (the live file is still the 2026-10-07 one, with `overwrite=True`), then the bulk reprocess. The XPCS variables' `"Incomplete"` handling is a proposal for the XPCS owner (`context/xpcs-results-proposal.md` on branch `claude/gracious-wing-15f275`, not merged) **2026-10-19:** the live context is the grouped one (`install_context.py --install`, commit `7ca24e2`); `reprocess_safe start --fresh` moves companion caches aside; `scripts/prune_damnit_variables.py` removed the 37 stale columns; the failure modes of that day are pitfalls 32–34. |
| 18 | Photonise JUNGFRAU in the pass, jf1 and jf2 (session S2, 2026-10-13) | Pitfall 23; user decision 2026-10-13: both detectors, in the pass, settings recorded. First `scripts/w7_photonize_check.py` fixes the parameters at 9 keV (zero-level estimator per ASIC: median below 2 keV, then 3× the mean within ±1 keV, 3-px edge excluded — p010646's, validated bit for bit against fxs; dark-cell level rms and ADC-group offsets; cell-7 leftover; over-dispersion D; keV/(E·photons) along q and along a run). Then `clip(rint((x − level)/E), 0)` per ASIC before integrating, with `asic_level (n, 8)`, `n_level_missing`, E and `units = "photons"` stored, a Poisson variance, and readers that refuse to mix keV and photons. Old outputs renamed `*_kev.h5` first. Acceptance: the W4 rerun (already due) on r0423, both detectors, plus the jf1/jf2 overlap before/after (task 5). **Done, merged 2026-10-14** (`context/jungfrau-photonising.md`): W4 gates pass in photons (self-test 0.0, reference 7.4e-8 / 7.5e-8, 24 000 OK, no negative bins, σ ratio 1.012 / 1.002); photonising costs ~24 ms/frame/core (+45 % wall, 52 / 49 s on max-exfl226). Rename to `*_kev.h5` happens at the bulk reprocess |
| 19 | Speed and tooling (session S3, 2026-10-13) | (a) **pyFAI 2026.9.0** (PyPI 2026-09-15; 2026.5.0 locked now). p010646 measured 2D integration 15–20× faster after upgrading from 2024.9. Re-check pitfalls 1, 2, 5, 17, 18 against the new source, compare the AGIPD operator and JF NaN-trick bit-identity, time both versions in one job on one DAMNIT node, re-run the P4 gates. Changing the operator means reprocessing, not resuming (task 17b). (b) **Virtual CXI files** (`AGIPD1M`/`JUNGFRAU.write_virtual_cxi`, EXtra-data 1.24): build cost, read rates against our per-frame EXtra-data path at 1 and 36 processes, frame identity fields, what breaks when proc is taped, fit with Tier-2 (task 12) — a decision note in `context/performance.md`. (c) `analysis.common.slurm` (`prop_path`, `submit` through sbatch, no `srun`) and `-w/-t/--local/--dry-run` for node scripts (deleted 2026-10-27 with the node scripts, no caller). (d) A progress line every 10 % in `fan_out`. **Done, merged 2026-10-14** (`context/performance.md`): pyFAI 2026.9 is bit-identical and **no faster** for anything this project runs (p010646's 15–20× was from 2024.9; we were on 2026.5), and every re-checked pitfall still holds; virtual CXI not adopted (task 12). A follow-up session measures P6 (36 vs 72 workers) |
| 20 | Diagnostics, XPCS quality, droplet videos (session S4, 2026-10-13) | (a) `analysis.diagnostics` from stored outputs only: frame accounting, SVD of per-train curves, χ² of each train against the run median, relative std before/after per-train XGM × transmission, JF cell–XGM correlation as a lit-cell cross-check (p010646: lit r ≈ 0.98, dark ≤ 0.2). (b) Per-run XPCS SNR from `analysis.xpcs.snr` (§4.6 of `context/run-triage.md`). (c) An XPCS quality check from photon statistics — per-ROI P(k) histograms, negative-binomial contrast against the TTCF intercept, k̄ stability, pile-up — specified first in `context/xpcs-quality.md`. The histograms need proc, which goes to tape, so they are a raw product to extract now. (d) `*_frames_used` DAMNIT cells coloured by status code, reading the file without depending on the cluster variable. (e) A labelled MP4 per run of `MID_EXP_SAM/CAM/CAM5` in `usr/cached_files/droplet_videos/`, since the camera images go to tape with raw. No ffmpeg on PATH and our interpreter is not the EuXFEL env, so take ffmpeg from `imageio-ffmpeg` or a module. **Merged 2026-10-14** (`context/diagnostics.md`, `context/xpcs-quality.md`): diagnostics; per-run XPCS SNR with a fixed reference β (median TTCF intercept of r450 and r493–496, ≈ 0.048 at low q to 0.019 at high q); droplet hits per train (`--phase classify`; r318 agrees with the water-band rule on 99.6 % of 705 trains); photon histograms inside the AGIPD pass; `*_frames_used` cells (all 459 AGIPD and ~420 JF tables green); droplet videos (r0423: 3002 images = 3002 frames, 1.0 MB, 16.6 s). **Open:** Q2–Q4 of the quality spec; the videos for every run (~1 node-h, 0.2–0.4 GB), with the user's go. Science so far: r0450's single-frame β is 0.84–0.91 of the reference at low q, r0423's 0.35–0.50 of r0450's (Q3 open) **2026-10-19:** `agipd.diagnostics`, `xpcs.photon_statistics`, `*_frames_used` and `xpcs.score` are filled for every run; the score now reads the hit results file (P6). `camera.video` / `camera.first_image` are still not computed for every run. |
| 21 | AGIPD setup per run range (deferred, 2026-10-13) | `usr/masks` holds beamtime masks for a JF shadow (05-09) and a beamstop (05-11), and the stray-light pattern changed between r433 and r443, yet one mask serves every run. Inventory from the window sums first (change points in the beamstop, shadow and near-beam regions), then run-range tables only where a range really differs (`../transfer_context/templates/run_range_table.py`: exactly one entry per run, a missing file is an error, never overwrite a file a processed run used). The JF geometry is out of scope: constant after r52, and earlier runs are not used (user, 2026-10-13) |
| 22 | Hit-train XPCS (`<pkg>.xpcs`, branch `claude/xpcs-hit-trains`, 2026-10-18/19) | Replaces `xpcs_offline` by a pass that reads only the droplet-hit trains: an extract-once photon store in `usr/cached_files` (outlives proc), exact integer sums, a reader that makes every float (`context/xpcs-hits.md`). **P0–P5 done in the worktree, not merged:** P4 on r0423 and r0493 to r0496 (G1 against extra-speckle: the elementwise gate holds for the TTCFs; g2 differs by 6.7e-8 absolute, **proved to be extra-speckle's float32** — it differs from its own float64 by the same, ours is 2.1e-9 from that; `Averaging.ES` applies its noisy-ring rule; `max_partner_gap` default 1 on B4's evidence). P5: `analysis.xpcs.damnit` (the only importer of the classifier) and `xpcs_hits_preprocessing` / `_g2` / `_ttcf` / `_overview` / `_frames_used` in `src/amore/context.py`; `tests/test_context.py` now states each preprocessing variable's dependencies (the hit pass is a second read of proc: it needs the whole run's `agipd_saxs`). **Left, with the user's OK:** merge, `scripts/install_context.py --install`, then the hits' first pass per run before its proc is taped; P6 (`analysis.xpcs.snr` (moved from `analysis.triage.xpcs` and `analysis.diagnostics.xpcs` on 2026-10-28) onto the reader, retire `xpcs_pipeline`); the open problems of §10 (serial block writes dominate extraction, the 5 mm envelope margin, r0423's hit set, r0450's missing signals, whether a mesh run needs a partner policy other than gap 1) **2026-10-19: merged and run on every run** (459 of 459 hit results complete). Added: the pixel-outlier mask of extra-speckle's statistic, computed from the hit trains and applied at the merge (results schema 2, `context/xpcs-hits.md` §12), every old extra-speckle figure and data column reproduced from the store (`analysis.xpcs.figures`, `.legacy`, `.detector`, `.g2fit`), the per-train TTCF for hits only (`xpcs.ttcf_per_train`), `xpcs.score` and the triage SNR on the hit store (P6), and the hit-recognition cells `hits.summary` / `hits.geometry`. `xpcs_pipeline` and the extra-speckle import are gone from the context. Open: whether `max_partner_gap` 1 costs too much on sparse mesh runs (r250 SNR −32 %), r0423's hit set, and the §10 problems. **2026-10-20:** twelve rings of 0.045 nm⁻¹ with the fcc 111 and 200 bands left out (`RINGS_NM`, `rings_nm` intervals, results schema 3), `REFERENCE_BETA` measured again for them, the outliers figure no longer draws pixels flagged in every train (`context/xpcs-hits.md` §13). Reprocess the XPCS **without `--fresh`** (it would move the photon stores aside). **2026-10-27:** the rings leave out `BRAGG_WINDOWS_NM` (0.57–0.65, 0.68–0.74 nm⁻¹, now defined only in `analysis.xpcs.config`): nine rings, 0.540–0.585, 0.640–0.685 and 0.725–0.770 dropped with their `REFERENCE_BETA` entries; results schema stays 3, but the ring sha moves the merge hash, so every stored twelve-ring file reads not complete until the XPCS re-merge **without `--fresh`** (`context/xpcs-hits.md` §14). |
