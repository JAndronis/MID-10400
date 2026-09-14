# JUNGFRAU WAXS integrator (`jungfrau_waxs`) — design and implementation guide

Context file for `<pkg>.waxs`. CLAUDE.md open task 2.

Read `CLAUDE.md` first, then `context/agipd-saxs-integrator.md` — this file is written as a *delta*
against that one and does not repeat its reasoning. Where a section here is silent, the AGIPD
design applies unchanged; §4 says exactly which parts those are.

Work phase by phase (§7). Present a plan before creating files. **W0–W4 are done and W5 is
implemented but unrun**; §7 carries the state of each, and §8 the testing conventions.

---

## 1. Scope

**In scope (v1).** For every *lit* JUNGFRAU frame of a run in the proc data, per detector:
per-q-bin 1-D sufficient statistics, a per-frame status ledger, full provenance, written to
scratch. Two JUNGFRAU-500Ks → two output files (§3 D2).

**Out of scope (v1):** gain-stage reconstruction (proc data is already corrected), AGIPD SAXS
(its own file), XCCA/XPCS, background subtraction and transmission — post hoc on the stored sums,
exactly as for AGIPD.

**Priority:** data quality > completeness guarantees > speed.

---

## 2. What the JUNGFRAU data actually is

Measured on **r0423, train 2637695397 — one train, both detectors.** "Kept region" means
`data.mask == 0` **and** `.edf == 0` (§6 O2: the `.edf` files are native pyFAI masks, so non-zero
is excluded); the whole-detector figures are given where they differ, because that difference is
itself a finding.

| Property | jf1 | jf2 | Consequence |
|---|---|---|---|
| dtype | `float32` | `float32` | **not** integer counts; `sparse.integrate_frame`'s dtype check rejects it, correctly |
| Units | **keV** — single-photon peak 8.87 | **keV** — peak 9.12 | 9.04 keV expected; charge sharing biases a photon peak low. The Poisson term is in keV, not counts (§3 D3) |
| Layout | `(train, cell, 512, 1024)`, 16 memory cells per train, fixed | same | rectangular, not AGIPD's ragged frame table |
| **Lit cells** | ~~0–7~~ **`{0…6, 15}`** | ~~0–7~~ **`{0…6, 15}`** | **Corrected — the `0–7` was array *positions*, not cell ids (§6 O4, CLAUDE.md pitfall 4).** Lit cells 42–53 % of pixels > 4.5 keV; dark cells ≤ 0.004 %. Integrating all 16 halves I(q) (§3 D4). And this set holds for the r0379–r0500 science block only: the proposal used **four** readout patterns (§6 O4) |
| Lit / dark mean, kept region | +5.67 / −0.16 keV | +5.32 / −0.00 keV | — |
| Readout σ, dark cells, kept region | **0.323 keV** | **0.317 keV** | the noise floor of the error model, and the two agree. Measured per run (§3 D3); a run that reads all 16 cells has no dark cell and falls back to a per-detector median |
| Exactly zero, lit | 0.002 % | 0.004 % | **dense**; no sparsity to exploit (§3 D1) |
| Negative, lit, kept region | 20.8 % | 16.2 % | 45.6 % / 31.8 % over the whole detector. Either way the "no negative counts" data check is wrong here |
| Dynamic mask set | 1.08 % | 1.08 % | same masking design as AGIPD |
| Bits present | {0, 1, 21, 22} | {0, 1, 21, 22} | 0 `OFFSET_OUT_OF_THRESHOLD`, 1 `NOISE_OUT_OF_THRESHOLD`, 21 `WRONG_GAIN_VALUE`, 22 `NON_STANDARD_SIZE`. Different from AGIPD's {0,1,7,8,9,12,13}; 21 is new |
| Bit 22 `NON_STANDARD_SIZE` | set, 0.97 % | set, 0.97 % | unlike AGIPD, where it is never set — so **no separate seam mask is needed** (§3 D5) |
| **Extreme pixels `data.mask` misses** | 10, **all dynamically flagged** | **15, none flagged** | up to ±1.8e5 keV, present in *every* cell. `data.mask` alone is not sufficient on jf2 (§3 D6) |
| max \|x\|, kept region | 73.5 keV | 64.2 keV | the extremes are all outside the kept region |
| Static mask `.edf` | uint8 0/1, (512, 1024), **75.9 %** non-zero | **84.4 %** non-zero | native pyFAI masks from silx view, so non-zero = excluded — pass straight to `mask=`, no inversion (§6 O2). One per detector, genuinely different |
| Usable pixels per cell, kept region | 126 125 | 81 767 | 24 % / 16 % of the module |
| Geometry | `jf1.poni`, dist 231.96 mm, `orientation` 4 | `jf2.poni`, dist 232.28 mm, `orientation` 3 | pyFAI `Jungfrau` detector, (512, 1024), 75 µm, Si 320 µm. PONI files, not an EXtra-geom object (§5 R4) |
| Wavelength in PONI | 1.371507e-10 m = **9.0400 keV** | identical | matches `cfg.photon_energy_kev`; a third independent vote in CLAUDE.md open task 15 |
| q range, whole detector | 10.0–24.0 nm⁻¹ | 5.0–20.4 nm⁻¹ | AGIPD SAXS covers 0.077–1.067 |
| **q range actually populated** | **11.5–23.7** | **9.8–18.5** | after masking. This is the range to quote, and it changes the gap in §6 O6 |

Both detectors agree on dtype, units, the lit-cell set, the bit set, the mask fraction and the
readout noise, which is what makes these numbers worth building on. Still **one train of one run**:
the lit-cell split and the bit set must be re-checked across runs, including a crystallised one,
before they are treated as run-invariant (§6 O4).

*That re-check happened, and the lit-cell row is the one it moved* — twice. First the set itself
(`{0…6, 15}`, not `0–7`, W1); then, over the whole proposal, the discovery that there is no single
set to name at all (§6 O4, 2026-09-14). The bit set, the mask fractions, the q ranges and the
readout noise all survived unchanged.

---

## 3. Decisions

**D1 — Dense pyFAI per frame. Not the sparse kernel.** 0.005 % of pixels are exactly zero and
45.5 % are negative: the gather/bincount path exists only because AGIPD frames are 0.7–1.8 %
non-zero, and here it would gather essentially every pixel while paying the indexing overhead. Use
`ai.integrate1d(..., method=("full","csc","cython"), variance=...)` per frame and keep
`res.sum_signal` / `res.sum_normalization` / `res.sum_variance` as the stored sums, so §9 of the
AGIPD file — pooling, post hoc scaling, the reducers — carries over unchanged.

*This is the single decision that differs most from AGIPD, and it is measured, not assumed.*

**D2 — One output file per detector.** `{output_root}/r{run:04d}/jungfrau_waxs_{det}.h5`, same
schema as the AGIPD file. The two detectors have different geometries, different static masks,
different q ranges and different orientations; a shared file would need a detector axis on every
dataset and a per-detector operator, for no gain. Combining happens **only at the plot** (§6 O5).

**D3 — Error model: `Var(x) = σ_read² + E·x`, in keV², stored unclamped.** `Σc²·x` is the Poisson
variance of *integer photon counts*; on keV-valued data with a 45 % negative tail it is not a
variance at all and can go negative. For a pixel holding `x` keV of deposited energy from `E`-keV
photons the photon count is `x/E`, whose Poisson variance is `x/E`, which in keV² is `E·x`; the
readout term adds `σ_read² ≈ 0.10 keV²`. Pass this per-pixel array to pyFAI as `variance=` — never
`ErrorModel.POISSON` (CLAUDE.md pitfall 2). **Confirm `E` and σ_read per run** rather than
hardcoding this train's numbers: `cells.CellAccumulator` measures σ_read from the dark cells of the
sampled trains and recovers 0.3230 / 0.3175 keV on r0423.

*Amended 2026-09-13, measured.* This expression is **negative for 23–26 % of kept pixels** (x
reaches −1.7 keV on r0423), and integrated it leaves **5 of 4000 occupied per-frame bins** with
`sum_variance ≤ 0` on jf1, none on jf2. Clamping the Poisson term at `max(x, 0)` removes every
negative bin but biases the noise floor upward on each negative-noise pixel, and **that bias does
not cancel on pooling** while the unbiased form's negatives do. So the sums are **stored
unclamped** — the same "store sufficient statistics, decide post hoc" rule as AGIPD §9 — and the
per-frame negative bins are counted into `/frames/n_negative_variance_bins`, where the fact stays
visible. `common.writer.pooled_per_train` gives σ = 0 for any bin whose *pooled* variance is still
non-positive and counts those in `attrs["negative_variance_bins"]`, rather than rooting a negative
number.

*Amended 2026-09-14: σ_read has three sources, in this order.* "Measure it per run" assumed every
run has a dark cell to measure from. 213 of the proposal's runs read all 16 storage cells (D4′) and
have none, so `run._error_model` falls back rather than raising:

1. `cfg.read_noise_kev` — an explicit value wins outright.
2. The run's own dark cells, which is what this decision asks for.
3. `cfg.read_noise_fallback_kev`, defaulted per detector from `DEFAULT_READ_NOISE_KEV` —
   **jf1 0.3440, jf2 0.3195 keV**, the median over the 159 (run, detector) results per detector
   where the dark side is unambiguous. jf1's spans 0.3153–0.3724 (±8.3 %), jf2's 0.3147–0.3370
   (±3.5 %).

`ErrorModel.source` records which of the three it was, so a stored file never has to be guessed at.

*There is no fourth branch that refuses.* The first draft of this had one — "no measurement and no
fallback" — and `ReadNoiseUnavailable` with it. It could not fire: `__post_init__` resolves
`read_noise_fallback_kev` from `DEFAULT_READ_NOISE_KEV`, so a constructed config never holds
`None` and there is no way to express "no fallback". Resolving it at construction is deliberate and
is what puts the number *actually used* inside `config_hash` — were the field left `None` and
filled in at use, a change to the constant would let resume merge blocks computed at two different
σ_read (CLAUDE.md pitfall 12). The exception was removed 2026-09-14 rather than made reachable,
because working rule 2's guarantee is already met by `ErrorModel.source`: every stored file says
which of the three it used, so "refuse anything that fell back" is a question the provenance
answers. The fallback is affordable because σ_read is a small lever: integrating the r0423 jf1
frames with it wrong by **+41 %** moves σ(q) by at most **0.19 %** at that run's occupancy and
**0.95 %** at the lower occupancy of the all-16 runs, against a per-frame σ/I of 3–7 %. It is in
`config_hash` all the same — when it is used, it changes a stored number.

**D4 — Lit-cell selection is mandatory, and it comes from the data.** Eight of the sixteen cells
hold no photons at all (0.00 % of pixels above half a photon, against 42–49 % in the lit eight —
array *positions* 0–7 as first recorded, cell ids `{0…6,15}` as later measured, §6 O4). Integrating all 16
would halve I(q) and add a dark-frame background. Unlike AGIPD's open task 3, this does not wait
on LITFRM: the split is unambiguous in the frames themselves. v1 selects lit cells by a measured
per-cell threshold, records the selected set in provenance, and **fails loudly if the lit set is
not what the config expects** — a silent change in the cell pattern between runs must not pass.

**D4′ — Amended 2026-09-14, measured over the whole proposal. The gate is the *shape*, not a named
set.** The sentence above has a hidden premise: that there is one pattern to expect. There is not.
Over **746 (run, detector) results spanning r0001–r0500** the proposal used four:

| pattern | cells | runs | n |
|---|---|---|---|
| 8 cells from 15 | `{0…6, 15}` | 379–500 | 114 |
| all 16 | `{0…15}` | 54–378 | 213 |
| 1 cell | `{15}` | 52, 53, 56 | 3 |
| none (no beam) | `{}` | 1–42, 68–69, 140, … | 43 |

Every one is a run of consecutive memory cells **mod 16** — a JUNGFRAU storage-cell sequence, set
by `storageCellStart` and `storageCells` on the control device. That is a far stronger thing to
hold the pass to than any one set, because it is a property of *every* valid readout pattern rather
than of this beamtime. So:

- `config.is_storage_cell_sequence` defines the shape, and `cells.CellClassification.check_structure`
  raises `ImplausibleLitCells` — a subclass of `UnexpectedLitCells` — when the measured set is not
  one. This runs on **every** run. A failure there means the *classification* is wrong, not that the
  run is unusual, and the message names `lit_fraction_min` and `lit_gap_ratio` because those are
  what to look at.
- `cfg.expected_lit_cells` becomes `None` by default and now only *pins*: set it to
  `EXPECTED_LIT_CELLS` when reprocessing the r0379–r0500 science block must refuse anything that
  has drifted. Because it refuses **before the output file is opened**, it cannot change a stored
  number, so by CLAUDE.md pitfall 12 it is excluded from `config_hash`
  (`WAXS_OPERATIONAL_FIELDS`) and pinning it does not invalidate files written without it.
- **A run with no lit cell returns `None` rather than raising.** No beam means no I(q), which is a
  result; 43 of the proposal's runs are like that, and raising would take a whole DAMNIT reprocess
  down on runs that are simply dark. `run._no_lit_cells` logs the brightest cell and its fraction
  against the floor, so the verdict carries its evidence. `plan.build_plan` therefore accepts an
  empty set too and plans zero rows — whether that is worth integrating is the pass's call, not the
  row model's — and `damnit.combined_curve` returns `None` when either detector produced nothing.

**Where the threshold comes from, and why the old one was wrong.** Pooling all 16 × 746 per-cell
fractions and sorting them leaves exactly **one** wide multiplicative gap, `5.654e-05 → 4.083e-03`,
a factor of 72; every other step in the whole sample is at most 1.5. `lit_fraction_min` sits at the
geometric centre of that empty band — **4.805e-4**, with a factor of 8.5 of margin on each side —
and the classification is *identical* for any floor between 1e-4 and 1e-3, a decade-wide plateau.

The floor it replaced was **0.10**, chosen as "the midpoint" between 42–53 % and 0.08 % on the one
run then available. That is a fraction of *scattered intensity*, so it moves with the beam: the lit
cells of a bright run sit at 0.45 and those of a dim one at 0.09, and a threshold at 0.10 cuts
through the middle of the population. It produced **67 physically impossible sets** such as
`[5, 7, 9, 10]` and `[1, 12]` — which is exactly what `check_structure` now catches.

`lit_gap_ratio` (default 30) then subdivides the cells that clear the floor, so a uniformly
attenuated run — every lit cell pushed towards the floor together — still splits on the step
between lit and dark rather than on the absolute level. On the 746 results it never fires; the
floor alone reproduces all four patterns. Both knobs *do* change which cells are integrated and so
which rows exist, so unlike `expected_lit_cells` they stay **in** `config_hash`.

> **Provenance gap.** The 746-result sweep is recorded here and in
> `analysis.waxs.cells`' module docstring, but its JSON is **not in the repo**, which §7's
> "keep benchmark JSONs next to the script" rule asks for. Re-run it, or find and commit it,
> before any of these numbers is quoted as measured elsewhere.

**D5 — No seam mask; `static_bad` is the `.edf` file alone.** `agipd_asic_seams()` has no JUNGFRAU
counterpart and needs none: bit 22 `NON_STANDARD_SIZE` is set in `data.mask` (§2), which is exactly
what it exists to supply for AGIPD. The `.edf` is a native pyFAI mask and shares pyFAI's polarity,
so it is loaded and used as-is — `fabio.open(path).data != 0` — with its sha256 in `config_hash`
the way `pixel_mask_file` is for AGIPD.

~~Combine it with the per-frame `data.mask` and pass the union to `integrate1d(mask=)` per frame.~~

**D5′ — Amended 2026-09-13, measured. The per-frame mask is a NaN, not a `mask=` argument.**
pyFAI keys its cached sparse matrix on a checksum of the mask, so a mask that changes per frame
rebuilds the full-split CSC matrix *every frame* — its own docstring calls that "a very time
consuming operation" (`pyFAI/integrator/common.py`, `setup_sparse_integrator`). Measured on r0423:
**16.8 ms/frame**.

Build the engine **once** with the `.edf` static mask, and carry the per-frame `data.mask` as
**NaN in the data and the variance arrays**. pyFAI's preprocessing drops a non-finite pixel from
the numerator *and* the normalisation, which is exactly the exact-denominator property AGIPD has to
reconstruct sparsely. Measured: **bit-identical** sums — max relative difference `0.0e+00` on
`sum_signal`, `sum_normalization` and `sum_variance`, over all 500 bins, on **both** detectors — at
**2.3–2.9 ms/frame** with a single cached engine.

The option exists here and not for AGIPD because JUNGFRAU frames are `float32`. AGIPD's `int16`
cannot carry NaN, which is precisely what forced the sparse denominator correction there. The
consequence is that the WAXS pass needs **no gather kernel at all**: `analysis.saxs.sparse` stays
where it is, and `waxs/masks.py` has no per-cell base mask to build.

Because the whole design rests on that equivalence, `waxs/selftest.py` gates it on real frames —
the NaN path against a reference that *does* pass the union to `integrate1d(mask=)` — so a change
in how pyFAI preprocesses non-finite values cannot pass silently. That replaces the AGIPD §6.6
gate, which would be dense-vs-dense here and prove nothing (§4).

A NaN already present in `data.adc` would be indistinguishable from that sentinel, which is why
D6's check rejects a non-finite value on any pixel the static mask keeps.

**D6 — A value-range data check. NB: its stated motivation was wrong.**

*Corrected 2026-09-13.* On the cluster, with each detector's own `data.mask`, **every** extreme
pixel is flagged: 130 on jf1 and 204 on jf2 above 1000 keV, none unflagged, none reaching the
integrator. The claim below — that jf2 carries pixels `data.mask` misses — came from comparing
jf2's data against **jf1's mask**: `data/proc_mask_jf2_r423.nc` is a byte-identical copy of the jf1
export (`module=1`), the same mix-up as §6 O3, which was fixed for the data file and not for the
mask. Re-export it before trusting anything else measured from it.

The check stays, on the weaker and still-sufficient grounds that it is nearly free and that the
`.edf` is what currently keeps those pixels out of the integrator — an arrangement worth verifying
per frame rather than assuming. What follows is kept for the record, struck through:

~~jf2 carries 15 pixels
reaching ±1.8e5 keV — some 20 000 photons where the lit-cell mean is 5.3 — in **every** memory
cell, and **none of them is flagged in `data.mask`**. jf1 has 10 such pixels and all 10 are
flagged, so this is not a property of the detector type but of the individual module's
calibration. One such pixel dominates its q bin outright.

The `.edf` static mask happens to cover all 25 (§6 O2), so under convention A they never reach the
integrator — but relying on that silently is exactly the failure mode the AGIPD status ledger
exists to prevent.~~

v1 checks the value range per frame and routes a frame carrying a pixel outside a configured bound
to `DATA_CHECK_FAILED` with the count in the ledger, rather than trusting either mask to have
caught it. Note the AGIPD check itself — integer dtype, no negative counts — is wrong here in both
halves (§2), so this replaces it rather than adding to it.

---

## 4. The shared layer, and what is JUNGFRAU's own

*Rewritten 2026-09-13: the detector-agnostic parts were moved out of
`analysis.saxs` into `analysis.common` rather than imported across from the SAXS package. The
`analysis.saxs.*` modules keep re-export shims, so the whole AGIPD test suite passes unedited and
`AgipdSaxsConfig.config_hash()` is byte-identical — no existing `agipd_saxs.h5` is invalidated.*

| `analysis.common` module | Contents | Notes |
|---|---|---|
| `status.py` | `FrameStatus`, `DataCheckFailed` | As-is. Codes are detector-independent; only the exception's docstring generalised off `image.data` |
| `cpu.py` | `physical_cores`, `file_sha256`, `THREAD_ENV`, `set_thread_env`, `default_pool`, `phase`, `package_versions` | Was split across AGIPD's `config`, `worker` and `run` |
| `plan.py` | `TrainRecord`, `Block`, `RunPlan`, `build_blocks`, `evenly_spaced` | The row model. Each pass keeps its own `build_plan` and `run_checks` |
| `masks.py` | `MaskSource`, `StaticMask`, `UnexpectedMaskBits`, `bits_to_mask`, `describe_bits`, `frame_bad` | The mask vocabulary and the per-frame union |
| `writer.py` | `FrameTableWriter`, `ConfigHashMismatch`, `IncompleteRun`, `pooled_per_train`, `as_handle`, `q_centers` | The layout, ledger, resume and label checks. The per-frame column schema is class attributes, so each pass declares its own. The `PassConfig` protocol requires `operational_fields`, and provenance records **that** set rather than the shared constant — the two passes exclude different ones |

| `analysis.waxs` module | Verdict |
|---|---|
| `config.py` | **New.** `JungfrauWaxsConfig`, one per detector; `config_for(proposal, run, detector)` fills the per-detector PONI and `.edf` paths. Also `WAXS_OPERATIONAL_FIELDS` — the shared six plus `expected_lit_cells` (D4′) — and `is_storage_cell_sequence` |
| `operator.py` | **New, and much smaller than AGIPD's.** No LUT is lifted out: under D5′ pyFAI keeps the matrix and the pass keeps only the q axis, the solid angle, the static mask and the PONI text, hashed together |
| `masks.py` | **New, simpler.** The `.edf` alone. No seams (D5), no per-cell base mask (D5′ removes the reason for one) |
| `cells.py` | **New.** Lit-cell selection (`split_lit_dark`, the floor-plus-gap split of D4′), the storage-cell-sequence check, and the readout-noise measurement, all off the same sampled trains |
| `integrate.py` | **New.** `ErrorModel` and the pure per-frame dense integration. The AGIPD `sparse.py` is not used and not moved |
| `plan.py` | **New, same shape.** The one thing that does not transfer is the row model: see §5 R6 |
| `worker.py`, `writer.py`, `run.py`, `damnit.py` | **New**, following the AGIPD shapes |
| `selftest.py` | **New, and it is not the AGIPD gate.** Dense-vs-dense would be an identity check; this one gates D5′'s NaN equivalence on real frames |

## 5. Rules that differ from the AGIPD integrator (§3 there)

Rules 1, 2, 3, 6, 7, 8 and 9 apply **unchanged**. Two are replaced:

**R4 — Geometry comes from a PONI file.** AGIPD forbids PONI files because an EXtra-geom object is
the single source of positions. For JUNGFRAU there is no `.geom` in `usr/geometry`; the source of
truth is `jf1.poni` / `jf2.poni`, which carry the detector, the pixel size, the distance, the PONI
offsets and the orientation together. Load with `AzimuthalIntegrator.sload(poni)` and **assert**
the wavelength in the file matches `cfg.photon_energy_kev` rather than overriding it silently —
the PONI carries its own wavelength, and a mismatch means the file was refined at a different
energy. Record the PONI's sha256 in `config_hash` the way the geometry file is for AGIPD.

**R5 — Identity.** `JUNGFRAU` is a `MultimodDetectorBase`, **not** an `XtdfDetectorBase`, so its
keydata is `MultimodKeyData`, which has `train_id_coordinates()` and **no `pulse_id_coordinates()`
or `cell_id_coordinates()`**. Cell identity comes from `data.memoryCell`; note that
`JUNGFRAU.cell_ids()` reads it from the **first train only** and asserts consistency across
modules, so it is a per-run constant, not per-frame coordinates — read `data.memoryCell` per train
if that assumption is not verified for this beamtime. Pulse identity does not exist in the reader
at all and must be aligned from `XrayPulses`/LITFRM onto the cell axis. **Until that alignment is
established, store the cell id and the train id and leave pulse id absent — do not synthesise one
from position** (CLAUDE.md pitfall 4). The rectangular layout makes position *within* a train
meaningful (it is the cell index), but the train axis is still addressed by label.

**R6 — `JUNGFRAU.frame_counts` counts entries, not frames.** `JUNGFRAU` is a
`MultimodDetectorBase`, whose `frame_counts` is the INDEX *entry* count — one per train — while
each entry holds `_frames_per_entry` memory cells (16 here). `AGIPD1M.frame_counts` counts frames
directly. So a train's rows are the *lit cells of its single entry*, not its entry count, and
assuming otherwise is a silent off-by-sixteen. EXtra-data's reader assumes one entry per train
throughout (`buffer_shape` is `(modules, trains) + entry_shape`), so a train with two entries
cannot be represented at all and `plan.build_plan` refuses the run rather than reading it wrong.

**R7 — One module per detector.** Each of this experiment's JUNGFRAU-500Ks is a single module with
its own PONI, its own `.edf` and its own q range, so `plan.open_detector` refuses a multi-module
selection outright. A consequence for the ledger: `MISSING_MODULES` cannot arise — a train has its
one module or none, and none is `NO_FRAMES`.

**R8 — No `decompress_threads`.** That argument is `XtdfImageMultimodKeyData`'s, i.e. AGIPD's;
`MultimodKeyData.ndarray` does not take it, so CLAUDE.md pitfall 3 does not arise on this read
path. The parent still pins `EXTRA_NUM_THREADS=1` before the pool exists.

---

## 6. Open questions

- **O1 — Resolved 2026-09-13 from `lsxfel` on `/gpfs/exfel/d/proc/MID/202601/p010400/r0423`.**

  | | jf1 | jf2 |
  |---|---|---|
  | Detector name | `MID_EXP_JF500K1` | `MID_EXP_JF500K2` |
  | Corrected source | `MID_EXP_JF500K1/CORR/JNGFR01:daqOutput` | `MID_EXP_JF500K2/CORR/JNGFR02:daqOutput` |
  | Legacy source | `…/DET/JNGFR01:daqOutput` → the CORR one | `…/DET/JNGFR02:daqOutput` → the CORR one |
  | `first_modno` | 1 | 2 |
  | Files | `CORR-R0423-JNGFR01-S{seq:05d}.h5`, 6 × 500 trains | `CORR-R0423-JNGFR02-S{seq:05d}.h5`, 6 × 500 trains |

  Keys are `data.adc`, `data.mask` and `data.memoryCell`, as the component's defaults expect.

  Three consequences, each pinned by a test:

  1. **Corrected data uses `/CORR/`.** Since 2026/1 (`_data_is_raw`: *"corrected data always uses
     /CORR/ in its source names"*), and `_source_corr_pat` matches only that. A mock written with
     the old `/DET/` names silently takes the raw-name fallback path instead — which is what this
     one did until the names were known.
  2. **The legacy `/DET/` alias does not become a second module**, though it easily could look as
     if it should. It is an `h5py.SoftLink` listed in `METADATA/dataSourceId`, and
     `DataCollection.instrument_sources` **includes** legacy names — only `detector_sources`
     subtracts them — so `_source_matches` really does iterate over it. What keeps it out is
     `_source_corr_pat` matching `/CORR/` alone. If that ever widened, two sources would collapse
     onto one module number and `MultimodDetectorBase.__init__`'s own assertion would fire.
  3. **Auto-detection cannot work.** Both names match `_det_name_pat`, so
     `_find_detector_name` raises *"Multiple detectors found"* against a whole run. `config.py`
     therefore fills `detector_name` and `first_modno` from the detector rather than leaving them
     to it, and the filled name enters `config_hash` — which Karabo source a run was integrated
     from is part of what identifies the result.

  A related trap the config now refuses outright: `dataclasses.replace(cfg, detector="jf2")` keeps
  jf1's `detector_name`, PONI and mask, and integrating one detector's frames through the other's
  geometry yields a plausible-looking I(q) and no error anywhere — the AGIPD beam-centre trap
  (CLAUDE.md pitfall 15) in another guise. Use `config_for(proposal, run, detector)`.
- **O2 — Resolved 2026-09-13: non-zero = excluded.** `jf1_mask.edf` and `jf2_mask.edf` are native
  **pyFAI** masks written from silx view, so they carry pyFAI's own polarity, which the installed
  source states twice — `integrate1d_ng`: *"array with 0 for valid pixels, all other are masked"*;
  `detectors/_common.py`: *"the mask with valid pixel to 0"* — and a polarity test confirms
  (masking half the module non-zero drops `sum_normalization` to 0.494 of unmasked). Pass them
  straight to `mask=`; no inversion anywhere, and the same convention as the AGIPD pixel mask.

  Three measurements had independently pointed the same way, and they are worth keeping because
  they are what would catch a *future* mask written the other way round: restricted to the
  `.edf`-zero region the two detectors agree on readout noise (0.323 vs 0.317 keV) and photon peak
  (8.87 vs 9.12 keV), where over the whole detector jf2's dark-cell σ is **469 keV**, 1500× jf1's;
  all 25 unmasked extreme pixels of D6 fall in the non-zero region; and non-zero pixels are
  2.0–2.7× enriched in dynamically-flagged ones.

  *What did **not** discriminate:* integrating both ways through the PONI gives a plausible smooth
  curve either way — the scattering is diffuse enough that a wrong mask produces no obvious
  nonsense. Do not reach for that test on the next mask question.
- **O3 — Resolved 2026-09-13.** The `jf2` export was a copy of `jf1`; re-exported and verified
  distinct (`module=2`, different sha256). §2 now carries both detectors.
- **O4 — Resolved twice, and the second answer reversed the first. Closed 2026-09-14: the set is
  *not* run-invariant; its shape is.**

  *First pass, 2026-09-13, r0423 + r0426.* `scripts/w1_facts.py`, both detectors, eight sampled
  trains each: the lit set is identical in all four, `{0,1,2,3,4,5,6,15}`. Cells 0–6 and 15 hold
  33–40 % of kept pixels above half a photon; cells 7–14 sit at 1e-6 to 4e-5 — five orders of
  magnitude of margin. ~~A single global `expected_lit_cells` is therefore right; no per-run
  expression is needed.~~ **That conclusion was drawn from two runs out of 373 and is wrong.**

  **Cell 7 is dark and cell 15 is lit**, which is not what §2 recorded. The earlier `(0…7)` came
  from indexing an exported train by array *position*: positions 0–7 are the lit ones, and the
  `data.memoryCell` values they carry are 0–6 and 15. That is exactly the inference CLAUDE.md
  pitfall 4 forbids — and D4's loud failure is what caught it, on the first cluster run, before a
  single frame was integrated. The design worked; the constant was wrong.

  *Second pass, 2026-09-14, r0001–r0500.* Over 746 (run, detector) results the proposal used
  **four** readout patterns, tabulated in D4′: `{0…6,15}` for r0379–r0500, all 16 for r0054–r0378,
  `{15}` for three runs, and nothing at all for 43. `{0,1,2,3,4,5,6,15}` is the *science block's*
  set, not the proposal's. What **is** invariant is that every pattern is a storage-cell sequence
  mod 16, and that is what the pass now gates on (D4′); the named set survives as
  `EXPECTED_LIT_CELLS`, an expectation a reprocess may pin, never the default.

  *The lesson is about the shape of the evidence, not about JUNGFRAUs.* Two runs agreeing looked
  like invariance and was in fact one plateau of a four-level step. The same pattern of reasoning
  is live in this file's §6 O5 (a χ² measured on one featureless run) and in CLAUDE.md open task 4
  (a beam centre agreed on one train), so it is worth naming.

  Cell 15 runs consistently a little below 0–6 (0.333 against 0.352 on jf2 r0423, 0.363 against
  0.385 on jf1 r0423). That is the usual JUNGFRAU first-storage-cell behaviour and a reason to look
  at cell 15 separately before pooling it with the others.
- **O5 — Resolved 2026-09-13: fit a scale factor over the overlap.** After masking the two populate
  11.5–23.7 (jf1) and 9.8–18.5 (jf2), so they overlap over ~11.5–18.5 nm⁻¹. `analysis.waxs.combine`
  fits a single multiplicative factor for jf2 against jf1 over that range and merges them into one
  curve; `damnit.combined_curve` is the DAMNIT-facing form, and the overview draws jf2 on jf1's
  scale with the factor in the legend.

  The original worry — that pooling needs both `N` arrays on the same absolute scale, which is
  untested across different solid-angle coverage and different masks — is answered by construction:
  a *fitted* factor absorbs whatever the relative normalisation is. What it cannot absorb is a
  difference in **shape**, which is what `reduced_chi2` and `residual_rms` report.

  **Where that cross-check is blind, measured.** A relative error between the two q axes is
  degenerate with a scale factor whenever the overlap is featureless: for a power law
  `I(1.05·q) = 1.05⁻²·I(q)` exactly. On a smooth `100/q² + 0.5`, a 5 % q shift leaves χ²ᵣ at 0.22 —
  invisible. Put a Bragg-like peak in the overlap and the same shift gives χ²ᵣ 776, and 1 % gives
  77. So this is a real cross-check on the two PONIs **only on a run whose overlap has a feature**:
  it bites on r0426, not on r0423. A good χ² on r0423 says the two detectors agree in shape, not
  that either q axis is right.

  Implementation notes: only jf2 is interpolated (jf1 keeps its own bins, so the result is on a real
  q axis); the fit is weighted least squares through the origin, with a `median`-of-ratios
  alternative for outlier-heavy data; `combine_files` works off the stored sums, which is the path
  that gives a meaningful χ² because the per-cell grid carries no errors.

  **First real run, r0423, 2026-09-13** — both detectors with their own PONIs and masks:

      factor 0.98823 ± 0.00005   over 11.53–18.44 nm⁻¹, 285 bins
      reduced χ² 54.1            residual rms 0.74 %

  **The factor is the headline: 0.988, within 1.2 % of unity.** Two independent PONIs, two
  independent masks, different solid-angle coverage and different q ranges, and the two detectors
  agree on *absolute* I(q) to a bit over one percent with nothing tuned. That is a real
  cross-validation of the whole normalisation chain — the Σc·Ω denominator, the masks, the
  geometry — and it was not guaranteed. O5's original worry, that the two `N` arrays might not be
  on the same absolute scale, is answered: they nearly are.

  **The χ² of 54 means the error bars are small, not that the curves disagree.** `residual_rms /
  sqrt(χ²ᵣ)` puts the claimed per-bin error at **0.10 %** against an actual shape difference of
  **0.74 %** — so a real, systematic, 7σ disagreement that is nonetheless sub-percent. Read χ²
  alone and it sounds catastrophic; it is not. The 0.10 % is itself a check on D3: pooling ~2.3e6
  photons per bin gives a Poisson floor of 0.066 %, so the error model is producing errors of the
  right size, which is the first independent evidence that it is calibrated and not just
  self-consistent. (`OverlapScaling.sigma_over_intensity` is this number.)

  **What the 0.74 % is, is not yet settled, and r0423 cannot settle it.** Two candidates:

  1. **The polarisation correction, applied to neither detector.** pyFAI's azimuthal modulation
     amplitude is `sin²(2θ)/2`, which across this overlap runs from **3.1 % at q = 11.5 to 7.8 % at
     q = 18.4**. The two detectors sit at different azimuths, so each averages a different part of
     that modulation, and the difference grows with q. Note what this does to CLAUDE.md open task 5:
     the "≤ 5.4e-4 at q_max" recorded there is the **AGIPD** number and does not transfer — at WAXS
     angles polarisation is two orders of magnitude larger and is the leading systematic here.
  2. **A small relative error between the two q axes.** On a featureless curve this is largely
     degenerate with the scale factor, but only largely: the measured 5 % q-shift test left a
     0.69 % rms residual, which is the same size as what is seen here.

  `OverlapScaling.residual_slope_per_nm` exists to separate them from a leftover normalisation
  offset — flat in q means the factor already absorbed it, a tilt means something angle-dependent
  that no factor can. It does **not** separate candidate 1 from candidate 2, both of which tilt.
  What does: **r0426**, where Bragg peaks make a q-axis error shift peak positions while
  polarisation does not (§6 O5's featureless blind spot, in the one case where it lifts); and
  applying the polarisation correction to see whether χ² falls.
- **O6 — Closed 2026-09-13 by decision, not by measurement.** AGIPD covers 0.077–1.067 nm⁻¹ and
  JUNGFRAU populates 9.8–23.7, so the gap is 1.07–9.8 nm⁻¹ with no overlap to cross-normalise
  against. No combined SAXS+WAXS curve is planned at any point, so the question does not arise and
  the warning it prompted has been removed from the overview figure. If that ever changes, the
  constraint is unchanged: two disconnected pieces with a free relative scale between them, and the
  predecessor's fcc peaks at 0.6/0.7/1.0 nm⁻¹ sit in the AGIPD range, not here.

---

## 7. Phases and acceptance

**W0 — shared layer. Done 2026-09-13.** The detector-agnostic parts of `analysis.saxs` moved to
`analysis.common` (§4), with re-export shims left behind. Gate: the whole existing test suite —
213 passed, 1 skipped — passes with **no test file edited**, and
`AgipdSaxsConfig(proposal=10400, run=423).config_hash()` is byte-identical
(`5f589d2a8090d075…`), so no stored `agipd_saxs.h5` is invalidated.

**W1 — data and geometry facts. Done 2026-09-13 on max-exfl484**, over r0423 and r0426, both
detectors, eight sampled trains each (`scripts/w1_facts.py`, JSON beside itself):

    python scripts/w1_facts.py --runs 423 426 --detectors jf1 jf2

| Question | Answer |
|---|---|
| **O1**, sources | `MID_EXP_JF500K{1,2}/CORR/JNGFR0{1,2}:daqOutput`, one module each, 16 cells/entry, 1 entry/train, **3000 trains** per run. The legacy `/DET/` alias is present and correctly ignored. Wired into `config.DETECTOR_NAMES` / `DETECTOR_MODNOS`; the mock reproduces the `/CORR/` layout including the soft link |
| Files present? | Yes. `usr/geometry/jf{1,2}.poni` (430 / 432 B) and `usr/masks/jf{1,2}_mask.edf` (524 800 B), sha256 recorded |
| q populated | jf1 **11.53–23.68**, jf2 **9.80–18.46** nm⁻¹ — the §2 figures, now from the real PONIs. Overlap 11.53–18.46 |
| Static mask | 75.92 % / 84.39 % — as recorded |
| Bits | `{0, 1, 21, 22}` on all four, none unexpected |
| **O4**, lit cells | `{0,1,2,3,4,5,6,15}`, identical across both runs and both detectors — **Not** `{0…7}`. Read this as *these two runs agree*, not as invariance: the r0001–r0500 sweep later found four patterns (§6 O4, §3 D4′) |
| Readout noise | jf1 0.359 (r0423) / 0.337 (r0426); jf2 0.318 / 0.318. jf1's moves 6 % between runs, which is why D3 measures it per run instead of hardcoding |
| **Is `data.mask` train-invariant?** | **Yes** over the sampled trains: zero differing pixels, all four combinations |
| **Does `roi` save I/O?** | **No, and it cannot** |
| Extreme pixels | 130 (jf1) / 204 (jf2) above 1000 keV, **all flagged** by `data.mask`, none reaching the integrator — which corrects §3 D6 |

**`roi` is a dead end, for two independent reasons.** `data.adc` and `data.mask` are both chunked
`(1, 16, 512, 1024)` — one chunk per train spanning *all sixteen cells* — and `data.mask` is
gzip-compressed, so a partial read still costs a whole chunk fetch and decompress. And the lit set
`{0…6, 15}` is not contiguous, so there is no single window to ask for anyway. The worker's
read-everything-and-select-in-memory is not a compromise; it is the only sensible shape. (`data.adc`
is uncompressed, `data.mask` is not.)

**`data.mask` being train-invariant is a real optimisation, not yet taken.** Zero pixels differ
across the sampled trains on either detector in either run. The mask read is half the pass's I/O,
so reading it once per run would nearly halve the wall time. But eight trains out of 3000 is not
proof, and being wrong means silently integrating against a stale mask — so this needs a whole-run
check first, and the gain should be measured against W4's timing rather than assumed.

**Gate: met.** The §2 table is confirmed run-to-run, amended where it was wrong (the lit set, D6),
and both I/O questions have numeric answers.

**W2 — operator and error model. Done 2026-09-13**, on the real r0423 train of **both** detectors.

| Gate | Verdict |
|---|---|
| PONI load, shape and wavelength assertion | pass; fires on a PONI refined at 9.000 keV |
| Method resolution `("full","csc","cython")` | pass, asserted on the probe |
| σ_read measured from the dark cells | 0.3230 (jf1) / 0.3175 (jf2) keV — the §2 numbers |
| Lit cells measured from the data | array positions 0–7 on both; 42–53 % against ≤ 0.08 %. The *cell ids* are `{0…6,15}` (§6 O4) — the synthetic geometry here carries no `data.memoryCell` |
| Poisson recovery on a synthetic frame | model mean variance within 0.04 % of the true one |
| **NaN equivalence (D5′)** | **max rel S, N, V all exactly `0.0e+00`**, 8 frames per detector, all 500 bins, one cached engine |
| Per-frame cost | 2.3–2.9 ms against 16.8 ms for the per-frame-mask path |

The q values in that run are **not** real — the PONI files are on GPFS, so a synthetic geometry was
used. Nothing gated above depends on where the beam centre is.

**W3 — masks, plan, worker, writer. Done 2026-09-13**, on a mock JUNGFRAU run built with
`extra_data.tests.mockdata.jungfrau.JUNGFRAUModule` (which already writes the right dtypes, so
unlike the AGIPD mock the images need no recasting). Covered: dropped train, zero-entry train,
multi-entry refusal, block never straddling a gap, repeated `memoryCell` → `LABEL_MISMATCH`,
`DATA_CHECK_FAILED` on an extreme pixel, worker exception → `WORKER_ERROR`, broken pool →
`NOT_PROCESSED` and raise, resume completing only missing blocks, config-hash mismatch refused, a
real spawned pool, the D4 loud failure end to end, and a run with no dark cell. Invariants: rows
written by label, no NaN anywhere in the file.

*Extended 2026-09-14 for D4′ and the D3 fallback:* the floor-and-gap split as a pure function, the
storage-cell-sequence table (including two of the 67 impossible sets the 0.10 floor produced), an
impossible set raising `ImplausibleLitCells`, a pinned set that the data contradicts stopping the
run while the same run unpinned integrates cleanly, a run with **no** lit cell returning `None`
with its evidence in the log, an empty lit set planning zero rows rather than being refused, and
σ_read taking each of its three sources in turn — and that the fallback **cannot** be cleared,
which is the assertion that caught the dead `ReadNoiseUnavailable` branch (D3).

*First cluster run, 2026-09-13:* both scripts failed immediately on max-exfl484 with
`ValueError: buffer source array is read-only`, in `build_operator`, for every run and detector.
The cause is CLAUDE.md pitfall 18 — the static mask was frozen read-only and handed straight to
`integrate1d(mask=)`, which the Linux pyFAI wheels reject and the macOS ones accept, so the whole
local suite passed. Fixed by copying before freezing and keeping a writable `static_mask_2d` on the
operator; `tests/waxs/test_waxs_operator.py` now pins both. The scripts recorded only `repr(error)`
and no traceback, which is what made this cost a round trip — they capture and print the traceback
now.

**W4 — on-node acceptance, r0423: ACCEPTED on BOTH detectors 2026-09-13** (max-exfl484, 36
workers, full runs).

> **Both accepted files are now stale, 2026-09-14.** D4′ and D3's fallback added
> `lit_gap_ratio` and `read_noise_fallback_kev` to the config and moved `lit_fraction_min`'s
> default, and all three are result-affecting and so in `config_hash`. Every
> `jungfrau_waxs_*.h5` written before that date carries the old hash and will be **refused, not
> resumed** — the same situation as CLAUDE.md pitfall 12's 2026-09-13 note. Rerun W4 on both
> detectors before W5. The numbers below are still the right *targets*; they are no longer a
> current verdict.

| Gate | jf1 | jf2 |
|---|---|---|
| 0 configuration | pass | pass — 3001 trains, 24 000 frames, 36 workers, config hash matches, lit cells `{0…6,15}` |
| A self-test | **exactly 0.0** | **exactly 0.0** — max rel S, N and V against the per-frame-mask reference |
| B reference | **6.5e-08** | **8.0e-08** — 3 trains each, all 500 bins populated, empty-bin sets equal |
| C timing | **17.5 s** | **20.1 s** — against a 600 s target; efficiency 0.748 / 0.655 |
| D ledger | pass | pass — 24 000/24 000 OK, bits exactly `{0,1,21,22}`, 3000 of 3001 trains own rows |

The one train owning no rows is **2637698397**, the last of the run, on both detectors — named by
the ledger's train-level reporting, which was added because jf1's frame ledger reconciled perfectly
without ever mentioning it.

**D5′ holds at full scale.** The NaN path and the per-frame-mask reference agree to *exactly* zero
on real frames — the same result as the one-train W2 gate, now over a whole run integrated by 36
spawned workers.

**Per frame per core: read_data 11.6 / 11.9, read_mask 5.0 / 5.2, integrate 3.0 / 2.7 ms, total
19.7 / 19.8** (jf1 / jf2). The two detectors agree to within 4 % on every stage, which is itself
worth having: the cost is set by the read, not by anything detector-specific. Two things worth
keeping:

- **`integrate` did not degrade under load.** 3.0 ms against 2.3–2.9 ms measured on one idle core,
  where the AGIPD pass found every stage 1.3–1.6× worse with 36 workers competing for memory
  bandwidth. Dense pyFAI over 24 000 frames is not bandwidth-bound the way a sparse kernel over
  465 000 frames is. The AGIPD warning does not transfer, and now there is a number for that.
- **The read dominates, and it is `data.adc`, not `data.mask`.** 11.6 against 5.0 ms, because
  `data.adc` is stored uncompressed (33.5 MB/train) while `data.mask` is gzip-compressed and mostly
  zeros. This **corrects the W1 estimate**: reading the mask once per run would save 5.0/19.7 = 25 %
  of worker time, about 26 % of the wall — not the "nearly half" guessed from the two datasets
  having the same logical size. At 17.5 s for a whole run that is 4 s, against the risk of silently
  integrating a stale mask. **Not worth taking.** The train-invariance measurement stands as a
  recorded fact, not as a pending optimisation.

**The unclamped variance, and why jf2 is fifteen times worse.** Frames carrying at least one
non-positive variance bin: **493 of 24 000 (2.1 %) on jf1, 9162 (38.2 %) on jf2**, at most 2 and 6
bins respectively in any one frame. The difference is not the negative-pixel fraction — jf1 has
*more* of those (45 % against 33 %) — it is pixels per bin: jf2's `.edf` masks 84.4 % against
75.9 %, leaving 81 767 kept pixels against 126 125, so roughly 164 per bin against 252. Thinner
bins scatter further, and a thin bin at the edge of the q range scatters furthest.

This is D3 working, not failing, but it does mean **per-frame σ is unusable on a third of jf2's
frames** and anyone wanting per-frame errors must select on the count. The pooled σ is unaffected,
which is the claim the design rests on — so the W4 ledger now pools the run the way §9 pools it and
reports how many bins are *still* non-positive afterwards, rather than leaving "pooling fixes it"
as an assertion. `tests/waxs/test_waxs_writer.py` forces per-frame negatives on a written file and
checks both that they cancel and that `pooled_per_train` says so when they do not — the cancellation
is a property of the reducer, so it is tested against the reducer rather than against the script
that happens to report it.

*Remaining:* nothing on W4. Both detectors are accepted.

**Script.**
`scripts/w4_acceptance.py`, one detector at a time, verdict as JSON beside itself:

    python scripts/w4_acceptance.py --run 423 --detector jf1 --workers 36
    python scripts/w4_acceptance.py --run 423 --detector jf2 --workers 36

| Gate | Asks |
|---|---|
| 0 configuration | the whole run, on the workers claimed, at this config hash, with the configured lit set? |
| A self-test | did the NaN path equal the per-frame-mask reference on real frames? (read back from provenance) |
| B reference | does an **independent** integration reproduce what was stored? |
| C timing | wall time, ms/frame/core per stage, parallel efficiency, serial fraction |
| D ledger | every frame accounted for, expected cells and bits, negative-variance bins counted |

Gate B is the one that earns its keep: it re-integrates a few fully-OK trains the slow way —
`integrate1d(mask = static | dynamic)` per frame — and compares against what the *writer stored*,
so the row-to-train map, the `f4` storage and the whole chain are inside the comparison rather than
just the kernel. Its ability to fail is tested against a deliberately corrupted stored row.

Only the wall time is a verdict; a stage over `BUDGET_MS` is reported, not failed. **Budget under
load, not one core at a time** — the AGIPD P4 found every single-core figure 1.3–1.6× optimistic,
and the 2.3–2.9 ms/frame above was measured on one idle core of a laptop, which is why `BUDGET_MS`
here is set well above it.

**W5 — DAMNIT. Implemented 2026-09-13; not yet run on the cluster.** Four variables in
`src/amore/context.py`, all thin wrappers over `analysis.waxs.damnit`: `jungfrau_waxs_jf1`,
`jungfrau_waxs_jf2` (each returning the `(trainId, cellId, q)` grid — 48 MB at 3000 trains × 8 cells
× npt 500, against 0.93 GB for the AGIPD per-pulse grid), `jungfrau_waxs_overview`, which draws
the two as **two traces with their overlap shaded** (§6 O5) and states the AGIPD gap on the figure
(§6 O6), and `jungfrau_waxs_combined`. `tests/test_context.py` checks every integration wrapper's
body is an import and a call, by AST rather than by line count.

**A bulk reprocess survives a dark run.** D4′'s `None` return runs through the whole surface:
`jungfrau_waxs` returns `None` for a run with no lit cell, `combined_curve` returns `None` when
either detector produced nothing, and `overview_figure` already drew with one or both missing. 43
of the proposal's runs are in that state, so this is the difference between a reprocess finishing
and a reprocess stopping on r0001.

*Remaining:* run them on r0423 and r0426. Needs the cluster and the real PONI files — **and a W4
rerun first**, because the config hash moved (see W4's note). Pre-flight `scripts/w1_facts.py`
over the full run list: with D4′ it no longer fails on a run whose pattern merely differs, but
`lit_matches_expected` still reports against the science-block set, which is what tells you which
block a run belongs to before you integrate it.

---

## 8. Testing conventions

**The suite covers `src/` only.** `scripts/` is not unit-tested: `tests/saxs/test_p4_acceptance.py`,
`tests/waxs/test_waxs_w1_facts.py` and `tests/waxs/test_waxs_w4_acceptance.py` were removed
2026-09-14. 406 tests collected.

The consequence is worth being explicit about, because it is a real trade. The acceptance scripts
are the things that decide whether a run is accepted, and nothing now re-checks their gate logic
between edits — a change to `stage_ledger` or `_lit_as_configured` is caught by running the script
on a node, not by pytest. When editing one, run it on r0423 before trusting it.

What was *not* lost: the one test in that set whose subject was `src` — D3's claim that per-frame
negative variances cancel on pooling — moved to `tests/waxs/test_waxs_writer.py`, where it tests
`pooled_per_train` directly instead of the script that happens to report it. Nothing else in the
three files asserted anything about `analysis.*`.

**The mock run costs 135 MB.** `data/adc` is dense `float32` over 16 cells of 512×1024, because the
real detector is. `tests/waxs/conftest.py`'s `mock_run_factory` caches one run per **resolved**
keyword signature — bound against `write_mock_run`'s own defaults, so passing a default explicitly
does not silently write a second identical copy — and holds it for the whole session. So the
suite's peak is 135 MB × the number of distinct signatures. Three things keep that bounded:

- `data/mask` is recreated gzip+shuffle before it is filled, `(1, 16, 512, 1024)` chunks as the
  real CORR files use. Its content is a few hundred set bits in a field of zeros, so it goes from
  134 MB to 0.74 MB and both reading and writing get *faster*. `adc` is left alone: dense noise,
  15 % for 2.3 s of deflate per run, paid back on every read.
- `fill=False` creates the datasets unwritten — kilobytes, not megabytes — for tests that read only
  metadata (source names, INDEX, train ids). Anything reading a frame gets the fill value silently,
  so it is only for those.
- `pyproject.toml` sets `tmp_path_retention_policy = "failed"` and `tmp_path_retention_count = 1`.
  pytest's defaults keep the last *three* sessions' basetemps, which is gigabytes of synthetic
  detector frames outliving the run that made them.

**Add a new mock signature deliberately.** Each one is another 135 MB for the session.
