# JUNGFRAU WAXS integrator (`jungfrau_waxs`) — design and implementation guide

Context file for `<pkg>.waxs`. CLAUDE.md open task 2.

Read `CLAUDE.md` first, then `context/agipd-saxs-integrator.md` — this file is written as a *delta*
against that one and does not repeat its reasoning. Where a section here is silent, the AGIPD
design applies unchanged; §4 says exactly which parts those are.

Work phase by phase (§7). Present a plan before creating files. Nothing here is implemented yet.

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
| **Lit cells** | **0–7** | **0–7** | confirmed on both. Lit cells 42–53 % of pixels > 4.5 keV; dark cells ≤ 0.004 %. Integrating all 16 halves I(q) (§3 D4) |
| Lit / dark mean, kept region | +5.67 / −0.16 keV | +5.32 / −0.00 keV | — |
| Readout σ, dark cells, kept region | **0.323 keV** | **0.317 keV** | the noise floor of the error model, and the two agree |
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

**D4 — Lit-cell selection is mandatory, and it comes from the data.** Cells 8–15 hold no photons
at all (0.00 % of pixels above half a photon, against 42–49 % in cells 0–7). Integrating all 16
would halve I(q) and add a dark-frame background. Unlike AGIPD's open task 3, this does not wait
on LITFRM: the split is unambiguous in the frames themselves. v1 selects lit cells by a measured
per-cell threshold, records the selected set in provenance, and **fails loudly if the lit set is
not what the config expects** — a silent change in the cell pattern between runs must not pass.

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

**D6 — A value-range data check, because `data.mask` is not sufficient.** jf2 carries 15 pixels
reaching ±1.8e5 keV — some 20 000 photons where the lit-cell mean is 5.3 — in **every** memory
cell, and **none of them is flagged in `data.mask`**. jf1 has 10 such pixels and all 10 are
flagged, so this is not a property of the detector type but of the individual module's
calibration. One such pixel dominates its q bin outright.

The `.edf` static mask happens to cover all 25 (§6 O2), so under convention A they never reach the
integrator — but relying on that silently is exactly the failure mode the AGIPD status ledger
exists to prevent. v1 therefore checks the value range per frame and routes a frame carrying a
pixel outside a configured bound to `DATA_CHECK_FAILED` with the count in the ledger, rather than
trusting either mask to have caught it. Note the AGIPD check itself — integer dtype, no negative
counts — is wrong here in both halves (§2), so this replaces it rather than adding to it.

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
| `writer.py` | `FrameTableWriter`, `ConfigHashMismatch`, `IncompleteRun`, `pooled_per_train`, `as_handle`, `q_centers` | The layout, ledger, resume and label checks. The per-frame column schema is class attributes, so each pass declares its own |

| `analysis.waxs` module | Verdict |
|---|---|
| `config.py` | **New.** `JungfrauWaxsConfig`, one per detector; `config_for(proposal, run, detector)` fills the per-detector PONI and `.edf` paths |
| `operator.py` | **New, and much smaller than AGIPD's.** No LUT is lifted out: under D5′ pyFAI keeps the matrix and the pass keeps only the q axis, the solid angle, the static mask and the PONI text, hashed together |
| `masks.py` | **New, simpler.** The `.edf` alone. No seams (D5), no per-cell base mask (D5′ removes the reason for one) |
| `cells.py` | **New.** Lit-cell selection and the readout-noise measurement, off the same sampled trains |
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
  (CLAUDE.md pitfall 14) in another guise. Use `config_for(proposal, run, detector)`.
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
- **O4 — Is the lit-cell split run-invariant?** Measured on one train of one run. If it varies,
  D4's "fail loudly" becomes the mechanism that catches it, but the config needs to express the
  expected set per run rather than globally. **`scripts/w1_facts.py` answers this**: run it over
  r0423 and r0426 and read `o4_lit_cells_by_run` and `o4_lit_cells_invariant` out of its JSON.
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

**W1 — data and geometry facts. Partly done; the rest needs Maxwell.**

- **O1 — done 2026-09-13.** Source names, module numbers, file naming and the legacy-alias
  behaviour are settled and wired into `config.DETECTOR_NAMES` / `DETECTOR_MODNOS`; the mock run now
  reproduces the real `/CORR/` layout including the soft-linked `/DET/` alias.

Remaining, all needing the cluster. **`scripts/w1_facts.py` answers every one of them in a single
pass** and writes its findings as JSON beside itself:

    python scripts/w1_facts.py --runs 423 426 --detectors jf1 jf2

- Confirm `usr/geometry/jf{1,2}.poni` and `usr/masks/jf{1,2}.edf` exist and that the `.edf` sha256s
  match the `data/jf{1,2}_mask.edf` copies used here (those carry a `_mask` suffix the cluster ones
  do not).
- **O4**, re-check the lit-cell set and the bit set on a crystallised run (r0426). D4's loud failure
  is the mechanism that catches a change, but the config has to be able to express a per-run set.
- **Is `data.mask` train-invariant?** It is uint32 and the same size as the data, so reading it
  doubles the I/O (≈ 200 GB per detector per run). If the dynamic bits never change between trains,
  one read per run replaces 3000. Measure; do not assume.
- **Does `roi` save I/O?** `MultimodKeyData.ndarray(roi=(np.s_[0:8],))` would read only the lit
  cells — note the *tuple*, because EXtra-data concatenates it onto an index expression and a bare
  slice raises `TypeError` — but it slices the *array* axis while the lit set is defined by
  `data.memoryCell` values, and whether it saves anything depends on the proc chunk layout. v1
  reads all sixteen and selects in memory (worker docstring), which is correct and twice the I/O.
  `scripts/w1_facts.py` times both.

**W2 — operator and error model. Done 2026-09-13**, on the real r0423 train of **both** detectors.

| Gate | Verdict |
|---|---|
| PONI load, shape and wavelength assertion | pass; fires on a PONI refined at 9.000 keV |
| Method resolution `("full","csc","cython")` | pass, asserted on the probe |
| σ_read measured from the dark cells | 0.3230 (jf1) / 0.3175 (jf2) keV — the §2 numbers |
| Lit cells measured from the data | (0…7) on both; 42–53 % against ≤ 0.08 % |
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
real spawned pool, the D4 loud failure end to end, and `ReadNoiseUnavailable` when a run has no
dark cell. Invariants: rows written by label, no NaN anywhere in the file.

*First cluster run, 2026-09-13:* both scripts failed immediately on max-exfl484 with
`ValueError: buffer source array is read-only`, in `build_operator`, for every run and detector.
The cause is CLAUDE.md pitfall 17 — the static mask was frozen read-only and handed straight to
`integrate1d(mask=)`, which the Linux pyFAI wheels reject and the macOS ones accept, so the whole
local suite passed. Fixed by copying before freezing and keeping a writable `static_mask_2d` on the
operator; `tests/waxs/test_waxs_operator.py` now pins both. The scripts recorded only `repr(error)`
and no traceback, which is what made this cost a round trip — they capture and print the traceback
now.

**W4 — on-node acceptance, r0423. Script written; the run needs Maxwell.**
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

**W5 — DAMNIT. Implemented 2026-09-13; not yet run on the cluster.** Three variables in
`src/amore/context.py`, all thin wrappers over `analysis.waxs.damnit`: `jungfrau_waxs_jf1`,
`jungfrau_waxs_jf2` (each returning the `(trainId, cellId, q)` grid — 48 MB at 3000 trains × 8 cells
× npt 500, against 0.93 GB for the AGIPD per-pulse grid) and `jungfrau_waxs_overview`, which draws
the two as **two traces with their overlap shaded** (§6 O5) and states the AGIPD gap on the figure
(§6 O6). `tests/test_context.py` now checks every integration wrapper's body is an import and a
call, by AST rather than by line count.

*Remaining:* run them on r0423 and r0426. Needs the cluster and the real PONI files.
