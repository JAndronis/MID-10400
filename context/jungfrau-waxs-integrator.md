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

**D3 — Error model: `Var(x) = σ_read² + E·x`, in keV².** `Σc²·x` is the Poisson variance of
*integer photon counts*; on keV-valued data with a 45 % negative tail it is not a variance at all
and can go negative. For a pixel holding `x` keV of deposited energy from `E`-keV photons the
photon count is `x/E`, whose Poisson variance is `x/E`, which in keV² is `E·x`; the readout term
adds `σ_read² ≈ 0.10 keV²`. Pass this per-pixel array to pyFAI as `variance=` — never
`ErrorModel.POISSON` (CLAUDE.md pitfall 2). **Confirm `E` and σ_read per run** rather than
hardcoding this train's numbers.

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

Combine it with the per-frame `data.mask` and pass the union to `integrate1d(mask=)` per frame;
do **not** put it on `detector.mask` instead. A detector-level mask is static, and the per-frame
bits have to join it anyway, so one union at the call site keeps a single code path and keeps the
denominator exact per frame (AGIPD §6.4's reason, unchanged).

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

## 4. What is inherited from `analysis.saxs` unchanged

Verified against the installed source, not assumed.

| Module | Verdict |
|---|---|
| `status.py` | **As-is.** Codes and the data-check exception are detector-independent |
| `selftest.py` | **As-is.** Works off `op.shape` / `op.omega` and touches zero config fields. Under D1 it becomes a dense-vs-dense identity check, so weaken or retire the gate rather than pretend it proves something |
| `writer.py` | **Type change only.** Touches exactly five config fields — `config_hash`, `method`, `npt`, `output_file`, `overwrite` — all detector-agnostic. Take a `Protocol` instead of the concrete `AgipdSaxsConfig`; the ledger, resume, label checks and both reducers of §9 carry over |
| `operator.py` | **Mostly.** The body only calls `to_pyfai_detector()` / `to_distortion_array()`. Under §5 R4 the JUNGFRAU path loads a PONI instead, so `build_operator` gains a branch; the hashing, save/load and the `("full","csc","cython")` assertion are unchanged |
| `sparse.py` | **Not used** (D1). Left to the AGIPD pass |
| `plan.py` | **New, same shape.** `frame_counts` and `frames_per_train` are set in `MultimodDetectorBase.__init__`, so they exist on `JUNGFRAU` too and the row model survives; the run checks are AGIPD components and need JUNGFRAU equivalents |
| `masks.py` | **New, simpler.** No seams (D5), a different bit set, and the per-cell base mask is probably unnecessary — with 16 fixed cells the cell axis is cheap to carry exactly |
| `worker.py`, `config.py`, `run.py` | **New.** `run.py`'s orchestration shape is reusable; it calls the builders by name |

---

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

---

## 6. Open questions

- **O1 — Source names and keys.** Unverified for this proposal. Get them from `lsxfel` on r0423;
  do not hardcode. `JUNGFRAU._main_data_key` is `data.adc` and `_mask_data_key` is `data.mask`,
  and `_det_name_pat` matches several MID naming conventions, so `detector_name` will very likely
  have to be passed explicitly with two JUNGFRAUs present.
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
  expected set per run rather than globally.
- **O5 — How to combine the two detectors in a plot.** After masking they populate 11.5–23.7 (jf1)
  and 9.8–18.5 (jf2), so they **overlap over 11.5–18.5 nm⁻¹** — this is not concatenation. Plot as
  two traces with the overlap visible: it exposes any normalisation mismatch between the detectors
  instead of hiding it, and the overlap is wide enough to be a genuine cross-check on the two
  PONIs. Pooling onto a common grid needs the two `N` arrays on the same absolute scale, which is
  untested and involves different solid-angle coverage and different masks; do that only after the
  two traces are seen to agree.
- **O6 — AGIPD ↔ JUNGFRAU q continuity.** AGIPD covers 0.077–1.067 nm⁻¹; JUNGFRAU populates
  9.8–23.7 after masking. The **gap is 1.07–9.8 nm⁻¹** — wider than the raw detector ranges
  suggest, and with no overlap to cross-normalise against. Any combined SAXS+WAXS curve is two
  disconnected pieces with a free relative scale between them. Say so explicitly wherever such a
  plot is produced, and note the predecessor's fcc peaks at 0.6/0.7/1.0 nm⁻¹ sit in the AGIPD
  range, not here.

---

## 7. Phases and acceptance

**W1 — data and geometry facts.** Adapt benchmark stages 1 and 7 to JUNGFRAU; confirm O1, O2, O3
and O4 across several runs, including a crystallised one (r0426). Gate: the §2 table is confirmed
run-to-run, or amended with what varies.

**W2 — operator and error model.** PONI loading with the R4 wavelength assertion; dense
integration with an explicit `variance=`; the D3 constants measured per run rather than
hardcoded. Gate: a synthetic frame with known Poisson statistics recovers its input variance, and
the wavelength assertion fires on a deliberately mismatched PONI.

**W3 — masks, plan, worker, writer.** Static `.edf` ∪ `data.mask`, lit-cell selection with the D4
loud failure, the row model over the rectangular cell axis, the AGIPD writer behind its Protocol.
Gate: the AGIPD P3 test list (dropped train, missing modules, zero-frame train, worker error,
killed worker, resume, config-hash mismatch) on a JUNGFRAU mock run.

**W4 — on-node acceptance, r0423.** Both detectors, full run. Gate: pooled I(q) matches a dense
pyFAI reference; every non-OK status accounted for; the lit-cell set matches the config; timing
recorded. **Budget under load, not one core at a time** — see the AGIPD P4 note, where every
single-core figure came in 1.3–1.6× optimistic.

**W5 — DAMNIT integration.** Two variables backed by `analysis.waxs.damnit`, plus the combined
overview of O5.
