# Photonising JUNGFRAU (`analysis.waxs.photons`): measurements and design

Context file for CLAUDE.md open task 18 (pitfalls 23 and 24), session S2. Read
`context/jungfrau-waxs-integrator.md` first; this file is a delta against it.

User decision, 2026-10-13: *"it measures in keV so it must be photonized"*, on **both** detectors,
inside the pass, with the settings recorded.

---

## 1. Phase 1: what the data say (W7, 2026-10-13)

`scripts/w7_photonize_check.py`, JSON `scripts/w7_photonize_check_20261013T135756Z.json`, Slurm
25286569 on max-exfl356 (upex, EPYC 7542), 4.5 min. 100 evenly spaced trains each of **r0423**
(science block, lit `{0…6, 15}`, dark 7–14) and **r0370** (ferritin 50 mg/ml without PEG, all 16
cells lit, no dark cell), jf1 and jf2. Both runs: `gainMode` dynamic, `settings` gain0,
`storageCellStart` 15 on both detectors (raw `…/DET/CONTROL`), XGM photon energy 9.000 keV on every
train, proc corrected with the same-day dark of 2026-05-11 (`proc/r*/calibration_metadata_*.yml`).
Per-train arrays: `scratch/s2_jf_photonise/w7/w7_r{run}_{det}_arrays.npz`.

The estimator is p010646's (`transfer_context/templates/jf_photon_count.py`): per 256 × 256 ASIC,
leaving out its 3 edge pixels, the median of the pixels below 2 keV, then three times the mean of
the pixels within ±1 keV of the current estimate. Pixel sets, all `data.mask`-good and finite:
**A** every such pixel, **K** the ones the static `.edf` keeps (the integrated pixels), **S** the ones
it masks (shadowed or excluded, but read out).

| Question | Answer |
|---|---|
| **ASIC tiling** | Bit 22 `NON_STANDARD_SIZE` covers exactly rows {255, 256} and columns {255, 256, 511, 512, 767, 768}, with no bit-22 pixel outside the 3-px ASIC edges. The template's 2 × 4 tiling of 256² is the proc layout, on both detectors |
| **Pixels per ASIC** | Set A: ≥ 55 414 of 62 500 inner pixels on every ASIC of both detectors, in every frame. Set K: jf1 `[0, 36950, 34273, 9739, 33230, 10583, 0, 0]`, jf2 `[0, 6435, 35249, 38998, 0, 0, 0, 0]`. Those ASICs that set K leaves empty integrate nothing either. No ASIC ever lacked a level in set A |
| **The level moves per module on jf1, per frame on jf2** | jf1: train-to-train rms 0.13–0.16 keV per ASIC, the same in lit and dark cells. ASIC correlation 0.95, 95 % of the variance module-common. Between-train 0.13–0.16 keV, within-train 0.02–0.03 keV. This is p010646's common mode. jf2: train-to-train 0.03–0.04 keV, between-train 0.011–0.013, within-train 0.03–0.04, correlation 0.65. **No per-train jump on jf2** |
| **Lit cells sit higher than dark ones** | Same train, cells 8–14 as reference: **+0.127 keV (jf1) / +0.128 (jf2)** on average, and the shadow set S shows the same (+0.125 / +0.131). So it is a real zero-level shift, not the estimator picking up signal. By cell (jf1; jf2 within 0.01): 0 +0.13, 1 +0.17, 2 +0.11, 3 +0.16, 4 +0.11, 5 +0.16, 6 +0.10, **15 +0.06**: odd cells higher, and cell 15, acquired first, lowest. It grows with intensity: by occupancy tercile +0.108 / +0.126 / +0.148 keV (jf1), the same on jf2. r0370, at a third of r0423's occupancy, shows the same odd/even pattern at a third of the amplitude (0.02–0.05 keV) |
| **Cell 7, read straight after lit cell 6** | Its level is +0.09 keV above cells 8–14 on both detectors, cell 8 +0.02, then flat: a decaying leftover of the lit cells' shift. Its spatial leftover is small: the slope of cell 7's image against the lit image is 0.008 (jf1) / 0.017 (jf2), against a floor of 0.004–0.007 in cells 9–14, i.e. ≤ 1 % of the lit signal (p010646 saw ~3 %) |
| **Dark-cell mean** | Over 100 trains, cells 8–14: **−0.04 keV (jf1), −0.00 (jf2)**. The −0.16 keV in §2 of the integrator file and in pitfall 23 was one train of jf1's common mode, whose rms is 0.15 keV |
| **What σ_read lumps together** | As `waxs/cells.py` measures it (one std over pixel × dark cell × train, union mask): 0.354 (jf1) / 0.319 (jf2) keV, reproducing W1. After subtracting each frame's ASIC level: 0.317 / 0.316. Pure per-pixel, per-cell temporal noise: **0.295 / 0.299**. So the two detectors have the same pixel noise, and jf1's extra 0.04–0.06 keV is its common mode. `DEFAULT_READ_NOISE_KEV`'s jf1/jf2 difference is the common mode |
| **Static offsets** | Per 64-column ADC group, relative to its ASIC: rms 0.021 (jf1) / 0.015 (jf2) keV, max 0.049 / 0.037, stable between cells to 0.001. Per pixel: rms 0.042 / 0.040. Between dark cells: 0.009–0.010 / 0.005–0.007 |
| **Estimator convergence** | 10 against 3 iterations: ≤ 0.0009 keV. Median only against 3 iterations: ≤ 0.004 keV |
| **Estimator against shadow (lit frames)** | A − S: +0.003 (jf1) / −0.004 (jf2) keV mean, 0.008 std. K − S: +0.011 / −0.031, partly static (K − A in dark frames +0.002 / −0.012). r0370, no dark cell: A − S +0.001 / −0.003, std 0.006 |
| **Sensitivity to a level error** | A 0.1 keV level error moves **1.0–1.1 %** of the one-photon pixels. E/2 sits in the spectrum's valley, but the valley is only 0.23–0.28 of the photon peak, because the charge-sharing continuum fills it. So rounding damps an additive offset only ~2.5× against integrating keV; subtracting the per-frame level is what removes it |
| **Single-photon peak** (level-subtracted) | jf1 8.96 keV (0.991 E), jf2 8.91 (0.985 E) on r0423; 8.99 / 8.90 on r0370. Valley at 4.4–4.8 keV |
| **keV / (E · photons)** | Level-subtracted: **1.010 (jf1) / 1.004 (jf2)** on r0423, 1.013 / 1.002 on r0370. Rounding loses 0.4–1.3 % of the energy to charge sharing (p010646: 4.6 % at 15 keV). Over q: 1.001–1.015 / 0.988–1.013 (2–98 %), rising slowly with q on jf1. Along r0423, whose occupancy *falls* from 0.50 to 0.36 (jf1): first and last decile 1.0096 / 1.0108, so no drift beyond 0.1 % |
| **What the keV pass carries** | Raw keV / (E · photons): 1.031 (jf1) / 1.035 (jf2) on r0423. The zero level adds 3 % to today's keV I(q), more where I(q) falls: jf1 1.023 at q 12 nm⁻¹ → 1.040 at 23. Over the overlap 11.5–18.4 nm⁻¹ jf1 runs 1.018–1.035 and jf2 1.025–1.041, a different shape on each detector, which is §6 O5's third candidate. Per train, jf1's 0.15 keV common mode is 0.15 / (E · 0.45) ≈ **3.7 %** of its keV intensity; jf2's 0.035 keV is ≈ 1.0 % |
| **E/2** | Count ratio 2.006 (jf1) / 2.000 (jf2) on r0423, 2.008 / 1.998 on r0370; 1.98–2.01 across q |
| **9.000 instead of 9.04 keV** | +0.32 % photons (r0423), +0.27 % (r0370). Open task 15 moves counts by 0.3 % as well as q |
| **Over-dispersion D** | **Under-dispersed**: per-frame median D = 0.934 (jf1) / 0.932 (jf2) at r0423's 0.40–0.45 photons/pixel, and 0.955 / 0.946 at r0370's 0.14–0.16. D falls with occupancy: 1.00 at 0.03, 0.975 at 0.1, 0.95 at 0.25, 0.92 at 0.6, 0.89 at 1.0. p010646 saw the opposite (D ≈ 1.7 at 3 photons/pixel). So a Poisson variance here *overstates* the per-pixel variance by 5–7 % |
| **Extreme pixels, kept region** | 0 in the 100 sampled trains of either run |

## 2. What phase 1 fixes for phase 2

- **The level comes from each frame itself, per ASIC.** Dark cells cannot stand in for it: lit
  cells sit 0.06–0.17 keV higher, depending on the cell and on intensity. A per-run constant cannot
  follow jf1's per-train jumps (0.15 keV rms).
- **Pixel set A** (every `data.mask`-good pixel, `.edf`-masked ones included). It agrees with the
  photon-free shadow set S to 0.004 keV, keeps ≥ 55 k pixels on every ASIC, and does not depend on
  where the `.edf` happens to leave pixels. Edge 3 px, cut 2 keV, window ±1 keV, 3 iterations, as
  the template.
- **Minimum pixels for a level: 1000.** Derived, not measured to fail: the window mean of n pixels of
  0.30 keV noise has σ ≈ 0.30/√n, which is 0.0095 keV at n = 1000. Through the 1 %/0.1 keV
  sensitivity that is ≤ 0.1 % in counts, an order below the estimator's own 0.008 keV scatter
  against S. It never fires on these detectors (minimum 55 414).
- **E = `cfg.photon_energy_kev` (9.04)**, stored per frame. Counting at 9.000 would add 0.3 %.
- **Variance: Poisson on the counts.** Given D = 0.93–0.95 it is conservative by 5–7 % in variance;
  W4's gate E compares it with the empirical frame-to-frame scatter.
- **σ_read is a diagnostic from now on.** As measured it is mostly pixel noise (0.30 keV) plus jf1's
  common mode.

---

## 3. Phase 2: the pass counts photons (implemented 2026-10-13)

**Per frame, in this order** (`analysis.waxs.integrate.prepare_frame`):

1. the union mask: `(data.mask & mask_bits) != 0` or the static `.edf`;
2. `extreme_pixels` drops the kept pixels that are non-finite or beyond `±max_abs_kev`
   (pitfalls 21 and 24). This happens before counting, so a gain-switched or NaCl-spot value
   never reaches a count;
3. `photons.zero_levels` estimates each ASIC's level from the frame itself, over the `data.mask`-good
   pixels (`level_pixels = "all_good"`, `.edf`-masked ones included). This is float64, stored as
   float32. An ASIC with fewer than `level_min_pixels` usable pixels, or with no value below the cut,
   has **no level**, and its integrable pixels are dropped and counted in `n_level_missing`. A level
   far outside ±1 keV (a broken pedestal, say) therefore gives no level rather than a wrong one;
4. `photons.count_photons`: `clip(rint((x − float32(level)) / float32(E)), 0)`, in float32, from the
   *stored* values, so any frame can be recounted from proc bit for bit;
5. pyFAI integrates the counts with every excluded pixel as NaN, `variance = counts`: Poisson,
   `Σc²·n` per bin, the AGIPD form. Negative variance bins can no longer occur, and the ledger asserts
   it.

The worker passes the **dynamic** mask alone, because the level uses `.edf`-masked pixels and the
operator supplies the static mask. `integrate_frame(..., timings=)` splits the worker's time into
`photonise` and `integrate`.

**Config** (all in `config_hash`, by-name pickling unchanged):

| Field | Default | Note |
|---|---|---|
| `count_energy_kev` | `None` → `photon_energy_kev` | Resolved in `__post_init__`, like the noise fallback. Separate so that the count can follow open task 15 without moving q |
| `level_pixels` | `"all_good"` | or `"kept"` (the `.edf`-kept pixels only) |
| `level_cut_kev`, `level_window_kev`, `level_iterations`, `level_edge_px` | 2.0, 1.0, 3, 3 | the template's |
| `level_min_pixels` | 1000 | derived in §2 |

`read_noise_kev` and `read_noise_fallback_kev` stay in the hash. σ_read no longer enters the
variance, but it is still stored (`cells/read_noise_kev`, `provenance/error_model`), and a field that
changes a stored number belongs in the hash.

**Stored**, beside the existing columns:

| Where | What |
|---|---|
| `frames/asic_level` (n, 8) f4 | Zero level per ASIC in keV, as counted with. NaN for an ASIC with no level in an integrated frame (0.0 is a real level), 0 in rows never integrated. The one column exempt from "no NaN in the file" |
| `frames/n_level_missing` u1 | ASICs that had pixels to integrate but no level |
| `frames/count_energy_kev` f4 | The E counted at, per frame |
| `frames/n_photons` u4 | Photons over the integrated pixels |
| `frames/energy_valid` | **Unchanged meaning**: raw keV over the integrated pixels, before any level is subtracted. Do not read `units` into it |
| `provenance.attrs["units"]` | `"photons"`, written at layout time, so even an unfinished file says what it counts. A file without it is a keV file |
| `provenance/photon_counting` | the settings and a one-line method string |
| `provenance/photon_levels` | the per-ASIC median and frame-to-frame rms of the level, and the missing-level counts |

**Readers.** `writer.file_units(path)` returns `"photons"` or `"keV"`. `per_cell` and `pooled_per_train`
carry `units` on their result. `combine.require_same_units` raises `UnitsMismatch`, and
`combine_files`, `damnit.combined_curve` and `damnit.overview_figure` call it, because a fitted
factor between a keV and a photon curve would absorb the unit change as a detector mismatch. The
overview's y-label follows the unit. The writer refuses to resume into a file whose `units` differ
or are absent (`SchemaMismatch`). In practice the config hash already differs, so this is a second,
explicit guard.

**Shared layer.** `common.writer.FrameTableWriter` gains `FRAME_ARRAYS` (per-frame `(n, k)` columns)
and `UNITS`. Both are empty for AGIPD, whose output digest is unchanged (the SAXS snapshot test).

**Speed.** The estimator costs 5.7–8.6 ms per frame on one login-node core (synthetic frames at
0.15–1.0 photons/pixel), against ~2 ms for the dense integration. Iterating the window over the
below-cut values whenever the window ends below the cut is exact and halves the cost. The node
figure is in §4.

**Tests.**
- `tests/waxs/test_waxs_photons.py`:
  - per-ASIC offsets under Poisson photons are recovered to < 5 eV;
  - a common-mode jump is followed;
  - counts equal the truth;
  - **E/2 exactly doubles** them;
  - too few pixels and an above-cut level give no level;
  - the estimator matches the template's `baseline` to 1e-12 keV;
  - counts are reproducible from the stored float32 values.
- `test_waxs_integrate.py`:
  - a masked ASIC is not counted;
  - a fully masked one costs nothing;
  - the NaN-versus-mask gate holds on counted frames;
  - extreme pixels are dropped before counting.
- `test_waxs_writer.py`:
  - the mock now carries a zero level per ASIC and train plus a lit-cell shift, and the stored
    `asic_level` of every frame recovers it to 0.01 keV;
  - a file without `units` is not resumed.
- `test_waxs_combine.py` and `test_waxs_damnit.py`: keV and photon inputs are refused.
- `test_waxs_scatter.py`: the σ ratio is 1 for a pure counting process with 20 % flux jitter.
- `tests/common/test_pinned_digests.py` and the WAXS output digest were updated deliberately.

---

## 4. Acceptance: W4 on r0423, keV against photons (2026-10-13)

Slurm 25287845, then the repeat 25287881, both on **max-exfl226** (allcpu, Intel Gold 6240, 2 × 18
cores, the DAMNIT node type in CLAUDE.md), with 36 workers and pyFAI 2026.5.0. No 9374F node was free,
so these timings are *not* comparable with the September W4 targets (17.5 / 20.1 s on max-exfl484, a
9374F). keV and photons, however, ran on the same node type, so they are comparable with each other.
- **keV:** main's code (deploy clone `ac24b89`, the W4 rerun due since 2026-09-14),
  `scratch/s2_jf_photonise/kev/`.
- **Photons:** this branch (`b6c55f8`, a venv without the dev group), `scratch/s2_jf_photonise/photons/`.
- JSONs: `scripts/w4_acceptance_r0423_{jf1,jf2}_{kev,photons}_*.json` and
  `scripts/w7_before_after_r0423_20261013T144611Z.json`.

| Gate | keV jf1 / jf2 | photons jf1 / jf2 |
|---|---|---|
| 0 configuration | pass: 3001 trains, 24 000 frames, lit `{0…6,15}` | pass, `units = photons` |
| A self-test (NaN vs mask) | 0.0 / 0.0 | **0.0 / 0.0** |
| B reference | 6.5e-8 / 8.0e-8 | **7.4e-8 / 7.5e-8**. Template levels against stored `asic_level`: **0.0 keV** difference, missing-level patterns equal |
| C wall (target 600 s) | 36.2 / 32.7 s | 52.3 / 49.3 s (repeat; the first run's jf2 took 80.7 s at efficiency 0.39 with the same worker CPU time, a one-off) |
| C ms/frame/core | read 13.5 + mask 11.6 + frame 12.0 (jf1) | read 12.3 + mask 11.7 + **photonise 23.6** + integrate 5.0 (jf1) |
| D ledger | 24 000 OK; frames with a negative-variance bin 493 / 9162, still negative after pooling 0 / **8** | 24 000 OK; **0 / 0** negative bins; 0 missing levels; E 9.04 on every frame |
| E σ ratio (frames / model) | 0.976 / 0.953 | **1.012 / 1.002** (overlap 1.009 / 1.003) |

- **The counting costs about 24 ms per frame per core on this node under full load**, about +45 % of
  wall. The pyFAI step itself is 5 ms. The keV path's 12 ms "integrate" also included building the
  keV variance. A final timing on a 9374F node follows S3's pyFAI 2026.9 upgrade. If it ever
  matters, estimating only the ASICs that have integrable pixels would save 40–60 % (jf1 integrates
  5 of 8 ASICs, jf2 3 of 8), at the cost of the common-mode diagnostic on the rest.
- **The Poisson σ is the scatter the frames show**, to 1.2 % on jf1 and 0.2 % on jf2. The per-pixel
  D of 0.93 predicts 0.97, so a train's frames differ by slightly more than counting noise: about
  0.3 σ, plausibly cell-to-cell shape differences. Per-frame σ is now usable on every frame. Before,
  jf2 had a non-positive bin in 38 % of its frames, and 8 bins stayed negative even after pooling.
- **Gate E cannot see jf1's common mode**, because it is shared by every cell of a train. What does
  see it is keV over photons train by train, on the same frames: **3.81 % rms on jf1 and 0.35 % on
  jf2**. W7 predicted 3.7 % for jf1, and 1.0 %/√8 for jf2, whose level moves per frame. The per-train
  spread of jf1's intensity drops from 0.1695 to 0.1653, and √(0.1653² + 0.0381²) = 0.1696: the keV
  spread is the photon spread plus that zero-level noise. jf2's stays at 0.1600. The 16 % itself is
  the sample and the beam.
- **keV over E · photons** across the run: 1.036 (jf1) / 1.035 (jf2), rising with q (jf1 1.029 →
  1.045 over 11.5–23.7 nm⁻¹), as W7 found on 100 trains.

**jf1/jf2 overlap (CLAUDE.md open task 5), 11.53–18.44 nm⁻¹, 285 bins:**

| | factor | χ²ᵣ | residual rms | slope | residual at q_low / q_high | σ/I |
|---|---|---|---|---|---|---|
| keV | 0.98823 ± 0.00005 | 54.1 | 0.743 % | 2.57e-3 /nm | −1.12 % / +0.66 % | 0.101 % |
| photons | **0.99076 ± 0.00005** | **31.4** | **0.640 %** | **2.56e-3 /nm** | −1.12 % / +0.65 % | 0.114 % |

The keV row reproduces 2026-09-13 exactly. Photon counting moves the factor by +0.26 % and removes
0.1 % of the residual, but the **tilt is unchanged to 0.4 %**. So the uncorrected zero level (§6 O5's
third candidate) is *not* what tilts jf2 against jf1. That leaves polarisation and a relative
q-axis error, which r0426 (with Bragg peaks in the overlap) and the polarisation correction can
separate.
