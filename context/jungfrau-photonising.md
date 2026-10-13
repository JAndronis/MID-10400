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
