# Run triage (`analysis.triage`): design and implementation guide

Context file for `<pkg>.triage`. It characterises every run of p010400 from what DAMNIT already
holds and ranks candidates for the "red box". Read `CLAUDE.md` first.

The file follows the integrator guides' conventions:
- prose plus pseudocode (working rule 3)
- every threshold with a traceable source (rule 2)
- one phase at a time (rule 1, §9)

**State:**
- T0 (this spec) was reviewed on 2026-10-01 and revised to the keep rule in §1.
- **T1 is done:** acceptance passed on the real data on 2026-10-01 (§4.2, §9).
- T2–T4 are not started.

---

## 1. Purpose, scope and the keep rule

The beamline staff asked which runs to keep. As of 2026-10-01:
- DAMNIT outputs and reduced data stay on disk.
- **Raw and proc go to tape.**
- A limited selection of raw *or* proc can be kept on disk in a **"red box"**. Its size is not
  known.
- Tape is not deletion. A taped run can be restored. The red box holds what will be actively
  reanalysed.

**The keep rule** (user, 2026-10-01): a run is a red-box candidate **if its dynamics are
extractable**, which is a question of XPCS signal-to-noise. The **only** exception is a run with
low SNR that **has a use as a background**.

The SNR is the **analytical multispeckle SNR** of one g2 point (§4.6), built from speckle contrast,
photons per pixel, pixel count and frame count:
- per 30 s slice of the run
- extractable means at least one slice reaches **SNR ≥ 3**
- the contrast β is measured from the data

The user chose all three on 2026-10-01; this replaced an earlier Γ–q exponent criterion.

Two things are *not* keep criteria:
- Crystallisation is a content flag. Its kinetics live in DAMNIT's 1D SAXS, which stays.
- XCCA is a low-priority analysis for now (user, 2026-10-01). XPCS ranks above it.

**In scope:**
- **The XPCS SNR of every run**, per 30 s slice (§4.6). The per-train TTCFs supply the measured
  contrast and an empirical check of the noise.
- Per-run content and quality metrics from DAMNIT: SAXS, WAXS, the droplet fit and run metadata.
- A per-train characterisation of the SSHEX mesh scans: on/off droplet, on-droplet-only XPCS,
  and use as a background.
- A ranked red-box list with cumulative raw and proc TB.
- A barebones report: CSV, JSON and a short markdown summary.

**Out of scope:**
- The keep decision itself, which belongs to the group.
- Re-integration and background subtraction. The triage reads stored data and never alters it.
- Commissioning runs r1–42 beyond the inventory (user, 2026-10-01: tape, no analysis).

**Priority:** data quality > completeness guarantees > speed, as everywhere in this repository.

---

## 2. Inputs, and what they actually hold

Every input is opened **read-only**:
- DAMNIT's sqlite, `usr/Shared/amore/runs.sqlite`, opened as `file:…?mode=ro`.
- The per-run DAMNIT HDF5 files, `usr/Shared/amore/extracted_data/p10400_r{run}.h5`.
- Raw, only for the SSHEX motor positions and, in T3, the pulse-rank XGM.

| Source | Content | Measured on, 2026-10-01 |
|---|---|---|
| sqlite `runs` | `run_type`, `sample` (stale, §3), `scan_type`, `n_trains`, `pulses`, `raw_size`/`proc_size` [TB], run-mean `total_transmission`, `xgm_intensity` | 508 runs, r1–510; raw total **994.8 TB**, proc 65.2 TB |
| `xpcs_ttcf` | per train: TTCF_raw − TTCF_off, (trainId, roi_q=7, frame_1, frame_2), float32, **NaN where either frame's kbar < 3e-3** (`extra_speckle/xpcs/tools.py:251-255`) | **The input to §4.6.** r423: (3000, 7, 155, 155), 1.47 GB on disk; one process reads it and runs the jackknife in **28 s**. r120 (mesh): (833, 7, 350, 350), 0.13 GB. **Finite fraction** per ROI: r423 1.00 for ROIs 0–3, 0.99, 0.67, 0.00; r120 0.09–0.19 (mostly off-droplet frames below kbar) |
| `xpcs_mean_dataset` | `g2` (roi_q, tau), `ttcf_mean` | **g2 is TTCF_raw − TTCF_off, so it decays to 0, not 1** (`tools.py:271-313`). It is the train mean, the last 10 lags are dropped, and lags beyond 10 are log-binned by `nanmedian` (`tools.py:370-398`). There are no error bars and no fit parameters (`pipeline.py:586-596`). The triage uses it only as a cross-check (§9 T2) |
| ROI geometry | `roi_q` centres in **Å⁻¹**: 0.029, 0.040, 0.051, 0.0605, 0.067, 0.072, 0.0805. Edges are the context's `args["q_range"]` = [0.0225, 0.035, 0.045, 0.056, 0.065, 0.069, 0.075, 0.086] (`src/amore/context.py`) | DAMNIT stores the centres only |
| Lag axis | lag k ⇒ τ = k / 2.2569 MHz = **k × 0.4431 µs**. Usable lags k = 1 … F − 11: up to **60 µs** in static runs (F = 155) and **138 µs** in mesh runs (F = 350) | matches DAMNIT's `rep_rate` (2.2569 MHz) and the stored `tau` |
| `agipd_saxs` | `intensity` (trainId, pulseId, q) float32, `n_frames` 0/1. An empty slot is 0, never NaN | q **0.0813–1.0570 nm⁻¹, 2000 uniform bins of 4.881e-4 nm⁻¹** (r423; deployed `npt=2000`). **It is k̄ in photons/pixel/frame**: the pass divides by Σ c·Ω, where Ω is pyFAI's *relative* solid angle (`saxs/operator.py:181`), ≈ 1 at these angles |
| `scratch/agipd_saxs/r{run}/agipd_saxs.h5` | the pass's per-frame `frames/normalization` (n, npt) = Σ c·Ω over kept pixels | **N_pix per bin.** r423, per XPCS ROI (nm⁻¹): 0.225–0.35 → 34 752 px, 0.35–0.45 → 100 813, 0.45–0.56 → 152 793, 0.75–0.86 → 150 035. 461 runs have a file. Written by DAMNIT with `overwrite=True` |
| `jungfrau_waxs_jf1/jf2` | `intensity` (trainId, cellId, q=200), lit cells only | jf1 q 11.54–23.66 nm⁻¹. All 16 cells up to r378, `{0…6,15}` from r379 |
| `droplet_fit` | `volume` [mm³], `radius`/`center` (trainId, dim x/y) [camera px, 13.9 µm], `phi` | absent or all-NaN for r1–42, r52–56 and r349 |
| `xgm_intensity`, `total_transmission` | per train; the XGM is the mean over **all** pulses | see D4 |
| `.errors/<var>` | the exception text of a failed variable | the XPCS family failed on r52, 56, 57, 140, 356 and 382; `agipd_saxs` on r356 and r382 |
| raw `MID_EXP_SAM/HEXAPOD/SSHEX` | `X/Z.actualPosition.value` per train | verified with `lsxfel` on r120. Must be verified per run |

Missing products among the non-calibration runs:
- `agipd_saxs`: r349, r356, r382
- `jungfrau_waxs_jf1`: also r68, r69, r140
- XPCS: also r52, r56, r57, r140

A missing product is recorded as missing, never imputed.

### 2.1 The SSHEX "mesh" is a fly scan (r120, r318)

- **Z sweeps 1 mm continuously** in each of 11 X columns 0.1 mm apart (~72 trains per column).
  `scan_type` reports a `dmesh`.
- **The camera rides with the hexapod.** On r120 the droplet centre in the camera does not follow
  X (r = 0.01).
- **The droplet is oblate**, so only **6–23 % of trains hit it**: 50–192 of 718–843 on the runs
  probed.
- **The on/off separation is large in SAXS, small in WAXS.** Mid-q SAXS (0.3–0.5 nm⁻¹) per flux
  on/off is **14–28×**; the WAXS ring is ~2.5× (r120). This sets D5.
- **r307** (attenuator 0.0047, the first run at Z ±0.25) did not separate in a median-split
  probe, so the classifier must be able to report ambiguity.
- **All-train XPCS on a mesh run carries no signal.** r120 gives A₀/σ ≈ 0 in every plain ROI
  (§2.3), so mesh extractability can only be judged on `on` trains (§4.6).

### 2.2 What the mesh period is

`droplet_fit` shows **~14 complete droplet evaporation series** in r68–337, ~18 runs each, from a
fresh droplet of V ≈ 2–3 mm³ to a plateau:

| Runs | Attenuator | Sample (§3) |
|---|---|---|
| r68–69 | 1.0e-4 | silica NP |
| r70–87 | 0.00102 | ferritin + PEG 6K. No g2: every ROI fell below the kbar filter |
| r88–185 | 0.0123 | ferritin + PEG 6K |
| r186–239 | 0.051 | ferritin + PEG 6K |
| r240–307 | 0.101 (r307: 0.0047) | ferritin + PEG 6K |
| r308–337 | 0.101, Z ±0.25 | pure ferritin (r336–337 are non-mesh, same droplet) |

Mesh runs are ~840 trains at 350 pulses. Static runs are ~3000 trains at 155 pulses, at
attenuators 1.0, 0.6 and 0.506.

### 2.3 Probes behind this spec (session scratch, not in the repo; T2 re-derives them)

**The analytical noise matches the measured one.** On r423, ROI 0.040 Å⁻¹ (0.35–0.45 nm⁻¹):
- k̄ = 0.0167, N_pix = 100 813, F − 1 = 154 pairs per train, 3000 trains.
- The analytical noise of one g2 point is √2 / (k̄ √(N_pix (F−1) N_trains)) = **3.9e-4**.
- The block jackknife on the per-train TTCFs (§4.6, 50 blocks) gives **3.8e-4** per lag.
- Without the √2 for the TTCF_off subtraction, the prediction would be 2.8e-4.

**Jackknife fits** (§4.6; 50 blocks; plain ROIs; single exponential):

| Run | Kind | A₀/σ per ROI | σ_Γ/Γ | Γ–q exponent |
|---|---|---|---|---|
| r450 | static PEG 6K, late | 126, 115, 109, 48 | 0.05–0.07 | 0.61 ± 0.08 |
| r400 | static no-PEG, late | 49, 52, 86, 43 | 0.04–0.05 | 0.41 ± 0.06 |
| r423 | static PEG 6K, early | 6.7, 3.6, 3.2, 0.9 | 0.25 – 1900 | unresolved |
| r120 | mesh, **all trains** | −0.2, −0.4, 0.4, 0.1 | — | — |

- The late runs give A up to ~0.045 at 0.029 Å⁻¹. Those are the runs §4.6 measures β on.
- The exponents are well below 1: neither diffusion (2) nor shear (1). This may be a systematic of a
  single exponential on off-subtracted run means (O7). The triage does not use the exponent.

**Narrow (Bragg-like) SAXS excess** (§4.3 method; fraction of trains with z > 8, a probe value
that §5 replaces):

| Run | Kind | 0.57–0.65 | 0.68–0.74 | 0.98–1.05 |
|---|---|---|---|---|
| r370 / r410 / r423 | static no-PEG / PEG 1K / PEG 6K early | 0.000 / 0.000 / 0.004 | 0.000 / 0.000 / 0.000 | **1.000 all** |
| r425 / r426 | static PEG 6K, crystallised | 0.994 / 0.890 | 0.810 / 0.751 | 1.000 |
| r114 / r130 | mesh, on-droplet, att 0.0123 | 1.000 / 0.987 | 1.000 / 0.881 | 1.000 |
| r120 | mesh, on-droplet, late | 0.040 | 0.000 | 1.000 |

- **The 0.98–1.05 window fires in every run**, so it is instrumental. It is plausibly the median
  filter meeting the grid's upper edge at 1.057 (D6).
- **The XPCS Bragg-q ROI at 0.0605 Å⁻¹** rises and falls along every PEG 6K droplet. g2(τ_min)
  ×1e-3 runs r423 → r426: 22 → 1106 → 2780 → 475, and r112 → r120: 5279 → 8108 → 2009 → 552 → 732.
  Its control maxima are 11.8 (no-PEG static, r400), 8.7 (PEG 1K, r419) and 18.0 (pure-ferritin
  mesh, r331).
- **The 0.072 Å⁻¹ ROI is not clean.** It reaches 132.6 in the pure-ferritin mesh droplet (r320),
  near the ferritin form-factor minimum. Each Bragg ROI is therefore held against its own control
  bound (§4.7).

---

## 3. Run classes and sample identity

### 3.1 Classes

Classes are assigned in this order. The first rule that matches wins.

```
if run_type starts with "Calibration" or "Dark":        calib_dark
elif run_type starts with "Test":                       test
elif run <= 42:                                         commissioning      # user, 2026-10-01
elif "SSHEX" in scan_type:                              mesh_sshex
elif effective_sample in {LaB6, Air, LUDOX, Silica NP}: reference
else:                                                   static_droplet
```

| Class | Runs | Raw TB | Proc TB |
|---|---|---|---|
| static_droplet | 146 (r336–496) | 524.9 | 33.8 |
| mesh_sshex | 255 (r68–335) | 330.4 | 23.5 |
| commissioning | 38 (r1–42) | 63.3 | 4.3 |
| reference | 19: r52–57, 62–64, 288, 382–385, 389, 497–500 | 49.9 | 3.1 |
| calib_dark | 46 | 20.6 | 0 |
| test | 4 (r58–61) | 5.6 | 0.4 |

### 3.2 Sample overrides (`triage.samples.OVERRIDES`)

DAMNIT's `sample` is myMdC's label and is stale for several runs. The overrides map run ranges to
the true sample, each with its source. They also carry a `background_role` (§7).

| Runs | True sample | DAMNIT label | Source | background_role |
|---|---|---|---|---|
| r289–303 | ferritin 50 + PEG 6K | Silica NP | user, 2026-10-01 | — |
| r308–335 | ferritin 50, no PEG | PEG6K / Ferritin_50 / Silica NP | user, 2026-10-01 (r308–309 checked by the user) | — |
| r463 | PEG 6K buffer until ~train 1550 (explosion, a bubble), then a fragment | Ferritin_50_PEG6K_5p | memory, 2026-09-29 | excluded |
| r464–466 | PEG 6K 5 % + 150 mM NaCl buffer, no ferritin | Ferritin_50_PEG6K_5p | user, 2026-09-28 | SAXS buffer for the 12 May PEG 6K droplets |
| r53–55 | air / empty beam | Air | label; r55 confirmed by the user, 2026-09-28 | empty beam, WAXS |

**The table holds corrections and roles only.** Labels the user confirmed are left alone:
- r68–69 are silica NP (2026-10-01).
- The `Ferritin_50` block r336–408 is ferritin without PEG (2026-09-28).

A blanket r336–408 entry would wrongly relabel the silica runs r382–385 and r389 inside that
range.

---

## 4. Metrics

Every metric is computed per run. For mesh runs, metrics are also computed per class of train
(§4.4). "Flux" F is the per-train XGM × `total_transmission`, used inside one run. Comparing flux
across runs needs D4.

### 4.1 Inventory (T1)

Recorded per run:
- class, effective sample, override source, background_role
- raw/proc TB, n_trains, pulses per train, attenuator
- XGM mean and coefficient of variation over trains
- per DAMNIT variable: `present`, `errored` (with the first line of `.errors/<var>`) or `absent`
- JF readout block: `all16` or `lit{0…6,15}`
- stray-light era: `11may` up to r433, `12may` from r443 (memory, saxs-background-subtraction)

### 4.2 Droplet series (T1)

Evaporation never increases a droplet's volume. So the question at each run boundary is whether
the volume went **up** by more than the fit can scatter.

```
lv        = log10(volume) of a run's finite fits
null_d(r) = differences between adjacent 20-train medians of lv inside run r   (n_d(r) of them)
σ(r)      = 1.4826 · MAD(null_d(r) − median(null_d(r)))      # the run's own fit noise, at this window
ν(r)      = 0.37 · (n_d(r) − 1)          # dof of a MAD (Gaussian efficiency 0.37, Rousseeuw & Croux 1993)
            fewer than 3 differences → σ = median over all runs, taken as known (ν = ∞), flag noise_fallback

for consecutive fitted runs a < b (runs without a fit between them are skipped, not breaks):
    d      = median(lv_b[first 20]) − median(lv_a[last 20])
    z      = d / sqrt(σ(a)² + σ(b)²)
    z_cut  = Student t, one-sided, α = 0.1 / n_boundaries, at the Welch–Satterthwaite dof of σ(a), σ(b)
    significant ⇔ z > z_cut
    kind   = sample_change   if the effective sample changes between a and b
             fresh           if significant ∧ d > d_fresh          # d_fresh: §5
             increase        if significant ∧ d ≤ d_fresh          # chained, flagged
             continuing      otherwise
    label_suspect ⇔ sample_change ∧ ¬significant       # one droplet, two labels?
    underpowered  ⇔ continuing ∧ d_fresh / σ ≤ z_cut   # an injection could not have been seen here
break the series at b  ⇔  fresh or sample_change
```

**Why Student t.** The z is studentised by noise estimated from a few dozen windows: ~42 for a
mesh run, 150 for a static one. A normal cut ignores that, and inflates the false count. In a
simulation of clean runs (`tests/triage/test_series.py`), the t cut holds 0.07–0.09 expected
false flags per ten-run triage against a design of 0.1. With the normal cut, the first test
history raised two.

**`σ` overstates a boundary's noise by √2, deliberately.** It is the noise of a *difference* of
two window medians, and it is used for both sides of the boundary. A boundary also carries a pause
and any change of camera conditions, which the in-run null does not see.

- **A run without a fit is a member, not a break**, if its class carries a droplet (static, mesh
  or reference). r349 sits inside the r338–369 droplet the notebooks already treat as one.
  Calibration, dark and test runs are never members. r353–355 are calibration runs taken during
  that droplet; they are skipped without breaking it.
- **A pause is not a break.** r378 → r379 drops from 0.79 to 0.21 mm³ across 16 minutes, and the
  notebooks use r370–381 as one droplet.

**In-run steps**, per run:

```
step       = max |d − median(d)| over the run's adjacent window-median differences   (max_jump_dlog)
step_z     = step / σ(r)                                                               (max_jump_z)
step_cut   = Student t, two-sided, α = 0.1 / (all differences of all runs), dof ν(r)
volume_jump   ⇔ step_z > step_cut ∧ step ≥ d_fresh      # as large as an injection
volume_glitch ⇔ step_z > step_cut ∧ step <  d_fresh      # significant but glitch-sized
```

Significance alone is not enough. Static runs with fit noise of ~0.1 % make 1 % shape wobbles
"significant". Before the size split, 54 runs were flagged; most of them were 0.4–7 % steps.

**T1 result on the real data** (2026-10-01, `scripts/run_triage.py --phase inventory`, 39 s on a
login node):
- 508 runs; 407 boundaries.
- d_fresh = 0.122, from the empty band between +0.087 and +0.171.
- **Fresh:** every injection, including the top-up r143→144.
- **`increase`:** r208→209 (+0.046), r256→257 (+0.051), r301→302 (+0.087). r159→160 (+0.038),
  an `increase` under the normal cut, is now ordinary noise.
- **`volume_jump`:** r149 (+0.29), r463 (+0.81, explosion) and r466 (+0.24, the unusable last
  block). 51 runs carry glitch-sized steps.
- **`label_suspect`:** **r389 → r390** (−0.015). One droplet runs from 3.15 to 2.17 mm³ across
  the label change `Silica NP` → `Ferritin_50`. **Checked by the user: the labels are right.**
  r389 is a silica droplet cut short before it evaporated, and r390 a fresh ferritin droplet
  that happened to start at a similar volume. So `label_suspect` is a prompt to check, not a
  verdict; this was its false positive. Verdicts live in `samples.CHECKED_LABEL_CHANGES` and
  are reported as `label_checked`, so a checked change is not asked about again.
- **`underpowered`:** 69 continuing boundaries, in three groups:
  - r96–129, where the droplet fits oscillate by up to 0.1 dex per window. This is where the
    0.78 → 1.46 mm³ step r96→97 hides, so r93–128 being one series is unproven.
  - r228–238.
  - the small droplets r318–337 and r352–370.

Per series: runs, V₀, V_min, the counts of each flag, and t* with its source
(`analysis.cache.droplet_timeline`, T3).

Per run:
- V/V₀ start and end
- fit coverage
- robust centre jitter in x and y [µm]
- σ(r), `max_jump_dlog`, `max_jump_z` and its cut, `volume_glitch`, `volume_jump`

### 4.3 SAXS (T2)

**Completeness:** the fraction of (train, pulse) slots with `n_frames == 1`, and the fraction of
trains with every pulse present.

**Stability, static runs:**

```
s_t = pulse-mean I(q) over 0.3–0.5 nm⁻¹ / F_t
r_t = s_t / running_median(s, 51 trains) − 1
outlier_t ⇔ |r_t| > 5 · 1.4826 · MAD(r)
```

Under Gaussian noise the 5-robust-σ cut predicts 0.002 false outliers in 3000 trains. Recorded:
the outlier fraction, and p1/p50/p99 of r.

**Narrow (Bragg-like) excess, per train** (a content flag, §1):

```
I_t(q)  = mean over pulses with n_frames == 1
base_t  = running_median(I_t, width = round(0.030 nm⁻¹ / Δq))   # 61 bins at Δq = 4.881e-4
res_t   = I_t / base_t − 1
σ_t     = 1.4826 · MAD(res_t over Q_noise)
          Q_noise = 0.25–0.95 nm⁻¹ minus 0.55–0.76
z_t[w]  = max over q in window w of res_t / σ_t
windows: B1 0.57–0.65, B2 0.68–0.74, C 0.40–0.50 (control), B3 0.98–1.05 (recorded, never flagged)
bragg_t ⇔ max(z_t[B1], z_t[B2]) > z_star        (§5)
```

- **B1** covers the fcc doublet at 0.584 + 0.633 nm⁻¹ and its merged 0.610 under the other
  beam-centre convention, so the flag does not depend on open task 4.
- **B2** covers the predecessor's 0.7 nm⁻¹ peak.
- Per run: the fraction of Bragg trains, the first and last Bragg train, and the medians of z per
  window.
- Mesh runs use their `on` trains only.

### 4.4 Mesh train classes (T2)

```
s_t = pulse-mean SAXS over 0.3–0.5 nm⁻¹ / F_t          # primary (D5)
w_t = jf1 ring 17–23 nm⁻¹, cell-mean / F_t              # cross-check
pool log10(s_t) over every mesh run in one attenuator block
find the widest histogram band with no counts between the low and the high mode
s_lo, s_hi = that band's edges
on   ⇔ s_t > s_hi;  off ⇔ s_t < s_lo;  edge ⇔ otherwise
```

- If the block has no empty band, the threshold falls back to the histogram minimum and the run
  is flagged `classification_ambiguous`. r307 is the expected case.
- The cross-check reports how far `w_t` agrees with the SAXS classes.
- **Off-droplet sub-classes** follow `notebooks/lowq_report.ipynb` cell 17 (r318), extended to the
  fly scan. Each off train stores `dist_mm`, the hexapod distance from the bounding box of the run's
  `on` trains.
- `beside` versus `beam_only` is decided in **T3**, at the measured `d_near` where the per-flux
  low-q background stops changing with `dist_mm` (§8 O1).

### 4.5 WAXS (T2)

Recorded per run:
- presence of jf1/jf2 and the JF readout block
- the ring mean per flux, on and off for mesh runs
- `attrs["data_check"]`
- where the scratch per-frame file `scratch/jungfrau_waxs/r{run}/jungfrau_waxs_jf1.h5` exists with
  the current schema: the fraction of frames with `n_extreme_pixels > 0`, the NaCl indicator of
  open task 16 and pitfall 21

### 4.6 XPCS SNR: the keep criterion (T2)

**Definition.** The analytical multispeckle SNR of one g2 point, from Möller, Sprung, Madsen & Gutt
(2019), *IUCrJ* 6:794 (arXiv:1906.09102, eqs. 8 and 18), after Lumma et al. 2000 and Falus et al.
2006:

```
SNR = β · k̄ · √(N_pix · N_fr · N_rep)          valid for k̄ ≪ 1 (here 1e-3 … 0.1)
β    speckle contrast         k̄     photons per pixel per frame
N_pix pixels in the ROI       N_fr  frames            N_rep  repetitions (trains)
```

- Speckle and pixel size enter through the contrast: β = β_cl · β_res, with
  β_res = ((2/w²) ∫₀ʷ (w − v) (sin(v/2)/(v/2))² dv)², w = 2πP/S and S = λL/a (eqs. 12–13).
- At MID the speckle (~58 µm at a ~15 µm beam, ~7 m) is far smaller than the 200 µm AGIPD pixel,
  so β is a few percent. Leonau et al. 2026, *J. Synchrotron Rad.* 33:725 (arXiv:2506.08668).

**As applied here**, per plain ROI and per 30 s slice:

```
slice    = 300 consecutive trains (30 s at 10 Hz; user 2026-10-01); a final part-slice of
           ≥ 150 trains is its own slice, a shorter one joins the previous slice
usable   = trains with every frame present; for mesh runs also train_class == on (§4.4)
k̄(q)     = mean of DAMNIT's agipd_saxs intensity over the usable trains' frames and the ROI's
           q bins (photons/pixel/frame, §2)
N_pix(q) = Σ over the ROI's q bins of frames/normalization in the pass's scratch file,
           median over 10 OK frames of the run (§2)
F        = frames per train; the shortest lag has F − 1 pairs per train
φ(q)     = fraction of finite k = 1 entries of xpcs_ttcf over the slice's usable trains.
           Pairs dropped by extra-speckle's kbar < 3e-3 filter carry no signal
SNR(q)   = β(q) · k̄ · sqrt(N_pix · (F − 1) · φ · n_usable) / √2
           the √2: the pipeline subtracts TTCF_off (the previous train), an independent estimate
           with the same noise, so the variance doubles (§2.3: 3.9e-4 predicted, 3.8e-4 measured)
extractable slice ⇔ max over plain ROIs of SNR ≥ 3                  (user, 2026-10-01)

per run:
    n_slices, n_extractable_slices
    best_snr, with its ROI and slice
    snr_run     the same formula over all usable trains
    t_needed_s  = 30 s · (3 / best_snr)²     # the slice length at which the best ROI reaches 3
extractable ⇔ n_extractable_slices ≥ 1
```

- **ROI mapping.** The XPCS ROI edges (Å⁻¹ × 10 → nm⁻¹) select the SAXS bins whose centres fall
  inside them.
- **Geometry caveat.** extra-speckle builds its own geometry (`geom=None` → motor encoders,
  pitfall 16), so its ROI pixels may not be ours. k̄ and N_pix come from our geometry. The
  measured-noise check below is what bounds the mismatch.
- **Mesh runs.** A 30 s slice of a mesh run holds only its `on` trains, ~20–70 of 300. That is the
  honest count of useful trains at that time resolution.
- **Plain ROIs only:** 0.029, 0.040, 0.051 and 0.0805 Å⁻¹, i.e. the ROIs whose edges miss B1 and B2
  (§4.3). Bragg-q ROIs carry crystallite speckle and are a content flag (§4.7).

**β(q) is measured from the data** (user, 2026-10-01):

```
per static run with XPCS, per plain ROI: the block-jackknife g2 (below), fitted with
    g2(τ) = A·exp(−2Γτ) + C,  A ≥ 0, Γ > 0, C free        # extra-speckle's model, xpcs/fit.py:120-163
eligible run ⇔ fit converged ∧ A/σ_A ≥ 10 ∧ 5·τ_1 ≤ 1/(2Γ) ≤ τ_max
β(q) = median of A over eligible runs;  report n_eligible and the 16–84 % spread of A
β_res(q) from S = λL/a (a = 15 µm, the MID paper's beam) is reported beside it, not used
```

- **Why only slow runs.** A is the model's τ → 0 intercept. A run whose decay is fast against the
  first lag shows a low A, because its contrast is gone before τ_1.
- **The two numbers in "eligible" are design choices, for review (O9).**
  - A/σ_A ≥ 10 fixes each run's β to 10 %.
  - 5τ_1 ≤ 1/(2Γ) keeps the τ → 0 extrapolation within a fifth of the decay time.
- **β is a property of the optics, and the optics never changed.** The CRL lens insertion state
  (raw `SA2_XTD1_CRL/MDL/CRL` and `MID_XTD6_CRL/MDL/CRL`, keys `actual.isInsertedLens{1…10}.value`,
  verified on r423) is identical in all 25 sample runs probed from r62 to r497: XTD1 lenses 3 and 5,
  XTD6 lenses 3, 5 and 6, constant within each run. Only the air run r55 differs (XTD6 lens 2 as
  well). So **one β(q)** serves the experiment. The triage records each run's state and flags
  `optics_differs` for a run outside the majority state; such a run gets no SNR until a β exists
  for its state.

**Block jackknife, for the β fits and the noise check:**

```
blocks: B = min(50, n_usable) contiguous blocks of usable trains
        (the jackknife variance is itself uncertain by sqrt(1/(2(B−1))) ≈ 10 % at B = 50)
per block b, per ROI: S_b = nansum over trains of TTCF, N_b = count of finite entries
g2(k)      = nanmean over i of [ΣS / ΣN][i, i+k],   k = 1 … F − 11
g2_(−b)(k) = the same with block b left out
σ_g2(k)    = sqrt((B−1)/B · Σ_b (g2_(−b)(k) − mean)²)
A, σ_A, Γ  = weighted fit (weights 1/σ_g2²), with σ from refitting each g2_(−b)
noise check: ρ = mean σ_g2(k = 1…3) / (√2 / (k̄ √(N_pix (F−1) φ n_usable)))
             expected 1; r423: 3.8/3.9
```

- **Cross-check against DAMNIT:** the all-train g2 must agree with DAMNIT's stored `g2` at k ≤ 10,
  within σ_g2 (§9 T2).

### 4.7 Crystallinity from the XPCS Bragg ROIs (T2, content flag)

```
for each Bragg ROI k ∈ {0.0605, 0.067, 0.072} Å⁻¹:  ratio[k] = A0[k] / a0_star[k]   (§5)
xpcs_bragg ⇔ any present k with ratio[k] > 1
```

---

## 5. Thresholds: derivation and control sets

No threshold is a constant in the code. Each is derived when the triage runs, or from a stated
requirement, and written to the JSON with its source.

**Control sets** (confirmed by the user, 2026-10-01): static **no-PEG** r336–408 (minus the
`.errors` runs and the silica runs), **PEG 1K** r409–419, and the **pure-ferritin mesh** r308–335
(`on` trains).

| Threshold | Source | Rule |
|---|---|---|
| SNR ≥ 3 per 30 s slice (§4.6) | user, 2026-10-01 | the keep criterion. A run is extractable when at least one slice reaches it in a plain ROI |
| A/σ_A ≥ 10 and 5τ_1 ≤ 1/(2Γ) (β eligibility, §4.6) | design choice, **for review (O9)** | each run fixes β to 10 %; the τ → 0 extrapolation stays within a fifth of a decay time |
| `z_star` (Bragg, §4.3) | controls, all trains | the 99.9th percentile of `max(z[B1], z[B2])`. It must still flag r425/426, r114 and r130 (§9) |
| `a0_star[k]`, per Bragg ROI (§4.7) | controls' A0[k] | the **maximum**: with 11–57 control runs per ROI a percentile would be meaningless. At derivation time: 18.0e-3 (0.0605), 43.9e-3 (0.067) and 132.6e-3 (0.072), all from the pure-ferritin mesh |
| `s_lo`, `s_hi` (§4.4) | each mesh attenuator block | empty band |
| `z_cut`, `step_cut` (§4.2) | multiple testing | Student t at α = 0.1 / n_tests (one-sided for boundaries, two-sided for in-run steps), with the dof of the MAD noise estimate (ν = 0.37·(n − 1), Welch–Satterthwaite across a boundary). The expected false count is 0.1; checked by simulation |
| `d_fresh` (§4.2) | the significant same-sample boundary increases | the geometric centre of the widest empty band in log₁₀ d. Linear d would put the band among the large injections, 1.19 → 1.38. T1, 2026-10-01: between +0.087 and +0.171, so **0.122** (×1.32 in volume). It also sizes in-run steps: jump against glitch |
| 5 robust σ (§4.3) | Gaussian tail | 0.002 expected false outliers per 3000 trains |
| `d_near` (§4.4) | far-off-droplet trains | measured in T3 |

---

## 6. Stored products and output schema

**Per-run reduction file**, `scratch/run_triage/triage_r{run:04d}_{hash}.nc`:
- Written through `analysis.cache`'s helpers (`_cache_path`, `_load`, `_write`, `_attrs`,
  `_source`): keyed on its inputs and on the DAMNIT file's size and mtime, written atomically,
  rebuilt when stale.
- Unscaled (memory, cache-holds-raw-data-only): flux and transmission sit beside the data and are
  never divided in.

What it holds:
- **Over trainId:**
  - `saxs_band` (band: lowq 0.10–0.20, mid 0.30–0.50, hi 0.80–0.95), the pulse-mean band
    intensities
  - `n_pulses_with_frames`
  - `z` (window: B1, B2, B3, C)
  - `waxs_ring_jf1`, `xgm_uJ`, `total_transmission`
  - `volume_mm3`, `radius_*_px`, `center_*_px`
  - mesh runs: `hexapod_x_mm`, `hexapod_z_mm`, `train_class`, `dist_mm`
- **XPCS** is reduced in one pass over `xpcs_ttcf` to a per-train, per-ROI, per-lag **sum and
  count of the finite diagonal entries**, (trainId, roi, lag). That is all §4.6 needs: g2 over
  any train set (slices, blocks, `on` trains) is Σ sum / Σ count, and φ is the k = 1 count. The
  full matrices are never kept. T2a holds these in memory and writes only the per-run summaries;
  whether to cache the arrays (≈ 20 MB per static run) is decided when the full sweep's cost is
  measured.
- The thresholds and fits are applied in the report step, so a re-derived threshold never needs a
  re-read.

**Report** (T4), `scripts/run_triage_{stamp}.json` plus `.csv` beside it (`_common.write_report`):
- one row per run, holding:
  - every §4 metric
  - `extractable`, `n_slices`, `n_extractable_slices`, `best_snr` (with its ROI and slice),
    `snr_run`, `t_needed_s`, `beta_group`, `beta_borrowed`
  - per plain ROI: `kbar`, `n_pix`, `phi`, `snr_run`, and the noise-check ratio `rho`
  - a per-slice table in the JSON: (run, slice, ROI) → `kbar`, `phi`, `n_usable`, `snr`
  - `flags`, `reasons` (strings naming each metric and value), `series_id`, `background_role`
  - `red_box_rank`, `red_box_kind`, `cum_proc_tb`, `cum_raw_tb`
- the JSON also carries a `series` table, a `thresholds` block and a `pre_tape_extraction` list
- `run_triage_{stamp}.md`: one page of counts per class and level, the series table and the top
  of the ranking

---

## 7. Ranking for the red box

**Eligibility** follows §1's rule. Within a tier the order is the stated key, then run number.
Cumulative TB is summed down the whole list, so the group cuts wherever the quota falls.

| Tier | Contents | Order within the tier |
|---|---|---|
| K1 | `extractable` runs of any class: static, mesh, and silica XPCS standards, which are judged like samples | `n_extractable_slices` descending, then `best_snr` |
| B | runs with a `background_role` (§3.2), plus the mesh background representatives chosen in T3 | by the extractable runs that need them |
| C | calibration outside the SNR rule: LaB6 r497–500 and r52/56 (geometry, open task 4) | flagged for the group, no recommendation |
| — | everything else, including commissioning, test and calib_dark | tape. Their DAMNIT 1D data and flags stay. **Shown with `t_needed_s`**, so the group sees which runs a longer slice would admit |

- **`red_box_kind`** is `proc` by default. It is `raw` only where an open task needs data proc
  lacks: r423/r426 (open task 6) and r480 (open task 16). The user confirmed this "for now",
  2026-10-01.
- **A background run** needs red-box space only for 2D work, because its 1D curve stays in DAMNIT.
  Once the §7.1 extraction exists for it, its kind becomes `none` with the reason
  `background_1d_and_2d_extracted`.
- **Crystallisation (`bragg`, `xpcs_bragg`) and XCCA suitability** are reported per run but never
  move a run between tiers (§1).

### 7.1 Pre-tape extraction

For each mesh attenuator block, and for the static runs at each stray-light era, a background
needs one 2D product that DAMNIT does not hold. It is `analysis.cache.detector_sums` (2D photon
sums, ~4 MB), taken over:
- the `beam_only` trains, and
- the `on` trains

of one representative run. This is what the χ-sector background analysis needs (memory,
saxs-background-subtraction). The cost is measured on one run before the list is executed
(rule 4).

---

## 8. Decisions and open questions

**Decisions:**

| # | Decision | Why |
|---|---|---|
| D1 | DAMNIT is the source; raw only for SSHEX positions and, in T3, pulse-rank XGM | DAMNIT stays after taping |
| D2 | SAXS q in nm⁻¹. XPCS q stored in Å⁻¹ and **converted on output**, unit in the column name | pitfall 19 |
| D3 | Products unscaled, flux beside them | memory, cache-holds-raw-data-only |
| D4 | Classification uses DAMNIT's train-mean XGM. Flux **across runs** (T3) uses `utils.train_pulse_energy(raw, n_pulses=155)` | per-pulse XGM ramp (memory): train means over 350 and 155 pulses are not on one scale |
| D5 | Mesh classes from mid-q SAXS; WAXS is the cross-check | 14–28× against ~2.5× (§2.1) |
| D6 | B3 (0.98–1.05) recorded, never flagged | fires in every run (§2.3) |
| D7 | Thresholds derived when the triage runs, or from a stated requirement (§5) | rule 2 |
| D8 | **`xpcs_ttcf` is read**, not only `xpcs_mean_dataset` | it supplies φ per slice, the β fits, the noise check and `on`-only selection for mesh runs. Measured cost: 28 s per static run in one process (§2) |
| D9 | r1–42 inventory only | user, 2026-10-01 |
| D10 | Keep rule = analytical XPCS SNR ≥ 3 in at least one 30 s slice of a plain ROI; backgrounds are the only exception | user, 2026-10-01; §4.6. It replaced the Γ–q exponent criterion, which the user did not want to rely on |
| D13 | β is measured from the data (median τ → 0 intercept of slow, well-measured runs). The speckle/pixel model β_res is a cross-check | user, 2026-10-01 |
| D14 | The analytical noise carries √2 for the TTCF_off subtraction | measured: 3.9e-4 predicted against 3.8e-4 jackknife on r423 (§2.3) |
| D11 | XCCA and crystallinity are content flags, never ranking criteria | user, 2026-10-01: XCCA is low priority; Bragg kinetics survive in DAMNIT 1D |
| D12 | Dynamics only from the plain ROIs | Bragg-ROI "g2" is crystallite speckle (§2.3), and the 0.072 ROI misbehaves without crystals |

**Open questions:**

| # | Question | Next step |
|---|---|---|
| O1 | `d_near` for beside vs beam_only | measured in T3 |
| O2 | Does the parasitic per flux depend on the attenuator? | T3, across the four mesh blocks |
| O3 | Is the 11 May stray-light pattern the same from r68 to r433? | T3: 2D sums per block (§7.1) against r423 |
| O4 | r97–110: a refilled droplet or a failing fit? | T1 flags it; read a camera image |
| O5 | Locate the beam on the droplet geometrically (beam_cam = b₀ − k·(X, Z)) | optional, T3 |
| O6 | Is scratch retained after the data move? | §6 files regenerate from DAMNIT; §7.1 sums do not, so they go under `usr/` |
| O7 | Sub-linear Γ–q exponents (0.4–0.6, §2.3): physics or method? | outside the triage, which no longer uses them; worth checking with a stretched-exponential model and without the off-subtraction before anyone quotes them |
| O9 | β eligibility: A/σ_A ≥ 10 and 5τ_1 ≤ 1/(2Γ) are design choices | user review; T2 reports how β moves if they are relaxed to 5 and 2 |
| O8 | Does XPCS on a mesh `on` subset suffer from the scan motion (the hexapod moves in ~72 % of trains)? | T3: compare `on` trains with the motor at rest and in motion |

---

## 9. Phases and acceptance

| Phase | Content | Acceptance |
|---|---|---|
| **T0** | this spec | reviewed by the user (done 2026-10-01, revised to §1's rule) |
| **T1** | `triage.samples`, `triage.inventory`, `triage.series`; tests on a mock sqlite and mock DAMNIT files of real shape | every run classified with §3.1's counts and TB (raw total 994.8). Series reproduce r443–452, r464–466, r338–369 (r349 a member without a fit, calibration r353–355 skipped), r370–381 (across the r378→379 pause) and r308–337. §4.2's known cases: r143→144 fresh, and `volume_jump` in r463 and r466. These are checked by the script; the `increase`, `label_suspect` and `underpowered` boundaries are reported for review. **Passed 2026-10-01** |
| **T2** | `triage.reduce`, `triage.classify`, `triage.xpcs`; `scripts/run_triage.py --phase reduce`; first on r114, r120, r130, r318, r370, r400, r410, r423, r425, r426, r450, then all runs | **XPCS:** the all-train g2 matches DAMNIT's `g2` at k ≤ 10 within σ_g2. The noise-check ratio ρ is 1 within the jackknife's ~10 % on the static acceptance runs (r423: 0.97). β(q) is measured with n_eligible ≥ 3 per plain ROI. The per-slice SNR is reported for every acceptance run, with r450 and r400 extractable. **Mesh:** r318's classes match lowq_report (`on` only at Z ≈ 177.6, X 1.54–2.24). **Bragg:** `z_star` flags ≤ 0.1 % of control trains and does flag r425/426, r114 and r130. **Cost:** per run, measured; it sets the sweep's workers (rule 4) and decides the §6 storage question |
| **T3** | background characterisation (§4.4 sub-classes, O1–O3, O8, the §7.1 list) | `d_near` measured; per-block background table; extraction list with measured cost |
| **T4** | `triage.report`; CSV, JSON and markdown | every run has a tier, a kind and reasons; cumulative TB are monotone; tier rules unit-tested on a synthetic table |

**Testing** follows `context/jungfrau-waxs-integrator.md` §8:
- mock inputs of real dtype and shape, including a synthetic TTCF stack with known contrast, Γ
  and Poisson-level noise, so §4.6's jackknife, β and SNR are tested against ground truth
- `uv run pytest tests/triage` needs no data and no Maxwell
- `scripts/run_triage.py` is checked by running it on a node, not by pytest (rule 7)
