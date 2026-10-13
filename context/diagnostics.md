# Run diagnostics (`<pkg>.diagnostics`), XPCS SNR, frame cells and droplet videos

Context file for CLAUDE.md open task 20 (session S4, 2026-10-13). Read `CLAUDE.md` first;
the XPCS keep rule is `context/run-triage.md` §4.6, the photon-statistics quality check
`context/xpcs-quality.md`.

**State (2026-10-13):**
- Phase A (`analysis.diagnostics.frames`, `.curves`) is done: 22 mock tests, ruff clean,
  smoke-tested on r0423 and r0426.
- Phase B (`analysis.diagnostics.xpcs`, a measured β) is done: §4.
- Phases C–F are to come.

---

## 1. Rules

- **Read only final, small files**: the passes' frame tables
  (`scratch/agipd_saxs/r{run}/agipd_saxs.h5`, `scratch/jungfrau_waxs/r{run}/jungfrau_waxs_jf{1,2}.h5`)
  and DAMNIT's per-run file (`usr/Shared/amore/extracted_data/p10400_r{run}.h5`). They stay
  on disk after raw and proc go to tape (memory, storage-tiers-red-box).
- **Stored data stay raw** (memory, cache-holds-raw-data-only). Every normalisation divides an
  in-memory copy and is named in `attrs["normalised_by"]`.
- **Per-train flux** = DAMNIT `xgm_intensity` (mean over all pulses of the train) ×
  `total_transmission`, joined on trainId (outer: a train missing either is NaN and dropped).
  Never per pulse (memory, saxs-background-subtraction). Valid inside one run only
  (run-triage D4).
- **Plain functions** on xarray/pandas, no classes (transfer_context/01).
- **Units**: `frames.units(path)` reads `units` from `/provenance` attrs, then the root. Absent
  means the detector default: AGIPD `photons`, JUNGFRAU `keV` (every JF file before S2's
  photonising). Per S2 (2026-10-13, provisional until S2 lands): `units` covers
  `signal`/`normalization`/`variance` only; `energy_valid` stays raw keV in photon files, and
  `n_photons` is the per-frame photon total, which `default_scalar_column` prefers.
- **Labels**: rows marked by `FrameTableWriter.mark` (WORKER_ERROR, NOT_PROCESSED) carry no
  trainId/cellId (zero fill). `frame_accounting` books them as `unlabelled`, never as cell 0.

## 2. Functions

| Function | What |
|---|---|
| `frame_accounting(path, by="auto")` | frames per status code; per `cellId` for JF (`by=` any label column). `attrs["trains"]`: the plan's train statuses — a `NO_FRAMES` train owns no rows |
| `frame_scalar(path, column=None)` | a per-frame column on (trainId, cellId/pulseId), OK frames only, placed by stored labels |
| `per_train(path)` | `pooled_per_train` + `units`, `units_recorded`, `detector` |
| `per_train_flux(damnit_file)` | `xgm_uJ`, `total_transmission`, `flux` per train |
| `band_mean(per_train, q_range)` | unweighted mean over the bins in range, per train |
| `svd(curves, k, flux=, q_range=)` | uncentred SVD of curves / flux / run-median curve. `noise_floor` = σ_rms·(√n_trains + √n_q), the Marchenko–Pastur edge for i.i.d. noise (Gavish & Donoho 2014); `n_above_noise` |
| `chi2_vs_median(per_train, flux=, q_range=)` | reduced χ² of each train against the run median with the stored σ; reads high by ≈ π/(2N) |
| `xgm_scatter(signal, flux, detrend_window=51)` | plain and robust relative spread before and after dividing by the flux, their ratio, r; again on detrended series |
| `cell_xgm_correlation(grid, reference, detrend_window=51)` | r per cell against a per-train reference, Fisher-z ±1σ |
| `detrend(x, 51)` | x / running median − 1 |

The detrend window, 51 trains, is the stability window of `context/run-triage.md` §4.3.

## 3. Findings (2026-10-13, r0423 and r0426, stored outputs only)

**The lit-cell cross-check does not transfer from p010646 as stated.** On p010646 lit JF
cells followed the XGM at r ≈ 0.98. Here, JF1 energy totals per cell (`energy_valid`, 3000
trains, all eight lit cells):

| Run | r vs XGM, plain | r vs XGM, detrended | r vs AGIPD train total, detrended |
|---|---|---|---|
| r0423 | −0.20 … −0.22 | 0.20 … 0.26 | 0.78 … 0.86 |
| r0426 | 0.15 … 0.18 | 0.18 … 0.24 | 0.75 … 0.83 |

- **Plain r is set by two slow trends**: the droplet evaporates and the XGM drifts, each on its
  own. Detrending removes them.
- **Both detectors share a train-to-train fluctuation that the train-mean XGM does not see.**
  The JF cells correlate with each other at 0.70–0.92 (detrended, r0423) and with the AGIPD train
  total at ≈ 0.8, but with the XGM at ≈ 0.25.
- **Dividing by the per-train flux does not reduce the train-to-train scatter.** AGIPD train
  total against XGM × transmission: detrended relative spread 7.8 % (r0423) / 13.6 % (r0426),
  flux 4.7 % / 4.9 %, r 0.29 / 0.16, ratio after/before 1.01 / 1.00.
- **Not explained here; candidates:** droplet position or shape jitter in the beam (the path
  length changes train to train); beam pointing through the CRLs and apertures downstream of the
  XGM; the XGM measuring a different pulse population from the detectors' (per-pulse XGM is
  known to be distorted at 2.26 MHz). The droplet-fit centre jitter per train is in DAMNIT and
  is the obvious next test.
- **Consequence for the normalisation rule.** Per-train XGM × transmission remains the right
  *slow* normalisation (pulse-resolved is worse). It does not remove train-to-train scatter on
  these runs, so per-train curves carry ~8–14 % level noise that a detector-internal reference
  (e.g. the WAXS ring or AGIPD high-q) would remove. Which reference to use is the user's call,
  not the diagnostics'.
- **JF cell → pulse.** Against single AGIPD pulses, each JF cell's best match moves through the
  train monotonically (cell 15 → pulse rank 5, cell 0 → 24, …, cell 6 → 144, r0423), but the
  peak is flat (0.70 against 0.69 for neighbours, median 0.54). It shows an ordering, not a
  pulse mapping.

**Frame accounting:** r0423 and r0426 JF1 hold 24 000 OK frames (3000 per lit cell); r0423
AGIPD holds 465 000 OK frames, and the plan has one `NO_FRAMES` train with no rows.

**SVD and χ² on r0423 JF1** (all 200 bins, each train divided by its flux): singular values
776, 18.4, 11.1, 5.9, 4.7 against a noise floor of 1.7, so 35 components lie above noise;
χ² against the median: median 61, p99 483. Both are dominated by the evaporation drift, which is
a real change of shape and level. Read them per slice, or against a running median, before
looking for outliers.

---

## 4. Per-run XPCS SNR (phase B)

`xpcs.xpcs_snr(run, load_beta(report))` is `triage.xpcs.measure_run` + `summarise` for one run,
with β read from a triage XPCS report. It returns `summary` (the verdict), `slices` (SNR per
ROI per 30 s slice) and `rois` (k̄, N_pix, φ, ρ, the fit, the β used). An unknown β gives SNR
NaN and `extractable` None. The same change was made in `triage.xpcs.summarise`, and
`scripts/run_triage.py` now records a run without a TTCF as an error instead of dropping it.

**Why the 2026-10-01 report had β = NaN.**
- The report measured r423 alone; r450 and r400 were absent.
- r423's fits are not β-eligible: A/σ_A is 3.1 (< 10), and 1/(2Γ) = 0.53 µs is less than
  5τ₁ = 2.2 µs.
- So no run fixed β, every SNR was NaN, and `summarise` turned that into
  `extractable: False`.

**β measured 2026-10-13** (`scripts/run_triage_xpcs_20261013T141114Z.json`;
`run_triage.py --phase xpcs` on its 11 acceptance runs; one allcpu node, 8 workers, 283 s,
job 25287690):

| ROI [Å⁻¹] | β (median A) | n eligible | p16–p84 |
|---|---|---|---|
| 0.029 | 0.0228 | 7 | 0.015–0.050 |
| 0.040 | 0.0182 | 8 | 0.009–0.040 |
| 0.051 | 0.0151 | 8 | 0.008–0.030 |
| 0.0805 | 0.0137 | 6 | 0.008–0.019 |

- **Every acceptance run is extractable.** The best 30 s slice reaches SNR 8.9 (r370) to 49.8
  (r441). r423 reaches 15, and 10 of its 10 slices pass.
- **The DAMNIT g2 cross-check passes** (no ROI outside z = 3.8).
- **The report's `passed` is False, on the noise check alone.** ρ (jackknife σ / analytical σ)
  is 0.9–1.05 on the runs with moderate SNR (r423, r425, r426, r370, r410). It is 1.5–3.5 on the
  high-SNR runs: r400, r441, r450, r451, r495 and r398. The likely reading is that at SNR ~100,
  real block-to-block change of g2 along the run is resolved. The contiguous-block jackknife is
  built to show it (it inflates σ rather than hiding drift). That makes the analytical SNR
  optimistic by ρ on exactly the runs that pass by a factor of 10 or more, so no verdict here
  changes. Whether the triage's tolerance should apply only where the SNR is moderate is a
  decision for the user.
- **β is not one optical number.** The eligible intercepts span a factor of ~3 per ROI: late
  PEG 6K runs (r450, r451, r495) give A ≈ 0.05 at 0.029 Å⁻¹; r398 and r400 give 0.018–0.023;
  r425 and r441 give 0.008–0.016. A is β times the fraction of contrast that survives the first
  lag. Decorrelation faster than τ₁ (advection, CLAUDE.md firm boundary 3) lowers A without
  showing as dynamics. So the median is an effective contrast, and the SNR inherits its factor-2
  uncertainty.
- **An SNR pass does not mean a resolved decay.** The criterion is the SNR of one g2 point at
  τ → 0 (user, 2026-10-01). r423 passes, yet its fitted decay at 0.029 Å⁻¹ sits at about one
  lag (0.53 µs), so its dynamics are mostly faster than the window.
