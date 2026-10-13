# Run diagnostics (`<pkg>.diagnostics`), XPCS SNR, frame cells and droplet videos

Context file for CLAUDE.md open task 20 (session S4, 2026-10-13). Read `CLAUDE.md` first;
the XPCS keep rule is `context/run-triage.md` §4.6, the photon-statistics quality check
`context/xpcs-quality.md`.

**State:** phase A (`analysis.diagnostics.frames`, `.curves`) done 2026-10-13: 22 mock
tests, ruff clean, smoke-tested on r0423 and r0426. Phases B–F follow the plan below.

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
