# Run diagnostics (`<pkg>.diagnostics`), XPCS SNR, frame cells and droplet videos

> **Variable names.** Since 2026-10-19 the DAMNIT columns are `Group` classes (`agipd.saxs`, `xpcs.g2`, ...); the flat names used below (`agipd_saxs`, `xpcs_hits_g2`, `xpcs_score`, ...) are the earlier ones. The map is `context/damnit-integrity.md` §8.1; `analysis.damnit_names` reads both.

Context file for CLAUDE.md open task 20 (session S4, 2026-10-13). Read `CLAUDE.md` first;
the XPCS keep rule is `context/run-triage.md` §4.6, the photon-statistics quality check
`context/xpcs-quality.md`.

**State (2026-10-14):**
- Phase A (`analysis.diagnostics.frames`, `.curves`) is done: 22 mock tests, ruff clean,
  smoke-tested on r0423 and r0426.
- Phase B (`analysis.diagnostics.xpcs`, a fixed reference β) is done: §4.
- Phase C (the photon-statistics spec, `context/xpcs-quality.md`) is reviewed.
- Phase H (droplet hits, `triage.classify`; the per-run score with k̄ per ROI) is done: §5.
- Q1 (the photon-statistics pass, `analysis.saxs.photon_stats`, and its reader
  `analysis.diagnostics.photon_stats`) passed 2026-10-14 on r0423, r0450 and r318:
  `context/xpcs-quality.md` §7.1. Its product is
  `usr/cached_files/agipd_photon_stats/r{run:04d}/agipd_photon_stats.h5`, written for those three
  runs only.
- Phase D (`*_frames_used` cells) passed 2026-10-14: §6.
- Phase E (droplet videos, `analysis.camera`) passed 2026-10-14 on r0423, r318 and r0426: §7.
- Phase F (`notebooks/diagnostics.ipynb`) is left to another session (user, 2026-10-14).
- The DAMNIT layout (user, 2026-10-14): one preprocessing variable per detector kind reads
  proc; every other column reads its files (§8). The photon statistics come out of the
  `agipd_saxs` pass itself (`context/xpcs-quality.md` §7.2).

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

`xpcs.xpcs_snr(run)` is `triage.xpcs.measure_run` + `summarise` for one run, with β the fixed
`triage.xpcs.REFERENCE_BETA` table (or a report's β, `load_beta`, for comparison). It returns `summary` (the verdict), `slices` (SNR per
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

**β fixed, 2026-10-13 (user).** "A fitted contrast that is different to each run will be a
pitfall": the β above (median over eligible runs) moved with the run set. It is replaced by one
table from the runs with an almost complete g2: r450, r493, r494, r495 and r496
(`triage.xpcs.REFERENCE_BETA`, measured by job 25288023,
`scripts/run_triage_xpcs_20261013T152014Z.json`).

| ROI [Å⁻¹] | β | A over the five runs |
|---|---|---|
| 0.029 | 0.0478 | 0.0474–0.0499 |
| 0.040 | 0.0390 | 0.0379–0.0400 |
| 0.051 | 0.0295 | 0.0293–0.0306 |
| 0.0805 | 0.0186 | 0.0171–0.0189 |

A/σ_A is 19–68 per run. 1/(2Γ) is 4.5–10.7 µs at 0.029, against a 60 µs window.

The verdicts with the fixed β come from job 25288102 (`scripts/run_triage_xpcs_20261013T152419Z.json`;
the 11 acceptance runs plus r493, r494 and r496; 119 s):
- **The fixed β is reproduced from its own runs exactly** (`reference_beta_reproduced`, all
  agree). The DAMNIT g2 cross-check passes. `passed` is still False on the noise check alone (ρ
  1.5–3.5 on the high-SNR runs, unchanged, since ρ does not depend on β).
- **Every run is extractable, by a wide margin.** The best 30 s slice reaches SNR 19 (r370, a
  600-train run) up to 107 (r441). r423 reaches 32 and r426 25. The 30 s slice at which SNR = 3
  would be reached, `t_needed_s`, is ≤ 0.75 s for all 14 runs.
- **So for static runs the keep rule does not discriminate.** It will separate runs only where
  k̄ or the train count is low: mesh runs (on-trains only), heavily attenuated runs and short
  runs. That is a consequence of the rule the user chose, not something the diagnostics should
  change. The caveat above stands: SNR at τ → 0 is not a resolved decay.

---

## 5. The per-run score over droplet hits (phase H, 2026-10-14)

Every diagnostic of a run is computed over its droplet hits only (user, 2026-10-13):
`analysis.triage.classify.hits(run)["train_ids"]`; method and results in run-triage §4.4
"As built". `diagnostics.xpcs.xpcs_score(run)` combines:
- `hits`: the classification summary (counts, flags, the WAXS check);
- `kbar`: `kbar_rois(run, hits)`, the mean k̄ per XPCS ROI over the hits. It comes from
  extra-speckle's `scratch/xpcs/r{run:04d}/kbar.nc` (k̄ per train, frame and ROI; 453 runs), or
  from our own k̄ when that file is missing;
- `n_rois_kbar_ok` / `n_plain_rois_kbar_ok`: ROIs with mean k̄ ≥ `KBAR_MIN` = 3e-3 photons per pixel
  per frame (user, 2026-10-14; also extra-speckle's default `kbar_threshold`,
  pipeline/pipeline.py:89). The user wrote "ph/pix/s" but defined k̄ "per acquisition"; a frame is
  the acquisition, so the threshold is per frame;
- `snr`: `xpcs_snr(run, select=hits)`.

**extra-speckle's k̄ against ours** (r0423, all trains): 0.02046 / 0.01525 / 0.01101 / 0.00729 /
0.00499 / 0.00370 / 0.00232 against 0.02054 / 0.01533 / 0.01113 / 0.00744 / 0.00511 / 0.00378 /
0.00238. They agree to 0.4–3 % per ROI, so the geometry worry of pitfall 16 does not show at
the ROI-mean level.

| Run | Kind | Hits | ROIs at k̄ ≥ 3e-3 (plain) | Best 30 s SNR over hits |
|---|---|---|---|---|
| r120 | mesh | 95 of 834 | 3 / 7 (1 / 4) | 1.7, not extractable |
| r318 | mesh | 259 of 722 | 4 / 7 (3 / 4) | 10.0, extractable |
| r307 | mesh | 46 of 726 | 0 / 7 | 0.0 |
| r423 | static | 3000 of 3001 | 6 / 7 (3 / 4) | 32.2 |

On r120 the mean k̄ over all trains is 4e-4 in every ROI. Over the hits it is 2.5–5.4e-3, which
is why the score is taken over the hits.

## 6. Frames used per detector (phase D, 2026-10-14)

Three DAMNIT variables, appended to `src/amore/context.py`: `agipd_frames_used`,
`jf1_frames_used`, `jf2_frames_used` (titles "AGIPD / JF1 / JF2 frames used"). Each is an import
plus a call to `analysis.diagnostics.damnit.frames_used(proposal, run_no, detector)`.
- **What it reads.** The pass's frame table at the path its own config names
  (`analysis.saxs.damnit.config_for(...).output_file`, `analysis.waxs.config.config_for(...)`).
  There is no `var#` dependency, so asking for the cell never relaunches the cluster job
  (CLAUDE.md pitfall 26). It is not a cluster variable either: it reads one status column.
- **What it shows.** `Cell(data=used, summary_value="used / total (pct%)")`, where used is the
  number of OK rows and total the number of rows.
  - Green: every frame OK.
  - Red: any WORKER_ERROR or LABEL_MISMATCH, or a file HDF5 cannot read.
  - Amber: anything else (NOT_PROCESSED = incomplete, MISSING_MODULES, NO_FRAMES,
    DATA_CHECK_FAILED), or "busy".
  - When not green, the code that set the colour follows, e.g. `· WORKER_ERROR 1`.
  - `Skip` when the file does not exist.
- **Refreshing.** The cell shows the file as it was when the cell ran. After a reprocess, run
  `damnit reprocess --match "frames used"` (user decision, 2026-10-13). In damnit 0.4.0
  `--match` compares variable **titles**, case-insensitive, not names.
- **"busy" is rare inside DAMNIT.** `import extra_speckle` sets
  `HDF5_USE_FILE_LOCKING=FALSE` for the whole process (`extra_speckle/xcca/tools.py:2`), and the
  context imports it at the top. Inside DAMNIT, then, a file being written reads as it stands
  (amber, NOT_PROCESSED rows), or red if HDF5 trips over it mid-write.
  - The same switch means HDF5 does not refuse a second writer in any DAMNIT job. That is
    pitfall 26's double write, with S1's lock file as the only guard (told to S1 on
    2026-10-14).
  - The mock test of "busy" sets `HDF5_USE_FILE_LOCKING=TRUE` for itself and its writer.
    HDF5 reads the variable at every open, so this works after `extra_speckle` has been
    imported.

**Acceptance.** ctxrunner `exec 10400 <run> proc --var …` on a scratch copy of the context, run
as DAMNIT runs it: `python -m ctxrunner`, with ctxsupport on `PYTHONPATH` and the copy's
directory as cwd, so `analysis_helpers` imports. Results:
- r0423: all three green, 465 000 / 465 000 (AGIPD) and 24 000 / 24 000 (each JF), 0.2–1.3 s
  per cell;
- r0016: AGIPD 1440 / 1440 and JF1 86 064 / 86 064 green; JF2 skipped (no file).
- **Every existing file, read-only** (`frames_used` over every run directory, login node,
  2026-10-14): all green. That is 459 AGIPD, 421 JF1 and 418 JF2 frame tables. The other runs
  have no file and would be skipped. A cold AGIPD file takes about 3 s, because GPFS has to
  fetch the status column of a 2.3 GB file.
- 9 mock tests (`tests/diagnostics/test_diagnostics_damnit.py`) cover the colours, Skip, busy and
  unreadable files, and the paths. `tests/test_context.py` checks that the three have no
  dependency, are not cluster jobs, share "frames used" in their titles, and are an import plus
  a call.

## 7. Droplet videos (phase E, 2026-10-14)

**The camera.** `MID_EXP_SAM/CAM/CAM5:daqOutput`, key `data.image.pixels`: (494, 659) uint16,
one image per train (10 Hz), no gaps on r0016, r0060, r0318 and r0423 (`camera_facts`). Also in
raw: CAM1 (2052 × 2456) and two XTD6 beam viewers; none of them gets a video.

**`analysis.camera`** (no DAMNIT import), adapted from p010646 (`transfer_context/09`):
- `write_camera_video(run, path, label=, binning=1, fps=30, crf=23, workers=8)`;
- `count_frames(path)`, which decodes every frame with the same ffmpeg;
- `first_image(run)`, which reads one image;
- `camera_facts(run)` and `gaps(train_ids)`.

How a video is made:
- **Grey levels** are the 0.5 / 99.5 % percentiles of 8 images spread over the run.
- **Frames** are cropped to even sides (659 → 658) and labelled with Pillow: the run, the train
  ID, and the seconds since the first image.
- **Order.** A bounded thread pool (blocks of 4 images, at most 2 × workers waiting) feeds
  ffmpeg in train order. Each block checks the train IDs the reader returns against the ones it
  asked for (pitfall 4).
- **Encoding:** H.264 with yuv420p, preset veryfast, `+faststart`, at 30 fps (3× real time). The
  MP4 `comment` records the source, trains, image count, gaps, limits, binning, fps, CRF and
  version.
- **Writing.** The file is written under a temporary name in the same directory and renamed
  into place. A second job on the same run (pitfall 26) therefore cannot leave a truncated
  video, and a failed write keeps the old one.
- **ffmpeg** is `imageio-ffmpeg` 0.6.0's static ffmpeg 7.0.2 with libx264, added with `pillow`
  to the `euxfel` extra (uv.lock adds only imageio-ffmpeg). It has no ffprobe.
- **Output:** `usr/cached_files/droplet_videos/r{run:04d}_cam5.mp4`.

**DAMNIT** (`analysis.diagnostics.damnit`; context variables "Droplet video" and "Droplet first
image"):
- `droplet_video` is `cluster=True` and raw. It writes the video, decodes it again, and raises
  if the frame count is not the number of images. It returns `Cell(path, "N frames, G gaps")`,
  and `Skip` without images.
- `droplet_first_image` is raw and not a cluster job. It returns the first image as a (y, x)
  DataArray with its trainId, which DAMNIT shows as a thumbnail, and reads only that image.

**Acceptance** (job 25295822, max-wn074, EPYC 7402, 16 threads; JSON
`scripts/droplet_video_acceptance_20261014T072337Z.json`):

| Run | Images | Decoded (ours / EuXFEL ffprobe) | Size | Per frame | Wall | Limits |
|---|---|---|---|---|---|---|
| r0423 | 3002 | 3002 / 3002 | 1.00 MB | 331 B | 16.6 s | 65–3042 |
| r318 (mesh) | 722 | 722 / 722 | 0.44 MB | 609 B | 3.9 s | 493–3629 |

- **ctxrunner on r0426**, from a scratch copy:
  - `droplet_first_image`: computed in 0.85 s, 310 MB peak RSS for the whole process;
  - `droplet_video` with `--cluster-job`: computed in 14.3 s, 470 MB peak. It stored
    "3002 frames, 0 gaps" and the path; the file is 1.04 MB.
- **Looked at, not only counted.** The first, middle and last frames of r0423 and r318 show the
  droplet's silhouette with a legible label, and r0423's droplet shrinks over 300 s. The files
  are small because the backlit images are smooth (median frame-to-frame change 0.14–0.22 grey
  levels), not because they are blank.
- **A bulk run, if wanted.** The proposal's ~617 000 trains come to about 0.2–0.4 GB and about
  1 h of one node at these rates. `mmlsquota` reports no quota-enabled fileset for p010400, and
  exflu has 480 TB free (2026-10-14). Not run: it needs the user's go, or comes with the
  DAMNIT reprocess once the context is installed.

## 8. The DAMNIT layout: who computes, who reads (user, 2026-10-14)

The user wanted the table to say which variable computes what, and no detector data read
twice. So **two variables read detector proc, and every other column reads the files they
write.** In `src/amore/context.py`:

| Column (title) | Kind | What it does |
|---|---|---|
| `agipd_preprocessing` ("AGIPD preprocessing") | cluster, proc | `analysis.saxs.damnit.preprocess`: one read of AGIPD proc, writing whichever of the frame table, window sums and photon statistics are incomplete. A complete run is answered from its files. The cell is a status per product (green when all are complete and OK); a failure is a red cell, not an exception, so each reader still says what its own file lacks |
| `agipd_saxs` ("AGIPD I(q)") | reader, depends on `agipd_preprocessing` | the per-pulse grid from the frame table; same name and content as before |
| `agipd_iq_overview` | reader | unchanged |
| `photon_statistics` | reader, depends on `agipd_preprocessing`, `agipd_saxs`, `xgm_intensity`, `total_transmission` | `analysis.diagnostics.damnit.photon_statistics`: k̄ and β_sf per plain XPCS ROI and per ring, the pooled P(k), and the reference contrast, over the droplet hits |
| `photon_statistics_overview` | reader | β_sf against q with the reference; P(k) against the negative binomial; k̄ against q with 3e-3. Off-scale rings are marked at the edge with their values |
| `diagnostics_overview` | reader, same inputs as `photon_statistics` | SVD against the noise floor, χ² of each train against the median, the 0.3–0.5 nm⁻¹ band against the flux; over hits |
| `jungfrau_preprocessing` ("JUNGFRAU preprocessing") | cluster, proc | `analysis.waxs.damnit.preprocess`: JF1, then JF2, each answered from its file when complete. Status per detector: OK counts, "dark" (no lit cell, no file), or failed |
| `jungfrau_waxs_jf1` / `_jf2` | readers, depend on `jungfrau_preprocessing` | the grid from the file; skipped without one |
| `jungfrau_waxs_overview` / `_combined` | readers | unchanged, except that their inputs are optional: a dark detector no longer skips them |
| `*_frames_used`, `xpcs_score` | proc, not cluster, no dependency | read stored files: the frame tables; DAMNIT's TTCF and extra-speckle's `kbar.nc` |
| `droplet_video`, `droplet_first_image` | raw | as §7 |

- **Hits inside the cluster job.** `photon_statistics` and `diagnostics_overview` need this
  run's droplet hits. Inside DAMNIT's cluster job, though, this run's `agipd_saxs` reaches the
  per-run file only when the job ends.
  - So they take `agipd_saxs`, `xgm_intensity` and `total_transmission` as dependencies.
    `triage.classify.signals_from` builds the per-train signals from those values with the
    estimator `train_signals` uses. The two agree exactly on r318 and in the tests.
  - `hits(run, signals=...)` uses them for this run. A mesh block's other runs come from their
    files; one not yet in DAMNIT is left out and listed in `summary["block_missing"]`.
  - ctxrunner re-runs raw dependencies such as XGM and transmission inside the proc job, on
    `open_run(data="all")` (damnit 0.4.0, `Pipeline.execute`).
- **`xpcs_score` reads stored values.** It has no `var#` dependency on the XPCS variables,
  because S1's side session has proposed reworking those (`context/xpcs-results-proposal.md`
  on `claude/gracious-wing-15f275`). It therefore shows the run as of its last processing:
  refresh with `--match "XPCS score"`, as with the frames-used cells.
- **Reprocessing.** `scripts/reprocess_safe.py` matches the new titles: "AGIPD
  preprocessing", "AGIPD I(q)", "Photon statistics", "Diagnostics overview", "JUNGFRAU
  preprocessing" and the JF ones. The statistics are a companion it test-reads and never moves.
- **Install.** None of this reaches DAMNIT until the context is installed with S1's
  `scripts/install_context.py`. DAMNIT reads `usr/Shared/amore/context.py` (pitfall 27).

**Checked on real data, outside DAMNIT** (login node, 2026-10-14), with DAMNIT's stored values
standing in for the in-memory ones:
- **`photon_statistics`**:
  - r0423: static, 3000 hit trains, weighted over 301 windows, β per ROI 0.0151 / 0.0118 /
    0.0134 / 0.46, 14 s;
  - r318: mesh, 259 hits, 20 windows, 0.40 / 0.097 / 0.51 / 2.6, 7 s.

  Both equal the Q1 acceptance (`context/xpcs-quality.md` §7.1).
- **`diagnostics_overview`**: 18 s and 12 s.
- **`xpcs_score`**: r0423 "3000 hits · k̄ ok 6/7 (3/4 plain) · SNR 32.2", r318 "259 hits · k̄ ok
  4/7 (3/4 plain) · SNR 10.0", both green, as in §5.

**Acceptance, through DAMNIT's own runner** (job 25297808, max-exfl473, commit 4bcbb55 on S1's
efd6b0a, 2026-10-14). ctxrunner ran on a scratch copy of the context, as DAMNIT runs it, for
r0423 (static) and r318 (mesh). Both runs' products were already complete.
- **The proc stage** (`--var` the frames-used cells and `xpcs_score`) takes 33 s and 19 s.
  - Frames used: r0423 465 000 / 24 000 / 24 000, r318 252 000 / 11 552 / 11 552, all green.
  - XPCS score: r0423 "3000 hits · k̄ ok 6/7 (3/4 plain) · SNR 32.2", r318 "259 hits · k̄ ok 4/7
    (3/4 plain) · SNR 10.0".
- **The cluster stage** (`--cluster-job`, both preprocessing variables, every reader, the
  overviews; XGM and the transmissions re-run as dependencies) takes 59 s and 36 s, with peak
  RSS 4.4 and 2.2 GB.
  - Every variable is computed, and none fails or is skipped.
  - `agipd_preprocessing` answers from the files in 5.3–5.5 s, and `jungfrau_preprocessing`
    in 0.4–0.5 s, writing nothing.
  - The frame tables and window sums keep their 2026-10-07 mtimes.
  - The two Q1 statistics files were finalised again. Their provenance had kept an
    `operator_sha256` the completeness check now reads as a mismatch, and `is_complete` is
    true for both runs afterwards. r0450's file shows the same leftover until its next
    preprocessing.
- **Fixed after this run.** `agipd_preprocessing` was red on both runs, because r0423 has one
  train (r318 two) with no frames. The window sums record them as `NO_FRAMES`, which the
  pass treats as final. The cell now judges the sums by `pixel_sums.FAILED`, as
  `run_agipd_saxs` does, and reads "frames 465000 OK · sums 3000 OK · stats 465000 OK".
