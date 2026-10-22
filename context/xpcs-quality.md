# XPCS quality from photon statistics — specification

Context file for CLAUDE.md open task 20 (c). Read `CLAUDE.md`, then `context/run-triage.md`
§4.6 (the keep rule and its analytical SNR) and `context/diagnostics.md` §4 (the fixed β).

**State:** spec, written 2026-10-13 (session S4) and **reviewed by the user the same day**
(decisions in §9). Nothing is implemented. Every number comes from `scripts/xpcs_quality_probe.py` on 20 trains of r0423 and
of r0450. There are two reports beside it:
- `xpcs_quality_probe_20261013T153938Z.json`: job 25288430, max-cssb024, Xeon Gold 6226.
- `xpcs_quality_probe_20261013T165909Z.json`: job 25288745, max-wn135, EPYC 75F3. It adds the
  in-worker cost and the ring storage. Its contrast numbers are identical to the first.

---

## 1. Why, and why now

The keep rule rests on the analytical SNR, β · k̄ · √(N_pix · (F − 1) · φ · n) / √2. Two of its
inputs are assumptions that the stored products cannot test:
- **β is one fixed table** (`REFERENCE_BETA`, from r450 and r493–496). Whether every run has
  that contrast is not checked anywhere.
- **The noise term assumes speckle statistics**: negative-binomial photon counts with contrast
  β. Hot pixels, photonisation errors (two photons read as three), charge sharing and crystallite
  speckle all break that, and nothing would show it.

A histogram of photon counts per pixel and frame, P(k), answers both:
- **Single-frame contrast β_sf.** One femtosecond pulse freezes the sample, so the contrast of
  a single frame is the optical contrast, untouched by dynamics. The TTCF intercept A is the
  contrast that survives the first lag (444 ns). β_sf > A means decorrelation faster than τ₁,
  which CLAUDE.md firm boundary 3 says lowers contrast without showing as dynamics.
- **The shape of P(k)** against the negative binomial. Its departures are the artefact flags.
- **k̄ per frame and per pulse rank**, which the frame table already holds through I(q). It is
  listed here only as a cross-check.

**Why now:** the per-pixel counts live only in proc, which goes to tape. The stored products
cannot rebuild P(k):
- the frame table holds q-binned Σc·x and Σc²·x, not Σk(k − 1) per pixel;
- the window sums add 1550 frames per pixel.

Once proc is taped, P(k) can only be extracted again by restoring it.

## 2. Definitions

Per frame f and ROI r, over the frame's valid pixels (the pass's own `frame_bad`: static mask,
seams, `image.mask` bits):

```
n      = number of valid pixels
h(k)   = pixels with k photons, k = 0 … K; h(K+1) = pixels with more than K
S1     = Σ k          S2 = Σ k(k − 1)          exact, whatever K is
```

Post hoc, with w_p each pixel's run-mean count from the per-pixel window sums
(`usr/cached_files/agipd_pixel_sums`, CLAUDE.md task 1 §15), scaled to mean 1 over its ROI:

```
W1 = Σ w_p,  W2 = Σ w_p²      over the frame's valid pixels; the per-cell base mask stands
                              in for the frame's mask (§2.1)
μ_p,f = a_f · w_p             the expected count of pixel p in frame f, a_f = S1 / W1

single-frame contrast, pooled over any set of frames:
    1 + β_sf = Σ_f S2 / Σ_f [(S1² − S1) · W2 / W1²]
```

- **Why this estimator.** E[k(k − 1)] = (1 + β)·μ² holds for a negative binomial and for any
  intensity distribution with that contrast. Pooling S2 and the expected second moment
  separately avoids the bias of averaging ratios. Subtracting S1 removes the bias of estimating
  a_f² from the same frame (Var S1 ≈ S1). Weighting by w_p removes the intensity variation
  across a ring, which would otherwise read as contrast. A pixel with w_p = 0 adds nothing to
  any sum.
- **σ:** jackknife over trains (contiguous blocks over a run).
- **Shape checks.** For a negative binomial with contrast β, exactly and at any k̄:

  ```
  R2 = 2·h(0)·h(2) / h(1)²   = 1 + β
  R3 = 3·h(1)·h(3) / (2·h(2)²) = (1 + 2β) / (1 + β)
  ```

  R3 well above its expectation (Poisson σ from the counts) flags excess multi-photon pixels:
  hot pixels, photonisation, crystallite speckle. R2 pooled over a wide ring also carries the
  intensity variation, so it is an artefact check, not the contrast estimate.

### 2.1 The base mask in W1, W2

The writer cannot compute W1 and W2: the w_p come from the window sums, which the same pass
writes. They are computed afterwards, from the per-cell base masks the pass stores (`/masks`),
plus the static mask. A frame's own flags beyond its cell's are `n_frame_specific`: about 31
pixels per frame over the whole detector (CLAUDE.md, `image.mask` bit 12), fewer within the
rings' ~4.5 × 10⁵ pixels. Those pixels are left out of S1 and S2 but kept in W1 and W2, which
moves 1 + β by about their fraction, ≤ 10⁻⁴.
The probe's σ_β is ≥ 2 × 10⁻³.

## 3. What a few trains show (2026-10-13, 20 trains = 2945 frames per run)

| ROI [nm⁻¹] | run | k̄ | β_sf (weighted) | β, unweighted | TTCF A (triage) | REFERENCE_BETA | R3 / expected |
|---|---|---|---|---|---|---|---|
| 0.225–0.35 | r0450 | 0.061 | 0.043 ± 0.003 | 0.054 | 0.050 | 0.048 | 1.167 ± 0.017 / 1.041 |
| 0.35–0.45 | r0450 | 0.048 | 0.032 ± 0.002 | 0.038 | 0.040 | 0.039 | 1.086 ± 0.014 / 1.031 |
| 0.45–0.56 | r0450 | 0.038 | 0.029 ± 0.002 | 0.035 | 0.031 | 0.029 | 1.095 ± 0.016 / 1.028 |
| 0.75–0.86 | r0450 | 0.011 | 0.033 ± 0.006 | 0.064 | 0.017 | 0.019 | 1.14 ± 0.10 / 1.03 |
| 0.225–0.35 | r0423 | 0.021 | 0.007 ± 0.006 | 0.023 | (fast decay) | 0.048 | 0.96 ± 0.08 / 1.01 |
| 0.35–0.45 | r0423 | 0.016 | 0.011 ± 0.004 | 0.021 | (fast decay) | 0.039 | 1.01 ± 0.07 / 1.01 |
| 0.45–0.56 | r0423 | 0.011 | 0.018 ± 0.004 | 0.034 | (fast decay) | 0.029 | 1.20 ± 0.10 / 1.02 |
| 0.75–0.86 | r0423 | 0.0024 | 0.46 ± 0.11 | 0.58 | (fast decay) | 0.019 | 3.8 ± 1.7 / 1.3 |

What this says, as far as 20 trains can:

- **On a reference run, the single-frame contrast is close to the TTCF intercept.** On r0450,
  β_sf at the three lower-q ROIs is 0.79–0.94 of A. A lower β_sf than A is the opposite of what
  fast decorrelation would give, so either the TTCF is normalised differently (extra-speckle
  builds its own geometry, pitfall 16) or the single frame loses contrast to charge sharing,
  which adds independent counts to one frame but not to a correlation between two. Telling them
  apart is phase Q3.
- **r0423's single-frame contrast is about a third of r0450's** at the same q: 0.007–0.018
  against 0.029–0.043, with the narrow rings agreeing (§4). Same optics (CRL state, run-triage
  §4.6), same attenuator. A candidate, not a finding: r0423 is a fresh droplet, 1.6× thicker
  along the beam than r0450 (chord 2a_x from `droplet_fit`: 1.71 mm against 1.07 mm; r0426
  1.32, r0493/496 1.22). At 9.04 keV and q = 0.4 nm⁻¹ (2θ = 8.7 mrad) the path differences
  are about 40 nm across 1 mm of sample, L·(2θ)²/2, and about 130 nm across a 15 µm beam,
  w·2θ. Both are of the order of the longitudinal coherence λ/(Δλ/λ): ≈ 70 nm for a SASE
  bandwidth of 2 × 10⁻³, ≈ 1 µm behind a Si(111) monochromator. With pink beam, a thicker
  droplet shows less contrast. **Whether the monochromator was in the beam is not settled.**
  Raw r0423 and r0450 carry identical `MID_XTD6_MONO` (TX 0, TY 2, energy changer 9.0925) and
  `MID_OPT_MONO` (TX −15, TY 97, energy changer 8.99999535, target 9.0) settings, but their
  in/out positions are not known here. Ask MID before quoting this. The settings being
  identical does rule the monochromator out as the cause of the r0423/r0450 difference. The
  energy changers report their setpoint, not the beam: the real energy is 9.04 keV (user,
  2026-10-13, from the undulator spectrum and the beamline staff; CLAUDE.md open task 15).
  **If this holds, the fixed β overstates the SNR of early droplets by up to ×3.** r0423's best
  slice would fall from 32 to about 9, still above 3. Phase Q2 measures β_sf against droplet
  volume along a whole series.
- **The weighting matters, by exactly w_cv².** The unweighted estimator reads high by the
  squared spread of mean intensity across the ring: r0450 wide ROIs 0.0112 / 0.0063 / 0.0065
  against w_cv² 0.0112 / 0.0063 / 0.0067; r0423 0.0165 / 0.0104 / 0.0160 against
  0.0172 / 0.0113 / 0.0175. In the narrow rings the same bias is 0.002–0.006 at q ≤ 0.56. So a
  wrong or missing weight costs a wide ROI 20–60 % of β, and a narrow ring 5–15 %. Narrow rings
  (Δq = 0.025) with weights reproduce the wide ROIs' β_sf: r0450 0.025–0.050, mean ≈ 0.03 at
  0.35–0.56.
- **The negative-binomial shape fails mildly on r0450.** R3 is 5–12 % above expectation at
  4–7σ in the three lower-q ROIs: too many 3-photon pixels. A ring mixture can lift R3 by
  ~w_cv² ≈ 1 %, not 5–12 %. Candidates: AGIPD photonisation, crystallite speckle (r0450 is a
  late PEG 6K droplet) or pile-up. It does not change β_sf, which uses only the first two
  factorial moments, but it is what the flag is for. **Corrected by Q1 (§7.1):** a pooled
  histogram also mixes frames of different k̄, which lifts R3 by A3_frame = 1.02–1.04 on a whole
  run. In a wide ROI the rings' different k̄ add to that. So most of the 5–12 % is mixing.
  What survives, measured per frame, is a 2–10 % excess at third order.
- **0.75–0.86 nm⁻¹ on r0423 is dominated by artefacts.** k̄ = 0.0024, and a few pixels give
  k = 5–12 in single frames (13 counts at k = 5, 7 at k = 12). That makes β_sf 0.46 and R3 3.8.
  These are the pixels `max_count` already records per frame. The ROI needs an outlier rule
  (§6, Q1) before its contrast means anything at low k̄.
- **No clear pulse dependence within a train at this precision.** Each quarter of the train
  has σ ≈ 2× the run's. β_sf by quarter on r0450 is 0.036 / 0.052 / 0.042 / 0.042,
  0.030 / 0.031 / 0.031 / 0.033 and 0.032 / 0.031 / 0.029 / 0.022 in the three lower-q ROIs.
  The last ROI's fall in the final quarter is a ~2σ hint of heating within the train (CLAUDE.md
  boundary 4), nothing more. k̄ of the first 20 pulses over the rest is 0.97 (r0450) and 1.005
  (r0423); the spread over pulse ranks is 2.3–2.6 %.
- **K = 15 is enough.** No count exceeded 12 in any ROI; h(K+1) = 0.

## 4. ROIs

**17 narrow rings, Δq = 0.025 nm⁻¹, tiling the four plain XPCS ROIs**:
0.225–0.35 (5 rings), 0.35–0.45 (4), and 0.45–0.56 and 0.75–0.86 (4 each, 0.0275 wide).
- Every sum here is additive, so each XPCS ROI is the exact sum of its rings, and the
  comparison with the TTCF needs no second product.
- **No ring touches a Bragg window**, B1 0.57–0.65 or B2 0.68–0.74 nm⁻¹ (CLAUDE.md firm
  boundary 2). This is asserted in code, as in the probe. The fcc 1.00–1.03 peak of r0426 and
  the instrumental B3 0.98–1.05 lie above the last ring.
- **Pixels are assigned by their centre's q in our geometry** (`build_operator`: the configured
  beam centre and pixel corners, pitfall 15). They are the SAXS pass's pixels, not
  extra-speckle's (its geometry comes from motor encoders, pitfall 16).
- **Rings move when the beam-centre convention does** (open task 4). The file stores the
  pixel → ring map's sha256, so a reprocess under another centre is a new product, never merged
  with the old one. The map is integer labels, so a last-bit difference in a pixel's q between
  hosts moves that pixel only if it lies within ~1e-16 of a ring edge (not checked). The
  completeness check compares the label arrays themselves, not their sha. **The operator's sha256 is
  host-dependent**: `omega` differs by 1e-16 between CPU types (r0423 vs r0450, this probe;
  S1 found 119 of 459 AGIPD files built on the other class). So compatibility with the integrator
  goes through S1's `analysis.saxs.complete.operator_differences` (exact coef/bins/indptr/q,
  omega at rtol 1e-12; branch `claude/s1-damnit-integrity-1548a4`, 7ebeaa8), not through the
  sha.

## 5. Storage — per frame, 17 rings (user, 2026-10-13)

Measured with gzip level 1, one chunk per train, K = 15, uint32 counts. The per-frame record is
`hist (n, R, K+2)`, `S1`, `S2 (n, R)`, plus trainId and pulseId:

| Granularity | ROIs | Per frame | r0423-size run (465 k frames) | Proposal (1.486 × 10⁸ frames, 459 runs) |
|---|---|---|---|---|
| per frame | 4 XPCS | 50–61 B (320 uncompressed) | 23–28 MB | 7–9 GB |
| **per frame** | **17 rings** | **163–206 B (1308 uncompressed)** | **76–96 MB** | **24–31 GB** |
| per train (frames summed) | 17 rings | — | ≈ 3.9 MB uncompressed | ≈ 0.8 GB uncompressed (617 077 trains) |

- **Per frame** (user): it keeps every later grouping open — per pulse rank (heating, CLAUDE.md
  boundary 4), per slice, per droplet-volume bin, per hit train of a mesh scan (§5.1).
- **17 rings rather than 4 ROIs**, because they give the more realistic contrast, which is the
  criterion the user set:
  - every XPCS-ROI number is the exact sum of its rings, so nothing is lost;
  - in a ring the result depends 2–5× less on the per-pixel weights (§3: 0.002–0.006 against
    0.006–0.017);
  - the weights are what is uncertain wherever a run-mean pixel pattern is not the pattern of
    the frames at hand: mesh scans (§5.1), and droplets whose shadow or stray light changes
    along the run (memory, saxs-background-subtraction).

  The cost is 3.4× the storage.
- Location: `usr/cached_files/agipd_photon_stats/r{run:04d}/`, beside the window sums. It
  survives the move to tape, and the middata ACL already allows writes there.

### 5.1 Mesh scans: extract every frame, score only the hits

The extraction is per frame and identical for every run, so mesh scans need nothing special at
that stage. **The diagnostics do**, because most mesh trains miss the droplet. Mid-q SAXS per
flux (0.3–0.5 nm⁻¹, DAMNIT `agipd_saxs` / XGM × transmission, log₁₀, 2026-10-13):

| Run | Off-droplet peak | Hit cluster | Between the two |
|---|---|---|---|
| r120 | −4.3 (725 of 833 trains) | −3.1 to −2.8 (≈ 52, 6 %) | ≈ 56, spread over 1.1 dex |
| r318 | −5.15 (443 of 720) | −3.95 to −3.4, peak −3.8 (≈ 246, 34 %) | ≈ 31 |

The off peak and the hit cluster are about 1.3 dex apart, ≈ 20× (run-triage §2.1: 14–28× on its
probe runs). The trains between
them are partial overlaps of the beam with the droplet's edge. Those are exactly the trains where
surface scattering and refraction change the pixel pattern, so they belong to neither class.

So, for every diagnostic built on these products — β_sf and the shape checks here, the XPCS SNR
(`triage.xpcs.measure_run(select=)`), and the SVD, χ² and XGM scatter of
`context/diagnostics.md`:

```
hits(run) -> trainIds
    mesh run:   train_class == on            (run-triage §4.4: pooled log10(s_t / F_t) per
                                              attenuator block, empty band; `edge` and `off`
                                              excluded; `classification_ambiguous` reported, r307)
    static run: every train except a droplet out of the beam — s_t / F_t below the run's own
                running median by more than the §4.3 cut (5 robust σ)
diagnostic(run) = diagnostic over hits(run) only; n_hits always reported beside it
```

- **The per-pixel weights of a mesh run come from its hit trains only**, never from the whole
  run's window sums. The 10-train windows mix classes, so the weights are built from the windows
  whose ten trains are all hits. Where too few exist, the unweighted estimator in the narrow
  rings is used, and its w_cv² bias (§3) is reported with it.
- The hit classifier is `analysis.triage.classify.hits(run)`, **built 2026-10-14 (phase H)**. It
  is documented in run-triage §4.4 "As built": mesh runs by block, static runs by an out-of-beam
  rule. Every diagnostic takes its `train_ids`.

## 6. Where it is computed — now inside the `agipd_saxs` pass (user, 2026-10-14)

**Changed on 2026-10-14** (user: "minimize duplicate calculations… if one variable calculates
detector sums, another variable should make use of them. Same with statistics"). The
`agipd_saxs` workers now histogram every frame from the read they already make, with
`AgipdSaxsConfig.photon_stats` on by default. The separate pass is kept, as
`run_photon_stats`, for backfills. §7.2 has the design and its measurement. What follows is
the 2026-10-13 record.

The user chose a separate pass. Ideally the histograms would come out of an XPCS worker, beside
the correlations, but those are extra-speckle's, and building them into it is not in reach now.

**Rejected: inside the `agipd_saxs` worker** (kept for the record). `integrate_frame` already
holds the valid nonzero pixels (`keep`) and the pixels whose mask differs from the cell's base
mask (`diff`). The histogram needs only `label[keep]` and `label[diff]`, plus each cell's base
count per ring. Measured at 0.15–0.41 ms/frame (EPYC 75F3), against reads of 4.2–10.6 ms (data)
and 8.2–14.8 ms (mask) per frame over the two nodes: 1–3 % of the read. A standalone sparse
version that rescans the frame costs 2.5–4.6 ms; the dense one 10–15 ms. All three agree on
every frame (0 mismatches over 2 × 2945 frames).

**Chosen: a separate pass**, `analysis.photon_stats` plus a node script, one run per job,
workers over trains:
- **It reads `image.data` and `image.mask`**: ≈ 12–25 ms/frame, 1.6–3.2 h of CPU per 465 k-frame
  run, ≈ 14–29 node-hours for the proposal on 36 cores. The mask is read rather than taken from
  the stored per-cell base masks. The 0.022 % of pixels whose flags vary between trains can carry
  counts, and in S2 one such pixel weighs as much as a speckle (data quality > speed).
- **The histogram is the sparse form** (2.5–4.6 ms/frame), or the in-worker form if the pass
  builds its own per-cell base masks.
- **The pass's own geometry and masks** (`build_operator`, `build_static_bad`, `frame_bad`,
  `cfg.mask_bits`); its compatibility with the integrator is checked with
  `operator_differences`.
- **Its own file and hash**: ring edges, pixel → ring labels, K. Frames are labelled from the
  reader's coordinates (trainId, pulseId, cellId), never by position, so they can be matched to
  the `agipd_saxs` frame table.
- **It must run before the run's proc goes to tape**, and is independent of the bulk reprocess.
- **Bulk extraction only with the user's OK**, after the per-run time and size are measured on a
  few runs.

- **Virtual CXI as the input** (S3, job 25288100, EPYC 9374F, branch
  `claude/s3-speed-tooling-b02c32` commit 90977be, its `context/performance.md` §3).
  - Read one frame per slice, AGIPD data are 10–30 % cheaper than
    `AGIPD1M[...].ndarray(decompress_threads=1)`. A multi-frame slice is ~12× slower
    (45 against 2.6 ms/frame).
  - **A missing source file reads as the fill value without an error**: AGIPD data as all-zero
    counts, which here would be a valid-looking "no photons" histogram, and the mask as
    0xFFFFFFFF. A pass over virtual CXI must check every `virtual_sources()` file exists first;
    EXtra-data refuses in the same situation.
  - ROI reads save nothing, since an AGIPD chunk is a whole module.
  - Pulse IDs follow detector numbering (2, 4, …), not `XrayPulses`' 200, 202, ….
  - Option W reads proc once for the integrator anyway, so none of this applies there.

## 7. As built (Q1, 2026-10-14)

**The pass: `analysis.saxs.photon_stats`**, beside the window sums since it is AGIPD's.
- `PhotonStatsConfig(saxs=AgipdSaxsConfig(...))` with `rings_nm` (`default_rings()`: the 17
  rings, Bragg overlap refused) and `k_max` = 15. Its hash covers what changes a count: rings,
  K, sdd, energy, beam centre, mask bits, seams, min_modules, and the geometry and pixel-mask
  files' sha256. Not `npt`, the method, workers or outputs (pitfall 12). It pickles by name
  (pitfall 20).
- `run_photon_stats(cfg)` builds the plan (`build_plan`, the frame table's rows), the ring
  map (`ring_labels`: `build_operator`, pixel-centre q) and the static mask, then fans blocks
  out (`fan_out`). Workers read `image.data` and `image.mask` (`decompress_threads=1`), check
  labels as the integrator does, and run `frame_stats` (the sparse form).
- The file is `PhotonStatsWriter`, a `FrameTableWriter`: trainId, reader_pulseId, cellId and
  status per frame, plus `n_valid`, `hist (n, 17, 17)`, `s1`, `s2` and `max_k`. `/rings` holds
  the edges, the pixel → ring labels with their sha256, and the operator's sha256. It shares
  the frame table's resume, label checks, ledger and hash refusal.
- Output: `usr/cached_files/agipd_photon_stats/r{run:04d}/agipd_photon_stats.h5`.

**Reading it back: `analysis.diagnostics.photon_stats`**, plain functions.
- `read_photon_stats(path, trains=None)`: OK frames, as (frame, ring, k).
- `run_weights(path, pixel_sums_path=, masks_path=, trains=None)`: W1, W2 per (cell, ring). The
  per-pixel means come from the window sums, every window or only those wholly inside
  `trains` (a mesh run's hits). The cells' base masks are the SAXS pass's `masks.npz`.
- `contrast(ds, weights=None, groups=None)`: β per ring, or per XPCS ROI with
  `groups=roi_groups(ds)`. A group pools S2 and the per-ring expectations, so intensity
  differences between rings cancel as well. σ by jackknife over 50 contiguous blocks of trains.
- `nb_checks(ds, beta)`: R2, R3, their expectations and `r3_z`. The sign means something:
  extra 3-photon pixels read above; hot pixels far in the tail inflate the moment β and read
  below (mock test: |z| > 5 with three 10-photon pixels per frame). **On real runs it is not a
  speckle test** (§7.1): the pooled ratios carry the spread of k̄ between frames and between
  pixels, and over a whole run that alone is worth 15–30σ. Q4 replaces it.
- Still to come (Q3): `compare_ttcf`, β_sf over the TTCF intercept per XPCS ROI.

**Why not extra-speckle's XSVS.** `extra_speckle.xsvs.calc_xsvs_agipd` computes P(k) per
(train, pulse, q-bin), checked 2026-10-14 in the installed source. But:
- `PhotonCalculator` keeps only k ≤ `ph_max` (default 5), and its docstring speaks of 1-, 2- and
  3-photon pixels;
- it counts pixels per train, not per frame;
- it stores no exact Σk(k − 1);
- its q-bins and masks come from its `Setup` (pitfall 16).

It stays a possible cross-check for Q3.

### 7.1 Q1 acceptance (2026-10-14)

`scripts/photon_stats_acceptance.py --runs 423 450 318 --workers 32`: job 25295674 on
max-wn072 (EPYC 7402), commit 1f1dd34 plus Q1. Its JSON is
`photon_stats_acceptance_20261014T064115Z.json`. Then
`scripts/photon_stats_shape_check.py` on the login node, JSON
`photon_stats_shape_check_20261014T064804Z.json`.

**The gates pass.**
- Every frame is OK on all three runs.
- On r0423 and r0450, the probe's 20 trains (evenly spaced, as in §3) reproduce its
  narrow-ring counts exactly in all 17 rings.
- The mock tests pass: β is recovered within its σ, and every histogram equals the dense count
  frame by frame (13 tests).

| Run | Frames | Wall, 32 workers | Frames/s | File | Per frame |
|---|---|---|---|---|---|
| r0423 | 465 000 | 340 s | 1366 | 92.8 MB | 200 B |
| r0450 | 465 000 | 365 s | 1272 | 113.0 MB | 243 B |
| r318 (mesh) | 252 000 | 205 s | 1230 | 36.6 MB | 145 B |

- **Cost.** Per frame and core, the workers spend:
  - 3.2–4.3 ms reading `image.data`;
  - 6.7–6.8 ms reading `image.mask`;
  - 7.3–8.3 ms on the per-frame mask, the data check and the histogram.

  17–19 ms in all, which a separate pass pays because it re-reads both datasets (§6).
- **Projection at these rates.** The proposal's 1.486 × 10⁸ frames cost 22–36 GB and about
  32 h of one node at 32 workers. Not run: a bulk extraction needs the user's go.

**β_sf per XPCS ROI, whole run, weighted** (± jackknife):

| ROI [Å⁻¹] | r0450 | REFERENCE_BETA | r0423 | r318, 259 hits |
|---|---|---|---|---|
| 0.02875 | 0.0425 ± 0.0005 | 0.0478 | 0.0150 ± 0.0006 | 0.40 ± 0.04 |
| 0.04 | 0.0328 ± 0.0005 | 0.0390 | 0.0118 ± 0.0004 | 0.097 ± 0.009 |
| 0.0505 | 0.0267 ± 0.0004 | 0.0295 | 0.0134 ± 0.0006 | 0.51 ± 0.04 |
| 0.0805 | 0.0221 ± 0.0008 | 0.0186 | 0.46 ± 0.07 | 2.6 ± 0.2 |

- **r0450.** The ratio to REFERENCE_BETA is 0.84–0.91 at the three lower-q ROIs and 1.19 at
  0.0805. The 20-train probe agrees (0.043 / 0.032 / 0.029). Explaining it is Q3.
- **r0423** reads 0.35–0.50 of r0450 at the three lower-q ROIs (the probe said about a third).
  Its 0.0805 ROI is dominated by outlier pixels, as the probe found.
- **The "unweighted" column of the acceptance JSON pools per-ring unweighted estimates.** It
  reads only 0.002–0.006 high, the within-ring w_cv². The wide-ROI bias of §3 (0.006–0.017)
  includes the spread between rings, which pooling per ring removes.
- **On r318's hits, β_sf is not speckle contrast.**
  - In rings 0, 1, 3 and 8, where g3 also passes (|z| ≤ 1.2), it is 0.019–0.057
    (σ 0.005–0.017).
  - Rings 2, 5, 10 and 13–16 read 0.37–3.5. Counts reach 28 in ring 10.
  - The weights come from the 20 windows wholly inside the hits. A hit's pixel pattern
    changes from frame to frame (droplet edge, surface scattering, §5.1), and run-mean weights
    cannot follow it.
  - The per-run score (`diagnostics.xpcs.xpcs_score`) is unaffected: it uses k̄ and the fixed β,
    not β_sf (§9, decision 3).

**The shape check, measured** (`photon_stats_shape_check`). A pooled histogram is a mixture.
In the low-k̄ limit its R3 is the negative binomial's times two factors:
- the frame factor A3_frame = ⟨m⟩⟨m³⟩/⟨m²⟩², over frames, with m = S1/W1 (k̄ varies between
  frames: SASE pulse energy, transmission);
- the pixel factor A3_pix = ⟨w⟩⟨w³⟩/⟨w²⟩², over the ring's pixels.

| | r0450 rings 0–12 | r0423 rings 0–12 | r318 hits |
|---|---|---|---|
| A3_frame | 1.022–1.039 | 1.022–1.023 | 1.072–1.099 |
| r3_z against the negative binomial | +15 to +31 | +2.7 to +9.8 | (outlier rings, below) |
| r3_z after A3_frame × A3_pix | +3.6 to +11 | −0.6 to +7.3 | (outlier rings) |
| g3 / expected, per frame | 1.024–1.10 (z 7–21) | 1.02–3.3 (z 1.7–9.5) | (outlier rings) |

- **The frame factor is most of r0450's failure.** A pooled ratio over many frames is
  therefore no speckle test, and `nb_checks` is kept only as a record of the raw ratios.
- **The per-frame check.** S3 = Σ k(k − 1)(k − 2)·h_k is per frame, and it is normalised per
  frame like β:

  ```
  g3 = Σ_f S3 / Σ_f S1(S1 − 1)(S1 − 2)·W3/W1³ = (1 + β)(1 + 2β) for a negative binomial
  ```

  Neither mixing factor touches it.
- **On r0450 it still reads 2.4–10 % high** in rings 0–12. That is an intensity distribution
  heavier-tailed than a gamma at third order: real, small, cause open (photonisation,
  crystallites, frame-specific pixel patterns).
- **A3_pix finds rings where a few pixels carry the run-mean weight.**
  - r0423: ring 11 (1.15), rings 13–16 (1.09–4.9);
  - r318: rings 2 (10), 14 (48), 15 (5.0), 16 (2.1);
  - against 1.002–1.04 elsewhere.

  It catches most of the absurd β_sf (r0423 rings 13, 14, 16; r318 rings 2 and 13–16), but
  not all of them, and not only them:
  - r318's rings 5 and 10 read 0.37 and 2.0 at A3_pix 1.02 and 1.18. Their pixels misbehave
    in some frames, not on average.
  - r0423's ring 11 reads a sane 0.013 at 1.15.

  So the Q4 outlier rule has to work per frame (counts impossible for the frame's k̄), with
  A3_pix as a pointer, not the rule.
- **No overflow:** no count reached K + 1 = 16 on r0423 or r0450.

### 7.2 Inside the `agipd_saxs` pass (user, 2026-10-14)

**Design** (`analysis.saxs.run`, `.worker`, `.photon_stats`, `.complete`):
- **Config.** `AgipdSaxsConfig.photon_stats` (default True) and `photon_stats_root`. Both
  are excluded from the frame table's hash, so no stored hash moved (pitfall 12).
- **Worker.** It gets a `PhotonSpec`: the ring map, the static mask, K and the blocks to
  histogram. In the frame loop it calls `histogram_frame` after the frame's mask and data
  check, so a frame has one status in both files.
- **Output.** `agipd_photon_stats.h5` is the standalone pass's format and hash, so the two
  paths write interchangeable files.
- **Locking and completeness.** The file joins S1's locked outputs and `is_complete`.
  - `photon_stats_reasons`: the frame table's checks, then the ring edges, K and the pixel
    → ring map, label by label.
  - The operator's sha256 sits in `/rings`, not in provenance.
- **Only incomplete products are opened.** A run whose frame table is complete gets its sums
  or statistics with no base masks, no self-test, no `operator.npz`/`masks.npz`, and neither
  the frame table nor the sums reopened.
- **Standalone.** `run_photon_stats` remains for backfills. It is now locked too, and returns
  without writing when its file is complete.

**Acceptance** (`scripts/photon_stats_fused_acceptance.py`, job 25297714, max-exfl473, EPYC
9374F, 32 workers, commit 80a09e1, r0423; JSON
`photon_stats_fused_acceptance_20261014T121159Z.json`). Four passes into a scratch directory:

| Pass | Wall | Per frame and core |
|---|---|---|
| statistics off, cold | 225 s | data 2.42, mask 4.58, integrate 3.11, sums 1.23 ms |
| fused | 261 s | data 2.34, mask 4.37, integrate 3.12, sums 1.22, **statistics 2.33** ms |
| statistics off, warm | 192 s | data 2.37, mask 4.42, integrate 3.14, sums 1.23 ms |
| backfill into the warm run | 211 s | data 2.37, mask 4.46, **statistics 4.20** ms |

- **Every gate passes.**
  - The fused statistics equal the backfilled ones and Q1's standalone file in every column
    and in the ring map.
  - The fused frame table equals the warm one dataset by dataset.
  - The backfill left the frame table, the sums, `operator.npz` and `masks.npz` with the same
    inode, mtime and size.
  - All 465 000 frames are OK.
- **The cost is +36 % wall on a warm r0423** (261 against 192 s), +21 % of worker CPU.
  - The parent's block writing is 80–84 s either way, so it is not the cause.
  - The histogram's 2.33 ms/frame is the sparse `frame_stats`, which rescans each frame.
  - §6 measured an in-worker form at 0.15–0.41 ms that reuses the integrator's nonzero and
    mask-difference indices. It is not built: it changes `integrate_frame`.
- **A backfill costs one read of proc.** About 211 s per 465 k-frame run here, against 340 s
  for the standalone pass on an EPYC 7402 (§7.1; not the same CPU).
- **Preemption.** The first submission (25297694, `allcpu`) was preempted and requeued under
  the same job id, and its stale locks would have refused the restart on another node. S1
  fixed that in `analysis.common.lock` (efd6b0a, `SLURM_RESTART_COUNT`). DAMNIT submits to
  `allcpu` too.

## 8. Phases and acceptance

| Phase | Content | Acceptance |
|---|---|---|
| **Q0** | this spec | reviewed 2026-10-13 (§9) |
| **Q1** | the separate pass (`analysis.saxs.photon_stats`, `scripts/photon_stats_acceptance.py`), its file and hash, the reader (`analysis.diagnostics.photon_stats`); mock tests with a synthetic negative binomial of known β, one with injected hot pixels, and the mock AGIPD run frame by frame | β recovered within its σ from the mock; histograms equal the dense computation frame by frame; on r0423 and r0450 the probe's 20 trains reproduce its numbers exactly; time and size per run measured on r0423, r0450 and one mesh run (r318). **Passed 2026-10-14** (§7.1): all frames OK, 17/17 rings exact on both runs, 340 / 365 / 205 s and 200 / 243 / 145 B per frame |
| **H** | mesh hits: run-triage T2 `classify` (§4.4) and the static out-of-beam rule (§5.1); every diagnostic takes the hit trains | **passed 2026-10-14**: r318 agrees with the water-band rule on 99.6 % of trains; r68–69 and r140 ambiguous, r309–311 all on; r307 reports 46 hits with WAXS z 0.8 (flagged by the WAXS check, not by the classifier); hit fractions per run in the report (r120 11 %, r318 36 %) |
| **Q1b** | the statistics from the `agipd_saxs` pass's own read (§7.2) | **passed 2026-10-14** on r0423: fused = backfill = standalone, frame table unchanged, backfill touches nothing else; +36 % wall |
| **Q2** | β_sf along one droplet series against V/V₀, per 30 s slice, on hits only | a trend with droplet volume measured or excluded at stated precision. A diagnostic only: the keep rule keeps the fixed β (§9) |
| **Q3** | β_sf against the TTCF intercept on the reference runs, with the TTCF rebuilt on our pixels | the 0.79–0.94 ratio on r0450 explained: geometry (our ROIs vs extra-speckle's), normalisation, or charge sharing |
| **Q4** | the outlier rule for low-k̄ rings (pixels whose counts are impossible for the ring's k̄; A3_pix finds the rings, §7.1), and a per-run shape flag from the per-frame third factorial moment g3 (W3 beside W1, W2), replacing the pooled R3 | r0423 at 0.75–0.86 nm⁻¹ gives a β_sf within 2σ of its neighbours once the rule is applied; g3 on r0450 within its σ, or its excess stated |

## 9. Decisions (user, 2026-10-13)

1. **A separate pass**, not the `agipd_saxs` worker (§6). The ideal would be an XPCS worker
   beside the correlations, but those are extra-speckle's. **Reversed 2026-10-14:**
   - the statistics are written by the `agipd_saxs` pass from its own read, on by default;
   - in DAMNIT, one `agipd_preprocessing` variable reads proc, and `agipd_saxs`,
     `photon_statistics` and the diagnostics read its files (§7.2; `context/diagnostics.md`
     §8).
2. **Per frame** (§5). Rings or ROIs was left to whichever gives more realistic results: 17
   rings, for the reason in §5.
3. **The keep rule keeps the fixed β** (`REFERENCE_BETA`). β_sf is a diagnostic; the user is
   satisfied with the visibility result as it stands, and may come back to it.
4. **Mesh scans are not dropped.** Their diagnostics are computed on droplet hits only, which
   needs the hit classifier first (§5.1, phase H).
5. The photon energy is **9.04 keV**. The 9.000 readbacks are setpoints (§3).
