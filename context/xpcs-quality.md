# XPCS quality from photon statistics — specification

Context file for CLAUDE.md open task 20 (c). Read `CLAUDE.md`, then `context/run-triage.md`
§4.6 (the keep rule and its analytical SNR) and `context/diagnostics.md` §4 (the fixed β).

**State:** spec only, written 2026-10-13 (session S4), **for the user's review**. Nothing is
implemented. Every number comes from `scripts/xpcs_quality_probe.py` on 20 trains of r0423 and
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
  identical does rule the monochromator out as the cause of the r0423/r0450 difference.
  **If this holds, the fixed β overstates the SNR of early droplets by up to ×3.** r0423's best
  slice would fall from 32 to about 9, still above 3. Phase Q2 measures β_sf against droplet
  volume along a whole series.
- **The weighting matters.** The unweighted estimator reads 0.006–0.016 higher, and twice as
  high at 0.75–0.86 nm⁻¹ (w_cv 0.18–0.37 there, 0.08–0.13 at lower q). Narrow rings
  (Δq = 0.025) with weights reproduce the wide ROIs' β_sf: r0450 0.025–0.050, mean ≈ 0.03 at
  0.35–0.56.
- **The negative-binomial shape fails mildly on r0450.** R3 is 5–12 % above expectation at
  4–7σ in the three lower-q ROIs: too many 3-photon pixels. A ring mixture can lift R3 by
  ~w_cv² ≈ 1 %, not 5–12 %. Candidates: AGIPD photonisation, crystallite speckle (r0450 is a
  late PEG 6K droplet) or pile-up. It does not change β_sf, which uses only the first two
  factorial moments, but it is what the flag is for.
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

## 5. Storage

Measured with gzip level 1, one chunk per train, K = 15, uint32 counts. The per-frame record is
`hist (n, R, K+2)`, `S1`, `S2 (n, R)`, plus trainId and pulseId:

| Granularity | ROIs | Per frame | r0423-size run (465 k frames) | Proposal (1.486 × 10⁸ frames, 459 runs) |
|---|---|---|---|---|
| per frame | 4 XPCS | 50–61 B (320 uncompressed) | 23–28 MB | 7–9 GB |
| per frame | 17 rings | 163–206 B (1308 uncompressed) | 76–96 MB | 24–31 GB |
| per train (frames summed) | 17 rings | — | ≈ 3.9 MB uncompressed | ≈ 0.8 GB uncompressed (617 077 trains) |

- **Per frame keeps every later choice open**: per pulse rank (heating, CLAUDE.md boundary 4),
  per slice, per droplet-volume bin, the frames of a TTCF block.
- **Per train loses the pulse axis.** β_sf per frame is useless on its own (σ ≈ 0.1 at these
  k̄), so per-train sums lose nothing for slices or runs, only for within-train questions.
- Recommendation: **per frame, 17 rings** (24–31 GB, about half the window sums' 53 GB), in
  `usr/cached_files/agipd_photon_stats/r{run:04d}/`
  beside the window sums. Like them it survives the move to tape, and the middata ACL already
  lets DAMNIT write there.

## 6. Where it is computed

**Option W, inside the `agipd_saxs` worker** (recommended), as the per-pixel window sums are
(§15 of the integrator guide):
- **Cost.** `integrate_frame` already holds the valid nonzero pixels (`keep`) and the pixels
  where the frame's mask differs from its cell's base mask (`diff`). The histogram needs only
  `label[keep]` and `label[diff]`, plus each cell's base count per ring, built once. Measured:
  0.15–0.41 ms/frame (EPYC 75F3, both rings and ROIs), against reads of 4.2–10.6 ms (data)
  and 8.2–14.8 ms (mask) per frame over the two nodes: 1–3 % of the read. A standalone sparse
  version that rescans the frame costs 2.5–4.6 ms; the dense one 10–15 ms. All three agree on
  every frame (0 mismatches over 2 × 2945 frames). The
  window sums cost +11.5 % wall (their phase S2, 2026-10-07).
- **It needs no extra read of proc.** The one bulk AGIPD reprocess planned after the merges
  (S1 → S3 → S2 → S4) produces it for every run. **The user has to coordinate that.** The
  histograms must be in the code that reprocess runs, so S4's phase Q1 has to merge first.
- **A separate file with its own hash**: ring edges, pixel → ring sha256, K. The operator
  is checked with `operator_differences`, as above.
  It is never truncated by `overwrite`, like the window sums, and has its own completeness
  check (S1's `is_complete` pattern).
- **A standalone backfill**, `run_photon_stats(cfg)`, like `run_pixel_sums`, for runs the
  reprocess misses. It reads proc, so it works only before taping.

**Option P, a separate pass:** a script over proc per run. It has to read `image.data` and
`image.mask` again: ≈ 12–25 ms/frame, 1.6–3.2 h of CPU per 465 k-frame run, ≈ 14–29
node-hours for the proposal on 36 cores. It is independent of the reprocess's timing and of the
integrator code. Choose it only if the reprocess cannot wait for Q1.

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

## 7. Reading it back (to implement with Q1)

`analysis.diagnostics.photon_stats`, plain functions:

```
read(run) -> xr.Dataset (frame, ring, k) + (frame,) trainId, pulseId, cellId
weights(run) -> w_p per ring, from the window sums; W1, W2 per (cell, ring) from /masks
contrast(ds, by=None | "slice" | "pulse_quarter" | trainIds, rings=XPCS ROI or ring)
    -> β_sf, σ (jackknife over trains), n frames
nb_checks(ds, ...) -> R2, R3, their expectations and Poisson σ
compare_ttcf(run, ...) -> β_sf / A per XPCS ROI, using triage.xpcs.measure_run
```

## 8. Phases and acceptance

| Phase | Content | Acceptance |
|---|---|---|
| **Q0** | this spec | the user's review |
| **Q1** | the worker hook, the file and its hash, `run_photon_stats`, the reader; mock tests with a synthetic negative binomial of known β (and one with injected hot pixels) | β recovered within its σ from the mock; histograms equal the dense computation frame by frame; on r0423 and r0450 (20 trains) the file reproduces this probe's numbers exactly; worker cost ≤ 1 ms/frame on the DAMNIT partition |
| **Q2** | β_sf along one droplet series (r443–452, or r423–426) against V/V₀, per 30 s slice | a trend with droplet volume measured or excluded at stated precision; the user decides whether β per slice replaces `REFERENCE_BETA` in the keep rule |
| **Q3** | β_sf against the TTCF intercept on the reference runs, with the TTCF rebuilt on our pixels | the 0.79–0.94 ratio on r0450 explained: geometry (our ROIs vs extra-speckle's), normalisation, or charge sharing |
| **Q4** | the outlier rule for low-k̄ rings (pixels whose counts are impossible for the ring's k̄), and R3 per run as a flag | r0423 at 0.75–0.86 nm⁻¹ gives a β_sf within 2σ of its neighbours once the rule is applied |

## 9. Decisions for the user

1. **Option W or P** (§6). W needs Q1 merged before the bulk reprocess.
2. **Per frame or per train** (§5), and 4 ROIs or 17 rings.
3. **Whether β_sf may ever replace the fixed β.** The probe hints that β depends on the droplet
   (r0423 against r0450). The user ruled out a fitted contrast per run (2026-10-13). β_sf is
   not fitted to dynamics, but it is per run. Q2 gives the evidence; the rule stays as it is
   until then.
