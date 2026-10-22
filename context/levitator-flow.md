# s443 (r0443–452, r0447 left out): levitator and flow in the XPCS g2, 2026-10-21

Method: g2 pooled per run over the usable hit trains (`analysis.cache.pool_g2`, `binned=False`, so every lag
1…144 × 443.08 ns, up to 63.8 µs). KWW fit as in `analysis.xpcs.kww.fit_rings` (β fixed at `REFERENCE_BETA`).
Errors come from 20 contiguous train blocks per run. The figures and every number quoted here
(`levitator_numbers.json`) are in `figures/levitator_flow/` (gitignored). The scripts that made them were scratch
(`g2_analysis.py`, `train_spectra.py`, `refine.py`, `ttc.py`, `figures.py` in a session's `/tmp` scratchpad) and
are not kept in the repository, so reproducing this means rewriting them from the method above.

## 1. Periodic recorrelation: present in r0445 and r0446, absent or weak elsewhere

- **r0445 (t/t* 0.61–0.90) and r0446 (0.91–1.21) show recorrelation echoes** at τ = 9.80 ± 0.05, 19.8 ± 0.2 and
  29.9 ± 0.3 µs (r0445), and at 9.77 ± 0.03, 19.9 ± 0.1 and 30.3 ± 0.5 µs (r0446).
  - Period T = 9.84 ± 0.02 µs, so **f = 101.6 ± 0.3 kHz**.
  - The first echo rises 0.13–0.14 β_ref above the KWW fit (fixed-shape fit z ≈ 26 and 36). c2 at 10 µs is
    1.5–1.9× the smooth curve, and there is a dip at T/2.
  - Successive echoes damp by a factor of 0.3–0.5 per period.
  - The sinusoid scan peaks at 90–102 kHz with T² per d.o.f. ≈ 25. The comb test gives z = 13.6 and 15.5 at
    T ≈ 9.4–9.8 µs.
  - The echoes are visible as stripes at t₂ − t₁ = T and 2T in the pooled two-time map of r0446.
- **The echoes are not instrumental.**
  - T is not a whole number of frames (22.2), and k = 2 and 3 fall at 44.8 and 67.5–68 frames.
  - The echoes appear in 2 of 9 runs, although detector and pulse pattern are identical in all 9.
  - g2 is normalised pairwise per train (Σ on / Σ s_a s_b/cnt), so an intensity modulation divides out.
- **Elsewhere** the fixed-shape echo height at T is:

  | Run | Echo height / β_ref |
  |---|---|
  | r0444 | 0.019 ± 0.005 (weak) |
  | r0450 | 0.009 ± 0.003 |
  | r0451 | 0.010 ± 0.004 |
  | r0443, r0448, r0449, r0452 | below 0.01 |

  So in those runs the echo is absent at a 3σ sensitivity of about 0.01 β_ref.
- **q dependence (r0445, r0446).** The echo depth r(T) − r(T/2) is 0.26 → 0.10 β_ref over q = 0.25–0.66 nm⁻¹,
  and the echo width is 1.0–1.7 µs.
  - Small displacements would give a depth that grows as q². The measured depth is nearly flat in q, which points
    to saturation: a periodic relative displacement with qA ≳ 1 already at 0.25 nm⁻¹, so A ≳ 4 nm.
  - From the width, the oscillatory velocity difference is v ≈ 1/(q σ_τ) ≈ 2 mm/s amplitude.
  - Only 15–25 % of β comes back. Either only part of the illuminated volume oscillates, or the motion is not
    strictly periodic.
- **Pulse-axis spectra** (per train, |FFT|² averaged over trains, so insensitive to phase):
  - c2(t₁, t₁ + d) for d = 1, 2, 4 shows no line at f or 2f, at any run (3σ: < 0.002–0.004 absolute modulation).
  - The pooled two-time maps show no phase-locked modulation along t₁.
  - The per-pulse AGIPD intensity carries a line at **193 ± 3 kHz** in every droplet run tested (r318 and r372,
    350 pulses; r394–r495, 155 pulses). It is absent in the XGM. Its origin is not identified: it lies near, but
    apparently below, 2f = 203 kHz. There is a weak bump at about 103–107 kHz.
- **Drive frequency in raw: not recorded.** r0450 raw has no levitator, function-generator or transducer source.
  The only frequency keys are timing-system clocks (`MID_EXP_SYS/TSYS/UTC-2-S4` `resultingFreq*` = 54167) and
  device update rates. The 101.6 kHz comes from the XPCS alone; confirm it from the levitator's settings in the
  logbook or with the operators.

## 2. Γ ∝ qⁿ per run (resolved rings, binned g2; error = the larger of the fit error ×√χ²ᵣ and a 10-block jackknife)

| Run | t/t* | n | Rings | Δv = Γ/q (n = 1) | Γ₀ + V q: V |
|---|---|---|---|---|---|
| r0443 | 0.00–0.30 | 1.99 ± 0.59 | 4 | 4.1 mm/s | (ill-defined) |
| r0444 | 0.30–0.60 | 0.96 ± 0.29 | 8 | 1.2 mm/s | 1.0 ± 0.4 mm/s |
| r0445 | 0.61–0.90 | 0.82 ± 0.80 | 7 | 0.09 mm/s | (ill-defined) |
| r0446 | 0.91–1.21 | 1.66 ± 0.24 | 12 | 0.20 mm/s | 0.36 ± 0.04 mm/s |
| r0448 | 1.52–1.81 | 0.78 ± 0.07 | 12 | 0.23 mm/s | 0.18 ± 0.01 mm/s |
| r0449 | 1.82–2.12 | 0.82 ± 0.08 | 12 | 0.26 mm/s | 0.21 ± 0.02 mm/s |
| r0450 | 2.12–2.42 | 0.65 ± 0.06 | 12 | 0.36 mm/s | 0.24 ± 0.02 mm/s |
| r0451 | 2.43–2.72 | 0.79 ± 0.04 | 12 | 0.33 mm/s | 0.26 ± 0.02 mm/s |
| r0452 | 2.73–3.03 | 0.67 ± 0.10 | 12 | 0.26 mm/s | 0.17 ± 0.03 mm/s |

- **Judgement.** The late runs r0448–452 are 13–26σ from n = 2, so they are not diffusion-dominated.
  - Under CLAUDE.md epistemic boundary 3 they belong on the shear/advection side.
  - n is below 1, and Γ₀ + V q needs Γ₀ = 2–5 × 10⁴ s⁻¹. So the data also call for a contribution that barely
    depends on q, with χ²ᵣ still 6–60, so neither model is complete.
  - Implied Δv across the scattering volume (qΔvτ ~ 1): 0.2–0.4 mm/s late, and 1–4 mm/s early (r0443/444, where
    few rings are resolved).
  - The oscillatory velocity amplitude from the echoes (~2 mm/s) is of the same order or larger.
- **Unreliable rows.**
  - r0446's n = 1.66: the KWW fit does not model the echoes, which distort Γ.
  - r0443: only 4 rings are resolved.
  - r0445: n is unconstrained.
- **Caveat.** β is fixed at β_ref, so contrast lost before the first lag reads as a faster Γ (fit_rings's
  docstring). That bears on n and is the other session's question.

## 3. Two-time maps

- **r0446:** the echo stripes run parallel to the diagonal at t₂ − t₁ ≈ 10 and 20 µs. They appear in the pooled
  map, so they are not phase-locked to the train. Phase-locked structure along t₁ at f is absent: the power
  along t₁ at f is 0.2–2.2× the median, which is noise.
- **r0448, r0450:** no stripes.
- **Two further observations, not levitator:**
  - Every run shows a block of excess correlation among the first ~31 frames (t < 13.7 µs), with an edge at
    frame ~31. It is train-locked, so it is likely the pulse pattern or the detector. Not investigated.
  - In r0450, c2 near the diagonal falls at t₁ ≳ 40 µs. Decorrelation speeds up through the train, which is
    consistent with dose-driven dynamics (boundary 4) and could supply the q-weak Γ₀. Not quantified.

## Figures (in `figures/levitator_flow/`)

- `s443_g2_residuals_full_lag.png`: residual about KWW per run, every lag
- `s443_g2_echo_r0446_r0450.png`: c2/β with fits, echo against no echo
- `s443_g2_periodogram.png`: sinusoid scan and comb test
- `s443_echo_depth_vs_q.png`: echo depth against q
- `s443_gamma_q_exponent.png`: Γ(q), n and Δv against t/t*
- `s443_pulse_axis_spectra.png`: per-pulse intensity (AGIPD, XGM) and c2(t₁, t₁ + 4) spectra
- `s443_ttc_r0446_r0448_r0450.png`: pooled two-time maps and their off-diagonal residual
