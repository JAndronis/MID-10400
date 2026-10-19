# Hit-train XPCS (`analysis.xpcs`) — design, P0 facts, state

Context file for the replacement of extra-speckle's `xpcs_offline` by a pass that reads only
the droplet-hit trains. Read `CLAUDE.md` first (pitfalls 4, 8, 12, 17, 20, 26, 29, 30, 31), then
`context/xpcs-quality.md` (the histogram product that shares this read of proc) and
`context/diagnostics.md` §5 (how hits are decided).

**State (2026-10-19, branch `claude/xpcs-hit-trains`).** P0 (facts), P1 (config, rings,
kernel, partners), P2 (store, extract, run) and P3 (reader) are implemented and tested on a mock
run: `tests/xpcs`, no test needing Maxwell or data. P4 (node acceptance,
`scripts/xpcs_hits_acceptance.py`) ran on 2026-10-18: §10; its two open gates were closed on
2026-10-19 (G1's g2 and the noisy-ring rule, `max_partner_gap`). P5 (the DAMNIT wiring,
`analysis.xpcs.damnit`, five `xpcs_hits_*` variables in `src/amore/context.py`, `tests/test_context.py`)
is implemented and was run through ctxrunner on r0493: §11. **Not done:** installing the context
(`install_context.py --install`, the user's go), P6 (moving `analysis.triage.xpcs` onto the reader
and retiring `xpcs_pipeline`). The old `xpcs_pipeline` and `xpcs_*` variables are untouched.

---

## 1. Why this shape

- **Hits are known late.** The hit rule needs the whole run's `agipd_saxs` (a running
  percentile over 301 trains; mesh blocks span runs), so the XPCS pass is a *second* read of
  proc, after the first. It therefore takes the train ids as an input (`HitSet`) and knows
  nothing of the classifier, which another session is changing. Only the future `damnit.py`
  imports `analysis.triage.classify`.
- **Proc goes to tape.** What XPCS needs from it is raw photons, which no stored product holds.
  So the pass first *extracts* the hit trains once into a compact store under `usr/cached_files`,
  then derives every statistic from the store. A different hit rule, partner policy, ring
  edge, mask policy or XSVS later costs minutes of CPU, not a restore from tape.
- **Two files, two lifetimes.** The photon store is raw, accumulating and irreplaceable; the
  results file is derived, per hit set, and rebuilt freely.

```
proc ──extract (once per hit train)──▶ agipd_ring_photons.h5 ──merge (minutes)──▶ xpcs_hits.h5 ──XpcsHits──▶ g2, TTCF, diagonals
        reads image.data + image.mask      raw counts + flags over the            exact int32 sums per        every float is made here
        of the 16 modules                  envelope, nothing else                 train and ring
```

## 2. Files

Both live in `usr/cached_files/` (`DEFAULT_XPCS_ROOT`), one directory per run, written under
`analysis.common.lock.output_lock`.

### `agipd_ring_photons/r####/agipd_ring_photons.h5`

One group `trains/t<12-digit id>` per extracted train; the union of whichever trains were
extracted, in any order. A group's `status` attribute is written **last**: a group without one
is what a dead writer leaves and counts as unfinished.

| Dataset | Content |
|---|---|
| `envelope/pixels` | int32 flat detector indices of the stored pixels (951 744 on r0423) |
| `base/{cells,ptr,pixel}` | each memory cell's usual flagged pixels (the **flag base**), as envelope positions |
| `trains/t…/pulse_ids`, `cell_ids` | per frame, from the reader's coordinates (pitfall 4) |
| `trains/t…/frame_ptr`, `pixel_step`, `count` | photon events as a CSR: pixel positions **delta-coded within a frame**, count uint8 |
| `trains/t…/flag_ptr`, `flag_pixel` | each frame's flagged pixels as the symmetric difference from `base[cell]` |
| attrs `status`, `sequence` | `FrameStatus`; the proc sequence file the train sits in |

A train that gave no photons has the attributes and no datasets. The envelope is the annulus
the rings span dilated by `envelope_margin_mm` (5 mm, provisional), **bad pixels kept**: the
static mask, the ring edges and the kernel all act afterwards, so none of them is in the store's
identity. The 5 mm exceeds the 4.0 mm beam-centre disagreement of open task 4, so a later move
of the centre stays inside what was stored.

### `xpcs_hits/r####/xpcs_hits.h5`

Dense, addressed by position in the hit set (`hits/train_id`). Integers only.

| Dataset | Shape |
|---|---|
| `sums/on`, `sums/off` | `(T, R, P, P)` int32, chunked per train |
| `sums/s_on`, `s_off_cur`, `s_off_prev` | `(T, R, P)` int32 |
| `sums/cnt`, `cnt_pair` | `(T, R)` int32 |
| `hits/status`, `partner`, `pair_status`, `has_off`, `sequence` | `(T,)` |
| `pattern/pulse_ids`, `rings/{edges_nm,label}` | the shared pulse pattern and the ring map used |

1.35 MB a train raw at R = 7, P = 155; **0.19 MB after gzip** (§7). Changed inputs (hit set, merge hash, partner table) move the
old file to `xpcs_hits.prev.h5` and rebuild it: it is derived.

## 3. Types and modules (`src/analysis/xpcs/`)

| File | Owns |
|---|---|
| `hits.py` | `HitSet`: sorted unique uint64 ids, `sha256`, `source` (provenance, not hashed). `HitSet.of` validates once; empty is allowed |
| `config.py` | `XpcsConfig` (nests `AgipdSaxsConfig`; pickles by name; `extract_hash`, `merge_hash`), `Kernel`, `MaskPolicy`, `RING_EDGES_NM` |
| `rings.py` | `pixel_map` (the SAXS pass's `build_operator` and `ring_labels_from`, or a masks file; a file's rings are authoritative, so no static mask is applied on top), `Envelope`, `Rings` |
| `kernel.py` | pure: `partners` → `Pairing`/`PairStatus`; `train_sums` → `TrainSums` |
| `store.py` | `TrainPhotons`, `FlagBase`, `RingPhotonStore`, delta coding |
| `extract.py` | the proc worker: `encode_train` (pure), `process_block` |
| `merge.py` | the store worker: `process_block` |
| `result.py` | `ResultWriter`, `result_reasons`, `XpcsHits` (the reader), `read` |
| `run.py` | `run_xpcs_hits`: locks, completeness, the two `fan_out` stages |
| `__init__.py` | `xpcs_hits(cfg, hits)` = `run_xpcs_hits` then `XpcsHits` |

Call chain: `xpcs_hits` → `run_xpcs_hits` → `extract` / `merge` workers → `kernel` / `store`.

## 4. The kernel

Per ring, with `X` the train's frames (`P` rows over the ring's good pixels), `Y` its partner's:

```
on[a,b]  = Σ_p X[a,p] X[b,p]        s_on[a] = Σ_p X[a,p]        cnt = #good pixels
off[a,b] = Σ_p X[a,p] Y[b,p]        s_off_cur, s_off_prev: the same sums over the pair mask
good     = ring ≥ 0  and  not flagged in any frame of this train      (MaskPolicy.ANY_FRAME)
pair     = good(train) and good(partner)
```

Exact integers in int64, checked below 2³¹ and stored as int32, so they pool in any order and
the result cannot depend on worker count, block size or restarts.

**Partner = the previous usable hit within `max_partner_gap` train ids** (`Kernel.CLEAN`;
default **1**, the previous train only, since 2026-10-19 on B4's evidence, §10).

```
partners(ids ascending, ok, sequence, kernel, gap):
    last = None; last_failed = None
    for each hit t:
        if not ok[t]:           NOT_OK; last_failed = t; continue
        CLEAN: if last and t − last ≤ gap                             PAIRED  (partner = last)
               elif last_failed within gap and after last             PARTNER_FAILED
               elif no last                                           FIRST_HIT
               else                                                   GAP_TOO_LARGE
        ES:    PAIRED if last is in the same sequence file else FIRST_HIT
        last = t
```

There is **no "next hit" fallback**: it would count a pair twice and let a later hit change an
earlier train's baseline. A train with no partner has no baseline, which is NaN downstream, never
1. `tests/xpcs/test_xpcs_partners.py` pins that a later hit never changes an earlier train.

**`Kernel.ES`** is extra-speckle's `calc_xpcs.py` loop to the bit, for the equivalence gate: the
chain restarts at each proc sequence file, the first train of a chain is paired with its own mask
(its baseline is its own photon sum, exactly 1 once normalised), and the pair mask `msk·msk_prev`
leaks into the raw sums. `tests/xpcs/test_xpcs_kernel.py` checks `on`, `off`, `s_on`, `s_off`,
`s_cnt` equal to `TTCFCalculator` and `TTCFSparseCalculator` run on the same frames, in the
installed extra-speckle's own code. The test loads `extra_speckle/xpcs/kernel.py` **by path**:
`import extra_speckle` imports the XCCA tools, which switch HDF5 file locking off for the whole
process (pitfall 29).

## 5. Identity, idempotence, failure

Two hashes (pitfall 12), neither covering anything operational (`n_workers`,
`trains_per_block`, `flag_base_trains`, `output_root`, `allow_incomplete`, `overwrite`):

- `extract_hash` = run, **envelope sha256**, `mask_bits`, `min_modules`, detector name. Nothing
  a later analysis chooses is in it. A store extracted under another hash is **refused and never
  replaced** (`StoreMismatch`); `overwrite` does not apply to it.
- `merge_hash` = `extract_hash` + the ring labels' sha256 (ring edges, ring masks file and the
  static mask all act through it) + `kernel` + `mask_policy` + `max_partner_gap`.

Behaviour, each pinned by `tests/xpcs/test_xpcs_run.py` unless noted:

- A complete run is answered from the file; inode, mtime and sha256 of both files are unchanged.
- A larger hit set extracts only the new trains. A pass with nothing to extract opens the
  store read-only, so changing the gap or the kernel rebuilds the sums and leaves the store's
  bytes alone.
- A hit not in the run is a `ValueError` before anything is written. An empty hit set is a
  result (a zero-train file), and creates no store.
- Statuses are per train: `MISSING_MODULES`, `NO_FRAMES` (from the plan), `LABEL_MISMATCH` (frame
  count or ids disagree with the plan, or at merge the pulse pattern differs from the commonest
  one), `DATA_CHECK_FAILED` (non-integer or out-of-0…255 counts, or at merge another frame
  count) are final and counted; `WORKER_ERROR` and `NOT_PROCESSED` are retried. Unfinished
  trains after extraction raise `IncompleteRun` unless `allow_incomplete`.
- A held lock raises `OutputBusy`; both files are locked for the whole pass.
- 1 vs 3 workers, blocks of 4 vs 1: store and results have the same `h5_digest`.

## 6. The reader

`XpcsHits` reads the sums on demand; every float is made here.

- `ttcf(Averaging.POOLED)`: `raw = Σon / Σ(s_a s_b / cnt)` over the usable trains, the baseline
  the same over the trains that have one, symmetrised **after** summing, `raw_paired` over those
  same trains, `c2 = raw_paired − off`. 0/0 is NaN. A frame without photons is a NaN row, where
  extra-speckle gives 0.
- `ttcf(Averaging.ES)`: each train normalised on its own with `norm == 0 → 1`, entries under
  `kbar_min = 3e-3` set to NaN (**no source for 3e-3 has been found**; offered only here, `None`
  disables it), the mean over trains, minus the mean baseline.
  `tests/xpcs/test_xpcs_result.py` compares it with `normalize_ttcf` and `kbar_filter` run on
  extra-speckle's own sums.
- `g2(Averaging.ES)` also applies extra-speckle's **noisy-ring rule** (`ES_NAN_THRESH = 0.025`,
  `nan_thresh=None` disables it): `reduce_ttcf(mask_noisy=True)` leaves out a ring when
  `np.isnan(ring_map).sum() > nan_thresh * ring_map.size` (read from
  `extra_speckle/xpcs/tools.py::_reduce_ttcf_regular`; the pipeline's `mask_noisy` defaults to
  True), where the map is the baseline-subtracted mean TTCF, `P × P` entries, diagonal and both
  triangles counted. `roi_q` then lists the rings kept. The pooled view has no such rule: a
  ring with no photons is NaN and says so.
- `g2()`: lags `1 … P−1−10` (extra-speckle drops the last 10 unless fewer than 20 would remain);
  pooled over each diagonal, so a pair of frames weighs with its photon statistics. `sigma` is
  the jackknife error of `c2` over 50 contiguous blocks of trains (leave-one-block-out of all
  four sums at once), `sigma_g2` that of `g2`; same estimator as
  `analysis.triage.xpcs.jackknife_g2`, which a test confirms. Binning (`binned=True`) is
  extra-speckle's `g2_to_bins(50, 20, 10)` shape but **sums numerators and denominators** before
  dividing, not medians.
- `diagonals()`: an `analysis.triage.xpcs.TtcfDiagonals` of per-train `raw − off` diagonal
  sums and finite counts, for `measure_run`. Only trains with a baseline get a row.
- `coverage()`: where every hit went (`by_status`, `by_pairing`, usable, with a baseline);
  `accounted` is true when no train dropped silently.
- `kbar()`, `per_train()` (extra-speckle's `TTCF_raw`/`TTCF_off` layout, float32; large).
- **Time axis.** τ = frame lag × pulse-id step / `BASE_CLOCK_HZ`, with `BASE_CLOCK_HZ =
  1.3e9 / 288 = 4.5139 MHz`. A DAMNIT `rep_rate` (MHz), if given, must agree within 1 % or the
  reader raises; unevenly spaced pulses raise `NonUniformPattern`.

## 7. P0 facts (r0423 and r0426)

`scripts/xpcs_hits_facts.py` (not unit-tested), 50 trains spread evenly over each run's 3000 OK
trains as a stand-in for the hit set. Jobs 25336955 (r0423, max-wn055) and 25336956 (r0426,
max-wn032), commit a34519d. JSON beside the script:
`scripts/xpcs_hits_facts_20261018T203755Z.json` (r0423), `…T203843Z.json` (r0426).

| Question | Answer |
|---|---|
| **B0.** Do the 16 modules cut the run into the same sequence files? | **Yes**, on both runs: 12 sequences each, every `(first, last, n_trains)` identical across modules. `Kernel.ES` can chain per train. The extract worker takes the minimum sequence over modules and does not check agreement, so a run with a module cut elsewhere would go unnoticed: run the script on it first |
| Frames per train | 155 on all 3000 OK trains; pulse ids 2, 4, …, 310 in all 50 sampled trains; `XrayPulses`: 3002 trains with period 2 base clocks, 155 pulses, constant pattern |
| Pulse clock | `XrayPulses.bunch_repetition_rate` = 4 513 888.9 Hz (1.3 GHz / 288), as extra-speckle's `convert_pulse_to_mhz` (`12·24/1.3e9`). A step of 2 is **443.1 ns**, 2.2569 MHz — DAMNIT's `rep_rate`. The nominal 4.5 MHz / 444.4 ns of the design is 0.3 % off |
| Envelope (margin 5 mm) | 951 744 px; **779 561 good ring pixels** (35 281, 102 220, 155 280, 152 781, 70 091, 111 021, 152 887 per ring) |
| Counts | int16; maximum 81 (r0423) and 71 (r0426) inside the envelope, so uint8 holds them. 1.48 M events a train on r0423 (9 548 a frame), 1.85 M on r0426 |
| `image.mask` | bits `0x3383`. 37 194 px flagged per frame (3.9 % of the envelope); 37 166 of them in the cell's usual set. 25 767 pixels are flagged in every memory cell, 116 688 in some cell: **the cells differ**. 1 211 pixels per cell vary between trains (max 1 356) |
| What ANY_FRAME costs | A pixel flagged in any frame of a train: **76 852 (8.1 % of the envelope), of which 43 528 are good ring pixels (5.6 % of 779 561)** |
| Read cost (one process) | `image.data` 1.12 s, `image.mask` 2.26 s, encode 1.56 s a train on r0423 (1.28 / 2.30 / 1.57 on r0426): ~3.4 core-s to read, ~5 with the encode. Not the ~2 core-s of the design; the page-cache state was not controlled and this is one process, not 36 |

**B1, bytes per hit train** (gzip 4 + shuffle, group per train, file size; r0423 / r0426):

| Candidate | MB per train |
|---|---|
| events, absolute positions | 2.26 / 2.75 |
| **events, delta-coded within a frame** | **1.66 / 2.01** |
| flags, every flagged pixel listed | 3.25 / 3.25 |
| flags, packed bits per frame | 3.05 / 3.05 |
| flags, difference from the train's majority mask | 1.40 / 1.40 |
| **flags, difference from the cell's base (base from other trains)** | **0.0147 / 0.0149** |
| the base itself, once per run | 3.2 |

The cells share their bad pixels with the same cell of other trains, not with the other cells of
their own train, so the per-cell base is 95× smaller than the per-train one. Chosen layout:
delta events + per-cell base = **1.68 MB per hit train on r0423, 2.02 on r0426**, i.e. about
16.8 GB per 10 000 hit trains (20.2 on r0426) plus the base. The results file adds about 0.2 MB a train (§7, smoke test).
The user still sets the budget; DAMNIT's `xpcs_ttcf` holds about 2 GB a run today.

**Smoke test on real data (not acceptance).** Job 25337041 (max-wn143, 8 workers, one train a
block): `xpcs_hits` on 8 consecutive r0423 trains from the train table, starting from nothing.

| | |
|---|---|
| wall, including worker start, flag base, extract and merge | 23.6 s |
| coverage | 8 usable, 7 PAIRED, 1 FIRST_HIT |
| photon store | 17.3 MB: 3.2 MB flag base, ~1.5 MB envelope, ~1.6 MB a train |
| results | 1.53 MB (0.19 MB a train, gzip of the 1.35 MB raw) |
| `train_sums` in the workers | 3.3 core-s for 8 trains, 0.41 s a train |
| complete rerun | 0.68 s, answered from the file |
| first lag | 443.08 ns; `rep_rate_mhz=2.26` accepted |

It shows the pass runs on real frames at real size and that the sums are cheap next to the read.
It checks nothing against extra-speckle: that is P4.

## 8. Differences from the approved design, and why

1. **Flag base per memory cell, not per train** (design: "differs from the cell/train base").
   B1 above decides it. The base comes from `flag_base_trains` (5) evenly spread trains at store
   creation and is not part of the store's identity (any base is lossless).
2. **Events are delta-coded** (B1: 26 % smaller).
3. **The static mask is in `merge_hash`, not `extract_hash`.** The envelope keeps statically bad
   pixels, so the mask changes no stored photon; putting it in the extract hash would refuse a
   store that proc cannot refill. The envelope sha256 replaces "geometry/operator sha" there
   (the operator sha differs by node, pitfall 30), and replaces the ring edges and margin.
4. **Pooling options are reader arguments**, not config and not in `merge_hash`: they change no
   stored number.
5. **Pulse clock 1.3 GHz / 288, not 4.5 MHz** (amendment 4): see §7. The `rep_rate` check is kept.
6. `PairStatus.NOT_OK` is a fifth member, for a train that is not paired because it has no
   photons. `MaskPolicy` has `ANY_FRAME` (as extra-speckle) and `STATIC_ONLY` (a diagnostic).
7. `XpcsConfig` nests `AgipdSaxsConfig` and exposes `proposal`/`run` from it (no `run_no`), and
   refuses `min_modules != 16`: a missing module reads as zero photons with no flag.
8. `run_xpcs_hits` returns the results path and `xpcs_hits` wraps it with the reader.
9. The results file is dense by hit position (design: per train), so a block writes rows and
   a resume is a status column.
10. Test modules are named `test_xpcs_*.py`: the test directories are not packages, so basenames
    must be unique across `tests/`.
11. **`max_partner_gap` defaults to 1** (user, 2026-10-19; design said 10, provisional until B4).
    A sparse hit set (a mesh run) loses the baseline of every train whose previous hit is not the
    previous train; that is a NaN baseline, never a 1 (§4).
12. **`Averaging.ES` g2 applies the noisy-ring rule** (§6).
13. **P5 is one reader of proc more than the SAXS design** (§11): the hit rule needs the whole
    run's `agipd_saxs`, so `xpcs_hits_preprocessing` depends on it and is a second read.

## 9. Open

See §10 for what P4 settled and left open.

- **P5**: done in the worktree (§11); open is `install_context.py --install`, the user's go.
- **P6**: `triage.xpcs.measure_run`, `inventory` and `xpcs_score` onto `read()`. Until then
  `analysis.triage.xpcs.ROI_EDGES_A` and `RING_EDGES_NM` are two lists; a test pins them equal.

## 10. P4, node acceptance (2026-10-18)

`scripts/xpcs_hits_acceptance.py` (not unit-tested; one JSON per run beside it). Jobs 25337632
(preempted and requeued, rerun as 25337789: hits, main, G1, G2), 25337825 (G1 to A1 on r0423),
25337869 (G1 rerun after fixing the script's ring mapping), 25337826 to 25337829 (r0493 to r0496).
Everything is under `scratch/xpcs_hits_acceptance/r####/`; **nothing was written to
`usr/cached_files`** (the store's identity is its envelope hash and a store is never replaced, so
a provisional 5 mm margin does not belong in the product directory yet). extra-speckle ran in a
separate process (`es-child`), pitfall 29.

**Hit sets are provisional.** `analysis.triage.classify.hits` as committed in this worktree.
r0493 to r0496: DAMNIT's signals (2994, 2934, 2955, 2972 hits; the first 400 are used for
r0494 to r0496, all 2994 for r0493). **r0423 has no usable signals**: DAMNIT's file has no
`agipd_saxs` (reprocess under way) and `scratch/run_triage/signals_r0423_c42659e9af48.nc`
(created 2026-10-18 21:04 UTC) holds 200 train ids 2637862276 to 2637862475, none of them trains
of r0423 (2637695397 to 2637698397). So r0423 uses 200 consecutive OK trains from the frame table
(`agipd_saxs_before_reprocess_20261014T233033Z.h5`) as a stand-in: G1, G2, G4 to G6, A1 do not
depend on what a hit is. **r0450 has no signals cache and no DAMNIT `agipd_saxs`: skipped.**
The cache for r0423 is wrong and should be rebuilt by whoever owns it.

| Gate | Result |
|---|---|
| **G1** Kernel.ES + Averaging.ES on extra-speckle's ring masks (`all_masks.npy`), r0423, 200 trains | **Pass, with a g2 gate defined honestly (below).** Same train ids, pair status as expected (199 PAIRED, 1 FIRST_HIT), NaN patterns identical. Max elementwise relative difference: `TTCF_raw` 2.7e-7, `TTCF_off` 2.5e-7, `kbar` 5.9e-8, `TTCF` (mean raw) 8.9e-7, first lag 443.08 ns on both. `g2` (c2, binned, the six rings extra-speckle kept): max absolute difference 6.7e-8, max elementwise relative 5.7e-4 (4.8e-6 of the largest value); the literal 1e-6 relative gate cannot be met by any code that is not extra-speckle's float32. §10.1 |
| G1, noisy rings | extra-speckle's `reduce_ttcf(mask_noisy=True)` drops a ring whose TTCF has more than 2.5 % NaN (ring 6, 0.075 to 0.086 Å⁻¹ here). **Since 2026-10-19 `XpcsHits.g2(Averaging.ES)` applies the same rule** and returns the same six rings on r0423 (checked on the saved results file: `roi_q` 0 to 5) |
| **G2** ring pixels, extra-speckle against ours | 783 124 against 779 561 ring pixels, **28 151 differ, 0 unexplained**: rebuilt from extra-speckle's parameters, the rings are reproduced exactly (0 pixels, max dq 9e-8 nm⁻¹). Photon energy 9.000 against 9.04 keV: 27 656 pixels. sdd 7.5315962 against 7.531 m: 495. 12.3984 against pyFAI's hc: 11 pixels move at that step and none of the final differences is left to it. Beam centre (rounded), open against half-open edges, the pixel mask file (identical), ASIC seams (already inside the pixel mask in every ring): 0. DAMNIT's real call passes `geom=None` (encoder geometry): its rings differ from the explicit `geom_latest.geom` ones in 689 pixels (max dq 3.2e-4 nm⁻¹) and from ours in 27 472 |
| **G3** CLEAN against extra-speckle c2 at lags 1, 10, 100 per ring | r0423, identical rings, against the real `xpcs_offline`: rings 0 to 5 within 0.6 sigma of CLEAN; ring 6 (0.75 to 0.86 nm⁻¹) differs by up to 6.7 sigma (extra-speckle c2 0.03 to 0.07 against about 0 for CLEAN; extra-speckle drops this ring itself as noisy). Not the chain's first train: leaving it out changes nothing. r0493 to r0496 (own geometry, against Kernel.ES on the same store): rms z 0.56, 0.42, 0.52, 0.60, **max |z| 1.0 to 1.5**, no row above 2 sigma |
| **G4** workers and blocks, 96 trains of r0423 | **Pass.** 36 and 8 workers, blocks of 8, 1 and 3: the sha256 of `sums/*`, of `hits/ pattern/ rings/` and of the store's `trains/*` are identical in all four |
| **G5** rerun | **Pass** on r0423, r0493 to r0496: inode, mtime and sha256 of store and results unchanged; 0.5 to 0.7 s |
| **G6** | extract 14.4 trains/s at 36 workers on r0493 (0.0695 s a train, 2994 trains); 0.43 to 0.49 s a train on 200 to 400 trains, where 25 to 50 blocks cannot keep 36 workers busy. Worker core-seconds a train: 0.32 read data, 0.57 read mask, 0.65 encode on r0493 (r0423: 0.74, 1.4, 1.85). **The parent's serial block writes took 167 s of r0493's 208 s extraction** (56 ms a train): the bottleneck, not the reads. **Fixed 2026-10-19 by block files merged with H5Ocopy (§10.2): 280 to 238 s paired on max-cfel029.** Kernel 244 core-ms a train on r0493 (537 on r0423 and 410 in the smoke test), 65 s for 2994 trains at 36 workers. Store 2.29 MB a train on r0493 (1.79 on r0423), results 0.23 MB: **25.2 GB per 10 000 hits** (20.0 on r0423). `g2()` over 2994 trains 9.3 s (pooled, binned or not), `ttcf` 8.4 s |
| **B4** partner gap (pooled baseline against gap 1, jackknife over 50 blocks, rms z over ring and lags 1 to 60) | r0493 (1150 trains): gap 2 **2.2**, 5 3.3, 10 3.3, 50 3.9; the mean baseline shift at gap 2 is 1.1e-3, as large as c2 itself (about 3e-3). r0494 to r0496 (350 trains): gap 2 1.06 to 1.15, 5 1.3 to 1.9, 10 1.6 to 2.1, 50 2.6 to 3.3. r0423 stand-in (150 trains): about 1.0 at every gap to 50. **Recommended `max_partner_gap` = 1** (the previous train only); 10, the current default, is not supported: the baseline moves by more than its error on every PEG 6K reference run |
| **E3** our g2 against `analysis.triage.xpcs` on DAMNIT's `xpcs_ttcf`, same trains | max |z| over lags 1 to 10 below 1.0 for every ring of r0493 (400 or 2994 shared trains), below 0.8 for r0494 to r0496, 0.3 on r0423 (200); ring 6 of r0423 1.6. rms z over all lags 0.12 to 0.83. The two estimators and the geometries differ (G2), and agree within their errors |
| **A1** store against `agipd_photon_stats` (r0423 and r0493, 100 trains, 15 500 frames each) | **Exact**: s1 and n_valid equal in every frame and ring (max difference 0), on the file's own 17 ring labels. The store's envelope holds every ring pixel |

JSONs: `scripts/xpcs_hits_acceptance_r0423_20261018T220056Z.json` (G2 to A1; its `g1` is
superseded), `…T220440Z.json` (G1), `…r0493_20261018T220641Z.json`, `…r0494…`, `…r0495…`,
`…r0496…`. The first r0423 report (hits, main, G1, G2) was overwritten by a rerun; its main stage
is `scratch/xpcs_hits_acceptance/r0423/main.json`.

Changes made for acceptance: the extract worker now marks a train whose modules sit in different
proc sequence files `LABEL_MISMATCH` instead of chaining it under the minimum (test first).

**Open problems after P4.**

- ~~`Averaging.ES` g2 does not apply extra-speckle's noisy-ring rule~~ closed 2026-10-19 (§6).
- ~~The parent's serial writes dominate extraction~~ closed 2026-10-19 (§10.2): workers write
  block files, the parent only merges them. What limits extraction now is the workers' read and
  encode time, which on max-cfel029 is 1.8 times what it was on max-m004 (same code).
- ~~`max_partner_gap` is still 10~~ set to 1 by the user on 2026-10-19. A sparse hit set (mesh
  runs) loses the baseline of most trains; whether a mesh run needs its own policy is open.
- ~~The G1 g2 float32 explanation is unproven~~ proved on 2026-10-19 (§10.1).
- The 5 mm envelope margin, the flag-base trains, r0423's hit set and r0450's missing signals.
- `Kernel.ES` still assumes the 16 modules cut sequences identically; the extract worker now
  refuses a train where they do not, but a run is not checked up front.

### 10.2 Block files: workers compress, the parent merges (2026-10-19)

The 167 s of serial `put` in G6 was gzip level 4 on seven datasets a train plus a flush per train
in the one process allowed to write. `scripts/bench_store_write.py` timed the ways out on 300 real
r0493 trains; the user's rule was to merge when that is close to writing directly. Arm B won on that
rule: workers write one temporary block file per block (`write_block_file`, the store's own layout,
gzip 4, written as `.tmp` and `os.replace`d), and the parent copies finished groups into the store
with `Group.copy` (H5Ocopy), which does not recompress: 5.5 ms a train in the parent against 76 ms
serial, identical stored bytes.

**Behaviour.**

- The parent stays the only writer to `agipd_ring_photons.h5`. Block files live in
  `agipd_ring_photons.blocks/` beside the store and are created, merged and removed under the
  store's output lock.
- `RingPhotonStore.merge_block` copies a group without attributes, checks the copy against its
  source (dataset names, shapes, dtypes, stored bytes), and only then sets `sequence` and `status`,
  so a group that carries a status is complete and a crash leaves it unfinished. A final group is
  skipped and an unfinished one replaced, as `put` does, so replaying a block writes nothing.
  The block file is deleted only after that.
- `recover_blocks` runs at the start of an extraction: `.tmp` files are removed, finished leftover
  block files (a worker finished, the parent died first) are merged, and one that cannot be read or
  carries another extract hash is renamed `.bad` and logged. The directory is removed when empty.
- A block that fails in a worker writes no block file; the parent marks its trains `WORKER_ERROR`
  as before and the next run retries them.
- `ExtractResult` carries the block file path in place of pickled photons. `SCHEMA`, `GZIP` and the
  extract hash are unchanged, so existing stores stay valid.

**Measured** (r0493, 2994 hits from nothing into a scratch store, 36 workers, max-cfel029; the
real store was only read). Paired in one job, previous tree (`git archive` of 401c5d1) first:

| | serial `put` | block files |
|---|---|---|
| extract stage wall (rings, plan, flag base, fan-out) | 279.9 s | **238.0 s** |
| parent seconds writing | 219.7 | 21.3 (7 ms a train) |
| worker core-s: read data / read mask / encode | 1139 / 2029 / 3464 | 1138 / 2028 / 3448 |
| worker core-s writing block files | 0 | 224 (0.075 s a train) |
| store bytes | 6 857 440 216 | 6 855 525 989 |

Every one of the 2994 trains of the new store equals the real store's, field by field
(`TrainPhotons` arrays and dtypes, attributes, stored bytes per dataset, flag base, envelope), in
both arms. The first block-file run alone (job 25338246) gave 241.0 s. The 208 s of G6 was on
max-m004, whose workers took 955 / 1703 / 1955 core-s; on max-cfel029 the same code takes
1139 / 2029 / 3464, so against its own node the change is 280 to 238 s, not 208 to 238. The time
now left is the workers: 6838 core-s over 36 workers is 190 s, plus about 13 s of setup and a
spin-up of about 50 s before the first block lands. The parent's merge runs beside them and is
not on the critical path. **Why encode is 1.8 times slower on max-cfel029 is not established**
(a node effect, since both arms show it).

**Merge at real size.** `merge_block` of 240 new trains into a copy of the real store (6.86 GB,
2994 groups) took 5.77 ms a train, into an empty store 5.85 ms (ratio 0.99): the cost does not grow
with the file. A replay of the same blocks wrote nothing in 0.23 ms a train at either size. The
copies read back through `get`.

JSONs: `scripts/bench_extract_blocks_r0493_20261019T002300Z.json` (block files, then the scale
test), `scripts/bench_extract_blocks_paired_r0493.json` (paired). Tests: `tests/xpcs/test_xpcs_store_blocks.py`
and three at the end of `tests/xpcs/test_xpcs_run.py`.

**Not verified.** G1 and G2 were not rerun: they test the merge stage, which reads the store through
`get`, and `get` returned identical arrays. The read-back before a block file is deleted compares
metadata and stored sizes, not decompressed content. A kill (not an exception) of the parent
mid-merge was simulated in tests, never on a node. Only gzip 4 was merged; the block-file
directory on GPFS has not been tried with preempted and requeued jobs.

### 10.1 G1's g2: extra-speckle's float32, proved (2026-10-19)

`scripts/xpcs_g1_float32.py` (read-only, login node, no extra-speckle code run; JSON
`scripts/xpcs_g1_float32_r0423.json`) takes the files the r0423 acceptance left in
`scratch/xpcs_hits_acceptance/r0423/` and computes the same six-ring, 30-bin g2 four ways. Its
re-implementation of `_reduce_ttcf_regular` and `g2_to_bins` first reproduces extra-speckle's saved
`g2_c2.nc` **exactly** (180 values, difference 0.0), so the diagonal index (`np.diagonal(.., offset=k)`
for k = 1 to 144), `skip_last = 10`, the `nanmean` order and the log-bin medians are the same
conventions as ours. Then, on the same 200 trains:

| Comparison (180 values) | max abs | max rel elementwise | max rel to largest |
|---|---|---|---|
| extra-speckle float32 (stored) against extra-speckle's own per-train files averaged and subtracted in **float64** | **6.73e-8** | 5.6e-4 | 4.8e-6 |
| ours (`Averaging.ES`) against the stored float32 | **6.72e-8** | 5.7e-4 | 4.8e-6 |
| ours against extra-speckle's float64 | **2.1e-9** | 3.1e-5 | 1.5e-7 |
| ours, per-train inputs rounded to float32 first, against float64 | 2.3e-9 | 2.8e-5 | 1.7e-7 |

Reading. extra-speckle's float32 g2 differs from its **own** float64 value by exactly as much as
ours differs from it (6.7e-8 either way, 0.56 float32 epsilon), and ours sits **30 times closer to
the float64 value** (2.1e-9). The cause is the float32 means: `raw` and `off` are means of 200 values
near 1 held in float32 (their own float32 mean against float64: 8.9e-7 relative, 1.5e-4 absolute),
and c2 is their difference, 0.014 at most and 2e-5 at least, so a rounding of 6e-8 on each is 5.6e-4
of the smallest c2. Relative error is the wrong measure for a difference of two numbers near 1. The
remaining 2e-9 is extra-speckle's per-train float32 arithmetic (the `TTCF_raw` entries differ from
ours by up to 2.7e-7), averaged over 200 trains; rounding our per-train values to float32 leaves it
unchanged (2.3e-9), so it is not our storage. No discrepancy in our code was found.

**The honest gate for g2** (`G2_ABS_TOLERANCE` in `scripts/xpcs_hits_acceptance.py`): maximum
absolute difference at most 2 float32 epsilon (2.4e-7) against extra-speckle's stored g2, the same
rings kept, NaN pattern identical. Measured 6.7e-8. The 1e-6 elementwise-relative gate stays for
`TTCF_raw`, `TTCF_off`, `kbar` and the mean TTCF, which are not differences of near-equal numbers.
It does not test ours against extra-speckle's float64, because extra-speckle never computes one;
the table above is the evidence for that.

## 11. P5, the DAMNIT wiring (2026-10-19)

**Design (approved).** One new proc reader, `xpcs_hits_preprocessing`, and readers of its file.

| Variable | Title | Depends on | Reads |
|---|---|---|---|
| `xpcs_hits_preprocessing` | XPCS hits preprocessing | `agipd_saxs`, `xgm_intensity`, `total_transmission` | **proc** (cluster) |
| `xpcs_hits_g2` | XPCS hits g2 | `xpcs_hits_preprocessing`, `rep_rate` (optional) | results file (cluster, as `agipd_saxs`) |
| `xpcs_hits_ttcf` | XPCS hits TTCF | `xpcs_hits_preprocessing` | results file |
| `xpcs_hits_overview` | XPCS hits overview | `xpcs_hits_g2`, `xpcs_hits_ttcf` | the two values |
| `xpcs_hits_frames_used` | XPCS hits frames used | nothing (pitfall 26) | results file; its title shares "frames used" with the other three |

- `analysis.xpcs.damnit` is the **only** importer of `analysis.triage.classify` (a test walks the
  package and pins it). It resolves the `HitSet` with
  `hits(run, proposal=, signals=signals_from(saxs, xgm, transmission))` as `photon_statistics` does,
  because inside the cluster job this run's `agipd_saxs` is not in the per-run file until the job
  ends. It builds `XpcsConfig(saxs=analysis.saxs.damnit.config_for(p, r))`, so the SAXS defaults
  and the XPCS defaults (`max_partner_gap` 1, `ring_edges_nm`, 5 mm envelope), and `output_root`
  is the default `usr/cached_files`.
- **A cell, not an exception, for every outcome:** green (every hit usable), amber (some hit has a
  final non-OK status, or the output is busy), red (the hit rule or the pass failed, or a hit is
  `WORKER_ERROR`, `LABEL_MISMATCH` or unfinished), grey (`no hits`: an empty hit set is a result,
  the pass still runs and writes its zero-train file, and the readers `Skip` it).
- The readers refuse a results file with an unfinished hit (`IncompleteRun`), as `agipd_saxs` does.
- `rep_rate` (MHz) goes to `XpcsHits(path, rep_rate_mhz)`: a pulse pattern implying another rate
  refuses to make a lag axis (`RepRateMismatch`, 1 %). The pass itself never uses it.
- The wrappers are a docstring, one `ImportFrom` and a `Return`; no keyword argument, no
  `overwrite`, no module global, no `from __future__ import annotations`.
- `tests/test_context.py`: the old assertion that every `PREPROCESSING` entry has no dependency is
  now a per-entry table (`PREPROCESSING[name] = (call, dependencies)`), with the reason in a comment:
  hits need the whole-run `agipd_saxs`, so the second proc read is unavoidable.
  `tests/xpcs/test_xpcs_damnit.py` covers the module with mocks: hit resolution with the
  signals, the pass called with a `HitSet` and the SAXS defaults, overrides, empty hits (grey),
  failure cells, busy, colours, the readers.

**ctxrunner (pitfall 25), `scripts/xpcs_hits_context_check.py`.** Passed on r0493: §11.1.

### 11.1 What ran, and what did not

`scripts/xpcs_hits_context_check.py` (not unit-tested), job 25338041 (max-wn097, 36 workers, 12 min 19 s),
commit 00f6c09, JSON `scripts/xpcs_hits_context_check_r0493_20261018T224254Z.json`. Python is the
**deploy venv's**, with `PYTHONPATH` on this worktree's `src` (the report records
`analysis_package`; the deploy clone's `analysis` has no `xpcs.damnit` until the merge).

| Step | Result |
|---|---|
| `python -m ctxrunner ctx` on a scratch copy of `src/amore/context.py` + `analysis_helpers.py` | exit 0 **and** `error_info` is `None` in the pickle; 53 variables; the five `xpcs_hits_*` present; `xpcs_pipeline` still there |
| `exec 10400 493 all --cluster-job --var xpcs_hits_overview` | exit 0, no error line. Ran, in order: `agipd_preprocessing` 2.0 s (answered from complete files; `is_complete` was true before), `agipd_saxs` 11 s, `xgm_intensity`, `total_transmission`, `rep_rate`, **`xpcs_hits_preprocessing` 639 s**, `xpcs_hits_g2` 19.6 s, `xpcs_hits_ttcf` 18.0 s, `xpcs_hits_overview` 0.4 s |
| `exec … --var xpcs_hits_frames_used` | exit 0, 0.005 s |
| The hit cell | 2994 hits (the same count as the P4 hit set), 2994 usable, 2989 with a baseline; pairing 2989 PAIRED, 1 FIRST_HIT, **4 GAP_TOO_LARGE** at gap 1; green |
| g2, first lag (read back with `rep_rate` 2.2569) | 443.08 ns; c2 0.043, 0.038, 0.030, 0.057, 0.028, 0.038, 0.022 for the seven rings, jackknife errors 0.0006 to 0.007 |
| Where the files went | store 6.4 GB (2.1 MB a train) and results 658 MB under `scratch/xpcs_hits_p5/out/`; `usr/cached_files` untouched |

How it was kept off the product directory: the context file ran **unchanged**; the harness wraps
`analysis.xpcs.damnit.config_for` in the ctxrunner subprocess to set `output_root`. Everything else
(`damnit`, the hit rule on DAMNIT's real `runs.sqlite`, the real AGIPD frame table) is the production
path. The values are written to `extracted_data/` in the scratch work directory.

Not run, and why: `install_context.py` and `damnit read-context` (the live context and the shared
`runs.sqlite` are not to be touched); a mesh run (r0493 is a static PEG reference, so the block path
of `hits()` and a sparse hit set at gap 1 were not exercised here); a requeue of the DAMNIT job
(`allcpu` preemption; the lock takeover is pinned by the unit tests of `output_lock`, not by this
run); the `rep_rate` mismatch against a real DAMNIT value other than the pattern's own.
**Cost to know before the bulk run.** The pass took 639 s for 2994 hits from nothing, 3 times the
extract-only figure of §10 (208 s): the merge and the serial block writes follow it (§10 G6). The
photon store is 6.4 GB for r0493; at 2 to 2.3 MB a hit train that is about 20 to 23 GB per 10 000
hits, as budgeted.

**Installing, later (the user's go).** From a clean main checkout after the merge:
`python scripts/install_context.py` (dry run: diff, ctxrunner evaluation of both files), read the
diff (it should add only the `xpcs_hits_*` block), then `--install`. Then, per run, either let
DAMNIT run the new column, or reprocess through `scripts/reprocess_all.sh xpcs-hits [RUNS]` (`context/damnit-integrity.md` §7.1: never a bare `--match "XPCS hits"`, which would run the AGIPD pass for a run that lacks it). The pass takes the output
lock and answers a complete run from its files, so a relaunch writes nothing. A run's first pass
extracts its hits from proc (about 14 trains/s at 36 workers on r0493, §10), so run it before the
run's proc goes to tape; the photon store is what outlives proc.

