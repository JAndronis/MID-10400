# Hit-train XPCS (`analysis.xpcs`) — design, P0 facts, state

> **Variable names.** Since 2026-10-19 the DAMNIT columns are `Group` classes (`agipd.saxs`, `xpcs.g2`, ...); the flat names used below (`agipd_saxs`, `xpcs_hits_g2`, `xpcs_score`, ...) are the earlier ones. The map is `context/damnit-integrity.md` §8.1; `analysis.damnit_names` reads both.

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
| `pattern/pulse_ids`, `rings/{bounds_nm,label}` | the shared pulse pattern and the ring map used; `bounds_nm` is `(R, 2)` (schema 3, §13) |

1.35 MB a train raw at R = 7, P = 155; **0.19 MB after gzip** (§7). Changed inputs (hit set, merge hash, partner table) move the
old file to `xpcs_hits.prev.h5` and rebuild it: it is derived.

## 3. Types and modules (`src/analysis/xpcs/`)

| File | Owns |
|---|---|
| `hits.py` | `HitSet`: sorted unique uint64 ids, `sha256`, `source` (provenance, not hashed). `HitSet.of` validates once; empty is allowed |
| `config.py` | `XpcsConfig` (nests `AgipdSaxsConfig`; pickles by name; `extract_hash`, `merge_hash`), `Kernel`, `MaskPolicy`, `RINGS_NM`, `ES_RINGS_NM` |
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
- **`finalised` is set only on a file that reads back** (2026-10-20). `ResultWriter.finalise`
  writes the provenance, closes the file, `fsync`s it and reads every dataset in full from a
  fresh handle (`analysis.common.readback`); only then does it reopen it and set `finalised`. A
  file that fails is removed and the pass raises `UnreadableOutput`: its rows all say `OK`, so a
  resume would only try to finalise it again, and a rerun rebuilds it from the store. Why: the
  `usr` quota filled during that day's reprocess, and r0452, r0453, r0460 and r0469 were
  finalised with `sums/s_on`, `s_off_cur` and `s_off_prev` raising `filter returned failure
  during read`; `result_reasons` and `reprocess_safe.py`'s completeness check read attributes and
  statuses only, so both called them complete. Cost on r0424 (3000 hit trains, 0.87 GB on disk,
  6.9 GB decompressed): 12.2 s of a 341 s pass (3.6 %), merged from its store into scratch on
  max-exfl461 at 32 workers, sums identical to the production file; reading that production file
  back on its own takes 11.9 s.
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
- **P6 (2026-10-19)**: `triage.xpcs.measure_run(run, XpcsHits)`, `diagnostics.xpcs.xpcs_score` / `xpcs_snr` /
  `kbar_rois` and `scripts/run_triage.py --phase xpcs` read the results file only (lag from the pulse ids,
  N_pix from `XpcsHits.pixels`, k-bar from `kbar`, diagonals from `diagonals(kbar_min=3e-3)`); slices are
  anchored to the first usable hit. `measure_run_legacy` and the DAMNIT readers stay for gate e3. Against the
  old path on r0493, r0240, r0250 the run SNR is 2 to 10 % lower and the verdict unchanged: N_pix is the
  store's (3.5 to 9 % fewer pixels than the full ring) and a hit without a previous-train baseline has no
  `raw - off`, where extra-speckle paired it. `inventory` is still to move.
  `analysis.triage.xpcs.ROI_EDGES_A` and `ES_RINGS_NM` are two lists; a test pins them equal. The
  pass's own rings are `RINGS_NM` since 2026-10-20 (§13).

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
  and the XPCS defaults (`max_partner_gap` 1, `rings_nm`, 5 mm envelope), and `output_root`
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


## 12. The pixel-outlier mask, and what else `xpcs_offline` did (design, 2026-10-19)

User decision (2026-10-19): the hit pass reproduces the **whole** behaviour of the old
`xpcs_pipeline` column (one `xpcs_offline` call, `git show HEAD:src/amore/context.py`, the
`xpcs_pipeline` variable), restricted to hit trains. The first gap is extra-speckle's pixel-outlier
mask: on r0493 it removes 2415 ring pixels the hit pass kept, pixels whose mean depends on the
memory cell in 32-frame blocks (0.96 / 1.13 / 0.93 / 0.96 of the mean), which leave 32-frame blocks
in the pooled raw and off maps. It is to be **kept, computed from the hit trains, and applied at the
merge**, so the store is untouched.

### 12.1 What extra-speckle computes (read from the installed source, confirmed on r0493)

`pipeline.xpcs_offline(outlier_masking=True, outlier_thresholds=(0.75, 1.75),
outlier_q_bounds=None)` (the DAMNIT call passes only `outlier_masking=True`; `log.txt` of
`scratch/xpcs/r0493` records the thresholds) calls `stats.tools.mask_outlier(mean_pixelcell, setup)`:

```
mean_pixelcell[f, p] = Σ_t x[t,f,p]·g[t,f,p] / Σ_t g[t,f,p]      g = not image.mask[t,f,p] and not static[p]
                                                                 (stats.calc_stats, mask_dynamic=True; 0/0 → NaN)
                                                                 f = frame position in the train (pulse_coord 'idx'), every train of the run
s[p]   = sqrt(nanmean_f mean_pixelcell[f,p]²)                     "std_of_mean": an RMS over frames, not a std
c_i    = the 300 bin centres of saxs.get (pyFAI integrate1d, npt=300 hard-wired, q_A^-1, whole detector)
ratio[p] = s[p] / nanmedian{ s[p'] : c_i < q[p'] ≤ c_{i+1} }      for the interval p falls in
ratio[p] = 1.25 (the mean of the thresholds) when p is in no interval
outlier[p] = ratio[p] < 0.75  or  ratio[p] > 1.75                 NaN ratio → not an outlier
setup.mask |= outlier                                             then every ROI mask, so the TTCFs and g2 lose them
```

So the statistic is **one number per pixel**, not per memory-cell block: a pixel whose mean differs
between cell blocks has a large RMS over frames and is removed whole. Nothing in it works per
block. `mask_TTCF_diagonal_boxes` (32×32 diagonal blocks) exists in extra-speckle and was **off**.

Check (`scratch`, read-only, extra-speckle's own `mean_pixelcell.npy` and `SAXS.nc` for r0493, q
from our geometry at extra-speckle's 9.000 keV and 7.5315962 m): the formula above gives 3275
outliers (1933 low, 1342 high) against the 3278 of `outlier_mask.npy`, 13 pixels different, the
size of the encoder-vs-file geometry difference of G2. The 300 centres run 0.008234 to 0.105113
Å⁻¹ in steps of 3.2401e-4.

### 12.2 Design

- **Config.** `XpcsConfig.outlier_thresholds: tuple[float, float] | None = (0.75, 1.75)` (None
  switches it off; the production values, not the figure's guess) and `outlier_q_bins: int = 300`
  (extra-speckle's hard-wired `npt`). Both enter `merge_hash` as
  `"outlier": None | {"thresholds", "q_bins"}`. Not in `extract_hash`: the store is unchanged.
- **Statistic, over the hit trains.** The trains are the prelude's OK ones (one pulse pattern),
  *all* of them, whether paired or not: the mask is a property of the run's pixels, so it does not
  depend on `max_partner_gap` or the kernel. Per frame position `f` and envelope pixel `p`:
  `S[f,p]` = photons where the frame does not flag `p` (the flag base XOR the frame's listed
  difference, i.e. `mask_dynamic=True`), `C[f,p]` = trains in which it is not flagged. The
  candidates are the envelope pixels that are not statically bad (ring or margin: the median of a
  q bin at a ring edge needs the pixels just outside the ring, as extra-speckle's covers the whole
  detector). The q bins are extra-speckle's construction on our geometry: 300 equal bins over the
  q range of the detector's not-statically-bad pixels, intervals between consecutive centres,
  1.25 outside them.
- **Where it enters.** `outlier[p]` is folded into the ring labels (`label = −1`) before the merge
  workers start, like the static mask. `train_sums`, `MaskPolicy.ANY_FRAME` (which still acts per
  train on top), the pair mask and both kernels are unchanged, so `Averaging.ES` and `POOLED`
  read the same sums and stay consistent. `XpcsHits.pixels` (N_pix) follows the applied labels.
- **What the results file records.** `rings/label` is the applied map; new
  `outlier/{ratio (float32, n_env), mask (bool, n_env), label_before (int8)}`, attrs `thresholds`,
  `q_bins`, `n_trains`, `sha256` of the mask, and the same sha in provenance. `SCHEMA` 1 → 2.
- **Hash: the inputs, not the mask's sha.** The mask is a deterministic function of the store, the
  hit set (whose OK trains are fixed by the prelude), the static rings and the settings, all of
  which `merge_hash` already names. Putting the mask's own sha256 in `merge_hash` would force the
  statistic pass to run before a complete run could be answered from the file, which is G5's
  0.5 s rerun becoming a minute. The mask's sha256 is stored and recorded for audit instead, and a
  resumed merge (same identity) reads the mask back from the file rather than recomputing it.
- **Cost.** One more read of every OK hit train from the store (`store.get` ≈ 0.12 s a train,
  2.7 M events a train on r0493, measured on a login node), by `fan_out` over blocks of at most 257
  trains (so a block's sums fit uint16: 257 × 255) and at least one per worker. Each block returns
  a dense `(P, n_env)` uint16 sum (295 MB at r0493's size) and its flag counts sparse; the parent
  adds them into uint32. A sparse return does not pay: 8 trains give 19.7 M distinct (frame, pixel)
  keys out of 21.5 M events. Expected on r0493 at 36 workers: ~15 s of reads in the workers,
  ~10 GB through the pipes, under a minute in all (to be measured, §12.4). Memory: ~0.45 GB a
  worker, ~1.2 GB in the parent. r0423 scales with its hit count.
- **A block that fails** cannot be left out (it would change the mask), so a failed block stops the
  merge with an error, before any results row is written; a rerun retries it.
- **Not supported with `ring_masks_file`** (no q per pixel): `XpcsConfig` refuses
  `ring_masks_file` with `outlier_thresholds` set. The G1 harness sets it to None.
- **Invalidated.** Every `xpcs_hits.h5` (schema 1, and the merge hash moved): the next pass moves it
  to `xpcs_hits.prev.h5` and rebuilds it from the store, minutes, no proc. That is
  `usr/cached_files/xpcs_hits/r0493/xpcs_hits.h5` and anything under `scratch/xpcs_hits_p5`. No
  photon store is invalidated.

```
outlier_mask(store, ok_trains, envelope, static_bad, q_nm, thresholds, q_bins):
    S = zeros((P, n_env), uint32); flagged = zeros((P, n_env), int32)
    for block in blocks(ok_trains, ≤ 257):                       # workers
        for t in block: s16[f, p] += x[t,f,p] where not flag[t,f,p]; flagged += flag[t]
    C = n_trains − flagged
    s = sqrt(nanmean_f (S/C)²)  over candidates (not static_bad), NaN where C = 0 in every frame
    centres = linspace bins over q of the detector's good pixels; ratio as §12.1
    return ratio < lo or ratio > hi
```

### 12.3 Gap list: `xpcs_offline` as the DAMNIT column called it, against the hit pass

| Old behaviour (argument) | Here |
|---|---|
| every train of the run (`selected_trains=None`) | **differently, on purpose**: the hit trains only (the point of the pass) |
| `remove_outer_points=False` | reproduced (nothing removed) |
| geometry `geom=None` (motor encoders, pitfall 16), `px/py` 607.46/672.08, `sdd` 7531.5962 mm, photon energy from the run (9.000 keV) | **differently, on purpose**: `geom_latest.geom`, 7.531 m, 9.04 keV (G2: 28 151 ring pixels differ, every one explained) |
| `q_range` 0.0225 … 0.086 Å⁻¹, seven rings | reproduced (`ES_RINGS_NM`) until 2026-10-20; since then **differently, on purpose**: twelve rings with the Bragg bands left out (§13) |
| `mask=MASK_PATH` | reproduced (plus ASIC seams, already inside it) |
| `mask_dynamic=True`: per-frame `image.mask` in the stats, any frame of a train in the TTCF | reproduced (`MaskPolicy.ANY_FRAME`; per frame in the outlier statistic) |
| `calc_stats=True`: `mean_pixelcell`, `mean_intensity` | `mean_intensity` reproduced on ring pixels (`figures.mean_intensity`); `mean_pixelcell` is computed inside the outlier pass and not kept as a product |
| `refine_beamcenter=False` | reproduced (no fit; `figure_beamcenter` is a check) |
| **`outlier_masking=True`, (0.75, 1.75), whole q range** | **missing → implemented here (§12.2)** |
| `intensity_masking=False` (`xgm_source` used only by it) | reproduced (no train removed); `figure_intensity_outliers` draws it |
| `run_saxs_analysis=True` (`SAXS.nc`, per-frame I(q), pyFAI 300 bins) | **differently**: `figures.saxs_curves` bins the store's annulus (plain means, no pixel splitting); the AGIPD SAXS pass is the product |
| `run_xpcs_analysis=True`, `sparsify=True`, `photonize=False`, `extra_filtering=False`, all pulses | reproduced bit for bit by `Kernel.ES` (G1); `Kernel.CLEAN` **differs on purpose** (§4: previous hit within `max_partner_gap`, no chain restart, no own-mask first train) |
| `kbar_masking=True`, `kbar_threshold=3e-3` | reproduced in `Averaging.ES` only; the pooled view has no k-bar cut, on purpose (no source for 3e-3) |
| `subtract_TTCF_off=True` | reproduced (`c2 = raw − off`) |
| `mask_noisy=True`, `nan_thresh=0.025` | reproduced in `Averaging.ES` (§6) |
| `mask_TTCF_diagonal_boxes=False` | reproduced (not done) |
| `bin_g2=True`, `(50, 20, 10)` | reproduced (ES view: medians as extra-speckle; pooled: summed numerators, on purpose) |
| `rep_rate` from DAMNIT | reproduced, as a check (`RepRateMismatch`) |
| `fit_result=True` (`xpcs.fit`, `dr=True`) | reproduced (`figures.fit_g2`) |
| `run_xsvs_analysis`, `run_xcca_analysis` False | nothing to reproduce |
| a train with missing modules or a short frame count is dropped by `chunk_trains_agipd` | reproduced as statuses (§5), counted rather than dropped |
| g2 without error bars | addition: jackknife `sigma` |

Nothing else in the call is missing. The open item is the outlier mask; everything marked
"differently" is a recorded decision.

### 12.4 Implemented and checked on r0493 (2026-10-19)

`analysis.xpcs.outliers` (statistic, worker, sink), `XpcsConfig.outlier_thresholds` /
`outlier_q_bins`, `run._outlier_mask` before the merge, `result` schema 2 (`outlier/`,
`XpcsHits.outlier()`), `figures` (the applied mask in `figure_outliers`, `figure_q_rings`,
`q_curves`). Tests: `tests/xpcs/test_xpcs_outliers.py` (the statistic against a dense rewrite of
`mask_outlier`, flag base vs listed flags, uint16 bound, hash, labels and `cnt`, workers, a complete
rerun answered without the statistic, a failed block refuses). The test `cfg` fixture sets
`outlier_thresholds=None` so the kernel tests keep their independent rings. The G1 harness passes
`outlier_thresholds=None` with `ring_masks_file`.

Job 25339400 (max-wn164, 36 workers), `scratch/xpcs_hits_outlier/verify.py`, the real store
through a symlink, results under `scratch/xpcs_hits_outlier/root` (`usr/cached_files` not written),
JSON `scratch/xpcs_hits_outlier/r0493/verify.json`:

| | |
|---|---|
| outlier pass | **25.9 s** for 2994 trains (whole merge 140.6 s wall) |
| pixels | 2617 in the envelope, **2402 in a ring**; extra-speckle 2415 in our rings: **2385 common**, 30 only theirs, 17 only ours, every one of the 47 with our ratio within 0.004 of a threshold (0.746 to 0.761, 1.748 to 1.751): bin edges and geometry (9.04 vs 9.000 keV, all trains vs hits) |
| 32-frame blocks, pooled maps | gone: diagonal block means of `raw` at 0.0605 Å⁻¹ 1.122/1.121/1.109/1.108/1.111 → 1.099/1.098/1.099/1.098/1.100; block spread of `off` 0.0047 → 0.0018 (0.0605), 0.0044 → 0.0009 (0.067) |
| g2, `Averaging.ES` vs extra-speckle's `g2.nc`, median relative difference | 0.0605 Å⁻¹ **11.7 % → 0.17 %**; 0.067 Å⁻¹ **32.2 % → 1.35 %** |

Figures: `figures/xpcs_hits/r0493_outlier_maps.png`, `r0493_outlier_mask_vs_es.png`,
`r0493_figure_outliers.png`, `r0493_figure_q_rings.png`. Left visible: a faint line at frame ≈ 30
in the 0.0605 Å⁻¹ raw map, and single-frame spikes in I(q) of the first frames that the RMS over 155
frames does not catch (extra-speckle's statistic has the same blind spot). Not run: r0423 (no hit
set, §10).

## 13. Rings without the Bragg bands, β per ring, the outlier figure (2026-10-20)

**Why.** extra-speckle's ROIs at 0.0605 and 0.072 Å⁻¹ (0.56–0.65 and 0.69–0.75 nm⁻¹) hold the fcc
111 and 200 reflections, and on crystallised runs (r424) their c2 read ~1.0–1.3 against ~0.005 for
the others: Bragg speckle, not dynamics (CLAUDE.md epistemic boundary 2 applies to XPCS too).

**The bands, measured** (all under the 2026-09-13 beam centre; figures in `figures/xpcs_rois/`):

| measure | 111 | 200 |
|---|---|---|
| pooled I(q) excess > 3 % over r0423 (local-median baseline), r0424 / r0426 / r0488 | 0.593–0.631 nm⁻¹ | 0.693–0.715 |
| p90 over 438 runs of the fraction of hit-train pixels with outlier ratio > 1.75 (`outlier/ratio`) | 0.598–0.618 | 0.698–0.713 |

0.704 / 0.608 = 1.158 against fcc's 2/√3 = 1.155. A move of the beam centre (open task 4) splits
111 into 0.584 + 0.633; it also changes every pixel's q, so it re-merges anyway and the bands are
to be measured again then.

**The rings** (`analysis.xpcs.config.RINGS_NM`): twelve, 0.045 nm⁻¹ wide, 0.225 → 0.585, 0.640 →
0.685, 0.725 → 0.860, i.e. 111 and 200 left out with ≥ 0.008 nm⁻¹ margin. The outer bounds are
extra-speckle's floats (`10 * 0.0225` = 0.22499999999999998, not 0.225), so the envelope sha256 and
every photon store are unchanged; any other outer bound would refuse every store (`StoreMismatch`)
and need proc. Pixels per ring (static mask applied): 7k, 13k, 22k, 43k, 53k, 55k, 67k, 75k, 78k,
79k, 67k, 52k. `triage.roi_kinds` still tags 0.05625, 0.06625 and 0.07475 Å⁻¹ "bragg" (they touch
the wider SAXS windows `BRAGG_WINDOWS_NM`), so `xpcs.score` takes the best of the other nine. The score cell's ROI counts come from the results file (`n_rois`, `n_plain_rois`; commit
`50a1939`), e.g. r0424 "k̄ ok 10/12 (7/9 plain)", amber: every ring must meet k̄ ≥ 3e-3, and the two
outermost miss it there; r450 "12/12 (9/9 plain) · SNR 68.6".

**Code.** `XpcsConfig.rings_nm: tuple[(lo, hi), ...]`, ascending and disjoint, replaces the
contiguous `ring_edges_nm`; a gap is in no ring but inside the envelope (first lo to last hi).
`ES_RINGS_NM` keeps extra-speckle's seven for the equivalence gates (`scripts/xpcs_hits_acceptance.py`
sets it). Results schema 3: `rings/bounds_nm (R, 2)` replaces `rings/edges_nm`;
`XpcsHits.bounds_nm`, `DetectorMean.ring_bounds_A`, `triage.roi_edges_from(bounds, n)` follow.

**β.** `REFERENCE_BETA` holds the twelve rings' β (`ES_REFERENCE_BETA`, the 2026-10-13 table, is
kept inside it and on its own for the photon statistics, which still bin on extra-speckle's ROIs;
no centre is shared). Measured on r450 and r493–496 merged from their stores with the new rings
into `scratch/xpcs_rois_20261020` (store linked, 348–494 s a run on max-exfl461 at 32 workers,
no proc read), then `run_triage.py --phase xpcs` (`scripts/run_triage_xpcs_20261020T114609Z.json`):
every one of the 60 fits converged; β 4.88 % at 0.02475 Å⁻¹ falling smoothly to 1.74 % at 0.08375,
with no step where the Bragg ROIs were.

**Outliers figure.** The spikes left in `XPCS outliers` "after" panel (§12.4 "single-frame spikes
… the RMS does not catch") are **not** pixels the sums use. On r0424 all 15 pixels behind them have
a NaN ratio: `image.mask` flags them in every frame of every hit train (6043 ring pixels, checked
on sampled trains), so `MaskPolicy.ANY_FRAME` removes them from every train, but
`detector_mean` reads the store's photons without flags. With them out, 0.75/1.75 and 0.8/1.3 leave
the same spikes (r0423 1, r0424 155, r0426 468, all at the Bragg peaks on the crystallised runs),
so the thresholds stay (user, 2026-10-20). `figure_outliers` now leaves out of both panels the
pixels `detector_mean` finds flagged in some frame of every train it reads
(`DetectorMean.always_flagged`) and counts them in the title. A NaN ratio alone is not that: an
empty q bin gives 0/0, and the first version, keyed on NaN, drew nothing on r1 (no beam) and
raised. Reading the flags makes the store figures slower (r1 `xpcs.outliers` 11 → 28 s, r450 65 s).

**Reprocess.** The merge hash moved (ring sha and schema), so every `xpcs_hits.h5` is rebuilt from
its store: `reprocess_safe.py start RUNS -d xpcs` **without `--fresh`**. `reprocess_all.sh
xpcs-hits` passes `--fresh`, which moves the photon stores aside and would re-extract from proc.

## 14. The rings leave out the Bragg windows (2026-10-27)

**Why.** §13's rings left out only the measured bands (0.585–0.640 and 0.685–0.725 nm⁻¹), so
three of them touched the wider windows `BRAGG_WINDOWS_NM` (0.57–0.65 and 0.68–0.74 nm⁻¹) and
`roi_kinds` tagged them "bragg". The user set those windows as authoritative (2026-10-27).

**The windows.** `BRAGG_WINDOWS_NM` and `touches_bragg` live in `analysis.xpcs.config` alone.
`triage.roi_kinds` and `saxs.photon_stats.default_rings` import them from there; nothing else
restates a band edge.

**The rings.** `RINGS_NM` is computed: 0.045 nm⁻¹ rings stepped up from extra-speckle's lower
bound and down from its upper bound, each stopping before the windows. That gives nine rings,
0.225 → 0.540 and 0.770 → 0.860, all "plain". They are §13's nine that touched no window, with the
same edges to within a float ULP, and the outer bounds are still extra-speckle's floats, so the
envelope and every photon store stay valid. Dropped: 0.540–0.585, 0.640–0.685 and 0.725–0.770
nm⁻¹ (centres 0.05625, 0.06625, 0.07475 Å⁻¹), and with them their three `REFERENCE_BETA`
entries; the other nine β values stand.

**Stored files.** The results schema stays 3, because the layout is unchanged. The ring sha moves
with the bounds, so the merge hash moves and every twelve-ring `xpcs_hits.h5` reads "merged under
another configuration". The readers take the ring count from the file, so a twelve-ring file
still opens; `fixed_beta` gives NaN for its three dropped centres, which are "bragg" there anyway.

**Reprocess.** Re-merge every run from its store: `reprocess_safe.py start RUNS -d xpcs`
**without `--fresh`** (§13), then refresh `xpcs.score` and the `*_frames_used` cells.
