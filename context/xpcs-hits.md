# Hit-train XPCS (`analysis.xpcs`) — design, P0 facts, state

Context file for the replacement of extra-speckle's `xpcs_offline` by a pass that reads only
the droplet-hit trains. Read `CLAUDE.md` first (pitfalls 4, 8, 12, 17, 20, 26, 29, 30, 31), then
`context/xpcs-quality.md` (the histogram product that shares this read of proc) and
`context/diagnostics.md` §5 (how hits are decided).

**State (2026-10-18, branch `claude/xpcs-hit-trains`).** P0 (facts), P1 (config, rings,
kernel, partners), P2 (store, extract, run) and P3 (reader) are implemented and tested on a mock
run: `tests/xpcs`, 137 tests, none needing Maxwell or data. **Not done:** P4 (acceptance on
real runs against extra-speckle), P5 (the DAMNIT wrappers, `analysis.xpcs.damnit`,
`tests/test_context.py`, `install_context.py`), P6 (moving `analysis.triage.xpcs` onto the
reader and retiring `xpcs_pipeline`). `src/amore/context.py` is untouched.

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
| `rings.py` | `pixel_map` (the SAXS pass's `build_operator` and `ring_labels_from`, or a masks file), `Envelope`, `Rings` |
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

**Partner = the previous usable hit within `max_partner_gap` train ids** (`Kernel.CLEAN`).

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

## 9. Open

- **P4** (needs a node and the user's go): G1 `Kernel.ES` + `Averaging.ES` against
  `xpcs_offline(selected_trains=…)` on the same rings (their qmsk through `ring_masks_file`);
  G2 per-pixel attribution of every ring difference (9.04 vs 9.000 keV, 7.531 vs 7.5315962 m,
  seams, outlier mask, edges); G3 CLEAN against ES c2; G4 8 vs 36 workers; G5 rerun writes
  nothing; G6 sizes against an agreed budget. Benchmarks B2 (extract, `trains_per_block`),
  B3 (`train_sums`) and B4 (`max_partner_gap`, now 10) are unmeasured.
- **P5**: `analysis.xpcs.damnit` (the only caller of the classifier), `xpcs_hits_*` variables,
  the third `PREPROCESSING` entry in `tests/test_context.py`, ctxrunner on a scratch copy
  (pitfall 25), `install_context.py`.
- **P6**: `triage.xpcs.measure_run`, `inventory` and `xpcs_score` onto `read()`. Until then
  `analysis.triage.xpcs.ROI_EDGES_A` and `RING_EDGES_NM` are two lists; a test pins them equal.
- The sequence-agreement assumption (§7), the 5 mm margin and the gap of 10 are provisional.
- `g2()` over a few thousand trains has been tested for correctness on small inputs only.
