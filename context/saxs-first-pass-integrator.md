# SAXS first-pass integrator (AGIPD) — design and implementation guide

Context file for `<pkg>.saxs`. This guide replaces `analysis_helpers.integrate_run`.

Read `CLAUDE.md` first. It holds the scientific context, the environment, the data-format facts
(§ "Data files and formats found") and general pitfalls. This file does not repeat them.

Work phase by phase (§10). Present a plan before creating files in each phase. Do not start a
phase until the previous phase's acceptance criteria pass.

---

## 1. Scope

**In scope (v1).** For every AGIPD frame of a run in the proc data:
- per-q-bin 1D sufficient statistics;
- a per-frame status ledger;
- full provenance, written to scratch.

Per-train pooled I(q) is returned to DAMNIT.

**Out of scope (v1):**
- WAXS JUNGFRAU (CLAUDE.md open task 2);
- XCCA I(q,φ) (second pass);
- XPCS;
- bulk chunk reader (I3);
- GPU, dask, OpenMP;
- XGM normalisation, transmission correction, V/V₀ pairing and background subtraction. These are
  post hoc modules working on the stored sums (§9).

**Priority:** data quality > completeness guarantees > speed.

---

## 2. Decisions and the measurements behind them

| Decision | Evidence (r0423 unless noted) |
|---|---|
| Sparse integration over photon hits instead of dense pyFAI per frame | Data are photon counts, 0.7–1.8 % nonzero per frame. Sparse path matches pyFAI full-split CSC to 6e-8 with no empty-bin disagreements; 6.2 vs 19.0 ms/frame |
| Exact per-frame masking with a per-cell base mask plus sparse correction | Static bits are per memory cell (4.05 % always, 0.022 % vary). Bits 12/13 are per frame but few (~31 px/frame), zero-valued and intensity-independent, also in crystallised r0426. The mask read costs 8.2 ms/frame; kept for data quality |
| EXtra-data HDF5 path, `decompress_threads=1` explicitly | Threaded path is 2.1–3.2× slower on these files (per-chunk index lookups) |
| One single-threaded worker per physical core, spawned processes | Read scaling 88 % at 32 processes; CSR with 16 threads is 7.6× less efficient per core than 1 thread; crossing sockets is slower still |
| No GPU, no dask | Workload is CPU-bound decode + light integration; DAMNIT cannot request GPUs; a static partition of trains needs no task graph |
| Store Σc·x, Σc·Ω, Σc²·x per frame | At high q a frame has ~1 photon per bin. Sums give exact pooling and errors for any later grouping. pyFAI's Poisson model inflates variance ~87× on this data |
| Seam mask + custom mask on top of `image.mask` | `NON_STANDARD_SIZE` is never set, so double-width ASIC-edge pixels are unflagged. The old pipeline applied both masks |

**Budget:** data read 4.6 + mask read 8.2 + sparse integration 6.2 ≈ 19 ms/frame/core.
Target for r0423 (~465 k frames): ≈ 4.7 min on 36 workers.

---

## 3. Integrator rules

These add to the general pitfalls in CLAUDE.md.

1. **Operator method.** Built with `("full", "csc", "cython")`. Assert
   `(res.method.split_lower, res.method.algo_lower, res.method.impl_lower)` equals it.
2. **Variance.** Always Σ c²·x. Never pyFAI `ErrorModel.POISSON`.
3. **EXtra-data reads** of `image.*` pass `decompress_threads=1`. The **parent** sets
   `EXTRA_NUM_THREADS=1` and `OMP_NUM_THREADS=1` in `os.environ` *before the pool is constructed*.
   A spawned child inherits the environment at process start, whereas a `ProcessPoolExecutor`
   initializer runs only after the child has imported the worker module — and therefore numpy —
   so setting the variables there is too late for thread pools that already exist. Workers
   re-assert them anyway, which only helps libraries imported lazily.
4. **Geometry.** The geometry object is the single source.
   - Use `AzimuthalIntegrator(detector=geom.to_pyfai_detector(), dist=sdd, wavelength=λ)`, PONI = 0.
   - No `setFit2D`, no second `set_pixel_corners`, no PONI files.
5. **Identity.** Train, pulse and cell identity come from reader coordinates. Rows are written by
   label.
6. **Processes.** Spawn only. Worker functions live in `<pkg>`.
7. **No NaN sentinels.** Every frame has a status code (§8).
8. **Single writer.** Only the parent process opens the output file for writing.
9. **Hot loop scope.** Nothing in the hot loop depends on XGM, transmission or background.

---

## 4. Architecture

```
DAMNIT cluster job (one node, one run)
  └─ <pkg>.saxs.run.run_first_pass(cfg)                          [parent]
       ├─ plan.build_plan          trains, frame counts, rows, blocks, run checks
       ├─ operator.build_operator  sparse full-split operator + Ω, hashed, saved
       ├─ masks.build_static_bad   ASIC seams ∪ custom mask ∪ lobe mask (I4a), hashed
       ├─ masks.build_base_masks   per-cell base mask + base denominators, saved
       ├─ selftest.run_selftest    sparse vs pyFAI on real frames → abort on failure
       ├─ ProcessPoolExecutor(spawn, n_workers, initializer=worker.init)
       │     worker.process_block(block) → BlockResult     [one core each, read + integrate]
       ├─ writer.write_block       label checks, rows by label, ledger, flush
       └─ writer.finalise          provenance, pooled per-train I(q) → DAMNIT
```

Modules under `src/<pkg>/saxs/`:

| Module | Contents |
|---|---|
| `config.py` | Frozen config dataclass and hash |
| `plan.py` | Trains, frame counts, rows, blocks, run checks |
| `operator.py` | Sparse operator build/save/load |
| `masks.py` | Static and per-cell base masks, denominators |
| `sparse.py` | Gather + bincount kernels; the pure per-frame integration (§6.4) |
| `worker.py` | Per-block read and frame loop; calls `sparse.integrate_frame` |
| `writer.py` | HDF5 schema, ledger, resume |
| `selftest.py` | Sparse vs pyFAI gate |
| `status.py` | Frame status codes (§8) and the data-check exception |
| `run.py` | Orchestration |
| `cli.py` | Standalone entry point |

The reader layer (`plan.py`, reads in `worker.py`) must not import DAMNIT, so it can back the
pyBeamtime EuXFEL reader later.

---

## 5. Config (`config.py`)

Frozen dataclass. The hash covers every field, the `<pkg>` git commit and the sha256 of every
input file.

| Field | Default | Notes |
|---|---|---|
| `proposal`, `run` | — | from DAMNIT `meta#proposal`, `meta#run_number` |
| `detector_name` | `None` | auto-detect via `AGIPD1M`; record the detected name |
| `geometry_file` | `.../usr/geometry/geom_latest.geom` | sha256 recorded; assert corner z spread < 1 mm so distance is not applied twice |
| `sdd_m` | 7.531 | |
| `photon_energy_kev` | 9.04 | compare with `XGM(run).photon_energy_by_train()`; fail if not constant or > 0.1 % apart |
| `npt` | 500 | unit `q_nm^-1`; q range from geometry, recorded |
| `mask_bits` | `0xFFFFFFFF` | every bit present in these files marks unusable pixels |
| `expected_bits` | {0, 1, 7, 8, 9, 12, 13} | other bits present → provenance flag + warning |
| `use_asic_seams` | `True` | `extra_geom.agipd_asic_seams()` repeated over 16 modules |
| `custom_mask_file` | `.../usr/Shared/IA/custom_agipd_mask.npy` | non-zero = excluded; assert shape (16, 512, 128) or (512, 128) |
| `lobe_mask_file` | `/gpfs/exfel/u/usr/MID/202601/p010400/masks/mask_2026-09-08_AGIPD_SAXS.npy` | I4 decided (a): anisotropic low-q lobe. Non-zero = excluded; assert shape (16, 512, 128) or (512, 128); sha256 recorded; OR'd into `static_bad` |
| `base_mask_trains` | 8 | spread evenly over the run |
| `selftest_frames` | 8 | from 2 trains |
| `trains_per_block` | 4 | unit of scheduling, ledger and resume (not memory) |
| `n_workers` | physical cores | SMT only after P6 |
| `min_modules` | 16 | trains with fewer → `MISSING_MODULES` |
| `output_root` | `/gpfs/exfel/exp/MID/202601/p010400/scratch/saxs_first_pass` | file `r{run:04d}/saxs_first_pass.h5` |
| `allow_incomplete` | `False` | if `False`, raise when any frame status ≠ OK |
| `overwrite` | `False` | if `False`, refuse an existing file with a different config hash |

---

## 6. Algorithms

`SHAPE = (16*512, 128)`, `NPIX = SHAPE[0]*SHAPE[1]`. The flattened pixel order is
module × ss × fs. It equals `ndarray()[:, k].reshape(-1)` when all 16 modules are present in
sorted order.

### 6.1 Operator (`operator.py`, parent, once per job)

```
geom  = AGIPD_1MGeometry.from_crystfel_geom(cfg.geometry_file)
ai    = AzimuthalIntegrator(detector=geom.to_pyfai_detector(), dist=cfg.sdd_m, wavelength=λ)
probe = ai.integrate1d(ones(SHAPE, f32), cfg.npt, method=("full","csc","cython"), unit="q_nm^-1")
assert resolved(probe.method) == ("full","csc","cython")
eng   = ai.engines[probe.method].engine
data, rows, indptr = eng.lut                       # CSC: column j = pixel j
omega = ai.solidAngleArray(SHAPE).ravel().astype(f64)
q     = eng.bin_centers
save operator.npz {data, rows, indptr, omega, q, nnz}; op_hash = sha256(arrays)
```

### 6.2 Sparse kernels (`sparse.py`)

```
def gather(cols, weights, squared=False):
    starts = indptr[cols]; counts = indptr[cols+1] - starts
    idx    = repeat(starts - (cumsum(counts) - counts), counts) + arange(counts.sum())
    coeff  = data[idx] ** (2 if squared else 1)
    return bincount(rows[idx], weights=coeff * repeat(weights, counts), minlength=npt)   # f64
```

### 6.3 Static and base masks (`masks.py`, parent)

```
static_bad = zeros(NPIX, bool)
if cfg.use_asic_seams:   static_bad |= repeat(agipd_asic_seams()[None], 16, 0).ravel()
if cfg.custom_mask_file: static_bad |= broadcast_to(load(file), (16, 512, 128)).ravel() != 0
if cfg.lobe_mask_file:   static_bad |= broadcast_to(load(file), (16, 512, 128)).ravel() != 0  # I4a
static_hash = sha256(packbits(static_bad))

trains = evenly spaced cfg.base_mask_trains over OK trains
for tid in trains:
    m, cells = image.mask.ndarray(decompress_threads=1), cell_id_coordinates()
    for k, cell in enumerate(cells):
        votes[cell] += ((m[:, k].reshape(-1) & cfg.mask_bits) != 0) | static_bad     # uint8
        n[cell] += 1
        bits_present |= bitwise_or.reduce(m[:, k], axis=None)
base_bad[cell] = votes[cell] * 2 > n[cell]                        # majority; ⊇ static_bad
D[cell]        = gather(flatnonzero(~base_bad[cell]), omega[~base_bad[cell]])
D_static       = gather(flatnonzero(~static_bad), omega[~static_bad])   # for unseen cells
save masks.npz {cells, packbits(base_bad), D, n_samples, packbits(static_bad), D_static,
                static_sha256, operator_sha256, bits_present, unexpected_bits, sha256}
# sha256 is verified on load; operator_sha256 ties the masks to the operator they were built with
```

A cell that appears later without having been sampled uses `base_bad = static_bad` and
`D_static`. The per-frame correction stays exact; count such cells in provenance.

### 6.4 Per frame (`worker.py`)

```
x = data[:, k].reshape(-1)          # int16 photon counts
m = mask[:, k].reshape(-1)
bad  = ((m & cfg.mask_bits) != 0) | static_bad           # full scans
nz   = flatnonzero(x)
if nz.size and x[nz].min() < 0: status = DATA_CHECK_FAILED
keep = nz[~bad[nz]]
xs   = x[keep].astype(f64)
S    = gather(keep, xs)                                  # Σ c·x
V    = gather(keep, xs, squared=True)                    # Σ c²·x   (Poisson, unfloored)
diff = flatnonzero(bad != base_bad[cell])                # frame-specific flags
sign = where(base_bad[cell][diff], +1.0, -1.0)
N    = D[cell] + gather(diff, sign * omega[diff])        # Σ c·Ω over valid pixels
scalars: photons_valid = xs.sum(), n_bad = bad.sum(), n_frame_specific = diff.size,
         max_count = x[nz].max() if nz.size else 0
```

`image.data` must have an integer dtype. Otherwise the block fails with `DATA_CHECK_FAILED`;
there is no silent dense fallback.

### 6.5 Block (`worker.py`)

```
init(paths, cfg):   re-assert thread env = 1 (the effective setting comes from the parent,
                    §3 rule 3); load operator.npz and masks.npz (unpack once);
                    det = AGIPD1M(open_run(cfg.proposal, cfg.run, data="proc"),
                                  detector_name=cfg.detector_name, min_modules=cfg.min_modules)

process_block(block):
    for tid in block.train_ids:
        sel  = det.select_trains(by_id[[tid]])
        kx   = sel["image.data"]; km = sel["image.mask"]
        x    = kx.ndarray(decompress_threads=1); m = km.ndarray(decompress_threads=1)
        tids = kx.train_id_coordinates(); pids = kx.pulse_id_coordinates(); cells = kx.cell_id_coordinates()
        assert all(tids == tid) and len(tids) == x.shape[1] == block.expected_frames[tid]
        bits_present |= bitwise_or.reduce(m, axis=None)
        per frame k: §6.4
    return BlockResult(trainId, pulseId, cellId per frame; S, N, V as f32 (n, npt); scalars;
                       status per frame; timings {read_data, read_mask, integrate, cpu};
                       bits_present)
```

Worker memory is about one train of data plus mask (~1 GB).

### 6.6 Self-test gate (`selftest.py`, parent, before the pool starts)

```
for (x, m, cell) in cfg.selftest_frames frames from 2 OK trains:
    bad = ((m & cfg.mask_bits) != 0) | static_bad
    f   = where(bad, nan, x).astype(f32).reshape(SHAPE)
    ref = eng.integrate_ng(f, variance=x.astype(f32).reshape(SHAPE), solidangle=omega.reshape(SHAPE))
    S, N, V = §6.4
    rel(a, b) = |a - b| / max(|b|, 1e-12)
    require: (N > 0) == (ref.normalization > 0)
             max rel(S, ref.signal), rel(N, ref.normalization), rel(V, ref.variance) < 1e-6   on N > 0
on failure: write provenance, raise SelfTestFailed; the pool is never started
```

### 6.7 Orchestration (`run.py`)

```
def run_first_pass(cfg, pool_factory=None):
    set_thread_env()                          # parent-side, before any pool exists (§3 rule 3)
    out    = writer.open_or_create(cfg)       # refuse on config-hash mismatch unless cfg.overwrite
    plan   = plan.build_plan(cfg)
    op     = operator.build_operator(cfg)
    static = masks.build_static_bad(cfg, op)
    base   = masks.build_base_masks(cfg, plan, op, static)
    selftest.run_selftest(cfg, plan, op, static, base)
    todo   = [b for b in plan.blocks if not out.block_complete(b)]   # resume
    # pool_factory defaults to ProcessPoolExecutor; injecting it lets the BrokenProcessPool and
    # WORKER_ERROR paths be tested without killing real processes or putting test hooks in worker.py
    with (pool_factory or default_pool)(cfg.n_workers, mp_context=get_context("spawn"),
                             initializer=worker.init, initargs=(paths, cfg)) as pool:
        futures = {pool.submit(worker.process_block, b): b for b in todo}
        for fut in as_completed(futures):
            b = futures[fut]
            try:                   res = fut.result()
            except BrokenProcessPool: out.mark_remaining(NOT_PROCESSED); raise
            except Exception as e: out.mark(b, WORKER_ERROR, repr(e)); continue
            out.write_block(b, res, plan)      # validates labels & counts, rows by label, flush
    out.finalise(provenance(cfg, plan, op, static, base, timings, bits_present))
    if not cfg.allow_incomplete and out.any_not_ok(): raise IncompleteRun(out.status_summary())
    return out.pooled_per_train()              # xarray (trainId, q)
```

### 6.8 Plan and run checks (`plan.py`)

```
dc     = open_run(cfg.proposal, cfg.run, data="proc")
det    = AGIPD1M(dc, detector_name=cfg.detector_name, min_modules=cfg.min_modules)
counts = det.frame_counts                  # pandas Series indexed by trainId
first  = counts.cumsum() - counts          # row offsets in the flat frame table
status: dc.train_ids not in det.train_ids → MISSING_MODULES; counts == 0 → NO_FRAMES
blocks = consecutive OK trains, cfg.trains_per_block each
run checks → provenance flags (no frame selection in v1):
    XrayPulses(dc).pulse_counts() vs counts           → "frames != x-ray pulses"
    XrayPulses(dc).is_constant_pattern()
    AGIPD1MQuadrantMotors(dc).positions(compressed=True) has > 1 entry → "quadrants moved"
    CalibrationData.from_correction(proposal, run, detector_name) → constants summary
```

---

## 7. Output schema (`writer.py`)

File: `{output_root}/r{run:04d}/saxs_first_pass.h5`. Per-frame datasets are chunked by one train
and compressed with gzip level 1.

```
/frames/trainId            (n,)       u8
/frames/reader_pulseId     (n,)       u8     pulse_id_coordinates(), as read
/frames/cellId             (n,)       u2
/frames/status             (n,)       u1     §8, initialised NOT_PROCESSED
/frames/signal             (n, npt)   f4     Σ c·x
/frames/normalization      (n, npt)   f4     Σ c·Ω over valid pixels
/frames/variance           (n, npt)   f4     Σ c²·x
/frames/photons_valid      (n,)       f4
/frames/n_bad_pixels       (n,)       u4
/frames/n_frame_specific   (n,)       u4
/frames/max_count          (n,)       u2
/trains/trainId, /trains/first, /trains/count, /trains/status
/q/centers                 (npt,)     f8     attrs: unit="nm^-1"
/operator/{data,rows,indptr,omega}           attrs: sha256
/masks/{cells, base_bad_packed, D, n_samples, static_bad_packed, D_static,
        bits_present, unexpected_bits}                attrs: sha256, static_sha256,
                                                     operator_sha256, sampled trains
/provenance                attrs: config JSON + hash, package versions, <pkg> commit, host,
                           n_workers, input-file sha256s, photon energy (config and XGM),
                           calibration constants summary, run-check flags, timing summary
```

Size: ≈ 2.8 GB for r0423 at npt = 500. The frame arrays stay `(n, npt)`: I4 is decided as
option (a), a static pixel mask, so there are no per-region sums.

**Resume.** A block is complete when all its frames have status ≠ NOT_PROCESSED. Rerunning with
the same config hash processes only incomplete blocks.

---

## 8. Status codes

| Code | Name | Meaning |
|---|---|---|
| 0 | OK | integrated |
| 1 | MISSING_MODULES | fewer than `min_modules` modules in the train |
| 2 | NO_FRAMES | zero frames in the train |
| 3 | LABEL_MISMATCH | reader train IDs or frame count disagree with the plan |
| 4 | DATA_CHECK_FAILED | non-integer dtype or negative counts |
| 5 | WORKER_ERROR | exception in the worker (message in the ledger) |
| 255 | NOT_PROCESSED | initial value |

---

## 9. Post hoc usage and DAMNIT wrapper

For any group G of OK frames:

```
I(q) = Σ_G S / Σ_G N
σ(q) = sqrt(Σ_G V) / Σ_G N
```

With a per-frame scale a_f (e.g. 1/I0 or 1/T): numerator Σ a_f·S_f, variance Σ a_f²·V_f,
denominator unchanged. XGM `pulse_energy()` is indexed by pulse index. Align it explicitly to
`reader_pulseId`/`cellId` before use.

DAMNIT context file (thin wrapper only):

```python
@Variable(title="SAXS first pass", cluster=True)
def saxs_first_pass(run, proposal: "meta#proposal", run_no: "meta#run_number"):
    from <pkg>.saxs.config import FirstPassConfig
    from <pkg>.saxs.run import run_first_pass
    return run_first_pass(FirstPassConfig(proposal=proposal, run=run_no))
```

The default `slurm_time` (2 h) covers ~5 min per run.

---

## 10. Phases and acceptance

**P1 — operator, sparse kernels, self-test** (no Maxwell needed)
- Unit tests on synthetic frames with a real-sized geometry (EXtra-geom test quad positions),
  at 0.5 %, 2 % and 5 % occupancy.
- Random static masks, per-frame flags, and frame/cell disagreement in both signs.
- Gate: S, N and V match pyFAI to < 1e-6 and the same bins are empty in both paths.

**P2 — masks**
- Seam and custom-mask construction (shape and convention asserts).
- Majority vote; unseen-cell fallback; exactness for frame flags differing from the cell in either
  direction; unexpected-bit flagging.

**P3 — plan, worker, writer on a mock run.** I4 is decided: option (a), a static lobe mask OR'd
into `static_bad`; the frame schema stays `(n, npt)`.
- Mock run: EXtra-data `write_file` with `AGIPDModule(raw=False)`, then rewrite `image/data` as
  int16 photon counts and `image/mask` as uint32 static + dynamic bits. Both chunked
  (1, 512, 128), shuffle + gzip.
- Test cases:
  - dropped train;
  - train with < 16 modules;
  - zero-frame train;
  - worker exception → WORKER_ERROR;
  - killed worker → NOT_PROCESSED and raise;
  - resume completes only missing blocks;
  - config-hash mismatch refused.
- Invariants: rows written by label; no NaN anywhere.

**P4 — on-node acceptance, r0423**
- Full run with 36 workers; self-test passes.
- For 3 trains, pooled I(q) matches a dense pyFAI reference (engine with explicit `variance`)
  to < 1e-6.
- Per-stage timing against the §2 budget; wall time ≤ ~6 min; every non-OK status accounted for.

**P5 — DAMNIT integration**
- Cluster variable on r0423 and r0426 via `context_python`.
- Returns (trainId, q) pooled I(q); fails loudly if incomplete.

**P6 — hyperthreading**
- Compare 36 vs 72 workers end to end.
- Keep 72 only for a throughput gain ≥ 15 % with no memory pressure.

---

## 11. Integrator open items

- **I1 — Lit-frame selection.** v1 integrates all frames and flags a mismatch between frames and
  X-ray pulses. Selection logic waits for the LITFRM source and keys from `lsxfel`.
- **I2 — Polarisation.** Not applied in v1.
  - Effect ≤ 5.4e-4 at q_max = 1.07 nm⁻¹ before azimuthal averaging.
  - Needs the detector-frame ↔ lab-horizontal orientation and the polarisation factor confirmed
    (CLAUDE.md task 5).
  - Can be applied post hoc as a per-(cell, q-bin) factor from the base masks, without
    reprocessing.
- **I3 — Bulk chunk reader** (for the XCCA pass): one `chunk_iter` per dataset, contiguous `pread`,
  ISA-L inflate, `zlib_into` unshuffle.
  - In-memory decode floor: 0.9 ms (data) and 1.6 ms (mask) per frame.
  - Build only with its own bit-exact benchmark measuring CPU and wall time.
- **I4 — Anisotropic low-q lobe** (r0423; CLAUDE.md task 8). **Decided: option (a), a static
  pixel mask.** `cfg.lobe_mask_file` points at `/gpfs/exfel/u/usr/MID/202601/p010400/masks/mask_2026-09-08_AGIPD_SAXS.npy`;
  `masks.build_static_bad` OR's it into `static_bad` exactly as it does the ASIC seams and the
  custom mask, and its sha256 goes into provenance.
  - *Why (a).* The mask is defined in pixel space, so it needs no φ range and no re-derivation of
    the old integrator's φ ≈ 278–330°, and it is unaffected by the unresolved geometry question
    (CLAUDE.md task 4). Storage and the `(n, npt)` frame schema are unchanged, so P3 is unblocked.
  - *What it costs.* The lobe pixels are absent from the first-pass sums. Changing the exclusion
    later means reprocessing, and the lobe's per-frame amplitude is **not** recoverable from this
    pass — the lobe-amplitude vs droplet-volume correlation (CLAUDE.md task 8) needs its own pass.
  - *Rejected: (b) per-region sums.* K = 2 regions would keep the lobe recoverable post hoc, but at
    ×K storage (≈ 5.6 GB for r0423 at npt = 500) and a wider schema, for an analysis that is not
    part of the first pass.

---

## 12. APIs to verify against installed source before use

- **EXtra-data:**
  - `open_run(proposal, run, data="proc")`
  - `AGIPD1M(dc, detector_name=, min_modules=)`
  - `.select_trains(by_id[...])`
  - `.frame_counts`
  - `det["image.*"].ndarray(decompress_threads=)`
  - `.train_id_coordinates()`, `.pulse_id_coordinates()`, `.cell_id_coordinates()`
- **EXtra:**
  - `XrayPulses(dc).pulse_counts()`, `.is_constant_pattern()`
  - `XGM(dc).photon_energy_by_train()`
  - `CalibrationData.from_correction(proposal, run, detector_name)`
  - `AGIPD1MQuadrantMotors(dc).positions(compressed=True)`
  - `extra.calibration.BadPixels`
- **EXtra-geom:**
  - `AGIPD_1MGeometry.from_crystfel_geom()`
  - `.to_pyfai_detector()`
  - `agipd_asic_seams()`
- **pyFAI 2026.5.0:**
  - `integrate1d(..., method=tuple)`
  - `res.method.{split,algo,impl}_lower`
  - `ai.engines[...].engine.{lut, bin_centers, nnz, integrate_ng(weights, variance=, solidangle=)}`
  - result fields `signal`, `normalization`, `variance`
  - `ai.solidAngleArray(shape)`
- **DAMNIT:**
  - `@Variable(cluster=True)`
  - `meta#proposal`, `meta#run_number`
  - `damnit db-config context_python`

---

## 13. Do not port from `analysis_helpers.integrate_run`

- **Method string `"csc"`** — resolves to no-split NumPy histogram.
- **The OpenCL path** — silently falls back to the slowest CPU engine.
- **Three geometries per run:**
  - CPU workers ignore `beamcenter`;
  - the q axis comes from another integrator;
  - `setFit2D` is applied on re-shifted corners.
- **`reshape(unstacked_shape)` with -1 plus offset arithmetic.**
- **In-place `data /= i0` before integration.**
- **`astype(float32)` + NaN masking of int16 data.**
- **Unreliable concurrency:**
  - pasha fork from a non-main thread;
  - unchecked futures;
  - `mp.Queue.empty()` loop condition;
  - silent `join(timeout)`.
- **`pulseId = np.arange(n_pulses)`.**
- **`xgm.wavelength()` mid-run.**
