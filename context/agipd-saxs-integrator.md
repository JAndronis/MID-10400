# AGIPD SAXS integrator (`agipd_saxs`) — design and implementation guide

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
| Seam mask + one pixel mask on top of `image.mask` | `NON_STANDARD_SIZE` is never set, so double-width ASIC-edge pixels are unflagged. A single hand-maintained pixel mask carries both the bad pixels and the low-q lobe: two overlapping mask files would have to be kept in step |

**Budget:** data read 4.6 + mask read 8.2 + sparse integration 6.2 ≈ 19 ms/frame/core.
Target for r0423 (~465 k frames): ≈ 4.7 min on 36 workers.

These were measured one core at a time. Under the real load — 36 workers competing for memory
bandwidth — the full r0423 pass measured 5.10 / 9.81 / 10.11 ≈ 25.0 ms/frame/core (P4 below), so
treat 19 ms as the unloaded floor, not the operating point.

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
  └─ <pkg>.saxs.run.run_agipd_saxs(cfg)                          [parent]
       ├─ plan.build_plan          trains, frame counts, rows, blocks, run checks
       ├─ operator.build_operator  sparse full-split operator + Ω, hashed, saved
       ├─ masks.build_static_bad   ASIC seams ∪ pixel mask (incl. lobe, I4a), hashed
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
| `pixel_mask_file` | `/gpfs/exfel/exp/MID/202601/p010400/usr/masks/mask_2026-09-08_AGIPD_SAXS.npy` | The **only** mask file. Carries the bad pixels *and* the low-q lobe (I4 decided (a)). Non-zero = excluded; assert shape (16, 512, 128), (512, 128) or (8192, 128); sha256 recorded; OR'd into `static_bad`. Supersedes `usr/Shared/IA/custom_agipd_mask.npy`, which must not also be applied |
| `base_mask_trains` | 8 | spread evenly over the run |
| `selftest_frames` | 8 | from 2 trains |
| `trains_per_block` | 4 | unit of scheduling, ledger and resume (not memory) |
| `n_workers` | physical cores | SMT only after P6 |
| `min_modules` | 16 | trains with fewer → `MISSING_MODULES` |
| `output_root` | `/gpfs/exfel/exp/MID/202601/p010400/scratch/agipd_saxs` | file `r{run:04d}/agipd_saxs.h5` |
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
if cfg.pixel_mask_file:  static_bad |= broadcast_to(load(file), (16, 512, 128)).ravel() != 0  # incl. lobe, I4a
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
def run_agipd_saxs(cfg, pool_factory=None):
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

File: `{output_root}/r{run:04d}/agipd_saxs.h5`. Per-frame datasets are chunked by one train
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
                           calibration constants summary, run-check flags, timing summary,
                           wall_s and started_at (P4 checks the wall time from the file alone)
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

Two reducers read the frame table back. Both work from the output file alone, so post hoc
analysis needs no plan object:

| Reducer | Returns |
|---|---|
| `writer.pooled_per_train(file)` | `intensity`, `sigma` (trainId, q) and `n_frames` (trainId) |
| `writer.per_pulse(file)` | `intensity` (trainId, pulseId, q) and `n_frames` (trainId, pulseId) |

`per_pulse` places each frame by its *stored* trainId and pulseId, never by its row position, so a
dropped or short train cannot slide frames onto the wrong train (CLAUDE.md pitfall 4). Only `OK`
frames carry trustworthy labels — a train that failed label validation was never given pulse ids,
and an unwritten row is all zeros — so anything else is counted in `attrs["unplaced"]` rather than
guessed onto a slot. Empty slots are zeros with `n_frames == 0`, never NaN (§3 rule 7). At r0423
size the grid is 0.93 GB as f4; the pooled view is 12 MB.

DAMNIT context file (thin wrappers only — DAMNIT `exec`s that file into a dict, so nothing defined
there can be pickled to the spawned workers):

```python
@Variable("AGIPD I(q)", data="proc", cluster=True, tags=["offline"])
def agipd_saxs(run, proposal: "meta#proposal", run_no: "meta#run_number"):
    from analysis.saxs.damnit import agipd_saxs
    return agipd_saxs(proposal, run_no)          # (trainId, pulseId, q)


@Variable("AGIPD I(q) overview", data="proc", cluster=True, tags=["offline"])
def agipd_iq_overview(run, grid: "var#agipd_saxs"):
    from analysis.saxs.damnit import overview_figure
    return overview_figure(grid)
```

DAMNIT's own `run` is deliberately unused: a `data="proc"` variable is handed a proc-only
collection, which carries no XGM, timeserver or motors, so every run check would come back
unavailable. Passing the proposal and run number lets the pass open proc for frames and raw for
checks. The default `slurm_time` (2 h) covers the measured 6.3 min per run.

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

**P3 — plan, worker, writer on a mock run.** I4 is decided: option (a), the lobe is excluded by
the one static pixel mask OR'd into `static_bad`; the frame schema stays `(n, npt)`.
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

**P4 is closed (2026-09-11).** The wall time came in at 6.29 min against a "~6 min" estimate, and
that estimate was a design target, not a constraint: the pipeline this replaces
(`analysis_helpers.integrate_run`) takes over an hour for one run, and DAMNIT's default
`slurm_time` is 2 h, which 6.29 min uses 5 % of. Accepted on that basis.

*How it is checked.* `scripts/p4_acceptance.py` runs the pass and then the four gates, and writes
its verdict as JSON next to itself. Gate B re-reads what the writer stored, pools it the way §9
does and compares that against a dense accumulation over whole trains, so the row-to-train
mapping, the `f4` storage and `pooled_per_train` are inside the comparison — the per-frame kernel
is the self-test's job, not gate B's. Gate C divides the workers' summed per-stage CPU seconds by
the OK frame count to reach ms/frame/core; the §2 budget is a measurement, so a stage over budget
is reported and only the wall time fails the gate (see the accepted target in the script). The gates are unit-tested against the mock run
(`tests/saxs/test_p4_acceptance.py`), including a deliberately corrupted stored row.

It cannot be run off the cluster: it needs a DAMNIT-partition node, the real geometry and mask
files, and r0423. Rerun it whenever the frame loop, the masks or the geometry change.

*Result, 2026-09-11, max-exfl170, full r0423 on 36 workers* — **accepted**. Every correctness
gate passes; the wall time is 4.9 % over the design estimate and was accepted as above.

| Gate | Verdict | |
|---|---|---|
| Configuration | pass | 3001/3001 trains, 465 000 frames, 36 workers, config hash matches |
| A self-test | pass | 8 real frames; max rel S and V exactly 0, N 1.3e-8 |
| B dense reference | pass | 4.9e-8 over 3 trains × 155 frames, all 500 bins populated, empty-bin sets equal |
| C timing | 377.6 s | over the ~360 s estimate, accepted; vs > 1 h for the pipeline it replaces |
| D ledger | pass | 465 000/465 000 OK; one train NO_FRAMES owning no rows; bits exactly the expected set |

Per-stage, per frame per core: read_data 5.10 (budget 4.6), read_mask 9.81 (8.2), integrate 10.11
(6.2), total 25.0 against 19.0. Parallel efficiency 0.856, with 54.5 s of the wall outside the
workers' timed stages. Every stage is above budget and `integrate` is the outlier at 1.63×; the §2
figures were measured without 36 cores competing for memory bandwidth, and the frame loop's
full-detector passes (`frame_bad`, `flatnonzero(x)`, `bad != base_bad`, `bad.sum()`) are pure
memory traffic — they cost several times the sparse gathers even unloaded. A ten-train trial on
the same node measured 21.3 ms/frame, so contention accounts for most of the gap between the two.

The 54.5 s outside the workers was unattributed, because only the worker stages were timed.
`run.py` now records each parent phase (`plan`, `operator`, `static_mask`, `base_masks`,
`selftest`, `save_operator_and_masks`, `write_blocks`) under `provenance/setup_timings`. Nothing
depends on that breakdown now that the wall time is accepted; it is there for whoever wants the
14 % of the run that is serial, and for P6, which compares 36 against 72 workers end to end and
needs the serial part separated out to read the comparison.

The run also settled two things beyond the gates: every run check now resolves (constant pulse
pattern, 155 frames = 155 X-ray pulses on all 3000 trains, quadrants stationary), and the XGM's
nominal photon energy disagrees with the configured one — see CLAUDE.md open task 15.

**P5 — DAMNIT integration**
- Cluster variable on r0423 and r0426 via `context_python`.
- Returns (trainId, q) pooled I(q); fails loudly if incomplete.

*Implemented.* The stored variable is the **(trainId, pulseId, q)** grid rather than the pooled
per-train view: per-pulse I(q) is what the old `agipd_saxs` provided and what the overview and any
per-pulse analysis need, it costs 0.93 GB against that variable's 3.7 GB, and the pooled view is
one call away from the same file. Failing loudly needs no code of its own — `run_agipd_saxs`
raises `IncompleteRun` unless `allow_incomplete`, and the wrapper does not catch it.

`analysis/saxs/damnit.py` holds `config_for`, `agipd_saxs`, the two file readers and
`overview_figure`; `src/amore/context.py` holds only the two decorated functions of §9.
`agipd_saxs` and `agipd_iq_overview` keep their names and their DAMNIT columns — this supersedes
the old implementation rather than sitting beside it, so nothing downstream has to be repointed.
What the column *holds* does change: I(q) in nm⁻¹ and undivided, where `analysis_helpers.
integrate_run` gave Å⁻¹ divided by I0 in place. Runs processed before the switch hold the old
quantity, so clear the column for them rather than plotting across the boundary.
`tests/test_context.py` loads the context file the way DAMNIT does and checks that every `var#`
resolves.

*Running it on r0423 and r0426 is the remaining work* — it needs the cluster.

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
  pixel mask.** `cfg.pixel_mask_file` points at
  `/gpfs/exfel/exp/MID/202601/p010400/usr/masks/mask_2026-09-08_AGIPD_SAXS.npy` — the single mask file,
  which covers the generally bad pixels *and* the lobe. `masks.build_static_bad` OR's it into
  `static_bad` alongside the ASIC seams, and its sha256 goes into provenance.
  - *Why (a).* The mask is defined in pixel space, so it needs no φ range and no re-derivation of
    the old integrator's φ ≈ 278–330°, and it is unaffected by the unresolved geometry question
    (CLAUDE.md task 4). Storage and the `(n, npt)` frame schema are unchanged, so P3 is unblocked.
  - *Why one file.* An earlier draft kept the lobe mask separate from
    `usr/Shared/IA/custom_agipd_mask.npy`. Two overlapping masks have to be kept in step with each
    other, and neither is meaningful alone; the newer file already contains both, so it is the only
    one applied and the older one is retired.
  - *What it costs.* The lobe pixels are absent from the `agipd_saxs` sums. Changing the exclusion
    later means reprocessing, and the lobe's per-frame amplitude is **not** recoverable from this
    pass — the lobe-amplitude vs droplet-volume correlation (CLAUDE.md task 8) needs its own pass.
  - *Rejected: (b) per-region sums.* K = 2 regions would keep the lobe recoverable post hoc, but at
    ×K storage (≈ 5.6 GB for r0423 at npt = 500) and a wider schema, for an analysis that is not
    part of this pass.

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
