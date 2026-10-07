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
- WAXS JUNGFRAU (CLAUDE.md open task 2) — §14 maps which of this design carries over;
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
4. **Geometry.** The geometry object is the single source of detector *positions*. Where the beam
   sits on it is `cfg.beam_center`, and the two conventions are not interchangeable:
   - `cfg.beam_center is None` → `AzimuthalIntegrator(detector=geom.to_pyfai_detector(), dist=sdd,
     wavelength=λ)`, PONI = 0 at the geometry origin.
   - `cfg.beam_center == (px, py)` → `detector.set_pixel_corners(geom.to_distortion_array())` and
     then `ai.setFit2D(sdd_mm, px, py)`. **Both steps or neither.** The two corner arrays have
     different origins — the distortion array's is the corner of the assembled bounding box, so its
     coordinates are all positive; `to_pyfai_detector()`'s is the geometry origin — so `setFit2D`
     on the unreplaced array places the beam where neither convention means. This is what
     `extra_speckle.setup.configuration.ConfigSAXS` does and what the agreed `(px, py)` were
     derived against; `test_operator.py` cross-checks our construction against it.
   - `setFit2D` also sets `ai.dist`, so `sdd` goes through it in mm rather than being set twice.
   - Still no PONI files.
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
| `pixel_sums.py` | Per-pixel window sums: window spec, per-train kernel, writer, reader (§15) |
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
| `sdd_m` | 7.531 | the agreed beam centre came with 7531.5962 mm, 0.6 mm / 0.008 % away; unresolved |
| `beam_center_px` | 607.4598195630211 | Fit2D `centerX`, in the `to_distortion_array()` frame — see §3 rule 4 |
| `beam_center_py` | 672.076693118667 | Fit2D `centerY`, same frame. Set with `px` or not at all; both `None` means PONI = 0 |
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
| `pixel_sum_trains` | 10 | trains per window of per-pixel sums (§15); `None` writes none. Operational for the frame table, hashed by the window sums; needs `min_modules == 16` |
| `pixel_sums_root` | `/gpfs/exfel/exp/MID/202601/p010400/usr/cached_files/agipd_pixel_sums` | file `r{run:04d}/agipd_pixel_sums.h5`; not scratch, which does not survive the move to tape |

---

## 6. Algorithms

`SHAPE = (16*512, 128)`, `NPIX = SHAPE[0]*SHAPE[1]`. The flattened pixel order is
module × ss × fs. It equals `ndarray()[:, k].reshape(-1)` when all 16 modules are present in
sorted order.

### 6.1 Operator (`operator.py`, parent, once per job)

```
geom  = AGIPD_1MGeometry.from_crystfel_geom(cfg.geometry_file)
det   = geom.to_pyfai_detector()
if cfg.beam_center is None:                        # PONI = 0 at the geometry origin
    ai = AzimuthalIntegrator(detector=det, dist=cfg.sdd_m, wavelength=λ)
else:                                              # §3 rule 4: both steps together
    det.set_pixel_corners(geom.to_distortion_array())
    ai = AzimuthalIntegrator(detector=det, wavelength=λ)
    ai.setFit2D(cfg.sdd_m * 1e3, *cfg.beam_center) # also sets ai.dist
    assert isclose(ai.dist, cfg.sdd_m)
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
| `writer.pooled_per_train(file)` | a Dataset: `intensity`, `sigma` (trainId, q) and `n_frames` (trainId) |
| `writer.per_pulse(file)` | a **DataArray** `intensity` (trainId, pulseId, q), with `n_frames` (trainId, pulseId) as a non-dimension coordinate |

`per_pulse` returns a DataArray, not a Dataset, for two reasons that both come from DAMNIT: it is
what `analysis_helpers.integrate_run` returned, and DAMNIT renders a 3-D DataArray in the table as
`float32: (3000, 155, 500)` while a Dataset shows only `Dataset (930.49MB)`. Carrying `n_frames`
as a coordinate keeps the companion array without turning it into a Dataset; it survives the
netCDF round trip DAMNIT stores it through.

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

**P1 operator, sparse kernels, self-test — done.** Synthetic frames at 0.5/2/5 % occupancy on a
real-sized geometry, random static masks and frame/cell disagreement in both signs; S, N and V
match pyFAI to < 1e-6 with the same bins empty in both paths. `tests/saxs/test_operator.py`,
`test_sparse.py`, `test_selftest.py`.

**P2 masks — done.** Seam and pixel-mask construction, majority vote, unseen-cell fallback,
exactness for frame flags differing from the cell either way, unexpected-bit flagging.
`tests/saxs/test_masks.py`.

**P3 plan, worker, writer — done**, on a mock run (EXtra-data `write_file` with
`AGIPDModule(raw=False)`, `image/data` rewritten as int16 counts and `image/mask` as uint32, both
chunked (1, 512, 128), shuffle + gzip). Covered: dropped train, < 16 modules, zero-frame train,
worker exception → WORKER_ERROR, killed worker → NOT_PROCESSED and raise, resume completes only
missing blocks, config-hash mismatch refused. Invariants: rows written by label, no NaN anywhere.

**P4 on-node acceptance, r0423 — accepted 2026-09-11** (max-exfl170, full run, 36 workers).

| Gate | Verdict | |
|---|---|---|
| Configuration | pass | 3001/3001 trains, 465 000 frames, 36 workers, config hash matches |
| A self-test | pass | 8 real frames; max rel S and V exactly 0, N 1.3e-8 |
| B dense reference | pass | 4.9e-8 over 3 trains × 155 frames, all 500 bins populated, empty-bin sets equal |
| C timing | 377.6 s | over the ~360 s estimate, accepted; vs > 1 h for the pipeline it replaces |
| D ledger | pass | 465 000/465 000 OK; one train NO_FRAMES owning no rows; bits exactly the expected set |

The wall time was a design target, not a constraint: `analysis_helpers.integrate_run` takes over an
hour for one run and DAMNIT's default `slurm_time` is 2 h, of which 6.29 min is 5 %.

*Why every stage came in over the §2 budget.* Per frame per core: read_data 5.10 (budget 4.6),
read_mask 9.81 (8.2), integrate 10.11 (6.2), total 25.0 against 19.0; `integrate` is the outlier at
1.63×. The §2 figures were measured without 36 cores competing for memory bandwidth, and the frame
loop's full-detector passes (`frame_bad`, `flatnonzero(x)`, `bad != base_bad`, `bad.sum()`) are
pure memory traffic — several times the cost of the sparse gathers even unloaded. A ten-train trial
on the same node measured 21.3 ms/frame, so contention accounts for most of the gap. **Treat 19 ms
as the unloaded floor, not the operating point** — this is the trap to avoid when budgeting WAXS.

Parallel efficiency 0.856, with 54.5 s (14 %) of the wall outside the workers' timed stages.
`run.py` records each parent phase (`plan`, `operator`, `static_mask`, `base_masks`, `selftest`,
`save_operator_and_masks`, `write_blocks`) under `provenance/setup_timings` so P6 can separate the
serial part; nothing else depends on it.

`scripts/p4_acceptance.py` runs the pass and the gates and writes its verdict as JSON beside
itself. The gates are **not** unit-tested: the test suite covers `src/` only, so a change to the
gates is caught by running the script on a node, not by pytest. Gate B re-reads what the writer stored and pools it
the way §9 does, so the row-to-train map, the `f4` storage and `pooled_per_train` are all inside
the comparison; the per-frame kernel is the self-test's job. It needs a DAMNIT-partition node, the
real geometry and mask files, and r0423. **Rerun it whenever the frame loop, the masks or the
geometry change** — including the 2026-09-13 beam-centre change, re-gated 2026-10-07 on max-cfel029
(EPYC 9374F): all gates pass, gate B 4.5e-8, wall 157–179 s on 36 workers (§15.5).

The run also settled that every run check resolves (constant pulse pattern, 155 frames = 155 X-ray
pulses on all 3000 trains, quadrants stationary) and surfaced the XGM photon-energy disagreement
(CLAUDE.md open task 15).

**P5 DAMNIT integration — implemented; not yet run on the cluster.** `agipd_saxs` and
`agipd_iq_overview` keep their names and columns, backed by `analysis/saxs/damnit.py`
(`config_for`, `agipd_saxs`, the two file readers, and `overview_figure`, which draws the two
maps with `extra.utils.imshow2` — the helper the old overview used, so the colour bars and the
coordinate-labelled axes match it); `src/amore/context.py` holds
only the two decorated wrappers of §9. Two choices worth keeping:

- The stored variable is the **(trainId, pulseId, q)** DataArray, not the pooled per-train view:
  per-pulse I(q) is what the old `agipd_saxs` gave and what the overview needs, it costs 0.93 GB
  against 3.7 GB, and the pooled view is one call away from the same file.
- Failing loudly needed no code — `run_agipd_saxs` raises `IncompleteRun` unless
  `allow_incomplete`, and the wrapper does not catch it.

What the column *holds* changed twice: from Å⁻¹ I0-divided to nm⁻¹ undivided (P5), then the q axis
moved again with the beam centre (2026-09-13). Clear the column and reprocess rather than plotting
across either boundary. `tests/test_context.py` loads the context file the way DAMNIT does and
checks every `var#` resolves.

*Remaining:* run it on r0423 and r0426. Needs the cluster.

**P6 hyperthreading — open.** Compare 36 vs 72 workers end to end; keep 72 only for a throughput
gain ≥ 15 % with no memory pressure.

---

## 11. Integrator open items

- **I1 — Lit-frame selection.** v1 integrates all frames and flags a mismatch between frames and
  X-ray pulses. Selection waits for the LITFRM source and keys from `lsxfel`.
- **I2 — Polarisation.** Not applied in v1. Effect ≤ 5.4e-4 at q_max = 1.07 nm⁻¹ before azimuthal
  averaging. Needs the detector-frame ↔ lab-horizontal orientation and the factor confirmed
  (CLAUDE.md task 5). Applicable post hoc as a per-(cell, q-bin) factor from the base masks,
  without reprocessing.
- **I3 — Bulk chunk reader** (for the XCCA pass): one `chunk_iter` per dataset, contiguous `pread`,
  ISA-L inflate, `zlib_into` unshuffle. In-memory decode floor 0.9 ms (data) and 1.6 ms (mask) per
  frame. Build only with its own bit-exact benchmark measuring CPU and wall time.
- **I4 — Anisotropic low-q lobe. Decided: option (a), a static pixel mask.** The single
  `cfg.pixel_mask_file` carries the bad pixels *and* the lobe, OR'd into `static_bad` by
  `masks.build_static_bad`; no φ range and no re-derivation of the old integrator's φ ≈ 278–330°,
  and it is unaffected by the unresolved geometry question. Rejected: per-region sums, which would
  keep the lobe recoverable at ×K storage (≈ 5.6 GB for r0423) and a wider schema, for an analysis
  that is not part of this pass. *Cost:* the lobe pixels are absent from the sums, so the
  lobe-amplitude vs droplet-volume correlation (CLAUDE.md task 8) needs its own pass, and changing
  the exclusion later means reprocessing.

---

## 12. APIs to verify against installed source before use

- **EXtra-data:** `open_run(proposal, run, data="proc")`;
  `AGIPD1M(dc, detector_name=, min_modules=)`; `.select_trains(by_id[...])`; `.frame_counts`;
  `det["image.*"].ndarray(decompress_threads=)`; `.train_id_coordinates()`;
  `.pulse_id_coordinates()`; `.cell_id_coordinates()`
- **EXtra:** `XrayPulses(dc).pulse_counts()`; `.is_constant_pattern()`;
  `XGM(dc).photon_energy_by_train()`;
  `CalibrationData.from_correction(proposal, run, detector_name)`;
  `AGIPD1MQuadrantMotors(dc).positions(compressed=True)`; `extra.calibration.BadPixels`
- **EXtra-geom:** `AGIPD_1MGeometry.from_crystfel_geom()`; `.to_pyfai_detector()`;
  `.to_distortion_array()`; `agipd_asic_seams()`
- **pyFAI 2026.5.0:** `integrate1d(..., method=tuple)`; `res.method.{split,algo,impl}_lower`;
  `ai.engines[...].engine.{lut, bin_centers, nnz, integrate_ng(weights, variance=, solidangle=)}`;
  result fields `signal`, `normalization`, `variance`; `ai.solidAngleArray(shape)`;
  `ai.setFit2D(directDist_mm, centerX, centerY)`
- **DAMNIT:** `@Variable(cluster=True)`; `meta#proposal`; `meta#run_number`;
  `damnit db-config context_python`

---

## 13. Do not port from `analysis_helpers.integrate_run`

CLAUDE.md pitfalls 1, 2, 3, 7, 13–16 carry the general cases. Specific to this module:

- **Three geometries per run** — CPU workers ignore `beamcenter`, the q axis comes from another
  integrator, and `setFit2D` is applied on re-shifted corners.
- **`reshape(unstacked_shape)` with -1 plus offset arithmetic** (CLAUDE.md pitfall 4).
- **In-place `data /= i0` before integration** — irreversible; normalise the stored sums (§9).
- **`astype(float32)` + NaN masking of int16 data** — the mask belongs in the operator's
  denominator, not in the values.
- **`pulseId = np.arange(n_pulses)`** — use `XrayPulses` / reader coordinates.
- **`xgm.wavelength()` mid-run** — raises if not constant; use `*_by_train()`.
- **Unreliable concurrency** — pasha forked from a non-main thread, unchecked futures, an
  `mp.Queue.empty()` loop condition, a silent `join(timeout)`. Failures end as NaN rows.

---

## 14. What carries over to the WAXS integrator (CLAUDE.md open task 2)

Nothing here is a WAXS design decision — the JUNGFRAU-500K data have **not** been inspected
(dtype, compression, frames per train, gain and mask bits, value distribution all unknown). This
records which parts of this design are detector-agnostic and which rest on an AGIPD fact that has
to be re-measured before it can be reused.

| Part | Carries over? |
|---|---|
| Sufficient statistics S, N, V and the pooling of §9 | **Yes, if** the data are counts. `V = Σc²·x` is the Poisson variance of integer photons; on float ADU it is not a variance at all and the error model has to be re-derived |
| Status ledger (§8), rows by label, no NaN sentinels | **Yes.** Detector-independent |
| Output schema (§7), resume, provenance, config hash | **Yes**, but two JUNGFRAUs means either a detector axis or one file each — decide before writing, it is a schema change afterwards |
| Plan / worker / writer split, spawned pool, single writer | **Yes.** `plan.py` needs a JUNGFRAU frame-count source instead of `AGIPD1M.frame_counts` |
| Sparse integration over photon hits (§6.2, §6.4) | **Only if sparse.** The 6.2 vs 19.0 ms/frame win comes from 0.7–1.8 % nonzero. Measure occupancy first (benchmark stage 1); a dense WAXS detector wants dense pyFAI per frame and this whole kernel is the wrong shape |
| Per-cell base mask (§6.3) | **Unclear.** It exists because AGIPD's static bits are per memory cell. Whether JUNGFRAU has an analogous per-cell structure is unmeasured; if not, `static_bad` plus the per-frame correction is the whole story and `masks.py` simplifies |
| Operator build (§6.1) and §3 rule 4 | **Partly.** The full-split assertion and the beam-centre discipline carry. The geometry does not: `usr/geometry` has no JUNGFRAU `.geom`, but `masks_and_calibration/data/calibration/` holds `jf1.poni`, `jf2.poni` and per-detector masks (`jf1_mask.edf`, `jf2_mask.edf`). PONI files are forbidden for AGIPD by rule 4 because a geometry object exists; for JUNGFRAU they may be the only source, so rule 4 needs an explicit JUNGFRAU clause rather than being quietly broken |
| `EXPECTED_BITS`, `agipd_asic_seams()`, `min_modules=16`, `mask_bits` blanket | **No.** All AGIPD-specific. The blanket `mask_bits` is only correct while every bit present marks an unusable pixel (CLAUDE.md pitfall 6) — re-establish that per detector |
| The 19 ms/frame budget | **No.** Re-measure. And see P4: budget under load, not one core at a time |

First step, per CLAUDE.md open task 2: adapt benchmark stages 1 and 7 to JUNGFRAU and get the
value distribution, occupancy, dtype and gain/mask semantics. The sparse-vs-dense fork and the
error model both hang on that measurement, so it comes before any spec.

---

## 15. Per-pixel window sums (`pixel_sums.py`)

**Why.** Low-q studies (stray-light lobes, droplet shadow, slice-wise buffer subtraction, mask
optimisation) need per-pixel photon sums. `analysis.cache.detector_sums` re-reads proc for them,
~80 CPU-s per 50-train block. The pass already decompresses every frame once, and proc moves to
tape (memory `storage-tiers-red-box`), after which these sums are the only 2D record of a run. They
also keep the 1D result recomputable: with the default `mask_bits`, over the frames of one window

```
Σ_f S_f = Σ_{j ∉ static} c_j · counts_j      Σ_f V_f = Σ_{j ∉ static} c_j² · counts_j
Σ_f N_f = Σ_{j ∉ static} c_j · Ω_j · valid_frames_j
```

so window-pooled I(q) can be rebuilt under any geometry, beam centre or photon energy (CLAUDE.md
open tasks 4 and 15) without proc.

**Decided 2026-10-07 (user):** window length 10 trains; a separate file; stored under
`usr/cached_files`, not scratch, because scratch does not survive the move to tape; a sums-only
mode to backfill the runs whose 1D file already exists.

### 15.1 Definition

For each window, per pixel `(module, slow, fast)`:

- `counts` = Σ of `image.data` over the frames whose `image.mask` is 0 for that pixel;
- `valid_frames` = how many such frames.

All mask bits, independent of `cfg.mask_bits`; no static mask, no seams, no normalisation (the
stored data are raw, with metadata only). This is `analysis.cache.detector_sums` exactly, and every
window equals `detector_sums(run, <its member trains>)` bit for bit.

**Windows are train-id ranges**, anchored at the run's first train t₀ (the first entry of the plan's
train table): window k holds `t₀ + k·L ≤ trainId < t₀ + (k+1)·L`, `L = cfg.pixel_sum_trains`. A
window is therefore a fixed span of time, membership comes from the label (CLAUDE.md pitfall 4), and
a dropped or failed train never shifts a later window. The last window is partial and stored with
its range like any other.

**Which trains are summed.** A train is summed whole or not at all. It is summed when the worker's
label check passes (as for the frame table), `image.data` has an integer dtype and no value in the
train is negative. Otherwise it is left out and the train table records the `FrameStatus` that left
it out. Trains that own no rows (`MISSING_MODULES`, `NO_FRAMES`) keep their plan status. Window sums
need all 16 modules — a missing module would read as fill values that pass `mask == 0` — so the
config refuses `pixel_sum_trains` with `min_modules != 16`.

### 15.2 Where it accumulates

In the worker, per train, right after the label check, on the arrays already read (before the
frame loop, which does not modify them):

```
for m in range(16):
    valid = mask[m] == 0
    counts[m] += where(valid, data[m], 0).sum(0, dtype=int32)    # one train: < 2^31
    valid_frames[m] += valid.sum(0, dtype=int32)
```

Per module, so the temporaries are ~20 MB on top of a worker's ~1 GB. A worker returns, per
scheduling block, one partial per window it touched (int32 after a range check, ~8.4 MB pickled)
plus the trains it left out. The parent adds partials into int64 accumulators and writes a window
when the last scheduling block touching it has returned or failed; values are range-checked before
the int32 cast. A partial for a window already written (resume) is ignored.

### 15.3 File, hash and resume

`{pixel_sums_root}/r{run:04d}/agipd_pixel_sums.h5`, default root
`/gpfs/exfel/exp/MID/202601/p010400/usr/cached_files/agipd_pixel_sums`. When `run_agipd_saxs` is
given an `output_path`, the file goes beside it instead.

```
/windows/start_trainId   (W,)              u8   t₀ + k·L
/windows/n_trains        (W,)              u4   trains summed
/windows/n_frames        (W,)              u4   frames summed
/windows/written         (W,)              u1   1 once the window is final
/windows/counts          (W, 16, 512, 128) i4   chunks (1, 1, 512, 128), shuffle + gzip 1
/windows/valid_frames    (W, 16, 512, 128) i4   same
/trains/trainId          (T,)              u8   the plan's train table
/trains/window           (T,)              i4
/trains/status           (T,)              u1   OK = summed, else why not; NOT_PROCESSED until written
/provenance              attrs: config_hash, config, hash_fields, detector_name, window_trains,
                         window_origin_trainId, n_windows, definition, bits_present, host,
                         package_versions, timings, wall_s, status_summary, block_errors
```

Measured on two cached `detector_sums` files, int32 + shuffle + gzip-1 stores a window in
0.75–1.3 MB against 8.4 MB raw.

**Two hashes, each over its own numbers (CLAUDE.md pitfall 12).** `pixel_sum_trains` and
`pixel_sums_root` are operational for the frame table (`SAXS_OPERATIONAL_FIELDS`), so adding them
left every existing `agipd_saxs.h5` hash unchanged. The window sums' hash covers only the proposal,
the run, the *resolved* detector name, `min_modules` and `pixel_sum_trains` — not geometry, beam
centre, photon energy, `npt` or any mask, none of which changes a stored count.

**Never truncated by the pass.** `cfg.overwrite` recreates the frame table only. A window-sums file
whose hash differs is refused even with `overwrite=True`: once proc is on tape it cannot be rebuilt,
so removing one is a decision made by hand.

**Resume.** A written window is final. The blocks to process are those with `NOT_PROCESSED` frames
plus those touching an unwritten window; a block wanted only for its windows runs with integration
skipped (read and sum, no frame-table write). A killed job leaves its open windows unwritten, and a
rerun rebuilds them from scratch, never half-filled. A failed block's trains are recorded as
`WORKER_ERROR`, the window is written and the run raises `IncompleteRun`; but unlike the frame
table, where `WORKER_ERROR` stays until the rows are reset, such a window is **not** final. A worker
error says nothing about the data, so the next run reopens every window holding one and rebuilds it
from scratch (sums only for blocks the frame table already holds). That is also how a backfill
retries a transient read failure: run it again. `DATA_CHECK_FAILED` and `LABEL_MISMATCH` are
properties of the data and stay final. Writing a window updates only its own rows of
`/trains/status`, so a run that dies mid-rebuild leaves the file as it found it.

### 15.4 Entry points and reader

| Call | Does |
|---|---|
| `run_agipd_saxs(cfg)` | the pass; writes the window sums too unless `pixel_sum_trains is None` |
| `run_pixel_sums(cfg)` | sums only: plan, read, sum. Never opens `agipd_saxs.h5` and builds no operator, masks or self-test, so it is independent of that file's hash — all 459 existing ones were written with `npt=2000`, which a full pass at the default config refuses. For backfilling runs whose 1D file exists |
| `pixel_sums.window_table(path)` | one row per window: range start, trains, frames, written |
| `pixel_sums.window_sums(path, windows)` | the sum over those windows |
| `pixel_sums.detector_sums(run, train_ids=None, *, windows=None)` | the same Dataset `analysis.cache.detector_sums` returns. `train_ids` must be exactly a union of whole windows; otherwise it raises and names the windows that cover them, rather than summing more trains than asked |

Notebook switch: `cache.detector_sums(run, tids, label=…)` → `pixel_sums.detector_sums(run, tids)`.

### 15.5 Phases

- **S1 — code and tests.** `tests/saxs/test_pixel_sums.py` on the mock run: every window equals
  `cache.detector_sums` of its member trains on a run with a dropped train, a short-module train, a
  zero-frame train, a partial last window and scheduling blocks straddling windows; left-out trains
  (negative counts, worker error); resume after a killed pool; windows added to a complete frame
  table without reintegration; the sums-only mode writes the same data; the window-pooled S, N, V
  of the frame table equal the gather over the window sums (1e-6, the P4 gate-B tolerance); hashes;
  the reader. The frame table's datasets are byte-identical before and after.
- **S2 — on a node — done 2026-10-07** (Slurm 25172553, max-cfel029, EPYC 9374F in DAMNIT's
  `allcpu` partition, exclusive, 36 workers, r0423; every Gold-6140/6240 node was allocated). Four
  runs in order, JSONs `scripts/p4_acceptance_r0423_20261007T15*.json`:

  | Run | Wall | Worker ms/frame/core | `pixel_sums` | Parent `write_blocks` | Gates |
  |---|---|---|---|---|---|
  | sums off, cold | 178.7 s | 10.39 | — | 64.0 s | A 0 B C D pass |
  | sums on, warm | 174.7 s | 11.31 | 1.25 | 78.9 s | A 0 B C D **E E2** pass |
  | sums off, warm | 156.7 s | 9.99 | — | 64.7 s | A 0 B C D pass |
  | sums only, warm | 115.7 s | 7.99 | 1.25 | 15.8 s | **E** pass |

  The overhead is on − warm off: **+18.0 s wall (+11.5 %)**, +1.3 ms/frame/core in the workers (the
  other stages moved by < 0.07 ms), +14 s of parent write time. **Gate E:** all 300 non-empty
  windows of 301 equal the `cache._sum_job` reference exactly, and the three spot checks through
  `cache.detector_sums` itself; 3000 trains, 465 000 frames summed, one `NO_FRAMES` train.
  **Gate E2:** max relative difference 1.8e-8 (S), 3.5e-8 (N), 3.2e-8 (V) over 300 windows, all
  empty bins agree. The sums-only file is identical to the full pass's, dataset by dataset;
  280 MB for r0423, 0.93 MB per window. The absolute times are this node's, not P4's Gold-6140
  (377.6 s); the A/B is the measurement. Gate B passing on all three full runs also re-gates the
  2026-09-13 beam-centre change (4.5e-8).
- **S3 — backfill** of the runs whose 1D file exists, with `run_pixel_sums`, before proc is taped.
  Size, from the 459 existing 1D files: 1.49e8 frames (320 × r0423), 619 175 trains, ~62 100
  windows, so ~58 GB at the measured 0.93 MB per window. At the measured sums-only rate on the S2
  node that is ~10.7 h of pool time plus per-run start-up; cold reads cost ~14 % more there (cold
  vs warm sums-off). Not yet run.
