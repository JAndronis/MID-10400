# Performance and tooling: measured verdicts (open task 19)

Context file for CLAUDE.md open task 19, written in session S3 (2026-10-13). It records what was
measured, where, with which versions, and what was decided. Read `CLAUDE.md` first.

**State:**
- (c) SLURM helpers: done (`analysis.common.slurm`, `scripts/_common.py`).
- (d) `fan_out` progress lines: done.
- (a) pyFAI 2026.9: measured, **no speed gain, bit-identical results**. The lock is upgraded on
  branch `claude/s3-speed-tooling-*`; it merges only after the user's review.
- (b) virtual CXI: measured, **not adopted** (§3). No pipeline change.

---

## 1. Running on the cluster

### 1.1 The DAMNIT partition

| Fact | Source |
|---|---|
| DAMNIT submits this proposal's cluster variables to **`allcpu`**, `--clusters maxwell`, no `-c`, no `--exclusive` | `metameta.slurm_partition` in `usr/Shared/amore/runs.sqlite`; `damnit/backend/extraction_control.py` `_slurm_cluster_opts` (damnit 0.4.0) |
| `allcpu` is `select/linear`: a job gets **whole nodes** | `scontrol show config` |
| With `--cpus-per-task=36`, the job's affinity and cgroup were the **whole node** — 72 CPUs on a Gold 6140, 128 on an EPYC 9374F | probe jobs 25285874/5, 2026-10-13 |
| Every `allcpu` node class has 2 threads per core; classes run from 2 × 10 cores (E5-2640 v4, 40 CPUs) to 2 × 64 (EPYC 9554) | `sinfo -p allcpu -o "%c %z %f"` |

So a DAMNIT job runs on whatever class is free, and `physical_cores()` (pitfall 10) sizes the
pool to it: 20 workers on an E5-2640 node, 64 on a 9374F. Timings from DAMNIT jobs are
therefore not comparable with each other unless the host is checked; the provenance records it.

### 1.2 `analysis.common.slurm` and the script flags

Node scripts submit themselves:

```
python scripts/p4_acceptance.py --run 423 [426 …] -w 36 [-t 01:00:00] [--constraint 9374F]
                                [--after JOBID] [--dry-run | --local]
```

- Without `--local`, one job per run. The batch script calls the same script with the same
  options, the run alone and `--local`. Inside a job (`SLURM_JOB_ID` set) the script works in
  place.
- **`-w` is physical cores**: both the worker count and what the job asks for. The batch script
  requests `-c 2w`, because SLURM counts hardware threads. `-c 36` alone can land on a
  2 × 10-core node and put 36 workers on 20 cores (user decision 2026-10-13).
- Paths in the batch script are written as `/gpfs/exfel/exp/MID/202601/p010400/<sub>/…`
  (`prop_path`): resolved tier paths (`/gpfs/exfel/u/usr/…`) and private aliases (`~/p010400`)
  are rewritten, and a venv's `bin/python` keeps its name so that the venv is kept (pitfall 22).
- `export EXTRA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` before
  the command; `-N 1 -n 1`; never `srun` (pitfall 28).
- `--after JOBID` adds `--dependency=afterok:JOBID --kill-on-invalid-dep=yes`.
- `--local` with several runs: each run is tried, a failure is logged with its traceback, and the
  exit status is non-zero, naming the failed runs.
- Logs: `scratch/slurm_logs/<script>/slurm-<jobid>.out`.
- Argparse prefix matching is off in these scripts (`--dry` is refused), because the command line
  is rewritten by exact option name.
- Importing `analysis.common.slurm` imports numpy (through `analysis.common`); `_common` imports
  it lazily, after the scripts pin the thread count.

Adopted by `p4_acceptance.py`, `bench_pyfai.py`, `bench_io.py`. Not (yet) by `w1_facts.py`,
`w4_acceptance.py`, `w6_data_check.py`. `w1_facts` compares runs within one process (O4), so it
must stay one job for all runs.

### 1.3 Progress lines

`analysis.common.run.fan_out` logs one INFO line per tenth of the blocks on
`analysis.common.run`, for example (r0423, P4, 2026-10-13):

```
75/750 blocks (10 %), 46500/465000 frames, 1821 frames/s, elapsed 0:00:26, remaining ~0:03:50
```

A block counts when it is written or marked `WORKER_ERROR`. The clock starts when the pool is
requested, so worker start-up is included. The lines appear only if the process lets INFO from
`analysis` through (`_common.configure_logging()` does).

### 1.4 Benchmark venvs

Timing venvs are built **without the dev group**, as `scripts/shared_env.sh` builds the deploy
venv: the dev group's `zlib_into` switches EXtra-data to its threaded path, 2.1–3.2× slower on the
mask (CLAUDE.md read-performance table). For this study:

```
UV_PYTHON_INSTALL_DIR=usr/Software/uv-python UV_LINK_MODE=copy \
UV_PROJECT_ENVIRONMENT=scratch/s3_speed/venv-pyfai2026.X \
uv sync --locked --no-dev --extra euxfel --extra plotting --python <uv-python 3.12.11>
```

---

## 2. pyFAI 2026.5.0 → 2026.9.0

### 2.1 Verdict

**Upgrade: safe, and neither faster nor slower for anything this project runs.** Every array
compared is bit-identical. Every per-frame timing is within ±6 % (run-to-run noise). The
15–20× gain p010646 saw (`transfer_context/08`) was 2024.9 → 2026.9; this project was already
on 2026.5. The changelog's one numerical item, detector `orientation` as a pure re-indexing of
pixels, does not touch us: AGIPD (`extra_geom.to_pyfai_detector()`) and both JUNGFRAU PONIs
carry orientation 3, the default, in both versions.

**Merging the upgrade means a reprocess of every pass output.** S1's `is_complete` /
`check_resumable` (branch `claude/s1-damnit-integrity-*`, open task 17b) keeps pyFAI in its
version gate (`GATING_PACKAGES` in `analysis/common/writer.py`). The user decided this on
2026-10-13, with the bit-identity evidence above in hand. So once pyFAI 2026.9 is on main:
- every existing `agipd_saxs.h5` and `jungfrau_waxs_*.h5` reads "not complete";
- resuming one raises `OperatorMismatch`;
- every file must be reprocessed through `scripts/reprocess_safe.py`, never with `overwrite=True`.

This is intended. Schedule the merge together with the bulk reprocess that already follows S2
(merge order S1 → S3 → S2 → S4), not on its own.

### 2.2 Measurement

`scripts/bench_pyfai.py`, job 25287833, max-cfel030 (AMD EPYC 9374F), 2026-10-13. One process,
single-threaded, the two venvs run one after the other on the same r0423 frames: AGIPD
32 frames (16 from each of trains 2637696897–98), jf1 32 lit frames (4 trains). Median of 3
passes after a warm-up. JSON: `scripts/bench_pyfai_r0423_20261013T143944Z.json`.

| Path | 2026.5.0 | 2026.9.0 | ratio |
|---|---|---|---|
| AGIPD operator build (once per run) | 0.530 s | 0.522 s | 1.02 |
| AGIPD sparse kernel (the pass; no pyFAI inside — control) | 3.12 ms/frame | 3.13 | 1.00 |
| AGIPD dense pyFAI reference (self-test, gate B) | 8.55 ms/frame | 8.32 | 1.03 |
| JF operator build | 0.099 s | 0.091 s | 1.09 |
| **JF `integrate_frame`** (dense pyFAI, NaN mask; the WAXS hot loop) | **2.88 ms/frame** | **2.72** | 1.06 |
| JF the same through `mask=` (pitfall 17) | 17.4 ms/frame | 16.6 | 1.05 |
| AGIPD `integrate2d` 500 × 360, `("full","csr","cython")` | 12.7 ms/frame | 13.2 | 0.96 |
| … first call (builds the matrix) | 0.270 s | 0.269 s | 1.01 |
| AGIPD `integrate2d`, `("bbox","csr","cython")` | 12.7 ms/frame | 12.4 | 1.02 |
| AGIPD `integrate2d`, `("no","histogram","cython")` | 11.6 ms/frame | 11.6 | 1.00 |

These are one unloaded core. P4 measured AGIPD stages ~1.3× slower under 36 loaded workers
(`agipd-saxs-integrator.md` §2), so these numbers compare versions; they are not an operating
point. `integrate2d` at ~12 ms/frame single-threaded means a 2D pass over r0423 would cost about
as much as the 1D one does now; it is not a reason to avoid 2D for open task 4 or XCCA.

**Identity (26 arrays, `np.array_equal`):** the AGIPD operator's `coef`, `bins`, `indptr`, `omega`
and `q`; the dense reference's S, N, V; `intensity`, `radial`, `azimuthal`, `sum_signal`,
`sum_normalization` of `integrate2d` for all three methods, the χ axis included; the jf1
sums. All equal. Both operator sha256s are equal between versions (AGIPD and JF).

**Where 2026.9 does move numbers: detector orientation ≠ 3.** The unit-test fixture's jf1
PONI uses orientation 4, which deliberately makes it differ from jf2's. Its pinned output
digest (`tests/waxs/test_waxs_output_snapshot.py`) moved: per-frame sums by at most 4.1e-6
relative, pooled I(q) by 2.6e-7, with the same empty bins and no other dataset changed. The 2026.5
output reproduces the old pin exactly, so pyFAI is the only cause. The pin is updated. No real
geometry of this proposal uses orientation ≠ 3. A new PONI that does must be checked.

**P4 under 2026.9** (job 25287834, max-cfel030, 36 workers, r0423, 2026-10-13, output in
`scratch/s3_speed/p4/`): **all gates pass**. Self-test max rel S 0, N 1.48e-8, V 0; gate B
4.464e-8, the same as on 2026-10-07; 465 000/465 000 OK; window sums E and E2 pass. Wall 199.6 s
against 174.7 s on 2026-10-07 (max-cfel029, 2026.5). The workers' stages are the same within
noise (read_data 2.44 / 2.37, read_mask 4.68 / 4.38, integrate 3.27 / 3.31, pixel_sums
1.24 / 1.25 ms/frame/core), and the parallel efficiency is lower (0.753 / 0.836). The difference
is outside the integration, which does not call pyFAI. JSON:
`scripts/p4_acceptance_r0423_20261013T144648Z.json`.

**Imports and the DAMNIT context**, both versions: `extra_speckle`, `extra_speckle.xcca`,
`extra_speckle.saxs`, `extra.components`, `extra.calibration` import.
`extra.applications.xcca` does not import in either version: the toolbox is not in
euxfel-EXtra 2026.1.1 (CLAUDE.md pitfall 9). Both context files — the repo's `src/amore/context.py`
and the live `usr/Shared/amore/context.py` — evaluate through `ctxrunner.py ctx` on scratch copies,
37 variables each.

**`ctxrunner ctx` exits 0 when the context fails.** It pickles `(ctx, error_info)` and returns
normally either way, so the exit code proves nothing. Check `error_info is None` in the pickle
(unpickling needs `damnit/ctxsupport` on `PYTHONPATH`). Both context files also need the live
directory's `analysis_helpers*.py` on the path.

### 2.3 The pitfalls, re-checked in 2026.9

Measured by `bench_pyfai.py` in both versions. The answers are the same in both.

| Pitfall | 2026.9 | Evidence |
|---|---|---|
| 1. Method strings | **still holds**: `method="csc"` resolves to `("no","histogram","python")` | probe `pitfall1_string_csc` |
| 1. OpenCL fallback warning | **still holds**: asked for `("full","csr","opencl")` without pyopencl, pyFAI warns "`IntegrationMethod(1d int, full split, CSR, cython)` will be used" and runs `("full","histogram","cython")` | probe `pitfall1_tuple_full_csr_opencl` |
| 2. Poisson floor | **still holds**: `error_model="poisson"` gives 99.3× the variance of `variance=x` on an r0423 AGIPD frame | probe `pitfall2_…` |
| 5. Engine cache | unchanged: keyed by the method, so a second `npt` rebuilds into the same slot | probe `pitfall5_engine_keys` |
| 17. Per-frame `mask=` rebuilds the matrix | **still holds**: 16.6 vs 2.7 ms/frame. The NaN path is still bit-identical to `mask=` (S, N, V, max rel diff 0.0) | `jf.mask_path` |
| 18. Read-only inputs | **still holds on Linux**: a read-only mask, data or variance raises `ValueError: buffer source array is read-only`. `solidAngleArray` returns its cached object, writeable, and `np.ascontiguousarray` returns it unchanged | probe `pitfall18` |

The Python layer changed little where these live (`integrator/common.py`: an `auto_gc` option and
`orientation` passed to the full-split builders; `engines/*`: ruff clean-up, `.shape =` →
`reshape`). The Cython sources are not in the wheel, which is why 2 and 18 are probed and not
read.

### 2.4 The operator sha256 depends on the node's instruction set

Found while comparing the P4 output with production. It is not about pyFAI, but it decides how
an operator check may be written (open task 17b).

| Built on | CPU | sha256 |
|---|---|---|
| production r0423–r0428 (DAMNIT, 2026-10-07, pyFAI 2026.5) | E5-2640 v4, EPYC 75F3 | `2bec8d7c…` |
| probe 25287917 / 25287918 (pyFAI 2026.9) | EPYC 75F3 / E5-2640 v4 — AVX2, no AVX-512 | `2bec8d7c…` |
| this study, both pyFAI versions; probes 25287882 / 25287883 | EPYC 9374F / Gold 6140 — AVX-512 | `a59609c8…` |

Same venv, same code, same input files (the config hashes are equal). Only `omega` differs:
**105 693 of 1 048 576 pixels, by one ULP** (max relative 1.1e-16). `coef`, `bins`, `indptr` and
`q`, which make up the sparse matrix, are bit-identical. The q axis and every stored sum agree
far below float32 resolution, and gate B is 4.464e-8 on both. With numpy's AVX-512 kernels
switched off (`NPY_DISABLE_CPU_FEATURES`) the AVX-512 nodes still give `a59609c8`. So the
dispatch happens below numpy, presumably in the math library behind pyFAI's solid-angle
computation. The library was not pinned down further.

Consequence: an `is_complete` or resume check that compares a stored `operator_sha256` with one
rebuilt on the current node refuses a valid file whenever DAMNIT lands on a different CPU class.
`allcpu` mixes them. Compare the sparse matrix and q exactly, and Ω to a tolerance, rather than
the digest. The worker-side check (`worker.init` against the parent's sha) is safe: both are
built on the same node.

S1 adopted this on 2026-10-13 (branch `claude/s1-damnit-integrity-*`, commit 7ebeaa8):
- **How the check compares now:** AGIPD `coef`/`bins`/`indptr`/`q` exactly and `omega` to
  rtol 1e-12; JF mask exactly, `q`/`omega` to rtol 1e-6.
- **Reproduced** on a Gold 6126 (job 25287937).
- **How common it is:** a status over every stored file found 119 of 459 AGIPD files, and 140
  of 418 per JUNGFRAU detector, written on the other CPU class. All are complete under the new
  check.

---

## 3. Virtual CXI files

### 3.1 Verdict

**Do not adopt virtual CXI files in either pass.**
- **AGIPD:** read frame by frame, a virtual file is 10–30 % cheaper than our EXtra-data reads.
  Read a train at a time, it is 12× dearer.
- **JUNGFRAU:** it is slower however it is read.
- **Tape:** it fails silently once proc is taped: an AGIPD module whose file is gone reads as
  zero photons.
- **When one is used** (e.g. an XCCA or XPCS prototype while proc is on disk):
  - read it one frame per slice;
  - check that every source file exists before trusting a value (§3.4);
  - treat it as a disposable index, rebuilt in seconds, never as a product.

The bigger lever for AGIPD read speed is still the bulk chunk reader (integrator I3: decode
floor 0.9 + 1.6 ms/frame against the 2.4 + 4.7 the pass pays now). P6 (36 vs 72 workers) is
the other.

### 3.2 Measurement

`scripts/bench_io.py`, job 25288100, max-cfel029 (AMD EPYC 9374F), r0423, 2026-10-13.
EXtra-data 1.24.0, h5py 3.16.0, HDF5 2.0.0, the no-dev venv of §1.4. Data and mask are timed
separately, at 1 and 36 processes. Every (variant, process count) reads its own trains, spread
over the run, so the page cache favours none: 4 or 2 trains per process for AGIPD, 8 or 2 for
jf1. JSON: `scripts/bench_io_r0423_20261013T152146Z.json`.

**Build** (`write_virtual_cxi`, after the detector object exists):

| | AGIPD | jf1 |
|---|---|---|
| wall | 0.82 s (opening the run and the detector: ~13 s more) | 0.10 s |
| size | 30 MB (per-frame ids and `experiment_identifier` strings) | 3.1 MB |
| shape | `(465000, 16, 512, 128)` | `(48000, 1, 512, 1024)`: all 16 cells, dark ones too |
| virtual mappings | 192 per dataset (16 modules × 12 sequence files) | 6 |
| datasets | `data`, `mask`, `gain` | `data`, `mask`, `gain` |
| fill value | data `0` (int16), mask `0xFFFFFFFF`, gain `0` | data `NaN`, mask `0xFFFFFFFF` |

EXtra-data writes `mask` only when the source group also has `gain`: it takes that as the sign of
calibrated data. Our proc files have it, for both detectors. EXtra-data 1.24.0's
`write_cxi.py` prints one `### Source: …` line per module and key; it ships that way (the
installed file matches the wheel's RECORD).

**Per-frame read cost, ms per frame per process:**

| AGIPD | data, 1 proc | mask, 1 proc | data, 36 procs | mask, 36 procs | frames/s, 36 procs |
|---|---|---|---|---|---|
| EXtra-data `["image.*"].ndarray(decompress_threads=1)` per train (the pass) | 3.68 | 5.57 | 4.05 | 7.38 | 2417 |
| VDS, one slice per train (`ds[r0:r1]`) | **45.3** | **49.7** | 47.1 | 50.2 | 363 |
| VDS, one slice per frame (`ds[r]`, 16 chunks) | 2.63 | 5.03 | 3.68 | 6.48 | 3124 |

| jf1, per *lit* frame (8 of 16 per train) | data, 1 proc | mask, 1 proc | data, 36 procs | mask, 36 procs | lit frames/s, 36 procs |
|---|---|---|---|---|---|
| EXtra-data `["data.*"].ndarray()` per train (the pass; reads all 16) | 4.61 | 5.28 | 11.2 | 7.53 | 675 |
| VDS, one slice per train (16 rows, one source chunk) | 52.0 | 50.9 | 56.6 | 52.4 | 248 |
| VDS, the lit rows one at a time | 4.81 | **31.9** | 4.60 | 35.4 | 402 |
| … with a 256 MiB chunk cache on the VDS file | 3.02 | 31.9 | 4.60 | 34.9 | 484 |

The EXtra-data rows sit above P4's in-pass figures (read_data 2.44, read_mask 4.68 ms under 36
workers). The benchmark reads isolated trains spread over the run, where P4 reads blocks of
consecutive trains and GPFS read-ahead helps. Compare the rows of one table with each other, not
with P4.

What the numbers say:
- **Multi-frame slices of the virtual dataset are an order of magnitude slower** than the same
  bytes read frame by frame, on both detectors and at both process counts. The cause was not
  isolated. Its practical form is that a reader must loop over frames.
- **AGIPD frame by frame** beats the pass's EXtra-data read by 1.0 / 0.5 ms (1 process) and
  0.4 / 0.9 ms (36) for data / mask. Taking the 36-process figures, that is 1.3 ms of P4's
  11.6 ms/frame/core, at most ~10 % of the pass's wall time.
- **JUNGFRAU** stores `data/adc` *uncompressed* (float32, chunks `(1, 16, 512, 1024)`), while
  `mask` and `gain` are gzip over the same 32 MiB chunk. Read one cell at a time, every lit cell
  decompresses the whole mask chunk again: 31.9 against 5.28 ms. A 256 MiB `rdcc_nbytes` on the
  VDS file changes nothing for the mask, so the cache set there does not reach the source datasets.
  The pass's whole-train read is already the right shape for these chunks.

### 3.3 Frame identity

Checked on 5 trains spread over r0423, each against EXtra-data's coordinates and frame data:

- **AGIPD:** `entry_1/trainId`, `entry_1/pulseId` and `entry_1/cellId` (virtual, per module,
  identical across modules), plus `experiment_identifier` (`"train:pulse"` strings). All
  equal to EXtra-data's `train_id_` / `pulse_id_` / `cell_id_coordinates()`. Data and mask are
  bit-identical. Same 3000 trains. The `pulseId` is the detector's own numbering (2, 4, 6, … on
  r0423), the same as our pass stores, **not** the X-ray pulse IDs `XrayPulses` reports (200, 202,
  …). Match the two by order (`transfer_context/03`).
- **JUNGFRAU:** `entry_1/trainId` and `entry_1/memoryCell` (plus a virtual `cellId` of the same
  values). **No pulse IDs**, as fxs's CXI lacked them (`transfer_context/10`). One row per
  memory cell, dark cells included, in storage order (cell 15 first, pitfall 4). Lit cells must
  be selected by `memoryCell`, never by row. Data and mask are bit-identical; the same 3000 trains.

### 3.4 When proc is taped, and who can read it

- **Source paths are absolute, in the proposal form**
  (`/gpfs/exfel/exp/MID/202601/p010400/proc/r0423/CORR-…`), as EXtra-data opened them.
- **A missing source reads as the fill value, silently.** Shown on a mock run written like the
  real files (`bench_io --phase taped`):
  - One module's file was moved away. The virtual file still opened and read: that module's
    int16 data came back **all zero**, which looks like "no photons", and its mask all
    `0xFFFFFFFF`. The other modules still read normally.
  - A reader of data alone gets a plausible frame with a dark module. A reader that applies the
    mask drops the module, but only if it reads the mask.
  - EXtra-data on the same directory refuses: `No data found with >= 16 modules present`.
- **Detecting it** is cheap: `all(os.path.exists(s.file_name) for s in
  ds.virtual_sources())`, 192 `stat` calls for AGIPD. Any VDS reader must do it, and must also
  check the per-train frame count against its own expectation.
- **After the move to tape** every virtual file over proc is a file of fill values. It is not
  worth keeping: rebuilding takes ~14 s from a restored proc, and it holds no data of its own
  (cf. the window sums, which are the record because they do).
- **Access:** the files take scratch's default ACL (`user::r--`, `group::r--`, group `upex`,
  `other::---`), the same as `agipd_saxs.h5`. Anyone who can read our scratch outputs and proc
  can read them, DAMNIT's jobs included. Nothing about them is specific to one user.

### 3.5 Tier-2 access (open task 12) and the photon-statistics pass (S4)

- **What a VDS offers:** a flat `(frame, module, ss, fs)` array with frame identity beside it,
  in one file that h5py, extra-speckle or a notebook can open without EXtra-data. For per-frame
  random access, such as XPCS ROI series or XCCA frame selection, that is simpler code. It is
  also no slower than EXtra-data, if read frame by frame.
- **What it does not offer:**
  - Any saving on ROI reads. The chunk is a whole module, so `ds[r, m, y0:y1, x0:x1]` still
    decompresses all 512 × 128 pixels of that module. Only the module choice saves anything,
    and EXtra-data's module selection does the same.
  - Any lasting value once proc is taped (§3.4).
- **For S4's photon statistics,** which must be extracted from proc before taping, a VDS is
  acceptable as the input index if every job checks its sources first. It is not a reason to
  delay. The output is what must survive, and the VDS cannot.
