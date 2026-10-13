# Performance and tooling: measured verdicts (open task 19)

Context file for CLAUDE.md open task 19, written in session S3 (2026-10-13). It records what was
measured, where, with which versions, and what was decided. Read `CLAUDE.md` first.

**State:**
- (c) SLURM helpers: done (`analysis.common.slurm`, `scripts/_common.py`).
- (d) `fan_out` progress lines: done.
- (a) pyFAI 2026.9: measured, **no speed gain, bit-identical results**. The lock is upgraded on
  branch `claude/s3-speed-tooling-*`; it merges only after the user's review.
- (b) virtual CXI: §3.

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

---

## 3. Virtual CXI files

PENDING (phase b).
