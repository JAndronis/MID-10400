# DAMNIT and pass integrity (open task 17)

Read this when working on the lock, the completeness check, resume, the DAMNIT context file,
`scripts/install_context.py` or `scripts/reprocess_safe.py`. It covers CLAUDE.md pitfalls 26
(double launch) and 27 (live vs repo context). Session S1, 2026-10-13.

## 1. Facts verified 2026-10-13

**DAMNIT 0.4.0.** These were read from the installed source, not from the docs.

- **Which damnit.** The listener runs damnit from `amore-beta`
  (`metameta.damnit_python = …/amore-beta/.pixi/envs/default/bin/python3.13`). `module load exfel
  damnit/stable` gives `amore-mid`. Both are 0.4.0, and their `ctxsupport/` is byte-identical to the
  deploy venv's.
- **Context evaluation.** DAMNIT evaluates a context file with `python -m ctxrunner ctx
  <context.py> <out.pkl>`. It runs from the amore directory, with
  `PYTHONPATH=<damnit>/ctxsupport` (`backend/extract_data.py:get_context_file`). `-m` puts the
  cwd on `sys.path`, which is how `from analysis_helpers import …` resolves.
- **`ctx` exits 0 even for a broken context.** It pickles `(ctx, error_info)`, and a failure
  shows up only as a non-`None` `error_info`. Under `-m` the `ContextFile` pickles as
  `__main__.ContextFile`, so DAMNIT unpickles it with `ContextFileUnpickler`.
- **`exec <proposal> <run> <raw|proc|all>`** loads `context.py` from the cwd and writes its
  results to `extracted_data/` in the cwd. `--var` takes variable names, `--match` takes title
  substrings, and `--cluster-job` is needed for `cluster=True` variables, which are skipped
  without it.
- **Dependencies are re-run.** Selecting a variable (`ContextFile.filter`) adds all of its
  dependencies back. So requesting `agipd_iq_overview` runs `agipd_saxs` again in the same job.
  That is the mechanism behind pitfall 26.
- **`damnit reprocess [--in DIR] [--match TITLE] runs…`.**
  - `--match` is a case-insensitive substring of the title, not a regex.
  - Several runs go out as one Slurm job array named `p10400-damnit`, at most `concurrent_jobs` (15) at a
    time. One run goes out as a job named `r<run>-p10400-damnit`, with the run number unpadded.
  - The first job of a batch also updates the variable list in `runs.sqlite`.
- **Partitions and clusters.** Cluster variables use partition `allcpu`. Non-cluster jobs go to
  cluster `solaris`, so a queue query needs `--clusters=all`.
- **The DB was read with `sqlite3 …?mode=ro` only.**

**Slurm (Maxwell).**
- `squeue --clusters=all -j <id>` **exits 1 even for a running job**, because the other
  clusters print "Invalid job id specified". Read the verdict from stdout, never from the return code.
- `MinJobAge` is 90 s: a finished job leaves the queue listing after that.
- An array task is listed as `<array>_<task>`, not by its own `SLURM_JOB_ID`.

**Quota** (`mmlsquota -j <fileset> exflu`, where the fileset comes from `mmlsattr -L <dir>`):

| Fileset | Holds | Used | Quota |
|---|---|---|---|
| `bt_scratch_MID_202601_p010400` | scratch | 6.8 TB | 193 TB |
| `bt_usr_MID_202601_p010400` | usr | 2.4 TB | 5.4 TB |

**Stored outputs** (provenance attributes only):
- **AGIPD.** All 459 `scratch/agipd_saxs` files use `npt` 500, were written 2026-10-07 with
  `overwrite=True` and pyFAI 2026.5.0, and every row is OK.
- **JUNGFRAU.** 836 of the 839 `scratch/jungfrau_waxs` files use `npt` 200 (2026-09-14). The
  exceptions are r0016 and r0017 jf1, at `npt` 500, and r0041 jf1, at `npt` 500 with **pyFAI
  2026.9.0** (written 2026-10-09 on max-exfl462, not by the deploy env).
- **r0423.** `is_complete` holds for AGIPD at the defaults and for jf1 and jf2 at `npt=200`
  (§3). The operator built on 2026-10-13 hashes to the one stored on 2026-10-07.

**Operator build cost** (r0423 provenance, `setup_timings.operator`): AGIPD 1.2 s, JF 0.17 s.

## 2. The lock (`analysis.common.lock`)

**The file.** `output_lock(path)` creates `<path>.lock` with `O_CREAT | O_EXCL`, which is atomic
across GPFS nodes. It writes JSON into it: output, host, pid, user, `SLURM_JOB_ID`,
`SLURM_CLUSTER_NAME`, the start time, and a random token. The token means a process only ever
removes its own lock. The lock is held for as long as the pass writes.

**When another process finds the lock**, it decides whether the holder is still alive:

| Holder | Test | Verdict |
|---|---|---|
| Ran in a Slurm job (not ours) | `slurm_job_state` | listed and not in `FINISHED_STATES` → alive. `COMPLETING` counts as alive |
| No job, this host | `kill(pid, 0)` | the pid is ours → alive (re-entry); the pid exists → alive |
| No job, another host | — | **cannot be checked → `OutputBusy`**, with instructions to delete it by hand |
| Unparsable, < 60 s old | — | being written → `OutputBusy` |
| `squeue` failed for another reason | — | `JobStateUnknown` → `OutputBusy`. A lock is never taken on a guess |

**Taking over a dead holder's lock.**
1. Rename the lock aside.
2. Check that the file moved aside is the one judged stale.
3. Delete it and retry the create.

If the lock changed hands between the read and the rename, the moved file is linked back
(`os.link` refuses an occupied slot) and `OutputBusy` is raised. The takeover is logged at WARNING.
It is also written into the output's provenance: `lock_takeover = {previous, reason, at}`, beside
`lock`, the holder's record without its token.

**Why stale locks happen.** A job that hits its time limit or is killed leaves its lock behind:
neither SIGKILL nor Slurm's SIGTERM runs `finally`.

**A second job is refused, not made to wait** (user decision 2026-10-13). In DAMNIT that shows as an error
in that job's cell. The holder's own result arrives normally.

**Which files are locked:**
- `run_agipd_saxs` locks the frame table, then the window sums (when `pixel_sum_trains` is set).
  `operator.npz` and `masks.npz` sit in the frame table's directory and are covered by its lock.
- `run_pixel_sums` locks the window sums only.
- `run_jungfrau_waxs` locks its own file.

## 3. Completeness (`analysis.{saxs,waxs}.complete.is_complete`)

**The check is read-only.** It returns a `Completeness`, which is true when complete and
otherwise carries its reasons. A frame table is complete when:
- it exists and opens, and was finalised (`status_summary` is in provenance);
- its config hash equals the current one, and it has every column this pass writes;
- no row is `NOT_PROCESSED` or `WORKER_ERROR`, and `trains/count` accounts for every row;
- the pyFAI version recorded in it equals the installed one (`GATING_PACKAGES`). This gate is kept for AGIPD
  too, by user decision on 2026-10-13, although AGIPD numbers depend on pyFAI only through the
  operator;
- `provenance.operator_sha256` equals `/operator.sha256`, and both equal the sha256 of the
  operator this pass would build now.

**The window sums (AGIPD) are complete when:**
- the hash (with the detector name resolved) and the schema match, and the file was finalised;
- every window is written;
- no train is unfinished;
- the trains are exactly the frame table's.

**How the would-be operator sha256 is obtained: the operator is built.**
- **Inside a pass**, the check reuses the operator the pass builds anyway, so it costs nothing extra.
- **Elsewhere** (`expected_operator_sha256`), the build is memoised per process on the config with
  the proposal and run zeroed, since no operator input depends on the run. A script checking 459
  runs builds it once.
- **Why not compare library versions instead:** an exact sha needs no assumption about what a pyFAI or
  EXtra-geom release changes.

**The early return.**
- **Where.** At the top of each `run_*`, after the operator build (which reads only the geometry or
  PONI): if no lock file exists and the output is complete, the result comes from the file.
  Otherwise the pass takes the lock and checks again, and a complete run still returns from the file.
- **Raises on the same terms as a full pass.** `IncompleteRun` for final non-OK rows such as
  `LABEL_MISMATCH`. For JF, also `UnexpectedLitCells` when `expected_lit_cells` is pinned and
  the stored lit set differs.
- **What it writes.** Nothing: no lock file, no npz, no attribute. On a complete file the inode and
  mtime are unchanged (tests, and r0423 read-only on 2026-10-13).
- **`overwrite=True` skips the early return, but never the lock.** `run_pixel_sums` ignores
  `overwrite` here as everywhere (§15.3 of the AGIPD context file).
- **A live lock refuses even a complete run**, because the file may be mid-write.

**Not covered.** A changed *input* — proc re-corrected after the output was written — is not
detected. The frame table never checked its plan against the file; the window sums still do, on
resume.

## 4. Resume

- **`WORKER_ERROR` is retried.** `block_complete` treats `UNFINISHED` = {`NOT_PROCESSED`,
  `WORKER_ERROR`} as not done. `LABEL_MISMATCH` and `DATA_CHECK_FAILED` are properties of the data
  and still count as done.
  - A retried block's earlier message moves from `block_errors` to `block_errors_resolved`.
  - Before this change, a frame-table `WORKER_ERROR` stayed until its rows were reset by hand.
    The window sums already rebuilt such windows.
- **`FrameTableWriter.check_resumable`** refuses an existing file unless `overwrite`:
  - `ConfigHashMismatch` — the config hash differs;
  - `SchemaMismatch` — a column is missing;
  - `OperatorMismatch`, a subclass of `ConfigHashMismatch` — the operator or the pyFAI version
    differs.

  `open_or_create` calls it. The `run_*` functions also call it under the lock, before the plan, masks
  and self-test, so a file that can only be reprocessed is refused in a second.
  Without the operator check, a run that `is_complete` rejected for its pyFAI version would
  resume, find no work, and finalise the file with the new version stamped on old rows.
- **`package_versions` is now written when the file is created**, as well as at finalise. So a run
  that died before finalising can still be checked.
- **`operator.npz` and `masks.npz` are written atomically**: `analysis.common.arrays.atomic_savez`
  writes a temporary file, fsyncs it, then `os.replace`s it.

## 5–7. Context, install, reprocess

To be filled in by Phases B and C.
