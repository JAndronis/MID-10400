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
  (§3). On an AVX2 node (EPYC 75F3), the operator built on 2026-10-13 hashes to the one stored on
  2026-10-07.

**Operator build cost** (r0423 provenance, `setup_timings.operator`): AGIPD 1.2 s, JF 0.17 s.
Reading the stored operator back costs about 0.01 s: 28 MB for AGIPD, 2 MB for JF.

**The operator sha256 depends on the node's instruction set** (S3, 2026-10-13, r0423, pyFAI
2026.5 and 2026.9).
- AVX2 nodes (EPYC 75F3, E5-2640 v4), where production r0423–r0428 were written, build
  `2bec8d7c…`.
- AVX-512 nodes (EPYC 9374F, Gold 6140) build `a59609c8…`.
- Only `omega` differs: 105 693 of 1 048 576 pixels, by 1 ULP (at most 1.1e-16 relative).
  `coef`, `bins`, `indptr` and `q` are bit-identical.
- `allcpu` mixes both classes, so **never gate on an exact sha256 against an operator rebuilt on
  another node.**
- The worker's own sha256 check (`worker.init`) is safe: parent and workers share a node.
- **Reproduced by S1** on a Gold 6126 (max-cssb009, Slurm job 25287937, 2026-10-13). The AGIPD
  operator built there hashes to `a59609c8…`, against `2bec8d7c…` stored; `omega` differs on
  105 693 pixels, by at most 1.11e-16; `coef`, `bins`, `indptr` and `q` are identical. With the
  array comparison, `is_complete` holds for AGIPD, jf1 and jf2. The sha256 comparison it replaced
  would have refused all three. Script and log: `scratch/slurm_logs/s1_integrity/isa_check*`.

**pyFAI 2026.5 → 2026.9** (S3): bit-identical on everything this repository uses.

## 2. The lock (`analysis.common.lock`)

**The file.** `output_lock(path)` creates `<path>.lock` with `O_CREAT | O_EXCL`, which is atomic
across GPFS nodes. It writes JSON into it: output, host, pid, user, `SLURM_JOB_ID`,
`SLURM_CLUSTER_NAME`, the start time, and a random token. The token means a process only ever
removes its own lock. The lock is held for as long as the pass writes.

**When another process finds the lock**, it decides whether the holder is still alive:

| Holder | Test | Verdict |
|---|---|---|
| Ran in a Slurm job (not ours) | `slurm_job_state` | listed and not in `FINISHED_STATES` → alive. `COMPLETING` counts as alive |
| This job, an earlier `SLURM_RESTART_COUNT` | — | **dead**: the first attempt of a job that was preempted and requeued under the same id |
| This job, the same attempt, this host | `kill(pid, 0)` | as for "No job, this host" below |
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

**Requeued jobs.**
- `allcpu` preempts with `PreemptMode=REQUEUE` (`JobRequeue=1`), and a requeued job keeps its job
  id. S4's job 25297694 was "CANCELLED DUE TO PREEMPTION" and went back to PENDING on 2026-10-14.
- So the requeued attempt meets the lock its killed first attempt left. Asking Slurm about its own
  id would only say "running", and on another node the pid cannot be checked.
- The lock therefore records `SLURM_RESTART_COUNT` ("the number of times the job has been
  restarted", `man sbatch`; absent before any restart). A lock from the same job id at a lower
  count is stale.
- A lock written before this field existed reads as count 0.

**A second job is refused, not made to wait** (user decision 2026-10-13). In DAMNIT that shows as an error
in that job's cell. The holder's own result arrives normally.

**HDF5's own file lock does not protect anything inside DAMNIT.**
- `extra_speckle/xcca/tools.py:2` sets `os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"` at import
  time. The context file imports `extra_speckle.pipeline` at the top, so every ctxrunner process
  and every cluster job runs with it off, and the pass's spawned workers inherit it.
- A second writer is therefore not refused by HDF5. The `O_EXCL` lock file above does not depend
  on HDF5 and is the only guard.
- Found by S4, 2026-10-14; checked against the deploy venv: unset before the import, `FALSE` after.

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
- `provenance.operator_sha256` equals `/operator.sha256` (both written by the same job);
- the stored operator equals the one this pass would build now, **array by array**, not by sha256
  (§1):
  - AGIPD: `coef`, `bins`, `indptr` and `q` exactly; `omega` to `OMEGA_RTOL` = 1e-12 relative,
    against a 1.1e-16 spread between node classes;
  - JF: the static mask exactly; `q` and `omega` to 1e-6, because `omega` is stored as float32.

**The window sums (AGIPD) are complete when:**
- the hash (with the detector name resolved) and the schema match, and the file was finalised;
- every window is written;
- no train is unfinished;
- the trains are exactly the frame table's.

**How the would-be operator is obtained: it is built, on the checking node.**
- **Inside a pass**, the check reuses the operator the pass builds anyway, so it costs nothing extra.
- **Elsewhere** (`expected_operator`), the build is memoised per process on the config with the
  proposal and run zeroed, since no operator input depends on the run. A script checking 459 runs
  builds it once.
- **Why not compare library versions instead:** comparing the arrays needs no assumption about
  what a pyFAI or EXtra-geom release changes.

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
  - `OperatorMismatch`, a subclass of `ConfigHashMismatch` — the operator arrays (same rule as
    §3) or the pyFAI version differ.

  `open_or_create` calls it. The `run_*` functions also call it under the lock, before the plan, masks
  and self-test, so a file that can only be reprocessed is refused in a second.
  Without the operator check, a run that `is_complete` rejected for its pyFAI version would
  resume, find no work, and finalise the file with the new version stamped on old rows.
- **`package_versions` is now written when the file is created**, as well as at finalise. So a run
  that died before finalising can still be checked.
- **`operator.npz` and `masks.npz` are written atomically**: `analysis.common.arrays.atomic_savez`
  writes a temporary file, fsyncs it, then `os.replace`s it.

## 5. Reconciling the live and repo context (2026-10-13)

DAMNIT reads `usr/Shared/amore/context.py` (pitfall 27). The repo copy is now authoritative and
is installed with `scripts/install_context.py` (§6). The live file of 2026-10-07 (sha256
`21e42116…`) differed in five hunks. Each was decided by the user:

| Hunk | Live | Decision |
|---|---|---|
| `selected_trains = None` | a stub, used nowhere | dropped |
| XPCS `remove_outer_points` | an `if run_type == "XPCS"` branch, both arms `False`. The XPCS arm carried commented-out overrides: `motor_moving='MID_EXP_SAM/MDL/DATA_SELECTOR_2'`, a `motor_key` holding a source *and* a key, and `motor_range=(-24, 24)` | the repo's one-line `False`. Same behaviour. If it is ever set True, extra_speckle's defaults apply (`MID_SAE_FSSS/MOTOR/SCANNERX`, `actualPosition`, `(-22.5, 22.5)`) unless the overrides are added back |
| `agipd_saxs(..., overwrite=True)` | — | dropped |
| `jungfrau_waxs(..., npt=200, overwrite=True)` ×2 | — | `overwrite` dropped. `npt` 200 kept, but moved into `analysis.waxs.damnit.DAMNIT_OVERRIDES` |
| comments and the JF1 docstring | — | the repo's |

**Why `npt` lives in the package, not the context file.** 836 of the 839 stored JF files are at
`npt` 200. The config default stays 500, which W4 ran at. `damnit_config()` is the one place
that says what DAMNIT runs, so `reprocess_safe` and `is_complete` judge DAMNIT's files under the
config DAMNIT actually uses.

**`tests/test_context.py` enforces both:**
- no call in the context file passes `overwrite`;
- the integration wrappers pass no keyword at all.

Both fail on the live file of 2026-10-07 (lines 378, 393 and 401).

**`analysis_helpers.py`.** The repo's 7.8 KB copy is the live 24 KB v2 minus nine functions the
context no longer calls (`integrate_run`, `get_integrator`, `integrate_frame`, …). The five it
imports have the same code, apart from docstrings and one f-string. The user confirmed that
nobody else imports the live copy, so the repo's copy is installed with the context.

**Not reconciled, because it is the same in both:** `xpcs_pipeline` returns `"Incomplete"`, or
`"Done"`, and its 14 dependants read a module-global `XPCS_RESULTS` that only that job fills.
This is for the XPCS owner (follow-up).

**ctxrunner check** (deploy venv, damnit 0.4.0, scratch copy of the repo context and helpers):
`error_info` None, 37 variables, 22 of them `cluster=True`. A copy with a dangling `var#` also
exits 0, with `ctx = None` and the traceback in `error_info`. The pickled `ContextFile` has its
functions stripped, so its dependency graph reads as empty; check the dependency wiring with
`tests/test_context.py` instead.

## 6. `scripts/install_context.py`

DAMNIT reads `usr/Shared/amore`, and others edit it, so an install is guarded. Without
`--install` the script is a dry run. In order:

1. **Clean source.** Refuse a dirty `src/amore`, so the recorded commit is what is installed.
2. **Show what changes.** Print the diff live → repo for `context.py` and `analysis_helpers.py`,
   and a 12-character **live-state id** naming both live files' sha256s.
3. **Guard.** The live sha256s must equal the ones recorded by the last install, in
   `backups/installed.json`. The first time there is no record, so `--accept-live <id>` is
   required. A stale id — the files changed after the diff was read — is refused.
4. **Evaluate both contexts as DAMNIT does.** Run `<context_python> -m ctxrunner ctx` in a scratch
   copy, with the listener's own `ctxsupport` on `PYTHONPATH`. Both settings come from
   `runs.sqlite`, opened read-only. Fail unless the pickled `error_info` is `None`. Report the
   variables added or removed.
5. **Deploy gate.** The commit must be an ancestor of the deploy clone's HEAD, so DAMNIT never
   evaluates a context calling code it does not have. **Install only after S1 is merged.**
6. **`--install`.**
   - Write a timestamped backup of both live files, `install_<stamp>.diff` and
     `install_<stamp>.sha256` to `usr/Shared/amore/backups/`.
   - Re-check each live sha256, then replace each file atomically (temp file + `os.replace`),
     keeping its mode. The owner becomes the installer.
   - Write `installed.json`: sha256s, commit, time, user, backups. Append the same to
     `install_log.jsonl`.

**Real dry run, 2026-10-13** (HEAD `7ebeaa8`, report `scripts/install_context_*.json`):
- both contexts evaluate, 37 variables each, none added or removed;
- refused for exactly two reasons: first install (`--accept-live 44561ba487b3` for the live
  state of 2026-10-07), and the commit not yet deployed;
- the live files were untouched and no `backups/` was created.

**Exercised end to end on a copy of the live directory** (`--live-dir`, with this worktree as
`--deploy-dir`): first-install refusal, a wrong id refused, the install itself (contents equal the
repo, mode 644 kept, backup equal to the old live file), a clean rerun, and an edit after the
install refused with the edited file left alone.

## 7. `scripts/reprocess_safe.py`

`status | start | check | cleanup | restore <runs> -d agipd|jf1|jf2 [--dry-run] [-j N]`. Runs are
numbers, ranges `a-b`, or `all`. The config each detector is judged under is DAMNIT's:
`analysis.saxs.damnit.config_for` and `analysis.waxs.damnit.damnit_config`.

**`status`** shows, per run and detector: the output, complete or the reasons it is not, a lock,
a DAMNIT job, and backups with their check verdict.

**What counts as busy.** A job named `r<run>-p10400-damnit`, from `squeue --clusters=all` over all
users. Any active job array `p10400-damnit` counts every run as busy, because squeue cannot say
which runs it covers.

**`start`.** Refuses everything unless every selected run is idle:
- no DAMNIT job and no lock;
- no leftover backup;
- room on the fileset: free ≥ 1.2 × the outputs' size, from `mmlsattr -L` → fileset,
  `df` → filesystem, `mmlsquota -j … --block-size K`.

It then renames each output to `<stem>_before_reprocess_<UTC stamp>.h5` and runs `bash -lc 'module
load exfel damnit/stable && damnit reprocess --in usr/Shared/amore --match … <runs>'`. The
`--match` titles:
- AGIPD: `AGIPD I(q)`, which also matches the overview;
- JF: `JF1 I(q)` or `JF2 I(q)`, plus `JUNGFRAU I(q)`, so the overview and combined curve are
  rebuilt; the other detector is answered from its file.

`--no-trigger` renames only. **The AGIPD window sums are never moved.**

**`check`.** Requires exactly one backup, no job and no lock. It then needs `is_complete` on the
new output, every dataset of it read back (and of the window sums for AGIPD), and the backup's
frame count and trains, unless `--accept-count-change`. It writes `<backup>.check.json` with the
verdict and the new output's (inode, mtime, size).

**`cleanup`** deletes a backup only if its check passed and the new output is still the file that
was checked.

**`restore`** keeps the new output as `<stem>_failed_reprocess_<stamp>.h5` and renames the backup
back.

**Exercised on copies of r0423's JF files** under proposal scratch (`--root`, DAMNIT never
called):
- `--dry-run` moves nothing; `--no-trigger` moves the output aside;
- `start` refuses a leftover backup;
- `check` fails with no new output, passes a complete one, and fails one with 16 frames
  `WORKER_ERROR`;
- `cleanup` deletes the passed backup, keeps the failed one, and keeps one whose output was touched
  after its check;
- `restore` puts the backup back and keeps the failed output.

**Real runs, read-only, 2026-10-13.**
- **`start 423-428 -d agipd --dry-run`:** 14.1 GB would move; 190.9 TB free on
  `bt_scratch_MID_202601_p010400`; no busy run.
- **`status all -d agipd -d jf1 -d jf2 -j 4`** (5.8 min, login node):
  - AGIPD: all 459 complete; r0356 and r0382 have no output.
  - jf1 and jf2: 418 complete each, and 43 dark runs without a file.
  - Incomplete: r0016, r0017 and r0041 jf1, at `npt` 500 (r0041 also pyFAI 2026.9.0), which are
    commissioning runs not used.
- **The array comparison matters in practice.** 119 of the 459 AGIPD files carry the AVX-512
  operator sha256 `a59609c8…`, and 140 of the 418 per JF detector carry their class's other sha256.
  An exact-sha256 gate would have called about a quarter of the archive incomplete.
- **The sha256 shortcut.** When the stored sha256 equals the one built on the checking node, the
  arrays are not read back (they are covered by it). Otherwise they are read and compared.

### 7.1 The XPCS hits product (`-d xpcs`, added 2026-10-19)

`scripts/reprocess_safe.py -d xpcs` covers `analysis.xpcs` (`context/xpcs-hits.md`).

- **What moves.** `xpcs_hits.h5` (derived, rebuilt from the store in minutes). **The photon store
  `agipd_ring_photons.h5` never moves**: proc cannot refill it once taped. `check` test-reads it
  as a companion, as it does the AGIPD window sums. The pass itself keeps a result made for other
  inputs as `xpcs_hits.prev.h5`; that is not one of this script's backups.
- **Titles.** `XPCS hits preprocessing`, `g2`, `TTCF`, `overview`. `XPCS hits frames used` is left
  to `reprocess_all.sh refresh`, because it reads the file the cluster job is still writing. None
  is a substring of an old extra-speckle title, and `reprocess_all.sh xpcs` no longer matches the
  bare `XPCS`. `python scripts/match_titles.py --self-test` lists every title of `context.py` per
  `--match` and asserts both properties.
- **Refusals.** `xpcs_hits_preprocessing` depends on `agipd_saxs`, so `start -d xpcs` refuses a run
  with no AGIPD output, a locked or incomplete one (judged at the real location whatever `--root`
  says), a run with no store and no proc directory, and any command that also names another
  detector. The refusal is for the whole command, so name the runs.
- **What `complete` means here.** Readable, schema current, finalised, no `WORKER_ERROR`,
  `NOT_PROCESSED` or `LABEL_MISMATCH` hit train, and the store holds every OK hit train under the
  hash the result recorded; final non-OK trains (`NO_FRAMES`, ...) are notes. It cannot tell the
  current hit set or merge hash from the recorded ones (that needs the classifier and the
  geometry, which a script must not import): `xpcs_hits_preprocessing` judges those and answers a
  complete run from its file. A changed hit count needs `--accept-count-change`.
- **Order.** `reprocess_all.sh launch` and let every job end, then `reprocess_all.sh xpcs-hits
  [RUNS]`, then `reprocess_safe.py check RUNS -d xpcs` and `cleanup`, then `reprocess_all.sh refresh`.

## 8. Acceptance (2026-10-13)

| Gate | Result |
|---|---|
| pytest, ruff | 654 passed, 12 skipped; ruff clean on every file touched. `src/analysis/utils.py` carries 27 ruff findings that were already on `main` and are untouched here |
| Mock tests | lock conflict, stale-lock takeover recorded in provenance, early return with inode and mtime unchanged and no npz or lock left, `WORKER_ERROR` retry, resume refused after a pyFAI or operator change, a 1-ULP `omega` from the other node class still complete — `tests/common/test_lock.py`, `tests/saxs/test_integrity.py`, `tests/waxs/test_waxs_integrity.py` |
| ctxrunner on the reconciled context | evaluates, 37 variables (§5) |
| **Real run** | below |

**The real-run test** (Slurm job 25293843, max-cssb006, Gold 6126, AVX-512):
- **What ran.** DAMNIT's own interpreter with `PYTHONPATH=<S1 worktree>/src` ran `ctxrunner exec
  10400 423 all --cluster-job --var agipd_iq_overview` on a scratch copy of the reconciled
  context. ctxrunner pulled in `agipd_saxs` as a dependency.
- **The pre-check.** The worktree's `analysis` was the one imported, and r0423 was complete, judged
  through the array comparison because the file was written on AVX2.
- **What happened.** `agipd_saxs` was "complete; answered from the file, nothing written" (13.9 s),
  and the overview took 3.4 s; 59 s wall in all, 3.6 GB peak RSS.
- **The verdict: pass.** `agipd_saxs.h5`, `operator.npz`, `masks.npz` and `agipd_pixel_sums.h5`
  kept their inode, mtime, size and sha256. No file appeared, no lock was left, and the fragment
  holds both variables with no `.errors`.
- **Where.** Probe, job script and log: `scratch/slurm_logs/s1_integrity/phaseD/`. The scratch copy
  wrote only its own `extracted_data/`, never `runs.sqlite`.

**Left for after the merge** (needs the user's OK):
- `python scripts/install_context.py --install --accept-live 44561ba487b3`, provided the live files
  have not changed since the 2026-10-07 state that id names;
- then `reprocess_safe.py` for the bulk reprocess, after S3 and S2.
