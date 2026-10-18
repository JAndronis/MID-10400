#!/usr/bin/env python
"""P5 check: the context file with the ``xpcs_hits_*`` variables, run as DAMNIT runs it.

For each run, in a scratch directory (never the live context, never ``read-context``,
CLAUDE.md pitfall 25):

1. copy ``src/amore/context.py`` and ``analysis_helpers.py`` beside each other;
2. ``python -m ctxrunner ctx`` and read ``error_info`` in the pickle (``ctx`` exits 0
   even for a broken context), then list the ``xpcs_hits_*`` variables;
3. ``python -m ctxrunner exec <proposal> <run> all --cluster-job --var
   xpcs_hits_overview``: that variable and every dependency, which includes the
   AGIPD pass (answered from its files when they are complete, so run
   ``analysis.saxs.complete.is_complete`` first);
4. ``exec --var xpcs_hits_frames_used`` (the one non-cluster variable);
5. read the cells back from ``extracted_data/``.

The context file runs unchanged. Only the harness redirects the pass's output root
(``analysis.xpcs.damnit.config_for`` is wrapped in the subprocess), so the photon
store and the results land under ``--out-root`` and not in ``usr/cached_files``.

    python scripts/xpcs_hits_context_check.py --run 493 -w 36
    python scripts/xpcs_hits_context_check.py --run 493 -w 36 --local

Needs Maxwell: proc, raw, DAMNIT's files. Not unit-tested (CLAUDE.md working rule 7).
"""

from __future__ import annotations

from _common import add_job_args, dispatch, report_header, write_report  # noqa: E402

import analysis
from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path("/gpfs/exfel/exp/MID/202601/p010400/scratch/xpcs_hits_p5")
BOOTSTRAP = """
import sys
import analysis.xpcs.damnit as d
orig = d.config_for
root = {root!r}
d.config_for = lambda p, r, **kw: orig(p, r, **({{"output_root": root}} | kw))
import ctxrunner
ctxrunner.main({argv!r})
"""

log = logging.getLogger("scripts")


def ctxsupport() -> Path:
    import damnit

    return Path(damnit.__file__).parent / "ctxsupport"


def child_env(workdir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ctxsupport()), str(REPO / "src"), env.get("PYTHONPATH", "")]
    )
    env["MPLBACKEND"] = "Agg"
    env["EXTRA_NUM_THREADS"] = env["OMP_NUM_THREADS"] = "1"
    return env


def check_context(workdir: Path) -> dict[str, Any]:
    """``ctx`` the way DAMNIT does, then read ``error_info`` from the pickle."""
    for name in ("context.py", "analysis_helpers.py"):
        shutil.copy(REPO / "src" / "amore" / name, workdir / name)
    started = time.perf_counter()
    done = subprocess.run(
        [sys.executable, "-m", "ctxrunner", "ctx", "context.py", "out.pkl"],
        cwd=workdir,
        env=child_env(workdir),
        capture_output=True,
        text=True,
    )
    sys.path.insert(0, str(ctxsupport()))  # the pickle holds ctxrunner's classes
    with open(workdir / "out.pkl", "rb") as handle:
        ctx, error_info = pickle.load(handle)
    names = sorted(n for n in (ctx.vars if ctx else {}) if n.startswith("xpcs_hits"))
    return {
        "returncode": done.returncode,
        "error_info": error_info,
        "n_variables": len(ctx.vars) if ctx else 0,
        "xpcs_hits_variables": {
            n: {
                "title": ctx.vars[n].title,
                "cluster": ctx.vars[n].cluster,
                "data": str(ctx.vars[n].data),
            }
            for n in names
        },
        "old_xpcs_pipeline_present": bool(ctx and "xpcs_pipeline" in ctx.vars),
        "seconds": time.perf_counter() - started,
    }


def exec_variables(
    workdir: Path, root: Path, proposal: int, run: int, variable: str, cluster: bool
) -> dict[str, Any]:
    argv = ["exec", str(proposal), str(run), "all", "--var", variable]
    if cluster:
        argv.insert(4, "--cluster-job")
    code = BOOTSTRAP.format(root=str(root), argv=argv)
    started = time.perf_counter()
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=workdir,
        env=child_env(workdir),
        capture_output=True,
        text=True,
    )
    (workdir / f"exec_{variable}.log").write_text(done.stdout + "\n---\n" + done.stderr)
    return {
        "argv": argv,
        "returncode": done.returncode,
        "seconds": time.perf_counter() - started,
        "log": str(workdir / f"exec_{variable}.log"),
        "errors": [line for line in done.stderr.splitlines() if " ERROR " in line][:20],
        "computed": [
            line.split("Computed ")[1]
            for line in done.stderr.splitlines()
            if "Computed " in line
        ],
        "skipped": [line for line in done.stderr.splitlines() if "Skip" in line][:10],
    }


def read_cells(workdir: Path, proposal: int, run: int) -> dict[str, Any]:
    import h5py

    out: dict[str, Any] = {}
    for path in sorted((workdir / "extracted_data").glob("*.h5")):
        with h5py.File(path, "r") as f:
            for name in f:
                if not name.startswith(("xpcs_hits", "agipd_saxs", "rep_rate")):
                    continue
                group = f[name]
                entry: dict[str, Any] = {"file": path.name}
                if "data" in group:
                    data = group["data"]
                    entry["dtype"] = str(data.dtype)
                    entry["shape"] = list(data.shape)
                for key, value in group.attrs.items():
                    entry[key] = value.tolist() if hasattr(value, "tolist") else value
                if (
                    "data" in group
                    and group["data"].dtype.kind in "OS"
                    and not group["data"].shape
                ):
                    raw = group["data"][()]
                    text = raw.decode() if isinstance(raw, bytes) else str(raw)
                    entry["text"] = text[:600]
                out[f"{path.name}:{name}"] = entry
    return out


def run_one(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO)
    workdir = args.out_root / "ctx" / f"r{args.run:04d}"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    root = args.out_root / "out"
    report: dict[str, Any] = report_header(
        proposal=args.proposal,
        run=args.run,
        out_root=str(root),
        workdir=str(workdir),
        analysis_package=analysis.__file__,
    )
    report["ctx"] = check_context(workdir)
    if report["ctx"]["error_info"] is not None:
        write_report(report, args.json, f"xpcs_hits_context_check_r{args.run:04d}")
        return 1
    report["exec_overview"] = exec_variables(
        workdir, root, args.proposal, args.run, "xpcs_hits_overview", cluster=True
    )
    report["exec_frames_used"] = exec_variables(
        workdir, root, args.proposal, args.run, "xpcs_hits_frames_used", cluster=False
    )
    report["cells"] = read_cells(workdir, args.proposal, args.run)
    report["outputs"] = (
        sorted(str(p.relative_to(root)) for p in root.rglob("*.h5"))
        if root.exists()
        else []
    )
    write_report(report, args.json, f"xpcs_hits_context_check_r{args.run:04d}")
    print(
        json.dumps(
            {k: report[k] for k in ("exec_overview", "exec_frames_used")}, default=str
        )[:3000]
    )
    return int(
        report["exec_overview"]["returncode"] != 0
        or report["exec_frames_used"]["returncode"] != 0
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--run", type=int, nargs="+", required=True)
    add_job_args(parser, workers=36, time="01:00:00")
    parser.add_argument("--out-root", type=Path, default=SCRATCH)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if len(args.run) > 1 and args.json:
        parser.error("--json names one run's file")
    return dispatch(
        args,
        args.run,
        lambda run: run_one(argparse.Namespace(**vars(args) | {"run": run})),
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
