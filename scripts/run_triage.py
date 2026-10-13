#!/usr/bin/env python
"""Run triage for p010400 (context/run-triage.md).

T1, ``--phase inventory``:
* one row per run from DAMNIT's database and files: class, effective sample,
  sizes, and which products exist
* the droplet series, and per-run droplet metrics
* the §9 T1 acceptance checks

T2a, ``--phase xpcs``:
* the analytical XPCS SNR (§4.6) per run and 30 s slice
* β measured from the runs themselves
* the noise check ρ and the cross-check against DAMNIT's own g2
* by default only on the T2a acceptance runs; ``--runs all`` takes every static and
  silica run with an XPCS result

Everything is read from DAMNIT, the SAXS pass's scratch files and raw, read-only.
Nothing is written next to them.

    python scripts/run_triage.py --phase inventory
    python scripts/run_triage.py --phase xpcs --workers 8

Writes run_triage_<phase>_<stamp>.json beside this script, or at --json, plus a CSV
of the per-run rows next to it. Not unit-tested (CLAUDE.md rule 7): check a change
by running it.
"""

from __future__ import annotations

# Before numpy or h5py import anything that builds a thread pool.
from _common import report_header, write_report  # noqa: E402

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.common.cpu import default_pool  # noqa: E402
from analysis.saxs.config import DEFAULT_OUTPUT_ROOT as SAXS_ROOT  # noqa: E402
from analysis.triage.inventory import (  # noqa: E402
    DAMNIT_ROOT,
    build_inventory,
    damnit_file,
    read_run_table,
)
from analysis.triage.series import load_droplet, segment  # noqa: E402
from analysis.triage.stats import z_for_tests  # noqa: E402
from analysis.triage.xpcs import (  # noqa: E402
    N_BLOCKS,
    SNR_MIN,
    crl_state,
    measure_beta,
    measure_run,
    summarise,
)

#: §9 T1: series the notebooks and the user already established, as run lists.
KNOWN_SERIES: dict[str, list[int]] = {
    "r443-452": list(range(443, 453)),
    "r464-466 buffer": [464, 465, 466],
    "r338-369 (r349 no fit; calibration r353-355 skipped)": [
        r for r in range(338, 370) if r not in (353, 354, 355)
    ],
    "r370-381 (across the r378-379 pause)": list(range(370, 382)),
    "r389 silica, cut short": [389],
    "r390-393": list(range(390, 394)),
    "r308-337": list(range(308, 338)),
}
#: §4.2 known cases.
KNOWN_FRESH = [(143, 144)]
KNOWN_JUMPS = [463, 466]
#: §3.1, raw total over every run.
RAW_TOTAL_TB = 994.8
#: §9 T2a: early and late static droplets of each sample, crystallised r425/426, and
#: enough late, slow runs that β has several eligible runs per ROI.
XPCS_ACCEPTANCE = [370, 398, 400, 410, 423, 425, 426, 441, 450, 451, 495]
#: §4.6 noise check: ρ within twice the jackknife's own relative uncertainty.
RHO_TOLERANCE = 2 * float(np.sqrt(1 / (2 * (N_BLOCKS - 1))))


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-ready rows: NaN and NaT become None."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def inventory_phase(args) -> dict[str, Any]:
    t0 = time.perf_counter()
    runs = read_run_table(args.damnit_root / "runs.sqlite")
    inv = build_inventory(runs, args.damnit_root, args.proposal)
    droplets = {}
    for run in inv.run[inv.var_droplet_fit == "present"]:
        d = load_droplet(damnit_file(int(run), args.damnit_root, args.proposal), run)
        if d is not None:
            droplets[int(run)] = d
    result = segment(inv, droplets)
    table = inv.merge(result.runs, on="run", how="left")
    table["errors"] = table.errors.map(json.dumps)

    by_class = (
        table.groupby("cls")
        .agg(n=("run", "size"), raw_tb=("raw_tb", "sum"), proc_tb=("proc_tb", "sum"))
        .round(1)
        .reset_index()
    )
    series = result.series
    members = {
        sid: set(r) for sid, r in zip(series.series_id, series.runs, strict=True)
    }
    kinds = {
        (int(b), int(a)): k
        for b, a, k in zip(
            result.boundaries.before,
            result.boundaries.after,
            result.boundaries.kind,
            strict=True,
        )
    }
    flagged = table.set_index("run").volume_jump.fillna(False).astype(bool)
    checks = {
        "raw_total_tb": {
            "expected": RAW_TOTAL_TB,
            "got": round(float(table.raw_tb.sum()), 1),
        },
        "known_series": {
            name: {
                "expected": runs_,
                "found_in": [sid for sid, m in members.items() if set(runs_) <= m],
                "exact": any(set(runs_) == m for m in members.values()),
            }
            for name, runs_ in KNOWN_SERIES.items()
        },
        "known_fresh": {f"r{a}->r{b}": kinds.get((a, b)) for a, b in KNOWN_FRESH},
        "known_jumps": {f"r{r}": bool(flagged.get(r, False)) for r in KNOWN_JUMPS},
    }
    checks["raw_total_tb"]["passed"] = (
        checks["raw_total_tb"]["got"] == checks["raw_total_tb"]["expected"]
    )
    passed = (
        checks["raw_total_tb"]["passed"]
        and all(v["exact"] for v in checks["known_series"].values())
        and all(v == "fresh" for v in checks["known_fresh"].values())
        and all(checks["known_jumps"].values())
    )
    bounds = result.boundaries
    increases = bounds[bounds.kind == "increase"]
    flagged_runs = table[table.volume_jump.fillna(False).astype(bool)]
    return {
        "phase": "inventory",
        "damnit_root": str(args.damnit_root),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "n_runs": len(table),
        "by_class": records(by_class),
        "thresholds": result.thresholds,
        "checks": checks,
        "passed": passed,
        "increase_boundaries": records(increases),
        "label_suspect_boundaries": records(bounds[bounds.label_suspect]),
        "n_underpowered_boundaries": int(bounds.underpowered.sum()),
        "volume_jump_runs": [int(r) for r in flagged_runs.run],
        "n_volume_glitch_runs": int(table.volume_glitch.fillna(False).sum()),
        "series": records(series),
        "boundaries": records(result.boundaries),
        "runs": records(table),
        "_table": table,
    }


def _measure(job):
    """One run in a worker: its optics state and its `RunMeasurement`, or the error."""
    import extra_data as ed

    run, proposal, damnit_path, scratch_path, rep_rate_hz = job
    t0 = time.perf_counter()
    try:
        optics = crl_state(ed.open_run(proposal, run, data="raw"))
        m = measure_run(run, Path(damnit_path), Path(scratch_path), rep_rate_hz)
        return run, optics, m, None, time.perf_counter() - t0
    except Exception as error:  # noqa: BLE001 - recorded per run, never fatal
        return run, None, None, f"{type(error).__name__}: {error}", 0.0


def _roi_rows(m, beta) -> list[dict[str, Any]]:
    rows = []
    for r, kind in enumerate(m.kinds):
        fit = m.fits[r]
        rows.append(
            {
                "roi_q_A": float(m.roi_q_A[r]),
                "roi_nm": [float(x) for x in m.edges_nm[r]],
                "kind": kind,
                "kbar": float(m.kbar[r]),
                "n_pix": float(m.npix[r]),
                "phi": float(m.phi[r]),
                "rho": float(m.rho[r]),
                "damnit_max_z": float(m.damnit_max_z[r]),
                "A": fit.amplitude,
                "sigma_A": fit.sigma_amplitude,
                "gamma_per_s": fit.rate,
                "sigma_gamma": fit.sigma_rate,
                "beta_eligible": m.run in beta[r].runs,
            }
        )
    return rows


def xpcs_phase(args) -> dict[str, Any]:
    t0 = time.perf_counter()
    inv = build_inventory(read_run_table(args.damnit_root / "runs.sqlite"),
                          args.damnit_root, args.proposal).set_index("run")  # fmt: skip
    wanted = (inv.cls == "static_droplet") | (inv["sample"] == "silica_np")
    wanted &= inv.var_xpcs_ttcf == "present"
    if args.runs == ["all"]:
        runs = [int(r) for r in inv.index[wanted & (inv.cls != "mesh_sshex")]]
    else:
        runs = [int(r) for r in (XPCS_ACCEPTANCE if not args.runs else args.runs)]
    jobs = [
        (r, args.proposal, str(damnit_file(r, args.damnit_root, args.proposal)),
         str(Path(SAXS_ROOT) / f"r{r:04d}" / "agipd_saxs.h5"),
         float(inv.loc[r, "rep_rate_mhz"]) * 1e6)
        for r in runs
    ]  # fmt: skip
    with default_pool(min(args.workers, len(jobs))) as pool:
        results = list(pool.map(_measure, jobs))

    measured = {r: (opt, m, t) for r, opt, m, err, t in results if m is not None}
    errors = {r: err for r, _, _, err, _ in results if err is not None}
    states = pd.Series({r: v[0] for r, v in measured.items()})
    majority = states.mode().iloc[0] if len(states) else None
    same = [v[1] for r, v in measured.items() if v[0] == majority]
    beta = measure_beta(same)

    summaries, roi_tables, slice_rows = [], {}, []
    for r, (opt, m, seconds) in sorted(measured.items()):
        if opt != majority:
            summaries.append({"run": r, "optics_differs": opt})
            continue
        s = summarise(m, beta)
        slices = s.pop("slices")
        s.update(seconds=round(seconds, 1), sample=inv.loc[r, "sample"],
                 cls=inv.loc[r, "cls"], optics=opt)  # fmt: skip
        summaries.append(s)
        roi_tables[r] = _roi_rows(m, beta)
        for sl in slices:
            slice_rows.append(
                {"run": r, "slice": sl["slice"], "n_usable": sl["n_usable"],
                 "best_snr": sl["best"],
                 **{f"snr_{float(q):.4f}A": float(v)
                    for q, v in zip(m.roi_q_A, sl["snr"], strict=True)}}
            )  # fmt: skip

    plain = [r for r, k in enumerate(next(iter(measured.values()))[1].kinds)
             if k == "plain"] if measured else []  # fmt: skip
    rhos = {
        r: [row["rho"] for row in rows if row["kind"] == "plain"]
        for r, rows in roi_tables.items()
    }
    n_cmp = sum(
        np.isfinite(row["damnit_max_z"]) for rows in roi_tables.values() for row in rows
    )
    z_cut = z_for_tests(max(int(n_cmp) * 10, 1), two_sided=True)
    by_run = {s["run"]: s for s in summaries}
    checks = {
        "rho_within": {
            "tolerance": RHO_TOLERANCE,
            "runs_outside": {
                r: v for r, v in rhos.items()
                if any(np.isfinite(x) and abs(x - 1) > RHO_TOLERANCE for x in v)
            },
        },
        "damnit_g2": {
            "z_cut": z_cut,
            "outside": {
                r: [row["roi_q_A"] for row in rows if row["damnit_max_z"] > z_cut]
                for r, rows in roi_tables.items()
                if any(row["damnit_max_z"] > z_cut for row in rows)
            },
        },
        "beta_eligible_per_plain_roi": {
            float(next(iter(measured.values()))[1].roi_q_A[r]): beta[r].n_eligible
            for r in plain
        },
        "r450_r400_extractable": {
            r: by_run.get(r, {}).get("extractable") for r in (450, 400)
        },
    }  # fmt: skip
    passed = (
        not checks["rho_within"]["runs_outside"]
        and not checks["damnit_g2"]["outside"]
        and all(n >= 3 for n in checks["beta_eligible_per_plain_roi"].values())
        and all(checks["r450_r400_extractable"].values())
    )
    table = pd.DataFrame(
        [{k: v for k, v in s.items() if k != "snr_run"} for s in summaries]
    )
    return {
        "phase": "xpcs",
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "snr_min": SNR_MIN,
        "optics_majority": majority,
        "beta": {
            float(next(iter(measured.values()))[1].roi_q_A[r]): {
                "value": b.value, "n_eligible": b.n_eligible, "p16": b.p16,
                "p84": b.p84, "runs": list(b.runs),
            }
            for r, b in beta.items()
        } if measured else {},
        "checks": checks,
        "passed": bool(passed),
        "errors": errors,
        "runs": summaries,
        "rois": roi_tables,
        "slices": slice_rows,
        "_table": table,
    }  # fmt: skip


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["inventory", "xpcs"], default="inventory")
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--damnit-root", type=Path, default=DAMNIT_ROOT)
    parser.add_argument("--runs", nargs="*", default=None, help="run numbers, or 'all'")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def print_xpcs(report) -> None:
    print(f"\n{len(report['runs'])} runs in {report['elapsed_s']} s; optics "
          f"{report['optics_majority']}")  # fmt: skip
    print("beta per ROI:")
    for q, b in report["beta"].items():
        print(f"  {q:.4f} A^-1: {b['value']:.4f} (16-84 % {b['p16']:.4f}-{b['p84']:.4f}, "
              f"n = {b['n_eligible']}: {b['runs']})")  # fmt: skip
    cols = ["run", "sample", "n_usable", "n_slices", "n_extractable_slices",
            "best_snr", "best_roi_q_A", "t_needed_s", "extractable", "seconds"]  # fmt: skip
    table = pd.DataFrame(report["runs"])
    print(table[[c for c in cols if c in table]].round(2).to_string(index=False))
    print("\nper plain ROI (kbar, n_pix, phi, rho, damnit_max_z):")
    for r, rows in report["rois"].items():
        print(f"  r{r}: " + "; ".join(
            f"{x['roi_q_A']:.4f}: {x['kbar']:.2e} {x['n_pix']:.0f} {x['phi']:.2f} "
            f"{x['rho']:.2f} {x['damnit_max_z']:.1f}"
            for x in rows if x["kind"] == "plain"))  # fmt: skip
    if report["errors"]:
        print("errors:", report["errors"])
    print("\nchecks:", json.dumps(report["checks"], indent=1, default=str))
    print(f"\npassed: {report['passed']}")


def main(argv=None) -> int:
    args = parse_args(argv)
    phase = {"inventory": inventory_phase, "xpcs": xpcs_phase}[args.phase]
    report = report_header(proposal=args.proposal) | phase(args)
    table = report.pop("_table")
    path = write_report(report, args.json, f"run_triage_{args.phase}")
    table.to_csv(path.with_suffix(".csv"), index=False)
    print(f"wrote {path.with_suffix('.csv')}")
    if args.phase == "xpcs":
        print_xpcs(report)
        return 0 if report["passed"] else 1

    print(f"\n{report['n_runs']} runs in {report['elapsed_s']} s")
    print(pd.DataFrame(report["by_class"]).to_string(index=False))
    print("\nthresholds:", json.dumps(report["thresholds"], indent=1, default=str))
    print("\nchecks:", json.dumps(report["checks"], indent=1, default=str))
    print(f"\nincrease boundaries: {len(report['increase_boundaries'])}")
    for b in report["increase_boundaries"]:
        print(f"  r{b['before']}->r{b['after']}: d={b['d']:+.3f} z={b['z']:.1f}")
    for b in report["label_suspect_boundaries"]:
        verdict = b["label_checked"] or "NOT CHECKED"
        print(
            f"label suspect: r{b['before']}->r{b['after']} (d={b['d']:+.3f}): {verdict}"
        )
    print(f"underpowered boundaries: {report['n_underpowered_boundaries']}")
    print(f"volume jumps: {report['volume_jump_runs']}")
    print(f"glitch-sized steps: {report['n_volume_glitch_runs']} runs")
    print(f"\npassed: {report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
