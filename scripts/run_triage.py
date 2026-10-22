#!/usr/bin/env python
"""Run triage for p010400 (context/run-triage.md).

T1, ``--phase inventory``:
* one row per run from DAMNIT's database and files: class, effective sample,
  sizes, and which products exist
* the droplet series, and per-run droplet metrics
* the §9 T1 acceptance checks

T2a, ``--phase xpcs``:
* the analytical XPCS SNR (§4.6) per run and 30 s slice
* β from `REFERENCE_BETA`, one table for every run (user, 2026-10-13); the median
  over β-eligible runs and the reference runs' re-measurement are reported beside it
* the noise check ρ
* by default only on the T2a acceptance runs; ``--runs all`` takes every static and
  silica run with a hit-train XPCS results file (``--xpcs-root``)

T2b, ``--phase classify``:
* droplet hits for every mesh, static and reference run (``analysis.triage.classify``):
  mesh train classes per attenuator-and-sample block, out-of-beam trains of the rest
* each block's cut and each run's counts, flags and WAXS check, with their sources
* checks: r318 against lowq_report's own hit rule, the runs that must come out
  ambiguous (r68–69, r140) and all-on (r309–311)

The XPCS phase reads the hit-train results files and raw; the rest, DAMNIT and raw.
All read-only.
Nothing is written next to them.

    python scripts/run_triage.py --phase inventory
    python scripts/run_triage.py --phase xpcs --workers 8
    python scripts/run_triage.py --phase classify --workers 16

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
from analysis.triage.classify import (  # noqa: E402
    OFF_SIGMA,
    SEPARATION_BANDS,
    STATIC_QUANTILE,
    STATIC_WINDOW,
    classify_block,
    run_blocks,
    static_hits,
    train_signals,
    waxs_check,
)
from analysis.triage.inventory import (  # noqa: E402
    DAMNIT_ROOT,
    build_inventory,
    damnit_file,
    read_run_table,
)
from analysis.triage.series import load_droplet, segment  # noqa: E402
from analysis.triage.xpcs import (  # noqa: E402
    N_BLOCKS,
    REFERENCE_BETA_RUNS,
    SNR_MIN,
    crl_state,
    fixed_beta,
    measure_beta,
    measure_run,
    reference_beta,
    summarise,
)
from analysis.xpcs.config import DEFAULT_XPCS_ROOT  # noqa: E402

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


def results_path(run: int, root: str | Path = DEFAULT_XPCS_ROOT) -> Path:
    """A run's hit-train results file (`analysis.xpcs.result.read`'s path)."""
    return Path(root) / "xpcs_hits" / f"r{run:04d}" / "xpcs_hits.h5"


def _measure(job):
    """One run in a worker: its optics state and its `RunMeasurement`, or the error."""
    import extra_data as ed

    from analysis.xpcs.result import XpcsHits

    run, proposal, path = job
    t0 = time.perf_counter()
    try:
        optics = crl_state(ed.open_run(proposal, run, data="raw"))
        m = measure_run(run, XpcsHits(path))
        return run, optics, m, None, time.perf_counter() - t0
    except Exception as error:  # noqa: BLE001 - recorded per run, never fatal
        return run, None, None, f"{type(error).__name__}: {error}", 0.0


def _roi_rows(m, eligible) -> list[dict[str, Any]]:
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
                "A": fit.amplitude,
                "sigma_A": fit.sigma_amplitude,
                "gamma_per_s": fit.rate,
                "sigma_gamma": fit.sigma_rate,
                "beta_eligible": m.run in eligible[r].runs,
                "beta_reference": m.run in REFERENCE_BETA_RUNS,
            }
        )
    return rows


def _beta_block(roi_q, beta) -> dict[float, dict[str, Any]]:
    return {
        float(roi_q[r]): {"value": b.value, "n_eligible": b.n_eligible, "p16": b.p16,
                          "p84": b.p84, "runs": list(b.runs)}
        for r, b in beta.items()
    }  # fmt: skip


def xpcs_phase(args) -> dict[str, Any]:
    t0 = time.perf_counter()
    inv = build_inventory(read_run_table(args.damnit_root / "runs.sqlite"),
                          args.damnit_root, args.proposal).set_index("run")  # fmt: skip
    wanted = (inv.cls == "static_droplet") | (inv["sample"] == "silica_np")
    wanted &= pd.Series(
        [results_path(int(r), args.xpcs_root).exists() for r in inv.index],
        index=inv.index,
    )
    if args.runs == ["all"]:
        runs = [int(r) for r in inv.index[wanted & (inv.cls != "mesh_sshex")]]
    else:
        runs = [int(r) for r in (XPCS_ACCEPTANCE if not args.runs else args.runs)]
    jobs = [(r, args.proposal, str(results_path(r, args.xpcs_root))) for r in runs]
    with default_pool(min(args.workers, len(jobs))) as pool:
        results = list(pool.map(_measure, jobs))

    measured = {r: (opt, m, t) for r, opt, m, err, t in results if m is not None}
    errors = {r: err for r, _, _, err, _ in results if err is not None}
    states = pd.Series({r: v[0] for r, v in measured.items()})
    majority = states.mode().iloc[0] if len(states) else None
    same = [v[1] for r, v in measured.items() if v[0] == majority]
    roi_q = same[0].roi_q_A if same else np.zeros(0)
    beta = fixed_beta(roi_q)
    beta_eligible_median = measure_beta(same)  # cross-check only
    has_reference = {m.run for m in same} >= set(REFERENCE_BETA_RUNS)
    remeasured = reference_beta(same) if has_reference else None

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
        roi_tables[r] = _roi_rows(m, beta_eligible_median)
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
    by_run = {s["run"]: s for s in summaries}
    checks = {
        "rho_within": {
            "tolerance": RHO_TOLERANCE,
            "runs_outside": {
                r: v for r, v in rhos.items()
                if any(np.isfinite(x) and abs(x - 1) > RHO_TOLERANCE for x in v)
            },
        },
        "beta_eligible_per_plain_roi": {
            float(roi_q[r]): beta_eligible_median[r].n_eligible for r in plain
        },
        # REFERENCE_BETA is reproduced from its own runs by this code (None when the
        # reference runs are not all in this set)
        "reference_beta_reproduced": None if remeasured is None else {
            float(roi_q[r]): {
                "frozen": beta[r].value, "remeasured": remeasured[r].value,
                "agree": bool(abs(remeasured[r].value - beta[r].value)
                              <= 1e-6 + 1e-4 * beta[r].value),
            }
            for r in range(len(roi_q))
        },
        "r450_r400_extractable": {
            r: by_run.get(r, {}).get("extractable") for r in (450, 400)
        },
    }  # fmt: skip
    reproduced = checks["reference_beta_reproduced"]
    passed = (
        not checks["rho_within"]["runs_outside"]
        and all(checks["r450_r400_extractable"].values())
        and (reproduced is None or all(v["agree"] for v in reproduced.values()))
    )
    table = pd.DataFrame(
        [{k: v for k, v in s.items() if k != "snr_run"} for s in summaries]
    )
    return {
        "phase": "xpcs",
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "snr_min": SNR_MIN,
        "optics_majority": majority,
        "beta_source": f"REFERENCE_BETA, runs {list(REFERENCE_BETA_RUNS)}",
        "beta": _beta_block(roi_q, beta),
        "beta_eligible_median": _beta_block(roi_q, beta_eligible_median),
        "checks": checks,
        "passed": bool(passed),
        "errors": errors,
        "runs": summaries,
        "rois": roi_tables,
        "slices": slice_rows,
        "_table": table,
    }  # fmt: skip


def _signals(run: int):
    """One run's log10(s/F) and log10(ring/F), or why not, in a worker."""
    try:
        ds = train_signals(run)
    except Exception as error:  # noqa: BLE001 - recorded per run, never fatal
        return run, None, None, None, f"{type(error).__name__}: {error}"
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.log10((ds.saxs_mid / ds.flux).values)
        w = np.log10((ds.waxs_ring / ds.flux).values)
    return run, ds.trainId.values, x, w, None


def _r318_against_lowq(classes, tids) -> dict:
    """lowq_report cell 17's own rule on r318, train by train, against our classes.

    Its rule: hit ⇔ water (0.85–0.95 nm⁻¹) per flux > 2 × the median at the lowest Z.
    """
    import extra_data as ed
    import h5py
    import xarray as xr

    from analysis.diagnostics.curves import per_train_flux

    raw = ed.open_run(10400, 318, data="raw")
    src = "MID_EXP_SAM/HEXAPOD/SSHEX"
    z = raw[src, "Z.actualPosition"].xarray()
    path = damnit_file(318, DAMNIT_ROOT, 10400)
    with h5py.File(path, "r") as f:
        g = f["agipd.saxs"]
        q = g["q"][:]
        c = np.flatnonzero((q >= 0.85) & (q < 0.95))
        frames = g["n_frames"][:] > 0
        band = g["intensity"][:, :, c[0] : c[-1] + 1].mean(axis=-1)
        water = np.where(frames, band, 0).sum(1) / np.maximum(frames.sum(1), 1)
        water = xr.DataArray(water, dims="trainId", coords={"trainId": g["trainId"][:]})
    water = water / per_train_flux(path).flux
    ours = xr.DataArray(classes, dims="trainId", coords={"trainId": tids})
    ours, z, water = xr.align(ours, z, water, join="inner")
    low = z < float(z.min()) + 0.05
    hit = (water > 2 * float(water.where(low).median())).values
    lowq = pd.Series(np.where(hit, "hit", "miss"), name="lowq")
    table = pd.crosstab(pd.Series(ours.values, name="ours"), lowq)
    agree = int(
        ((ours.values == "on") & hit).sum() + ((ours.values == "off") & ~hit).sum()
    )
    decided = int(np.isin(ours.values, ("on", "off")).sum())
    return {"crosstab": table.to_dict(), "agreement": agree / max(decided, 1),
            "n_decided": decided}  # fmt: skip


def classify_phase(args) -> dict[str, Any]:
    t0 = time.perf_counter()
    blocks = run_blocks(read_run_table(args.damnit_root / "runs.sqlite"))
    wanted = blocks.index[
        blocks.cls.isin(["mesh_sshex", "static_droplet", "reference"])
    ]
    with default_pool(min(args.workers, len(wanted))) as pool:
        results = list(pool.map(_signals, [int(r) for r in wanted]))
    signals = {r: (t, x, w) for r, t, x, w, err in results if err is None}
    errors = {r: err for r, *_, err in results if err is not None}

    rows, cuts = [], {}
    is_mesh = {r: isinstance(blocks.loc[r, "block"], str) for r in signals}
    mesh = blocks.loc[[r for r in signals if is_mesh[r]]]
    for block, members in mesh.groupby("block").groups.items():
        out = classify_block({int(r): signals[int(r)][1] for r in members})
        cuts[block] = {k: v for k, v in out["block"].items()}
        for r in members:
            r = int(r)
            v = out["runs"][r]
            rows.append({"run": r, "kind": "mesh", "block": block,
                         **{k: v[k] for k in ("n_on", "n_edge", "n_off", "n_none",
                                              "ambiguous", "two_classes", "reference",
                                              "off", "sigma", "high",
                                              "plateau", "plateau_source")},
                         **waxs_check(signals[r][2], out["classes"][r])})  # fmt: skip
            if r == 318:
                r318 = _r318_against_lowq(out["classes"][r], signals[r][0])
    for r, (_, x, w) in signals.items():
        if is_mesh[r]:
            continue
        on = static_hits(x)
        cls = np.where(on, "on", np.where(np.isfinite(x), "off", "none"))
        rows.append({"run": r, "kind": "static", "block": None,
                     "n_on": int(on.sum()), "n_edge": 0,
                     "n_off": int((cls == "off").sum()),
                     "n_none": int((cls == "none").sum()), "ambiguous": False,
                     **waxs_check(w, cls)})  # fmt: skip
    table = pd.DataFrame(rows).sort_values("run")
    by_run = table.set_index("run")
    checks = {
        "r318_vs_lowq": r318 if 318 in by_run.index else None,
        "ambiguous_as_expected": {
            r: bool(by_run.loc[r, "ambiguous"]) for r in (68, 69, 140)
            if r in by_run.index
        },
        "all_on_as_expected": {
            # the beam never left the droplet: no off train, and at most 1 % of the
            # trains below a full hit's overlap (§4.4, 2026-10-18)
            r: bool(by_run.loc[r, "n_on"] > 0 and by_run.loc[r, "n_off"] == 0
                    and by_run.loc[r, "n_edge"]
                    <= 0.01 * (by_run.loc[r, "n_on"] + by_run.loc[r, "n_edge"]))
            for r in (309, 310, 311) if r in by_run.index
        },
    }  # fmt: skip
    passed = (
        checks["r318_vs_lowq"] is not None
        and checks["r318_vs_lowq"]["agreement"] >= 0.99
        and all(checks["ambiguous_as_expected"].values())
        and all(checks["all_on_as_expected"].values())
    )
    return {
        "phase": "classify",
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "sources": {
            "signal": "DAMNIT agipd_saxs 0.3-0.5 nm^-1 / (xgm_intensity x "
                      "total_transmission), run-triage D5",
            "off_band": f"{OFF_SIGMA} robust sigma of the run's own off class (§4.3)",
            "hit_cut": "centre of the widest empty band of the block's pooled values "
                       "relative to each run's off level, between the off band and "
                       "the hits' mode (§4.4 on the relative axis)",
            "separation": f"hits need a mode >= {SEPARATION_BANDS} off bands above "
                          "the off level",
            "static": f"more than {OFF_SIGMA} robust sigma below a running "
                      f"{STATIC_QUANTILE} quantile over {STATIC_WINDOW} trains, "
                      "residuals centred, iterated",
        },
        "blocks": cuts,
        "checks": checks,
        "passed": bool(passed),
        "errors": errors,
        "runs": records(table),
        "_table": table,
    }  # fmt: skip


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["inventory", "xpcs", "classify"], default="inventory"
    )
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--damnit-root", type=Path, default=DAMNIT_ROOT)
    parser.add_argument("--runs", nargs="*", default=None, help="run numbers, or 'all'")
    parser.add_argument("--xpcs-root", type=Path, default=Path(DEFAULT_XPCS_ROOT),
                        help="where xpcs_hits/r####/xpcs_hits.h5 live")  # fmt: skip
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def print_xpcs(report) -> None:
    print(f"\n{len(report['runs'])} runs in {report['elapsed_s']} s; optics "
          f"{report['optics_majority']}")  # fmt: skip
    print(f"beta per ROI ({report['beta_source']}):")
    for q, b in report["beta"].items():
        print(f"  {q:.4f} A^-1: {b['value']:.4f} (16-84 % {b['p16']:.4f}-{b['p84']:.4f}, "
              f"n = {b['n_eligible']}: {b['runs']})")  # fmt: skip
    cols = ["run", "sample", "n_usable", "n_slices", "n_extractable_slices",
            "best_snr", "best_roi_q_A", "t_needed_s", "extractable", "seconds"]  # fmt: skip
    table = pd.DataFrame(report["runs"])
    print(table[[c for c in cols if c in table]].round(2).to_string(index=False))
    print("\nper plain ROI (kbar, n_pix, phi, rho):")
    for r, rows in report["rois"].items():
        print(f"  r{r}: " + "; ".join(
            f"{x['roi_q_A']:.4f}: {x['kbar']:.2e} {x['n_pix']:.0f} {x['phi']:.2f} "
            f"{x['rho']:.2f}"
            for x in rows if x["kind"] == "plain"))  # fmt: skip
    if report["errors"]:
        print("errors:", report["errors"])
    print("\nchecks:", json.dumps(report["checks"], indent=1, default=str))
    print(f"\npassed: {report['passed']}")


def main(argv=None) -> int:
    args = parse_args(argv)
    phase = {
        "inventory": inventory_phase,
        "xpcs": xpcs_phase,
        "classify": classify_phase,
    }[args.phase]
    report = report_header(proposal=args.proposal) | phase(args)
    table = report.pop("_table")
    path = write_report(report, args.json, f"run_triage_{args.phase}")
    table.to_csv(path.with_suffix(".csv"), index=False)
    print(f"wrote {path.with_suffix('.csv')}")
    if args.phase == "xpcs":
        print_xpcs(report)
        return 0 if report["passed"] else 1
    if args.phase == "classify":
        print(f"\n{len(report['runs'])} runs in {report['elapsed_s']} s; "
              f"errors {report['errors']}")  # fmt: skip
        for block, cut in report["blocks"].items():
            print(f"  {block}: t_on {cut['t_on']:.3f} "
                  f"on_mode {cut['on_mode']:.2f} off_cut {cut['off_cut']:.3f} "
                  f"ambiguous {cut['ambiguous']}")  # fmt: skip
        print("\nchecks:", json.dumps(report["checks"], indent=1, default=str))
        print(f"\npassed: {report['passed']}")
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
