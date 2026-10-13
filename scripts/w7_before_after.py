#!/usr/bin/env python
"""keV against photons on one run: what photon counting changed (open task 18).

Reads two finished output trees of the WAXS pass — one written before photon
counting (keV) and one after (photons), same run, both detectors — and reports,
as JSON beside this script:

* **the jf1/jf2 overlap** (``analysis.waxs.combine.combine_files``, §6 O5 and
  CLAUDE.md open task 5): scale factor, χ²ᵣ, fractional residual rms, its slope
  in q and ``σ/I``, for each unit;
* **per detector, the stored σ against the frames** (``analysis.waxs.scatter``):
  the σ ratio and the spread between trains, over the whole range and over the
  overlap;
* **per detector, keV over photons train by train**: ``I_keV / I_photons`` on
  the bins both populate. Both files integrate the same frames, so whatever
  moves that ratio from train to train is the zero level the keV pass carried —
  W7 predicts ~3.7 % rms on jf1 (a per-train common mode) and far less on jf2.

Login node is enough: it reads two pairs of output files and nothing else.

    python scripts/w7_before_after.py --run 423 \\
        --kev-root .../s2_jf_photonise/kev --photon-root .../s2_jf_photonise/photons
"""

from __future__ import annotations

from _common import report_header, write_report  # noqa: E402

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from analysis.common.writer import pooled_per_train  # noqa: E402
from analysis.waxs.combine import combine_files  # noqa: E402
from analysis.waxs.config import DETECTORS  # noqa: E402
from analysis.waxs.scatter import frame_scatter  # noqa: E402
from analysis.waxs.writer import KEV, PHOTONS, file_units  # noqa: E402

#: The q range jf1 and jf2 share on r0423 (§6 O5).
OVERLAP_NM = (11.53, 18.44)


def output_file(root: Path, run: int, detector: str) -> Path:
    return root / f"r{run:04d}" / f"jungfrau_waxs_{detector}.h5"


def overlap(jf1: Path, jf2: Path) -> dict[str, Any]:
    combined, scaling = combine_files(jf1, jf2)
    return scaling.summary() | {
        "sigma_over_intensity": scaling.sigma_over_intensity,
        "units": combined.attrs.get("units"),
    }


def kev_over_photons(kev: Path, photons: Path, energy_kev: float) -> dict[str, Any]:
    """Per-train ``I_keV / (E · I_photons)`` on the bins both files populate."""
    a = pooled_per_train(kev)
    b = pooled_per_train(photons)
    common = np.intersect1d(a["trainId"].values, b["trainId"].values)
    a = a.sel(trainId=common)
    b = b.sel(trainId=common)
    present = (a["n_frames"].values > 0) & (b["n_frames"].values > 0)
    ia = a["intensity"].values[present]
    ib = b["intensity"].values[present]
    bins = (ia > 0).all(axis=0) & (ib > 0).all(axis=0)
    per_train = ia[:, bins].mean(axis=1) / (energy_kev * ib[:, bins].mean(axis=1))
    per_q = ia[:, bins].mean(axis=0) / (energy_kev * ib[:, bins].mean(axis=0))
    q = a["q"].values[bins]
    return {
        "n_trains": int(per_train.size),
        "n_bins": int(bins.sum()),
        "mean": float(per_train.mean()),
        "relative_std_between_trains": float(per_train.std(ddof=1) / per_train.mean()),
        "per_q_p02_p50_p98": [float(v) for v in np.percentile(per_q, [2, 50, 98])],
        "per_q_first_last_tenth": [
            float(per_q[: max(per_q.size // 10, 1)].mean()),
            float(per_q[-max(per_q.size // 10, 1) :].mean()),
        ],
        "q_range": [float(q.min()), float(q.max())],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--kev-root", type=Path, required=True)
    parser.add_argument("--photon-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    files = {
        unit: {det: output_file(root, args.run, det) for det in DETECTORS}
        for unit, root in ((KEV, args.kev_root), (PHOTONS, args.photon_root))
    }
    for unit, paths in files.items():
        for path in paths.values():
            stored = file_units(path)
            if stored != unit:
                raise SystemExit(f"{path} stores {stored}, expected {unit}")

    report = report_header(
        run=args.run,
        files={u: {d: str(p) for d, p in ps.items()} for u, ps in files.items()},
        overlap_nm=list(OVERLAP_NM),
        overlap={},
        scatter={},
        kev_over_photons={},
    )
    for unit, paths in files.items():
        report["overlap"][unit] = overlap(paths["jf1"], paths["jf2"])
        report["scatter"][unit] = {
            det: {
                "whole_range": frame_scatter(path),
                "overlap": frame_scatter(path, q_range=OVERLAP_NM),
            }
            for det, path in paths.items()
        }
    import h5py

    for det in DETECTORS:
        with h5py.File(files[PHOTONS][det], "r") as handle:
            energy = float(np.unique(handle["frames/count_energy_kev"][:])[-1])
        report["kev_over_photons"][det] = kev_over_photons(
            files[KEV][det], files[PHOTONS][det], energy
        )

    for unit in (KEV, PHOTONS):
        o = report["overlap"][unit]
        rms = 100 * o["residual_rms"]
        print(
            f"{unit:8s} overlap: factor {o['factor']:.5f} ± {o['factor_error']:.5f}, "
            f"chi2r {o['reduced_chi2']:.2f}, residual rms {rms:.3f} %, "
            f"slope {o['residual_slope_per_nm']:.2e}/nm"
        )
        for det in DETECTORS:
            sc = report["scatter"][unit][det]["whole_range"]
            print(
                f"         {det}: sigma ratio {sc['sigma_ratio_median']:.3f}, "
                f"per-train relative std {sc['relative_std_between_trains']:.4f}"
            )
    for det in DETECTORS:
        k = report["kev_over_photons"][det]
        print(
            f"{det} keV/(E photons): mean {k['mean']:.4f}, per-train rms "
            f"{100 * k['relative_std_between_trains']:.2f} %"
        )
    write_report(report, args.json, f"w7_before_after_r{args.run:04d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
