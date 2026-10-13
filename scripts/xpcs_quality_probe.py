#!/usr/bin/env python
"""Probe for context/xpcs-quality.md: photon statistics per XPCS ROI on a few trains.

For each run, on ``--trains`` trains spread through it, per frame and ROI:
* the photon-count histogram h(k), k = 0 … K, plus an overflow bin
* S1 = Σk and S2 = Σk(k − 1) over the valid pixels, exactly
* W1 = Σw and W2 = Σw² over the same pixels, with w each pixel's run-mean count
  from the per-pixel window sums (usr/cached_files/agipd_pixel_sums), relative to
  its ROI

and from them, pooled over the frames:
* k̄ = ΣS1 / Σn
* the single-frame contrast as a factorial moment, unweighted and weighted:
      1 + β = ΣS2 / Σ[(S1² − S1)·W2/W1²]          (w ≡ 1 gives the unweighted form)
  E[k(k − 1)] = (1 + β)·μ² holds for the negative binomial and for any speckle
  statistics with that contrast; the weights take out the intensity variation
  across a wide ring, which would otherwise read as contrast. σ by jackknife over
  trains.
* the negative-binomial shape checks 2·P0·P2/P1² = 1 + β and 3·P1·P3/(2·P2²) =
  (1 + 2β)/(1 + β), whose departure flags pile-up or photonisation artefacts
* β per quarter of the pulse train (heating within a train, CLAUDE.md boundary 4)
* the cost per frame of a dense and a sparse histogram, and of the in-worker form
  that reuses what `integrate_frame` already has (the valid nonzero pixels, and
  the pixels where the frame's mask differs from its cell's base mask); all three
  must agree
* the gzip size per frame of the stored record, for the four XPCS ROIs and for the
  narrow rings

The geometry, static mask and per-frame bad pixels are the SAXS pass's own
(`build_operator`, `build_static_bad`, `frame_bad`); the operator's sha256 is checked
against the pass's output file. Nothing is written except the JSON beside this script
and a sample HDF5 under scratch/xpcs_quality_probe/ (for the size measurement).

    python scripts/xpcs_quality_probe.py --runs 423 450 --trains 20

Not unit-tested (CLAUDE.md rule 7): run it on a node.
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

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common.masks import frame_bad  # noqa: E402
from analysis.common.plan import evenly_spaced  # noqa: E402
from analysis.saxs.config import NPIX, SHAPE, AgipdSaxsConfig  # noqa: E402
from analysis.saxs.masks import build_static_bad, load_masks  # noqa: E402
from analysis.saxs.operator import build_operator, geometry_from_config  # noqa: E402
from analysis.saxs.pixel_sums import window_sums, window_table  # noqa: E402
from analysis.triage.xpcs import (  # noqa: E402
    BRAGG_WINDOWS_NM,
    REFERENCE_BETA,
    roi_edges_nm,
    roi_kinds,
)

PROP = Path("/gpfs/exfel/exp/MID/202601/p010400")
SUMS_ROOT = PROP / "usr/cached_files/agipd_pixel_sums"
SAMPLE_ROOT = PROP / "scratch/xpcs_quality_probe"
TRIAGE_REPORT = Path(__file__).with_name("run_triage_xpcs_20261013T152419Z.json")
#: Narrow rings for β(q) and for the width effect: this wide [nm⁻¹], inside the plain
#: XPCS ROIs, never touching a Bragg window.
NARROW_WIDTH = 0.025


def roi_sets() -> dict[str, np.ndarray]:
    """``xpcs``: the four plain XPCS ROIs; ``narrow``: rings of NARROW_WIDTH in them."""
    edges = roi_edges_nm()
    plain = edges[[k == "plain" for k in roi_kinds(edges)]]
    narrow = []
    for lo, hi in plain:
        n = max(int(round((hi - lo) / NARROW_WIDTH)), 1)
        bounds = np.linspace(lo, hi, n + 1)
        narrow += list(zip(bounds[:-1], bounds[1:], strict=True))
    narrow = np.array(narrow)
    for lo, hi in np.vstack([plain, narrow]):
        for w_lo, w_hi in BRAGG_WINDOWS_NM:
            assert not (lo < w_hi and hi > w_lo), (lo, hi)
    return {"xpcs": plain, "narrow": narrow}


def labels_for(q: np.ndarray, edges: np.ndarray, usable: np.ndarray) -> np.ndarray:
    label = np.full(q.size, -1, dtype=np.int32)
    for r, (lo, hi) in enumerate(edges):
        label[(q >= lo) & (q < hi) & usable] = r
    return label


def frame_stats_dense(c, bad, label, w, n_rois, k_max):
    valid = (label >= 0) & ~bad
    lab, val, wv = label[valid], c[valid].astype(np.int64), w[valid]
    hist = np.bincount(
        lab * (k_max + 2) + np.minimum(val, k_max + 1), minlength=n_rois * (k_max + 2)
    ).reshape(n_rois, k_max + 2)
    s1 = np.bincount(lab, weights=val, minlength=n_rois)
    s2 = np.bincount(lab, weights=val * (val - 1), minlength=n_rois)
    w1 = np.bincount(lab, weights=wv, minlength=n_rois)
    w2 = np.bincount(lab, weights=wv * wv, minlength=n_rois)
    return hist, s1, s2, w1, w2


def frame_stats_sparse(c, bad, label, w, n_rois, k_max, roi_n, roi_w1, roi_w2):
    """The same from the nonzero pixels and the bad pixels only."""
    nz = np.flatnonzero(c)
    nz = nz[(label[nz] >= 0) & ~bad[nz]]
    lab, val = label[nz], c[nz].astype(np.int64)
    hist = np.bincount(
        lab * (k_max + 2) + np.minimum(val, k_max + 1), minlength=n_rois * (k_max + 2)
    ).reshape(n_rois, k_max + 2)
    s1 = np.bincount(lab, weights=val, minlength=n_rois)
    s2 = np.bincount(lab, weights=val * (val - 1), minlength=n_rois)
    gone = np.flatnonzero(bad & (label >= 0))
    lab_bad = label[gone]
    n_valid = roi_n - np.bincount(lab_bad, minlength=n_rois)
    w1 = roi_w1 - np.bincount(lab_bad, weights=w[gone], minlength=n_rois)
    w2 = roi_w2 - np.bincount(lab_bad, weights=w[gone] ** 2, minlength=n_rois)
    hist[:, 0] = n_valid - hist[:, 1:].sum(axis=1)
    return hist, s1, s2, w1, w2


def frame_stats_reuse(x, keep, diff, base_bad, label, n_rois, k_max, base_n):
    """The in-worker form: only `integrate_frame`'s ``keep`` and ``diff`` are touched.

    ``base_n`` is the valid-pixel count per ROI under the cell's base mask; ``diff``
    moves it by the pixels this frame flags differently, exactly as the integrator
    corrects its denominator.
    """
    lab = label[keep]
    inside = lab >= 0
    lab, val = lab[inside], x[keep][inside].astype(np.int64)
    hist = np.bincount(
        lab * (k_max + 2) + np.minimum(val, k_max + 1), minlength=n_rois * (k_max + 2)
    ).reshape(n_rois, k_max + 2)
    s1 = np.bincount(lab, weights=val, minlength=n_rois)
    s2 = np.bincount(lab, weights=val * (val - 1), minlength=n_rois)
    dl = label[diff]
    sign = np.where(base_bad[diff], 1, -1)[dl >= 0]
    n_valid = base_n + np.bincount(dl[dl >= 0], weights=sign, minlength=n_rois)
    hist[:, 0] = n_valid - hist[:, 1:].sum(axis=1)
    return hist, s1, s2


def contrast(s1, s2, n, w1, w2):
    """1 + β from pooled factorial moments; with w ≡ 1 pass w1 = w2 = n."""
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = (s1 * s1 - s1) * w2 / (w1 * w1)
        return s2.sum(axis=0) / expected.sum(axis=0) - 1.0


def jackknife(fn, groups: np.ndarray, *arrays):
    """Value and leave-one-group-out σ of fn over the frame axis."""
    full = fn(*arrays)
    ids = np.unique(groups)
    loo = np.array([fn(*(a[groups != g] for a in arrays)) for g in ids])
    b = ids.size
    sigma = np.sqrt((b - 1) / b * ((loo - loo.mean(axis=0)) ** 2).sum(axis=0))
    return full, sigma


def nb_shape(hist):
    """Pooled 2·P0·P2/P1² and 3·P1·P3/(2·P2²) per ROI, and the counts behind them."""
    h = hist.sum(axis=0).astype(np.float64)  # (R, K+2)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 2 * h[:, 0] * h[:, 2] / h[:, 1] ** 2
        r3 = 3 * h[:, 1] * h[:, 3] / (2 * h[:, 2] ** 2)
    return r2, r3, h


def probe_run(run: int, n_trains: int, k_max: int, report_rois: dict) -> dict:
    from extra_data import by_id, open_run
    from extra_data.components import AGIPD1M

    out: dict = {"run": run}
    cfg = AgipdSaxsConfig(proposal=10400, run=run)
    t0 = time.perf_counter()
    op, ai = build_operator(geometry_from_config(cfg), cfg)
    out["operator_build_s"] = round(time.perf_counter() - t0, 1)
    with h5py.File(cfg.output_file, "r") as f:
        stored = f["operator"].attrs["sha256"]
    out["operator_sha256_matches_pass"] = bool(stored == op.sha256)
    q = np.asarray(ai.array_from_unit(SHAPE, "center", "q_nm^-1")).ravel()
    static = build_static_bad(cfg).bad
    base = load_masks(cfg.output_file.with_name("masks.npz"))

    t0 = time.perf_counter()
    sums_path = SUMS_ROOT / f"r{run:04d}" / "agipd_pixel_sums.h5"
    table = window_table(sums_path)
    sums = window_sums(sums_path, np.flatnonzero(table.written.values))
    counts = sums.counts.values.reshape(-1).astype(np.float64)
    valid = sums.valid_frames.values.reshape(-1).astype(np.float64)
    mean = np.divide(counts, valid, out=np.zeros(NPIX), where=valid > 0)
    out["pixel_sums_read_s"] = round(time.perf_counter() - t0, 1)

    sets = {}
    for name, edges in roi_sets().items():
        label = labels_for(q, edges, ~static & (valid > 0))
        w = np.zeros(NPIX)
        for r in range(len(edges)):
            sel = label == r
            w[sel] = mean[sel] / mean[sel].mean()
        n_rois = len(edges)
        sets[name] = {
            "edges": edges, "label": label, "w": w, "n_rois": n_rois,
            "roi_n": np.bincount(label[label >= 0], minlength=n_rois),
            "roi_w1": np.bincount(label[label >= 0], weights=w[label >= 0],
                                  minlength=n_rois),
            "roi_w2": np.bincount(label[label >= 0], weights=w[label >= 0] ** 2,
                                  minlength=n_rois),
            "w_cv": [float(w[label == r].std()) for r in range(n_rois)],
            "base_n": {
                int(cell): np.bincount(label[(label >= 0) & ~base.base_bad[i]],
                                       minlength=n_rois)
                for i, cell in enumerate(base.cells)
            },
            "zero_mean_pixels": [int(((label == r) & (mean == 0)).sum())
                                 for r in range(n_rois)],
            "records": {k: [] for k in ("hist", "s1", "s2", "w1", "w2")},
        }  # fmt: skip

    proc = open_run(10400, run, data="proc")
    det = AGIPD1M(proc, min_modules=16)
    trains = evenly_spaced(np.asarray(proc.train_ids), n_trains)
    timing = {
        k: 0.0 for k in ("read_data", "read_mask", "bad", "dense", "sparse", "reuse")
    }
    train_of, pulse_of, mismatches, n_frames = [], [], 0, 0
    for tid in trains:
        sel = det.select_trains(by_id[[int(tid)]])
        t = time.perf_counter()
        data = sel["image.data"].ndarray(decompress_threads=1)
        timing["read_data"] += time.perf_counter() - t
        t = time.perf_counter()
        mask = sel["image.mask"].ndarray(decompress_threads=1)
        timing["read_mask"] += time.perf_counter() - t
        pulses = sel["image.data"].pulse_id_coordinates()
        cells = sel["image.data"].cell_id_coordinates()
        tids = sel["image.data"].train_id_coordinates()
        assert np.all(tids == tid), f"r{run}: frames of another train in {tid}"
        for f in range(data.shape[1]):
            c = data[:, f].reshape(-1)
            t = time.perf_counter()
            bad = frame_bad(mask[:, f], cfg.mask_bits, static)
            timing["bad"] += time.perf_counter() - t
            # what integrate_frame has computed by now (not timed here)
            cell = int(cells[f])
            base_bad = base.for_cell(cell)[0]
            nz = np.flatnonzero(c)
            keep = nz[~bad[nz]]
            diff = np.flatnonzero(bad != base_bad)
            for name, s in sets.items():
                args = (c, bad, s["label"], s["w"], s["n_rois"], k_max)
                t = time.perf_counter()
                dense = frame_stats_dense(*args)
                if name == "xpcs":
                    timing["dense"] += time.perf_counter() - t
                t = time.perf_counter()
                sparse = frame_stats_sparse(*args, s["roi_n"], s["roi_w1"], s["roi_w2"])
                if name == "xpcs":
                    timing["sparse"] += time.perf_counter() - t
                same = all(
                    np.allclose(a, b, rtol=1e-12, atol=1e-9)
                    for a, b in zip(dense, sparse, strict=True)
                )
                if base.has_cell(cell):
                    t = time.perf_counter()
                    reuse = frame_stats_reuse(
                        c,
                        keep,
                        diff,
                        base_bad,
                        s["label"],
                        s["n_rois"],
                        k_max,
                        s["base_n"][cell],
                    )
                    if name == "xpcs":
                        timing["reuse"] += time.perf_counter() - t
                    same &= all(
                        np.allclose(a, b, rtol=1e-12, atol=1e-9)
                        for a, b in zip(dense[:3], reuse, strict=True)
                    )  # fmt: skip
                mismatches += not same
                for key, value in zip(("hist", "s1", "s2", "w1", "w2"), dense,
                                      strict=True):  # fmt: skip
                    s["records"][key].append(value)
            train_of.append(int(tid))
            pulse_of.append(int(pulses[f]))
            n_frames += 1

    out["n_trains"] = len(trains)
    out["n_frames"] = n_frames
    out["dense_sparse_mismatches"] = mismatches
    out["ms_per_frame"] = {k: round(1e3 * v / n_frames, 3) for k, v in timing.items()}
    groups = np.array(train_of)
    pulse_rank = np.unique(pulse_of, return_inverse=True)[1]
    quarter = np.minimum(4 * pulse_rank // (pulse_rank.max() + 1), 3)

    for name, s in sets.items():
        rec = {k: np.array(v) for k, v in s["records"].items()}
        n = rec["hist"].sum(axis=2).astype(np.float64)  # (F, R) valid pixels
        kbar = rec["s1"].sum(axis=0) / n.sum(axis=0)
        b_uni, sb_uni = jackknife(contrast, groups, rec["s1"], rec["s2"], n, n, n)
        b_w, sb_w = jackknife(contrast, groups, rec["s1"], rec["s2"], n,
                              rec["w1"], rec["w2"])  # fmt: skip
        r2, r3, h = nb_shape(rec["hist"])
        by_quarter = [
            contrast(rec["s1"][quarter == qq], rec["s2"][quarter == qq],
                     n[quarter == qq], rec["w1"][quarter == qq],
                     rec["w2"][quarter == qq]).tolist()
            for qq in range(4)
        ]  # fmt: skip
        kbar_pulse = np.array(
            [
                rec["s1"][pulse_rank == p].sum(axis=0) / n[pulse_rank == p].sum(axis=0)
                for p in range(pulse_rank.max() + 1)
            ]
        )
        rows = []
        for r, (lo, hi) in enumerate(s["edges"]):
            nonzero = np.flatnonzero(h[r])
            row = {
                "q_nm": [float(lo), float(hi)],
                "n_pix": int(s["roi_n"][r]),
                "w_cv": s["w_cv"][r],
                "zero_mean_pixels": s["zero_mean_pixels"][r],
                "kbar": float(kbar[r]),
                "beta_unweighted": float(b_uni[r]),
                "sigma_unweighted": float(sb_uni[r]),
                "beta_weighted": float(b_w[r]),
                "sigma_weighted": float(sb_w[r]),
                "nb_r2_minus_1": float(r2[r] - 1),
                "nb_r3": float(r3[r]),
                "nb_r3_expected": float((1 + 2 * b_w[r]) / (1 + b_w[r])),
                "counts_k": [int(x) for x in h[r][: min(nonzero.max() + 1, k_max + 2)]],
                "overflow": int(h[r][k_max + 1]),
                "beta_by_pulse_quarter": [float(b[r]) for b in by_quarter],
                "kbar_pulse_rel_std": float(kbar_pulse[:, r].std() / kbar[r]),
                "kbar_first20_over_rest": float(
                    kbar_pulse[:20, r].mean() / kbar_pulse[20:, r].mean()
                ),
            }
            if name == "xpcs":
                ttcf = [x for x in report_rois.get(str(run), [])
                        if abs(x["roi_q_A"] * 10 - (lo + hi) / 2) < 1e-6]  # fmt: skip
                if ttcf:
                    row["ttcf_A"] = ttcf[0]["A"]
                    row["ttcf_sigma_A"] = ttcf[0]["sigma_A"]
                ref = [v[0] for key, v in REFERENCE_BETA.items()
                       if abs(key * 10 - (lo + hi) / 2) < 1e-6]  # fmt: skip
                row["reference_beta"] = ref[0] if ref else None
            rows.append(row)
        out[name] = rows
        out.setdefault("storage", {})[name] = sample_size(
            run, name, rec, groups, pulse_of, k_max
        )
    return out


def sample_size(run, name, rec, groups, pulse_of, k_max) -> dict:
    """Bytes per frame of the per-frame record, gzip level 1, one chunk per train."""
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_ROOT / f"r{run:04d}_{name}_sample.h5"
    per_train = int(np.bincount(np.unique(groups, return_inverse=True)[1]).max())
    hist = rec["hist"].astype(np.uint32)
    with h5py.File(path, "w") as f:
        for name, value in (
            ("hist", hist),
            ("s1", rec["s1"].astype(np.uint32)),
            ("s2", rec["s2"].astype(np.uint32)),
            ("trainId", groups.astype(np.uint64)),
            ("pulseId", np.asarray(pulse_of, dtype=np.uint64)),
        ):
            f.create_dataset(name, data=value, compression="gzip", compression_opts=1,
                             chunks=(per_train, *value.shape[1:]))  # fmt: skip
    size = path.stat().st_size
    return {
        "file": str(path),
        "bytes": size,
        "bytes_per_frame": size / hist.shape[0],
        "uncompressed_bytes_per_frame": hist[0].nbytes + 2 * 4 * hist.shape[1] + 16,
        "k_max": k_max,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, nargs="+", default=[423, 450])
    parser.add_argument("--trains", type=int, default=20)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    report_rois = json.loads(TRIAGE_REPORT.read_text())["rois"]
    t0 = time.perf_counter()
    runs = [probe_run(r, args.trains, args.k_max, report_rois) for r in args.runs]
    report = report_header(
        script="xpcs_quality_probe", elapsed_s=round(time.perf_counter() - t0, 1),
        trains=args.trains, k_max=args.k_max, narrow_width_nm=NARROW_WIDTH,
        triage_report=TRIAGE_REPORT.name, runs=runs,
    )  # fmt: skip
    write_report(report, args.json, "xpcs_quality_probe")
    for r in runs:
        print(f"\nr{r['run']}: {r['n_frames']} frames, ms/frame {r['ms_per_frame']}, "
              f"mismatches {r['dense_sparse_mismatches']}, "
              f"operator ok {r['operator_sha256_matches_pass']}")  # fmt: skip
        for row in r["xpcs"]:
            print(f"  {row['q_nm']}: kbar {row['kbar']:.4f} "
                  f"beta_w {row['beta_weighted']:.4f}±{row['sigma_weighted']:.4f} "
                  f"beta_u {row['beta_unweighted']:.4f} "
                  f"ttcf_A {row.get('ttcf_A')} ref {row.get('reference_beta')} "
                  f"r2-1 {row['nb_r2_minus_1']:.4f} r3 {row['nb_r3']:.3f}/"
                  f"{row['nb_r3_expected']:.3f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
