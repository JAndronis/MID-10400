#!/usr/bin/env python
"""Why the pooled 3-photon ratio fails on whole runs (context/xpcs-quality.md §7).

`nb_checks` compares R3 = 3·P1·P3/(2·P2²) of the run's pooled histogram with the
negative binomial's (1 + 2β)/(1 + β). Pooled over frames and pixels of different
mean, R3 also carries
* the frame factor A3_frame = <m><m³>/<m²>² over frames (m = S1/W1, from the
  per-frame factorial moments of S1), and
* the pixel factor A3_pix = <w><w³>/<w²>² over the ring's pixels.
This script prints r3_z before and after both factors, and the per-frame
normalised third factorial moment, which neither factor touches:

    g3 = Σ_f S3 / Σ_f S1(S1 − 1)(S1 − 2)·W3/W1³,   expected (1 + β)(1 + 2β)

with S3 = Σ_k k(k − 1)(k − 2)·h_k from the stored histogram (the overflow bin
k > K is left out and counted). σ(g3) by jackknife over 50 blocks of trains.
Mesh runs use their droplet hits only (`triage.classify.hits`).

Reads only the photon-statistics files, window sums and masks (login node):

    python scripts/photon_stats_shape_check.py --runs 450 423 318

Writes photon_stats_shape_check_<stamp>.json beside itself. Not unit-tested
(CLAUDE.md rule 7).
"""

from __future__ import annotations

from _common import report_header, write_report  # noqa: E402

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.diagnostics.photon_stats import (  # noqa: E402
    N_BLOCKS,
    contrast,
    read_photon_stats,
    run_weights,
)
from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.saxs.masks import load_masks  # noqa: E402
from analysis.saxs.photon_stats import PhotonStatsConfig  # noqa: E402
from analysis.saxs.pixel_sums import window_sums, window_table  # noqa: E402


def _round(values, digits=5) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(values)]


def pixel_moments(path, saxs, windows, cells):
    """W1, W2, W3 per (cell, ring), as `ring_weights` builds W1 and W2."""
    with h5py.File(path, "r") as f:
        labels = f["rings/labels"][:]
    sums = window_sums(saxs.pixel_sums_file, windows)
    counts = sums.counts.values.reshape(-1).astype(np.float64)
    valid = sums.valid_frames.values.reshape(-1).astype(np.float64)
    mean = np.divide(counts, valid, out=np.zeros_like(counts), where=valid > 0)
    n_rings = int(labels.max()) + 1
    w = np.zeros(labels.size)
    for r in range(n_rings):
        sel = labels == r
        if mean[sel].mean() > 0:
            w[sel] = mean[sel] / mean[sel].mean()
    masks = load_masks(saxs.output_file.with_name("masks.npz"))
    in_ring = labels >= 0
    out = np.zeros((3, len(masks.cells), n_rings))
    for i in range(len(masks.cells)):
        v = in_ring & ~masks.base_bad[i]
        lab = labels[v].astype(np.intp)
        for p in range(3):
            out[p, i] = np.bincount(lab, w[v] ** (p + 1), n_rings)
    idx = np.searchsorted(masks.cells, cells)
    return out[:, idx]


def one_run(run: int) -> dict:
    from analysis.triage.classify import hits

    saxs = AgipdSaxsConfig(proposal=10400, run=run)
    path = PhotonStatsConfig(saxs=saxs).output_file
    h = hits(run)
    trains = h["train_ids"] if h["kind"] == "mesh" else None
    ds = read_photon_stats(path, trains=trains)
    weights = run_weights(
        path, pixel_sums_path=saxs.pixel_sums_file,
        masks_path=saxs.output_file.with_name("masks.npz"), trains=trains,
    )  # fmt: skip
    beta = contrast(ds, weights).beta.values
    # the windows run_weights chose (it does not return them)
    table = window_table(saxs.pixel_sums_file)
    windows = [int(w) for w in np.flatnonzero(table.written.values)]
    if trains is not None:
        wanted = set(np.asarray(trains, dtype=np.uint64).tolist())
        with h5py.File(saxs.pixel_sums_file, "r") as f:
            t_ids, t_win = f["trains/trainId"][:], f["trains/window"][:]
        windows = [
            w for w in windows
            if (m := t_ids[t_win == w]).size and set(m.tolist()) <= wanted
        ]  # fmt: skip
    assert len(windows) == weights.attrs["n_windows"]
    W1, W2, W3 = pixel_moments(path, saxs, windows, ds.cellId.values)
    idx = np.searchsorted(weights.cellId.values, ds.cellId.values)
    np.testing.assert_allclose(W1, weights.w1.values[idx], rtol=1e-12)
    np.testing.assert_allclose(W2, weights.w2.values[idx], rtol=1e-12)

    s1 = ds.s1.values
    n = ds.n_valid.values.astype(np.float64)
    hist = ds.hist.values
    k_max = int(ds.attrs["k_max"])
    k = np.arange(hist.shape[-1], dtype=np.float64)
    fac3 = k * (k - 1) * (k - 2)
    fac3[k_max + 1] = 0.0  # the overflow bin's values are not stored
    s3 = (hist * fac3).sum(-1)

    with np.errstate(invalid="ignore", divide="ignore"):
        den3 = np.where(W1 > 0, s1 * (s1 - 1) * (s1 - 2) * W3 / W1**3, 0.0)
        g3 = s3.sum(0) / den3.sum(0)
        g3_expected = (1 + beta) * (1 + 2 * beta)
        tids = ds.trainId.values
        uniq = np.unique(tids)
        blocks = np.array_split(np.arange(uniq.size), min(N_BLOCKS, uniq.size))
        member = np.zeros(uniq.size, np.intp)
        for i, b in enumerate(blocks):
            member[b] = i
        block = member[np.searchsorted(uniq, tids)]
        nb = len(blocks)
        num_b = np.stack([s3[block == i].sum(0) for i in range(nb)])
        den_b = np.stack([den3[block == i].sum(0) for i in range(nb)])
        loo = (num_b.sum(0) - num_b) / (den_b.sum(0) - den_b)
        g3_sigma = np.sqrt((nb - 1) / nb * ((loo - loo.mean(0)) ** 2).sum(0))

        m1 = (s1 / W1).mean(0)
        m2 = (s1 * (s1 - 1) / W1**2).mean(0)
        m3 = (s1 * (s1 - 1) * (s1 - 2) / W1**3).mean(0)
        a3_frame = m1 * m3 / m2**2
        nn = n.mean(0)
        a3_pix = (W1.mean(0) / nn) * (W3.mean(0) / nn) / (W2.mean(0) / nn) ** 2
        p = hist.sum(0).astype(np.float64)
        p1, p2, p3 = p[:, 1], p[:, 2], p[:, 3]
        r3 = 3 * p1 * p3 / (2 * p2**2)
        r3_sigma = r3 * np.sqrt(1 / p1 + 1 / p3 + 4 / p2)
        r3_nb = (1 + 2 * beta) / (1 + beta)
        r3_mixed = r3_nb * a3_frame * a3_pix
    return {
        "run": run, "kind": h["kind"], "n_frames": int(ds.sizes["frame"]),
        "n_windows": len(windows),
        "ring_lo_nm": _round(ds.ring_lo, 4), "ring_hi_nm": _round(ds.ring_hi, 4),
        "beta": _round(beta), "r3": _round(r3), "r3_sigma": _round(r3_sigma),
        "r3_nb": _round(r3_nb), "a3_frame": _round(a3_frame),
        "a3_pix": _round(a3_pix), "r3_z_nb": _round((r3 - r3_nb) / r3_sigma, 2),
        "r3_z_mixed": _round((r3 - r3_mixed) / r3_sigma, 2),
        "g3": _round(g3), "g3_expected": _round(g3_expected),
        "g3_sigma": _round(g3_sigma), "g3_z": _round((g3 - g3_expected) / g3_sigma, 2),
        "overflow": [int(v) for v in hist[..., k_max + 1].sum(0)],
    }  # fmt: skip


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, nargs="+", default=[450, 423, 318])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    runs = [one_run(r) for r in args.runs]
    write_report(
        report_header(script="photon_stats_shape_check", runs=runs),
        args.json, "photon_stats_shape_check",
    )  # fmt: skip
    for r in runs:
        print(f"r{r['run']} ({r['kind']}, {r['n_frames']} frames, "
              f"{r['n_windows']} windows)")  # fmt: skip
        print(" ring   beta   r3_z  A3_frame  A3_pix  r3_z|mixed  g3/exp   g3_z")
        for i in range(len(r["beta"])):
            ratio = r["g3"][i] / r["g3_expected"][i]
            row = (r["beta"][i], r["r3_z_nb"][i], r["a3_frame"][i], r["a3_pix"][i],
                   r["r3_z_mixed"][i], ratio, r["g3_z"][i])  # fmt: skip
            print(" {:3d} {:7.4f} {:6.1f} {:8.4f} {:7.4f} {:9.1f} {:8.4f} {:6.1f}"
                  .format(i, *row))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
