"""Single-frame contrast and photon-statistics checks.

Reads the per-frame ring records of `analysis.saxs.photon_stats` and computes, for
any set of frames (a run's hits, a 30 s slice, a quarter of the pulse train):

    1 + β = Σ_f Σ_r S2 / Σ_f Σ_r (S1² − S1) · W2 / W1²

per ring, or per XPCS ROI by summing over its rings. W1 = Σw and W2 = Σw² over
the ring's valid pixels, w each pixel's run-mean count relative to its ring (from
the window sums); without weights W1 = W2 = n_valid, which reads β high by the
ring's intensity spread w_cv². The W are taken over each memory cell's base mask,
which moves 1 + β by ≤ 1e-4. σ by jackknife over contiguous blocks of trains.

The negative-binomial shape checks: 2·P0·P2/P1² = 1 + β and
3·P1·P3/(2·P2²) = (1 + 2β)/(1 + β), with Poisson errors from the counts. They are
taken on the pooled histogram, so they also carry the spread of k̄ between frames
and between pixels: R3 is lifted by ⟨m⟩⟨m³⟩/⟨m²⟩² over frames, which on a whole
run alone is worth many σ. So ``r3_z`` records the raw ratios; it is not a speckle
test over many frames. Pixels far out in the tail (hot
pixels) inflate Σk(k − 1) and with it β, so the low orders read below.

Nothing here alters the stored counts.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from analysis.common.status import FrameStatus

__all__ = [
    "N_BLOCKS",
    "contrast",
    "nb_checks",
    "read_photon_stats",
    "ring_weights",
    "roi_groups",
    "run_weights",
    "summarise",
    "summary_of",
]

#: Jackknife blocks of contiguous trains (as triage.xpcs: σ good to ~10 %).
N_BLOCKS = 50


def read_photon_stats(path: str | Path, trains=None):
    """The OK frames of a photon-statistics file, optionally only ``trains``.

    :returns: a Dataset over (frame, ring[, k]): ``n_valid``, ``hist``, ``s1``,
        ``s2``, ``max_k``; per frame ``trainId``, ``pulseId``, ``cellId``; per ring
        ``ring_lo``/``ring_hi`` [nm⁻¹]. ``attrs``: the file, ``k_max`` and the
        ring map's sha256.
    """
    import xarray as xr

    with h5py.File(path, "r") as f:
        frames = f["frames"]
        keep = frames["status"][:] == FrameStatus.OK
        tids = frames["trainId"][:]
        if trains is not None:
            keep &= np.isin(tids, np.asarray(trains, dtype=np.uint64))
        rows = np.flatnonzero(keep)
        data = {
            name: frames[name][rows] if rows.size else frames[name][:0]
            for name in ("n_valid", "hist", "s1", "s2", "max_k")
        }
        edges = f["rings/edges_nm"][:]
        attrs = {
            "file": str(path),
            "k_max": int(f["rings"].attrs["k_max"]),
            "labels_sha256": str(f["rings"].attrs["labels_sha256"]),
        }
        pulse = frames["reader_pulseId"][rows]
        cell = frames["cellId"][rows]
    k = np.arange(data["hist"].shape[-1])
    return xr.Dataset(
        {
            "n_valid": (("frame", "ring"), data["n_valid"].astype(np.int64)),
            "hist": (("frame", "ring", "k"), data["hist"].astype(np.int64)),
            "s1": (("frame", "ring"), data["s1"].astype(np.float64)),
            "s2": (("frame", "ring"), data["s2"].astype(np.float64)),
            "max_k": (("frame", "ring"), data["max_k"].astype(np.int64)),
        },
        coords={
            "trainId": ("frame", tids[rows]),
            "pulseId": ("frame", pulse),
            "cellId": ("frame", cell),
            "ring": np.arange(edges.shape[0]),
            "ring_lo": ("ring", edges[:, 0]),
            "ring_hi": ("ring", edges[:, 1]),
            "k": k,
        },
        attrs=attrs,
    )


def ring_weights(
    labels: np.ndarray,
    mean_counts: np.ndarray,
    base_bad: np.ndarray,
    cells: np.ndarray,
):
    """W1 = Σw and W2 = Σw² per (cell, ring) over each cell's base-mask-valid pixels.

    :param labels: each pixel's ring, −1 for none (the file's ``rings/labels``).
    :param mean_counts: each pixel's mean count per frame (window sums); pixels
        that were never valid carry 0 and weigh nothing.
    :param base_bad: (n_cells, NPIX) base masks of the SAXS pass (`load_masks`).
    :param cells: the cell ids of ``base_bad``'s rows.
    """
    import xarray as xr

    labels = np.asarray(labels)
    n_rings = int(labels.max()) + 1
    w = np.zeros(labels.size)
    for r in range(n_rings):
        sel = labels == r
        mean = mean_counts[sel].mean() if sel.any() else 0.0
        if mean > 0:
            w[sel] = mean_counts[sel] / mean
    in_ring = labels >= 0
    w1 = np.zeros((len(cells), n_rings))
    w2 = np.zeros((len(cells), n_rings))
    for i in range(len(cells)):
        valid = in_ring & ~base_bad[i]
        lab = labels[valid].astype(np.intp)
        w1[i] = np.bincount(lab, weights=w[valid], minlength=n_rings)
        w2[i] = np.bincount(lab, weights=w[valid] ** 2, minlength=n_rings)
    return xr.Dataset(
        {"w1": (("cellId", "ring"), w1), "w2": (("cellId", "ring"), w2)},
        coords={"cellId": np.asarray(cells), "ring": np.arange(n_rings)},
    )


def run_weights(
    path: str | Path,
    *,
    pixel_sums_path: str | Path,
    masks_path: str | Path,
    trains=None,
):
    """`ring_weights` for one run: its ring map, window sums and masks.

    :param trains: build the per-pixel means from the windows whose trains all lie
        in ``trains`` (a mesh run's hits); None for every written window.
    :param masks_path: the SAXS pass's ``masks.npz`` (per-cell base masks).
    """
    from analysis.saxs.masks import load_masks
    from analysis.saxs.pixel_sums import window_sums, window_table

    with h5py.File(path, "r") as f:
        labels = f["rings/labels"][:]
    table = window_table(pixel_sums_path)
    windows = np.flatnonzero(table.written.values)
    if trains is not None:
        wanted = set(np.asarray(trains, dtype=np.uint64).tolist())
        with h5py.File(pixel_sums_path, "r") as f:
            t_ids, t_win = f["trains/trainId"][:], f["trains/window"][:]
        windows = [
            int(w) for w in windows
            if (members := t_ids[t_win == w]).size
            and set(members.tolist()) <= wanted
        ]  # fmt: skip
        if not windows:
            raise ValueError("no written window lies wholly inside the given trains")
    sums = window_sums(pixel_sums_path, windows)
    counts = sums.counts.values.reshape(-1).astype(np.float64)
    valid = sums.valid_frames.values.reshape(-1).astype(np.float64)
    mean = np.divide(counts, valid, out=np.zeros_like(counts), where=valid > 0)
    masks = load_masks(masks_path)
    out = ring_weights(labels, mean, masks.base_bad, masks.cells)
    out.attrs.update(n_windows=len(windows), pixel_sums=str(pixel_sums_path),
                     masks=str(masks_path))  # fmt: skip
    return out


def roi_groups(ds) -> dict[float, list[int]]:
    """The rings of each plain XPCS ROI, keyed by its centre [Å⁻¹] (as in DAMNIT)."""
    from analysis.triage.xpcs import roi_edges_nm, roi_kinds

    edges = roi_edges_nm()
    centre = 0.5 * (ds.ring_lo.values + ds.ring_hi.values)
    groups = {}
    for (lo, hi), kind in zip(edges, roi_kinds(edges), strict=True):
        if kind == "plain":
            rings = [int(r) for r in np.flatnonzero((centre >= lo) & (centre < hi))]
            if rings:
                groups[round(float(lo + hi) / 20.0, 6)] = rings
    return groups


def _expected(ds, weights):
    """Per frame and ring: (S1² − S1)·W2/W1², the second factorial moment at β = 0."""
    s1 = ds.s1.values
    if weights is None:
        w1 = w2 = ds.n_valid.values.astype(np.float64)
    else:
        idx = np.searchsorted(weights.cellId.values, ds.cellId.values)
        if not np.array_equal(weights.cellId.values[idx], ds.cellId.values):
            raise ValueError("a frame's cell has no weights")
        w1, w2 = weights.w1.values[idx], weights.w2.values[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w1 > 0, (s1 * s1 - s1) * w2 / (w1 * w1), 0.0)


def contrast(ds, weights=None, groups: dict | None = None, n_blocks: int = N_BLOCKS):
    """β per ring (or per group of rings) over the frames in ``ds``.

    :param weights: `ring_weights`, or None for the unweighted estimator.
    :param groups: label -> rings to pool, e.g. `roi_groups`; None per ring.
    :returns: a Dataset over the group: ``beta``, ``sigma`` (jackknife over
        contiguous blocks of trains), ``kbar`` and ``n_frames``.
    """
    import xarray as xr

    expected = _expected(ds, weights)
    s2 = ds.s2.values
    s1, n = ds.s1.values, ds.n_valid.values
    if groups is None:
        groups = {int(r): [int(r)] for r in ds.ring.values}
    labels = list(groups)
    num = np.stack([s2[:, groups[g]].sum(axis=1) for g in labels], axis=1)
    den = np.stack([expected[:, groups[g]].sum(axis=1) for g in labels], axis=1)
    k1 = np.stack([s1[:, groups[g]].sum(axis=1) for g in labels], axis=1)
    nv = np.stack([n[:, groups[g]].sum(axis=1) for g in labels], axis=1)

    trains = ds.trainId.values
    uniq = np.unique(trains)
    blocks = np.array_split(np.arange(uniq.size), min(n_blocks, max(uniq.size, 1)))
    train_block = np.zeros(uniq.size, dtype=np.intp)
    for i, members in enumerate(blocks):
        train_block[members] = i
    block_id = train_block[np.searchsorted(uniq, trains)]
    b = len(blocks)
    num_b = np.stack([num[block_id == i].sum(axis=0) for i in range(b)])
    den_b = np.stack([den[block_id == i].sum(axis=0) for i in range(b)])
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = num.sum(axis=0) / den.sum(axis=0) - 1.0
        loo = (num_b.sum(0) - num_b) / (den_b.sum(0) - den_b) - 1.0
        kbar = k1.sum(axis=0) / nv.sum(axis=0)
    sigma = (
        np.sqrt((b - 1) / b * ((loo - loo.mean(axis=0)) ** 2).sum(axis=0))
        if b > 1
        else np.full(len(labels), np.nan)
    )
    return xr.Dataset(
        {
            "beta": ("group", beta),
            "sigma": ("group", sigma),
            "kbar": ("group", kbar),
            "n_frames": ("group", np.full(len(labels), trains.size)),
        },
        coords={"group": labels},
        attrs={"weighted": weights is not None, "n_blocks": b},
    )


def nb_checks(ds, beta=None):
    """The negative-binomial shape per ring, pooled over the frames in ``ds``.

    The expectations assume one k̄ for every frame and pixel; see the module
    docstring for what the spread of k̄ adds to the pooled ratios.

    :param beta: per-ring contrast for the expectations (e.g. `contrast`'s), or None
        to report the ratios alone.
    :returns: a Dataset over ring: ``r2`` = 2·P0·P2/P1² and its Poisson σ, ``r3`` =
        3·P1·P3/(2·P2²) and its σ, and with ``beta`` their expectations and ``r3_z``.
    """
    import xarray as xr

    h = ds.hist.sum("frame").values.astype(np.float64)
    p0, p1, p2, p3 = h[:, 0], h[:, 1], h[:, 2], h[:, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 2 * p0 * p2 / p1**2
        r3 = 3 * p1 * p3 / (2 * p2**2)
        r2_sigma = r2 * np.sqrt(4 / p1 + 1 / p2)
        r3_sigma = r3 * np.sqrt(1 / p1 + 1 / p3 + 4 / p2)
    out = xr.Dataset(
        {
            "r2": ("ring", r2),
            "r2_sigma": ("ring", r2_sigma),
            "r3": ("ring", r3),
            "r3_sigma": ("ring", r3_sigma),
            "max_k": ("ring", ds.max_k.max("frame").values if ds.sizes["frame"] else
                      np.zeros(h.shape[0], int)),
        },
        coords={"ring": ds.ring.values},
    )  # fmt: skip
    if beta is not None:
        b = np.asarray(beta, dtype=np.float64)
        out["r2_expected"] = ("ring", 1 + b)
        out["r3_expected"] = ("ring", (1 + 2 * b) / (1 + b))
        with np.errstate(invalid="ignore", divide="ignore"):
            out["r3_z"] = ("ring", (r3 - (1 + 2 * b) / (1 + b)) / r3_sigma)
    return out


def summarise(path: str | Path, weights=None, trains=None, n_blocks: int = N_BLOCKS):
    """A run's photon statistics in brief, per plain XPCS ROI and per ring.

    :param weights: `run_weights`, or None for the unweighted estimator.
    :param trains: the trains to use (a run's droplet hits); None for every one.
    :returns: a Dataset: per ROI (``roi``, its centre in Å⁻¹) ``beta``, ``sigma``,
        ``kbar`` and ``reference_beta`` (the fixed table); per ring ``ring_beta``,
        ``ring_sigma``, ``ring_kbar``, ``ring_max_k`` and the pooled histogram
        ``ring_hist`` (ring, k). ``attrs``: ``n_frames``, ``n_trains``, ``weighted``.
    """
    ds = read_photon_stats(path, trains=trains)
    if not ds.sizes["frame"]:
        raise ValueError(f"{path}: no OK frame among the trains asked for")
    out = summary_of(ds, weights, n_blocks)
    out.attrs["file"] = str(path)
    return out


def summary_of(ds, weights=None, n_blocks: int = N_BLOCKS):
    """`summarise` on frames already read (`read_photon_stats`)."""
    import xarray as xr

    from analysis.triage.xpcs import ES_REFERENCE_BETA

    groups = roi_groups(ds)
    rings = contrast(ds, weights, n_blocks=n_blocks)
    rois = contrast(ds, weights, groups=groups, n_blocks=n_blocks)
    centres = list(groups)
    return xr.Dataset(
        {
            "beta": ("roi", rois.beta.values),
            "sigma": ("roi", rois.sigma.values),
            "kbar": ("roi", rois.kbar.values),
            "reference_beta": ("roi", [ES_REFERENCE_BETA[q][0] for q in centres]),
            "ring_beta": ("ring", rings.beta.values),
            "ring_sigma": ("ring", rings.sigma.values),
            "ring_kbar": ("ring", rings.kbar.values),
            "ring_max_k": ("ring", ds.max_k.max("frame").values),
            "ring_hist": (("ring", "k"), ds.hist.sum("frame").values),
        },
        coords={
            "roi": centres,
            "ring": ds.ring.values,
            "ring_lo": ("ring", ds.ring_lo.values),
            "ring_hi": ("ring", ds.ring_hi.values),
            "k": ds.k.values,
        },
        attrs={
            "n_frames": int(ds.sizes["frame"]),
            "n_trains": int(np.unique(ds.trainId.values).size),
            "weighted": int(weights is not None),
        },
    )
