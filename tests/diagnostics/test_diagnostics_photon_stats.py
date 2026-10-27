"""Single-frame contrast and shape checks against negative-binomial frames of known β.

Each synthetic ring has an intensity gradient (w from 0.7 to 1.3, w_cv ≈ 0.17), so
the unweighted estimator must read high by about w_cv² and the weighted one must
not; injected hot pixels must show in the 3-photon shape check.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import pytest
import xarray as xr

from analysis.common.plan import RunPlan, TrainRecord
from analysis.diagnostics.photon_stats import (
    contrast,
    nb_checks,
    read_photon_stats,
    ring_weights,
    roi_groups,
)
from analysis.saxs.photon_stats import PhotonStatsWriter, default_rings, frame_stats

BETA, KBAR, N_PIX, K_MAX = 0.04, 0.08, 8000, 15


def nb_frames(rng, n_frames, n_rings=2, beta=BETA, kbar=KBAR, hot=0, n_pix=N_PIX):
    """Per-frame ring records of gamma-Poisson pixels with a known contrast.

    Pixel p of a ring has mean kbar·w_p; its intensity is Gamma(1/β) around that,
    so its counts are negative binomial with contrast β. ``hot`` pixels per frame
    get 10 photons whatever the intensity.
    """
    w = np.tile(np.linspace(0.7, 1.3, n_pix), n_rings)
    labels = np.repeat(np.arange(n_rings), n_pix).astype(np.int16)
    in_ring = np.ones(labels.size, bool)
    ring_n = np.full(n_rings, n_pix)
    bad = np.zeros(labels.size, bool)
    out = {k: [] for k in ("n_valid", "hist", "s1", "s2", "max_k")}
    m = 1.0 / beta
    for _ in range(n_frames):
        intensity = rng.gamma(m, kbar * w / m)
        counts = rng.poisson(intensity).astype(np.int16)
        if hot:
            counts[rng.choice(counts.size, hot, replace=False)] = 10
        rec = frame_stats(counts, bad, labels, ring_n, in_ring, K_MAX)
        for key, value in zip(out, rec, strict=True):
            out[key].append(value)
    ds = xr.Dataset(
        {k: (("frame", "ring", "k") if k == "hist" else ("frame", "ring"),
             np.array(v, dtype=np.float64 if k.startswith("s") else np.int64))
         for k, v in out.items()},
        coords={
            "trainId": ("frame", 1000 + np.arange(n_frames) // 10),
            "pulseId": ("frame", np.arange(n_frames) % 10),
            "cellId": ("frame", np.zeros(n_frames, np.uint16)),
            "ring": np.arange(n_rings),
            "ring_lo": ("ring", [0.35, 0.375][:n_rings]),
            "ring_hi": ("ring", [0.375, 0.4][:n_rings]),
        },
    )  # fmt: skip
    weights = ring_weights(labels, kbar * w, bad[None, :], np.array([0]))
    return ds, weights, w


@pytest.fixture(scope="module")
def frames():
    return nb_frames(np.random.default_rng(3), 1500)


def test_the_weighted_contrast_recovers_beta(frames):
    ds, weights, _ = frames
    c = contrast(ds, weights)
    assert c.attrs["weighted"]
    for r in range(2):
        assert abs(float(c.beta[r]) - BETA) < 4 * float(c.sigma[r])
        assert float(c.sigma[r]) < 0.01
        assert float(c.kbar[r]) == pytest.approx(KBAR, rel=0.02)


def test_the_unweighted_contrast_reads_high_by_the_ring_spread(frames):
    ds, weights, w = frames
    cv2 = (w[:N_PIX].std() / w[:N_PIX].mean()) ** 2
    plain, weighted = contrast(ds), contrast(ds, weights)
    excess = (plain.beta - weighted.beta).values
    np.testing.assert_allclose(excess, cv2 * (1 + BETA), rtol=0.15)


def test_rings_pool_into_rois(frames):
    ds, weights, _ = frames
    both = contrast(ds, weights, groups={"roi": [0, 1]})
    each = contrast(ds, weights)
    assert float(both.beta[0]) == pytest.approx(float(each.beta.mean()), abs=0.01)
    assert float(both.sigma[0]) < float(each.sigma.max())


def test_the_shape_check_passes_speckle_and_flags_hot_pixels(frames):
    ds, weights, _ = frames
    clean = nb_checks(ds, contrast(ds, weights).beta.values)
    assert (np.abs(clean.r3_z) < 4).all()
    # 10-photon pixels never reach P(3) but inflate Σk(k−1): the moment contrast
    # leaves the low orders behind, and the shape check says so, here from below
    hot, hot_w, _ = nb_frames(np.random.default_rng(4), 600, hot=3)
    beta = contrast(hot, hot_w).beta.values
    checks = nb_checks(hot, beta)
    assert (beta > 5 * BETA).all()
    assert (np.abs(checks.r3_z) > 5).all()
    assert (checks.max_k >= 10).all()


def test_roi_groups_map_rings_to_the_plain_xpcs_rois():
    rings = np.array(default_rings())
    ds = xr.Dataset(coords={"ring": np.arange(len(rings)),
                            "ring_lo": ("ring", rings[:, 0]),
                            "ring_hi": ("ring", rings[:, 1])})  # fmt: skip
    groups = roi_groups(ds)
    assert list(groups) == [0.02875, 0.04, 0.0505, 0.0805]
    assert [len(v) for v in groups.values()] == [5, 4, 4, 4]


@dataclass(frozen=True)
class StubConfig:
    npt: int
    k_max: int = K_MAX
    overwrite: bool = False

    @property
    def output_file(self):
        return "unused.h5"

    @property
    def operational_fields(self):
        return frozenset()

    def config_hash(self):
        return "stub"


def test_the_reader_keeps_ok_frames_and_selects_trains(tmp_path):
    n_trains, per_train, n_rings = 4, 3, 2
    records = [
        TrainRecord(1000 + t, per_train, t * per_train, 0) for t in range(n_trains)
    ]
    plan = RunPlan(tuple(records), (), n_trains * per_train, "FAKE")
    path = tmp_path / "ps.h5"
    with h5py.File(path, "w") as f:
        PhotonStatsWriter._create_layout(f, StubConfig(n_rings), plan, "stub")
        frames = f["frames"]
        frames["trainId"][:] = np.repeat(1000 + np.arange(n_trains), per_train)
        frames["cellId"][:] = np.tile(np.arange(per_train), n_trains)
        status = np.zeros(n_trains * per_train, np.uint8)
        status[4] = 5  # a worker error
        frames["status"][:] = status
        frames["s1"][:] = np.arange(n_trains * per_train)[:, None]
        f["rings"]["edges_nm"] = np.array([[0.35, 0.375], [0.375, 0.4]])
        f["rings"]["labels"] = np.zeros(10, np.int16)
        f["rings"].attrs.update(k_max=K_MAX, labels_sha256="x", operator_sha256="y")
    ds = read_photon_stats(path)
    assert ds.sizes["frame"] == 11 and 4 not in ds.s1.values[:, 0]
    some = read_photon_stats(path, trains=[1000, 1002])
    assert set(some.trainId.values) == {1000, 1002}
    assert some.sizes["k"] == K_MAX + 2


def test_the_summary_reports_rois_rings_and_the_reference(frames):
    from analysis.diagnostics.photon_stats import summary_of
    from analysis.triage.xpcs import REFERENCE_BETA

    ds, weights, _ = frames
    s = summary_of(ds, weights)
    assert list(s.roi.values) == [0.04]  # both rings lie in 0.35-0.45 nm^-1
    assert abs(float(s.beta[0]) - BETA) < 4 * float(s.sigma[0])
    assert float(s.reference_beta[0]) == REFERENCE_BETA[0.04][0]
    np.testing.assert_array_equal(
        s.ring_hist.sum("k").values, ds.n_valid.sum("frame").values
    )
    assert s.attrs["n_frames"] == ds.sizes["frame"] and s.attrs["weighted"] == 1


def test_the_contrast_sigma_is_the_scatter_of_independent_repeats():
    """Twelve runs of 200 single-ring frames, their moments taken straight from the
    counts; the jackknife σ of each must match the scatter of β between them."""
    n_rep, n_frames, n_pix = 12, 200, 2000
    rng = np.random.default_rng(0)
    w = np.linspace(0.7, 1.3, n_pix)
    m = 1.0 / BETA
    counts = rng.poisson(rng.gamma(m, KBAR * w / m, (n_rep, n_frames, n_pix)))
    s1 = counts.sum(axis=2, dtype=np.float64)
    s2 = (counts * (counts - 1)).sum(axis=2, dtype=np.float64)
    labels = np.zeros(n_pix, np.int16)
    weights = ring_weights(labels, KBAR * w, np.zeros((1, n_pix), bool), np.array([0]))
    betas, sigmas = [], []
    for i in range(n_rep):
        ds = xr.Dataset(
            {"s1": (("frame", "ring"), s1[i, :, None]),
             "s2": (("frame", "ring"), s2[i, :, None]),
             "n_valid": (("frame", "ring"), np.full((n_frames, 1), n_pix))},
            coords={"trainId": ("frame", np.arange(n_frames) // 10),
                    "cellId": ("frame", np.zeros(n_frames, np.uint16)),
                    "ring": [0]},
        )  # fmt: skip
        c = contrast(ds, weights, n_blocks=10)
        betas.append(float(c.beta[0]))
        sigmas.append(float(c.sigma[0]))
    # twelve repeats know their scatter to about 20 %, ten blocks theirs to 25 %
    assert np.std(betas, ddof=1) == pytest.approx(np.median(sigmas), rel=0.4)


def test_the_shape_ratio_sigmas_are_the_scatter_of_independent_repeats():
    """Each "ring" is one repeat of 10 000 negative-binomial pixels at k̄ = 0.5,
    where P2 and P3 weigh enough for a wrong σ formula to show."""
    rng = np.random.default_rng(0)
    n_rep, kbar = 300, 0.5
    counts = rng.negative_binomial(1 / BETA, 1 / (1 + kbar * BETA), (n_rep, 10_000))
    hist = np.stack([np.bincount(c, minlength=K_MAX + 2)[: K_MAX + 2] for c in counts])
    ds = xr.Dataset(
        {"hist": (("frame", "ring", "k"), hist[None]),
         "max_k": (("frame", "ring"), counts.max(axis=1)[None])},
        coords={"ring": np.arange(n_rep)},
    )  # fmt: skip
    out = nb_checks(ds, np.full(n_rep, BETA))
    # 300 repeats know their scatter to 4 %
    assert float(out.r2.std() / out.r2_sigma.median()) == pytest.approx(1, abs=0.1)
    assert float(out.r3.std() / out.r3_sigma.median()) == pytest.approx(1, abs=0.1)
    assert float(out.r3.mean()) == pytest.approx((1 + 2 * BETA) / (1 + BETA), abs=0.025)
