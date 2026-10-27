"""Fakes for the triage: DAMNIT's sqlite and per-run HDF5 files, in their real layout.

The layout is DAMNIT's as found on p10400: one group per variable, the
main dataset named as `analysis.triage.inventory.TRACKED` says, ``droplet_fit`` with
``trainId``/``volume``/``center``/``dim``, scalar byte strings under ``.errors``, and
a 0-d value for a variable that returned nothing.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from analysis.triage.inventory import RUN_COLUMNS, TRACKED

FIRST_TRAIN = 2637862276


def write_damnit_file(
    path: Path,
    *,
    n_trains: int = 100,
    volume=None,
    center=None,
    xgm=None,
    jf_cells=None,
    errors: dict[str, str] | None = None,
    empty: tuple[str, ...] = (),
) -> Path:
    tids = FIRST_TRAIN + np.arange(n_trains, dtype=np.uint64)
    with h5py.File(path, "w") as f:
        if volume is not None:
            g = f.create_group("droplet_fit")
            g["trainId"] = tids
            g["volume"] = np.asarray(volume, dtype=np.float32)
            g["dim"] = np.array([b"x", b"y"])
            g["center"] = (
                np.zeros((n_trains, 2), np.float32)
                if center is None
                else np.asarray(center, np.float32)
            )
            g["radius"] = np.full((n_trains, 2), 50.0, np.float32)
        if xgm is not None:
            g = f.create_group("xgm_intensity")
            g["Energy"] = np.asarray(xgm, dtype=np.float32)
            g["trainId"] = tids
        if jf_cells is not None:
            g = f.create_group("jungfrau_waxs_jf1")
            g["cellId"] = np.asarray(jf_cells, dtype=np.uint16)
            g["intensity"] = np.zeros((n_trains, len(jf_cells), 4), np.float32)
        for name in empty:
            f.create_group(name)[TRACKED[name]] = np.float64(np.nan)
        if errors:
            e = f.create_group(".errors")
            for name, text in errors.items():
                e[name] = text.encode()
    return path


def write_sqlite(path: Path, rows: list[dict]) -> Path:
    """A ``runs`` table with `RUN_COLUMNS` plus a preview column the triage ignores."""
    cols = list(RUN_COLUMNS) + ["agipd_iq_overview"]
    with closing(sqlite3.connect(path)) as con:
        con.execute(f"CREATE TABLE runs ({', '.join(cols)})")
        for row in rows:
            values = [row.get(c) for c in cols]
            con.execute(
                f"INSERT INTO runs VALUES ({', '.join('?' * len(cols))})", values
            )
        con.commit()
    return path


def run_row(run: int, **kw) -> dict:
    base = {
        "run": run,
        "start_time": 1778531000.0 + 600 * run,
        "run_length": "0:05:00",
        "run_type": "XPCS",
        "sample": "Ferritin_50_PEG6K_5p",
        "scan_type": None,
        "n_trains": 3000,
        "pulses": 155.0,
        "rep_rate": 2.256944,
        "raw_size": 4.7,
        "proc_size": 0.29,
        "total_transmission": 1.0,
        "xgm_intensity": 330.0,
        "agipd_iq_overview": b"\x89PNG",
    }
    base.update(kw)
    return base


#: Lit-pixel probability per frame of the seven extra-speckle XPCS rings.
ring_densities = (0.2, 0.1, 0.01, 0.005, 0.0005, 0.0002, 0.0001)


def write_hit_results(
    path: Path,
    *,
    n_trains: int = 60,
    n_frames: int = 14,
    n_env: int = 400,
    densities=ring_densities,
    poisson: bool = False,
    masked: dict[int, float] | None = None,
    failed: dict[int, int] | None = None,
    seed: int = 3,
):
    """A hit-train results file (`analysis.xpcs.result`) of synthetic photon trains.

    Ring r is the r-th XPCS ROI. By default it lights each pixel of each frame with
    probability ``densities[r]`` (photons of 1 to 3, mean 2, so k̄ ≈ 2 · density).
    With ``poisson`` each count is Poisson of mean ``densities[r]`` instead: a
    speckle-free field (β = 0), whose g2 noise is the analytical one. Each frame
    flags 2 % of the pixels; ``masked`` maps a train's offset from `FIRST_TRAIN` to
    a further fraction of the pixels flagged in that train alone. The trains are
    consecutive from `FIRST_TRAIN`. ``failed`` maps train id -> `FrameStatus` code
    of hits that gave no photons. Returns the open `XpcsHits`.
    """
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1] / "xpcs"))
    from xpcs_fakes import build_results, make_flags, make_train

    from analysis.common.status import FrameStatus
    from analysis.triage.xpcs import roi_edges_nm
    from analysis.xpcs.rings import Rings

    rng = np.random.default_rng(seed)
    n_rings = len(densities)
    label = rng.integers(0, n_rings, n_env).astype(np.int8)
    label.flags.writeable = False
    rings = Rings(tuple(map(tuple, roi_edges_nm()[:n_rings])), label, n_rings, "rings")
    per_pixel = np.asarray(densities)[label]
    masked = masked or {}
    failed = {t: FrameStatus(c) for t, c in (failed or {}).items()}
    ids = [
        int(FIRST_TRAIN) + i
        for i in range(n_trains)
        if int(FIRST_TRAIN) + i not in failed
    ]
    photons = []
    for t in ids:
        if poisson:
            dense = rng.poisson(per_pixel, (n_frames, n_env)).astype(np.int32)
        else:
            lit = rng.random((n_frames, n_env)) < per_pixel
            dense = (lit * rng.integers(1, 4, (n_frames, n_env))).astype(np.int32)
        flags = make_flags(rng, n_frames, n_env)
        if (fraction := masked.get(t - int(FIRST_TRAIN))) is not None:
            extra = np.flatnonzero(rng.random(n_env) < fraction)
            flags[0] = np.union1d(flags[0], extra)
        photons.append(make_train(t, dense, flags))
    path.parent.mkdir(parents=True, exist_ok=True)
    return build_results(path, photons, rings, failed=failed)


#: A mesh run's hits sit ON_DEX above its off level; the two spreads in dex.
ON_DEX, OFF_SD, ON_SD = 1.3, 0.02, 0.05


def partial(f):
    """rel of a train carrying a fraction ``f`` of a full hit's droplet signal."""
    return np.log10(1 + np.asarray(f) * (10**ON_DEX - 1))


def mesh_run(rng, off, hit_fraction, n=800, edge_fraction=0.02, partial_fraction=0.03):
    """log10(s/F) of one mesh run, and its true classes."""
    n_on = int(hit_fraction * n)
    n_edge = int(edge_fraction * n)
    n_part = int(partial_fraction * n)
    n_off = n - n_on - n_edge - n_part
    x = np.concatenate([
        off + OFF_SD * rng.standard_normal(n_off),
        off + ON_DEX + ON_SD * rng.standard_normal(n_on),
        off + rng.uniform(0.2, 0.5, n_edge),
        off + partial(rng.uniform(0.15, 0.6, n_part)),
    ])  # fmt: skip
    truth = np.array(
        ["off"] * n_off + ["on"] * n_on + ["edge"] * n_edge + ["partial"] * n_part
    )
    order = rng.permutation(n)
    return x[order], truth[order]


def write_damnit(path, saxs, *, n_frames=None, xgm=None, trans=None, waxs=None):
    """agipd_saxs (train, pulse, q), xgm_intensity, total_transmission, jf1."""
    n, p, nq = saxs.shape
    tids = FIRST_TRAIN + np.arange(n, dtype=np.uint64)
    q = np.linspace(0.08, 1.06, nq)
    with h5py.File(path, "w") as f:
        g = f.create_group("agipd_saxs")
        g["trainId"], g["q"] = tids, q
        g["pulseId"] = np.arange(p, dtype=np.uint64)
        g["intensity"] = saxs.astype(np.float32)
        g["n_frames"] = (np.ones((n, p)) if n_frames is None else n_frames).astype(
            np.uint8
        )
        if waxs is not None:
            j = f.create_group("jungfrau_waxs_jf1")
            j["trainId"], j["q"] = tids, np.linspace(11.5, 23.7, waxs.shape[2])
            j["cellId"] = np.arange(waxs.shape[1], dtype=np.uint16)
            j["intensity"] = waxs.astype(np.float32)
            j["n_frames"] = np.ones(waxs.shape[:2], np.uint8)
    for group, name, values in (
        ("xgm_intensity", "Energy", np.full(n, 500.0) if xgm is None else xgm),
        ("total_transmission", "__xarray_dataarray_variable__",
         np.full(n, 0.1) if trans is None else trans),
    ):  # fmt: skip
        xr.DataArray(
            np.asarray(values, float),
            dims="trainId",
            coords={"trainId": tids},
            name=name,
        ).to_dataset().to_netcdf(path, group=group, mode="a", engine="h5netcdf")
    return path, q


def in_memory(path):
    """What DAMNIT hands a variable: agipd_saxs as the DataArray the variable
    returned (``trainId``, ``pulseId``, ``q`` and ``n_frames`` as coordinates),
    XGM and transmission as per-train arrays, jf1 likewise or None."""

    def grid(g, frame_dim):
        return xr.DataArray(
            g["intensity"][()],
            dims=("trainId", frame_dim, "q"),
            coords={
                "trainId": g["trainId"][()],
                frame_dim: g[frame_dim][()],
                "q": g["q"][()],
                "n_frames": (("trainId", frame_dim), g["n_frames"][()]),
            },
            name="intensity",
        )

    with h5py.File(path, "r") as f:
        saxs = grid(f["agipd_saxs"], "pulseId")
        jf1 = (
            grid(f["jungfrau_waxs_jf1"], "cellId") if "jungfrau_waxs_jf1" in f else None
        )
    xgm = xr.open_dataset(path, group="xgm_intensity").load().Energy
    trans = xr.open_dataset(path, group="total_transmission").load()
    return saxs, xgm, trans.__xarray_dataarray_variable__, jf1


MESH_RUNS = (120, 121, 122)


def write_mesh_store(root: Path, seed: int = 6) -> Path:
    """A DAMNIT store of one mesh block (`MESH_RUNS`, 400 trains, 20 % hits) and the
    static r423 (300 hits, 20 trains off the beam, 300 hits), with its sqlite."""
    rng = np.random.default_rng(seed)
    (root / "extracted_data").mkdir(parents=True, exist_ok=True)
    for r in MESH_RUNS:
        x, _ = mesh_run(rng, -4.3, 0.2, n=400)
        level = 10.0**x  # s/F with F = 1 (xgm 10 × transmission 0.1)
        saxs = np.broadcast_to(level[:, None, None], (400, 2, 30)).copy()
        write_damnit(root / "extracted_data" / f"p10400_r{r}.h5", saxs,
                     xgm=np.full(400, 10.0))  # fmt: skip
    level = 10.0 ** np.r_[np.full(300, -3.0), np.full(20, -4.3), np.full(300, -3.0)]
    write_damnit(root / "extracted_data" / "p10400_r423.h5",
                 np.broadcast_to(level[:, None, None], (620, 2, 30)).copy(),
                 xgm=np.full(620, 10.0))  # fmt: skip
    rows = [
        run_row(r, scan_type="dmesh SSHEX", total_transmission=0.0123, pulses=350.0)
        for r in MESH_RUNS
    ]
    write_sqlite(root / "runs.sqlite", rows + [run_row(423)])
    return root
