#!/usr/bin/env python
"""How to write the hit-train photon store faster (``context/xpcs-hits.md`` §10 G6).

On r0493 the parent's serial ``RingPhotonStore.put`` calls took 167 of 208 s of
the extraction (56 ms a train): seven shuffled gzip-4 datasets and a flush per
train. This times the alternatives on the same real trains and checks that
every arm reads back, through ``RingPhotonStore.get``, arrays identical to the
source. No ``src/`` change: it only imports the store.

The source is an existing store, opened read-only; nothing is written to it.
Every arm writes under ``--work-dir`` in scratch, which is deleted at the end
(``--keep`` leaves it).

Arms (all on the same trains, in the same order):

``A``
    the current ``RingPhotonStore.put``, gzip 4, flush per train, in the parent.
``A_nf``
    the same datasets, flush removed: what the flush costs.
``A_raw_nf``
    no compression and no flush: the per-dataset HDF5 overhead alone.
``B_g4_b<k>``
    workers each write a block file of ``k`` trains (gzip 4, written under a
    temporary name, then renamed); the parent copies the groups into the final
    store with ``Group.copy`` (H5Ocopy). Worker write time and parent merge
    time are reported apart.
``C_g1_b<k>``, ``C_none_b<k>``
    as B at gzip 1 and with no compression.
``D``, ``D_bf``, ``D_py``
    workers precompress each dataset as one chunk, by HDF5's own filters in
    an in-memory file (``D``, ``D_bf``) or by numpy shuffle and
    ``zlib.compress`` (``D_py``); the parent creates the datasets and calls
    ``write_direct_chunk``. ``D`` flushes per train, ``D_bf`` per block.

Also measured on its own: shuffle plus ``zlib.compress`` of the same arrays
outside HDF5 (``zlib_floor``), which is the compression CPU every compressed
arm must pay somewhere.

Usage:

    python scripts/bench_store_write.py --run 493 -w 36 --constraint 9374F
    python scripts/bench_store_write.py --run 493 --local --n-trains 40

``BENCH_SRC`` names a ``src`` directory to import ``analysis`` from instead of
this checkout's, for a tree that does not have ``analysis.xpcs`` yet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_src = os.environ.get("BENCH_SRC") or str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, _src)

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

from _common import (  # noqa: E402
    add_job_args,
    configure_logging,
    dispatch,
    report_header,
    stamp,
    write_report,
)

from analysis.threadenv import set_thread_env  # noqa: E402

set_thread_env()

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.common.status import FrameStatus  # noqa: E402
from analysis.xpcs.store import (  # noqa: E402
    RingPhotonStore,
    TrainPhotons,
    _group_name,
    delta_encode,
)

log = logging.getLogger("scripts.bench_store_write")

SCRATCH = Path("/gpfs/exfel/exp/MID/202601/p010400/scratch")
CACHED = Path("/gpfs/exfel/exp/MID/202601/p010400/usr/cached_files")
#: Worker core-seconds a train that extraction spends besides the store write:
#: read data 0.32 + read mask 0.57 + encode 0.65 (context/xpcs-hits.md §10 G6).
EXTRACT_CORE_S = 1.54
FIELDS = (
    "pulse_ids",
    "cell_ids",
    "frame_ptr",
    "pixel",
    "count",
    "flag_ptr",
    "flag_pixel",
)

# ── what is written ────────────────────────────────────────────────────────────


def arrays_of(photons: TrainPhotons) -> list[tuple[str, np.ndarray]]:
    """The seven datasets ``RingPhotonStore.put`` writes, in its order."""
    return [
        ("pulse_ids", photons.pulse_ids),
        ("cell_ids", photons.cell_ids),
        ("frame_ptr", photons.frame_ptr.astype(np.uint32)),
        ("pixel_step", delta_encode(photons.pixel, photons.frame_ptr)),
        ("count", photons.count),
        ("flag_ptr", photons.flag_ptr.astype(np.uint32)),
        ("flag_pixel", photons.flag_pixel),
    ]


def write_train(parent: h5py.Group, photons: TrainPhotons, level: int) -> None:
    """``put``'s layout with the gzip level (0 = none) as a parameter."""
    group = parent.create_group(_group_name(photons.train_id))
    for name, array in arrays_of(photons):
        if array.size and level:
            group.create_dataset(
                name, data=array, compression="gzip", compression_opts=level,
                shuffle=True,
            )  # fmt: skip
        else:
            group.create_dataset(name, data=array)
    group.attrs["sequence"] = photons.sequence
    group.attrs["status"] = int(photons.status)


def shuffle_deflate(array: np.ndarray, level: int) -> bytes:
    """What HDF5's shuffle then deflate filters make of one chunk."""
    array = np.ascontiguousarray(array)
    size = array.dtype.itemsize
    raw = array.view(np.uint8).reshape(-1, size).T.tobytes() if size > 1 else array.tobytes()
    return zlib.compress(raw, level)


# ── workers ────────────────────────────────────────────────────────────────────

_source: RingPhotonStore | None = None


def _init(source_path: str) -> None:
    global _source
    set_thread_env()
    _source = RingPhotonStore.open(Path(source_path))


def _noop() -> int:
    time.sleep(0.4)
    return os.getpid()


def _timed() -> tuple[float, float]:
    return time.perf_counter(), time.process_time()


def _block_b(path: str, train_ids: list[int], level: int) -> dict:
    """Read the block's trains (untimed), then write them to ``path`` via a temp name."""
    assert _source is not None
    trains = [_source.get(t) for t in train_ids]
    tmp = path + ".tmp"
    w0, c0 = _timed()
    with h5py.File(tmp, "w") as handle:
        for photons in trains:
            write_train(handle, photons, level)
    os.replace(tmp, path)
    w1, c1 = _timed()
    return {"path": path, "n": len(trains), "wall": w1 - w0, "cpu": c1 - c0}


def hdf5_chunks(photons: TrainPhotons, level: int) -> list[tuple[str, str, int, bytes]]:
    """Each dataset's one chunk as HDF5's own C filters make it, via an in-memory file."""
    parts = []
    with h5py.File("scratch", "w", driver="core", backing_store=False) as handle:
        for name, array in arrays_of(photons):
            if array.size:
                ds = handle.create_dataset(
                    name, data=array, chunks=(array.size,), compression="gzip",
                    compression_opts=level, shuffle=True,
                )  # fmt: skip
                payload = ds.id.read_direct_chunk((0,))[1]
                parts.append((name, array.dtype.str, array.size, payload))
            else:
                parts.append((name, array.dtype.str, 0, b""))
    return parts


def python_chunks(photons: TrainPhotons, level: int) -> list[tuple[str, str, int, bytes]]:
    """The same chunks with numpy's shuffle and ``zlib.compress``."""
    return [
        (name, a.dtype.str, a.size, shuffle_deflate(a, level) if a.size else b"")
        for name, a in arrays_of(photons)
    ]


def _block_d(train_ids: list[int], level: int, python: bool) -> dict:
    """Read the block's trains (untimed), then compress their chunks."""
    assert _source is not None
    trains = [_source.get(t) for t in train_ids]
    make = python_chunks if python else hdf5_chunks
    w0, c0 = _timed()
    out = []
    for photons in trains:
        parts = make(photons, level)
        out.append((photons.train_id, photons.sequence, int(photons.status), parts))
    w1, c1 = _timed()
    return {"trains": out, "n": len(trains), "wall": w1 - w0, "cpu": c1 - c0}


# ── arms ───────────────────────────────────────────────────────────────────────


def new_store(path: Path, source: RingPhotonStore) -> RingPhotonStore:
    return RingPhotonStore.create(
        path, source.extract_hash, source.envelope(), source.base, {"bench": True}
    )


def payload_bytes(path: Path) -> int:
    """Stored (compressed) bytes of every train dataset, without HDF5 metadata."""
    total = 0
    with h5py.File(path, "r") as handle:
        def visit(_name: str, obj: object) -> None:
            nonlocal total
            if isinstance(obj, h5py.Dataset):
                total += obj.id.get_storage_size()
        handle["trains"].visititems(visit)
    return total


def verify(path: Path, source: RingPhotonStore, train_ids: list[int]) -> dict:
    """Every train read back through ``get`` equals the source, field for field."""
    bad: list[str] = []
    with RingPhotonStore.open(path) as store:
        have = set(store.train_ids().tolist())
        for tid in train_ids:
            if tid not in have:
                bad.append(f"{tid}: missing")
                continue
            a, b = source.get(tid), store.get(tid)
            if (a.status, a.sequence) != (b.status, b.sequence):
                bad.append(f"{tid}: status/sequence")
            for field in FIELDS:
                x, y = getattr(a, field), getattr(b, field)
                if x.dtype != y.dtype or not np.array_equal(x, y):
                    bad.append(f"{tid}: {field}")
        base_same = all(
            np.array_equal(getattr(store.base, f), getattr(source.base, f))
            for f in ("cells", "ptr", "pixel")
        )
        extra = len(have - set(train_ids))
    return {
        "identical": not bad and base_same and extra == 0,
        "trains_checked": len(train_ids),
        "mismatches": bad[:10],
        "n_mismatches": len(bad),
        "base_identical": base_same,
        "extra_trains": extra,
    }


def arm_serial(
    name: str, final: Path, source: RingPhotonStore, trains: list[TrainPhotons]
) -> dict:
    """A, A_nf, A_raw_nf: the parent writes every train itself."""
    level, flush = {"A": (4, True), "A_nf": (4, False), "A_raw_nf": (0, False)}[name]
    store = new_store(final, source)
    w0, c0 = _timed()
    for photons in trains:
        if name == "A":
            store.put(photons)
        else:
            write_train(store._f["trains"], photons, level)
            if flush:
                store._f.flush()
    store.close()
    w1, c1 = _timed()
    n = len(trains)
    return {
        "parent_wall_s_per_train": (w1 - w0) / n,
        "parent_cpu_s_per_train": (c1 - c0) / n,
        "worker_wall_s_per_train": 0.0,
        "worker_cpu_s_per_train": 0.0,
        "wall_s": w1 - w0,
    }


def arm_blocks(
    name: str,
    final: Path,
    work: Path,
    source: RingPhotonStore,
    source_path: Path,
    train_ids: list[int],
    level: int,
    block: int,
    n_workers: int,
) -> dict:
    """B and C: workers write block files, the parent merges them with H5Ocopy."""
    blocks = [train_ids[i : i + block] for i in range(0, len(train_ids), block)]
    store = new_store(final, source)
    merge_w = merge_c = 0.0
    worker_w = worker_c = 0.0
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(
        n_workers, mp_context=ctx, initializer=_init, initargs=(str(source_path),)
    ) as pool:
        list(cf.wait([pool.submit(_noop) for _ in range(n_workers)]).done)
        t0 = time.perf_counter()
        futures = [
            pool.submit(_block_b, str(work / f"block_{i:04d}.h5"), ids, level)
            for i, ids in enumerate(blocks)
        ]
        trains_group = store._f["trains"]
        for future in cf.as_completed(futures):
            result = future.result()
            worker_w += result["wall"]
            worker_c += result["cpu"]
            m0, mc0 = _timed()
            with h5py.File(result["path"], "r") as handle:
                for key in handle:
                    trains_group.copy(handle[key], key)
            store._f.flush()
            os.unlink(result["path"])
            m1, mc1 = _timed()
            merge_w += m1 - m0
            merge_c += mc1 - mc0
        store.close()
        wall = time.perf_counter() - t0
    n = len(train_ids)
    return {
        "parent_wall_s_per_train": merge_w / n,
        "parent_cpu_s_per_train": merge_c / n,
        "worker_wall_s_per_train": worker_w / n,
        "worker_cpu_s_per_train": worker_c / n,
        "wall_s": wall,
        "blocks": len(blocks),
    }


def arm_chunks(
    name: str,
    final: Path,
    source: RingPhotonStore,
    source_path: Path,
    train_ids: list[int],
    block: int,
    n_workers: int,
    flush_per_train: bool,
    python: bool,
) -> dict:
    """D: workers compress, the parent only creates datasets and writes chunks."""
    blocks = [train_ids[i : i + block] for i in range(0, len(train_ids), block)]
    store = new_store(final, source)
    parent_w = parent_c = 0.0
    worker_w = worker_c = 0.0
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(
        n_workers, mp_context=ctx, initializer=_init, initargs=(str(source_path),)
    ) as pool:
        list(cf.wait([pool.submit(_noop) for _ in range(n_workers)]).done)
        t0 = time.perf_counter()
        futures = [pool.submit(_block_d, ids, 4, python) for ids in blocks]
        trains_group = store._f["trains"]
        for future in cf.as_completed(futures):
            result = future.result()
            worker_w += result["wall"]
            worker_c += result["cpu"]
            p0, pc0 = _timed()
            for tid, sequence, status, parts in result["trains"]:
                group = trains_group.create_group(_group_name(tid))
                for key, dtype, size, payload in parts:
                    if size:
                        ds = group.create_dataset(
                            key, shape=(size,), dtype=np.dtype(dtype), chunks=(size,),
                            compression="gzip", compression_opts=4, shuffle=True,
                        )  # fmt: skip
                        ds.id.write_direct_chunk((0,), payload)
                    else:
                        group.create_dataset(key, data=np.zeros(0, dtype=np.dtype(dtype)))
                group.attrs["sequence"] = sequence
                group.attrs["status"] = status
                if flush_per_train:
                    store._f.flush()
            if not flush_per_train:
                store._f.flush()
            p1, pc1 = _timed()
            parent_w += p1 - p0
            parent_c += pc1 - pc0
        store.close()
        wall = time.perf_counter() - t0
    n = len(train_ids)
    return {
        "parent_wall_s_per_train": parent_w / n,
        "parent_cpu_s_per_train": parent_c / n,
        "worker_wall_s_per_train": worker_w / n,
        "worker_cpu_s_per_train": worker_c / n,
        "wall_s": wall,
        "blocks": len(blocks),
    }


def zlib_floor(trains: list[TrainPhotons]) -> dict:
    """Shuffle plus zlib on the seven arrays, outside HDF5."""
    out = {}
    raw = sum(a.nbytes for t in trains for _, a in arrays_of(t)) / len(trains)
    out["raw_bytes_per_train"] = raw
    w0, c0 = _timed()
    for photons in trains:
        for _, array in arrays_of(photons):
            if array.size and array.dtype.itemsize > 1:
                array.view(np.uint8).reshape(-1, array.dtype.itemsize).T.copy()
    out["numpy_shuffle_cpu_s_per_train"] = (_timed()[1] - c0) / len(trains)
    for level in (4, 1):
        w0, c0 = _timed()
        size = 0
        for photons in trains:
            for _, array in arrays_of(photons):
                if array.size:
                    size += len(shuffle_deflate(array, level))
        w1, c1 = _timed()
        n = len(trains)
        out[f"gzip{level}"] = {
            "cpu_s_per_train": (c1 - c0) / n,
            "wall_s_per_train": (w1 - w0) / n,
            "bytes_per_train": size / n,
        }
    return out


# ── driver ─────────────────────────────────────────────────────────────────────


def model_wall(row: dict, n_model: int, n_workers: int) -> float:
    """Extraction wall for ``n_model`` trains if the parent and the pool overlap.

    The pool needs ``EXTRACT_CORE_S`` plus its own write seconds a train over
    ``n_workers`` cores; the parent is a single serial thread. Whichever is
    longer sets the wall. A model, not a measurement.
    """
    pool = n_model * (EXTRACT_CORE_S + row["worker_wall_s_per_train"]) / n_workers
    parent = n_model * row["parent_wall_s_per_train"]
    return max(pool, parent)


def run_one(args: argparse.Namespace) -> int:
    configure_logging("scripts.bench_store_write")
    source_path = CACHED / "agipd_ring_photons" / f"r{args.run:04d}" / "agipd_ring_photons.h5"
    work = args.work_dir / f"r{args.run:04d}_{stamp()}"
    work.mkdir(parents=True)
    assert not str(work).startswith(str(CACHED)), "never write under cached_files"

    source = RingPhotonStore.open(source_path)
    ids = np.sort(source.train_ids())
    ok = ids[source.status(ids) == int(FrameStatus.OK)]
    pick = np.unique(np.linspace(0, ok.size - 1, args.n_trains).astype(int))
    train_ids = [int(t) for t in ok[pick]]
    log.info("%d OK trains in the store, using %d spread over the run", ok.size, len(train_ids))
    trains = [source.get(t) for t in train_ids]
    floor = zlib_floor(trains)

    k = args.block
    arms: dict[str, dict] = {}
    spec = [
        ("A", None), ("A_nf", None), ("A_raw_nf", None),
        (f"B_g4_b{k}", (4, k)), (f"B_g4_b{args.big_block}", (4, args.big_block)),
        (f"C_g1_b{k}", (1, k)), (f"C_none_b{k}", (0, k)),
        ("D", "d"), ("D_bf", "d"), ("D_py", "d"),
    ]  # fmt: skip
    for name, how in spec:
        runs = []
        for rep in range(args.repeats):
            final = work / f"{name}_{rep}.h5"
            block_dir = work / f"{name}_{rep}_blocks"
            block_dir.mkdir()
            if how is None:
                row = arm_serial(name, final, source, trains)
            elif how == "d":
                row = arm_chunks(
                    name, final, source, source_path, train_ids, k, args.workers,
                    flush_per_train=(name != "D_bf"), python=(name == "D_py"),
                )  # fmt: skip
            else:
                row = arm_blocks(
                    name, final, block_dir, source, source_path, train_ids,
                    how[0], how[1], args.workers,
                )  # fmt: skip
            row["payload_bytes_per_train"] = payload_bytes(final) / len(train_ids)
            row["file_bytes_per_train"] = final.stat().st_size / len(train_ids)
            row["verify"] = verify(final, source, train_ids)
            runs.append(row)
            log.info(
                "%s rep %d: parent %.1f ms/train, worker %.1f ms/train, wall %.2f s, "
                "%.2f MB/train, identical=%s",
                name, rep, row["parent_wall_s_per_train"] * 1e3,
                row["worker_wall_s_per_train"] * 1e3, row["wall_s"],
                row["payload_bytes_per_train"] / 1e6, row["verify"]["identical"],
            )  # fmt: skip
            final.unlink()
            shutil.rmtree(block_dir, ignore_errors=True)
        best = min(runs, key=lambda r: r["wall_s"])
        arms[name] = {
            "runs": runs,
            "best": best,
            "median_parent_wall_s_per_train": statistics.median(
                r["parent_wall_s_per_train"] for r in runs
            ),
            "model_wall_s": model_wall(best, args.n_model, args.workers),
            "all_identical": all(r["verify"]["identical"] for r in runs),
        }

    source.close()
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    report = report_header(
        run=args.run,
        source=str(source_path),
        n_trains=len(train_ids),
        workers=args.workers,
        block=k,
        big_block=args.big_block,
        repeats=args.repeats,
        n_model=args.n_model,
        extract_core_s_per_train=EXTRACT_CORE_S,
        bench_src=_src,
        zlib_floor=floor,
        arms=arms,
    )
    write_report(report, args.json, f"bench_store_write_r{args.run:04d}")
    print(f"\n{'arm':14s} {'parent ms':>10s} {'worker ms':>10s} {'cpu p/w ms':>13s} "
          f"{'wall s':>7s} {'model s':>8s} {'MB/train':>9s} ok")  # fmt: skip
    for name, arm in arms.items():
        b = arm["best"]
        print(
            f"{name:14s} {b['parent_wall_s_per_train'] * 1e3:10.2f} "
            f"{b['worker_wall_s_per_train'] * 1e3:10.2f} "
            f"{b['parent_cpu_s_per_train'] * 1e3:6.1f}/{b['worker_cpu_s_per_train'] * 1e3:<6.1f} "
            f"{b['wall_s']:7.2f} {arm['model_wall_s']:8.1f} "
            f"{b['payload_bytes_per_train'] / 1e6:9.3f} {arm['all_identical']}"
        )  # fmt: skip
    return 0 if all(a["all_identical"] for a in arms.values()) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=int, nargs="+", default=[493])
    parser.add_argument("--n-trains", type=int, default=300)
    parser.add_argument("--block", type=int, default=8, help="trains a block (the run's)")
    parser.add_argument("--big-block", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-model", type=int, default=2994, help="trains in the modelled wall")
    parser.add_argument("--work-dir", type=Path, default=SCRATCH / "bench_store_write" / "work")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    add_job_args(parser, workers=36, time="01:00:00")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return dispatch(
        args,
        args.run,
        lambda run: run_one(argparse.Namespace(**vars(args) | {"run": run})),
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
