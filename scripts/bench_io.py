#!/usr/bin/env python
"""Virtual CXI files against our EXtra-data reads (CLAUDE.md open task 19b).

Read-only on proc. Everything this writes goes to ``--work-dir`` in scratch,
and the verdict to JSON beside this script.

Phases, all in one job by default:

``build``
    ``write_virtual_cxi`` for AGIPD and for jf1 on one run: wall time, file
    size, the number of virtual mappings, and how the source files are named
    (absolute? which path form?), the datasets the file carries, their fill
    values, and the file's mode and ACL.
``identity``
    What frame identity the file carries (trainId, pulseId, cellId,
    memoryCell), checked against EXtra-data's coordinates and frame data on a
    few trains spread over the run.
``read``
    Per-frame read cost, data and mask timed separately, at each process count
    in ``--procs``. Every (variant, process count) reads its own trains, so the
    page cache never favours one variant. The variants:

    * AGIPD ``extra`` — ``AGIPD1M[...]["image.*"].ndarray(decompress_threads=1)``
      per train, as ``analysis.saxs.worker`` does;
    * AGIPD ``vds_train`` — one h5py slice of the virtual dataset per train;
    * AGIPD ``vds_frame`` — one slice per frame (16 module chunks each), the
      access a per-frame XCCA or XPCS reader would make;
    * JUNGFRAU ``extra`` — ``JUNGFRAU[...]["data.*"].ndarray()`` per train, as
      ``analysis.waxs.worker`` does (all 16 cells);
    * JUNGFRAU ``vds_train`` — the 16 rows of a train in one slice: exactly one
      source chunk, ``(1, 16, 512, 1024)``;
    * JUNGFRAU ``vds_lit`` / ``vds_lit_cache`` — the lit cells one row at a
      time, with HDF5's default chunk cache (1 MiB, smaller than one 32 MiB
      chunk) and with a 256 MiB one.
``taped``
    A mock run written like the real proc files (int16 counts, uint32 mask,
    chunks ``(1, 512, 128)``), a virtual CXI over it, and one module's file
    moved away: what the virtual file returns, and what EXtra-data does.

Usage:

    python scripts/bench_io.py --run 423 -w 36 --constraint 9374F
"""

from __future__ import annotations

# Before numpy, h5py or EXtra-data are imported: one thread per process, as in
# the passes, so that the process counts below mean cores.
from _common import (  # noqa: E402
    add_job_args,
    configure_logging,
    dispatch,
    report_header,
    stamp,
    write_report,
)

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

log = logging.getLogger("bench_io")

WORK_ROOT = Path("/gpfs/exfel/exp/MID/202601/p010400/scratch/s3_speed/cxi")
REPO = Path(__file__).resolve().parent.parent

#: Where the datasets sit in a CXI file written by EXtra-data 1.24.
CXI_DETECTOR = "entry_1/instrument_1/detector_1"

#: Variants per detector kind, in the order they run.
VARIANTS = {
    "agipd": ("extra", "vds_train", "vds_frame"),
    "jf1": ("extra", "vds_train", "vds_lit", "vds_lit_cache"),
}

#: Trains each process reads, per kind and process count. Small on purpose:
#: about 8 % of r0423's AGIPD trains and 11 % of its jf1 trains in all.
TRAINS_PER_PROCESS = {"agipd": {1: 4, 36: 2}, "jf1": {1: 8, 36: 2}}

#: HDF5's chunk cache for the ``vds_lit_cache`` variant: room for several
#: 32 MiB JUNGFRAU chunks.
BIG_CHUNK_CACHE = 256 * 1024**2

#: Trains spread over the run for the identity check.
IDENTITY_TRAINS = 5


# ── opening things ────────────────────────────────────────────────────────────


def open_detector(kind: str, proposal: int, run: int) -> Any:
    from extra_data import open_run

    dc = open_run(proposal, run, data="proc")
    if kind == "agipd":
        from extra_data.components import AGIPD1M

        return AGIPD1M(dc, min_modules=16)
    from analysis.waxs.config import config_for
    from analysis.waxs.plan import open_detector as open_jungfrau

    return open_jungfrau(config_for(proposal, run, kind), dc)


def cxi_path(work: Path, kind: str, run: int) -> Path:
    return work / f"r{run:04d}_{kind}.cxi"


def train_rows(path: Path) -> dict[int, tuple[int, int]]:
    """``{trainId: (first_row, n_rows)}`` from the CXI file's own trainId."""
    with h5py.File(path, "r") as handle:
        train_ids = handle["entry_1/trainId"][()]
    unique, first, counts = np.unique(train_ids, return_index=True, return_counts=True)
    return {
        int(t): (int(f), int(c)) for t, f, c in zip(unique, first, counts, strict=True)
    }


# ── build ─────────────────────────────────────────────────────────────────────


def build(kind: str, proposal: int, run: int, work: Path) -> dict[str, Any]:
    det = open_detector(kind, proposal, run)
    path = cxi_path(work, kind, run)
    started = time.perf_counter()
    det.write_virtual_cxi(str(path))
    wall = time.perf_counter() - started

    report: dict[str, Any] = {
        "path": str(path),
        "wall_s": wall,
        "size_bytes": path.stat().st_size,
        "mode": oct(path.stat().st_mode & 0o777),
        "acl": _getfacl(path),
    }
    with h5py.File(path, "r") as handle:
        names: list[str] = []
        handle.visit(names.append)
        report["objects"] = names
        for name in ("data", "mask", "gain"):
            key = f"{CXI_DETECTOR}/{name}"
            if key not in handle:
                continue
            dataset = handle[key]
            sources = dataset.virtual_sources()
            files = sorted({source.file_name for source in sources})
            report[name] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "fillvalue": repr(dataset.fillvalue),
                "n_mappings": len(sources),
                "n_source_files": len(files),
                "source_files_absolute": all(os.path.isabs(f) for f in files),
                "source_file_examples": files[:2],
                "source_files_exist": all(os.path.exists(f) for f in files),
            }
    return report


def _getfacl(path: Path) -> str:
    try:
        return subprocess.run(
            ["getfacl", "-p", str(path)], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error!r}"


# ── identity ──────────────────────────────────────────────────────────────────


def identity(kind: str, proposal: int, run: int, work: Path) -> dict[str, Any]:
    from extra_data import by_id

    det = open_detector(kind, proposal, run)
    path = cxi_path(work, kind, run)
    rows = train_rows(path)
    report: dict[str, Any] = {
        "trains_in_cxi": len(rows),
        "trains_in_extra_data": len(det.train_ids),
        "same_train_set": sorted(rows) == sorted(int(t) for t in det.train_ids),
    }
    picks = np.linspace(0, len(det.train_ids) - 1, IDENTITY_TRAINS).astype(int)
    checks = []
    with h5py.File(path, "r") as handle:
        entry = handle["entry_1"]
        report["entry_1_datasets"] = sorted(
            k for k in entry if isinstance(entry[k], h5py.Dataset)
        )
        data = handle[f"{CXI_DETECTOR}/data"]
        mask = handle[f"{CXI_DETECTOR}/mask"]
        for pick in picks:
            train_id = int(det.train_ids[pick])
            first, count = rows[train_id]
            selected = det.select_trains(by_id[[train_id]])
            check: dict[str, Any] = {"train_id": train_id, "rows": count}
            if kind == "agipd":
                key = selected["image.data"]
                x = key.ndarray(decompress_threads=1)
                m = selected["image.mask"].ndarray(decompress_threads=1)
                cells = entry["cellId"][first : first + count]
                check |= {
                    "frames_match": count == x.shape[1],
                    "pulse_ids_match": bool(
                        np.array_equal(
                            entry["pulseId"][first : first + count],
                            key.pulse_id_coordinates(),
                        )
                    ),
                    "cell_ids_match": bool(
                        np.array_equal(cells[:, 0], key.cell_id_coordinates())
                        and (cells == cells[:, :1]).all()
                    ),
                    # the VDS is (frame, module, ss, fs), EXtra-data (module, frame, …)
                    "data_match": bool(
                        np.array_equal(
                            data[first : first + count], np.swapaxes(x, 0, 1)
                        )
                    ),
                    "mask_match": bool(
                        np.array_equal(
                            mask[first : first + count], np.swapaxes(m, 0, 1)
                        )
                    ),
                }
            else:
                x = selected["data.adc"].ndarray().reshape(count, -1)
                m = selected["data.mask"].ndarray().reshape(count, -1)
                cells = selected["data.memoryCell"].ndarray().reshape(-1)
                check |= {
                    "memory_cells_match": bool(
                        np.array_equal(
                            entry["memoryCell"][first : first + count], cells
                        )
                    ),
                    "data_match": bool(
                        np.array_equal(
                            data[first : first + count].reshape(count, -1),
                            x,
                            equal_nan=True,
                        )
                    ),
                    "mask_match": bool(
                        np.array_equal(
                            mask[first : first + count].reshape(count, -1), m
                        )
                    ),
                }
            checks.append(check)
    report["checks"] = checks
    report["all_match"] = all(
        value
        for check in checks
        for key, value in check.items()
        if key.endswith("match")
    )
    return report


# ── read ──────────────────────────────────────────────────────────────────────

_STATE: dict[str, Any] = {}


def worker_init(kind: str, proposal: int, run: int, path: str, cache: int) -> None:
    set_thread_env()
    _STATE["kind"] = kind
    _STATE["det"] = open_detector(kind, proposal, run)
    _STATE["rows"] = train_rows(Path(path))
    kwargs = {"rdcc_nbytes": cache} if cache else {}
    handle = h5py.File(path, "r", **kwargs)
    _STATE["data"] = handle[f"{CXI_DETECTOR}/data"]
    _STATE["mask"] = handle[f"{CXI_DETECTOR}/mask"]
    if kind != "agipd":
        from analysis.waxs.config import EXPECTED_LIT_CELLS

        memory_cell = handle["entry_1/memoryCell"][()]
        _STATE["lit"] = np.isin(memory_cell, EXPECTED_LIT_CELLS)


def read_task(variant: str, train_ids: list[int]) -> dict[str, Any]:
    """Read ``train_ids`` one way; time data and mask apart."""
    from extra_data import by_id

    started_at = time.time()
    kind, det, rows = _STATE["kind"], _STATE["det"], _STATE["rows"]
    data, mask = _STATE["data"], _STATE["mask"]
    t_data = t_mask = 0.0
    frames = lit_frames = 0
    nbytes = 0

    for train_id in train_ids:
        first, count = rows[train_id]
        if variant == "extra":
            selected = det.select_trains(by_id[[train_id]])
            key_data, key_mask = (
                ("image.data", "image.mask")
                if kind == "agipd"
                else ("data.adc", "data.mask")
            )
            kwargs = {"decompress_threads": 1} if kind == "agipd" else {}
            t0 = time.perf_counter()
            x = selected[key_data].ndarray(**kwargs)
            t1 = time.perf_counter()
            m = selected[key_mask].ndarray(**kwargs)
            t2 = time.perf_counter()
            pieces = [(x, m)]
        elif variant == "vds_train":
            t0 = time.perf_counter()
            x = data[first : first + count]
            t1 = time.perf_counter()
            m = mask[first : first + count]
            t2 = time.perf_counter()
            pieces = [(x, m)]
        else:
            if variant == "vds_frame":
                wanted = range(first, first + count)
            else:  # vds_lit, vds_lit_cache
                wanted = [r for r in range(first, first + count) if _STATE["lit"][r]]
            t0 = time.perf_counter()
            xs = [data[r] for r in wanted]
            t1 = time.perf_counter()
            ms = [mask[r] for r in wanted]
            t2 = time.perf_counter()
            pieces = list(zip(xs, ms, strict=True))
        t_data += t1 - t0
        t_mask += t2 - t1
        frames += count
        lit_frames += (
            int(_STATE["lit"][first : first + count].sum())
            if "lit" in _STATE
            else count
        )
        nbytes += sum(x.nbytes + m.nbytes for x, m in pieces)

    return {
        "started_at": started_at,
        "ended_at": time.time(),
        "t_data": t_data,
        "t_mask": t_mask,
        "frames": frames,
        "lit_frames": lit_frames,
        "bytes": nbytes,
        "open_fds": len(os.listdir("/proc/self/fd")),
    }


def read_rates(
    kind: str, proposal: int, run: int, work: Path, procs: list[int]
) -> dict[str, Any]:
    """Every variant at every process count, each on trains of its own."""
    det = open_detector(kind, proposal, run)
    path = cxi_path(work, kind, run)
    rows = train_rows(path)
    usable = [int(t) for t in det.train_ids if int(t) in rows]
    plan = [(variant, n) for n in procs for variant in VARIANTS[kind]]
    assigned = allocate_trains(
        usable, [n * TRAINS_PER_PROCESS[kind][n] for _, n in plan]
    )

    results: dict[str, Any] = {}
    for (variant, n), mine in zip(plan, assigned, strict=True):
        per = TRAINS_PER_PROCESS[kind][n]
        tasks = [mine[i * per : (i + 1) * per] for i in range(n)]
        cache = BIG_CHUNK_CACHE if variant.endswith("_cache") else 0
        label = f"{variant}@{n}"
        log.info("%s %s: %d trains on %d processes", kind, label, len(mine), n)
        try:
            with get_context("spawn").Pool(
                n,
                initializer=worker_init,
                initargs=(kind, proposal, run, str(path), cache),
            ) as pool:
                outcomes = pool.starmap(
                    read_task, [(variant, task) for task in tasks], chunksize=1
                )
        except Exception as error:  # noqa: BLE001 - recorded per variant
            results[label] = {"error": repr(error), "traceback": traceback.format_exc()}
            continue
        results[label] = summarise(outcomes, n, kind)
    return results


def allocate_trains(trains: list[int], quotas: list[int]) -> list[list[int]]:
    """Disjoint train lists of the given sizes, interleaved over the whole run.

    Trains are taken evenly spaced over the run and dealt round-robin to the
    lists that still need some, so no variant reads only the start or only the
    end of it, and no train is read twice — a second read would come from the
    page cache.
    """
    needed = sum(quotas)
    if needed > len(trains):
        raise RuntimeError(f"needs {needed} trains, the run has {len(trains)}")
    picks = np.linspace(0, len(trains) - 1, needed).round().astype(int)
    lists: list[list[int]] = [[] for _ in quotas]
    turn = 0
    for train in (trains[i] for i in picks):
        while len(lists[turn % len(quotas)]) >= quotas[turn % len(quotas)]:
            turn += 1
        lists[turn % len(quotas)].append(train)
        turn += 1
    return lists


def summarise(outcomes: list[dict[str, Any]], n: int, kind: str) -> dict[str, Any]:
    frames = sum(o["frames"] for o in outcomes)
    lit = sum(o["lit_frames"] for o in outcomes)
    t_data = sum(o["t_data"] for o in outcomes)
    t_mask = sum(o["t_mask"] for o in outcomes)
    wall = max(o["ended_at"] for o in outcomes) - min(o["started_at"] for o in outcomes)
    summary = {
        "processes": n,
        "frames": frames,
        "data_ms_per_frame": 1e3 * t_data / frames,
        "mask_ms_per_frame": 1e3 * t_mask / frames,
        "wall_s": wall,
        "frames_per_s": frames / wall if wall > 0 else None,
        "gb_decompressed": sum(o["bytes"] for o in outcomes) / 1e9,
        "max_open_fds": max(o["open_fds"] for o in outcomes),
    }
    if kind != "agipd":
        # what the WAXS pass needs is the lit cells; a per-train read pays for
        # all sixteen
        summary |= {
            "lit_frames": lit,
            "data_ms_per_lit_frame": 1e3 * t_data / lit,
            "mask_ms_per_lit_frame": 1e3 * t_mask / lit,
            "lit_frames_per_s": lit / wall if wall > 0 else None,
        }
    return summary


# ── taped ─────────────────────────────────────────────────────────────────────


def taped(work: Path) -> dict[str, Any]:
    """Move one source file away and read the virtual file again."""
    from extra_data import RunDirectory
    from extra_data.components import AGIPD1M

    sys.path.insert(0, str(REPO / "tests" / "saxs"))
    from mockrun import write_mock_run

    root = work / "taped"
    shutil.rmtree(root, ignore_errors=True)
    mock = write_mock_run(root / "r0423", occupancy=0.05)
    # EXtra-data's CXI writer takes a file for calibrated data, and so writes
    # the mask, only when ``image/gain`` exists. The real proc files carry it
    # (uint8, chunks (1, 512, 128), gzip); the mock does not, so add it.
    for module_file in sorted(mock.path.glob("CORR-*.h5")):
        with h5py.File(module_file, "r+") as handle:
            for _name, group in _image_groups(handle):
                rows = group["data"].shape[0]
                group.create_dataset(
                    "gain",
                    shape=(rows, 512, 128),
                    dtype=np.uint8,
                    chunks=(1, 512, 128),
                    compression="gzip",
                    fillvalue=0,
                )
    path = root / "mock.cxi"
    AGIPD1M(RunDirectory(str(mock.path)), min_modules=16).write_virtual_cxi(str(path))
    module = 5
    with h5py.File(path, "r") as handle:
        before_x = handle[f"{CXI_DETECTOR}/data"][0, module]
        before_m = handle[f"{CXI_DETECTOR}/mask"][0, module]
        sources = sorted(
            {s.file_name for s in handle[f"{CXI_DETECTOR}/data"].virtual_sources()}
        )

    victim = next(Path(f) for f in sources if f"AGIPD{module:02d}" in f)
    aside = victim.with_suffix(".taped")
    victim.rename(aside)
    report: dict[str, Any] = {
        "module_removed": module,
        "before": {
            "data_nonzero": int(np.count_nonzero(before_x)),
            "mask_nonzero": int(np.count_nonzero(before_m)),
        },
    }
    try:
        with h5py.File(path, "r") as handle:
            after_x = handle[f"{CXI_DETECTOR}/data"][0, module]
            after_m = handle[f"{CXI_DETECTOR}/mask"][0, module]
            other_x = handle[f"{CXI_DETECTOR}/data"][0, module - 1]
            missing = [
                s.file_name
                for s in handle[f"{CXI_DETECTOR}/data"].virtual_sources()
                if not os.path.exists(s.file_name)
            ]
        report["vds_after"] = {
            "raised": False,
            "data_dtype": str(after_x.dtype),
            "data_all_zero": bool(not after_x.any()),
            "mask_all_ones": bool((after_m == np.uint32(0xFFFFFFFF)).all()),
            "neighbour_module_still_read": bool(other_x.any()),
            "missing_sources_detectable": len(missing),
        }
    except Exception as error:  # noqa: BLE001 - the answer
        report["vds_after"] = {"raised": True, "error": repr(error)}
    try:
        det = AGIPD1M(RunDirectory(str(mock.path)), min_modules=16)
        report["extra_data_after"] = {"raised": False, "trains": len(det.train_ids)}
    except Exception as error:  # noqa: BLE001 - the answer
        report["extra_data_after"] = {"raised": True, "error": repr(error)}
    finally:
        aside.rename(victim)
    return report


def _image_groups(handle: h5py.File) -> list[tuple[str, h5py.Group]]:
    """Every ``INSTRUMENT/<source>/image`` group of a module file."""
    found: list[tuple[str, h5py.Group]] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Group) and name.endswith("/image") and "data" in obj:
            found.append((name, obj))

    handle["INSTRUMENT"].visititems(visit)
    return found


# ── the job ───────────────────────────────────────────────────────────────────


def run_one(args: argparse.Namespace) -> int:
    configure_logging("bench_io")
    work = Path(args.work_dir) if args.work_dir else WORK_ROOT
    work.mkdir(parents=True, exist_ok=True)
    phases = args.phase
    destination = args.json or Path(__file__).with_name(
        f"bench_io_r{args.run:04d}_{stamp()}.json"
    )
    report: dict[str, Any] = report_header(
        run=args.run,
        proposal=args.proposal,
        work_dir=str(work),
        phases=phases,
        procs=args.procs,
        cpu=_cpu_model(),
        package_versions=_versions(),
        hdf5=h5py.version.hdf5_version,
    )
    steps = []
    for kind in ("agipd", "jf1"):
        if "build" in phases:
            steps.append(
                (
                    f"build/{kind}",
                    lambda k=kind: build(k, args.proposal, args.run, work),
                )
            )
        if "identity" in phases:
            steps.append(
                (
                    f"identity/{kind}",
                    lambda k=kind: identity(k, args.proposal, args.run, work),
                )
            )
        if "read" in phases:
            steps.append(
                (
                    f"read/{kind}",
                    lambda k=kind: read_rates(
                        k, args.proposal, args.run, work, args.procs
                    ),
                )
            )
    if "taped" in phases:
        steps.append(("taped", lambda: taped(work)))

    failed = False
    for name, step in steps:
        started = time.perf_counter()
        try:
            result = step()
        except Exception as error:  # noqa: BLE001 - recorded, the rest runs
            result = {"error": repr(error), "traceback": traceback.format_exc()}
            failed = True
        section, _, kind = name.partition("/")
        report.setdefault(section, {})
        if kind:
            report[section][kind] = result
        else:
            report[section] = result
        log.info("%s done in %.1f s", name, time.perf_counter() - started)
        # rewritten after every step, so a step that kills the job loses
        # only itself
        write_report(report, destination, "bench_io")
    return 1 if failed else 0


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _versions() -> dict[str, str]:
    from analysis.common.cpu import package_versions

    return package_versions()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--proposal", type=int, default=10400)
    parser.add_argument("--run", type=int, nargs="+", required=True)
    add_job_args(parser, workers=36, time="01:00:00")
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=["build", "identity", "read", "taped"],
        default=["build", "identity", "read", "taped"],
    )
    parser.add_argument("--procs", type=int, nargs="+", default=[1, 36])
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    known = sorted(TRAINS_PER_PROCESS["agipd"])
    for n in args.procs:
        if n not in known:
            parser.error(f"--procs: no train budget for {n}; known {known}")
    return args


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
