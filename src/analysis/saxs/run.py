"""Orchestration for one run.

The parent builds everything the workers need, gates on the self-test, then
fans blocks out to spawned single-threaded workers and writes their results
itself. Nothing in the hot loop depends on XGM, transmission or background
corrections: those are applied afterwards, to the stored sums.

Three files come out of one read: the frame table (``agipd_saxs.h5``), the
per-pixel window sums (``agipd_pixel_sums.h5``) and the per-frame photon
statistics (``agipd_photon_stats.h5``). Only the incomplete ones are opened: a run whose
frame table is complete gets its sums or statistics without the frame table,
the operator or the masks being written again. :func:`run_pixel_sums` writes
only the sums.

Both entry points first ask whether their output is already complete, and if it
is, answer from the file and write nothing. Otherwise they
hold a lock on every file they write for as long as they write it, so a second
DAMNIT job on the same run refuses rather than writing alongside.
"""

from __future__ import annotations

import logging
import platform
import socket
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.common.complete import Completeness, frame_table_reasons
from analysis.common.cpu import file_sha256, package_versions, phase, set_thread_env
from analysis.common.lock import OutputLock, lock_path, output_lock
from analysis.common.masks import frame_bad
from analysis.common.plan import Block, RunPlan, evenly_spaced
from analysis.common.run import FanOutTotals, base_provenance, fan_out
from analysis.common.status import FrameStatus
from analysis.common.writer import IncompleteRun, pooled_per_train, status_counts
from analysis.saxs import masks as masks_module
from analysis.saxs import operator as operator_module
from analysis.saxs import worker as worker_module
from analysis.saxs.complete import (
    _train_ids,
    is_complete,
    operator_differences,
    window_sums_reasons,
)
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.photon_stats import FILE_NAME as PHOTON_FILE_NAME
from analysis.saxs.photon_stats import (
    PhotonSpec,
    PhotonStatsConfig,
    PhotonStatsWriter,
    photon_provenance,
    photon_stats_reasons,
    ring_labels_from,
)
from analysis.saxs.pixel_sums import FAILED, FILE_NAME, PixelSumWriter, WindowSpec
from analysis.saxs.plan import build_plan
from analysis.saxs.selftest import run_selftest
from analysis.saxs.writer import AgipdSaxsWriter, per_pulse

__all__ = [
    "REDUCERS",
    "run_agipd_saxs",
    "run_pixel_sums",
]

#: What ``run_agipd_saxs`` may return. ``pooled`` is the pooled per-train I(q);
#: ``per_pulse`` is the (trainId, pulseId, q) grid, which is what the DAMNIT
#: variable stores; ``none`` skips the reduction for a caller that only wants
#: the file written.
REDUCERS = ("pooled", "per_pulse", "none")

log = logging.getLogger(__name__)


def run_agipd_saxs(
    cfg: AgipdSaxsConfig,
    *,
    dc: Any = None,
    geometry: Any = None,
    run_dir: Path | None = None,
    output_path: Path | None = None,
    work_dir: Path | None = None,
    pool_factory: Callable[..., Any] | None = None,
    reduce: str = "pooled",
    pixel_sums_path: Path | None = None,
    photon_stats_path: Path | None = None,
) -> Any:
    """Integrate every frame of a run and return pooled per-train ``I(q)``.

    The per-pixel window sums are written alongside unless
    ``cfg.pixel_sum_trains`` is ``None``, and the per-frame photon statistics
    unless ``cfg.photon_stats`` is False. A block the frame table already holds
    is read again only if one of its windows is unwritten or its frames are not
    histogrammed, and then is summed or histogrammed without being integrated.

    :param dc: an already-open ``DataCollection``; ``None`` opens the proc run.
    :param geometry: an ``AGIPD_1MGeometry``; ``None`` loads ``cfg.geometry_file``.
    :param run_dir: directory the workers open the run from, instead of
        resolving ``cfg.proposal``/``cfg.run``.
    :param pool_factory: builds the executor. Defaults to :func:`default_pool`;
        injecting it lets the broken-pool and worker-error paths be exercised
        without killing real processes or putting test hooks in ``worker``.
    :param reduce: which reduction to return, one of :data:`REDUCERS`. The
        output file is identical either way. This only chooses what is read
        back out of it before returning.
    :param pixel_sums_path: where the window sums go. ``None`` puts them
        beside ``output_path`` when that is given, else at
        ``cfg.pixel_sums_file``.
    :param photon_stats_path: where the photon statistics go, on the same terms
        (``cfg.photon_stats_file``).

    A run whose output is already complete
    (:func:`analysis.saxs.complete.is_complete`) is answered from the file and
    nothing is written, unless ``cfg.overwrite``. Otherwise both files are
    locked (:func:`analysis.common.lock.output_lock`) for as long as the pass
    writes them, and the check is made again under the lock.

    :raises IncompleteRun: some frame is not ``OK``, or some train with rows
        was left out of the window sums, and ``cfg.allow_incomplete`` is False.
        The output files are still written and closed first, so the ledgers
        explain what happened. A complete file that holds such frames raises
        the same, without being written.
    :raises analysis.common.lock.OutputBusy: another process is writing one of
        the files.
    :raises analysis.common.writer.OperatorMismatch: the frame table was
        integrated by another operator or pyFAI, so it cannot be resumed.
    """
    # Checked before any work, so a typo costs nothing rather than surfacing
    # after the whole run has been integrated.
    if reduce not in REDUCERS:
        raise ValueError(f"reduce must be one of {REDUCERS}, got {reduce!r}")

    # Before any pool exists, so spawned children inherit it.
    set_thread_env()
    # Wall time is recorded in provenance rather than left to whoever launched
    # the job. It spans the plan, the self-test, the pool and the writer.
    started_at = time.time()
    started = time.perf_counter()

    if pixel_sums_path is None:
        pixel_sums_path = (
            Path(output_path).parent / FILE_NAME if output_path else cfg.pixel_sums_file
        )
    if photon_stats_path is None:
        photon_stats_path = (
            Path(output_path).parent / PHOTON_FILE_NAME
            if output_path
            else cfg.photon_stats_file
        )
    photon_stats_path = Path(photon_stats_path)
    output_path = Path(output_path) if output_path else cfg.output_file
    work_dir = Path(work_dir) if work_dir is not None else output_path.parent
    guarded = [output_path]
    if cfg.pixel_sum_trains is not None:
        guarded.append(Path(pixel_sums_path))
    if cfg.photon_stats:
        guarded.append(photon_stats_path)

    # The operator first: it reads only the geometry, and the completeness
    # check needs its sha256 to tell a file integrated by another pyFAI apart.
    setup: dict[str, float] = {}
    with phase(setup, "operator"):
        if geometry is None:
            geometry = operator_module.geometry_from_config(cfg)
        op, ai = operator_module.build_operator(geometry, cfg)
    labels = None
    if cfg.photon_stats:
        with phase(setup, "rings"):
            labels = ring_labels_from(ai, PhotonStatsConfig.for_saxs(cfg))

    def complete() -> Completeness:
        return is_complete(
            output_path,
            cfg,
            pixel_sums_path=pixel_sums_path,
            operator=op,
            photon_stats_path=photon_stats_path,
            ring_labels=labels,
        )

    # Skipped while a lock exists, since the file may be mid-write; the check
    # under the lock below then decides.
    if not cfg.overwrite and not any(lock_path(p).exists() for p in guarded):
        if complete():
            return _from_file(
                cfg, output_path, pixel_sums_path, reduce, photon_stats_path
            )
    with ExitStack() as held:
        locks = {p: held.enter_context(output_lock(p)) for p in guarded}
        if not cfg.overwrite:
            verdict = complete()
            if verdict:
                return _from_file(
                    cfg, output_path, pixel_sums_path, reduce, photon_stats_path
                )
            log.info("%s", verdict)
        # Before the plan, the masks and the self-test, so a file that can only
        # be reprocessed is refused in a second rather than after a minute.
        AgipdSaxsWriter.check_resumable(
            cfg, output_path, operator_check=partial(operator_differences, op=op)
        )
        return _integrate(
            cfg,
            dc=dc,
            run_dir=run_dir,
            output_path=output_path,
            work_dir=work_dir,
            pool_factory=pool_factory,
            reduce=reduce,
            pixel_sums_path=Path(pixel_sums_path),
            photon_stats_path=photon_stats_path,
            op=op,
            ai=ai,
            labels=labels,
            setup=setup,
            started_at=started_at,
            started=started,
            locks=locks,
        )


def _from_file(
    cfg: AgipdSaxsConfig,
    output_path: Path,
    pixel_sums_path: Path,
    reduce: str,
    photon_stats_path: Path | None = None,
) -> Any:
    """Answer a complete run from its files, exactly as the pass would have.

    Opens them read-only. ``IncompleteRun`` is raised on the same terms as at
    the end of a pass: a complete file can still hold frames that are final
    without being ``OK``, such as ``LABEL_MISMATCH``.
    """
    with h5py.File(output_path, "r") as handle:
        summary: dict[str, Any] = status_counts(
            handle["frames/status"][:], nonzero_only=True
        )
        incomplete = any(name != FrameStatus.OK.name for name in summary)
        reduced = None
        if reduce == "pooled":
            reduced = pooled_per_train(handle)
        elif reduce == "per_pulse":
            reduced = per_pulse(handle)
    if cfg.pixel_sum_trains is not None:
        with h5py.File(pixel_sums_path, "r") as handle:
            trains = handle["trains/status"][:]
        incomplete |= bool(np.isin(trains, [int(code) for code in FAILED]).any())
        summary = {
            "frames": summary,
            "window_sums": status_counts(trains, nonzero_only=True),
        }
    if cfg.photon_stats and photon_stats_path is not None:
        with h5py.File(photon_stats_path, "r") as handle:
            photon = status_counts(handle["frames/status"][:], nonzero_only=True)
        incomplete |= any(name != FrameStatus.OK.name for name in photon)
        if not isinstance(summary.get("frames"), dict):
            summary = {"frames": summary}
        summary["photon_stats"] = photon
    log.info(
        "r%d: %s is complete; answered from the file, nothing written",
        cfg.run,
        output_path,
    )
    if incomplete and not cfg.allow_incomplete:
        raise IncompleteRun(f"not every frame reached OK: {summary}")
    return reduced


def _integrate(
    cfg: AgipdSaxsConfig,
    *,
    dc: Any,
    run_dir: Path | None,
    output_path: Path,
    work_dir: Path,
    pool_factory: Callable[..., Any] | None,
    reduce: str,
    pixel_sums_path: Path,
    photon_stats_path: Path,
    op: Any,
    ai: Any,
    labels: np.ndarray | None,
    setup: dict[str, float],
    started_at: float,
    started: float,
    locks: dict[Path, OutputLock],
) -> Any:
    """The pass itself, run under the locks :func:`run_agipd_saxs` holds.

    Only the incomplete products are opened. With the frame table complete, no
    base masks, self-test, operator or masks file are made: the workers read
    and sum or histogram without integrating.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_to_do = cfg.overwrite or bool(
        frame_table_reasons(
            output_path,
            cfg,
            AgipdSaxsWriter,
            partial(operator_differences, op=op),
        )
    )
    sums_to_do = cfg.pixel_sum_trains is not None and bool(
        window_sums_reasons(
            pixel_sums_path,
            cfg,
            frame_train_ids=None if frames_to_do else _train_ids(output_path),
        )
    )
    photon_cfg = PhotonStatsConfig.for_saxs(cfg) if cfg.photon_stats else None
    photon_to_do = photon_cfg is not None and bool(
        photon_stats_reasons(photon_stats_path, photon_cfg, labels)
    )

    with phase(setup, "plan"):
        plan = build_plan(cfg, dc=dc)
    with phase(setup, "static_mask"):
        static = masks_module.build_static_bad(cfg)
    if frames_to_do:
        with phase(setup, "base_masks"):
            base_masks, sampled = _build_base_masks(cfg, plan, op, static, dc)
        with phase(setup, "selftest"):
            _run_selftest(cfg, plan, op, ai, base_masks, dc)
        with phase(setup, "save_operator_and_masks"):
            operator_path = operator_module.save_operator(op, work_dir / "operator.npz")
            masks_path = masks_module.save_masks(base_masks, work_dir / "masks.npz")
        paths = worker_module.WorkerPaths(
            str(operator_path), str(masks_path), str(run_dir) if run_dir else None
        )
    else:
        paths = worker_module.WorkerPaths(None, None, str(run_dir) if run_dir else None)
    windows = WindowSpec.for_plan(plan, cfg.pixel_sum_trains) if sums_to_do else None

    with ExitStack() as stack:
        # The side products first: a file this run must not touch is refused
        # before ``overwrite`` has recreated the frame table for nothing.
        sums = (
            stack.enter_context(
                PixelSumWriter.open_or_create(cfg, plan, windows, pixel_sums_path)
            )
            if windows is not None
            else None
        )
        photon_out = (
            stack.enter_context(
                PhotonStatsWriter.open_or_create(photon_cfg, plan, photon_stats_path)
            )
            if photon_to_do
            else None
        )
        out = (
            stack.enter_context(
                AgipdSaxsWriter.open_or_create(
                    cfg,
                    plan,
                    output_path,
                    operator_check=partial(operator_differences, op=op),
                )
            )
            if frames_to_do
            else None
        )
        if out is not None:
            out.store_operator(op)
            out.store_masks(base_masks, sampled)
        if photon_out is not None:
            photon_out.store_rings(photon_cfg, labels, op.sha256)
        integrate = (
            {b.index for b in plan.blocks if not out.block_complete(b)}
            if out
            else set()
        )
        wanted = {b.index for b in sums.blocks_to_do(plan.blocks)} if sums else set()
        histogram = (
            {b.index for b in plan.blocks if not photon_out.block_complete(b)}
            if photon_out
            else set()
        )
        todo = [b for b in plan.blocks if b.index in integrate | wanted | histogram]
        skip = frozenset((wanted | histogram) - integrate)
        log.info(
            "%d of %d blocks to process: %d to integrate, %d to sum, %d to histogram",
            len(todo),
            len(plan.blocks),
            len(integrate),
            len(wanted),
            len(histogram),
        )
        if sums is not None:
            sums.begin(todo)
        spec = (
            PhotonSpec(labels, static.bad, photon_cfg.k_max, frozenset(histogram))
            if photon_out is not None
            else None
        )

        totals = fan_out(
            _Outputs(out, sums, skip, photon_out, frozenset(histogram)),
            todo,
            process=worker_module.process_block,
            n_workers=cfg.workers,
            initializer=worker_module.init,
            initargs=(paths, cfg, windows, skip, spec),
            pool_factory=pool_factory,
            accumulate=("unseen_cells",),
        )

        if out is not None:
            out.finalise(
                base_provenance(
                    cfg,
                    plan,
                    out,
                    started_at=started_at,
                    started=started,
                    totals=totals,
                    setup_timings=setup,
                    operator_sha256=op.sha256,
                    input_file_sha256={
                        name: (file_sha256(path) if path else None)
                        for name, path in cfg.input_files.items()
                    },
                    run_checks=plan.checks,
                )
                | {
                    "masks_sha256": base_masks.sha256,
                    "static_mask_sha256": static.sha256,
                    "static_mask_sources": [
                        {
                            "name": s.name,
                            "path": s.path,
                            "sha256": s.sha256,
                            "n_excluded": s.n_excluded,
                        }
                        for s in static.sources
                    ],
                    "unseen_cell_frames": int(totals.extra["unseen_cells"]),
                }
                | locks[output_path].provenance()
            )
        if sums is not None:
            sums.finalise(
                _pixel_provenance(cfg, plan, totals, started_at, started, setup)
                | locks[Path(pixel_sums_path)].provenance()
            )
        if photon_out is not None:
            photon_out.finalise(
                photon_provenance(
                    photon_cfg,
                    plan,
                    photon_out,
                    op.sha256,
                    started_at=started_at,
                    started=started,
                    totals=totals,
                    setup_timings=setup,
                )
                | {"static_mask_sha256": static.sha256}
                | locks[photon_stats_path].provenance()
            )

    # Every product read back as it now stands, written by this pass or not.
    return _from_file(cfg, output_path, pixel_sums_path, reduce, photon_stats_path)


def run_pixel_sums(
    cfg: AgipdSaxsConfig,
    *,
    dc: Any = None,
    run_dir: Path | None = None,
    pixel_sums_path: Path | None = None,
    pool_factory: Callable[..., Any] | None = None,
) -> Path:
    """Write only the per-pixel window sums of a run.

    For runs whose frame table already exists: this never opens
    ``agipd_saxs.h5`` and builds no operator, masks or self-test, since nothing
    is integrated, so a frame table written under another config (another
    ``npt``, say) neither blocks it nor is touched by it. A rerun resumes,
    doing only the windows not yet written.

    :param dc: an already-open ``DataCollection``; ``None`` opens the proc run.
    :param run_dir: directory the workers open the run from.
    :param pixel_sums_path: the file; ``None`` is ``cfg.pixel_sums_file``.
    :param pool_factory: builds the executor, as for :func:`run_agipd_saxs`.
    :returns: the file written or, when it was already complete, the file
        left untouched.
    :raises IncompleteRun: a train with rows was left out and
        ``cfg.allow_incomplete`` is False.
    :raises analysis.common.lock.OutputBusy: another process is writing it.
    """
    if cfg.pixel_sum_trains is None:
        raise ValueError("pixel_sum_trains is None: there are no windows to sum")
    set_thread_env()
    started_at = time.time()
    started = time.perf_counter()
    path = Path(pixel_sums_path) if pixel_sums_path else cfg.pixel_sums_file

    # ``overwrite`` never applies to the window sums, so it does not bypass
    # this check either.
    if not lock_path(path).exists() and not window_sums_reasons(path, cfg):
        return _sums_from_file(cfg, path)
    with output_lock(path) as lock:
        reasons = window_sums_reasons(path, cfg)
        if not reasons:
            return _sums_from_file(cfg, path)
        log.info("%s is not complete: %s", path, "; ".join(reasons))
        return _sum_windows(
            cfg,
            dc=dc,
            run_dir=run_dir,
            path=path,
            pool_factory=pool_factory,
            started_at=started_at,
            started=started,
            lock=lock,
        )


def _sums_from_file(cfg: AgipdSaxsConfig, path: Path) -> Path:
    """Answer a complete window-sums file as :func:`run_pixel_sums` would."""
    with h5py.File(path, "r") as handle:
        trains = handle["trains/status"][:]
    log.info("r%d: %s is complete; nothing written", cfg.run, path)
    if np.isin(trains, [int(code) for code in FAILED]).any() and not (
        cfg.allow_incomplete
    ):
        summary = status_counts(trains, nonzero_only=True)
        raise IncompleteRun(f"not every train was summed: {summary}")
    return path


def _sum_windows(
    cfg: AgipdSaxsConfig,
    *,
    dc: Any,
    run_dir: Path | None,
    path: Path,
    pool_factory: Callable[..., Any] | None,
    started_at: float,
    started: float,
    lock: OutputLock,
) -> Path:
    """The window-sums pass itself, run under the lock :func:`run_pixel_sums` holds."""
    setup: dict[str, float] = {}
    with phase(setup, "plan"):
        plan = build_plan(cfg, dc=dc)
    windows = WindowSpec.for_plan(plan, cfg.pixel_sum_trains)
    paths = worker_module.WorkerPaths(None, None, str(run_dir) if run_dir else None)

    with PixelSumWriter.open_or_create(cfg, plan, windows, path) as sums:
        todo = sums.blocks_to_do(plan.blocks)
        skip = frozenset(b.index for b in todo)
        log.info("%d of %d blocks to sum", len(todo), len(plan.blocks))
        sums.begin(todo)
        totals = fan_out(
            _Outputs(None, sums, skip),
            todo,
            process=worker_module.process_block,
            n_workers=cfg.workers,
            initializer=worker_module.init,
            initargs=(paths, cfg, windows, skip),
            pool_factory=pool_factory,
        )
        sums.finalise(
            _pixel_provenance(cfg, plan, totals, started_at, started, setup)
            | lock.provenance()
        )
        incomplete = sums.incomplete()
        summary = sums.status_summary()

    if incomplete and not cfg.allow_incomplete:
        raise IncompleteRun(f"not every train was summed: {summary}")
    return path


@dataclass(slots=True)
class _Outputs:
    """The ``out`` :func:`fan_out` writes through: frame table, sums, statistics.

    A block in ``skip`` was read only for its windows or its statistics. Its
    frame rows are already final in the file, so neither its result nor its
    failure may touch them. Only blocks in ``histogram`` reach the statistics.
    """

    frames: AgipdSaxsWriter | None
    sums: PixelSumWriter | None
    skip: frozenset[int]
    photon: PhotonStatsWriter | None = None
    histogram: frozenset[int] = frozenset()

    def write_block(self, block: Block, result: Any) -> None:
        integrated = block.index not in self.skip
        if result.integrated != integrated:
            raise ValueError(
                f"block {block.index}: the worker integrated={result.integrated}, "
                f"the parent asked for integrated={integrated}"
            )
        wanted = self.photon is not None and block.index in self.histogram
        if wanted != (result.photon is not None):
            raise ValueError(
                f"block {block.index}: the worker histogrammed="
                f"{result.photon is not None}, the parent asked for {wanted}"
            )
        if self.frames is not None and integrated:
            self.frames.write_block(block, result)
        if self.sums is not None:
            self.sums.write_block(block, result)
        if wanted:
            self.photon.write_block(block, result.photon)

    def mark(self, block: Block, status: FrameStatus, message: str = "") -> None:
        if self.frames is not None and block.index not in self.skip:
            self.frames.mark(block, status, message)
        if self.sums is not None:
            self.sums.fail_block(block, status, message)
        if self.photon is not None and block.index in self.histogram:
            self.photon.mark(block, status, message)

    def mark_remaining(self, status: FrameStatus) -> None:
        # Unwritten windows need no stamp: a rerun rebuilds them from scratch.
        if self.frames is not None:
            self.frames.mark_remaining(status)
        if self.photon is not None:
            self.photon.mark_remaining(status)


def _pixel_provenance(
    cfg: AgipdSaxsConfig,
    plan: RunPlan,
    totals: FanOutTotals,
    started_at: float,
    started: float,
    setup: dict[str, float],
) -> dict[str, Any]:
    """What the window-sums file records about the run that wrote it."""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "n_workers": cfg.workers,
        "n_blocks": len(plan.blocks),
        "started_at": started_at,
        "wall_s": time.perf_counter() - started,
        "bits_present": int(totals.bits_present),
        "timings": totals.timings,
        "setup_timings": {**setup, "write_blocks": totals.write_s},
        "run_checks": plan.checks,
    }


def _build_base_masks(
    cfg: AgipdSaxsConfig, plan: RunPlan, op: Any, static: Any, dc: Any
) -> tuple[Any, np.ndarray]:
    """Sample trains evenly over the run and majority-vote their masks."""
    from extra_data import by_id

    ok_trains = np.array(
        [t.train_id for t in plan.trains if t.status is FrameStatus.OK],
        dtype=np.uint64,
    )
    sampled = evenly_spaced(ok_trains, cfg.base_mask_trains)

    if dc is None:
        from extra_data import open_run

        dc = open_run(cfg.proposal, cfg.run, data="proc")
    from extra_data.components import AGIPD1M

    det = AGIPD1M(dc, detector_name=cfg.detector_name, min_modules=cfg.min_modules)

    accumulator = masks_module.BaseMaskAccumulator(
        op, static, mask_bits=cfg.mask_bits, expected_bits=cfg.expected_bits
    )
    for train_id in sampled.tolist():
        selected = det.select_trains(by_id[[train_id]])
        key = selected["image.mask"]
        accumulator.update(key.ndarray(decompress_threads=1), key.cell_id_coordinates())
    return accumulator.finalise(), sampled


def _run_selftest(
    cfg: AgipdSaxsConfig, plan: RunPlan, op: Any, ai: Any, base_masks: Any, dc: Any
) -> None:
    """Compare sparse against pyFAI on real frames before the pool starts."""
    from extra_data import by_id

    if dc is None:
        from extra_data import open_run

        dc = open_run(cfg.proposal, cfg.run, data="proc")
    from extra_data.components import AGIPD1M

    det = AGIPD1M(dc, detector_name=cfg.detector_name, min_modules=cfg.min_modules)
    ok_trains = [t.train_id for t in plan.trains if t.status is FrameStatus.OK]
    if not ok_trains:
        raise ValueError("no OK trains to self-test on")
    chosen = evenly_spaced(np.array(ok_trains, dtype=np.uint64), 2)

    engine = ai.engines[next(iter(ai.engines))].engine
    frames = []
    for train_id in chosen.tolist():
        selected = det.select_trains(by_id[[train_id]])
        data = selected["image.data"].ndarray(decompress_threads=1)
        mask_key = selected["image.mask"]
        mask = mask_key.ndarray(decompress_threads=1)
        cells = mask_key.cell_id_coordinates()
        for frame in range(min(data.shape[1], cfg.selftest_frames // 2 or 1)):
            cell = int(cells[frame])
            base_bad, base_denominator = base_masks.for_cell(cell)
            frames.append(
                (
                    data[:, frame].reshape(-1),
                    frame_bad(mask[:, frame], cfg.mask_bits, base_masks.static_bad),
                    base_bad,
                    base_denominator,
                )
            )
            if len(frames) >= cfg.selftest_frames:
                break
        if len(frames) >= cfg.selftest_frames:
            break

    report = run_selftest(engine, op, frames)
    log.info(
        "self-test passed on %d frames: max rel S %.2e, N %.2e, V %.2e",
        report.n_frames,
        report.max_rel_signal,
        report.max_rel_normalization,
        report.max_rel_variance,
    )
