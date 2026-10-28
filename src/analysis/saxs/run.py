"""Orchestration for one run.

The parent builds everything the workers need, gates on the self-test, then
fans blocks out to spawned single-threaded workers and writes their results
itself. Nothing in the hot loop depends on XGM, transmission or background
corrections: those are applied afterwards, to the stored sums.

Up to three files come out of one read (:mod:`analysis.saxs.products`): the
frame table, the per-pixel window sums and the per-frame photon statistics.
Only the incomplete ones are opened: a run whose frame table is complete gets
its sums or statistics without the frame table, the operator or the masks
being written again. :func:`run_pixel_sums` is the same pass with the window
sums alone.

A run whose files are complete is answered from them and nothing is written
(:func:`analysis.common.run.guarded_pass`). Otherwise every file is locked
for as long as the pass writes it, so a second DAMNIT job on the same run
refuses rather than writing alongside.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common.complete import Completeness
from analysis.common.cpu import phase, set_thread_env
from analysis.common.lock import OutputLock
from analysis.common.masks import frame_bad
from analysis.common.plan import Block, RunPlan, evenly_spaced
from analysis.common.run import fan_out, guarded_pass, reducer
from analysis.common.status import FrameStatus
from analysis.common.writer import pooled_per_train
from analysis.saxs import masks as masks_module
from analysis.saxs import operator as operator_module
from analysis.saxs import worker as worker_module
from analysis.saxs.complete import is_complete
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.operator import operator_differences
from analysis.saxs.photon_stats import FILE_NAME as PHOTON_FILE_NAME
from analysis.saxs.photon_stats import PhotonStatsConfig, ring_labels_from
from analysis.saxs.pixel_sums import FILE_NAME, WindowSpec
from analysis.saxs.plan import build_plan
from analysis.saxs.products import (
    FrameTable,
    PassRun,
    PhotonStats,
    Product,
    WindowSums,
    check_files,
    products_for,
    train_ids,
)
from analysis.saxs.selftest import run_selftest
from analysis.saxs.writer import AgipdSaxsWriter, per_pulse

__all__ = [
    "REDUCERS",
    "run_agipd_saxs",
    "run_pixel_sums",
]

#: What ``run_agipd_saxs`` may return, read from the frame table. ``pooled``
#: is the pooled per-train I(q); ``per_pulse`` is the (trainId, pulseId, q)
#: grid, which is what the DAMNIT variable stores; ``none`` skips the reduction
#: for a caller that only wants the files written.
REDUCERS = {"pooled": pooled_per_train, "per_pulse": per_pulse, "none": None}

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
    nothing is written, unless ``cfg.overwrite``. Otherwise every file is
    locked (:func:`analysis.common.lock.output_lock`) for as long as the pass
    writes it, and the check is made again under the lock.

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
    reduction = reducer(REDUCERS, reduce)

    # Before any pool exists, so spawned children inherit it.
    set_thread_env()
    # Wall time is recorded in provenance rather than left to whoever launched
    # the job. It spans the plan, the self-test, the pool and the writer.
    started_at = time.time()
    started = time.perf_counter()

    beside = Path(output_path).parent if output_path else None
    output_path = Path(output_path) if output_path else cfg.output_file
    if pixel_sums_path is None and beside is not None:
        pixel_sums_path = beside / FILE_NAME
    if photon_stats_path is None and beside is not None:
        photon_stats_path = beside / PHOTON_FILE_NAME
    work_dir = Path(work_dir) if work_dir is not None else output_path.parent

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
    products = products_for(
        cfg,
        output_path=output_path,
        pixel_sums_path=pixel_sums_path,
        photon_stats_path=photon_stats_path,
        op=op,
        labels=labels,
    )

    def integrate(locks: dict[Path, OutputLock]) -> Any:
        # Before the plan, the masks and the self-test, so a file that can only
        # be reprocessed is refused in a second rather than after a minute.
        AgipdSaxsWriter.check_resumable(
            cfg, output_path, operator_check=partial(operator_differences, op=op)
        )
        _integrate(
            PassRun(cfg, None, started_at, started, setup, op=op, labels=labels),
            products,
            dc=dc,
            ai=ai,
            run_dir=run_dir,
            work_dir=work_dir,
            pool_factory=pool_factory,
            locks=locks,
        )
        return answer()

    def answer() -> Any:
        # Every product read back as it now stands, written by this pass or not.
        log.info(
            "r%d: %s is complete; answered from the file, nothing written",
            cfg.run,
            output_path,
        )
        check_files(products, cfg, noun="frame reached OK")
        return reduction(output_path) if reduction else None

    return guarded_pass(
        [product.path for product in products],
        check=not cfg.overwrite,
        complete=lambda: is_complete(
            output_path,
            cfg,
            pixel_sums_path=pixel_sums_path,
            operator=op,
            photon_stats_path=photon_stats_path,
            ring_labels=labels,
        ),
        answer=answer,
        integrate=integrate,
    )


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
    doing only the windows not yet written. ``overwrite`` never applies to the
    window sums.

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
    sums = WindowSums(cfg, Path(pixel_sums_path or cfg.pixel_sums_file))

    def answer() -> Path:
        log.info("r%d: %s is complete; nothing written", cfg.run, sums.path)
        check_files([sums], cfg, noun="train was summed")
        return sums.path

    def integrate(locks: dict[Path, OutputLock]) -> Path:
        _integrate(
            PassRun(cfg, None, started_at, started, {}),
            [sums],
            dc=dc,
            ai=None,
            run_dir=run_dir,
            work_dir=None,
            pool_factory=pool_factory,
            locks=locks,
        )
        return answer()

    return guarded_pass(
        [sums.path],
        check=True,
        complete=lambda: Completeness(sums.path, tuple(sums.reasons(None))),
        answer=answer,
        integrate=integrate,
    )


def _integrate(
    run: PassRun,
    products: list[Product],
    *,
    dc: Any,
    ai: Any,
    run_dir: Path | None,
    work_dir: Path | None,
    pool_factory: Callable[..., Any] | None,
    locks: dict[Path, OutputLock],
) -> None:
    """The pass itself, run under the locks the caller holds.

    Only the incomplete products are opened. With the frame table complete, no
    base masks, self-test, operator or masks file are made: the workers read
    and sum or histogram without integrating.
    """
    cfg, setup = run.cfg, run.setup
    frames = next((p for p in products if isinstance(p, FrameTable)), None)
    frames_to_do = frames is not None and (frames.forced or bool(frames.reasons(None)))
    frame_ids = None if frames is None or frames_to_do else train_ids(frames.path)
    todo = [
        p
        for p in products
        if (p is frames and frames_to_do) or (p is not frames and p.reasons(frame_ids))
    ]

    with phase(setup, "plan"):
        run.plan = plan = build_plan(cfg, dc=dc)
    if run.op is not None:
        with phase(setup, "static_mask"):
            run.static = masks_module.build_static_bad(cfg)
    paths = worker_module.WorkerPaths(None, None, str(run_dir) if run_dir else None)
    if frames_to_do:
        work_dir.mkdir(parents=True, exist_ok=True)
        with phase(setup, "base_masks"):
            run.base_masks, run.sampled = _build_base_masks(
                cfg, plan, run.op, run.static, dc
            )
        with phase(setup, "selftest"):
            _run_selftest(cfg, plan, run.op, ai, run.base_masks, dc)
        with phase(setup, "save_operator_and_masks"):
            operator_path = operator_module.save_operator(
                run.op, work_dir / "operator.npz"
            )
            masks_path = masks_module.save_masks(run.base_masks, work_dir / "masks.npz")
        paths = worker_module.WorkerPaths(
            str(operator_path), str(masks_path), paths.run_dir
        )
    if any(isinstance(p, WindowSums) for p in todo):
        run.windows = WindowSpec.for_plan(plan, cfg.pixel_sum_trains)

    with ExitStack() as stack:
        # The side products first: a file this run must not touch is refused
        # before ``overwrite`` has recreated the frame table for nothing.
        for product in sorted(todo, key=lambda p: p is frames):
            product.open(stack, run)
        needed = frozenset().union(*(p.blocks for p in todo))
        blocks = [b for b in plan.blocks if b.index in needed]
        log.info(
            "%d of %d blocks to process: %s",
            len(blocks),
            len(plan.blocks),
            ", ".join(f"{len(p.blocks)} to {p.verb}" for p in todo),
        )
        for product in todo:
            product.begin(blocks)
        integrated = frames.blocks if frames_to_do else frozenset()
        photon = next((p for p in todo if isinstance(p, PhotonStats)), None)

        totals = fan_out(
            _Outputs(todo),
            blocks,
            process=worker_module.process_block,
            n_workers=cfg.workers,
            initializer=worker_module.init,
            initargs=(
                paths,
                cfg,
                run.windows,
                needed - integrated,
                photon.spec(run) if photon is not None else None,
            ),
            pool_factory=pool_factory,
            accumulate=("unseen_cells",),
        )
        for product in todo:
            product.finalise(run, totals, locks[product.path])


@dataclass(slots=True)
class _Outputs:
    """The ``out`` :func:`fan_out` writes through: every product being written."""

    products: list[Product]

    def write_block(self, block: Block, result: Any) -> None:
        for product in self.products:
            product.write(block, result)

    def mark(self, block: Block, status: FrameStatus, message: str = "") -> None:
        for product in self.products:
            product.mark(block, status, message)

    def mark_remaining(self, status: FrameStatus) -> None:
        for product in self.products:
            product.mark_remaining(status)


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
