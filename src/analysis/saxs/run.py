"""Orchestration for one run.

The parent builds everything the workers need, gates on the self-test, then
fans blocks out to spawned single-threaded workers and writes their results
itself. Nothing in the hot loop depends on XGM, transmission or background
corrections: those are applied afterwards, to the stored sums.

Two files come out of one read: the frame table (``agipd_saxs.h5``) and the
per-pixel window sums (``agipd_pixel_sums.h5``, context file §15).
:func:`run_pixel_sums` writes only the second, for runs whose frame table
already exists.
"""

from __future__ import annotations

import logging
import platform
import socket
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common.cpu import file_sha256, package_versions, phase, set_thread_env
from analysis.common.masks import frame_bad
from analysis.common.plan import Block, RunPlan, evenly_spaced
from analysis.common.run import FanOutTotals, base_provenance, fan_out
from analysis.common.status import FrameStatus
from analysis.common.writer import IncompleteRun
from analysis.saxs import masks as masks_module
from analysis.saxs import operator as operator_module
from analysis.saxs import worker as worker_module
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.pixel_sums import FILE_NAME, PixelSumWriter, WindowSpec
from analysis.saxs.plan import build_plan
from analysis.saxs.selftest import run_selftest
from analysis.saxs.writer import AgipdSaxsWriter

__all__ = [
    "REDUCERS",
    "run_agipd_saxs",
    "run_pixel_sums",
]

#: What ``run_agipd_saxs`` may return. ``pooled`` is the per-train I(q) of
#: ``per_pulse`` is the (trainId, pulseId, q) grid, which is
#: what the DAMNIT variable stores; ``none`` skips the reduction for a caller
#: that only wants the file written.
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
) -> Any:
    """Integrate every frame of a run and return pooled per-train ``I(q)``.

    The per-pixel window sums are written alongside unless
    ``cfg.pixel_sum_trains`` is ``None``. A block the frame table already holds
    is read again only if one of its windows is unwritten, and then is summed
    without being integrated.

    :param dc: an already-open ``DataCollection``; ``None`` opens the proc run.
    :param geometry: an ``AGIPD_1MGeometry``; ``None`` loads ``cfg.geometry_file``.
    :param run_dir: directory the workers open the run from, instead of
        resolving ``cfg.proposal``/``cfg.run``.
    :param pool_factory: builds the executor. Defaults to :func:`default_pool`;
        injecting it lets the broken-pool and worker-error paths be exercised
        without killing real processes or putting test hooks in ``worker``.
    :param reduce: which reduction to return, one of :data:`REDUCERS`. The
        output file is identical either way — this only chooses what is read
        back out of it before returning.
    :param pixel_sums_path: where the window sums go. ``None`` puts them
        beside ``output_path`` when that is given, else at
        ``cfg.pixel_sums_file``.

    :raises IncompleteRun: some frame is not ``OK``, or some train with rows
        was left out of the window sums, and ``cfg.allow_incomplete`` is False.
        The output files are still written and closed first, so the ledgers
        explain what happened.
    """
    # Checked before any work, so a typo costs nothing rather than surfacing
    # after the whole run has been integrated.
    if reduce not in REDUCERS:
        raise ValueError(f"reduce must be one of {REDUCERS}, got {reduce!r}")

    # Before any pool exists, so spawned children inherit it.
    set_thread_env()
    # Wall time is an acceptance criterion, so it is
    # recorded in provenance rather than left to whoever launched the job. It
    # spans the plan, the self-test, the pool and the writer.
    started_at = time.time()
    started = time.perf_counter()

    work_dir = (
        Path(work_dir)
        if work_dir is not None
        else (Path(output_path).parent if output_path else cfg.output_file.parent)
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    setup: dict[str, float] = {}
    with phase(setup, "plan"):
        plan = build_plan(cfg, dc=dc)
    with phase(setup, "operator"):
        if geometry is None:
            geometry = operator_module.geometry_from_config(cfg)
        op, ai = operator_module.build_operator(geometry, cfg)
    with phase(setup, "static_mask"):
        static = masks_module.build_static_bad(cfg)
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
    windows = (
        WindowSpec.for_plan(plan, cfg.pixel_sum_trains)
        if cfg.pixel_sum_trains is not None
        else None
    )
    if pixel_sums_path is None:
        pixel_sums_path = (
            Path(output_path).parent / FILE_NAME if output_path else cfg.pixel_sums_file
        )

    with ExitStack() as stack:
        # The sums first: a window-sums file this run must not touch is refused
        # before ``overwrite`` has recreated the frame table for nothing.
        sums = (
            stack.enter_context(
                PixelSumWriter.open_or_create(cfg, plan, windows, pixel_sums_path)
            )
            if windows is not None
            else None
        )
        out = stack.enter_context(
            AgipdSaxsWriter.open_or_create(cfg, plan, output_path)
        )
        out.store_operator(op)
        out.store_masks(base_masks, sampled)
        integrate = {b.index for b in plan.blocks if not out.block_complete(b)}
        wanted = {b.index for b in sums.blocks_to_do(plan.blocks)} if sums else set()
        todo = [b for b in plan.blocks if b.index in integrate | wanted]
        skip = frozenset(wanted - integrate)
        log.info(
            "%d of %d blocks to process, %d of them for their window sums only",
            len(todo),
            len(plan.blocks),
            len(skip),
        )
        if sums is not None:
            sums.begin(todo)

        totals = fan_out(
            _Outputs(out, sums, skip),
            todo,
            process=worker_module.process_block,
            n_workers=cfg.workers,
            initializer=worker_module.init,
            initargs=(paths, cfg, windows, skip),
            pool_factory=pool_factory,
            accumulate=("unseen_cells",),
        )

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
        )
        if sums is not None:
            sums.finalise(
                _pixel_provenance(cfg, plan, totals, started_at, started, setup)
            )
        incomplete = out.any_not_ok() or (sums is not None and sums.incomplete())
        summary = out.status_summary()
        if sums is not None:
            summary = {"frames": summary, "window_sums": sums.status_summary()}
        reduced = None
        if reduce == "pooled":
            reduced = out.pooled_per_train()
        elif reduce == "per_pulse":
            reduced = out.per_pulse()

    if incomplete and not cfg.allow_incomplete:
        raise IncompleteRun(f"not every frame reached OK: {summary}")
    return reduced


def run_pixel_sums(
    cfg: AgipdSaxsConfig,
    *,
    dc: Any = None,
    run_dir: Path | None = None,
    pixel_sums_path: Path | None = None,
    pool_factory: Callable[..., Any] | None = None,
) -> Path:
    """Write only the per-pixel window sums of a run (context file §15.4).

    For runs whose frame table already exists: this never opens
    ``agipd_saxs.h5`` and builds no operator, masks or self-test, since nothing
    is integrated. So the frame table's config hash does not matter here: the
    existing files were written with ``npt=2000``, and a full pass at the
    default config would refuse every one. A rerun resumes, doing only the
    windows not yet written.

    :param dc: an already-open ``DataCollection``; ``None`` opens the proc run.
    :param run_dir: directory the workers open the run from.
    :param pixel_sums_path: the file; ``None`` is ``cfg.pixel_sums_file``.
    :param pool_factory: builds the executor, as for :func:`run_agipd_saxs`.
    :returns: the file written.
    :raises IncompleteRun: a train with rows was left out and
        ``cfg.allow_incomplete`` is False.
    """
    if cfg.pixel_sum_trains is None:
        raise ValueError("pixel_sum_trains is None: there are no windows to sum")
    set_thread_env()
    started_at = time.time()
    started = time.perf_counter()
    path = Path(pixel_sums_path) if pixel_sums_path else cfg.pixel_sums_file

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
        sums.finalise(_pixel_provenance(cfg, plan, totals, started_at, started, setup))
        incomplete = sums.incomplete()
        summary = sums.status_summary()

    if incomplete and not cfg.allow_incomplete:
        raise IncompleteRun(f"not every train was summed: {summary}")
    return path


@dataclass(slots=True)
class _Outputs:
    """The ``out`` :func:`fan_out` writes through: the frame table and the sums.

    A block in ``skip`` was read only for its windows. Its frame rows are
    already final in the file, so neither its result nor its failure may touch
    them.
    """

    frames: AgipdSaxsWriter | None
    sums: PixelSumWriter | None
    skip: frozenset[int]

    def write_block(self, block: Block, result: Any) -> None:
        integrated = block.index not in self.skip
        if result.integrated != integrated:
            raise ValueError(
                f"block {block.index}: the worker integrated={result.integrated}, "
                f"the parent asked for integrated={integrated}"
            )
        if self.frames is not None and integrated:
            self.frames.write_block(block, result)
        if self.sums is not None:
            self.sums.write_block(block, result)

    def mark(self, block: Block, status: FrameStatus, message: str = "") -> None:
        if self.frames is not None and block.index not in self.skip:
            self.frames.mark(block, status, message)
        if self.sums is not None:
            self.sums.fail_block(block, status, message)

    def mark_remaining(self, status: FrameStatus) -> None:
        # Unwritten windows need no stamp: a rerun rebuilds them from scratch.
        if self.frames is not None:
            self.frames.mark_remaining(status)


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
