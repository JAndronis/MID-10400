"""Orchestration for one run.

The parent builds everything the workers need, gates on the self-test, then
fans blocks out to spawned single-threaded workers and writes their results
itself. Nothing in the hot loop depends on XGM, transmission or background
corrections: those are applied afterwards, to the stored sums.
"""

from __future__ import annotations

import logging
import platform
import socket
import time
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# Re-exported: ``default_pool`` is part of this module's documented surface.
from analysis.common.cpu import default_pool, package_versions, phase
from analysis.saxs import masks as masks_module
from analysis.saxs import operator as operator_module
from analysis.saxs import worker as worker_module
from analysis.saxs.config import AgipdSaxsConfig, file_sha256
from analysis.saxs.plan import RunPlan, build_plan
from analysis.saxs.selftest import run_selftest
from analysis.saxs.status import FrameStatus
from analysis.saxs.writer import AgipdSaxsWriter, IncompleteRun

__all__ = ["REDUCERS", "default_pool", "run_agipd_saxs"]

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
) -> Any:
    """Integrate every frame of a run and return pooled per-train ``I(q)``.

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

    :raises IncompleteRun: some frame is not ``OK`` and ``cfg.allow_incomplete``
        is False. The output file is still written and closed first, so the
        ledger explains what happened.
    """
    # Checked before any work, so a typo costs nothing rather than surfacing
    # after the whole run has been integrated.
    if reduce not in REDUCERS:
        raise ValueError(f"reduce must be one of {REDUCERS}, got {reduce!r}")

    # Before any pool exists, so spawned children inherit it.
    worker_module.set_thread_env()
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

    with AgipdSaxsWriter.open_or_create(cfg, plan, output_path) as out:
        out.store_operator(op)
        out.store_masks(base_masks, sampled)
        todo = [b for b in plan.blocks if not out.block_complete(b)]
        log.info("%d of %d blocks to process", len(todo), len(plan.blocks))

        timings: dict[str, float] = {}
        write_s = 0.0
        bits_present = 0
        unseen_cells = 0

        if todo:
            factory = pool_factory or default_pool
            try:
                with factory(
                    cfg.workers,
                    initializer=worker_module.init,
                    initargs=(paths, cfg),
                ) as pool:
                    futures = {
                        pool.submit(worker_module.process_block, block): block
                        for block in todo
                    }
                    for future in as_completed(futures):
                        block = futures[future]
                        try:
                            block_result = future.result()
                        except BrokenExecutor:
                            out.mark_remaining(FrameStatus.NOT_PROCESSED)
                            raise
                        except Exception as error:  # noqa: BLE001 - into the ledger
                            log.exception("block %d failed", block.index)
                            out.mark(block, FrameStatus.WORKER_ERROR, repr(error))
                            continue
                        write_started = time.perf_counter()
                        out.write_block(block, block_result)
                        write_s += time.perf_counter() - write_started
                        bits_present |= block_result.bits_present
                        unseen_cells += block_result.unseen_cells
                        for key, value in block_result.timings.items():
                            timings[key] = timings.get(key, 0.0) + value
            except BrokenExecutor:
                out.mark_remaining(FrameStatus.NOT_PROCESSED)
                raise

        out.finalise(
            {
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "n_workers": cfg.workers,
                # A run with fewer blocks than workers cannot use them all, so
                # the block count is what any efficiency figure divides by.
                "n_blocks": len(plan.blocks),
                "started_at": started_at,
                "wall_s": time.perf_counter() - started,
                "package_versions": package_versions(),
                "operator_sha256": op.sha256,
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
                "input_file_sha256": {
                    name: (file_sha256(path) if path else None)
                    for name, path in cfg.input_files.items()
                },
                "bits_present": int(bits_present),
                "unseen_cell_frames": int(unseen_cells),
                "timings": timings,
                "setup_timings": {**setup, "write_blocks": write_s},
                "status_summary": out.status_summary(),
                "run_checks": plan.checks,
            }
        )
        incomplete = out.any_not_ok()
        summary = out.status_summary()
        reduced = None
        if reduce == "pooled":
            reduced = out.pooled_per_train()
        elif reduce == "per_pulse":
            reduced = out.per_pulse()

    if incomplete and not cfg.allow_incomplete:
        raise IncompleteRun(f"not every frame reached OK: {summary}")
    return reduced


def _build_base_masks(
    cfg: AgipdSaxsConfig, plan: RunPlan, op: Any, static: Any, dc: Any
) -> tuple[Any, np.ndarray]:
    """Sample trains evenly over the run and majority-vote their masks."""
    from extra_data import by_id

    ok_trains = np.array(
        [t.train_id for t in plan.trains if t.status is FrameStatus.OK],
        dtype=np.uint64,
    )
    sampled = masks_module.evenly_spaced(ok_trains, cfg.base_mask_trains)

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
    chosen = masks_module.evenly_spaced(np.array(ok_trains, dtype=np.uint64), 2)

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
                    masks_module.frame_bad(
                        mask[:, frame], cfg.mask_bits, base_masks.static_bad
                    ),
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
