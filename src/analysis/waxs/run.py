"""Orchestration for one run and one detector (WAXS context file §3 D2).

The parent builds the geometry, measures the cell pattern and the readout
noise, gates on the self-test, then fans blocks out to spawned single-threaded
workers and writes their results itself. Nothing in the hot loop depends on the
XGM, transmission or background.

The two detectors never meet here: each gets its own config, its own pass and
its own file, and they are combined only at the plot (§6 O5).
"""

from __future__ import annotations

import json
import logging
import platform
import socket
import time
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from analysis.common.cpu import (
    default_pool,
    file_sha256,
    package_versions,
    phase,
    set_thread_env,
)
from analysis.common.masks import frame_bad
from analysis.common.plan import evenly_spaced
from analysis.common.status import FrameStatus
from analysis.waxs import worker as worker_module
from analysis.waxs.cells import CellAccumulator, CellClassification
from analysis.waxs.config import CELLS_PER_TRAIN, JungfrauWaxsConfig
from analysis.waxs.integrate import ErrorModel
from analysis.waxs.operator import WaxsOperator, build_operator
from analysis.waxs.plan import build_plan, open_detector
from analysis.waxs.selftest import run_selftest
from analysis.waxs.writer import IncompleteRun, JungfrauWaxsWriter

__all__ = ["REDUCERS", "run_jungfrau_waxs"]

#: What ``run_jungfrau_waxs`` may return. ``per_cell`` is the (trainId, cellId,
#: q) grid the DAMNIT variable stores; ``pooled`` is the per-train I(q);
#: ``none`` skips the reduction for a caller that only wants the file written.
REDUCERS = ("pooled", "per_cell", "none")

log = logging.getLogger(__name__)


def _no_lit_cells(cfg: JungfrauWaxsConfig, classification: CellClassification) -> None:
    """Log why a run produced nothing, and return nothing.

    Every run goes down the same path: measure the cells, integrate the lit
    ones. When none are lit there are no frames, so there is no I(q) — that is
    the honest result for a run that saw no beam, not a failure, and 43 of the
    proposal's runs are like that. Raising here would make a whole DAMNIT
    reprocess fall over on runs that are simply dark, so it logs the evidence
    and returns ``None`` instead.
    """
    brightest = max(
        zip(classification.cells, classification.lit_fraction, strict=True),
        key=lambda pair: pair[1],
    )
    log.warning(
        "run %d %s: no memory cell is lit, so there is nothing to integrate and "
        "no I(q) to return. The brightest cell is %d at a lit fraction of %.3e, "
        "against a floor of %.3e; every cell is below it. This is what a run "
        "with no beam on the sample looks like.",
        cfg.run,
        cfg.detector,
        int(brightest[0]),
        float(brightest[1]),
        cfg.lit_fraction_min,
    )
    return None


def run_jungfrau_waxs(
    cfg: JungfrauWaxsConfig,
    *,
    dc: Any = None,
    poni_file: str | Path | None = None,
    run_dir: Path | None = None,
    output_path: Path | None = None,
    pool_factory: Callable[..., Any] | None = None,
    reduce: str = "per_cell",
) -> Any:
    """Integrate every lit frame of one detector's run.

    :param dc: an already-open ``DataCollection``; ``None`` opens the proc run.
    :param poni_file: overrides ``cfg.poni_file``, for tests with a synthetic
        geometry.
    :param run_dir: directory the workers open the run from, instead of
        resolving ``cfg.proposal``/``cfg.run``.
    :param pool_factory: builds the executor. Defaults to
        :func:`analysis.common.cpu.default_pool`; injecting it lets the
        broken-pool and worker-error paths be exercised without killing real
        processes or putting test hooks in ``worker``.
    :param reduce: which reduction to return, one of :data:`REDUCERS`. The
        output file is identical either way.

    :returns: the reduction named by ``reduce``, or ``None`` when the run has no
        lit memory cell at all — see :func:`_no_lit_cells`.
    :raises analysis.waxs.cells.ImplausibleLitCells: the measured cell pattern
        is not a storage-cell sequence, so the classification is wrong (§3 D4).
    :raises analysis.waxs.cells.UnexpectedLitCells: ``cfg.expected_lit_cells``
        is pinned and the measured pattern is not it (§3 D4).
    :raises analysis.waxs.selftest.SelfTestFailed: the NaN path and the
        per-frame-mask reference disagree.
    :raises IncompleteRun: some frame carries a *blocking* status and
        ``cfg.allow_incomplete`` is False. ``DATA_CHECK_FAILED`` is not one by
        default — see :func:`_blocking` and ``cfg.allow_data_check_failures`` —
        but it is logged and recorded either way. The output file is written and
        closed first whatever happens, so the ledger explains it.
    """
    if reduce not in REDUCERS:
        raise ValueError(f"reduce must be one of {REDUCERS}, got {reduce!r}")

    # Before any pool exists, so spawned children inherit it (§3 rule 3).
    set_thread_env()
    started_at = time.time()
    started = time.perf_counter()

    opened_here = dc is None
    if opened_here:
        from extra_data import open_run

        dc = open_run(cfg.proposal, cfg.run, data="proc")

    setup: dict[str, float] = {}
    with phase(setup, "operator"):
        op, ai = build_operator(cfg, poni_file)
    with phase(setup, "detector"):
        det = open_detector(cfg, dc)
    with phase(setup, "cells"):
        classification, sampled = _classify_cells(cfg, op, det)
        classification.check_structure(CELLS_PER_TRAIN)
        if cfg.expected_lit_cells is not None:
            classification.check_expected(cfg.expected_lit_cells)
        if not classification.lit:
            return _no_lit_cells(cfg, classification)
        model = _error_model(cfg, classification)
    with phase(setup, "plan"):
        plan = build_plan(
            cfg,
            classification.lit,
            dc=dc,
            control_dc=None if opened_here else dc,
            det=det,
        )
    with phase(setup, "selftest"):
        report = _run_selftest(cfg, op, ai, model, classification, det, plan)

    with JungfrauWaxsWriter.open_or_create(cfg, plan, output_path) as out:
        out.store_operator(op)
        out.store_cells(classification, model, sampled)
        todo = [b for b in plan.blocks if not out.block_complete(b)]
        log.info("%d of %d blocks to process", len(todo), len(plan.blocks))

        timings: dict[str, float] = {}
        write_s = 0.0
        bits_present = 0

        if todo:
            factory = pool_factory or default_pool
            initargs = (
                cfg,
                op.sha256,
                model,
                classification.lit,
                str(run_dir) if run_dir else None,
            )
            try:
                with factory(
                    cfg.workers,
                    initializer=worker_module.init,
                    initargs=initargs,
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
                        for key, value in block_result.timings.items():
                            timings[key] = timings.get(key, 0.0) + value
            except BrokenExecutor:
                out.mark_remaining(FrameStatus.NOT_PROCESSED)
                raise

        out.finalise(
            {
                "detector": cfg.detector,
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "n_workers": cfg.workers,
                "n_blocks": len(plan.blocks),
                "started_at": started_at,
                "wall_s": time.perf_counter() - started,
                "package_versions": package_versions(),
                "operator_sha256": op.sha256,
                "poni_sha256": op.poni_sha256 or "",
                "static_mask_sha256": op.static_sha256,
                "q_range": list(op.q_populated),
                "input_file_sha256": {
                    name: (file_sha256(path) if path else None)
                    for name, path in cfg.input_files.items()
                },
                "bits_present": int(bits_present),
                "error_model": {
                    "read_noise_kev": model.read_noise_kev,
                    "photon_energy_kev": model.photon_energy_kev,
                    "source": model.source,
                    "n_samples": model.n_samples,
                    "sha256": model.sha256,
                    "note": (
                        "Var = sigma_read^2 + E*x in keV^2, stored unclamped; "
                        "per-frame negative bins are counted in "
                        "frames/n_negative_variance_bins"
                    ),
                },
                "cells": classification.summary(),
                "selftest": {
                    "n_frames": report.n_frames,
                    "max_rel_signal": report.max_rel_signal,
                    "max_rel_normalization": report.max_rel_normalization,
                    "max_rel_variance": report.max_rel_variance,
                    "tolerance": report.tolerance,
                },
                "timings": timings,
                "setup_timings": {**setup, "write_blocks": write_s},
                "status_summary": out.status_summary(),
                "data_check": out.data_check_summary(),
                "run_checks": plan.checks,
            }
        )
        summary = out.status_summary()
        data_check = out.data_check_summary()
        blocking = _blocking(cfg, summary)
        reduced = None
        if reduce == "pooled":
            reduced = out.pooled_per_train()
        elif reduce == "per_cell":
            reduced = out.per_cell()

    if data_check:
        # Never silent, whether or not it raises: a tolerated loss is still a
        # loss, and under §3 D6's unresolved question it could be a biased one.
        log.warning(
            "r%d %s: %d frames failed the value check and are absent from the "
            "result (%s)",
            cfg.run,
            cfg.detector,
            data_check["n"],
            json.dumps(data_check, sort_keys=True),
        )
    if blocking and not cfg.allow_incomplete:
        detail = f" data check: {data_check}" if data_check else ""
        raise IncompleteRun(
            f"not every frame reached OK: {summary}; blocking {blocking}.{detail}"
        )
    return reduced


def _blocking(cfg: JungfrauWaxsConfig, summary: dict[str, int]) -> dict[str, int]:
    """The non-``OK`` statuses that stop the run finishing.

    ``DATA_CHECK_FAILED`` is a property of the data rather than a failure of the
    pass — the frame was read, classified and recorded — so by default it is
    counted and not raised on; see ``cfg.allow_data_check_failures``. Everything
    else still blocks.
    """
    tolerated = {FrameStatus.OK.name}
    if cfg.allow_data_check_failures:
        tolerated.add(FrameStatus.DATA_CHECK_FAILED.name)
    return {name: n for name, n in summary.items() if name not in tolerated}


def _sample_trains(cfg: JungfrauWaxsConfig, det: Any) -> np.ndarray:
    train_ids = np.asarray(det.train_ids, dtype=np.uint64)
    if train_ids.size == 0:
        raise ValueError("the detector has no trains in this run")
    return evenly_spaced(train_ids, cfg.cell_sample_trains)


def _classify_cells(
    cfg: JungfrauWaxsConfig, op: WaxsOperator, det: Any
) -> tuple[CellClassification, np.ndarray]:
    """Measure the lit/dark split and the readout noise over sampled trains."""
    from extra_data import by_id

    sampled = _sample_trains(cfg, det)
    accumulator = CellAccumulator(cfg, op.static_bad)
    for train_id in sampled.tolist():
        selected = det.select_trains(by_id[[train_id]])
        data = np.asarray(selected["data.adc"].ndarray())[0, 0]
        mask = np.asarray(selected["data.mask"].ndarray())[0, 0]
        cells = np.asarray(selected["data.memoryCell"].ndarray()).reshape(-1)
        accumulator.update(data, mask, cells)
    return accumulator.finalise(), sampled


def _error_model(
    cfg: JungfrauWaxsConfig, classification: CellClassification
) -> ErrorModel:
    """sigma_read, in order of precedence: configured, measured, fallback.

    An explicit ``cfg.read_noise_kev`` wins outright. Otherwise the run's own
    dark cells are measured, which is what §3 D3 asks for. A run that reads
    every storage cell has no dark cell — 213 of the proposal's do — and falls
    back to ``cfg.read_noise_fallback_kev``, the per-detector median over the
    runs that do. ``ErrorModel.source`` records which of the three it was, so a
    stored file never has to be guessed at.

    **There is deliberately no fourth branch that raises.** An earlier draft had
    one, for "no measurement and no fallback" — but `JungfrauWaxsConfig` resolves
    ``read_noise_fallback_kev`` from
    :data:`~analysis.waxs.config.DEFAULT_READ_NOISE_KEV` in ``__post_init__``, so
    it is never ``None`` by the time a config exists and that branch could not
    fire. Resolving it there rather than here is what keeps
    the *used* number inside ``config_hash`` (CLAUDE.md pitfall 12): were the
    field left ``None`` and filled in at this point, a change to the constant
    would silently let resume merge blocks computed at two different sigma_read.

    The guarantee CLAUDE.md working rule 2 asks for — never a number without a
    traceable source — is met by ``ErrorModel.source`` instead of by an
    exception: every stored file says which of the three it used, so selecting
    against ``"fallback"`` is a question the provenance can answer.
    """
    if cfg.read_noise_kev is not None:
        return ErrorModel(
            cfg.read_noise_kev,
            cfg.photon_energy_kev,
            "configured",
            0,
            classification.dark,
        )
    if classification.read_noise_kev is not None:
        return ErrorModel(
            classification.read_noise_kev,
            cfg.photon_energy_kev,
            "measured",
            classification.read_noise_samples,
            classification.dark,
        )
    log.info(
        "run %d %s: all %d cells are lit, so sigma_read falls back to %.4f keV",
        cfg.run,
        cfg.detector,
        len(classification.cells),
        cfg.read_noise_fallback_kev,
    )
    return ErrorModel(
        cfg.read_noise_fallback_kev,
        cfg.photon_energy_kev,
        "fallback",
        0,
        classification.dark,
    )


def _run_selftest(
    cfg: JungfrauWaxsConfig,
    op: WaxsOperator,
    ai: Any,
    model: ErrorModel,
    classification: CellClassification,
    det: Any,
    plan: Any,
) -> Any:
    """Gate the NaN path against the per-frame-mask reference on real frames."""
    from extra_data import by_id

    ok_trains = plan.ok_train_ids()
    if ok_trains.size == 0:
        raise ValueError("no OK trains to self-test on")
    chosen = evenly_spaced(ok_trains, 2)

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for train_id in chosen.tolist():
        selected = det.select_trains(by_id[[int(train_id)]])
        data = np.asarray(selected["data.adc"].ndarray())[0, 0]
        mask = np.asarray(selected["data.mask"].ndarray())[0, 0]
        cells = np.asarray(selected["data.memoryCell"].ndarray()).reshape(-1)
        positions = {int(c): i for i, c in enumerate(cells.tolist())}
        for cell in classification.lit:
            if cell not in positions:
                continue
            position = positions[cell]
            frames.append(
                (
                    data[position],
                    frame_bad(mask[position], cfg.mask_bits, op.static_bad),
                )
            )
            if len(frames) >= cfg.selftest_frames:
                break
        if len(frames) >= cfg.selftest_frames:
            break

    report = run_selftest(ai, op, model, frames, max_abs_kev=cfg.max_abs_kev)
    log.info(
        "self-test passed on %d frames: max rel S %.2e, N %.2e, V %.2e",
        report.n_frames,
        report.max_rel_signal,
        report.max_rel_normalization,
        report.max_rel_variance,
    )
    return report
