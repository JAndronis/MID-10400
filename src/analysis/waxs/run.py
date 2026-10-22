"""Orchestration for one run and one detector.

The parent builds the geometry, measures the cell pattern and the readout
noise, gates on the self-test, then fans blocks out to spawned single-threaded
workers and writes their results itself. The workers count photons per ASIC
before integrating (:mod:`analysis.waxs.photons`), so the file stores photons.
Nothing here depends on the XGM, transmission or background, and each detector
gets its own config, pass and file.

A run whose file is already complete is answered from it with nothing written,
so a relaunched job cannot truncate a finished file. Otherwise the pass holds a
lock on the file for as long as it writes it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.common.cpu import (
    file_sha256,
    phase,
    set_thread_env,
)
from analysis.common.lock import OutputLock, lock_path, output_lock
from analysis.common.plan import evenly_spaced
from analysis.common.run import base_provenance, fan_out
from analysis.common.status import FrameStatus
from analysis.common.writer import IncompleteRun, pooled_per_train, status_counts
from analysis.waxs import worker as worker_module
from analysis.waxs.cells import (
    CellAccumulator,
    CellClassification,
    UnexpectedLitCells,
)
from analysis.waxs.complete import is_complete, operator_differences
from analysis.waxs.config import CELLS_PER_TRAIN, JungfrauWaxsConfig
from analysis.waxs.integrate import ErrorModel
from analysis.waxs.operator import WaxsOperator, build_operator
from analysis.waxs.photons import PhotonCounting
from analysis.waxs.plan import build_plan, open_detector
from analysis.waxs.selftest import run_selftest
from analysis.waxs.writer import JungfrauWaxsWriter, per_cell

__all__ = ["REDUCERS", "run_jungfrau_waxs"]

#: What ``run_jungfrau_waxs`` may return. ``per_cell`` is the (trainId, cellId,
#: q) grid the DAMNIT variable stores; ``pooled`` is the per-train I(q);
#: ``none`` skips the reduction for a caller that only wants the file written.
REDUCERS = ("pooled", "per_cell", "none")

log = logging.getLogger(__name__)


def _no_lit_cells(cfg: JungfrauWaxsConfig, classification: CellClassification) -> None:
    """Log why a run produced nothing, and return nothing.

    When no cell is lit there are no frames and so no I(q), which is the result
    for a run that saw no beam. Raising would stop a bulk DAMNIT reprocess on
    dark runs, so this logs the evidence and returns ``None``.
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
        :func:`analysis.common.cpu.default_pool`. Injecting it lets tests
        exercise the broken-pool and worker-error paths without real processes.
    :param reduce: which reduction to return, one of :data:`REDUCERS`. The
        output file is identical either way.

    :returns: the reduction named by ``reduce``, or ``None`` when the run has no
        lit memory cell (:func:`_no_lit_cells`).
    :raises analysis.waxs.cells.ImplausibleLitCells: the measured cell pattern
        is not a storage-cell sequence, so the classification is wrong.
    :raises analysis.waxs.cells.UnexpectedLitCells: ``cfg.expected_lit_cells``
        is pinned and the measured pattern is not it.
    :raises analysis.waxs.selftest.SelfTestFailed: the NaN path and the
        per-frame-mask reference disagree.
    :raises IncompleteRun: some frame carries a blocking status (:func:`_blocking`)
        and ``cfg.allow_incomplete`` is False. ``DATA_CHECK_FAILED`` is not
        blocking by default, but it is logged and recorded either way. The output
        file is written and closed first, so its frame table explains the raise.
    :raises analysis.common.lock.OutputBusy: another process is writing the
        file.
    :raises analysis.common.writer.OperatorMismatch: the file was integrated by
        another operator or pyFAI, so it cannot be resumed.

    A file that is already complete (:func:`analysis.waxs.complete.is_complete`)
    is answered from with nothing written, unless ``cfg.overwrite``. It raises
    on the same terms as a full pass.
    """
    if reduce not in REDUCERS:
        raise ValueError(f"reduce must be one of {REDUCERS}, got {reduce!r}")

    # Before any pool exists, so spawned children inherit it.
    set_thread_env()
    started_at = time.time()
    started = time.perf_counter()
    output_path = Path(output_path) if output_path is not None else cfg.output_file

    # The operator comes first because it reads only the PONI and the static
    # mask, and the completeness check compares against it.
    setup: dict[str, float] = {}
    with phase(setup, "operator"):
        op, ai = build_operator(cfg, poni_file)

    # Skipped while a lock exists, since the file may be mid-write; the check
    # under the lock below then decides.
    if not cfg.overwrite and not lock_path(output_path).exists():
        if is_complete(output_path, cfg, operator=op):
            return _from_file(cfg, output_path, reduce)
    with output_lock(output_path) as lock:
        if not cfg.overwrite:
            verdict = is_complete(output_path, cfg, operator=op)
            if verdict:
                return _from_file(cfg, output_path, reduce)
            log.info("%s", verdict)
        # Before the cells and the self-test, so a file that can only be
        # reprocessed is refused at once.
        JungfrauWaxsWriter.check_resumable(
            cfg, output_path, operator_check=partial(operator_differences, op=op)
        )
        return _integrate(
            cfg,
            op,
            ai,
            dc=dc,
            run_dir=run_dir,
            output_path=output_path,
            pool_factory=pool_factory,
            reduce=reduce,
            setup=setup,
            started_at=started_at,
            started=started,
            lock=lock,
        )


def _from_file(cfg: JungfrauWaxsConfig, path: Path, reduce: str) -> Any:
    """Answer a complete file exactly as the pass would have, read-only.

    The pinned lit-cell set and the blocking statuses are checked on the same
    terms as at the end of a pass, so pinning ``expected_lit_cells`` for a
    reprocess still refuses a run whose stored pattern differs.
    """
    with h5py.File(path, "r") as handle:
        summary = status_counts(handle["frames/status"][:], nonzero_only=True)
        lit = tuple(int(cell) for cell in handle["cells/lit"][:])
        data_check = json.loads(handle["provenance"].attrs.get("data_check", "{}"))
        reduced = None
        if reduce == "pooled":
            reduced = pooled_per_train(handle)
        elif reduce == "per_cell":
            reduced = per_cell(handle)
    if cfg.expected_lit_cells is not None and lit != tuple(cfg.expected_lit_cells):
        raise UnexpectedLitCells(
            f"{path} holds cells {list(lit)}, the config expects "
            f"{list(cfg.expected_lit_cells)}"
        )
    log.info(
        "r%d %s: %s is complete; answered from the file, nothing written",
        cfg.run,
        cfg.detector,
        path,
    )
    if data_check:
        log.warning(
            "r%d %s: the stored file records value-check exclusions (%s)",
            cfg.run,
            cfg.detector,
            json.dumps(data_check, sort_keys=True),
        )
    blocking = _blocking(cfg, summary)
    if blocking and not cfg.allow_incomplete:
        raise IncompleteRun(
            f"not every frame reached OK: {summary}; blocking {blocking}."
        )
    return reduced


def _integrate(
    cfg: JungfrauWaxsConfig,
    op: WaxsOperator,
    ai: Any,
    *,
    dc: Any,
    run_dir: Path | None,
    output_path: Path,
    pool_factory: Callable[..., Any] | None,
    reduce: str,
    setup: dict[str, float],
    started_at: float,
    started: float,
    lock: OutputLock,
) -> Any:
    """The pass itself, run under the lock :func:`run_jungfrau_waxs` holds."""
    opened_here = dc is None
    if opened_here:
        from extra_data import open_run

        dc = open_run(cfg.proposal, cfg.run, data="proc")

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
        counting = PhotonCounting.from_config(cfg)
    with phase(setup, "plan"):
        plan = build_plan(
            cfg,
            classification.lit,
            dc=dc,
            control_dc=None if opened_here else dc,
            det=det,
        )
    with phase(setup, "selftest"):
        report = _run_selftest(cfg, op, ai, counting, classification, det, plan)

    with JungfrauWaxsWriter.open_or_create(
        cfg,
        plan,
        output_path,
        operator_check=partial(operator_differences, op=op),
    ) as out:
        out.store_operator(op)
        out.store_cells(classification, model, sampled)
        todo = [b for b in plan.blocks if not out.block_complete(b)]
        log.info("%d of %d blocks to process", len(todo), len(plan.blocks))

        totals = fan_out(
            out,
            todo,
            process=worker_module.process_block,
            n_workers=cfg.workers,
            initializer=worker_module.init,
            initargs=(
                cfg,
                op.sha256,
                classification.lit,
                str(run_dir) if run_dir else None,
            ),
            pool_factory=pool_factory,
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
                "detector": cfg.detector,
                "poni_sha256": op.poni_sha256 or "",
                "static_mask_sha256": op.static_sha256,
                "q_range": list(op.q_populated),
                "units": out.UNITS,
                "photon_counting": counting.summary(),
                "photon_levels": out.level_summary(),
                "error_model": {
                    "variance": "poisson",
                    "read_noise_kev": model.read_noise_kev,
                    "photon_energy_kev": model.photon_energy_kev,
                    "source": model.source,
                    "n_samples": model.n_samples,
                    "sha256": model.sha256,
                    "note": (
                        "variance = photon count per pixel, Sum c^2 n per bin, "
                        "in photons^2. sigma_read is recorded as a diagnostic "
                        "and does not enter it"
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
                "data_check": out.data_check_summary(),
            }
            | lock.provenance()
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
        # Logged whether or not it raises, because a tolerated loss of pixels
        # can still bias the result.
        log.warning(
            "r%d %s: the value check excluded %d pixels across %d frames and "
            "refused %d outright (%s)",
            cfg.run,
            cfg.detector,
            data_check["n_pixels_excluded"],
            data_check["n_frames_with_excluded_pixels"],
            data_check["n_frames_refused"],
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

    ``DATA_CHECK_FAILED`` is a property of the data, so
    ``cfg.allow_data_check_failures`` decides. Everything else blocks.
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
    dark cells are measured. A run that reads every storage cell has no dark
    cell and falls back to ``cfg.read_noise_fallback_kev``.
    ``ErrorModel.source`` records which of the three it was.

    The fallback is never ``None`` here because ``JungfrauWaxsConfig`` resolves
    it in ``__post_init__``, which keeps the value used inside ``config_hash``.
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
    counting: PhotonCounting,
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

    bits = np.uint32(cfg.mask_bits)
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
            frames.append((data[position], (mask[position].reshape(-1) & bits) != 0))
            if len(frames) >= cfg.selftest_frames:
                break
        if len(frames) >= cfg.selftest_frames:
            break

    report = run_selftest(ai, op, counting, frames, max_abs_kev=cfg.max_abs_kev)
    log.info(
        "self-test passed on %d frames: max rel S %.2e, N %.2e, V %.2e",
        report.n_frames,
        report.max_rel_signal,
        report.max_rel_normalization,
        report.max_rel_variance,
    )
    return report
