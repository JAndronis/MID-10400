"""Orchestration of one run: extract the hit trains once, then merge them into sums.

Two stages, two files, each behind its own lock. The first reads proc and fills
the photon store with whichever hit trains it lacks; the second reads only the
store. A run whose results are already complete is answered from the file with
nothing written (CLAUDE.md pitfall 26), and a larger hit set later extracts only
its new trains.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from analysis.common.cpu import file_sha256, phase, set_thread_env
from analysis.common.lock import OutputLock, output_lock
from analysis.common.plan import (
    Block,
    RunPlan,
    TrainRecord,
    build_blocks,
    evenly_spaced,
)
from analysis.common.run import base_provenance, fan_out
from analysis.common.status import UNFINISHED, FrameStatus
from analysis.common.writer import IncompleteRun
from analysis.xpcs import extract as extract_module
from analysis.xpcs import merge as merge_module
from analysis.xpcs.config import XpcsConfig
from analysis.xpcs.hits import HitSet
from analysis.xpcs.kernel import partners
from analysis.xpcs.result import Prelude, ResultWriter, result_reasons
from analysis.xpcs.rings import (
    Envelope,
    Rings,
    build_envelope,
    build_rings,
    pixel_map,
    static_bad_for,
)
from analysis.xpcs.store import ABSENT, FlagBase, RingPhotonStore

__all__ = ["build_prelude", "run_xpcs_hits"]

log = logging.getLogger(__name__)


def run_xpcs_hits(
    cfg: XpcsConfig,
    hits: HitSet,
    *,
    dc: Any = None,
    geometry: Any = None,
    static_bad: np.ndarray | None = None,
    run_dir: Path | None = None,
    store_path: Path | None = None,
    output_path: Path | None = None,
    pool_factory: Callable[..., Any] | None = None,
) -> Path:
    """Extract the hit trains the store lacks, then write their sums.

    :param hits: the trains to correlate. Every one must be a train of the run.
    :param dc: an already-open proc ``DataCollection``; ``None`` opens the run.
    :param geometry: an AGIPD geometry object instead of ``cfg.saxs.geometry_file``.
    :param static_bad: the static mask instead of ``build_static_bad`` (tests).
    :param run_dir: the directory the workers open the run from.
    :param pool_factory: builds the executor; the tests inject an inline one.
    :returns: the results file, which :func:`analysis.xpcs.read` opens.
    :raises ValueError: a hit that is not a train of the run.
    :raises IncompleteRun: some hit train could not be extracted and
        ``cfg.allow_incomplete`` is False. The store keeps what was extracted.
    :raises analysis.common.lock.OutputBusy: another process is writing a file.
    :raises analysis.xpcs.store.StoreMismatch: the store was extracted under
        another hash. It is never replaced.
    """
    set_thread_env()
    started_at, started = time.time(), time.perf_counter()
    setup: dict[str, float] = {}
    store_path = Path(store_path) if store_path else cfg.store_file
    output_path = Path(output_path) if output_path else cfg.output_file

    with phase(setup, "rings"):
        pixels = pixel_map(cfg, geometry)
        envelope = build_envelope(pixels, cfg)
        if static_bad is None:
            static_bad = static_bad_for(cfg)
        rings = build_rings(pixels, envelope, static_bad, cfg)
    extract_hash = cfg.extract_hash(envelope.sha256)
    merge_hash = cfg.merge_hash(envelope.sha256, rings.sha256)

    if not cfg.overwrite and (verdict := result_reasons(output_path, hits, merge_hash)):
        log.info("r%d: %s; nothing written", cfg.run, verdict)
        return output_path

    with ExitStack() as held:
        locks = {
            p: held.enter_context(output_lock(p)) for p in (store_path, output_path)
        }
        if not cfg.overwrite and (
            verdict := result_reasons(output_path, hits, merge_hash)
        ):
            log.info("r%d: %s; nothing written", cfg.run, verdict)
            return output_path
        with phase(setup, "plan"):
            plan = _plan(cfg, hits, dc)
        extracted = _extract(
            cfg,
            hits,
            plan,
            envelope,
            extract_hash,
            store_path,
            pixels.operator_sha256,
            dc=dc,
            run_dir=run_dir,
            pool_factory=pool_factory,
            setup=setup,
            locks=locks,
            started_at=started_at,
            started=started,
        )
        if extracted.unfinished.size and not cfg.allow_incomplete:
            raise IncompleteRun(
                f"r{cfg.run}: {extracted.unfinished.size} of {len(hits)} hit trains "
                f"were not extracted (first: {extracted.unfinished[:5].tolist()}); "
                f"the store keeps the rest. Status: {extracted.summary}"
            )
        _merge(
            cfg,
            hits,
            rings,
            envelope,
            merge_hash,
            extract_hash,
            store_path,
            output_path,
            pixels.operator_sha256,
            plan=plan,
            pool_factory=pool_factory,
            setup=setup,
            locks=locks,
            started_at=started_at,
            started=started,
        )
    return output_path


# ── stage one: proc to the photon store ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Extracted:
    unfinished: np.ndarray
    summary: dict[str, int]


class _StoreSink:
    """The ``out`` :func:`fan_out` writes through for the extraction."""

    def __init__(self, store: RingPhotonStore) -> None:
        self.store = store

    def write_block(self, block: Block, result: Any) -> None:
        for photons in result.photons:
            self.store.put(photons)

    def mark(self, block: Block, status: FrameStatus, message: str = "") -> None:
        for train_id in block.train_ids:
            self.store.mark(train_id, status, message)

    def mark_remaining(self, status: FrameStatus) -> None:
        """Nothing to stamp: a train with no group is already unfinished."""

    def status_summary(self) -> dict[str, int]:
        return self.store.status_summary()


def _plan(cfg: XpcsConfig, hits: HitSet, dc: Any) -> RunPlan:
    from analysis.saxs.plan import build_plan

    plan = build_plan(cfg.saxs, dc=dc)
    foreign = hits.train_ids[~np.isin(hits.train_ids, plan.train_ids)]
    if foreign.size:
        raise ValueError(
            f"{foreign.size} hit train(s) are not trains of r{cfg.run} "
            f"(first: {foreign[:5].tolist()})"
        )
    return plan


def _extract(
    cfg: XpcsConfig,
    hits: HitSet,
    plan: RunPlan,
    envelope: Envelope,
    extract_hash: str,
    store_path: Path,
    operator_sha256: str | None,
    *,
    dc: Any,
    run_dir: Path | None,
    pool_factory: Callable[..., Any] | None,
    setup: dict[str, float],
    locks: dict[Path, OutputLock],
    started_at: float,
    started: float,
) -> _Extracted:
    if len(hits) == 0:
        return _Extracted(np.zeros(0, dtype=np.uint64), {})
    records = {t.train_id: t for t in plan.trains}
    wanted = [records[int(t)] for t in hits.train_ids]
    usable = [r for r in wanted if r.status is FrameStatus.OK]

    if store_path.exists():
        # A pass with nothing to extract reads the store and writes nothing, so a
        # change to how sums are made leaves the store's bytes alone.
        with RingPhotonStore.open(store_path, extract_hash) as probe:
            if probe.unfinished(hits.train_ids).size == 0:
                return _Extracted(np.zeros(0, dtype=np.uint64), probe.status_summary())
        store = RingPhotonStore.open(store_path, extract_hash, writable=True)
    else:
        with phase(setup, "flag_base"):
            base = FlagBase.empty()
            if usable:
                detector = extract_module.open_detector(cfg, dc, run_dir)
                sample = evenly_spaced(
                    np.array([r.train_id for r in usable], dtype=np.uint64),
                    cfg.flag_base_trains,
                )
                base = extract_module.sample_flag_base(
                    cfg, detector, sample, envelope.pixels
                )
        store = RingPhotonStore.create(
            store_path,
            extract_hash,
            envelope.pixels,
            base,
            {"flag_base_trains": [int(t) for t in sample] if usable else []},
        )

    with store:
        for record in wanted:
            if record.status is not FrameStatus.OK:
                store.mark(record.train_id, record.status)
        todo_ids = set(store.unfinished(hits.train_ids).tolist())
        todo = [r for r in usable if r.train_id in todo_ids]
        blocks = build_blocks(todo, cfg.trains_per_block)
        log.info(
            "r%d: %d of %d hit trains to extract in %d blocks",
            cfg.run,
            len(todo),
            len(hits),
            len(blocks),
        )
        totals = fan_out(
            _StoreSink(store),
            blocks,
            process=extract_module.process_block,
            n_workers=cfg.workers,
            initializer=extract_module.init,
            initargs=(
                cfg,
                envelope.pixels,
                store.base,
                str(run_dir) if run_dir else None,
            ),
            pool_factory=pool_factory,
        )
        left = store.unfinished(hits.train_ids)
        summary = store.status_summary()
        store.record_pass(
            base_provenance(
                cfg,
                SimpleNamespace(blocks=blocks),
                _StoreSink(store),
                started_at=started_at,
                started=started,
                totals=totals,
                setup_timings=setup,
                operator_sha256=operator_sha256 or "",
                input_file_sha256=_inputs(cfg),
                run_checks=plan.checks,
            )
            | {"hit_set_sha256": hits.sha256, "n_extracted": len(todo)}
            | locks[store_path].provenance()
        )
    return _Extracted(left, summary)


def _result_payload(cfg: XpcsConfig) -> dict[str, Any]:
    """The config without the fields that only change how the pass ran."""
    return {
        key: value
        for key, value in cfg.payload().items()
        if key not in cfg.operational_fields
    }


def _inputs(cfg: XpcsConfig) -> dict[str, str | None]:
    files = dict(cfg.saxs.input_files)
    files["ring_masks_file"] = cfg.ring_masks_file
    return {k: (file_sha256(v) if v else None) for k, v in files.items()}


# ── stage two: the store to sums ──────────────────────────────────────────────


def build_prelude(store: RingPhotonStore, hits: HitSet, cfg: XpcsConfig) -> Prelude:
    """Decide, from the store alone, each hit's status, partner and pulse pattern.

    The trains that did read share one pulse pattern, the commonest. A train
    with another frame count is ``DATA_CHECK_FAILED`` and one with the same count
    but other pulse ids ``LABEL_MISMATCH``: neither can be correlated lag for lag
    with the rest. A train left unfinished is ``WORKER_ERROR``, so it is retried.
    """
    ids = hits.train_ids
    status = store.status(ids)
    status[np.isin(status, [int(code) for code in UNFINISHED] + [ABSENT])] = (
        FrameStatus.WORKER_ERROR
    )
    ok = np.flatnonzero(status == FrameStatus.OK)
    patterns = {int(i): store.pulse_ids(int(ids[i])) for i in ok}
    counts = Counter(p.tobytes() for p in patterns.values())
    pattern = (
        np.frombuffer(counts.most_common(1)[0][0], dtype=np.uint32)
        if counts
        else np.zeros(0, dtype=np.uint32)
    )
    for i, own in patterns.items():
        if own.size != pattern.size:
            status[i] = FrameStatus.DATA_CHECK_FAILED
        elif not np.array_equal(own, pattern):
            status[i] = FrameStatus.LABEL_MISMATCH
    sequence = np.full(ids.size, -1, dtype=np.int32)
    for i in np.flatnonzero(status == FrameStatus.OK):
        sequence[i] = store.sequence(int(ids[i]))
    pairing = partners(
        ids, status == FrameStatus.OK, sequence, cfg.kernel, cfg.max_partner_gap
    )
    return Prelude(status, pairing, sequence, pattern.copy())


def _merge(
    cfg: XpcsConfig,
    hits: HitSet,
    rings: Rings,
    envelope: Envelope,
    merge_hash: str,
    extract_hash: str,
    store_path: Path,
    output_path: Path,
    operator_sha256: str | None,
    *,
    plan: RunPlan,
    pool_factory: Callable[..., Any] | None,
    setup: dict[str, float],
    locks: dict[Path, OutputLock],
    started_at: float,
    started: float,
) -> None:
    if len(hits) == 0:
        store = None
        ids = hits.train_ids
        prelude = Prelude(
            np.zeros(0, dtype=np.uint8),
            partners(ids, np.zeros(0, bool), np.zeros(0, int), cfg.kernel, 1),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.uint32),
        )
    else:
        with RingPhotonStore.open(store_path, extract_hash) as store:
            prelude = build_prelude(store, hits, cfg)

    attrs = {
        "kernel": cfg.kernel.value,
        "mask_policy": cfg.mask_policy.value,
        "max_partner_gap": cfg.max_partner_gap,
        "envelope_sha256": envelope.sha256,
        "config": json.dumps(_result_payload(cfg), sort_keys=True),
    }
    with ResultWriter.open_or_create(
        output_path,
        hits=hits,
        prelude=prelude,
        rings=rings,
        merge_hash=merge_hash,
        extract_hash=extract_hash,
        attrs=attrs,
        overwrite=cfg.overwrite,
    ) as out:
        ok_rows = np.flatnonzero(prelude.status == FrameStatus.OK)
        records = [
            TrainRecord(int(hits.train_ids[i]), 1, int(i), FrameStatus.OK)
            for i in ok_rows
        ]
        blocks = [
            b
            for b in build_blocks(records, cfg.trains_per_block)
            if not out.block_complete(b)
        ]
        partner_of = {
            int(t): int(p)
            for t, p in zip(hits.train_ids, prelude.pairing.partner, strict=True)
            if p
        }
        try:
            totals = fan_out(
                out,
                blocks,
                process=merge_module.process_block,
                n_workers=cfg.workers,
                initializer=merge_module.init,
                initargs=(
                    str(store_path),
                    extract_hash,
                    rings,
                    cfg.kernel,
                    cfg.mask_policy,
                    partner_of,
                ),
                pool_factory=pool_factory,
            )
        finally:
            merge_module.close()
        out.finalise(
            base_provenance(
                cfg,
                SimpleNamespace(blocks=blocks),
                out,
                started_at=started_at,
                started=started,
                totals=totals,
                setup_timings=setup,
                operator_sha256=operator_sha256 or "",
                input_file_sha256=_inputs(cfg),
                run_checks=plan.checks,
            )
            | {
                "hit_set_sha256": hits.sha256,
                "extract_hash": extract_hash,
                "config_operational": json.dumps(
                    {
                        k: v
                        for k, v in cfg.payload().items()
                        if k in cfg.operational_fields
                    }
                ),
            }
            | locks[output_path].provenance()
        )
        unfinished, summary = out.unfinished(), out.status_summary()
    if unfinished and not cfg.allow_incomplete:
        raise IncompleteRun(
            f"r{cfg.run}: {unfinished} hit train(s) could not be summed; a rerun "
            f"retries them (status {summary})"
        )
