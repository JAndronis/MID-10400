"""Orchestration shared by both passes: the worker fan-out and its provenance.

The parent owns the output file and every write to it. Workers only ever return
a block's results, so a failed block reaches the ledger instead of the file.
"""

from __future__ import annotations

import logging
import platform
import socket
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import BrokenExecutor, as_completed
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from analysis.common.cpu import default_pool, package_versions
from analysis.common.plan import Block
from analysis.common.status import FrameStatus

__all__ = ["FanOutTotals", "base_provenance", "fan_out"]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class FanOutTotals:
    """What a run accumulated across every block it processed."""

    #: Per-stage worker seconds, summed over blocks.
    timings: dict[str, float] = field(default_factory=dict)
    #: Parent-side seconds spent writing blocks into the output file.
    write_s: float = 0.0
    #: Union of the ``BadPixels`` bits any frame carried.
    bits_present: int = 0
    #: Sums of the extra per-block counters a pass asked for.
    extra: dict[str, int] = field(default_factory=dict)


class _Progress:
    """An INFO line each time another tenth of the blocks is done.

    A SLURM log has no terminal for a progress bar, and without these lines a
    pass that is slow and one that has hung look the same until the time
    limit. Fewer than ten blocks get a line each.
    """

    def __init__(self, todo: Sequence[Block]) -> None:
        self.n_blocks = len(todo)
        self.n_frames = sum(block.n_frames for block in todo)
        self.blocks = 0
        self.frames = 0
        self.tenths = 0
        self.started = time.perf_counter()

    def done(self, block: Block) -> None:
        """Count ``block`` as finished, written or failed alike."""
        self.blocks += 1
        self.frames += block.n_frames
        tenths = self.blocks * 10 // self.n_blocks
        if tenths == self.tenths and self.n_blocks >= 10:
            return
        self.tenths = tenths
        elapsed = time.perf_counter() - self.started
        rate = self.frames / elapsed if elapsed > 0 else float("inf")
        remaining = (self.n_frames - self.frames) / rate if rate > 0 else None
        log.info(
            "%d/%d blocks (%.0f %%), %d/%d frames, %.0f frames/s, elapsed %s, "
            "remaining ~%s",
            self.blocks,
            self.n_blocks,
            100 * self.blocks / self.n_blocks,
            self.frames,
            self.n_frames,
            rate,
            _duration(elapsed),
            "?" if remaining is None else _duration(remaining),
        )


def _duration(seconds: float) -> str:
    """``H:MM:SS``, to the second."""
    return str(timedelta(seconds=round(seconds)))


def fan_out(
    out: Any,
    todo: Sequence[Block],
    *,
    process: Callable[[Block], Any],
    n_workers: int,
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    pool_factory: Callable[..., Any] | None = None,
    accumulate: Iterable[str] = (),
) -> FanOutTotals:
    """Process ``todo`` in a worker pool, writing each result as it arrives.

    A block that raises is recorded as ``WORKER_ERROR`` and the run continues,
    because one bad block is not a reason to lose the rest. A pool that dies
    takes the run with it: every unwritten row is marked ``NOT_PROCESSED``
    first, so the ledger never leaves a row silently unaccounted for.

    Progress goes to this module's logger, at INFO, once per tenth of the
    blocks: frames/s, elapsed and remaining time, all from the moment the pool
    is asked for, so worker start-up counts.

    :param out: the open writer; the only thing that touches the output file.
    :param todo: blocks not already complete in that file.
    :param process: the worker entry point, submitted once per block.
    :param n_workers: pool size.
    :param initializer: worker setup, run once per process.
    :param initargs: its arguments, which must pickle.
    :param pool_factory: pool constructor, defaulting to spawned processes.
        Injected by the tests to run blocks inline.
    :param accumulate: extra ``BlockResult`` attributes to sum per pass.
    :returns: the totals, for the provenance record.
    """
    totals = FanOutTotals(extra=dict.fromkeys(accumulate, 0))
    if not todo:
        return totals

    factory = pool_factory or default_pool
    progress = _Progress(todo)
    try:
        with factory(n_workers, initializer=initializer, initargs=initargs) as pool:
            futures = {pool.submit(process, block): block for block in todo}
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
                    progress.done(block)
                    continue
                write_started = time.perf_counter()
                out.write_block(block, block_result)
                totals.write_s += time.perf_counter() - write_started
                totals.bits_present |= block_result.bits_present
                for name in totals.extra:
                    totals.extra[name] += getattr(block_result, name)
                for key, value in block_result.timings.items():
                    totals.timings[key] = totals.timings.get(key, 0.0) + value
                progress.done(block)
    except BrokenExecutor:
        out.mark_remaining(FrameStatus.NOT_PROCESSED)
        raise
    return totals


def base_provenance(
    cfg: Any,
    plan: Any,
    out: Any,
    *,
    started_at: float,
    started: float,
    totals: FanOutTotals,
    setup_timings: dict[str, float],
    operator_sha256: str,
    input_file_sha256: dict[str, str | None],
    run_checks: Any,
) -> dict[str, Any]:
    """The provenance keys both passes record, for a pass to merge its own into.

    :returns: the record. ``wall_s`` is measured here so that it covers the
        whole run rather than the part a caller remembered to time.
    """
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "n_workers": cfg.workers,
        "n_blocks": len(plan.blocks),
        "started_at": started_at,
        "wall_s": time.perf_counter() - started,
        "operator_sha256": operator_sha256,
        "input_file_sha256": input_file_sha256,
        "bits_present": int(totals.bits_present),
        "timings": totals.timings,
        "setup_timings": {**setup_timings, "write_blocks": totals.write_s},
        "status_summary": out.status_summary(),
        "run_checks": run_checks,
    }
