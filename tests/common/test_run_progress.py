"""``fan_out``'s progress lines: one per tenth of the blocks, on our logger only.

The pool runs blocks inline, all of them at submission, so the clock moves
where the parent collects them instead: the module's clock is a counter that
each write (or failure mark) advances by one second, so every rate and
duration in a line is exact.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from analysis.common import run as run_module
from analysis.common.plan import Block
from analysis.common.status import FrameStatus

FRAMES_PER_BLOCK = 10


class InlinePool:
    """Runs each block as it is submitted, as ``tests/saxs/test_run.py``'s does."""

    def __init__(self, n_workers, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        future: Future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as error:  # noqa: BLE001 - mirrors a real pool
            future.set_exception(error)
        return future


@dataclass
class FakeOut:
    clock: list[float]
    written: list[int] = field(default_factory=list)
    marked: list[tuple[int, FrameStatus]] = field(default_factory=list)

    def write_block(self, block, result):
        self.clock[0] += 1.0
        self.written.append(block.index)

    def mark(self, block, status, reason):
        self.clock[0] += 1.0
        self.marked.append((block.index, status))

    def mark_remaining(self, status):
        raise AssertionError("the pool did not die")


def blocks(n: int) -> list[Block]:
    return [
        Block(
            index=i,
            train_ids=(1000 + i,),
            expected_frames=(FRAMES_PER_BLOCK,),
            first_rows=(i * FRAMES_PER_BLOCK,),
        )
        for i in range(n)
    ]


@pytest.fixture
def clock(monkeypatch):
    """Seconds since the pool was asked for; each block takes one."""
    now = [0.0]
    monkeypatch.setattr(
        run_module, "time", SimpleNamespace(perf_counter=lambda: now[0])
    )
    return now


def run(todo, clock, *, fail=()):
    def process(block):
        if block.index in fail:
            raise RuntimeError(f"block {block.index}")
        return SimpleNamespace(bits_present=0, timings={"integrate": 1.0})

    out = FakeOut(clock)
    run_module.fan_out(
        out,
        todo,
        process=process,
        n_workers=2,
        initializer=lambda: None,
        initargs=(),
        pool_factory=InlinePool,
    )
    return out


def progress_lines(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and "blocks (" in record.getMessage()
    ]


def test_one_line_per_tenth_of_the_blocks(clock, caplog):
    caplog.set_level(logging.INFO, logger="analysis")
    run(blocks(25), clock)
    lines = progress_lines(caplog)
    assert [line.getMessage().split(" blocks")[0] for line in lines] == [
        "3/25", "5/25", "8/25", "10/25", "13/25",
        "15/25", "18/25", "20/25", "23/25", "25/25",
    ]  # fmt: skip


def test_a_line_carries_rate_elapsed_and_remaining(clock, caplog):
    caplog.set_level(logging.INFO, logger="analysis")
    run(blocks(20), clock)
    first, *_, last = (line.getMessage() for line in progress_lines(caplog))
    # two blocks of ten frames in two seconds; eighteen blocks to go
    assert first == (
        "2/20 blocks (10 %), 20/200 frames, 10 frames/s, elapsed 0:00:02, "
        "remaining ~0:00:18"
    )
    assert last == (
        "20/20 blocks (100 %), 200/200 frames, 10 frames/s, elapsed 0:00:20, "
        "remaining ~0:00:00"
    )


def test_fewer_than_ten_blocks_get_a_line_each(clock, caplog):
    caplog.set_level(logging.INFO, logger="analysis")
    run(blocks(3), clock)
    assert [line.getMessage()[:4] for line in progress_lines(caplog)] == [
        "1/3 ",
        "2/3 ",
        "3/3 ",
    ]


def test_a_failed_block_still_counts_towards_progress(clock, caplog):
    caplog.set_level(logging.INFO, logger="analysis")
    out = run(blocks(10), clock, fail={4})
    assert out.marked == [(4, FrameStatus.WORKER_ERROR)]
    assert len(progress_lines(caplog)) == 10
    assert progress_lines(caplog)[-1].getMessage().startswith("10/10 blocks (100 %)")


def test_the_lines_go_to_the_analysis_logger_only(clock, caplog):
    caplog.set_level(logging.INFO, logger="analysis")
    run(blocks(5), clock)
    assert {line.name for line in progress_lines(caplog)} == {"analysis.common.run"}


def test_the_lines_are_info_and_so_silent_at_warning(clock, caplog):
    caplog.set_level(logging.WARNING, logger="analysis")
    run(blocks(5), clock)
    assert progress_lines(caplog) == []


def test_nothing_to_do_logs_nothing(clock, caplog):
    caplog.set_level(logging.INFO, logger="analysis")
    run([], clock)
    assert progress_lines(caplog) == []
