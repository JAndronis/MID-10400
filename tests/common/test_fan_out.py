"""``fan_out``: a failed block reaches the ledger, a dead pool stops the run.

Blocks run inline in this process, through a pool that runs each one as it is
submitted, and land in a real frame table.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import BrokenExecutor, Future
from dataclasses import dataclass
from types import SimpleNamespace

import h5py
import pytest
from tiny_table import OK, PLAN, TinyConfig, TinyWriter, result_for

from analysis.common.plan import Block
from analysis.common.run import fan_out
from analysis.common.status import FrameStatus
from analysis.common.writer import FrameRows


class InlinePool:
    """Runs the initializer once and each block at submission, as a pool would."""

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


class DyingPool(InlinePool):
    """A pool whose worker dies at the second block it is given."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submitted = 0

    def submit(self, fn, *args):
        self.submitted += 1
        if self.submitted < 2:
            return super().submit(fn, *args)
        future: Future = Future()
        future.set_exception(BrokenExecutor("worker died"))
        return future


def process(block, *, fail=()):
    if block.index in fail:
        raise RuntimeError(f"block {block.index} could not be read")
    return CountedRows(
        result_for(block).columns,
        bits_present=1 << block.index,
        timings={"read": 1.0, "integrate": 0.5},
        n_extreme=block.index + 1,
    )


@dataclass(slots=True)
class CountedRows(FrameRows):
    """A block's rows plus one extra counter for ``accumulate`` to sum."""

    n_extreme: int = 0


def run(cfg, todo, *, fail=(), pool=InlinePool, **kwargs):
    with TinyWriter.open_or_create(cfg, PLAN) as out:
        return fan_out(
            out,
            todo,
            process=lambda block: process(block, fail=fail),
            n_workers=2,
            initializer=lambda: None,
            initargs=(),
            pool_factory=pool,
            **kwargs,
        )


@pytest.fixture
def cfg(tmp_path):
    return TinyConfig(output_dir=str(tmp_path))


def status_column(cfg):
    with h5py.File(cfg.output_file) as f:
        return f["frames/status"][:]


def test_a_block_that_raises_is_a_worker_error_and_the_rest_still_finish(cfg):
    run(cfg, PLAN.blocks, fail={1})

    assert status_column(cfg).tolist() == [OK] * 3 + [FrameStatus.WORKER_ERROR] * 5 + [
        OK
    ]
    with h5py.File(cfg.output_file) as f:
        errors = json.loads(f["provenance"].attrs["block_errors"])
    assert list(errors) == ["1"] and "could not be read" in errors["1"]


def test_a_dead_pool_raises_and_leaves_the_unwritten_rows_unprocessed(cfg):
    with pytest.raises(BrokenExecutor):
        run(cfg, PLAN.blocks, pool=DyingPool)

    # The first block may or may not have been collected before the pool died.
    status = status_column(cfg)
    assert set(status[:3]) <= {OK, FrameStatus.NOT_PROCESSED}
    assert (status[3:] == FrameStatus.NOT_PROCESSED).all()


def test_the_totals_add_up_over_the_written_blocks(cfg):
    totals = run(cfg, PLAN.blocks, fail={2}, accumulate=("n_extreme",))

    assert totals.bits_present == 0b11
    assert totals.extra == {"n_extreme": 1 + 2}
    assert totals.timings == {"read": 2.0, "integrate": 1.0}


def test_nothing_to_do_asks_for_no_pool(cfg):
    def no_pool(*args, **kwargs):
        raise AssertionError("a pool was started for no work")

    totals = run(cfg, [], pool=no_pool)
    assert totals.bits_present == 0 and totals.timings == {}


# ── progress ─────────────────────────────────────────────────────────────────
def blocks(n: int) -> list[Block]:
    return [
        Block(index=i, train_ids=(i,), expected_frames=(1,), first_rows=(i,))
        for i in range(n)
    ]


def progress(caplog, n: int, *, fail=()) -> list[tuple[int, int]]:
    """``(done, total)`` blocks of each progress line for a run of ``n`` blocks."""

    class Out:
        def write_block(self, block, result):
            pass

        def mark(self, block, status, message):
            pass

    def work(block):
        if block.index in fail:
            raise RuntimeError("boom")
        return SimpleNamespace(bits_present=0, timings={})

    caplog.set_level(logging.INFO, logger="analysis.common.run")
    fan_out(
        Out(),
        blocks(n),
        process=work,
        n_workers=1,
        initializer=lambda: None,
        initargs=(),
        pool_factory=InlinePool,
    )
    pattern = re.compile(r"(\d+)/(\d+) blocks")
    return [
        tuple(int(x) for x in match.groups())
        for record in caplog.records
        if record.name == "analysis.common.run"
        and record.levelno == logging.INFO
        and (match := pattern.search(record.getMessage()))
    ]


@pytest.mark.parametrize("n", [0, 3, 10, 25])
def test_one_progress_line_per_tenth_of_the_blocks(caplog, n):
    lines = progress(caplog, n)
    assert len(lines) == min(n, 10)
    assert all(total == n for _, total in lines)
    assert [done for done, _ in lines] == sorted({done for done, _ in lines})
    if n:
        assert lines[-1] == (n, n)


def test_a_failed_block_counts_towards_progress(caplog):
    assert progress(caplog, 10, fail={4})[-1] == (10, 10)
