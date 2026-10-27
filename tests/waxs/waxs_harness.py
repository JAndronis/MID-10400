"""Injected pools and the call every orchestration test makes.

The pool is injected rather than real wherever a test is about a failure path:
killing processes to exercise ``BrokenExecutor`` would put test hooks in
``worker``.
"""

from __future__ import annotations

from concurrent.futures import BrokenExecutor, Future

from analysis.waxs.run import run_jungfrau_waxs


class InlinePool:
    """Runs every block in this process, with the real initializer."""

    def __init__(self, n_workers, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        from analysis.waxs import worker

        worker._STATE = None

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as error:  # noqa: BLE001 - handed to the caller
            future.set_exception(error)
        return future


class FailingPool(InlinePool):
    """Every block raises, so the ledger takes ``WORKER_ERROR``."""

    def submit(self, fn, *args):
        future = Future()
        future.set_exception(RuntimeError("block exploded"))
        return future


class BrokenPool(InlinePool):
    """The executor dies, as a killed worker makes it."""

    def submit(self, fn, *args):
        future = Future()
        future.set_exception(BrokenExecutor("pool died"))
        return future


def run(cfg, mock, dc, **kwargs):
    """``run_jungfrau_waxs`` on a mock run, in this process unless told otherwise."""
    return run_jungfrau_waxs(
        cfg,
        dc=dc,
        run_dir=mock.path,
        output_path=kwargs.pop("output_path", None),
        pool_factory=kwargs.pop("pool_factory", InlinePool),
        **kwargs,
    )
