"""Synthetic trains, rings and the extra-speckle reference kernel for the tests."""

from __future__ import annotations

import importlib.util
from concurrent.futures import BrokenExecutor, Future
from pathlib import Path

import numpy as np
import pytest

from analysis.common.status import FrameStatus
from analysis.xpcs.rings import Rings
from analysis.xpcs.store import FlagBase, TrainPhotons

TRAIN_IDS = tuple(range(10000, 10010))
#: A train module 0 reports no frames for, and one nobody wrote a frame for.
SHORT, EMPTY = 10002, 10004
FRAMES = 12


def make_rings(rng, n_env=300, n_rings=3, outside=0.25) -> Rings:
    label = rng.integers(0, n_rings, n_env).astype(np.int8)
    label[rng.random(n_env) < outside] = -1
    label.flags.writeable = False
    return Rings(tuple(np.linspace(0.2, 0.5, n_rings + 1)), label, n_rings, "rings")


def make_dense(rng, n_frames, n_env, density=0.1) -> np.ndarray:
    return np.where(
        rng.random((n_frames, n_env)) < density,
        rng.integers(1, 4, (n_frames, n_env)),
        0,
    ).astype(np.int32)


def make_flags(rng, n_frames, n_env, rate=0.02) -> list[np.ndarray]:
    return [np.flatnonzero(rng.random(n_env) < rate) for _ in range(n_frames)]


def make_train(
    train_id: int, dense: np.ndarray, flags: list[np.ndarray], sequence: int = 0
) -> TrainPhotons:
    """A train built straight from its dense frames and per-frame flagged sets."""
    n_frames = dense.shape[0]
    rows, cols = np.nonzero(dense)
    pointers = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n_frames))])
    return TrainPhotons(
        train_id=train_id,
        status=FrameStatus.OK,
        sequence=sequence,
        pulse_ids=(200 + 2 * np.arange(n_frames)).astype(np.uint32),
        cell_ids=np.arange(n_frames, dtype=np.uint16),
        frame_ptr=pointers.astype(np.int64),
        pixel=cols.astype(np.uint32),
        count=dense[rows, cols].astype(np.uint8),
        flag_ptr=np.concatenate([[0], np.cumsum([f.size for f in flags])]).astype(
            np.int64
        ),
        flag_pixel=(
            np.concatenate(flags).astype(np.uint32)
            if flags
            else np.zeros(0, dtype=np.uint32)
        ),
        base=FlagBase.empty(),
    )


def load_extra_speckle_module(relative: str):
    """A file of the installed extra_speckle, loaded without importing the package.

    ``import extra_speckle`` imports its XCCA tools, which switch HDF5 file
    locking off for the whole process (CLAUDE.md pitfall 29); the kernel file
    needs only numpy.
    """
    spec = importlib.util.find_spec("extra_speckle")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("extra_speckle is not installed in this environment")
    path = Path(next(iter(spec.submodule_search_locations))) / relative
    if not path.exists():
        pytest.skip(f"{path} is missing from the installed extra_speckle")
    module_spec = importlib.util.spec_from_file_location(f"_es_{path.stem}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class InlinePool:
    """Runs blocks in this process, so failure paths need no real children.

    The initializer runs once, as a spawned worker would run it, and ``submit``
    returns an already-resolved future.
    """

    def __init__(self, n_workers, initializer=None, initargs=(), **kwargs):
        self.n_workers = n_workers
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
        except BaseException as error:  # noqa: BLE001 - mirrors a real pool
            future.set_exception(error)
        return future


class BrokenPool(InlinePool):
    """A pool that dies part-way through, as a killed worker would."""

    def submit(self, fn, *args):
        future: Future = Future()
        future.set_exception(BrokenExecutor("worker died"))
        return future
