"""Worker for the second stage: the store's photons into per-train sums.

Reads only the photon store, never proc, so it can run after proc is taped.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from analysis.common.cpu import set_thread_env
from analysis.common.plan import Block
from analysis.xpcs.config import Kernel, MaskPolicy
from analysis.xpcs.kernel import TrainSums, train_sums
from analysis.xpcs.rings import Rings
from analysis.xpcs.store import RingPhotonStore, TrainPhotons

__all__ = ["MergeResult", "close", "init", "init_from_store", "process_block"]

#: Trains kept in memory: the train being summed and, usually, its partner.
CACHE = 3


@dataclass(slots=True)
class MergeResult:
    block_index: int
    sums: list[TrainSums]
    bits_present: int = 0
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _State:
    store: RingPhotonStore
    rings: Rings
    kernel: Kernel
    policy: MaskPolicy
    partner_of: dict[int, int]
    cache: OrderedDict = field(default_factory=OrderedDict)


_STATE: _State | None = None


def init(
    store_path: str | Path,
    extract_hash: str,
    rings: Rings,
    kernel: Kernel,
    policy: MaskPolicy,
    partner_of: dict[int, int],
) -> None:
    """Process initializer: open the store read-only."""
    set_thread_env()
    init_from_store(
        RingPhotonStore.open(Path(store_path), extract_hash),
        rings,
        kernel,
        policy,
        partner_of,
    )


def init_from_store(
    store: RingPhotonStore,
    rings: Rings,
    kernel: Kernel,
    policy: MaskPolicy,
    partner_of: dict[int, int],
) -> None:
    global _STATE
    _STATE = _State(store, rings, kernel, policy, partner_of)


def close() -> None:
    """Close the store this process opened.

    Spawned workers die with theirs. An inline pool runs the initializer in the
    parent, which would otherwise keep the store open read-only and so refuse
    to reopen it for writing.
    """
    global _STATE
    if _STATE is not None:
        _STATE.store.close()
        _STATE = None


def _photons(state: _State, train_id: int) -> TrainPhotons:
    cache = state.cache
    if train_id in cache:
        cache.move_to_end(train_id)
        return cache[train_id]
    cache[train_id] = state.store.get(train_id)
    while len(cache) > CACHE:
        cache.popitem(last=False)
    return cache[train_id]


def process_block(block: Block) -> MergeResult:
    """Sum every train of the block against its partner."""
    import time

    if _STATE is None:
        raise RuntimeError("merge.init has not run in this process")
    state = _STATE
    started = time.perf_counter()
    sums = []
    for train_id in block.train_ids:
        partner = state.partner_of.get(train_id, 0)
        sums.append(
            train_sums(
                _photons(state, train_id),
                _photons(state, partner) if partner else None,
                state.rings,
                state.kernel,
                state.policy,
            )
        )
    return MergeResult(
        block.index, sums, 0, {"train_sums": time.perf_counter() - started}
    )
