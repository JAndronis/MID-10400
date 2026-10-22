"""The hit set: which trains the XPCS pass reads.

The pass takes the train ids as an input and knows nothing of the classifier
that chose them, so the hit rule can change without touching the pass. Only
``analysis.xpcs.damnit`` imports ``analysis.triage.classify``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["HitSet"]


@dataclass(frozen=True, slots=True)
class HitSet:
    """Sorted, unique train ids, validated once where they enter the pass."""

    train_ids: np.ndarray  # uint64 (n,), strictly increasing, read-only
    sha256: str
    #: Where the ids came from (a classifier version, a file). Provenance only,
    #: so it is not hashed: the same ids from two sources are the same hit set.
    source: str = ""

    @classmethod
    def of(cls, train_ids: Any, source: str = "") -> HitSet:
        """Validate ``train_ids``; an empty set is allowed (no hits is a result).

        :raises ValueError: not 1-D, not integer-valued, negative, unsorted or
            containing a duplicate. Nothing is sorted or deduplicated for the
            caller, because a classifier that returns either has a bug.
        """
        ids = np.asarray(train_ids)
        if ids.ndim != 1:
            raise ValueError(f"train ids must be 1-D, got shape {ids.shape}")
        if ids.size == 0:
            ids = ids.astype(np.uint64)
        elif not np.issubdtype(ids.dtype, np.integer):
            as_int = ids.astype(np.int64)
            if not np.array_equal(as_int, ids):
                raise ValueError("train ids must be integers")
            ids = as_int
        if ids.size and ids.min() < 0:
            raise ValueError("train ids must not be negative")
        ids = np.array(ids, dtype=np.uint64)
        if ids.size > 1:
            if (ids[1:] == ids[:-1]).any():
                raise ValueError("train ids contain duplicates")
            if (ids[1:] < ids[:-1]).any():
                raise ValueError("train ids must be sorted ascending")
        ids.flags.writeable = False
        return cls(ids, hashlib.sha256(ids.astype("<u8").tobytes()).hexdigest(), source)

    def __len__(self) -> int:
        return int(self.train_ids.size)
