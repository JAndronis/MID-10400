"""Array helpers shared by both passes."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["frozen_copy", "relative_difference"]


def relative_difference(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """``|got - want| / max(|want|, 1e-12)``, elementwise.

    The floor keeps a zero reference from producing a division warning and an
    infinite difference; a bin where both sides are zero reports 0.

    :param got: the value under test.
    :param want: the reference it is compared against.
    :returns: the relative difference, same shape as the inputs.
    """
    return np.abs(got - want) / np.maximum(np.abs(want), 1e-12)


def frozen_copy(array: np.ndarray, dtype: Any, *, cast: bool = False) -> np.ndarray:
    """A contiguous, read-only **copy**.

    The copy is the point: ``np.ascontiguousarray(x, dtype)`` returns ``x``
    itself when it already matches, so freezing the result would freeze an array
    pyFAI still owns — a solid-angle cache, or an engine's bin centres — and its
    Cython kernels reject a read-only buffer, surfacing as a failure somewhere
    unrelated later.

    :param array: the array to copy.
    :param dtype: the dtype the result must have.
    :param cast: convert to ``dtype`` when it differs. The default asserts
        instead, so that a change in an upstream layout surfaces loudly rather
        than being silently converted.
    :returns: the frozen copy.
    :raises TypeError: ``cast`` is False and ``array`` has another dtype.
    """
    if not cast and array.dtype != dtype:
        raise TypeError(f"expected dtype {np.dtype(dtype)}, got {array.dtype}")
    out = np.array(array, dtype=dtype, copy=True, order="C")
    out.flags.writeable = False
    return out
