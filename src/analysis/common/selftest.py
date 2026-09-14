"""The comparison primitive both passes' self-test gates are written against."""

from __future__ import annotations

import numpy as np

__all__ = ["relative_difference"]


def relative_difference(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """``|got - want| / max(|want|, 1e-12)``, elementwise.

    The floor keeps a zero reference from producing a division warning and an
    infinite difference; a bin where both sides are zero reports 0.

    :param got: the value under test.
    :param want: the reference it is compared against.
    :returns: the relative difference, same shape as the inputs.
    """
    return np.abs(got - want) / np.maximum(np.abs(want), 1e-12)
