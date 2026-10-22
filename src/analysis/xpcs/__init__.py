"""Hit-train XPCS: an extract-once ring-photon store and the sums derived from it.

``xpcs_hits(cfg, hits)`` reads the hit trains from proc once into a photon
store that outlives proc, merges them into exact per-train sums, and returns
an :class:`XpcsHits` that makes every float from them.
"""

from __future__ import annotations

from analysis.xpcs.config import Kernel, MaskPolicy, XpcsConfig
from analysis.xpcs.hits import HitSet
from analysis.xpcs.kernel import PairStatus
from analysis.xpcs.result import Averaging, Coverage, XpcsHits, read
from analysis.xpcs.run import run_xpcs_hits

__all__ = [
    "Averaging",
    "Coverage",
    "HitSet",
    "Kernel",
    "MaskPolicy",
    "PairStatus",
    "XpcsConfig",
    "XpcsHits",
    "read",
    "run_xpcs_hits",
    "xpcs_hits",
]


def xpcs_hits(
    cfg: XpcsConfig, hits: HitSet, *, rep_rate_mhz: float | None = None, **kwargs
) -> XpcsHits:
    """Correlate the hit trains of a run and return the result.

    :param rep_rate_mhz: DAMNIT's ``rep_rate``, checked against the pulse pattern.
    :param kwargs: those of :func:`run_xpcs_hits` (``dc``, ``geometry``, ...).
    """
    return XpcsHits(run_xpcs_hits(cfg, hits, **kwargs), rep_rate_mhz)
