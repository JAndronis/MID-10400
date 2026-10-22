"""Who pairs with whom, and the exact integer statistics of one train.

Both are pure: :func:`partners` is a function of the sorted hit list and the
trains' statuses, and :func:`train_sums` of a train's photons, its partner's and
the ring map. So the result does not depend on how blocks were split across
workers, and integer sums pool in any order.

Notation for a ring ``r``: ``X`` is the train's frames (``P`` rows, pixels as
columns), ``Y`` its partner's, ``m`` the mask in force.

    on[a,b]     = Σ_p X[a,p] X[b,p] m_p          raw two-time sum
    s_on[a]     = Σ_p X[a,p] m_p                 photons in frame a
    cnt         = Σ_p m_p                        pixels
    off[a,b]    = Σ_p X[a,p] Y[b,p] m_p          against the partner's frames
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from analysis.xpcs.config import Kernel, MaskPolicy
from analysis.xpcs.rings import Rings
from analysis.xpcs.store import TrainPhotons

__all__ = [
    "INT32_LIMIT",
    "Pairing",
    "PairStatus",
    "TrainSums",
    "partners",
    "train_sums",
]

INT32_LIMIT = 2**31


class PairStatus(IntEnum):
    """Why a train has the partner it has, or none."""

    PAIRED = 0
    #: No earlier usable hit (for ``Kernel.ES``, none earlier in its sequence file).
    FIRST_HIT = 1
    #: The previous usable hit is more than ``max_partner_gap`` trains back.
    GAP_TOO_LARGE = 2
    #: A hit within the gap exists but gave no photons, and no usable one does.
    PARTNER_FAILED = 3
    #: The train itself gave no photons, so it is not paired at all.
    NOT_OK = 255


@dataclass(frozen=True, slots=True)
class Pairing:
    """Each hit's partner (0: none) and the reason."""

    partner: np.ndarray  # uint64 (n,)
    status: np.ndarray  # uint8 (n,), PairStatus


def partners(
    train_ids: np.ndarray,
    ok: np.ndarray,
    sequence: np.ndarray,
    kernel: Kernel,
    max_gap: int,
) -> Pairing:
    """Pair each usable hit with the previous usable hit.

    Trains that gave no photons neither have a partner nor are one. There is no
    "next hit" fallback: it would count a pair twice and let a later hit change
    an earlier train's baseline.

    :param train_ids: the hit set, ascending.
    :param ok: whether each hit's photons are usable.
    :param sequence: the proc sequence file of each hit; only ``Kernel.ES`` reads it.
    :param max_gap: largest train-id distance to a partner under ``Kernel.CLEAN``.
    """
    n = len(train_ids)
    partner = np.zeros(n, dtype=np.uint64)
    status = np.full(n, PairStatus.NOT_OK, dtype=np.uint8)
    last = None  # index of the previous usable hit
    last_failed = None  # id of the previous unusable hit
    for i in range(n):
        if not ok[i]:
            last_failed = int(train_ids[i])
            continue
        tid = int(train_ids[i])
        if kernel is Kernel.ES:
            same_file = last is not None and sequence[last] == sequence[i]
            status[i] = PairStatus.PAIRED if same_file else PairStatus.FIRST_HIT
        elif last is not None and tid - int(train_ids[last]) <= max_gap:
            status[i] = PairStatus.PAIRED
        elif (
            last_failed is not None
            and tid - last_failed <= max_gap
            and (last is None or last_failed > int(train_ids[last]))
        ):
            status[i] = PairStatus.PARTNER_FAILED
        elif last is None:
            status[i] = PairStatus.FIRST_HIT
        else:
            status[i] = PairStatus.GAP_TOO_LARGE
        if status[i] == PairStatus.PAIRED:
            partner[i] = train_ids[last]
        last = i
    return Pairing(partner, status)


@dataclass(frozen=True, slots=True)
class TrainSums:
    """Exact integer statistics of one train (int32 on disk, overflow-checked).

    ``off`` and its three companions are zeros when the train has no baseline
    (``has_off`` False); that is not the same as a baseline of zero.
    """

    train_id: int
    partner_id: int  # 0: none
    has_off: bool
    on: np.ndarray  # (R, P, P)
    s_on: np.ndarray  # (R, P)
    cnt: np.ndarray  # (R,)
    off: np.ndarray  # (R, P, P)
    s_off_cur: np.ndarray  # (R, P)
    s_off_prev: np.ndarray  # (R, P)
    cnt_pair: np.ndarray  # (R,)


def good_pixels(photons: TrainPhotons, rings: Rings, policy: MaskPolicy) -> np.ndarray:
    """Envelope pixels a train correlates: in a ring and, under ``ANY_FRAME``,
    not flagged in any frame."""
    good = rings.label >= 0
    if policy is MaskPolicy.ANY_FRAME:
        good &= ~photons.flagged_any(rings.label.size)
    return good


def train_sums(
    cur: TrainPhotons,
    prev: TrainPhotons | None,
    rings: Rings,
    kernel: Kernel,
    policy: MaskPolicy = MaskPolicy.ANY_FRAME,
) -> TrainSums:
    """The sums of ``cur`` against its partner ``prev`` (``None``: no partner).

    ``Kernel.CLEAN`` correlates the train over its own good pixels and its
    baseline over the pixels good in both trains. ``Kernel.ES`` is
    extra-speckle's: with a partner both use the pair mask, without one the
    baseline is the train against its own mask, which makes it exactly the
    train's photon sum (and a baseline of 1 once normalised).
    """
    n_env = rings.label.size
    n_rings = rings.n_rings
    n_frames = cur.n_frames
    if prev is not None and prev.n_frames != n_frames:
        raise ValueError(
            f"train {cur.train_id} has {n_frames} frames, its partner "
            f"{prev.train_id} has {prev.n_frames}"
        )
    good = good_pixels(cur, rings, policy)
    pair = good & good_pixels(prev, rings, policy) if prev is not None else None
    own = pair if (kernel is Kernel.ES and pair is not None) else good

    x = cur.frames(n_env).tocsc().astype(np.int64)
    y = prev.frames(n_env).tocsc().astype(np.int64) if prev is not None else None
    ring_of = rings.label

    on = np.zeros((n_rings, n_frames, n_frames), dtype=np.int64)
    s_on = np.zeros((n_rings, n_frames), dtype=np.int64)
    cnt = np.zeros(n_rings, dtype=np.int64)
    off = np.zeros_like(on)
    s_off_cur = np.zeros_like(s_on)
    s_off_prev = np.zeros_like(s_on)
    cnt_pair = np.zeros_like(cnt)

    for ring in range(n_rings):
        columns = np.flatnonzero(own & (ring_of == ring))
        xr = x[:, columns]
        on[ring] = (xr @ xr.T).toarray()
        s_on[ring] = np.asarray(xr.sum(axis=1)).ravel()
        cnt[ring] = columns.size
        if pair is not None:
            columns = np.flatnonzero(pair & (ring_of == ring))
            xc, yp = x[:, columns], y[:, columns]
            off[ring] = (xc @ yp.T).toarray()
            s_off_cur[ring] = np.asarray(xc.sum(axis=1)).ravel()
            s_off_prev[ring] = np.asarray(yp.sum(axis=1)).ravel()
            cnt_pair[ring] = columns.size
        elif kernel is Kernel.ES:
            off[ring] = s_on[ring][:, None]
            s_off_cur[ring] = s_on[ring]
            s_off_prev[ring] = cnt[ring]
            cnt_pair[ring] = cnt[ring]

    for name, array in (
        ("on", on),
        ("s_on", s_on),
        ("off", off),
        ("s_off_cur", s_off_cur),
        ("s_off_prev", s_off_prev),
    ):
        if array.size and array.max() >= INT32_LIMIT:
            raise OverflowError(
                f"train {cur.train_id}: a {name} sum reaches {array.max()}, which "
                "int32 cannot hold"
            )
    return TrainSums(
        train_id=cur.train_id,
        partner_id=int(prev.train_id) if prev is not None else 0,
        has_off=pair is not None or kernel is Kernel.ES,
        on=on.astype(np.int32),
        s_on=s_on.astype(np.int32),
        cnt=cnt.astype(np.int32),
        off=off.astype(np.int32),
        s_off_cur=s_off_cur.astype(np.int32),
        s_off_prev=s_off_prev.astype(np.int32),
        cnt_pair=cnt_pair.astype(np.int32),
    )
