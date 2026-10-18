"""``xpcs_hits.h5``: the per-train integer sums, and the reader that turns them into g2.

Everything in the file is derived from the photon store and the hit set, so
changed inputs rebuild it (the previous file kept as ``xpcs_hits.prev.h5``)
rather than refuse. Rows are the hit set's, in order, and are addressed by
position: a train that gave no photons has a status and zero sums.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.common.complete import Completeness
from analysis.common.plan import Block
from analysis.common.status import UNFINISHED, FrameStatus
from analysis.xpcs.hits import HitSet
from analysis.xpcs.kernel import Pairing, TrainSums
from analysis.xpcs.rings import Rings

__all__ = [
    "SCHEMA",
    "Prelude",
    "ResultWriter",
    "result_reasons",
]

log = logging.getLogger(__name__)

SCHEMA = 1
GZIP = {"compression": "gzip", "compression_opts": 1, "shuffle": True}


@dataclass(frozen=True, slots=True)
class Prelude:
    """What the merge stage decides from the store before any sum is made."""

    status: np.ndarray  # uint8 (T,): the store's status, or the pattern check's
    pairing: Pairing
    sequence: np.ndarray  # int32 (T,)
    pattern: np.ndarray  # uint32 (P,) pulse ids shared by the usable trains

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for array in (self.status, self.pairing.partner, self.pairing.status):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(np.ascontiguousarray(self.sequence).tobytes())
        digest.update(np.ascontiguousarray(self.pattern).tobytes())
        return digest.hexdigest()


def _create(handle: h5py.File, name: str, shape: tuple[int, ...], dtype, chunks=None):
    """A zero-filled dataset; HDF5 cannot chunk or compress an empty one."""
    if all(shape) and chunks is not None:
        return handle.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks, fillvalue=0, **GZIP
        )
    return handle.create_dataset(name, shape=shape, dtype=dtype, fillvalue=0)


class ResultWriter:
    """The single writer of one run's ``xpcs_hits.h5``."""

    def __init__(self, handle: h5py.File, hits: HitSet) -> None:
        self._f = handle
        self._hits = hits

    @classmethod
    def open_or_create(
        cls,
        path: Path,
        *,
        hits: HitSet,
        prelude: Prelude,
        rings: Rings,
        merge_hash: str,
        extract_hash: str,
        attrs: dict[str, Any],
        overwrite: bool = False,
    ) -> ResultWriter:
        """Resume a file made for these exact inputs, else build it afresh.

        A file made for other inputs is moved to ``xpcs_hits.prev.h5`` first: it
        is derived, so replacing it loses minutes, not data.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = {
            "schema": SCHEMA,
            "merge_hash": merge_hash,
            "hits_sha256": hits.sha256,
            "prelude_sha256": prelude.sha256(),
        }
        if path.exists():
            if not overwrite and _identity(path) == identity:
                return cls(h5py.File(path, "r+"), hits)
            os.replace(path, path.with_name("xpcs_hits.prev.h5"))
            log.info("%s was made for other inputs; kept as xpcs_hits.prev.h5", path)
        handle = h5py.File(path, "w")
        cls._layout(handle, hits, prelude, rings, extract_hash, identity, attrs)
        return cls(handle, hits)

    @staticmethod
    def _layout(handle, hits, prelude, rings, extract_hash, identity, attrs) -> None:
        n, n_rings, n_frames = len(hits), rings.n_rings, prelude.pattern.size
        for key, value in identity.items():
            handle.attrs[key] = value
        handle.attrs["extract_hash"] = extract_hash
        handle.attrs["hits_source"] = hits.source
        handle.attrs["n_rings"] = n_rings
        handle.attrs["n_frames"] = n_frames
        for key, value in attrs.items():
            handle.attrs[key] = value

        initial = prelude.status.copy()
        initial[initial == FrameStatus.OK] = FrameStatus.NOT_PROCESSED
        handle.create_dataset("hits/train_id", data=hits.train_ids)
        handle.create_dataset("hits/status", data=initial)
        handle.create_dataset("hits/partner", data=prelude.pairing.partner)
        handle.create_dataset("hits/pair_status", data=prelude.pairing.status)
        handle.create_dataset("hits/sequence", data=prelude.sequence)
        handle.create_dataset("hits/has_off", data=np.zeros(n, dtype=np.uint8))
        handle.create_dataset("pattern/pulse_ids", data=prelude.pattern)
        handle.create_dataset("rings/edges_nm", data=np.asarray(rings.edges_nm))
        handle.create_dataset("rings/label", data=rings.label, **GZIP)
        handle["rings"].attrs["sha256"] = rings.sha256

        cube = (n, n_rings, n_frames, n_frames)
        plane = (n, n_rings, n_frames)
        row = (n, n_rings)
        for name in ("on", "off"):
            _create(handle, f"sums/{name}", cube, np.int32, (1, *cube[1:]))
        for name in ("s_on", "s_off_cur", "s_off_prev"):
            _create(handle, f"sums/{name}", plane, np.int32, (min(n, 64), *plane[1:]))
        for name in ("cnt", "cnt_pair"):
            _create(handle, f"sums/{name}", row, np.int32, (min(n, 256), n_rings))
        handle.create_group("provenance")
        handle.flush()

    # ── resume ───────────────────────────────────────────────────────────────
    def block_complete(self, block: Block) -> bool:
        rows = [self._row(block, t) for t in block.train_ids]
        status = self._f["hits/status"][rows]
        return not bool(np.isin(status, [int(code) for code in UNFINISHED]).any())

    def _row(self, block: Block, train_id: int) -> int:
        return block.frames_for(train_id)[0]

    # ── writing ──────────────────────────────────────────────────────────────
    def write_block(self, block: Block, result: Any) -> None:
        for sums in result.sums:
            self._write(self._row(block, sums.train_id), sums)
        self._f.flush()

    def _write(self, row: int, sums: TrainSums) -> None:
        f = self._f
        if int(f["hits/train_id"][row]) != sums.train_id:
            raise ValueError(
                f"row {row} is train {f['hits/train_id'][row]}, not {sums.train_id}"
            )
        if int(f["hits/partner"][row]) != sums.partner_id:
            raise ValueError(
                f"train {sums.train_id}: summed against {sums.partner_id}, the "
                f"pairing says {int(f['hits/partner'][row])}"
            )
        for name in ("on", "off", "s_on", "s_off_cur", "s_off_prev", "cnt", "cnt_pair"):
            f[f"sums/{name}"][row] = getattr(sums, name)
        f["hits/has_off"][row] = sums.has_off
        f["hits/status"][row] = FrameStatus.OK

    def mark(self, block: Block, status: FrameStatus, message: str = "") -> None:
        for train_id in block.train_ids:
            self._f["hits/status"][self._row(block, train_id)] = status
        if message:
            attrs = self._f["provenance"].attrs
            errors = json.loads(attrs.get("block_errors", "{}"))
            errors[str(block.index)] = message
            attrs["block_errors"] = json.dumps(errors)
        self._f.flush()

    def mark_remaining(self, status: FrameStatus) -> None:
        """Stamp every still-unprocessed row, e.g. after a broken pool."""
        current = self._f["hits/status"][()]
        current[current == FrameStatus.NOT_PROCESSED] = status
        self._f["hits/status"][...] = current
        self._f.flush()

    def status_summary(self) -> dict[str, int]:
        status = self._f["hits/status"][()]
        return {
            code.name: int((status == code).sum())
            for code in FrameStatus
            if (status == code).any()
        }

    def finalise(self, provenance: dict[str, Any]) -> None:
        group = self._f["provenance"]
        for key, value in provenance.items():
            group.attrs[key] = (
                value
                if isinstance(value, str | int | float)
                else json.dumps(value, default=str)
            )
        group.attrs["status_summary"] = json.dumps(self.status_summary())
        self._f.attrs["finalised"] = 1
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> ResultWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _identity(path: Path) -> dict[str, Any] | None:
    try:
        with h5py.File(path, "r") as handle:
            return {
                key: handle.attrs.get(key)
                for key in ("schema", "merge_hash", "hits_sha256", "prelude_sha256")
            }
    except OSError:
        return None


def result_reasons(path: str | Path, hits: HitSet, merge_hash: str) -> Completeness:
    """Why the results file at ``path`` is not complete; reasons empty if it is.

    Read-only, so a complete run is answered from the file with nothing written
    (CLAUDE.md pitfall 26). The prelude is not compared here: it is a function of
    the store and the hit set, and the merge hash already names the config.
    """
    path = Path(path)
    if not path.exists():
        return Completeness(path, ("no file",))
    try:
        with h5py.File(path, "r") as handle:
            attrs = handle.attrs
            reasons = []
            if attrs.get("schema") != SCHEMA:
                reasons.append(f"schema {attrs.get('schema')}, this code's is {SCHEMA}")
            if attrs.get("merge_hash") != merge_hash:
                reasons.append("it was merged under another configuration")
            if attrs.get("hits_sha256") != hits.sha256:
                reasons.append("it was made for another hit set")
            if "finalised" not in attrs:
                reasons.append("never finalised")
            status = handle["hits/status"][()]
            for code in sorted(UNFINISHED):
                if (status == code).any():
                    reasons.append(f"{int((status == code).sum())} trains {code.name}")
            return Completeness(path, tuple(reasons))
    except (OSError, KeyError, ValueError) as error:
        return Completeness(path, (f"unreadable: {error!r}",))
