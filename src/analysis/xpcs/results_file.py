"""``xpcs_hits.h5``: the layout of the per-train integer sums, its writer and checks.

The file is derived from the photon store and the hit set, so changed inputs
rebuild it (the previous file kept as ``xpcs_hits.prev.h5``) rather than refuse.
Rows are the hit set's, in order, addressed by position; a train that gave no
photons has a status and zero sums. The file holds exact integers only; every
float is made by the reader (`analysis.xpcs.reader`). This is the one module
that knows the file's dataset paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.common.complete import Completeness
from analysis.common.plan import Block
from analysis.common.readback import UnreadableOutput, sync, unreadable
from analysis.common.status import UNFINISHED, FrameStatus
from analysis.xpcs.hits import HitSet
from analysis.xpcs.kernel import Pairing, TrainSums
from analysis.xpcs.outliers import OutlierMask
from analysis.xpcs.rings import Rings

__all__ = [
    "SCHEMA",
    "SUM_NAMES",
    "Prelude",
    "ResultWriter",
    "read_header",
    "read_outlier",
    "read_ring_label",
    "read_sums",
    "result_identity",
    "result_reasons",
    "stored_outlier",
]

log = logging.getLogger(__name__)

#: Layout version; a file with another one is rebuilt. This layout stores the
#: outlier mask under ``outlier/``, the labels with it applied in ``rings/label``,
#: and the ring bounds as ``rings/bounds_nm`` ``(R, 2)``.
SCHEMA = 3
GZIP = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
#: Each per-train sum under ``sums/`` and the rank of one ring's entry: 2 for a
#: (frame, frame) map, 1 for a value per frame, 0 for a count.
SUM_RANKS = {
    "on": 2,
    "off": 2,
    "s_on": 1,
    "s_off_cur": 1,
    "s_off_prev": 1,
    "cnt": 0,
    "cnt_pair": 0,
}
SUM_NAMES = tuple(SUM_RANKS)
#: Rows per chunk, by rank: one map, or a few hundred counts, per chunk.
_CHUNK_ROWS = {2: 1, 1: 64, 0: 256}


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
        outlier: OutlierMask | None = None,
        static_label: np.ndarray | None = None,
        overwrite: bool = False,
    ) -> ResultWriter:
        """Resume a file made for these exact inputs, else build it afresh.

        A file made for other inputs is moved to ``xpcs_hits.prev.h5`` first: it
        is derived, so replacing it loses minutes, not data.

        :param rings: the labels the sums use, the outlier mask already applied.
        :param outlier: that mask, recorded under ``outlier/``.
        :param static_label: the labels before it, for the figures' "before".
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = result_identity(merge_hash, hits, prelude)
        if path.exists():
            if not overwrite and _identity(path) == identity:
                return cls(h5py.File(path, "r+"), hits)
            os.replace(path, path.with_name("xpcs_hits.prev.h5"))
            log.info("%s was made for other inputs; kept as xpcs_hits.prev.h5", path)
        handle = h5py.File(path, "w")
        cls._layout(handle, hits, prelude, rings, extract_hash, identity, attrs)
        if outlier is not None:
            _write_outlier(handle, outlier, static_label)
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
        handle.create_dataset(
            "rings/bounds_nm",
            data=np.asarray(rings.bounds_nm, dtype=np.float64).reshape(-1, 2),
        )
        handle.create_dataset("rings/label", data=rings.label, **GZIP)
        handle["rings"].attrs["sha256"] = rings.sha256

        for name, rank in SUM_RANKS.items():
            shape = (n, n_rings) + (n_frames,) * rank
            chunks = (min(n, _CHUNK_ROWS[rank]), *shape[1:])
            _create(handle, f"sums/{name}", shape, np.int32, chunks)
        handle.create_group("provenance")
        handle.flush()

    def block_complete(self, block: Block) -> bool:
        rows = [self._row(block, t) for t in block.train_ids]
        status = self._f["hits/status"][rows]
        return not bool(np.isin(status, [int(code) for code in UNFINISHED]).any())

    def _row(self, block: Block, train_id: int) -> int:
        return block.frames_for(train_id)[0]

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
        for name in SUM_NAMES:
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

    def unfinished(self) -> int:
        """Rows left ``NOT_PROCESSED`` or ``WORKER_ERROR``."""
        status = self._f["hits/status"][()]
        return int(np.isin(status, [int(code) for code in UNFINISHED]).sum())

    def status_summary(self) -> dict[str, int]:
        status = self._f["hits/status"][()]
        return {
            code.name: int((status == code).sum())
            for code in FrameStatus
            if (status == code).any()
        }

    def finalise(self, provenance: dict[str, Any]) -> None:
        """Record provenance, read the file back from disk, then mark it finalised.

        :raises UnreadableOutput: a dataset did not read back (a full disk quota
            does this); the file is removed rather than finalised.
        """
        group = self._f["provenance"]
        for key, value in provenance.items():
            group.attrs[key] = (
                value
                if isinstance(value, str | int | float)
                else json.dumps(value, default=str)
            )
        group.attrs["status_summary"] = json.dumps(self.status_summary())
        path = Path(self._f.filename)
        self._f.close()
        started = time.perf_counter()
        try:
            sync(path)
            failures = unreadable(path)
        except OSError as error:
            failures = [f"fsync: {error!r}"]
        if failures:
            # Its rows say OK, so a resume would finalise it again unchanged.
            # Removing it makes a rerun rebuild it from the store.
            path.unlink()
            raise UnreadableOutput(
                f"{path} did not read back and was removed; a rerun rebuilds it "
                f"from the photon store: {'; '.join(failures)}"
            )
        log.info("%s read back in %.1f s", path, time.perf_counter() - started)
        self._f = h5py.File(path, "r+")
        self._f.attrs["finalised"] = 1
        self._f.flush()
        sync(path)

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> ResultWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def result_identity(merge_hash: str, hits: HitSet, prelude: Prelude) -> dict[str, Any]:
    """What a results file must carry to be resumed rather than rebuilt."""
    return {
        "schema": SCHEMA,
        "merge_hash": merge_hash,
        "hits_sha256": hits.sha256,
        "prelude_sha256": prelude.sha256(),
    }


def _write_outlier(
    handle: h5py.File, outlier: OutlierMask, static_label: np.ndarray | None
) -> None:
    group = handle.create_group("outlier")
    if static_label is not None:
        group.create_dataset("label_before", data=static_label, **GZIP)
    group.create_dataset("ratio", data=outlier.ratio, **GZIP)
    group.create_dataset("mask", data=outlier.mask, **GZIP)
    group.attrs["thresholds"] = np.asarray(outlier.thresholds, dtype=np.float64)
    group.attrs["q_bins"] = outlier.q_bins
    group.attrs["n_trains"] = outlier.n_trains
    group.attrs["sha256"] = outlier.sha256
    handle.flush()


def stored_outlier(path: str | Path, identity: dict[str, Any]) -> OutlierMask | None:
    """The outlier mask of a results file made for ``identity``, else None.

    A resumed merge reads its mask back instead of recomputing it.
    """
    path = Path(path)
    if not path.exists() or _identity(path) != identity:
        return None
    with h5py.File(path, "r") as handle:
        if "outlier" not in handle:
            return None
        return _outlier_of(handle["outlier"])


def _outlier_of(group: h5py.Group) -> OutlierMask:
    mask = group["mask"][()].astype(bool)
    outlier = OutlierMask(
        group["ratio"][()],
        mask,
        tuple(float(t) for t in group.attrs["thresholds"]),
        int(group.attrs["q_bins"]),
        int(group.attrs["n_trains"]),
    )
    if outlier.sha256 != group.attrs["sha256"]:
        raise ValueError(f"{group.file.filename}: the outlier mask fails its sha256")
    return outlier


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

    Read-only, so a complete run is answered from the file with nothing written.
    The prelude is not compared: it is a function of the store and the hit set,
    and the merge hash already names the config.
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


def read_header(path: str | Path) -> dict[str, Any]:
    """The per-row arrays, the pulse pattern, the ring bounds and the file attrs.

    ``bounds_nm`` is ``(R, 2)`` [nm⁻¹], ``(0, 2)`` for rings from a masks file.
    """
    with h5py.File(path, "r") as f:
        return {
            "train_ids": f["hits/train_id"][()],
            "status": f["hits/status"][()],
            "partner_ids": f["hits/partner"][()],
            "pair_status": f["hits/pair_status"][()],
            "has_off": f["hits/has_off"][()].astype(bool),
            "pulse_ids": f["pattern/pulse_ids"][()],
            "bounds_nm": f["rings/bounds_nm"][()],
            "n_rings": int(f.attrs["n_rings"]),
            "n_frames": int(f.attrs["n_frames"]),
            "attrs": {k: v for k, v in f.attrs.items()},
        }


def read_sums(
    handle: h5py.File, rows: np.ndarray, names: tuple[str, ...] = SUM_NAMES, dtype=None
) -> dict[str, np.ndarray]:
    """The sums ``names`` of ``rows`` (ascending), cast to ``dtype`` if given."""
    out = {name: handle[f"sums/{name}"][rows.tolist()] for name in names}
    if dtype is not None:
        out = {name: array.astype(dtype) for name, array in out.items()}
    return out


def read_ring_label(path: str | Path) -> np.ndarray:
    """The ring label of every envelope pixel the sums use; -1 for none."""
    with h5py.File(path, "r") as f:
        return f["rings/label"][()]


def read_outlier(path: str | Path) -> tuple[OutlierMask, np.ndarray] | None:
    """The applied pixel-outlier mask and the ring labels before it; None if off."""
    with h5py.File(path, "r") as handle:
        if "outlier" not in handle:
            return None
        group = handle["outlier"]
        return _outlier_of(group), group["label_before"][()]
