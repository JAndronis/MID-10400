"""``agipd_ring_photons.h5``: the hit trains' raw photons, kept after proc is taped.

One group per train, so the file is the union of whichever trains were extracted
and its content does not depend on the order or the number of workers that
wrote it. Everything in it is raw (counts and ``image.mask`` flags over the
envelope, no ring, no static mask, no kernel), which is why it is keyed on the
envelope alone.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp

from analysis.common.status import UNFINISHED, FrameStatus

__all__ = [
    "ABSENT",
    "SCHEMA",
    "FlagBase",
    "RingPhotonStore",
    "StoreMismatch",
    "TrainPhotons",
    "block_dir_for",
    "delta_decode",
    "delta_encode",
    "write_block_file",
]

log = logging.getLogger(__name__)

#: Bump when what the file holds changes; the extract hash covers it.
SCHEMA = 1
#: What :meth:`RingPhotonStore.status` reports for a train with no group.
ABSENT = 255
GZIP = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def _empty(dtype: type) -> np.ndarray:
    return np.zeros(0, dtype=dtype)


@dataclass(frozen=True, slots=True)
class FlagBase:
    """Each memory cell's usual set of flagged pixels, as envelope positions.

    A memory cell flags nearly the same pixels in every train, so a frame's flags
    are stored as their difference from this base, a much shorter list than the
    flags themselves. Any base is lossless; a better one only makes the file
    smaller, which is why it is fixed once, when the store is created, and why it
    is not part of the store's identity.
    """

    cells: np.ndarray  # (n,) uint16, ascending
    ptr: np.ndarray  # (n + 1,) int64
    pixel: np.ndarray  # (nnz,) uint32, ascending within a cell

    @classmethod
    def empty(cls) -> FlagBase:
        """No base: every flagged pixel of every frame is listed."""
        return cls(_empty(np.uint16), np.zeros(1, dtype=np.int64), _empty(np.uint32))

    @classmethod
    def from_samples(
        cls, samples: Iterable[tuple[np.ndarray, list[np.ndarray]]]
    ) -> FlagBase:
        """The pixels flagged in more than half of a cell's sampled frames.

        :param samples: per sample train, its cell ids and each frame's flagged
            pixels. A cell never seen gets an empty base.
        """
        seen: dict[int, list[np.ndarray]] = {}
        for cells, flagged in samples:
            for cell, pixels in zip(cells.tolist(), flagged, strict=True):
                seen.setdefault(int(cell), []).append(pixels)
        ids = sorted(seen)
        parts = []
        for cell in ids:
            frames = seen[cell]
            pixels, votes = np.unique(np.concatenate(frames), return_counts=True)
            parts.append(pixels[votes * 2 > len(frames)].astype(np.uint32))
        return cls(
            np.asarray(ids, dtype=np.uint16),
            np.concatenate([[0], np.cumsum([p.size for p in parts])]).astype(np.int64),
            np.concatenate(parts) if parts else _empty(np.uint32),
        )

    def of_cell(self, cell: int) -> np.ndarray:
        index = int(np.searchsorted(self.cells, cell))
        if index == self.cells.size or self.cells[index] != cell:
            return _empty(np.uint32)
        return self.pixel[self.ptr[index] : self.ptr[index + 1]]

    def difference(self, cell: int, flagged: np.ndarray) -> np.ndarray:
        """A frame's flagged pixels as their symmetric difference from the base."""
        return np.setxor1d(flagged, self.of_cell(cell), assume_unique=True).astype(
            np.uint32
        )


@dataclass(frozen=True, slots=True)
class TrainPhotons:
    """One hit train: photons per frame and the pixels ``image.mask`` flagged.

    Pixel positions are indices into the envelope, not the detector. A frame's
    flagged set is its ``flag_pixel`` entries XOR the ``base`` of its memory
    cell. A train that is not ``OK`` carries no arrays.
    """

    train_id: int
    status: FrameStatus
    #: Proc sequence file the train sits in (the ``Kernel.ES`` chain).
    sequence: int
    pulse_ids: np.ndarray  # (P,) uint32
    cell_ids: np.ndarray  # (P,) uint16
    frame_ptr: np.ndarray  # (P + 1,) int64
    pixel: np.ndarray  # (nnz,) uint32, ascending within a frame
    count: np.ndarray  # (nnz,) uint8
    flag_ptr: np.ndarray  # (P + 1,) int64
    flag_pixel: np.ndarray  # (nf,) uint32, ascending within a frame
    base: FlagBase

    @property
    def n_frames(self) -> int:
        return int(self.pulse_ids.size)

    @classmethod
    def failed(cls, train_id: int, status: FrameStatus, sequence: int = -1):
        """A train that gave no photons, and why."""
        return cls(
            train_id,
            status,
            sequence,
            _empty(np.uint32),
            _empty(np.uint16),
            np.zeros(1, dtype=np.int64),
            _empty(np.uint32),
            _empty(np.uint8),
            np.zeros(1, dtype=np.int64),
            _empty(np.uint32),
            FlagBase.empty(),
        )

    def frames(self, n_env: int) -> sp.csr_matrix:
        """Photons as a ``(P, n_env)`` int32 sparse matrix."""
        return sp.csr_matrix(
            (
                self.count.astype(np.int32),
                self.pixel.astype(np.int32),
                self.frame_ptr.astype(np.int64),
            ),
            shape=(self.n_frames, n_env),
        )

    def flagged_frame(self, frame: int) -> np.ndarray:
        """One frame's flagged pixels, ascending."""
        listed = self.flag_pixel[self.flag_ptr[frame] : self.flag_ptr[frame + 1]]
        return np.setxor1d(
            listed, self.base.of_cell(int(self.cell_ids[frame])), assume_unique=True
        )

    def flagged_any(self, n_env: int) -> np.ndarray:
        """Pixels flagged in at least one frame, as a bool array over the envelope."""
        flagged = np.zeros(n_env, dtype=bool)
        for frame in range(self.n_frames):
            flagged[self.flagged_frame(frame)] = True
        return flagged


def delta_encode(pixel: np.ndarray, frame_ptr: np.ndarray) -> np.ndarray:
    """Positions as steps from the previous event, restarting at each frame.

    The first event of a frame keeps its absolute position. Gzip compresses the
    steps better than the positions.
    """
    step = np.diff(pixel.astype(np.int64), prepend=0)
    starts = frame_ptr[:-1]
    starts = starts[starts < pixel.size]
    step[starts] = pixel[starts]
    return step.astype(np.uint32)


def delta_decode(step: np.ndarray, frame_ptr: np.ndarray) -> np.ndarray:
    """Inverse of :func:`delta_encode`."""
    if step.size == 0:
        return step.astype(np.uint32)
    total = np.cumsum(step.astype(np.int64))
    counts = np.diff(frame_ptr)
    # what the running sum held when each frame began, taken off that frame
    before = np.concatenate([[0], total])[frame_ptr[:-1]]
    return (total - np.repeat(before, counts)).astype(np.uint32)


class StoreMismatch(RuntimeError):
    """An existing photon store was written under another extract hash.

    Never resolved by overwriting: proc cannot give the photons back once it is
    taped, so replacing a store is a decision for a person.
    """


def _group_name(train_id: int) -> str:
    return f"t{int(train_id):012d}"


def _put(group: h5py.Group, name: str, array: np.ndarray) -> None:
    """Gzip-compressed unless empty, which HDF5 cannot chunk."""
    if array.size:
        group.create_dataset(name, data=array, **GZIP)
    else:
        group.create_dataset(name, data=array)


def _write_train(trains: h5py.Group, photons: TrainPhotons) -> None:
    """One train's group, ``status`` last; the layout of ``put`` and of a block file."""
    group = trains.create_group(_group_name(photons.train_id))
    if photons.status is FrameStatus.OK:
        _put(group, "pulse_ids", photons.pulse_ids)
        _put(group, "cell_ids", photons.cell_ids)
        _put(group, "frame_ptr", photons.frame_ptr.astype(np.uint32))
        _put(group, "pixel_step", delta_encode(photons.pixel, photons.frame_ptr))
        _put(group, "count", photons.count)
        _put(group, "flag_ptr", photons.flag_ptr.astype(np.uint32))
        _put(group, "flag_pixel", photons.flag_pixel)
    group.attrs["sequence"] = photons.sequence
    group.attrs["status"] = int(photons.status)


def block_dir_for(store_path: Path) -> Path:
    """Where a run's block files live: beside the store, named after it."""
    store_path = Path(store_path)
    return store_path.with_name(store_path.stem + ".blocks")


def write_block_file(
    path: Path, extract_hash: str, trains: Iterable[TrainPhotons]
) -> None:
    """Write a worker's finished trains as a file of the store's own layout.

    The groups sit at the file's root, named as in the store's ``trains/``. The
    file is written under a ``.tmp`` name and renamed, so a block file that exists
    is whole: a crash leaves a ``.tmp`` or nothing. It never touches the store.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with h5py.File(tmp, "w") as handle:
            handle.attrs["schema"] = SCHEMA
            handle.attrs["extract_hash"] = extract_hash
            for photons in trains:
                _write_train(handle, photons)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _check_copy(source: h5py.Group, copied: h5py.Group) -> None:
    """A copied group against its source: datasets, shapes, dtypes, stored bytes."""
    if set(source) != set(copied):
        raise RuntimeError(
            f"{copied.name}: datasets {sorted(copied)} != {sorted(source)}"
        )
    for name, dataset in source.items():
        other = copied[name]
        if (
            other.shape != dataset.shape
            or other.dtype != dataset.dtype
            or other.id.get_storage_size() != dataset.id.get_storage_size()
        ):
            raise RuntimeError(f"{copied.name}/{name} differs from its block file")


class RingPhotonStore:
    """The union of the trains extracted so far, one group each.

    A group's ``status`` attribute is written last, so a group without one is
    what a crashed writer leaves behind and counts as unfinished. Only the parent
    process writes; workers open the file read-only after the parent has closed
    it.
    """

    def __init__(self, handle: h5py.File, path: Path, writable: bool) -> None:
        self._f = handle
        self.path = path
        self.writable = writable
        self.extract_hash: str = handle.attrs["extract_hash"]
        self.n_env: int = int(handle.attrs["n_env"])
        self.base = FlagBase(
            handle["base/cells"][()], handle["base/ptr"][()], handle["base/pixel"][()]
        )

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def create(
        cls,
        path: Path,
        extract_hash: str,
        envelope: np.ndarray,
        base: FlagBase,
        provenance: dict[str, Any] | None = None,
    ) -> RingPhotonStore:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.attrs["schema"] = SCHEMA
            handle.attrs["extract_hash"] = extract_hash
            handle.attrs["n_env"] = int(envelope.size)
            handle.create_dataset("envelope/pixels", data=envelope, **GZIP)
            handle.create_dataset("base/cells", data=base.cells)
            handle.create_dataset("base/ptr", data=base.ptr)
            _put(handle["base"], "pixel", base.pixel)
            handle.create_group("trains")
            handle.create_group("provenance")
            for key, value in (provenance or {}).items():
                handle["provenance"].attrs[key] = json.dumps(value, default=str)
        return cls.open(path, extract_hash, writable=True)

    @classmethod
    def open(
        cls, path: Path, extract_hash: str | None = None, *, writable: bool = False
    ) -> RingPhotonStore:
        """Open an existing store, refusing one written under another hash.

        :param extract_hash: what the caller's config and envelope say it should
            be; ``None`` opens whatever is there (a reader).
        :raises StoreMismatch: the hashes differ, or the schema is not this code's.
        """
        path = Path(path)
        handle = h5py.File(path, "r+" if writable else "r")
        try:
            stored, schema = (
                handle.attrs.get("extract_hash"),
                handle.attrs.get("schema"),
            )
            if schema != SCHEMA:
                raise StoreMismatch(
                    f"{path} has schema {schema}, this code's is {SCHEMA}"
                )
            if extract_hash is not None and stored != extract_hash:
                raise StoreMismatch(
                    f"{path} was extracted under hash {stored}, this run has "
                    f"{extract_hash}. Its photons cannot be re-read once proc is "
                    "taped, so it is never replaced automatically"
                )
            return cls(handle, path, writable)
        except BaseException:
            handle.close()
            raise

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> RingPhotonStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── ledger ───────────────────────────────────────────────────────────────
    def status(self, train_ids: np.ndarray) -> np.ndarray:
        """Each train's status, or :data:`ABSENT`; a group without one is unfinished."""
        trains = self._f["trains"]
        out = np.full(len(train_ids), ABSENT, dtype=np.uint8)
        for i, train_id in enumerate(np.asarray(train_ids).tolist()):
            group = trains.get(_group_name(train_id))
            if group is not None:
                out[i] = group.attrs.get("status", FrameStatus.NOT_PROCESSED)
        return out

    def unfinished(self, train_ids: np.ndarray) -> np.ndarray:
        """Trains that are absent, crashed mid-write or ended in a retried status."""
        status = self.status(train_ids)
        retried = [int(code) for code in UNFINISHED] + [ABSENT]
        return np.asarray(train_ids)[np.isin(status, retried)]

    def status_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for group in self._f["trains"].values():
            name = FrameStatus(int(group.attrs.get("status", 255))).name
            counts[name] = counts.get(name, 0) + 1
        return counts

    def train_ids(self) -> np.ndarray:
        return np.array([int(name[1:]) for name in self._f["trains"]], dtype=np.uint64)

    # ── writing ──────────────────────────────────────────────────────────────
    def put(self, photons: TrainPhotons) -> bool:
        """Store a train unless it is already final.

        :returns: whether anything was written. A second ``put`` of a finished
            train is a no-op, so a block replayed after a crash costs nothing.
        """
        self._require_writable()
        if not self._clear_unfinished(_group_name(photons.train_id)):
            return False
        _write_train(self._f["trains"], photons)
        self._f.flush()
        return True

    def _clear_unfinished(self, name: str) -> bool:
        """Drop ``name`` if it is unfinished; False when it is final and must stay."""
        trains = self._f["trains"]
        if name in trains:
            if trains[name].attrs.get("status", ABSENT) not in (
                *[int(c) for c in UNFINISHED],
                ABSENT,
            ):
                return False
            del trains[name]
        return True

    def merge_block(self, block_file: Path) -> int:
        """Copy a block file's trains in, then delete the file.

        Each group is copied without attributes, checked against its source, and
        only then given its ``status``, so a group that carries one is complete
        and a crash leaves it unfinished. A train already final is skipped and
        an unfinished one replaced, as :meth:`put` does, so a replayed block
        writes nothing. The file is deleted only once every train reads back.

        :returns: how many trains were written.
        :raises StoreMismatch: the block file was extracted under another hash.
        :raises RuntimeError: a copied group does not read back as its source;
            the block file is kept.
        """
        self._require_writable()
        block_file = Path(block_file)
        written = 0
        with h5py.File(block_file, "r") as handle:
            if handle.attrs.get("extract_hash") != self.extract_hash:
                raise StoreMismatch(
                    f"{block_file} was extracted under hash "
                    f"{handle.attrs.get('extract_hash')}, the store's is "
                    f"{self.extract_hash}"
                )
            trains = self._f["trains"]
            for name in handle:
                source = handle[name]
                if "status" not in source.attrs:
                    raise RuntimeError(f"{block_file}: {name} has no status")
                if not self._clear_unfinished(name):
                    continue
                trains.copy(source, name, without_attrs=True)
                _check_copy(source, trains[name])
                for key, value in source.attrs.items():
                    if key != "status":
                        trains[name].attrs[key] = value
                trains[name].attrs["status"] = source.attrs["status"]
                written += 1
            self._f.flush()
            for name in handle:
                if trains[name].attrs.get("status") is None:
                    raise RuntimeError(f"{name} did not read back from {block_file}")
        block_file.unlink()
        return written

    def recover_blocks(self, folder: Path) -> int:
        """Take up what a crashed run left in ``folder``, and clear the rest.

        Call under the store's output lock. A ``.tmp`` is half a block and is
        removed. A finished block file is merged as :meth:`merge_block` would;
        one that cannot be read or belongs to another hash is renamed ``.bad``
        and logged, kept for a person and out of the next run's way.

        :returns: how many trains the leftovers gave the store.
        """
        folder = Path(folder)
        if not folder.is_dir():
            return 0
        for tmp in folder.glob("*.tmp"):
            tmp.unlink()
        recovered = 0
        for block_file in sorted(folder.glob("*.h5")):
            try:
                recovered += self.merge_block(block_file)
            except (OSError, StoreMismatch) as error:
                log.warning(
                    "block file %s not usable (%s); kept as .bad", block_file, error
                )
                block_file.rename(block_file.with_name(block_file.name + ".bad"))
        return recovered

    def mark(self, train_id: int, status: FrameStatus, message: str = "") -> None:
        """Record a status with no photons: a final one, or an error to retry."""
        self._require_writable()
        name = _group_name(train_id)
        trains = self._f["trains"]
        if name in trains:
            if trains[name].attrs.get("status", ABSENT) not in (
                *[int(c) for c in UNFINISHED],
                ABSENT,
            ):
                return
            del trains[name]
        group = trains.create_group(name)
        if message:
            group.attrs["error"] = message[:1000]
        group.attrs["status"] = int(status)
        self._f.flush()

    def record_pass(self, provenance: dict[str, Any]) -> None:
        """Append one pass to the store's history."""
        self._require_writable()
        attrs = self._f["provenance"].attrs
        history = json.loads(attrs.get("history", "[]"))
        history.append(provenance)
        attrs["history"] = json.dumps(history, default=str)
        self._f.flush()

    def _require_writable(self) -> None:
        if not self.writable:
            raise PermissionError(f"{self.path} is open read-only")

    # ── reading ──────────────────────────────────────────────────────────────
    def envelope(self) -> np.ndarray:
        return self._f["envelope/pixels"][()]

    def get(self, train_id: int) -> TrainPhotons:
        """One train. A train without a finished group is an error, not a guess."""
        group = self._f["trains"].get(_group_name(train_id))
        if group is None or "status" not in group.attrs:
            raise KeyError(f"train {train_id} is not in {self.path}")
        status = FrameStatus(int(group.attrs["status"]))
        sequence = int(group.attrs.get("sequence", -1))
        if status is not FrameStatus.OK:
            return TrainPhotons.failed(train_id, status, sequence)
        frame_ptr = group["frame_ptr"][()].astype(np.int64)
        return TrainPhotons(
            train_id=train_id,
            status=status,
            sequence=sequence,
            pulse_ids=group["pulse_ids"][()],
            cell_ids=group["cell_ids"][()],
            frame_ptr=frame_ptr,
            pixel=delta_decode(group["pixel_step"][()], frame_ptr),
            count=group["count"][()],
            flag_ptr=group["flag_ptr"][()].astype(np.int64),
            flag_pixel=group["flag_pixel"][()],
            base=self.base,
        )

    def sequence(self, train_id: int) -> int:
        return int(self._f["trains"][_group_name(train_id)].attrs.get("sequence", -1))

    def pulse_ids(self, train_id: int) -> np.ndarray:
        return self._f["trains"][_group_name(train_id)]["pulse_ids"][()]
