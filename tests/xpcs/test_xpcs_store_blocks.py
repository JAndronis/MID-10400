"""Block files: workers write them, the parent alone merges them into the store."""

from __future__ import annotations

import hashlib

import h5py
import numpy as np
import pytest
from xpcs_fakes import make_dense, make_flags, make_train

from analysis.common.status import FrameStatus
from analysis.xpcs import store as store_module
from analysis.xpcs.store import (
    FlagBase,
    RingPhotonStore,
    TrainPhotons,
    block_dir_for,
    write_block_file,
)

N_ENV, P = 300, 6
HASH = "a" * 64
FIELDS = (
    "pulse_ids",
    "cell_ids",
    "frame_ptr",
    "pixel",
    "count",
    "flag_ptr",
    "flag_pixel",
)


@pytest.fixture
def envelope():
    return np.arange(N_ENV, dtype=np.int32) * 3


@pytest.fixture
def path(tmp_path):
    return tmp_path / "r0001" / "agipd_ring_photons.h5"


@pytest.fixture
def store(path, envelope):
    with RingPhotonStore.create(path, HASH, envelope, FlagBase.empty()) as s:
        yield s


def train(train_id, seed=0):
    rng = np.random.default_rng(seed or train_id)
    flags = make_flags(rng, P, N_ENV)
    return make_train(train_id, make_dense(rng, P, N_ENV), flags, sequence=2)


def block(tmp_path, name, trains, extract_hash=HASH):
    out = tmp_path / "blocks" / name
    write_block_file(out, extract_hash, trains)
    return out


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def stored_sizes(file, ids):
    with h5py.File(file, "r") as f:
        return {
            (g, k): ds.id.get_storage_size()
            for g in (f"trains/t{i:012d}" for i in ids)
            for k, ds in f[g].items()
        }


def same_train(a: TrainPhotons, b: TrainPhotons):
    assert (a.status, a.sequence) == (b.status, b.sequence)
    for name in FIELDS:
        assert getattr(a, name).dtype == getattr(b, name).dtype
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_a_merged_block_holds_its_trains_and_their_compressed_bytes(tmp_path, store):
    trains = [train(1), train(2), TrainPhotons.failed(3, FrameStatus.LABEL_MISMATCH, 4)]
    file = block(tmp_path, "b0.h5", trains)
    with h5py.File(file, "r") as f:
        in_block = {
            (f"trains/{g}", k): ds.id.get_storage_size()
            for g in f
            for k, ds in f[g].items()
        }
    assert store.merge_block(file) == 3
    for t in trains:
        same_train(store.get(t.train_id), t)
    store._f.flush()
    assert stored_sizes(store.path, [1, 2, 3]) == in_block
    assert not file.exists()


def test_writing_a_block_file_never_touches_the_store(tmp_path, path, store):
    store.merge_block(block(tmp_path, "b1.h5", [train(1)]))
    store._f.flush()
    before = sha(path)
    block(tmp_path, "b0.h5", [train(2)])
    assert sha(path) == before


def test_a_block_file_appears_whole_or_not_at_all(tmp_path, monkeypatch):
    file = block(tmp_path, "b0.h5", [train(1)])
    assert file.exists() and not list(file.parent.glob("*.tmp"))
    file.unlink()

    def boom(group, photons):
        raise OSError("disk full")

    monkeypatch.setattr(store_module, "_write_train", boom)
    with pytest.raises(OSError):
        block(tmp_path, "b0.h5", [train(1)])
    assert not list((tmp_path / "blocks").glob("*"))


def test_a_replayed_block_writes_nothing(tmp_path, store):
    first = block(tmp_path, "b0.h5", [train(1, seed=1)])
    assert store.merge_block(first) == 1
    replay = block(tmp_path, "b0.h5", [train(1, seed=2)])
    assert store.merge_block(replay) == 0
    same_train(store.get(1), train(1, seed=1))
    assert not replay.exists()


def test_a_final_status_survives_a_block_that_claims_otherwise(tmp_path, store):
    store.mark(1, FrameStatus.DATA_CHECK_FAILED)
    assert store.merge_block(block(tmp_path, "b0.h5", [train(1)])) == 0
    assert store.status(np.array([1], dtype=np.uint64)).tolist() == [
        FrameStatus.DATA_CHECK_FAILED
    ]


def test_unfinished_groups_are_replaced_by_the_block(tmp_path, store):
    store.mark(1, FrameStatus.WORKER_ERROR, "boom")
    store._f["trains"].create_group("t000000000002")  # a writer died mid-train
    store._f["trains/t000000000002"].create_dataset("count", data=np.ones(3, np.uint8))
    group = store._f["trains"].create_group("t000000000003")  # a copy died half way
    group.create_dataset("pixel_step", data=np.zeros(2, np.uint32))
    file = block(tmp_path, "b0.h5", [train(1), train(2), train(3)])
    assert store.merge_block(file) == 3
    ids = np.array([1, 2, 3], dtype=np.uint64)
    assert store.unfinished(ids).size == 0
    assert "error" not in store._f["trains/t000000000001"].attrs
    same_train(store.get(2), train(2))
    same_train(store.get(3), train(3))


def test_status_is_set_only_after_the_copy_checked_out(tmp_path, store, monkeypatch):
    file = block(tmp_path, "b0.h5", [train(1), train(2)])

    def refuse(source, copied):
        raise RuntimeError("read back differs")

    monkeypatch.setattr(store_module, "_check_copy", refuse)
    with pytest.raises(RuntimeError, match="read back"):
        store.merge_block(file)
    assert file.exists()
    ids = np.array([1, 2], dtype=np.uint64)
    assert store.unfinished(ids).tolist() == [1, 2]
    with pytest.raises(KeyError):
        store.get(1)
    monkeypatch.undo()
    assert store.merge_block(file) == 2
    same_train(store.get(1), train(1))


def test_recovery_merges_finished_leftovers_and_clears_the_rest(tmp_path, path, store):
    folder = block_dir_for(path)
    write_block_file(folder / "block_000001.h5", HASH, [train(1)])
    write_block_file(
        folder / "block_000002.h5",
        HASH,
        [train(2)],
    )
    (folder / "block_000003.h5.tmp").write_bytes(b"half a block")
    write_block_file(folder / "block_000004.h5", "b" * 64, [train(4)])
    (folder / "block_000005.h5").write_bytes(b"not hdf5")
    assert store.recover_blocks(tmp_path / "nothing") == 0
    assert store.recover_blocks(folder) == 2
    assert sorted(store.train_ids().tolist()) == [1, 2]
    assert not list(folder.glob("*.tmp"))
    assert not list(folder.glob("block_00000[12].h5"))
    # what could not be read back is kept for a person, out of the way of the next run
    assert sorted(p.name for p in folder.iterdir()) == [
        "block_000004.h5.bad",
        "block_000005.h5.bad",
    ]
