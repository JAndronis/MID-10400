"""The photon store file: what survives a crash, a replay and a foreign config.

Trains go in as the pass puts them, through a block file and ``merge_block``.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from xpcs_fakes import make_dense, make_flags, make_train, merge_trains

from analysis.common.status import FrameStatus
from analysis.xpcs.store import (
    ABSENT,
    FlagBase,
    RingPhotonStore,
    StoreMismatch,
    TrainPhotons,
)

N_ENV, P = 300, 6
HASH = "a" * 64


@pytest.fixture
def envelope():
    return np.arange(N_ENV, dtype=np.int32) * 3


@pytest.fixture
def path(tmp_path):
    return tmp_path / "r0001" / "agipd_ring_photons.h5"


def train(train_id, seed=0, base=None):
    rng = np.random.default_rng(seed)
    flags = make_flags(rng, P, N_ENV)
    photons = make_train(train_id, make_dense(rng, P, N_ENV), flags, sequence=2)
    return photons, flags


def test_a_train_comes_back_as_it_went_in(path, envelope):
    photons, flags = train(10)
    with RingPhotonStore.create(path, HASH, envelope, FlagBase.empty()) as store:
        assert merge_trains(store, [photons]) == 1
        back = store.get(10)
    assert back.status is FrameStatus.OK and back.sequence == 2
    for name in ("pulse_ids", "cell_ids", "frame_ptr", "pixel", "count", "flag_ptr"):
        np.testing.assert_array_equal(getattr(back, name), getattr(photons, name))
    np.testing.assert_array_equal(
        back.frames(N_ENV).toarray(), photons.frames(N_ENV).toarray()
    )
    for frame, expected in enumerate(flags):
        np.testing.assert_array_equal(back.flagged_frame(frame), expected)


def test_the_envelope_and_the_flag_base_travel_with_the_file(path, envelope):
    photons, flags = train(10)
    base = FlagBase.from_samples([(photons.cell_ids, flags)])
    RingPhotonStore.create(path, HASH, envelope, base).close()
    with RingPhotonStore.open(path, HASH) as store:
        np.testing.assert_array_equal(store.envelope(), envelope)
        np.testing.assert_array_equal(store.base.pixel, base.pixel)
        np.testing.assert_array_equal(store.base.cells, base.cells)
        assert store.n_env == N_ENV


def test_a_train_that_gave_no_photons_is_stored_with_its_status(path, envelope):
    with RingPhotonStore.create(path, HASH, envelope, FlagBase.empty()) as store:
        merge_trains(store, [TrainPhotons.failed(7, FrameStatus.LABEL_MISMATCH, 3)])
        back = store.get(7)
        assert back.status is FrameStatus.LABEL_MISMATCH and back.sequence == 3
        assert back.n_frames == 0
        assert store.status(np.array([7])).tolist() == [FrameStatus.LABEL_MISMATCH]


def test_only_absent_crashed_and_retried_trains_are_unfinished(path, envelope):
    ids = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
    with RingPhotonStore.create(path, HASH, envelope, FlagBase.empty()) as store:
        merge_trains(store, [train(1)[0]])  # finished
        store.mark(2, FrameStatus.DATA_CHECK_FAILED)  # final, a property of the data
        store.mark(3, FrameStatus.WORKER_ERROR, "boom")  # to be retried
        store._f["trains"].create_group("t000000000004")  # a writer died mid-train
        assert store.status(ids).tolist() == [
            0,
            4,
            5,
            FrameStatus.NOT_PROCESSED,
            ABSENT,
        ]
        assert store.unfinished(ids).tolist() == [3, 4, 5]


def test_a_final_status_is_not_overwritten_by_a_later_error(path, envelope):
    with RingPhotonStore.create(path, HASH, envelope, FlagBase.empty()) as store:
        merge_trains(store, [train(1)[0]])
        store.mark(1, FrameStatus.WORKER_ERROR, "late")
        assert store.status(np.array([1], dtype=np.uint64)).tolist() == [0]


def test_trains_extracted_in_separate_sessions_form_one_store(path, envelope):
    RingPhotonStore.create(path, HASH, envelope, FlagBase.empty()).close()
    for ids in ([5, 9], [7]):
        with RingPhotonStore.open(path, HASH, writable=True) as store:
            merge_trains(store, [train(tid, seed=tid)[0] for tid in ids])
    with RingPhotonStore.open(path, HASH) as store:
        assert sorted(store.train_ids().tolist()) == [5, 7, 9]
        assert store.get(7).n_frames == P


def test_a_store_extracted_under_another_hash_is_refused_untouched(path, envelope):
    RingPhotonStore.create(path, HASH, envelope, FlagBase.empty()).close()
    before = path.read_bytes()
    with pytest.raises(StoreMismatch, match="never replaced"):
        RingPhotonStore.open(path, "b" * 64, writable=True)
    assert path.read_bytes() == before


def test_a_reader_cannot_write(path, envelope):
    RingPhotonStore.create(path, HASH, envelope, FlagBase.empty()).close()
    with RingPhotonStore.open(path, HASH) as store, pytest.raises(PermissionError):
        store.mark(1, FrameStatus.WORKER_ERROR)


def test_a_file_of_another_schema_is_refused(path, envelope):
    RingPhotonStore.create(path, HASH, envelope, FlagBase.empty()).close()
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema"] = 99
    with pytest.raises(StoreMismatch, match="schema"):
        RingPhotonStore.open(path)


def test_a_train_the_store_never_finished_cannot_be_read(path, envelope):
    with RingPhotonStore.create(path, HASH, envelope, FlagBase.empty()) as store:
        store._f["trains"].create_group("t000000000001")
        with pytest.raises(KeyError):
            store.get(1)
        with pytest.raises(KeyError):
            store.get(2)
