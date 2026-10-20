"""Reading a written file back: every chunk of every dataset."""

from __future__ import annotations

import h5py
import numpy as np

from analysis.common import readback
from analysis.common.readback import unreadable


def write(path):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "sums/a", data=np.arange(4000).reshape(40, 100), chunks=(4, 100),
            compression="gzip", shuffle=True,
        )  # fmt: skip
        f.create_dataset("b", data=np.arange(3))
        f.create_dataset("empty", shape=(0, 5), dtype=np.int32)


def test_a_sound_file_reads_back(tmp_path):
    write(tmp_path / "x.h5")
    assert unreadable(tmp_path / "x.h5") == []


def test_one_corrupt_chunk_is_named_even_when_slabs_span_many(tmp_path, monkeypatch):
    path = tmp_path / "x.h5"
    write(path)
    with h5py.File(path, "r") as f:
        info = f["sums/a"].id.get_chunk_info_by_coord((36, 0))
    with open(path, "r+b") as raw:
        raw.seek(info.byte_offset)
        raw.write(b"\xff" * info.size)
    for read_bytes in (1, 1 << 28):
        monkeypatch.setattr(readback, "READ_BYTES", read_bytes)
        failures = unreadable(path)
        assert len(failures) == 1 and failures[0].startswith("sums/a:")


def test_a_file_that_does_not_open_is_reported(tmp_path):
    (tmp_path / "x.h5").write_bytes(b"not hdf5")
    assert unreadable(tmp_path / "x.h5")[0].startswith("opening:")
