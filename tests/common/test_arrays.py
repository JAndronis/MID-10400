"""``atomic_savez``: a reader sees the old archive or the new one, never half."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.common import arrays
from analysis.common.arrays import atomic_savez


def test_the_archive_lands_at_exactly_the_path_given(tmp_path):
    path = tmp_path / "operator.npz"
    assert atomic_savez(path, a=np.arange(3)) == path
    with np.load(path) as loaded:
        np.testing.assert_array_equal(loaded["a"], np.arange(3))
    assert [p.name for p in tmp_path.iterdir()] == ["operator.npz"]


def test_a_failed_write_leaves_the_old_archive_and_no_debris(tmp_path, monkeypatch):
    path = tmp_path / "masks.npz"
    atomic_savez(path, a=np.zeros(2))

    def dies_half_way(handle, **arrays_):
        handle.write(b"PK\x03\x04 truncated")
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(arrays.np, "savez", dies_half_way)
    with pytest.raises(OSError, match="quota"):
        atomic_savez(path, a=np.ones(2))
    monkeypatch.undo()
    with np.load(path) as loaded:
        np.testing.assert_array_equal(loaded["a"], np.zeros(2))
    assert [p.name for p in tmp_path.iterdir()] == ["masks.npz"]
