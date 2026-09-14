"""The mock-run cache must key on what was asked for, not on how it was asked.

Each mock run is tens to hundreds of MB that lives until the session ends, so a
cache that misses on an explicitly-passed default writes a second identical copy
of the same run. Both suites used to have their own factory and only one of them
got this right.
"""

from __future__ import annotations


def test_passing_a_default_explicitly_hits_the_cache(memoised_run_factory, tmp_path):
    written = []

    def write_mock_run(root, *, trains=3, detector_name="DET", cells=(0, 1)):
        written.append(root)
        return f"run-{trains}-{detector_name}-{tuple(cells)}"

    factory = memoised_run_factory(write_mock_run, "probe", open_run=lambda root: root)

    implicit = factory()
    explicit = factory(trains=3, detector_name="DET", cells=(0, 1))

    assert implicit[0] == explicit[0]
    assert len(written) == 1, "an explicitly-passed default wrote a second run"


def test_a_different_request_is_a_different_run(memoised_run_factory):
    written = []

    def write_mock_run(root, *, trains=3):
        written.append(root)
        return f"run-{trains}"

    factory = memoised_run_factory(write_mock_run, "probe", open_run=lambda root: root)
    factory()
    factory(trains=9)
    assert len(written) == 2


def test_a_list_argument_is_still_hashable(memoised_run_factory):
    """Keys are built from the bound arguments, which may hold lists."""
    written = []

    def write_mock_run(root, *, cells=None):
        written.append(root)
        return "run"

    factory = memoised_run_factory(write_mock_run, "probe", open_run=lambda root: root)
    factory(cells=[0, 1, 2])
    factory(cells=[0, 1, 2])
    assert len(written) == 1
