"""``physical_cores`` counts cores, not hyperthreads, within the affinity mask.

The default worker count comes from it. A node exposing 72 logical CPUs for 36
physical cores would otherwise run one worker per hyperthread while claiming one
per core.
"""

from __future__ import annotations

import os

from analysis.common.cpu import physical_cores


def fake_topology(root, cores: int):
    """Two hyperthreads per core: cpu ``c`` and ``c + cores`` are siblings."""
    for core in range(cores):
        group = f"{core},{core + cores}\n"
        for cpu in (core, core + cores):
            topology = root / f"cpu{cpu}" / "topology"
            topology.mkdir(parents=True)
            (topology / "thread_siblings_list").write_text(group)


def test_a_hyperthreaded_core_counts_once(tmp_path, monkeypatch):
    fake_topology(tmp_path, 36)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(72)))
    assert physical_cores(tmp_path) == 36


def test_only_the_cores_in_the_affinity_mask_count(tmp_path, monkeypatch):
    fake_topology(tmp_path, 4)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1, 4, 5})
    assert physical_cores(tmp_path) == 2


def test_no_topology_is_no_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1})
    assert physical_cores(tmp_path / "absent") is None
