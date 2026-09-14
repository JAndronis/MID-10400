"""The worker default must not silently decide whether to use hyperthreading.

A node exposing 72 logical CPUs for 36 physical cores would, if the default
came from the affinity mask, run the hyperthreaded configuration while claiming
one worker per core. Whether that is the right configuration is an open
question, and the default must not answer it.
"""

from __future__ import annotations

import os

from analysis.common import config as common_config
from analysis.common.cpu import physical_cores
from analysis.saxs.config import AgipdSaxsConfig


def _fake_topology(root, siblings):
    """Write a sysfs-shaped tree: ``siblings`` maps cpu id to its group."""
    for cpu, group in siblings.items():
        topology = root / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "thread_siblings_list").write_text(f"{group}\n")


def test_physical_cores_counts_a_hyperthreaded_core_once(tmp_path, monkeypatch):
    """36 physical cores exposed as 72 logical ones must count as 36."""
    siblings = {}
    for core in range(36):
        siblings[core] = f"{core},{core + 36}"
        siblings[core + 36] = f"{core},{core + 36}"
    _fake_topology(tmp_path, siblings)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(siblings), raising=False)

    assert physical_cores(tmp_path) == 36


def test_physical_cores_respects_an_affinity_mask(tmp_path, monkeypatch):
    """A job pinned to half the node is not told about the other half."""
    siblings = {}
    for core in range(4):
        siblings[core] = f"{core},{core + 4}"
        siblings[core + 4] = f"{core},{core + 4}"
    _fake_topology(tmp_path, siblings)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1, 4, 5}, raising=False)

    assert physical_cores(tmp_path) == 2


def test_physical_cores_is_none_without_sysfs(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    assert physical_cores(tmp_path / "absent") is None


def test_workers_defaults_to_physical_cores(monkeypatch):
    monkeypatch.setattr(common_config, "physical_cores", lambda *a, **k: 36)
    cfg = AgipdSaxsConfig(
        proposal=1, run=1, geometry_file=None, pixel_mask_file=None, n_workers=None
    )
    assert cfg.workers == 36


def test_explicit_n_workers_still_wins(monkeypatch):
    monkeypatch.setattr(common_config, "physical_cores", lambda *a, **k: 36)
    cfg = AgipdSaxsConfig(
        proposal=1, run=1, geometry_file=None, pixel_mask_file=None, n_workers=72
    )
    assert cfg.workers == 72
