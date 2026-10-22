"""Sample identity: the corrections to DAMNIT's labels, and nothing more."""

from __future__ import annotations

import pytest

from analysis.triage.samples import OVERRIDES, effective_sample, override_for


@pytest.mark.parametrize(
    ("run", "label", "expected"),
    [
        (289, "Silica NP", "ferritin50_peg6k"),
        (303, "Silica NP", "ferritin50_peg6k"),
        (308, "Ferritin_50_PEG6K_5p", "ferritin50"),
        (335, "Silica NP", "ferritin50"),
        (464, "Ferritin_50_PEG6K_5p", "peg6k_buffer"),
    ],
)
def test_stale_labels_are_corrected(run, label, expected):
    s = effective_sample(run, label)
    assert s.name == expected and s.label == label and "user" in s.source


@pytest.mark.parametrize(
    ("run", "label", "expected"),
    [
        (382, "Silica NP", "silica_np"),  # inside the ferritin block, still silica
        (389, "Silica NP", "silica_np"),  # cut short; r390 is a new droplet
        (336, "Ferritin_50", "ferritin50"),
        (68, "Silica NP", "silica_np"),  # confirmed, so not an override
        (423, "Ferritin_50_PEG6K_5p", "ferritin50_peg6k"),
    ],
)
def test_other_runs_keep_their_label(run, label, expected):
    s = effective_sample(run, label)
    assert s.name == expected and s.source == "DAMNIT label"


def test_background_roles_and_exclusions_travel_with_the_override():
    assert effective_sample(465, "x").background_role == "saxs_buffer_peg6k"
    assert effective_sample(55, "Air").background_role == "empty_beam"
    assert "explodes" in effective_sample(463, "x").exclude
    assert effective_sample(423, "Ferritin_50_PEG6K_5p").background_role is None


def test_an_unknown_label_passes_through_unguessed():
    s = effective_sample(450, "Something new")
    assert s.name == "Something new" and "not canonicalised" in s.source
    assert effective_sample(450, None).name == "unknown"


def test_no_run_is_covered_twice():
    for run in range(1, 600):
        override_for(run)  # raises on overlap
    assert all(o.first <= o.last for o in OVERRIDES)
