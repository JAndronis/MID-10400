"""Sample identity: the corrections to DAMNIT's labels, and nothing more."""

from __future__ import annotations

from analysis.triage.samples import OVERRIDES, effective_sample, override_for


def test_an_override_covers_its_first_and_last_run_and_no_other():
    """r289-303 are PEG 6K droplets labelled 'Silica NP' (the user, 2026-10-01)."""
    for run in (289, 303):
        s = effective_sample(run, "Silica NP")
        assert (s.name, s.label) == ("ferritin50_peg6k", "Silica NP")
        assert "user" in s.source
    for run in (288, 304):
        s = effective_sample(run, "Silica NP")
        assert (s.name, s.source) == ("silica_np", "DAMNIT label")


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
