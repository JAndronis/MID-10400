"""``damnit_names.stored``: which name a results file holds a product under."""

from __future__ import annotations

import pytest

from analysis.damnit_names import AGIPD_SAXS, JF1, LegacyNameUsed, stored


def test_the_current_name_wins_when_a_file_holds_both_generations():
    assert stored({"agipd.saxs", "agipd_saxs"}, AGIPD_SAXS) == "agipd.saxs"


def test_a_file_with_only_the_flat_name_resolves_to_it_and_says_so():
    with pytest.warns(LegacyNameUsed):
        assert stored({"jungfrau_waxs_jf1"}, JF1) == "jungfrau_waxs_jf1"


def test_a_missing_product_reports_the_name_now_in_use():
    assert stored(set(), "agipd_saxs") == "agipd.saxs"


def test_a_flat_name_asked_for_is_translated_first():
    assert stored({"jungfrau.jf1", "jungfrau_waxs_jf1"}, "jungfrau_waxs_jf1") == (
        "jungfrau.jf1"
    )
