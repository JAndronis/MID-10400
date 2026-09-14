"""How a config survives the trip to a spawned worker, for both passes.

Every pass pickles its config into a spawned pool, and both configs are frozen
slots dataclasses, which ``dataclasses`` pickles *by position* unless told
otherwise. A field added to either class then shifts every later field of a
state pickled by an older import, silently: see :mod:`analysis.common.config`
for the one that reached a real run. These tests pin the two properties that
prevent it — the state is keyed by name, and a state whose names are not this
class's is refused rather than partly applied.
"""

from __future__ import annotations

import dataclasses
import pickle

import pytest

from analysis.common.config import ConfigStateMismatch

pytest.importorskip("pyFAI")

from analysis.saxs.config import AgipdSaxsConfig  # noqa: E402
from analysis.waxs.config import JungfrauWaxsConfig  # noqa: E402


@pytest.fixture
def agipd():
    return AgipdSaxsConfig(
        proposal=10400, run=423, geometry_file=None, pixel_mask_file=None
    )


@pytest.fixture
def waxs():
    return JungfrauWaxsConfig(
        proposal=10400,
        run=423,
        detector="jf1",
        poni_file=None,
        static_mask_file=None,
    )


@pytest.fixture(params=["agipd", "waxs"])
def cfg(request):
    """Both passes' configs, so neither can be hardened without the other."""
    return request.getfixturevalue(request.param)


def field_names(cfg) -> list[str]:
    return [field.name for field in dataclasses.fields(cfg)]


def test_the_state_is_keyed_by_name_not_by_position(cfg):
    """The property the whole fix rests on: a name cannot shift, an index can."""
    state = cfg.__getstate__()
    assert isinstance(state, dict)
    assert sorted(state) == sorted(field_names(cfg))


def test_a_roundtrip_preserves_every_field(cfg):
    back = pickle.loads(pickle.dumps(cfg))
    assert back == cfg
    for name in field_names(cfg):
        assert getattr(back, name) == getattr(cfg, name), name


def test_a_state_missing_a_field_is_refused(cfg):
    """The reported failure: the parent's class predates a field this one has.

    Filling it from its default would be the silent behaviour — the whole point
    is that nobody knows what the older process meant by it.
    """
    state = cfg.__getstate__()
    del state["min_modules"]
    with pytest.raises(ConfigStateMismatch, match=r"missing here \['min_modules'\]"):
        pickle.loads(pickle.dumps(cfg)).__setstate__(state)


def test_a_state_carrying_an_unknown_field_is_refused(cfg):
    """The other direction: the parent's class is *newer* than this one."""
    state = cfg.__getstate__() | {"future_field": 1}
    with pytest.raises(ConfigStateMismatch, match=r"unknown here \['future_field'\]"):
        pickle.loads(pickle.dumps(cfg)).__setstate__(state)


def test_a_positionally_pickled_state_is_refused(cfg):
    """What a process running the pre-2026-09-14 module actually sends.

    A list of values with no names on it, which this class must not guess at
    however plausibly its own field count happens to match.
    """
    positional = [getattr(cfg, name) for name in field_names(cfg)]
    with pytest.raises(ConfigStateMismatch, match="pickled by position"):
        pickle.loads(pickle.dumps(cfg)).__setstate__(positional)


def test_the_reported_shift_raises_instead_of_assigning_none(waxs):
    """The exact 2026-09-14 skew, reproduced: 26 values arriving at 28 fields.

    Before the fix this assigned ``n_workers``' ``None`` to ``min_modules`` and
    ``cell_sample_trains``' ``8`` to ``max_abs_kev``, and the only complaint was
    a ``TypeError`` from inside ``extra_data``.
    """
    names = field_names(waxs)
    dropped = {"lit_gap_ratio", "read_noise_fallback_kev"}
    assert dropped < set(names), "this test describes a skew in these two fields"
    old_state = [getattr(waxs, name) for name in names if name not in dropped]
    assert len(old_state) == len(names) - 2

    restored = pickle.loads(pickle.dumps(waxs))
    with pytest.raises(ConfigStateMismatch, match="pickled by position"):
        restored.__setstate__(old_state)


def test_unpickling_revalidates(cfg):
    """``__setstate__`` bypasses ``__init__``, so it must run the checks itself.

    Without this a worker takes any value the parent's class would have refused
    and fails wherever that field is first used, in whatever library owns that
    line — which is how ``min_modules=None`` surfaced as a ``TypeError`` inside
    ``extra_data.components``.
    """
    state = cfg.__getstate__() | {"min_modules": 0}
    with pytest.raises(ValueError, match="min_modules must be positive"):
        pickle.loads(pickle.dumps(cfg)).__setstate__(state)

    state = cfg.__getstate__() | {"npt": 0}
    with pytest.raises(ValueError, match="npt must be positive"):
        pickle.loads(pickle.dumps(cfg)).__setstate__(state)


def test_a_config_is_still_hashable_and_comparable_after_a_roundtrip(cfg):
    """The dataclass machinery is untouched; only the state encoding changed."""
    back = pickle.loads(pickle.dumps(cfg))
    assert dataclasses.replace(back, run=424).run == 424
    assert back.config_hash() == cfg.config_hash()
    assert back.operational_fields == cfg.operational_fields
