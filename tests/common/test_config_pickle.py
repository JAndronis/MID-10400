"""How a config survives the trip to a spawned worker, for both passes.

Every pass pickles its config into a spawned pool, and both configs are frozen
slots dataclasses, which ``dataclasses`` pickles *by position* unless told
otherwise. A field added to either class then silently shifts every later
field of a state pickled by an older import. These tests pin the two properties
that prevent it: the state is keyed by name, and a state whose names are not
this class's is refused rather than partly applied.
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


def test_a_roundtrip_preserves_every_field(cfg):
    back = pickle.loads(pickle.dumps(cfg))
    assert back == cfg
    for name in field_names(cfg):
        assert getattr(back, name) == getattr(cfg, name), name


def test_a_state_missing_a_field_is_refused(cfg):
    """The parent's class predates a field this one has.

    Filling it from its default would hide the skew, and nobody knows what the
    older process meant by it.
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
    """A list of values with no names, as the dataclass default pickles a state.

    This class must not guess at it, however plausibly its own field count
    happens to match.
    """
    positional = [getattr(cfg, name) for name in field_names(cfg)]
    with pytest.raises(ConfigStateMismatch, match="pickled by position"):
        pickle.loads(pickle.dumps(cfg)).__setstate__(positional)


def test_unpickling_revalidates(cfg):
    """``__setstate__`` bypasses ``__init__``, so it must run the checks itself.

    Without this a worker takes any value the parent's class would have refused
    and fails wherever that field is first used, in whatever library owns that
    line (``min_modules=None`` fails as a ``TypeError`` inside
    ``extra_data.components``).
    """
    state = cfg.__getstate__() | {"min_modules": 0}
    with pytest.raises(ValueError, match="min_modules must be positive"):
        pickle.loads(pickle.dumps(cfg)).__setstate__(state)

    state = cfg.__getstate__() | {"npt": 0}
    with pytest.raises(ValueError, match="npt must be positive"):
        pickle.loads(pickle.dumps(cfg)).__setstate__(state)
