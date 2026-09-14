"""Config hashing and by-name pickling, shared by both passes.

The hash gates resume and gates reopening a file, so it covers only fields that
can change a stored number; :data:`OPERATIONAL_FIELDS` is what it excludes, and
a file records the set it was written under in its own provenance rather than
relying on this constant.

Both configs are frozen slots dataclasses sent to a spawned pool, where the
dataclass default pickles state *by position* — so a class whose field list has
moved assigns every later value to the wrong field, silently.
:func:`state_by_name` and :func:`restore_by_name` key the state by name instead,
and each config must call them from its own class body: ``dataclasses._add_slots``
reinstalls the positional pair unless ``__getstate__`` is in the class's own
``__dict__``, so a mixin does not work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from typing import Any

from analysis.common.cpu import file_sha256

__all__ = [
    "OPERATIONAL_FIELDS",
    "ConfigStateMismatch",
    "config_sha256",
    "restore_by_name",
    "result_fields",
    "state_by_name",
]

#: Config fields that change how a pass runs but not what it stores, and which
#: are therefore excluded from :func:`config_sha256`. A pass may exclude more;
#: read a stored file's set off ``config_operational_fields`` in its provenance,
#: never off this constant.
#:
#: Deliberately *absent*: ``base_mask_trains`` and ``cell_sample_trains`` choose
#: which trains are sampled, so they move the masks and the measured readout
#: noise that every row depends on. They look operational and are not.
OPERATIONAL_FIELDS: frozenset[str] = frozenset(
    {
        "n_workers",
        "trains_per_block",
        "selftest_frames",
        "output_root",
        "allow_incomplete",
        "overwrite",
    }
)


def result_fields(
    cfg: Any, operational: frozenset[str] = OPERATIONAL_FIELDS
) -> dict[str, Any]:
    """Every config field that can change a stored number, JSON-ready.

    ``frozenset`` fields are sorted and tuples become lists, so the rendering is
    stable across processes and Python versions.
    """
    payload: dict[str, Any] = {}
    for key, value in asdict(cfg).items():
        if key in operational:
            continue
        if isinstance(value, frozenset | set):
            value = sorted(value)
        elif isinstance(value, tuple):
            value = list(value)
        payload[key] = value
    return payload


def config_sha256(
    cfg: Any,
    input_files: dict[str, str | None],
    operational: frozenset[str] = OPERATIONAL_FIELDS,
) -> str:
    """sha256 over the result-affecting fields plus each input file's sha256."""
    payload = result_fields(cfg, operational)
    payload["input_file_sha256"] = {
        name: file_sha256(path) if path else None for name, path in input_files.items()
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


#: Appended to every :class:`ConfigStateMismatch`: the cause is almost always
#: one process holding an older import of the config module than the other.
_SKEW_HINT = (
    "The process that pickled this config and the one unpickling it are "
    "running different versions of its module — typically a notebook kernel "
    "that imported it before an edit while the workers it spawns import the "
    "current file. Restart the parent process."
)


class ConfigStateMismatch(RuntimeError):
    """A pickled config and the class unpickling it disagree on the fields."""


def state_by_name(cfg: Any) -> dict[str, Any]:
    """``__getstate__`` for a frozen slots config: one entry per field, by name."""
    return {field.name: getattr(cfg, field.name) for field in fields(cfg)}


def restore_by_name(cfg: Any, state: Any) -> None:
    """``__setstate__`` for a frozen slots config: assign by name, then validate.

    :raises ConfigStateMismatch: ``state`` is a positional list, or its field
        names are not exactly this class's.
    """
    name = type(cfg).__name__
    if not isinstance(state, dict):
        size = len(state) if isinstance(state, list | tuple) else "?"
        raise ConfigStateMismatch(
            f"{name} arrived as {size} values pickled by position, which is "
            "what a process running an older version of this module sends. "
            "Nothing in that state says which value is which, so none of it "
            f"can be trusted. {_SKEW_HINT}"
        )
    names = [field.name for field in fields(cfg)]
    missing = [key for key in names if key not in state]
    unknown = [key for key in state if key not in set(names)]
    if missing or unknown:
        raise ConfigStateMismatch(
            f"{name} was pickled by a class with a different field list: "
            f"missing here {missing}, unknown here {unknown}. {_SKEW_HINT}"
        )
    for key in names:
        object.__setattr__(cfg, key, state[key])
    # ``__setstate__`` bypasses ``__init__``, so nothing has validated these
    # values. Re-running ``__post_init__`` makes a worker refuse a bad config in
    # the config's own words, not at whatever line first uses the bad field.
    post_init = getattr(cfg, "__post_init__", None)
    if post_init is not None:
        post_init()
