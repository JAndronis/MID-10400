"""What a config hash is for — and what it must not cover.

The hash on an output file answers one question: *were these rows computed from
the same inputs under the same rules?* It gates resume and it gates reopening a
file, so anything it covers that does not change a stored number makes two
identical results look incompatible.

Six fields do exactly that. ``n_workers`` and ``trains_per_block`` schedule the
work, ``selftest_frames`` sizes a gate, ``output_root`` chooses where the bytes
land, ``allow_incomplete`` decides whether to raise at the end, and ``overwrite``
decides whether an existing file may be replaced. None of them can change a
single value in ``/frames``.

Leaving them in had two consequences, both found on the first attempt to run the
DAMNIT variables: a pass run with ``n_workers=36`` wrote a file that the same
pass with the default worker count refused, and — worse — a file written with
``overwrite=True`` could never match a later ``overwrite=False`` run, so it was
refused every single time.

They are still recorded in full in the provenance record, which is where
"how was this run" belongs. Only the hash narrows.

A pass may exclude *more* than these six, and the JUNGFRAU one does: see
:data:`analysis.waxs.config.WAXS_OPERATIONAL_FIELDS`. So the set a file was
written under is asked of the config — ``cfg.operational_fields`` — and stored
in provenance alongside the hash, rather than being read back off this constant
by a reader who may have a different version of it to hand.

Why both configs pickle by name
-------------------------------

Every config here is a ``@dataclass(frozen=True, slots=True)`` and every pass
sends one to a spawned pool. For frozen slots dataclasses, ``dataclasses``
installs a ``__getstate__`` that returns ``[getattr(self, f.name) for f in
fields(self)]`` and a ``__setstate__`` that zips that list back onto
``fields(self)`` — **by position**. So if the pickling class and the unpickling
class have different field lists, every value after the first difference is
assigned to the wrong field, silently.

That is not hypothetical. Adding ``lit_gap_ratio`` and ``read_noise_fallback_kev``
to :class:`analysis.waxs.config.JungfrauWaxsConfig` on 2026-09-14 made a
26-field state arrive at a 28-field class, and a notebook kernel holding the
older import shifted every field after index 15 by two:

======================== ==================== =========================
field                    intended             received
======================== ==================== =========================
``read_noise_kev``       ``None`` (measure)   ``1000.0`` keV
``max_abs_kev``          ``1000.0`` keV       ``8`` keV
``min_modules``          ``1``                ``None``
``allow_incomplete``     ``False``            *unset*
======================== ==================== =========================

Only the last of those raised — ``TypeError: '<=' not supported between
instances of 'NoneType' and 'int'``, from inside ``extra_data``, naming nothing
that would lead back here. Had ``min_modules`` landed on an int, the run would
have finished with a readout noise of 1000 keV and every frame above 8 keV
routed to ``DATA_CHECK_FAILED``, which is CLAUDE.md pitfall 14 with a new cause.
Note what did *not* catch it: the worker's ``operator_sha256`` check passed,
because every field ``build_operator`` reads sits before the insertion point.

So :func:`state_by_name` and :func:`restore_by_name` pickle these configs as
``{name: value}``, and each config calls them from its own class body — a mixin
would not do, because ``dataclasses._add_slots`` installs the positional pair
whenever ``__getstate__`` is absent from the class's *own* ``__dict__``. A state
whose names are not exactly the class's is refused rather than partly applied:
filling a missing field from its default is precisely the silent behaviour this
exists to prevent.
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
#: are therefore excluded from :func:`config_sha256`. Every pass excludes at
#: least these; a pass with a gate-only field of its own excludes that too, and
#: :data:`analysis.waxs.config.WAXS_OPERATIONAL_FIELDS` is the one that does.
#: Read a stored file's set off ``config_operational_fields`` in its provenance,
#: never off this constant.
#:
#: Note what is deliberately *absent*: ``base_mask_trains`` (AGIPD) and
#: ``cell_sample_trains`` (JUNGFRAU) choose which trains are sampled, and so
#: change the masks and the measured readout noise that every row depends on.
#: They look operational and are not.
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
            "what a process running a version of this module from before "
            "2026-09-14 sends. Nothing in that state says which value is which, "
            f"so none of it can be trusted. {_SKEW_HINT}"
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
    # ``__setstate__`` bypasses ``__init__``, so nothing has run ``__post_init__``
    # on these values. Re-running it is what makes a worker refuse a config its
    # own rules reject, in the config's own words, rather than at whatever line
    # first happens to use the offending field.
    post_init = getattr(cfg, "__post_init__", None)
    if post_init is not None:
        post_init()
