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
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from analysis.common.cpu import file_sha256

__all__ = ["OPERATIONAL_FIELDS", "config_sha256", "result_fields"]

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
