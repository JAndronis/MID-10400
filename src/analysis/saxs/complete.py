"""Whether a run's AGIPD output is complete — see :mod:`analysis.common.complete`.

Two files: the frame table and, unless ``cfg.pixel_sum_trains`` is ``None``, the
per-pixel window sums (context file §15). A run is complete only when both are.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.common.complete import (
    Completeness,
    frame_table_reasons,
    unfinished_reasons,
)
from analysis.saxs import operator as operator_module
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.pixel_sums import DATASETS, pixel_sums_hash
from analysis.saxs.writer import AgipdSaxsWriter

__all__ = ["expected_operator_sha256", "is_complete", "window_sums_reasons"]

#: Operator sha256 by run-neutral config hash, for this process.
_EXPECTED: dict[str, str] = {}


def expected_operator_sha256(cfg: AgipdSaxsConfig, geometry: Any = None) -> str:
    """The sha256 of the operator this pass would build for ``cfg`` now.

    Built for real rather than inferred from library versions, because it is
    cheap — 1.2 s on r0423 — and an exact comparison needs no assumption about
    what a pyFAI or EXtra-geom release changes. No input of the operator depends
    on the run, so it is memoised per process on the config with the proposal
    and run zeroed; a script checking hundreds of runs builds it once.

    :param geometry: an ``AGIPD_1MGeometry`` to build from instead of
        ``cfg.geometry_file``. Not memoised, since nothing identifies it.
    """
    if geometry is not None:
        return operator_module.build_operator(geometry, cfg)[0].sha256
    key = replace(cfg, proposal=0, run=0).config_hash()
    if key not in _EXPECTED:
        geometry = operator_module.geometry_from_config(cfg)
        _EXPECTED[key] = operator_module.build_operator(geometry, cfg)[0].sha256
    return _EXPECTED[key]


def window_sums_reasons(
    path: str | Path,
    cfg: AgipdSaxsConfig,
    *,
    frame_train_ids: np.ndarray | None = None,
) -> list[str]:
    """Why the window-sums file at ``path`` is not complete; empty when it is.

    :param frame_train_ids: the frame table's train ids, which the sums must
        cover exactly; ``None`` skips that comparison.
    """
    path = Path(path)
    if not path.exists():
        return ["no file"]
    try:
        with h5py.File(path, "r") as handle:
            missing = [name for name in DATASETS if name not in handle]
            if missing or "provenance" not in handle:
                return [f"lacks {missing or ['provenance']}"]
            attrs = handle["provenance"].attrs
            detector = attrs.get("detector_name")
            reasons = []
            if cfg.detector_name is not None and cfg.detector_name != detector:
                reasons.append(
                    f"summed {detector}, the config names {cfg.detector_name}"
                )
            stored, current = attrs.get("config_hash"), pixel_sums_hash(cfg, detector)
            if stored != current:
                reasons.append(f"hash {stored}, the current config's is {current}")
            if "status_summary" not in attrs:
                reasons.append("never finalised")
            written = handle["windows/written"][:].astype(bool)
            if not written.all():
                reasons.append(
                    f"{int((~written).sum())} of {written.size} windows unwritten"
                )
            reasons += unfinished_reasons(handle["trains/status"][:], "trains")
            if frame_train_ids is not None and not np.array_equal(
                handle["trains/trainId"][:], frame_train_ids
            ):
                reasons.append("its trains are not the frame table's")
    except (OSError, KeyError, ValueError) as error:
        return [f"unreadable: {error!r}"]
    return reasons


def is_complete(
    path: str | Path | None,
    cfg: AgipdSaxsConfig,
    *,
    pixel_sums_path: str | Path | None = None,
    operator_sha256: str | None = None,
    geometry: Any = None,
) -> Completeness:
    """Whether a run's frame table — and its window sums — are complete.

    Read-only: it opens both files for reading and writes nothing anywhere.

    :param path: the frame table; ``None`` is ``cfg.output_file``.
    :param pixel_sums_path: the window sums; ``None`` is ``cfg.pixel_sums_file``.
    :param operator_sha256: the operator to compare against, when the caller has
        already built it; ``None`` builds it (:func:`expected_operator_sha256`).
    :param geometry: passed to :func:`expected_operator_sha256`.
    """
    path = Path(path) if path is not None else cfg.output_file
    if operator_sha256 is None:
        operator_sha256 = expected_operator_sha256(cfg, geometry)
    reasons = frame_table_reasons(path, cfg, AgipdSaxsWriter, operator_sha256)
    if cfg.pixel_sum_trains is not None:
        sums = Path(pixel_sums_path) if pixel_sums_path else cfg.pixel_sums_file
        reasons += [
            f"window sums: {reason}"
            for reason in window_sums_reasons(
                sums, cfg, frame_train_ids=_train_ids(path)
            )
        ]
    return Completeness(path, tuple(reasons))


def _train_ids(path: Path) -> np.ndarray | None:
    try:
        with h5py.File(path, "r") as handle:
            return handle["trains/trainId"][:]
    except (OSError, KeyError):
        return None
