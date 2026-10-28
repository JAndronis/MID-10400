"""Whether a run's AGIPD output is complete, on top of :mod:`analysis.common.complete`.

Up to three files: the frame table; unless ``cfg.pixel_sum_trains`` is ``None``,
the per-pixel window sums; unless ``cfg.photon_stats`` is
False, the per-frame photon statistics (``analysis.saxs.photon_stats``). A run is
complete only when all of them are.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from analysis.common import complete as common
from analysis.common.complete import (
    Completeness,
    frame_table_reasons,
    unfinished_reasons,
)
from analysis.saxs import operator as operator_module
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.operator import SparseOperator
from analysis.saxs.pixel_sums import DATASETS, pixel_sums_hash
from analysis.saxs.writer import AgipdSaxsWriter

__all__ = [
    "OMEGA_RTOL",
    "expected_operator",
    "is_complete",
    "operator_differences",
    "photon_stats_reasons_for",
    "window_sums_reasons",
]

#: Relative tolerance on the solid angle. The same operator built on AVX2 and on
#: AVX-512 nodes differs in ``omega`` by one ULP on some pixels, and in nothing
#: else. Any real change to the geometry moves it by many orders of magnitude
#: more.
OMEGA_RTOL = 1e-12

#: The operator arrays a rebuild reproduces bit for bit on every node class.
EXACT_ARRAYS: tuple[str, ...] = ("coef", "bins", "indptr", "q")

#: Operators by run-neutral config hash, for this process.
_EXPECTED: dict[str, SparseOperator] = {}


def expected_operator(cfg: AgipdSaxsConfig, geometry: Any = None) -> SparseOperator:
    """The operator this pass would build for ``cfg`` now, on this node.

    :param geometry: an ``AGIPD_1MGeometry`` to build from instead of
        ``cfg.geometry_file``. Not memoised, since nothing identifies it.
    """
    if geometry is not None:
        return operator_module.build_operator(geometry, cfg)[0]
    return common.expected_operator(
        _EXPECTED,
        cfg,
        lambda: operator_module.build_operator(
            operator_module.geometry_from_config(cfg), cfg
        )[0],
    )


def operator_differences(handle: h5py.File, op: SparseOperator) -> list[str]:
    """How the operator stored in ``handle`` differs from ``op``; empty if not.

    :data:`EXACT_ARRAYS` must be equal, ``omega`` equal to :data:`OMEGA_RTOL`.
    """
    return common.operator_differences(
        handle,
        op.sha256,
        exact={name: getattr(op, name) for name in EXACT_ARRAYS},
        toleranced={"omega": op.omega},
        rtol=OMEGA_RTOL,
    )


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
    operator: SparseOperator | None = None,
    geometry: Any = None,
    photon_stats_path: str | Path | None = None,
    ring_labels: np.ndarray | None = None,
) -> Completeness:
    """Whether a run's frame table, window sums and statistics are complete.

    Read-only: it opens the files for reading and writes nothing anywhere.

    :param path: the frame table; ``None`` is ``cfg.output_file``.
    :param pixel_sums_path: the window sums; ``None`` is ``cfg.pixel_sums_file``.
    :param operator: the operator to compare against, when the caller has
        already built it; ``None`` builds it (:func:`expected_operator`).
    :param geometry: passed to :func:`expected_operator`.
    :param photon_stats_path: the statistics; ``None`` is ``cfg.photon_stats_file``.
    :param ring_labels: the pixel → ring map built now
        (``photon_stats.ring_labels_from``), when the caller has it; ``None``
        builds it.
    """
    path = Path(path) if path is not None else cfg.output_file
    op = operator if operator is not None else expected_operator(cfg, geometry)
    reasons = frame_table_reasons(
        path, cfg, AgipdSaxsWriter, partial(operator_differences, op=op)
    )
    if cfg.pixel_sum_trains is not None:
        sums = Path(pixel_sums_path) if pixel_sums_path else cfg.pixel_sums_file
        reasons += [
            f"window sums: {reason}"
            for reason in window_sums_reasons(
                sums, cfg, frame_train_ids=_train_ids(path)
            )
        ]
    if cfg.photon_stats:
        reasons += [
            f"photon statistics: {reason}"
            for reason in photon_stats_reasons_for(
                cfg, photon_stats_path, ring_labels, geometry
            )
        ]
    return Completeness(path, tuple(reasons))


def photon_stats_reasons_for(
    cfg: AgipdSaxsConfig,
    path: str | Path | None = None,
    ring_labels: np.ndarray | None = None,
    geometry: Any = None,
) -> list[str]:
    """Why the statistics the ``agipd_saxs`` pass writes for ``cfg`` are incomplete.

    The pixel → ring map is compared when the caller has it (``ring_labels``)
    or passes a ``geometry`` to build it from; otherwise the hash stands for it,
    since it covers everything the map is made of (geometry file, beam centre,
    distance, energy, rings).
    """
    from analysis.saxs.photon_stats import (
        PhotonStatsConfig,
        photon_stats_reasons,
        ring_labels_from,
    )

    pcfg = PhotonStatsConfig.for_saxs(cfg)
    if ring_labels is None and geometry is not None:
        ring_labels = ring_labels_from(
            operator_module.build_operator(geometry, cfg)[1], pcfg
        )
    path = Path(path) if path else cfg.photon_stats_file
    return photon_stats_reasons(path, pcfg, ring_labels)


def _train_ids(path: Path) -> np.ndarray | None:
    try:
        with h5py.File(path, "r") as handle:
            return handle["trains/trainId"][:]
    except (OSError, KeyError):
        return None
