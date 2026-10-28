"""Whether a run's AGIPD output is complete, on top of :mod:`analysis.common.complete`.

A run is complete only when every product the config asks for is
(:func:`analysis.saxs.products.products_for`): the frame table; unless
``cfg.pixel_sum_trains`` is ``None``, the per-pixel window sums, which must
cover the frame table's trains; unless ``cfg.photon_stats`` is False, the
per-frame photon statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from analysis.common.complete import Completeness
from analysis.saxs import operator as operator_module
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.operator import SparseOperator, expected_operator
from analysis.saxs.photon_stats import PhotonStatsConfig, ring_labels_from
from analysis.saxs.products import products_for, train_ids

__all__ = ["is_complete"]


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
        already built it; ``None`` builds it
        (:func:`analysis.saxs.operator.expected_operator`).
    :param geometry: an ``AGIPD_1MGeometry`` to build the operator and the
        pixel → ring map from, instead of ``cfg.geometry_file``.
    :param photon_stats_path: the statistics; ``None`` is ``cfg.photon_stats_file``.
    :param ring_labels: the pixel → ring map built now
        (``photon_stats.ring_labels_from``), when the caller has it. ``None``
        builds it from ``geometry`` when that is given; otherwise the hash
        stands for it, since it covers everything the map is made of.
    """
    path = Path(path) if path is not None else cfg.output_file
    if geometry is not None and (
        operator is None or (cfg.photon_stats and ring_labels is None)
    ):
        built, ai = operator_module.build_operator(geometry, cfg)
        operator = operator if operator is not None else built
        if cfg.photon_stats and ring_labels is None:
            ring_labels = ring_labels_from(ai, PhotonStatsConfig.for_saxs(cfg))
    if operator is None:
        operator = expected_operator(cfg)
    products = products_for(
        cfg,
        output_path=path,
        pixel_sums_path=pixel_sums_path,
        photon_stats_path=photon_stats_path,
        op=operator,
        labels=ring_labels,
    )
    frame_ids = train_ids(path)
    reasons = [
        f"{product.prefix}{reason}"
        for product in products
        for reason in product.reasons(frame_ids)
    ]
    return Completeness(path, tuple(reasons))
