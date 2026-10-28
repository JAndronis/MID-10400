"""The files the ``agipd_saxs`` pass writes from one read of proc.

Each is a :class:`Product`: the frame table (``agipd_saxs.h5``), the per-pixel
window sums (``agipd_pixel_sums.h5``) and the per-frame photon statistics
(``agipd_photon_stats.h5``). A product knows its file, why that file is not
complete, which status codes make it incomplete, which blocks it still needs,
and how to write and finalise it. The pass, :func:`analysis.saxs.complete.is_complete`
and the DAMNIT status cell loop over the products a config asks for.
"""

from __future__ import annotations

import logging
import platform
import socket
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np

from analysis.common.complete import frame_table_reasons, unfinished_reasons
from analysis.common.cpu import file_sha256, package_versions
from analysis.common.lock import OutputLock
from analysis.common.plan import Block, RunPlan
from analysis.common.run import FanOutTotals, base_provenance
from analysis.common.status import FrameStatus
from analysis.common.writer import IncompleteRun, status_counts
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.operator import SparseOperator, operator_differences
from analysis.saxs.photon_stats import (
    PhotonSpec,
    PhotonStatsConfig,
    PhotonStatsWriter,
    photon_provenance,
    photon_stats_reasons,
)
from analysis.saxs.pixel_sums import (
    DATASETS,
    FAILED,
    PixelSumWriter,
    WindowSpec,
    pixel_sums_hash,
)
from analysis.saxs.writer import AgipdSaxsWriter

__all__ = [
    "FrameTable",
    "PassRun",
    "PhotonStats",
    "Product",
    "WindowSums",
    "check_files",
    "products_for",
    "train_ids",
]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PassRun:
    """What the parent built for one pass, for the products to write with."""

    cfg: AgipdSaxsConfig
    plan: RunPlan
    started_at: float
    started: float
    setup: dict[str, float]
    op: SparseOperator | None = None
    labels: np.ndarray | None = None
    static: Any = None
    base_masks: Any = None
    sampled: np.ndarray | None = None
    windows: WindowSpec | None = None


@dataclass(slots=True)
class Product:
    """One output file of the pass.

    Between :meth:`open` and the end of the pass, :attr:`writer` is the file's
    open writer and :attr:`blocks` the indices of the blocks it still needs.
    A frame table sees only its own blocks: a block another product needed
    was read for that one, and must not touch its rows.
    """

    #: The key of this product in a status summary.
    key: ClassVar[str]
    #: Its name in the DAMNIT status line.
    name: ClassVar[str]
    #: What the pass does to a block for it, in the log.
    verb: ClassVar[str]
    #: Prefix of its reasons in a completeness verdict.
    prefix: ClassVar[str]
    #: Its status column.
    column: ClassVar[str]

    cfg: AgipdSaxsConfig
    path: Path
    writer: Any = field(default=None, init=False)
    blocks: frozenset[int] = field(default=frozenset(), init=False)

    @property
    def forced(self) -> bool:
        """Whether ``overwrite`` makes the pass redo it without asking."""
        return False

    def reasons(self, frame_train_ids: np.ndarray | None) -> list[str]:
        """Why the file is not complete; empty when it is.

        :param frame_train_ids: the frame table's train ids, which the window
            sums must cover exactly; ``None`` skips that comparison.
        """
        raise NotImplementedError

    def failures(self, counts: dict[str, int]) -> dict[str, int]:
        """The status codes in ``counts`` that make this file incomplete."""
        return {name: n for name, n in counts.items() if name != FrameStatus.OK.name}

    def open(self, stack: ExitStack, run: PassRun) -> None:
        raise NotImplementedError

    def begin(self, todo: list[Block]) -> None:
        """Called once with every block the pass will read."""

    def write(self, block: Block, result: Any) -> None:
        if block.index in self.blocks:
            self.writer.write_block(block, self._rows(result))

    def mark(self, block: Block, status: FrameStatus, message: str) -> None:
        if block.index in self.blocks:
            self.writer.mark(block, status, message)

    def mark_remaining(self, status: FrameStatus) -> None:
        self.writer.mark_remaining(status)

    def finalise(self, run: PassRun, totals: FanOutTotals, lock: OutputLock) -> None:
        raise NotImplementedError

    def _rows(self, result: Any) -> Any:
        raise NotImplementedError


@dataclass(slots=True)
class FrameTable(Product):
    """``agipd_saxs.h5``: one row of sums per frame, the operator, the masks."""

    key: ClassVar[str] = "frames"
    name: ClassVar[str] = "frames"
    verb: ClassVar[str] = "integrate"
    prefix: ClassVar[str] = ""
    column: ClassVar[str] = "frames/status"

    op: SparseOperator | None = None

    @property
    def forced(self) -> bool:
        return self.cfg.overwrite

    def reasons(self, frame_train_ids: np.ndarray | None) -> list[str]:
        return frame_table_reasons(
            self.path,
            self.cfg,
            AgipdSaxsWriter,
            partial(operator_differences, op=self.op),
        )

    def open(self, stack: ExitStack, run: PassRun) -> None:
        self.writer = stack.enter_context(
            AgipdSaxsWriter.open_or_create(
                run.cfg,
                run.plan,
                self.path,
                operator_check=partial(operator_differences, op=run.op),
            )
        )
        self.writer.store_operator(run.op)
        self.writer.store_masks(run.base_masks, run.sampled)
        self.blocks = frozenset(
            b.index for b in run.plan.blocks if not self.writer.block_complete(b)
        )

    def finalise(self, run: PassRun, totals: FanOutTotals, lock: OutputLock) -> None:
        cfg, static = run.cfg, run.static
        self.writer.finalise(
            base_provenance(
                cfg,
                run.plan,
                self.writer,
                started_at=run.started_at,
                started=run.started,
                totals=totals,
                setup_timings=run.setup,
                operator_sha256=run.op.sha256,
                input_file_sha256={
                    name: (file_sha256(path) if path else None)
                    for name, path in cfg.input_files.items()
                },
                run_checks=run.plan.checks,
            )
            | {
                "masks_sha256": run.base_masks.sha256,
                "static_mask_sha256": static.sha256,
                "static_mask_sources": [
                    {
                        "name": s.name,
                        "path": s.path,
                        "sha256": s.sha256,
                        "n_excluded": s.n_excluded,
                    }
                    for s in static.sources
                ],
                "unseen_cell_frames": int(totals.extra["unseen_cells"]),
            }
            | lock.provenance()
        )

    def _rows(self, result: Any) -> Any:
        return result.frames


@dataclass(slots=True)
class WindowSums(Product):
    """``agipd_pixel_sums.h5``: per-pixel photon sums per window of trains.

    Never truncated, ``overwrite`` or not: once proc is on tape the sums cannot
    be rebuilt. A train is a failure only with a code in ``pixel_sums.FAILED``;
    ``NO_FRAMES`` and ``MISSING_MODULES`` trains own no rows and are final.
    """

    key: ClassVar[str] = "window_sums"
    name: ClassVar[str] = "sums"
    verb: ClassVar[str] = "sum"
    prefix: ClassVar[str] = "window sums: "
    column: ClassVar[str] = "trains/status"

    def reasons(self, frame_train_ids: np.ndarray | None) -> list[str]:
        if not self.path.exists():
            return ["no file"]
        cfg = self.cfg
        try:
            with h5py.File(self.path, "r") as handle:
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
                stored, current = (
                    attrs.get("config_hash"),
                    pixel_sums_hash(cfg, detector),
                )
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

    def failures(self, counts: dict[str, int]) -> dict[str, int]:
        failed = {code.name for code in FAILED} | {"unreadable"}
        return {name: n for name, n in counts.items() if name in failed}

    def open(self, stack: ExitStack, run: PassRun) -> None:
        self.writer = stack.enter_context(
            PixelSumWriter.open_or_create(run.cfg, run.plan, run.windows, self.path)
        )
        self.blocks = frozenset(
            b.index for b in self.writer.blocks_to_do(run.plan.blocks)
        )

    def begin(self, todo: list[Block]) -> None:
        self.writer.begin(todo)

    def write(self, block: Block, result: Any) -> None:
        # Every block the pass read: the writer ignores what a written window
        # already holds.
        self.writer.write_block(block, result)

    def mark(self, block: Block, status: FrameStatus, message: str) -> None:
        self.writer.fail_block(block, status, message)

    def mark_remaining(self, status: FrameStatus) -> None:
        """Nothing: a rerun rebuilds an unwritten window from scratch."""

    def finalise(self, run: PassRun, totals: FanOutTotals, lock: OutputLock) -> None:
        self.writer.finalise(
            {
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "package_versions": package_versions(),
                "n_workers": run.cfg.workers,
                "n_blocks": len(run.plan.blocks),
                "started_at": run.started_at,
                "wall_s": time.perf_counter() - run.started,
                "bits_present": int(totals.bits_present),
                "timings": totals.timings,
                "setup_timings": {**run.setup, "write_blocks": totals.write_s},
                "run_checks": run.plan.checks,
            }
            | lock.provenance()
        )


@dataclass(slots=True)
class PhotonStats(Product):
    """``agipd_photon_stats.h5``: photon-count histograms per frame and ring."""

    key: ClassVar[str] = "photon_stats"
    name: ClassVar[str] = "stats"
    verb: ClassVar[str] = "histogram"
    prefix: ClassVar[str] = "photon statistics: "
    column: ClassVar[str] = "frames/status"

    #: The pixel → ring map built now; ``None`` lets the hash stand for it,
    #: since it covers everything the map is made of.
    labels: np.ndarray | None = None

    @property
    def photon_cfg(self) -> PhotonStatsConfig:
        return PhotonStatsConfig.for_saxs(self.cfg)

    def reasons(self, frame_train_ids: np.ndarray | None) -> list[str]:
        return photon_stats_reasons(self.path, self.photon_cfg, self.labels)

    def open(self, stack: ExitStack, run: PassRun) -> None:
        photon_cfg = self.photon_cfg
        self.writer = stack.enter_context(
            PhotonStatsWriter.open_or_create(photon_cfg, run.plan, self.path)
        )
        self.writer.store_rings(photon_cfg, run.labels, run.op.sha256)
        self.blocks = frozenset(
            b.index for b in run.plan.blocks if not self.writer.block_complete(b)
        )

    def spec(self, run: PassRun) -> PhotonSpec:
        """What a worker needs to histogram this product's blocks."""
        return PhotonSpec(
            run.labels, run.static.bad, self.photon_cfg.k_max, self.blocks
        )

    def finalise(self, run: PassRun, totals: FanOutTotals, lock: OutputLock) -> None:
        self.writer.finalise(
            photon_provenance(
                self.photon_cfg,
                run.plan,
                self.writer,
                run.op.sha256,
                started_at=run.started_at,
                started=run.started,
                totals=totals,
                setup_timings=run.setup,
            )
            | {"static_mask_sha256": run.static.sha256}
            | lock.provenance()
        )

    def _rows(self, result: Any) -> Any:
        return result.photon


def products_for(
    cfg: AgipdSaxsConfig,
    *,
    output_path: Path | None = None,
    pixel_sums_path: Path | None = None,
    photon_stats_path: Path | None = None,
    op: SparseOperator | None = None,
    labels: np.ndarray | None = None,
) -> list[Product]:
    """The products the pass writes for ``cfg``, the frame table first.

    A path left ``None`` is the config's own.
    """
    products: list[Product] = [
        FrameTable(cfg, Path(output_path or cfg.output_file), op=op)
    ]
    if cfg.pixel_sum_trains is not None:
        products.append(WindowSums(cfg, Path(pixel_sums_path or cfg.pixel_sums_file)))
    if cfg.photon_stats:
        products.append(
            PhotonStats(
                cfg, Path(photon_stats_path or cfg.photon_stats_file), labels=labels
            )
        )
    return products


def train_ids(path: Path) -> np.ndarray | None:
    """The train table of the frame table at ``path``; ``None`` if unreadable."""
    try:
        with h5py.File(path, "r") as handle:
            return handle["trains/trainId"][:]
    except (OSError, KeyError):
        return None


def check_files(products: list[Product], cfg: AgipdSaxsConfig, *, noun: str) -> None:
    """Raise on the same terms as the end of a pass, reading only the files.

    A complete file can still hold rows that are final without being ``OK``,
    such as ``LABEL_MISMATCH``.

    :param noun: what the message says did not reach ``OK``.
    :raises IncompleteRun: a product holds a failure and ``cfg.allow_incomplete``
        is False. The summary is per product, or flat for a single one.
    """
    summary: dict[str, Any] = {}
    incomplete = False
    for product in products:
        with h5py.File(product.path, "r") as handle:
            counts = status_counts(handle[product.column][:], nonzero_only=True)
        summary[product.key] = counts
        incomplete |= bool(product.failures(counts))
    if incomplete and not cfg.allow_incomplete:
        shown = summary if len(products) > 1 else next(iter(summary.values()))
        raise IncompleteRun(f"not every {noun}: {shown}")
