"""Frozen configuration of the hit-train XPCS pass.

Two products, two hashes (CLAUDE.md pitfall 12). The photon store holds raw
counts, which proc will not give back once it is taped, so what it is *keyed*
on is as narrow as it can be: :meth:`XpcsConfig.extract_hash` covers the pixel
envelope and the bit set, and nothing a later analysis chooses.
:meth:`XpcsConfig.merge_hash` adds what changes a stored sum (ring labels with
the static mask folded in, the kernel, the mask policy, the partner gap), so a
threshold or partner change rebuilds the sums from the store and never touches
the store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from analysis.common.config import restore_by_name, state_by_name
from analysis.common.cpu import physical_cores
from analysis.saxs.config import N_MODULES, AgipdSaxsConfig

__all__ = [
    "DEFAULT_XPCS_ROOT",
    "MERGE_SCHEMA",
    "RING_EDGES_NM",
    "STORE_SCHEMA",
    "XPCS_OPERATIONAL_FIELDS",
    "Kernel",
    "MaskPolicy",
    "XpcsConfig",
]

_PROPOSAL_ROOT = "/gpfs/exfel/exp/MID/202601/p010400"

#: Both products live in ``usr``, not scratch: scratch does not survive the move
#: to tape, and the photon store is the hit trains' only 2D record afterwards.
DEFAULT_XPCS_ROOT = f"{_PROPOSAL_ROOT}/usr/cached_files"

#: The XPCS ROI edges [nm⁻¹], as the DAMNIT context hands them to extra-speckle
#: (Å⁻¹ 0.0225 … 0.086). Seven contiguous rings.
RING_EDGES_NM: tuple[float, ...] = tuple(
    10 * edge for edge in (0.0225, 0.035, 0.045, 0.056, 0.065, 0.069, 0.075, 0.086)
)

#: Bump when what either file holds changes.
STORE_SCHEMA = 1
MERGE_SCHEMA = 1

#: Fields no hash covers: they change how the pass runs, not what it stores.
XPCS_OPERATIONAL_FIELDS: frozenset[str] = frozenset(
    {
        "n_workers",
        "trains_per_block",
        "flag_base_trains",
        "output_root",
        "allow_incomplete",
        "overwrite",
    }
)


class Kernel(Enum):
    """How a train is correlated with its partner."""

    CLEAN = "clean"
    """Partner is the previous OK hit within ``max_partner_gap`` train ids.
    The raw sums use the train's own mask; the pair mask enters the baseline
    only. A train without a partner has no baseline, which is NaN downstream."""

    ES = "es"
    """extra-speckle bit for bit, for the equivalence gate: the chain restarts
    at each proc sequence file, the first train of a chain is paired with its
    own mask (a baseline of exactly 1), and the pair mask leaks into the raw
    sums."""


class MaskPolicy(Enum):
    """Which pixels a train correlates."""

    ANY_FRAME = "any_frame"
    """Excluded if ``image.mask`` flags it in any frame of the train (as ES)."""

    STATIC_ONLY = "static_only"
    """Only the static mask and the ring map; ``image.mask`` is ignored. A
    diagnostic for how much the dynamic flags matter, not a production setting."""


@dataclass(frozen=True, slots=True)
class XpcsConfig:
    """Immutable run configuration.

    ``saxs`` supplies the geometry, the static mask, the bit set and the module
    requirement, so the pass uses the SAXS pass's pixel-centre q rather than a
    second geometry.
    """

    saxs: AgipdSaxsConfig
    ring_edges_nm: tuple[float, ...] = RING_EDGES_NM
    #: ``(R, 16, 512, 128)`` bool ring masks (e.g. extra-speckle's), instead of
    #: rings from the geometry. No dilation: the envelope is their union.
    ring_masks_file: str | None = None
    kernel: Kernel = Kernel.CLEAN
    mask_policy: MaskPolicy = MaskPolicy.ANY_FRAME
    #: Largest train-id distance to a partner. Provisional until benchmark B4.
    max_partner_gap: int = 10
    #: Envelope dilation. Provisional; it only has to exceed the beam-centre
    #: disagreement of open task 4 (4.0 mm) for a later move of the centre to
    #: stay inside the store.
    envelope_margin_mm: float = 5.0
    # ── scheduling and output (operational) ──────────
    n_workers: int | None = None
    trains_per_block: int = 8
    #: Trains whose masks give the photon store's flag base. Any base is
    #: lossless, so this only sets how small the file is.
    flag_base_trains: int = 5
    output_root: str = DEFAULT_XPCS_ROOT
    allow_incomplete: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.saxs.min_modules != N_MODULES:
            raise ValueError(
                f"the XPCS pass needs all {N_MODULES} modules, but saxs.min_modules "
                f"is {self.saxs.min_modules}: a missing module reads as zero photons "
                "with no flag, which would pass as a quiet part of the detector"
            )
        edges = tuple(self.ring_edges_nm)
        if len(edges) < 2 or any(
            b <= a for a, b in zip(edges, edges[1:], strict=False)
        ):
            raise ValueError(
                f"ring_edges_nm must be strictly increasing with at least two "
                f"values, got {edges}"
            )
        if not isinstance(self.kernel, Kernel):
            raise TypeError(f"kernel must be a Kernel, got {self.kernel!r}")
        if not isinstance(self.mask_policy, MaskPolicy):
            raise TypeError(
                f"mask_policy must be a MaskPolicy, got {self.mask_policy!r}"
            )
        if self.max_partner_gap < 1:
            raise ValueError(
                f"max_partner_gap must be positive, got {self.max_partner_gap}"
            )
        if self.envelope_margin_mm < 0:
            raise ValueError(
                "envelope_margin_mm must not be negative, got "
                f"{self.envelope_margin_mm}"
            )
        if self.flag_base_trains < 1:
            raise ValueError(
                f"flag_base_trains must be positive, got {self.flag_base_trains}"
            )
        if self.trains_per_block < 1:
            raise ValueError(
                f"trains_per_block must be positive, got {self.trains_per_block}"
            )
        if self.n_workers is not None and self.n_workers < 1:
            raise ValueError(f"n_workers must be positive, got {self.n_workers}")

    def __getstate__(self) -> dict[str, Any]:
        """Pickle by field name, not by position (CLAUDE.md pitfall 20)."""
        return state_by_name(self)

    def __setstate__(self, state: Any) -> None:
        restore_by_name(self, state)

    # ── identity ──────────────────────────────────────
    @property
    def proposal(self) -> int:
        return self.saxs.proposal

    @property
    def run(self) -> int:
        return self.saxs.run

    @property
    def workers(self) -> int:
        """``n_workers``, else one per *physical* core (CLAUDE.md pitfall 10)."""
        if self.n_workers is not None:
            return int(self.n_workers)
        return physical_cores() or 1

    @property
    def operational_fields(self) -> frozenset[str]:
        """Which fields the hashes leave out, for the provenance record."""
        return XPCS_OPERATIONAL_FIELDS

    @property
    def store_file(self) -> Path:
        """``{output_root}/agipd_ring_photons/r{run:04d}/agipd_ring_photons.h5``."""
        root = Path(self.output_root) / "agipd_ring_photons"
        return root / f"r{self.run:04d}" / "agipd_ring_photons.h5"

    @property
    def output_file(self) -> Path:
        """``{output_root}/xpcs_hits/r{run:04d}/xpcs_hits.h5``."""
        return (
            Path(self.output_root) / "xpcs_hits" / f"r{self.run:04d}" / "xpcs_hits.h5"
        )

    # ── hashes ────────────────────────────────────────
    def extract_hash(self, envelope_sha256: str) -> str:
        """What changes a stored photon: the envelope pixels and the flag bits.

        The envelope keeps statically bad pixels, so neither the static mask,
        the ring edges, the kernel nor any threshold is in here: all of them are
        recomputable from the store, which proc cannot replace.
        """
        return _sha256(
            {
                "schema": STORE_SCHEMA,
                "proposal": self.proposal,
                "run": self.run,
                "envelope": envelope_sha256,
                "mask_bits": self.saxs.mask_bits,
                "min_modules": self.saxs.min_modules,
                "detector_name": self.saxs.detector_name,
            }
        )

    def merge_hash(self, envelope_sha256: str, rings_sha256: str) -> str:
        """What changes a stored sum: the extract hash plus how it is combined.

        ``rings_sha256`` covers the ring edges, the ring masks file and the
        static mask, which all act through the label array.
        """
        return _sha256(
            {
                "schema": MERGE_SCHEMA,
                "extract": self.extract_hash(envelope_sha256),
                "rings": rings_sha256,
                "kernel": self.kernel.value,
                "mask_policy": self.mask_policy.value,
                "max_partner_gap": self.max_partner_gap,
            }
        )

    def payload(self) -> dict[str, Any]:
        """Every field, JSON-ready, for the provenance record."""
        return json.loads(json.dumps(asdict(self), default=_jsonable, sort_keys=True))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, frozenset | set):
        return sorted(value)
    return str(value)


def _sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
