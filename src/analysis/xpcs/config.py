"""Frozen configuration of the hit-train XPCS pass.

Two products, two hashes. The photon store holds raw
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
    "BRAGG_WINDOWS_NM",
    "DEFAULT_XPCS_ROOT",
    "MERGE_SCHEMA",
    "ES_RINGS_NM",
    "RINGS_NM",
    "STORE_SCHEMA",
    "XPCS_OPERATIONAL_FIELDS",
    "Kernel",
    "MaskPolicy",
    "XpcsConfig",
    "touches_bragg",
]

_PROPOSAL_ROOT = "/gpfs/exfel/exp/MID/202601/p010400"

#: Both products live in ``usr``, not scratch: scratch does not survive the move
#: to tape, and the photon store is the hit trains' only 2D record afterwards.
DEFAULT_XPCS_ROOT = f"{_PROPOSAL_ROOT}/usr/cached_files"


def _contiguous_nm(edges_A: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    nm = [10 * edge for edge in edges_A]
    return tuple(zip(nm[:-1], nm[1:], strict=True))


#: extra-speckle's seven contiguous ROIs, 0.0225 to 0.086 Å⁻¹, in nm⁻¹. The
#: equivalence gate against extra-speckle compares on these.
ES_RINGS_NM = _contiguous_nm((0.0225, 0.035, 0.045, 0.056, 0.065, 0.069, 0.075, 0.086))

#: The fcc 111 and 200 windows [nm⁻¹]. Bragg speckle inside them gives a
#: contrast near 1, so no XPCS ring and no plain triage ROI may touch one. They
#: enclose the measured Bragg excess of the crystallised runs with a margin.
BRAGG_WINDOWS_NM: tuple[tuple[float, float], ...] = ((0.57, 0.65), (0.68, 0.74))


def touches_bragg(lo: float, hi: float) -> bool:
    """Whether the interval ``(lo, hi)`` [nm⁻¹] overlaps a Bragg window."""
    return any(lo < w_hi and hi > w_lo for w_lo, w_hi in BRAGG_WINDOWS_NM)


#: The XPCS rings [nm⁻¹], 0.045 wide, clear of ``BRAGG_WINDOWS_NM``. The gap
#: between the two windows is narrower than a ring, so none sits there. The
#: outer bounds equal those of ``ES_RINGS_NM``, so both ring sets give the same
#: envelope and the same photon store.
RINGS_NM: tuple[tuple[float, float], ...] = _contiguous_nm(
    (0.0225, 0.027, 0.0315, 0.036, 0.0405, 0.045, 0.0495, 0.054)
) + _contiguous_nm((0.077, 0.0815, 0.086))

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
    #: ``(lo, hi)`` per ring [nm⁻¹], ascending and disjoint; gaps between rings
    #: are left out. The envelope spans the first ``lo`` to the last ``hi``.
    rings_nm: tuple[tuple[float, float], ...] = RINGS_NM
    #: ``(R, 16, 512, 128)`` bool ring masks (e.g. extra-speckle's), instead of
    #: rings from the geometry. No dilation: the envelope is their union, and the
    #: static mask is not applied on top of them.
    ring_masks_file: str | None = None
    kernel: Kernel = Kernel.CLEAN
    mask_policy: MaskPolicy = MaskPolicy.ANY_FRAME
    #: Largest train-id distance to a partner. 1 pairs with the previous train
    #: only, because the pooled baseline drifts with the gap on the PEG 6K
    #: reference runs.
    max_partner_gap: int = 1
    #: Envelope dilation. It exceeds the 4.0 mm between the two beam-centre
    #: conventions, so moving the centre from one to the other keeps every ring
    #: inside the store.
    envelope_margin_mm: float = 5.0
    #: extra-speckle's pixel-outlier mask (``mask_outlier``, with its production
    #: thresholds). A pixel whose RMS over frames of its per-frame mean lies
    #: outside these multiples of its q bin's median is removed from every ring.
    #: Computed from the hit trains at the merge; None switches it off.
    outlier_thresholds: tuple[float, float] | None = (0.75, 1.75)
    #: extra-speckle's q bins for that median (``saxs.get``'s hard-wired npt).
    outlier_q_bins: int = 300
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
        flat = [float(x) for ring in self.rings_nm for x in ring]
        if (
            not self.rings_nm
            or any(len(ring) != 2 for ring in self.rings_nm)
            or any(b <= a for a, b in zip(flat[::2], flat[1::2], strict=True))
            or any(b < a for a, b in zip(flat[1:-1:2], flat[2::2], strict=True))
        ):
            raise ValueError(
                "rings_nm must be (lo, hi) pairs with lo < hi, ascending and "
                f"disjoint, got {self.rings_nm}"
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
        if self.outlier_thresholds is not None:
            low, high = self.outlier_thresholds
            if not 0 <= low < high:
                raise ValueError(
                    "outlier_thresholds must be (low, high) with 0 <= low < high, "
                    f"got {self.outlier_thresholds}"
                )
            if self.ring_masks_file is not None:
                raise ValueError(
                    "the outlier mask needs each pixel's q, which ring_masks_file "
                    "does not give: set outlier_thresholds=None"
                )
        if self.outlier_q_bins < 2:
            raise ValueError(
                f"outlier_q_bins must be at least 2, got {self.outlier_q_bins}"
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
        """Pickle by field name, not position.

        A frozen slots dataclass pickles by position, so adding a field would
        shift every later field of a state pickled by an older import.
        """
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
        """``n_workers``, else one per physical core.

        ``os.sched_getaffinity`` would count hyperthreads too.
        """
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

        ``rings_sha256`` covers the ring bounds, the ring masks file and the
        static mask, which all act through the label array. The outlier mask
        enters through its settings: it is a function of the store, the hit set
        and the rings, so a complete run can be answered without recomputing it.
        """
        return _sha256(
            {
                "schema": MERGE_SCHEMA,
                "extract": self.extract_hash(envelope_sha256),
                "rings": rings_sha256,
                "kernel": self.kernel.value,
                "mask_policy": self.mask_policy.value,
                "max_partner_gap": self.max_partner_gap,
                "outlier": None
                if self.outlier_thresholds is None
                else {
                    "thresholds": [float(t) for t in self.outlier_thresholds],
                    "q_bins": self.outlier_q_bins,
                },
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
