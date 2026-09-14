"""Frozen configuration for the JUNGFRAU WAXS integrator (``jungfrau_waxs``).

One config per detector: the two JUNGFRAU-500Ks have different geometries,
static masks, q ranges and orientations, so each gets its own PONI, mask and
output file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.common.config import (
    OPERATIONAL_FIELDS,
    PassConfigMembers,
    restore_by_name,
    state_by_name,
)

__all__ = [
    "DETECTORS",
    "DETECTOR_MODNOS",
    "DETECTOR_NAMES",
    "PROC_FILE_PATTERN",
    "EXPECTED_BITS",
    "DEFAULT_READ_NOISE_KEV",
    "WAXS_OPERATIONAL_FIELDS",
    "EXPECTED_LIT_CELLS",
    "METHOD",
    "MODULE_SHAPE",
    "NPIX",
    "JungfrauWaxsConfig",
    "default_poni_file",
    "default_static_mask_file",
    "is_storage_cell_sequence",
]

#: One JUNGFRAU-500K module, slow-scan × fast-scan.
MODULE_SHAPE: tuple[int, int] = (512, 1024)
NPIX: int = MODULE_SHAPE[0] * MODULE_SHAPE[1]

#: Memory cells per train, fixed by the burst-mode readout unlike AGIPD's
#: ragged frame table. How many of them are *lit* is measured per run.
CELLS_PER_TRAIN: int = 16

#: The only integration method this pass accepts.
METHOD: tuple[str, str, str] = ("full", "csc", "cython")

#: The two detectors, by the short name used in paths and output filenames.
DETECTORS: tuple[str, ...] = ("jf1", "jf2")

#: EuXFEL source prefixes, taken from ``lsxfel``. Both names match
#: ``JUNGFRAU._det_name_pat``, so auto-detection is ambiguous with two JUNGFRAUs
#: in a run and the names must be given rather than found.
DETECTOR_NAMES: dict[str, str] = {
    "jf1": "MID_EXP_JF500K1",
    "jf2": "MID_EXP_JF500K2",
}

#: The module number in each detector's source name: ``JNGFR01`` / ``JNGFR02``.
#: Passed as ``first_modno`` so each single-module detector reports modno 1 and
#: ``n_modules`` 1, rather than jf2 claiming to be the second of two.
DETECTOR_MODNOS: dict[str, int] = {"jf1": 1, "jf2": 2}

#: Corrected JUNGFRAU files, one per detector per sequence.
PROC_FILE_PATTERN = "CORR-R{run:04d}-JNGFR{modno:02d}-S{seq:05d}.h5"

_PROPOSAL_ROOT = "/gpfs/exfel/exp/MID/202601/p010400"

#: ``BadPixels`` bits seen on both detectors. Bit 22 ``NON_STANDARD_SIZE`` is
#: set here, unlike AGIPD, which is why no separate seam mask is needed.
EXPECTED_BITS: frozenset[int] = frozenset({0, 1, 21, 22})

#: The memory cells that carry photons in the r0379–r0500 science block: eight
#: storage cells starting at 15, so the sequence wraps 15 → 0 → … → 6. Note it
#: is ``{0…6, 15}`` and not ``{0…7}``, which is the *array position* set.
#:
#: One of the proposal's four patterns, not *the* pattern — the pass measures the
#: set per run and checks its shape. Pin it through
#: :attr:`JungfrauWaxsConfig.expected_lit_cells` only to make a reprocess of that
#: block refuse anything that has drifted.
EXPECTED_LIT_CELLS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 15)

#: Readout noise in keV per detector, for runs that read all 16 storage cells
#: and so have no dark cell to measure it from. The per-detector median over
#: every run where the dark side is unambiguous. A measurement from the run's
#: own dark cells always wins; :func:`analysis.waxs.run._error_model` decides.
DEFAULT_READ_NOISE_KEV: dict[str, float] = {"jf1": 0.3440, "jf2": 0.3195}

#: The shared operational set plus two fields that only *gate*:
#: ``expected_lit_cells`` raises before the output file is opened and
#: ``allow_data_check_failures`` after it is finished, so neither can change a
#: value in ``/frames`` and pinning either does not invalidate existing files.
#:
#: ``lit_fraction_min`` and ``lit_gap_ratio`` are the opposite case and stay in
#: the hash: they decide which cells are integrated, so which rows exist at all.
WAXS_OPERATIONAL_FIELDS: frozenset[str] = OPERATIONAL_FIELDS | {
    "expected_lit_cells",
    "allow_data_check_failures",
}

#: Where the per-run output files are written.
DEFAULT_OUTPUT_ROOT = f"{_PROPOSAL_ROOT}/scratch/jungfrau_waxs"


def default_poni_file(detector: str) -> str:
    """pyFAI geometry for one detector, as ``usr/geometry/jf{n}.poni``."""
    return f"{_PROPOSAL_ROOT}/usr/geometry/{detector}.poni"


def default_static_mask_file(detector: str) -> str:
    """Native pyFAI static mask for one detector; non-zero means excluded.

    Named ``jf{n}_mask.edf``, unlike the PONIs, which are bare ``jf{n}.poni``.
    """
    return f"{_PROPOSAL_ROOT}/usr/masks/{detector}_mask.edf"


def is_storage_cell_sequence(lit: tuple[int, ...], n_cells: int) -> bool:
    """Is ``lit`` a run of consecutive cells mod ``n_cells``?

    Empty and full sets are sequences trivially. Everything else must be
    reachable as ``{(start + i) % n_cells}`` from one of its own members, which
    is what ``storageCellStart`` plus ``storageCells`` produces.
    """
    if not lit or len(lit) >= n_cells:
        return True
    wanted = set(lit)
    return any(
        {(start + i) % n_cells for i in range(len(lit))} == wanted for start in lit
    )


@dataclass(frozen=True, slots=True)
class JungfrauWaxsConfig(PassConfigMembers):
    """Immutable run configuration for one detector.

    ``poni_file`` and ``static_mask_file`` default from :attr:`detector` and may
    be ``None`` only when the caller supplies the geometry directly, which is
    what the unit tests do with a synthetic PONI. Every real run must set both,
    so that their sha256 enters :meth:`config_hash`.
    """

    proposal: int
    run: int
    detector: str
    #: EuXFEL source prefix. ``None`` fills it from :data:`DETECTOR_NAMES`;
    #: set it only for a run whose source names differ.
    detector_name: str | None = None
    #: ``None`` fills it from :data:`DETECTOR_MODNOS`.
    first_modno: int | None = None
    poni_file: str | None = None
    static_mask_file: str | None = None
    photon_energy_kev: float = 9.04
    npt: int = 500
    method: tuple[str, str, str] = METHOD
    unit: str = "q_nm^-1"
    # ── masks ─────────────────────────────────────────────────────────────
    mask_bits: int = 0xFFFFFFFF
    expected_bits: frozenset[int] = EXPECTED_BITS
    # ── lit-cell selection ────────────────────────────────────────────────
    #: Pins the lit set, refusing the run if the data disagree. ``None`` — the
    #: default — measures it and checks only that its shape is a storage-cell
    #: sequence, the check that holds for every run.
    expected_lit_cells: tuple[int, ...] | None = None
    #: Half a photon: a pixel above this held at least one.
    lit_threshold_kev: float = 4.5
    #: Fraction of kept pixels above :attr:`lit_threshold_kev` for a cell to
    #: count as lit. Sits in the empty band between the two populations, clear
    #: of both by 8.5×; any value from 1e-4 to 1e-3 classifies identically.
    lit_fraction_min: float = 4.805e-4
    #: Among the cells that clear the floor, a step larger than this splits them
    #: again, so a uniformly attenuated run still splits on the lit-to-dark step
    #: rather than on the absolute level.
    lit_gap_ratio: float = 30.0
    cell_sample_trains: int = 8
    # ── error model ───────────────────────────────────────────────────────
    #: ``None`` measures the readout noise from the dark cells of the sampled
    #: trains; a float overrides it even when a measurement is available.
    read_noise_kev: float | None = None
    #: Used only when the run reads every storage cell, leaving no dark cell to
    #: measure from. ``None`` is resolved from :data:`DEFAULT_READ_NOISE_KEV` in
    #: ``__post_init__``, so a constructed config never holds ``None`` here and
    #: the number actually used is the one that enters :meth:`config_hash`.
    read_noise_fallback_kev: float | None = None
    # ── data check ────────────────────────────────────────────────────────
    #: Values beyond ``±`` this are excluded pixel by pixel. Set far above the
    #: kept-region maximum, in the empty band below the artifact population.
    max_abs_kev: float = 1.0e3
    # ── run, scheduling and output ────────────────────────────────────────
    min_modules: int = 1
    #: Whether ``DATA_CHECK_FAILED`` frames on their own let the run finish.
    #: That status means a frame with **every** pixel excluded — a property of
    #: the data, not a failure of the pass — so one pathological frame does not
    #: take a bulk reprocess down. The count is logged at WARNING and recorded
    #: in ``provenance/data_check``. ``WORKER_ERROR``, ``NOT_PROCESSED`` and
    #: ``LABEL_MISMATCH`` still raise.
    allow_data_check_failures: bool = True
    trains_per_block: int = 8
    n_workers: int | None = None
    selftest_frames: int = 8
    output_root: str = DEFAULT_OUTPUT_ROOT
    allow_incomplete: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.detector not in DETECTORS:
            raise ValueError(
                f"detector must be one of {DETECTORS}, got {self.detector!r}"
            )
        # Filled here, not at use, so the source actually read enters
        # config_hash and provenance: it is part of what identifies the result.
        if self.detector_name is None:
            object.__setattr__(self, "detector_name", DETECTOR_NAMES[self.detector])
        if self.first_modno is None:
            object.__setattr__(self, "first_modno", DETECTOR_MODNOS[self.detector])
        if self.read_noise_fallback_kev is None:
            object.__setattr__(
                self,
                "read_noise_fallback_kev",
                DEFAULT_READ_NOISE_KEV[self.detector],
            )
        self._refuse_the_other_detector()
        if self.npt < 1:
            raise ValueError(f"npt must be positive, got {self.npt}")
        if self.photon_energy_kev <= 0:
            raise ValueError(
                f"photon_energy_kev must be positive, got {self.photon_energy_kev}"
            )
        if tuple(self.method) != METHOD:
            raise ValueError(
                f"method must be {METHOD}, got {tuple(self.method)}; "
                "it is given as a tuple because a method string resolves "
                "silently to a different integrator"
            )
        if not 0 <= self.mask_bits <= 0xFFFFFFFF:
            raise ValueError(
                f"mask_bits must fit a uint32 BadPixels field, got {self.mask_bits}"
            )
        if self.expected_lit_cells is not None:
            if not self.expected_lit_cells:
                raise ValueError(
                    "expected_lit_cells must name at least one cell, or be None "
                    "to measure the set from the data"
                )
            if not is_storage_cell_sequence(
                tuple(self.expected_lit_cells), CELLS_PER_TRAIN
            ):
                raise ValueError(
                    f"expected_lit_cells {list(self.expected_lit_cells)} is not a "
                    f"run of consecutive cells mod {CELLS_PER_TRAIN}, so no "
                    "JUNGFRAU storage-cell sequence can produce it"
                )
        if not 0.0 < self.lit_fraction_min < 1.0:
            raise ValueError(
                f"lit_fraction_min must lie in (0, 1), got {self.lit_fraction_min}"
            )
        if self.lit_gap_ratio <= 1.0:
            raise ValueError(f"lit_gap_ratio must exceed 1, got {self.lit_gap_ratio}")
        for name in ("read_noise_kev", "read_noise_fallback_kev"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.max_abs_kev <= 0:
            raise ValueError(f"max_abs_kev must be positive, got {self.max_abs_kev}")
        if self.cell_sample_trains < 1:
            raise ValueError(
                f"cell_sample_trains must be positive, got {self.cell_sample_trains}"
            )
        if self.min_modules < 1:
            raise ValueError(f"min_modules must be positive, got {self.min_modules}")
        if self.trains_per_block < 1:
            raise ValueError(
                f"trains_per_block must be positive, got {self.trains_per_block}"
            )
        if self.n_workers is not None and self.n_workers < 1:
            raise ValueError(f"n_workers must be positive, got {self.n_workers}")
        if self.selftest_frames < 1:
            raise ValueError(
                f"selftest_frames must be positive, got {self.selftest_frames}"
            )

    def __getstate__(self) -> dict[str, Any]:
        """Pickle by field name, not by position — see :mod:`analysis.common.config`."""
        return state_by_name(self)

    def __setstate__(self, state: Any) -> None:
        """Unpickle by field name; defined here because a mixin would not bind.

        ``dataclasses`` installs its own positional pair unless ``__getstate__``
        is in this class's own ``__dict__``.
        """
        restore_by_name(self, state)

    def _refuse_the_other_detector(self) -> None:
        """Refuse a config that mixes one detector with another's inputs.

        ``dataclasses.replace(cfg, detector="jf2")`` keeps whatever
        ``detector_name``, ``poni_file`` and ``static_mask_file`` were already
        set, so it would read jf2's frames through jf1's source name, geometry
        and mask. Nothing downstream would raise — a wrong PONI still yields a
        plausible-looking I(q), which is the same trap as the AGIPD beam centre (see
        pitfall 15) in another guise — so it is caught here.

        Only an unambiguous mix-up is refused: a field naming the *other*
        detector and not this one. Deliberately pointing a run at an unrelated
        path is still allowed.
        """
        others = [d for d in DETECTORS if d != self.detector]
        for other in others:
            if self.detector_name == DETECTOR_NAMES[other]:
                raise ValueError(
                    f"detector is {self.detector!r} but detector_name is "
                    f"{self.detector_name!r}, which belongs to {other!r}. "
                    "Build the config with config_for(proposal, run, detector) "
                    "rather than replacing `detector` on an existing one."
                )
        for field, path in (
            ("poni_file", self.poni_file),
            ("static_mask_file", self.static_mask_file),
        ):
            if path is None:
                continue
            name = Path(path).name
            if self.detector not in name and any(other in name for other in others):
                raise ValueError(
                    f"detector is {self.detector!r} but {field} is {path!r}, "
                    f"which names another detector. Integrating one detector's "
                    "frames through the other's geometry or mask yields a "
                    "plausible-looking I(q) and no error anywhere else."
                )

    @property
    def output_file(self) -> Path:
        """``{output_root}/r{run:04d}/jungfrau_waxs_{detector}.h5``."""
        return (
            Path(self.output_root)
            / f"r{self.run:04d}"
            / f"jungfrau_waxs_{self.detector}.h5"
        )

    @property
    def input_files(self) -> dict[str, str | None]:
        """The input files whose sha256 enters :meth:`config_hash`."""
        return {
            "poni_file": self.poni_file,
            "static_mask_file": self.static_mask_file,
        }

    @property
    def operational_fields(self) -> frozenset[str]:
        """Which fields :meth:`config_hash` excludes, for the provenance record."""
        return WAXS_OPERATIONAL_FIELDS


def config_for(proposal: int, run: int, detector: str, **overrides: object):
    """A config with the per-detector file defaults filled in."""
    overrides.setdefault("poni_file", default_poni_file(detector))
    overrides.setdefault("static_mask_file", default_static_mask_file(detector))
    return JungfrauWaxsConfig(
        proposal=proposal, run=run, detector=detector, **overrides
    )
