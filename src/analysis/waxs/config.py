"""Frozen configuration for the JUNGFRAU WAXS integrator (``jungfrau_waxs``).

One config per detector: the two JUNGFRAU-500Ks have different geometries,
different static masks, different q ranges and different orientations, so each
gets its own PONI, its own mask and its own output file (WAXS context file §3
D2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyFAI.units import hc  # keV·Å, derived from scipy CODATA (pyFAI/units.py:66)

from analysis.common.config import (
    OPERATIONAL_FIELDS,
    config_sha256,
    restore_by_name,
    state_by_name,
)
from analysis.common.cpu import physical_cores

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

#: Memory cells per train. Fixed by the burst-mode readout, unlike AGIPD's
#: ragged frame table; ``plan.build_plan`` still reads it per run.
CELLS_PER_TRAIN: int = 16

#: The only integration method this pass accepts (AGIPD context file §3 rule 1).
METHOD: tuple[str, str, str] = ("full", "csc", "cython")

#: The two detectors, by the short name used in paths and output filenames.
DETECTORS: tuple[str, ...] = ("jf1", "jf2")

#: EuXFEL source prefixes, from ``lsxfel`` on the r0423 proc files (§6 O1):
#: ``MID_EXP_JF500K1/CORR/JNGFR01:daqOutput`` and
#: ``MID_EXP_JF500K2/CORR/JNGFR02:daqOutput``.
#:
#: Both names match ``JUNGFRAU._det_name_pat``, so with two JUNGFRAUs in a run
#: ``_find_detector_name`` is ambiguous and auto-detection cannot work — which
#: is why these are filled in rather than left to it.
DETECTOR_NAMES: dict[str, str] = {
    "jf1": "MID_EXP_JF500K1",
    "jf2": "MID_EXP_JF500K2",
}

#: The module number in each detector's source name: ``JNGFR01`` / ``JNGFR02``.
#: Passed as ``first_modno`` so each single-module detector reports modno 1 and
#: ``n_modules`` 1, rather than jf2 claiming to be the second of two.
DETECTOR_MODNOS: dict[str, int] = {"jf1": 1, "jf2": 2}

#: Corrected JUNGFRAU files, one per detector per sequence. r0423 has six
#: sequences of 500 trains for each detector.
PROC_FILE_PATTERN = "CORR-R{run:04d}-JNGFR{modno:02d}-S{seq:05d}.h5"

_PROPOSAL_ROOT = "/gpfs/exfel/exp/MID/202601/p010400"

#: ``BadPixels`` bits seen on both detectors in r0423 (WAXS context file §2).
#: Bit 21 ``WRONG_GAIN_VALUE`` is new against AGIPD, and bit 22
#: ``NON_STANDARD_SIZE`` **is** set here, which is why no seam mask is needed
#: (§3 D5).
EXPECTED_BITS: frozenset[int] = frozenset({0, 1, 21, 22})

#: The memory cells that carry photons **in the science block, r0379–r0500**:
#: 0–6 and 15, not 0–7. Eight storage cells starting at 15, so the sequence wraps
#: 15 → 0 → … → 6.
#:
#: This is one of the proposal's four patterns, not *the* pattern: r0054–r0378
#: read all 16 cells, r0052/53/56 read only cell 15, and 43 runs saw no beam at
#: all. See :mod:`analysis.waxs.cells` for the full map. The pass therefore
#: measures the set per run and checks its *shape*; this constant is only the
#: expectation a caller may pin :attr:`JungfrauWaxsConfig.expected_lit_cells` to
#: when reprocessing the science block must be bit-reproducible.
#:
#: Cell 15 runs a little lower than 0–6 (0.333 against 0.352 on jf2 r0423),
#: which is the usual JUNGFRAU first-storage-cell behaviour and a reason to look
#: at it separately before pooling it with the rest.
#:
#: An earlier reading of ``(0…7)`` came from indexing an exported train by array
#: *position*: positions 0–7 are the lit ones, and the ``data.memoryCell`` values
#: they carry are 0–6 and 15. That is precisely the inference CLAUDE.md pitfall 4
#: forbids, and §3 D4's loud failure is what caught it.
EXPECTED_LIT_CELLS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 15)

#: Readout noise per detector, for runs that read all 16 storage cells and so
#: have no dark cell to measure it from.
#:
#: The median over the 159 (run, detector) results per detector where the dark
#: side is unambiguous, spanning r0001–r0500: jf1 0.3440 keV (range
#: 0.3153–0.3724, ±8.3 %), jf2 0.3195 keV (range 0.3147–0.3370, ±3.5 %). A
#: measurement from the run's own dark cells always wins over this; see
#: :func:`analysis.waxs.run._error_model` for the order of precedence and
#: :meth:`analysis.waxs.cells.CellAccumulator._read_noise` for why an error here
#: costs so little.
DEFAULT_READ_NOISE_KEV: dict[str, float] = {"jf1": 0.3440, "jf2": 0.3195}

#: The shared operational set plus ``expected_lit_cells``, which only *gates*.
#:
#: Pinning it makes the pass refuse a run whose measured pattern differs — it
#: raises before the output file is opened, so it can never change a value in
#: ``/frames``. By CLAUDE.md pitfall 12 it therefore stays out of the hash, and
#: pinning it for a reprocess does not invalidate files written without it.
#:
#: ``lit_fraction_min`` and ``lit_gap_ratio`` are the opposite case and stay in:
#: they decide which cells are integrated, and so which rows exist at all.
WAXS_OPERATIONAL_FIELDS: frozenset[str] = OPERATIONAL_FIELDS | {"expected_lit_cells"}

#: Where the per-run output files are written.
DEFAULT_OUTPUT_ROOT = f"{_PROPOSAL_ROOT}/scratch/jungfrau_waxs"


def default_poni_file(detector: str) -> str:
    """pyFAI geometry for one detector (WAXS context file §5 R4).

    ``usr/geometry`` is the same directory as
    ``/gpfs/exfel/u/usr/MID/202601/p010400/geometry``, the way ``usr/masks`` is
    for the AGIPD pixel mask.
    """
    return f"{_PROPOSAL_ROOT}/usr/geometry/{detector}.poni"


def default_static_mask_file(detector: str) -> str:
    """Native pyFAI static mask for one detector (§6 O2: non-zero = excluded).

    Named ``jf1_mask.edf`` / ``jf2_mask.edf``, unlike the PONIs, which are bare
    ``jf1.poni`` / ``jf2.poni``.
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
class JungfrauWaxsConfig:
    """Immutable run configuration for one detector.

    ``poni_file`` and ``static_mask_file`` default from :attr:`detector` and may
    be ``None`` only when the caller supplies the geometry directly, which is
    what the unit tests do with a synthetic PONI. Every real run must set both,
    so that their sha256 enters :meth:`config_hash`.
    """

    proposal: int
    run: int
    detector: str
    #: EuXFEL source prefix. ``None`` fills it from :data:`DETECTOR_NAMES`,
    #: which is what every real run wants: ``JUNGFRAU._det_name_pat`` matches
    #: both of this experiment's detectors, so auto-detection is ambiguous with
    #: two of them in the run. Set it explicitly only for a run whose source
    #: names differ from r0423's.
    detector_name: str | None = None
    #: ``None`` fills it from :data:`DETECTOR_MODNOS`.
    first_modno: int | None = None
    poni_file: str | None = None
    static_mask_file: str | None = None
    photon_energy_kev: float = 9.04
    npt: int = 500
    method: tuple[str, str, str] = METHOD
    unit: str = "q_nm^-1"
    # ── masks (§3 D5) ─────────────────────────────────────────────────────
    mask_bits: int = 0xFFFFFFFF
    expected_bits: frozenset[int] = EXPECTED_BITS
    # ── lit-cell selection (§3 D4) ────────────────────────────────────────
    #: Pins the lit set, refusing the run if the data disagree. ``None`` — the
    #: default — measures it instead and checks only that its *shape* is a
    #: storage-cell sequence, which is the check that holds for every run: the
    #: proposal used four different patterns between r0001 and r0500. Pin it to
    #: :data:`EXPECTED_LIT_CELLS` for the r0379–r0500 science block when a
    #: reprocess must refuse anything that has drifted.
    expected_lit_cells: tuple[int, ...] | None = None
    #: Half a photon: a pixel above this held at least one.
    lit_threshold_kev: float = 4.5
    #: Fraction of kept pixels above :attr:`lit_threshold_kev` for a cell to
    #: count as lit. The geometric centre of the one empty band in the pooled
    #: sample of 16 × 746 measured fractions — ``5.654e-05 → 4.083e-03``, a
    #: factor of 72 with every other step in the sample at most 1.5 — so it
    #: clears both populations by 8.5×, and any value from 1e-4 to 1e-3 gives
    #: the identical classification. See :mod:`analysis.waxs.cells`.
    lit_fraction_min: float = 4.805e-4
    #: Among the cells that clear the floor, a step larger than this splits them
    #: again. It never fires on the proposal's runs; it is what keeps a
    #: uniformly attenuated run splitting on the lit-to-dark step rather than on
    #: the absolute level.
    lit_gap_ratio: float = 30.0
    cell_sample_trains: int = 8
    # ── error model (§3 D3) ───────────────────────────────────────────────
    #: ``None`` measures the readout noise from the dark cells of the sampled
    #: trains, which is what the context file asks for; a float overrides it
    #: even when the measurement is available.
    read_noise_kev: float | None = None
    #: Used only when the run reads every storage cell, leaving no dark cell to
    #: measure from. ``None`` means "fill it from :data:`DEFAULT_READ_NOISE_KEV`"
    #: and is resolved in ``__post_init__``, so a *constructed* config never
    #: holds ``None`` here — there is no way to express "no fallback, refuse
    #: instead", and `analysis.waxs.run._error_model` says why that is right.
    #: Resolving it at construction rather than at use is what puts the number
    #: actually used into :meth:`config_hash` (CLAUDE.md pitfall 12).
    read_noise_fallback_kev: float | None = None
    # ── data check (§3 D6) ────────────────────────────────────────────────
    #: ``data.mask`` misses pixels reaching ±1.8e5 keV on jf2. Any |x| above
    #: this in a frame routes it to ``DATA_CHECK_FAILED`` rather than trusting
    #: either mask to have caught it. The kept-region maximum is ~74 keV.
    max_abs_kev: float = 1.0e3
    # ── run, scheduling and output ────────────────────────────────────────
    min_modules: int = 1
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
        # Filled here rather than left as None so that the source actually used
        # enters config_hash and the provenance record: which Karabo source a
        # run was integrated from is part of what identifies the result.
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
                f"method must be {METHOD} (AGIPD context file §3 rule 1), "
                f"got {tuple(self.method)}"
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
        """Unpickle by field name, refusing a state this class disagrees with.

        Defined here rather than inherited: ``dataclasses`` installs its own
        positional pair unless ``__getstate__`` is in this class's ``__dict__``.
        """
        restore_by_name(self, state)

    def _refuse_the_other_detector(self) -> None:
        """Refuse a config that mixes one detector with another's inputs.

        ``dataclasses.replace(cfg, detector="jf2")`` keeps whatever
        ``detector_name``, ``poni_file`` and ``static_mask_file`` were already
        set, so it would read jf2's frames through jf1's source name, geometry
        and mask. Nothing downstream would raise — a wrong PONI still yields a
        plausible-looking I(q), which is the AGIPD beam-centre trap (CLAUDE.md
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
        """``{output_root}/r{run:04d}/jungfrau_waxs_{detector}.h5`` (§3 D2)."""
        return (
            Path(self.output_root)
            / f"r{self.run:04d}"
            / f"jungfrau_waxs_{self.detector}.h5"
        )

    @property
    def workers(self) -> int:
        """Worker count: ``n_workers``, else one per *physical* core.

        The SMT question is the AGIPD pass's P6 and is not answered here
        (CLAUDE.md pitfall 10).
        """
        import os

        if self.n_workers is not None:
            return self.n_workers
        physical = physical_cores()
        if physical:
            return physical
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))
        return os.cpu_count() or 1

    @property
    def input_files(self) -> dict[str, str | None]:
        """The input files whose sha256 enters :meth:`config_hash`."""
        return {
            "poni_file": self.poni_file,
            "static_mask_file": self.static_mask_file,
        }

    @property
    def wavelength_m(self) -> float:
        """Photon wavelength in metres, from the configured photon energy.

        The PONI carries its own wavelength; ``operator.build_operator``
        asserts the two agree rather than letting either win silently (§5 R4).
        """
        return hc / self.photon_energy_kev * 1e-10

    def config_hash(self) -> str:
        """sha256 over the result-affecting fields plus each input file's sha256.

        Purely operational fields — worker count, block size, output root,
        ``allow_incomplete``, ``overwrite``, ``selftest_frames`` — are
        deliberately **excluded**: they change how the pass runs, never what it
        stores, and covering them made a rerun at a different worker count
        refuse its own output. See :mod:`analysis.common.config`. They are still
        recorded in full in the provenance record.

        ``expected_lit_cells`` joins them here: it refuses a run before the
        output file is opened, so it cannot change a stored number either.
        """
        return config_sha256(self, self.input_files, self.operational_fields)

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
