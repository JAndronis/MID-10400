"""Frozen configuration for the JUNGFRAU WAXS integrator (``jungfrau_waxs``).

One config per detector: the two JUNGFRAU-500Ks have different geometries,
different static masks, different q ranges and different orientations, so each
gets its own PONI, its own mask and its own output file (WAXS context file §3
D2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pyFAI.units import hc  # keV·Å, derived from scipy CODATA (pyFAI/units.py:66)

from analysis.common.cpu import file_sha256, physical_cores

__all__ = [
    "DETECTORS",
    "DETECTOR_MODNOS",
    "DETECTOR_NAMES",
    "PROC_FILE_PATTERN",
    "EXPECTED_BITS",
    "EXPECTED_LIT_CELLS",
    "METHOD",
    "MODULE_SHAPE",
    "NPIX",
    "JungfrauWaxsConfig",
    "default_poni_file",
    "default_static_mask_file",
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

#: The memory cells that carry photons, measured on r0423 for both detectors:
#: 42–53 % of pixels above half a photon against ≤ 0.08 % in cells 8–15. The
#: pass detects the set from the data and **fails** if it is not this (§3 D4);
#: the constant is the expectation, never the selection.
EXPECTED_LIT_CELLS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

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
    expected_lit_cells: tuple[int, ...] = EXPECTED_LIT_CELLS
    #: Half a photon: a pixel above this held at least one.
    lit_threshold_kev: float = 4.5
    #: Fraction of kept pixels above the threshold for a cell to count as lit.
    #: The measured gap is 42–53 % against ≤ 0.08 %, so anything in 0.01–0.30
    #: separates them; the midpoint is chosen to sit far from both.
    lit_fraction_min: float = 0.10
    cell_sample_trains: int = 8
    # ── error model (§3 D3) ───────────────────────────────────────────────
    #: ``None`` measures the readout noise from the dark cells of the sampled
    #: trains, which is what the context file asks for; a float overrides it.
    read_noise_kev: float | None = None
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
        if not self.expected_lit_cells:
            raise ValueError("expected_lit_cells must name at least one cell")
        if not 0.0 < self.lit_fraction_min < 1.0:
            raise ValueError(
                f"lit_fraction_min must lie in (0, 1), got {self.lit_fraction_min}"
            )
        if self.read_noise_kev is not None and self.read_noise_kev <= 0:
            raise ValueError(
                f"read_noise_kev must be positive, got {self.read_noise_kev}"
            )
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

    def _refuse_the_other_detector(self) -> None:
        """Refuse a config that mixes one detector with another's inputs.

        ``dataclasses.replace(cfg, detector="jf2")`` keeps whatever
        ``detector_name``, ``poni_file`` and ``static_mask_file`` were already
        set, so it would read jf2's frames through jf1's source name, geometry
        and mask. Nothing downstream would raise — a wrong PONI still yields a
        plausible-looking I(q), which is the AGIPD beam-centre trap (CLAUDE.md
        pitfall 14) in another guise — so it is caught here.

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
        """sha256 over every field plus the sha256 of each input file."""
        payload: dict[str, object] = asdict(self)
        payload["method"] = list(self.method)
        payload["expected_bits"] = sorted(self.expected_bits)
        payload["expected_lit_cells"] = list(self.expected_lit_cells)
        payload["input_file_sha256"] = {
            name: file_sha256(path) if path else None
            for name, path in self.input_files.items()
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def config_for(proposal: int, run: int, detector: str, **overrides: object):
    """A config with the per-detector file defaults filled in."""
    overrides.setdefault("poni_file", default_poni_file(detector))
    overrides.setdefault("static_mask_file", default_static_mask_file(detector))
    return JungfrauWaxsConfig(
        proposal=proposal, run=run, detector=detector, **overrides
    )
