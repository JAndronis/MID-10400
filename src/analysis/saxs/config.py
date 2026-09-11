"""Frozen configuration for the first-pass AGIPD SAXS integrator.

P1 scope: only the fields the operator, the sparse kernels and the self-test
need (context file §5). The masking, planning, worker and writer fields arrive
with their own phases.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from pyFAI.units import hc  # keV·Å, derived from scipy CODATA (pyFAI/units.py:66)

__all__ = [
    "DEFAULT_CUSTOM_MASK_FILE",
    "DEFAULT_GEOMETRY_FILE",
    "DEFAULT_LOBE_MASK_FILE",
    "EXPECTED_BITS",
    "FirstPassConfig",
    "file_sha256",
]

#: Flattened AGIPD-1M pixel grid, module × slow-scan × fast-scan (context §6).
SHAPE: tuple[int, int] = (16 * 512, 128)
NPIX: int = SHAPE[0] * SHAPE[1]

#: The only integration method this pipeline accepts (context file §3 rule 1).
METHOD: tuple[str, str, str] = ("full", "csc", "cython")

_PROPOSAL_ROOT = "/gpfs/exfel/exp/MID/202601/p010400"

#: CrystFEL geometry in use for this beamtime.
DEFAULT_GEOMETRY_FILE = f"{_PROPOSAL_ROOT}/usr/geometry/geom_latest.geom"

#: Hand-drawn AGIPD mask; non-zero = excluded.
DEFAULT_CUSTOM_MASK_FILE = f"{_PROPOSAL_ROOT}/usr/Shared/IA/custom_agipd_mask.npy"

#: Static pixel mask covering the anisotropic low-q lobe (integrator I4,
#: option (a)). This is the mask the live DAMNIT context used for the most
#: recent reintegration of the runs; ``usr/masks`` is the same directory as
#: ``/gpfs/exfel/u/usr/MID/202601/p010400/masks``.
DEFAULT_LOBE_MASK_FILE = f"{_PROPOSAL_ROOT}/usr/masks/mask_2026-09-08_AGIPD_SAXS.npy"

#: Where the per-run output file is written.
DEFAULT_OUTPUT_ROOT = f"{_PROPOSAL_ROOT}/scratch/saxs_first_pass"

#: ``BadPixels`` bits seen in r0423 and r0426 (CLAUDE.md, image.mask). Any
#: other bit present in a run is a provenance flag and a warning, not a
#: failure: every bit in these files marks an unusable pixel, so the blanket
#: ``mask_bits`` stays correct, but an unrecorded bit must be looked at.
EXPECTED_BITS: frozenset[int] = frozenset({0, 1, 7, 8, 9, 12, 13})


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FirstPassConfig:
    """Immutable run configuration.

    ``geometry_file`` may be ``None`` only when the caller supplies a geometry
    object directly (the synthetic geometry used by the P1 unit tests). Every
    real run must set it, so that its sha256 enters :meth:`config_hash`.
    """

    proposal: int
    run: int
    geometry_file: str | None = DEFAULT_GEOMETRY_FILE
    sdd_m: float = 7.531
    photon_energy_kev: float = 9.04
    npt: int = 500
    method: tuple[str, str, str] = METHOD
    unit: str = "q_nm^-1"
    # ── masks (P2, context file §6.3) ─────────────────────────────────────
    mask_bits: int = 0xFFFFFFFF
    expected_bits: frozenset[int] = EXPECTED_BITS
    use_asic_seams: bool = True
    custom_mask_file: str | None = DEFAULT_CUSTOM_MASK_FILE
    lobe_mask_file: str | None = DEFAULT_LOBE_MASK_FILE
    base_mask_trains: int = 8
    # ── run, scheduling and output (P3, context file §5) ──────────────────
    detector_name: str | None = None
    min_modules: int = 16
    trains_per_block: int = 4
    n_workers: int | None = None
    selftest_frames: int = 8
    output_root: str = DEFAULT_OUTPUT_ROOT
    allow_incomplete: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.npt < 1:
            raise ValueError(f"npt must be positive, got {self.npt}")
        if self.sdd_m <= 0:
            raise ValueError(f"sdd_m must be positive, got {self.sdd_m}")
        if self.photon_energy_kev <= 0:
            raise ValueError(
                f"photon_energy_kev must be positive, got {self.photon_energy_kev}"
            )
        if tuple(self.method) != METHOD:
            raise ValueError(
                f"method must be {METHOD} (context file §3 rule 1), "
                f"got {tuple(self.method)}"
            )
        if not 0 <= self.mask_bits <= 0xFFFFFFFF:
            raise ValueError(
                f"mask_bits must fit a uint32 BadPixels field, got {self.mask_bits}"
            )
        if self.base_mask_trains < 1:
            raise ValueError(
                f"base_mask_trains must be positive, got {self.base_mask_trains}"
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

    @property
    def output_file(self) -> Path:
        """``{output_root}/r{run:04d}/saxs_first_pass.h5`` (context file §7)."""
        return Path(self.output_root) / f"r{self.run:04d}" / "saxs_first_pass.h5"

    @property
    def workers(self) -> int:
        """Worker count: ``n_workers``, else one per core.

        SMT is only considered after P6, so this is the physical-core count
        where the platform reports it and ``os.cpu_count()`` otherwise.
        """
        if self.n_workers is not None:
            return self.n_workers
        return (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )

    @property
    def input_files(self) -> dict[str, str | None]:
        """The input files whose sha256 enters :meth:`config_hash`."""
        return {
            "geometry_file": self.geometry_file,
            "custom_mask_file": self.custom_mask_file,
            "lobe_mask_file": self.lobe_mask_file,
        }

    @property
    def wavelength_m(self) -> float:
        """Photon wavelength in metres."""
        return hc / self.photon_energy_kev * 1e-10

    def config_hash(self) -> str:
        """sha256 over every field plus the sha256 of each input file.

        The ``<pkg>`` git commit and the package versions join this in the
        provenance record written by ``writer.finalise`` (context file §7);
        they are deliberately not folded in here, so that a rerun of the same
        configuration can still resume.
        """
        payload: dict[str, object] = asdict(self)
        payload["method"] = list(self.method)
        payload["expected_bits"] = sorted(self.expected_bits)
        payload["input_file_sha256"] = {
            name: file_sha256(path) if path else None
            for name, path in self.input_files.items()
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
