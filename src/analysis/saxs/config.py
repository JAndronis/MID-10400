"""Frozen configuration for the AGIPD SAXS integrator (``agipd_saxs``).

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
    "DEFAULT_BEAM_CENTER_PX",
    "DEFAULT_BEAM_CENTER_PY",
    "DEFAULT_GEOMETRY_FILE",
    "DEFAULT_PIXEL_MASK_FILE",
    "EXPECTED_BITS",
    "AgipdSaxsConfig",
    "file_sha256",
    "physical_cores",
]

#: Flattened AGIPD-1M pixel grid, module × slow-scan × fast-scan (context §6).
SHAPE: tuple[int, int] = (16 * 512, 128)
NPIX: int = SHAPE[0] * SHAPE[1]

#: The only integration method this pipeline accepts (context file §3 rule 1).
METHOD: tuple[str, str, str] = ("full", "csc", "cython")

_PROPOSAL_ROOT = "/gpfs/exfel/exp/MID/202601/p010400"

#: CrystFEL geometry in use for this beamtime.
DEFAULT_GEOMETRY_FILE = f"{_PROPOSAL_ROOT}/usr/geometry/geom_latest.geom"

#: The single hand-maintained AGIPD pixel mask; non-zero = excluded.
#:
#: It is the mask the live DAMNIT context used for the most recent
#: reintegration of the runs, and it covers both the generally bad pixels and
#: the anisotropic low-q lobe (integrator I4, option (a)) — there is
#: deliberately no second mask to keep in step with it. The older
#: ``usr/Shared/IA/custom_agipd_mask.npy`` is superseded and must not be
#: combined with this one. ``usr/masks`` is the same directory as
#: ``/gpfs/exfel/u/usr/MID/202601/p010400/masks``.
DEFAULT_PIXEL_MASK_FILE = f"{_PROPOSAL_ROOT}/usr/masks/mask_2026-09-08_AGIPD_SAXS.npy"

#: Where the per-run output file is written.
DEFAULT_OUTPUT_ROOT = f"{_PROPOSAL_ROOT}/scratch/agipd_saxs"

#: Beam centre in Fit2D pixel coordinates, agreed with the beamline scientist.
#:
#: These are **not** in the frame ``AGIPD_1MGeometry.to_pyfai_detector()``
#: produces, which puts the origin at the geometry origin so that PONI = 0 is
#: the beam. They are in the frame of ``geom.to_distortion_array()``, whose
#: origin is the corner of the assembled bounding box and whose coordinates are
#: therefore all positive. ``operator.build_operator`` installs that corner
#: array before calling ``setFit2D``, which is what ``extra_speckle``'s
#: ``setup.configuration.ConfigSAXS`` does and what these numbers were derived
#: against. Applying them to the ``to_pyfai_detector()`` frame instead moves the
#: beam by the offset between the two origins, so the two steps belong
#: together.
#:
#: ``DEFAULT_BEAM_CENTER_PX`` is the fast-scan (pyFAI ``poni2``) coordinate and
#: ``DEFAULT_BEAM_CENTER_PY`` the slow-scan (``poni1``) one, matching Fit2D's
#: ``centerX`` / ``centerY``.
DEFAULT_BEAM_CENTER_PX: float = 607.4598195630211
DEFAULT_BEAM_CENTER_PY: float = 672.076693118667

#: ``BadPixels`` bits seen in r0423 and r0426 (CLAUDE.md, image.mask). Any
#: other bit present in a run is a provenance flag and a warning, not a
#: failure: every bit in these files marks an unusable pixel, so the blanket
#: ``mask_bits`` stays correct, but an unrecorded bit must be looked at.
EXPECTED_BITS: frozenset[int] = frozenset({0, 1, 7, 8, 9, 12, 13})


#: Linux CPU topology, where a core's hyperthread siblings are listed.
CPU_TOPOLOGY_ROOT = Path("/sys/devices/system/cpu")


def physical_cores(root: Path = CPU_TOPOLOGY_ROOT) -> int | None:
    """Physical cores available to this process, or ``None`` off Linux.

    Counts distinct hyperthread-sibling groups over the CPUs in this process's
    affinity mask, so a core contributes once however many threads it exposes
    and a cgroup-restricted job is not told about cores it cannot use.

    :param root: sysfs CPU topology root; the tests point it at a fixture.
    """
    if not hasattr(os, "sched_getaffinity"):
        return None
    groups: set[str] = set()
    for cpu in os.sched_getaffinity(0):
        try:
            siblings = (root / f"cpu{cpu}/topology/thread_siblings_list").read_text()
        except OSError:
            return None
        groups.add(siblings.strip())
    return len(groups) or None


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AgipdSaxsConfig:
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
    #: Beam centre, or ``None`` for PONI = 0 at the geometry origin. Set
    #: together or not at all; see :data:`DEFAULT_BEAM_CENTER_PX` for the frame
    #: they are expressed in, which is not the one PONI = 0 lives in.
    beam_center_px: float | None = DEFAULT_BEAM_CENTER_PX
    beam_center_py: float | None = DEFAULT_BEAM_CENTER_PY
    npt: int = 500
    method: tuple[str, str, str] = METHOD
    unit: str = "q_nm^-1"
    # ── masks (P2, context file §6.3) ─────────────────────────────────────
    mask_bits: int = 0xFFFFFFFF
    expected_bits: frozenset[int] = EXPECTED_BITS
    use_asic_seams: bool = True
    pixel_mask_file: str | None = DEFAULT_PIXEL_MASK_FILE
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
        if (self.beam_center_px is None) != (self.beam_center_py is None):
            raise ValueError(
                "beam_center_px and beam_center_py must be set together or "
                f"both left None, got {self.beam_center_px!r} and "
                f"{self.beam_center_py!r}; half a beam centre would silently "
                "fall back to PONI = 0 in the other axis"
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
        """``{output_root}/r{run:04d}/agipd_saxs.h5`` (context file §7)."""
        return Path(self.output_root) / f"r{self.run:04d}" / "agipd_saxs.h5"

    @property
    def workers(self) -> int:
        """Worker count: ``n_workers``, else one per *physical* core.

        SMT is P6's question, so the default must not answer it. On the DAMNIT
        node ``sched_getaffinity`` reports 72 logical CPUs for 36 physical
        cores, and using it would silently run the hyperthreaded configuration
        while claiming one worker per core.
        """
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
            "geometry_file": self.geometry_file,
            "pixel_mask_file": self.pixel_mask_file,
        }

    @property
    def beam_center(self) -> tuple[float, float] | None:
        """``(px, py)`` in Fit2D order, or ``None`` for PONI = 0."""
        if self.beam_center_px is None or self.beam_center_py is None:
            return None
        return (self.beam_center_px, self.beam_center_py)

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
