"""Frozen configuration for the AGIPD SAXS integrator (``agipd_saxs``).

One config per run, covering the operator, the masks, the plan and the output.
Anything that can change a stored number enters :meth:`config_hash`.
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
    "AgipdSaxsConfig",
    "DEFAULT_BEAM_CENTER_PX",
    "DEFAULT_BEAM_CENTER_PY",
    "DEFAULT_GEOMETRY_FILE",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PIXEL_MASK_FILE",
    "DEFAULT_PIXEL_SUMS_ROOT",
    "EXPECTED_BITS",
    "METHOD",
    "NPIX",
    "SAXS_OPERATIONAL_FIELDS",
    "SHAPE",
]

#: Modules of the AGIPD-1M.
N_MODULES = 16

#: Flattened AGIPD-1M pixel grid, module × slow-scan × fast-scan.
SHAPE: tuple[int, int] = (N_MODULES * 512, 128)
NPIX: int = SHAPE[0] * SHAPE[1]

#: The only integration method this pipeline accepts.
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

#: Where the per-pixel window sums are written (context file §15).
#:
#: Not scratch: scratch does not survive the move to tape, and once proc is
#: taped these sums are the only 2D record of a run left on disk.
DEFAULT_PIXEL_SUMS_ROOT = f"{_PROPOSAL_ROOT}/usr/cached_files/agipd_pixel_sums"

#: Fields the frame table's hash excludes. The shared operational set, plus the
#: two window-sum fields: those sums go to a separate file with a hash of its
#: own, so neither field can change a number in ``agipd_saxs.h5`` — and adding
#: them left the hash of every file already written unchanged.
SAXS_OPERATIONAL_FIELDS: frozenset[str] = OPERATIONAL_FIELDS | {
    "pixel_sum_trains",
    "pixel_sums_root",
}

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

#: ``BadPixels`` bits seen in r0423 and r0426. Any
#: other bit present in a run is a provenance flag and a warning, not a
#: failure: every bit in these files marks an unusable pixel, so the blanket
#: ``mask_bits`` stays correct, but an unrecorded bit must be looked at.
EXPECTED_BITS: frozenset[int] = frozenset({0, 1, 7, 8, 9, 12, 13})


@dataclass(frozen=True, slots=True)
class AgipdSaxsConfig(PassConfigMembers):
    """Immutable run configuration.

    ``geometry_file`` may be ``None`` only when the caller supplies a geometry
    object directly, which is what the unit tests do with a synthetic one. Every
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
    # ── masks ─────────────────────────────────────
    mask_bits: int = 0xFFFFFFFF
    expected_bits: frozenset[int] = EXPECTED_BITS
    use_asic_seams: bool = True
    pixel_mask_file: str | None = DEFAULT_PIXEL_MASK_FILE
    base_mask_trains: int = 8
    # ── run, scheduling and output ──────────────────
    detector_name: str | None = None
    min_modules: int = 16
    trains_per_block: int = 4
    n_workers: int | None = None
    selftest_frames: int = 8
    output_root: str = DEFAULT_OUTPUT_ROOT
    allow_incomplete: bool = False
    overwrite: bool = False
    # ── per-pixel window sums (context file §15) ──────
    #: Trains per window of per-pixel photon sums, or ``None`` for none. A
    #: window is a range of train ids from the run's first train, not a count
    #: of trains. ``overwrite`` never applies to these: see §15.3.
    pixel_sum_trains: int | None = 10
    pixel_sums_root: str = DEFAULT_PIXEL_SUMS_ROOT

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
                f"method must be {METHOD}, got {tuple(self.method)}; "
                "it is given as a tuple because a method string resolves "
                "silently to a different integrator"
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
        if self.pixel_sum_trains is not None:
            if self.pixel_sum_trains < 1:
                raise ValueError(
                    "pixel_sum_trains must be positive or None, got "
                    f"{self.pixel_sum_trains}"
                )
            if self.min_modules != N_MODULES:
                raise ValueError(
                    f"window sums need all {N_MODULES} modules, but min_modules is "
                    f"{self.min_modules}: a missing module reads as fill values "
                    "that pass image.mask == 0. Set pixel_sum_trains=None to "
                    "integrate with fewer modules"
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

    @property
    def output_file(self) -> Path:
        """``{output_root}/r{run:04d}/agipd_saxs.h5``."""
        return Path(self.output_root) / f"r{self.run:04d}" / "agipd_saxs.h5"

    @property
    def pixel_sums_file(self) -> Path:
        """``{pixel_sums_root}/r{run:04d}/agipd_pixel_sums.h5``."""
        return Path(self.pixel_sums_root) / f"r{self.run:04d}" / "agipd_pixel_sums.h5"

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
    def operational_fields(self) -> frozenset[str]:
        """Which fields :meth:`config_hash` excludes, for the provenance record."""
        return SAXS_OPERATIONAL_FIELDS
