"""Trains, frame counts, row offsets, blocks and run checks.

The plan is the run's identity table. Every row of the output file is addressed
by the train it belongs to, never by its position in an array: a dropped train
would otherwise shift every later frame onto the wrong trainId.

``build_plan`` takes an optional ``DataCollection`` so callers can supply a run
that is already open — or a mock one — instead of going through ``open_run``.
Nothing here imports DAMNIT, so this module can back the pyBeamtime EuXFEL
reader later.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

# Re-exported: the row model is detector-agnostic and now lives in
# ``analysis.common.plan``, but the SAXS package and its tests name it here.
from analysis.common.plan import Block, RunPlan, TrainRecord, build_blocks
from analysis.saxs.config import AgipdSaxsConfig
from analysis.saxs.status import FrameStatus

__all__ = ["Block", "RunPlan", "TrainRecord", "build_plan", "run_checks"]

log = logging.getLogger(__name__)


def _open_detector(cfg: AgipdSaxsConfig, dc: Any):
    from extra_data.components import AGIPD1M

    return AGIPD1M(dc, detector_name=cfg.detector_name, min_modules=cfg.min_modules)


def _open_control(cfg: AgipdSaxsConfig) -> Any:
    """Open the raw location, which is where the control sources live.

    Proc holds only the corrected detector files, so :func:`run_checks` needs
    its own collection. The plan itself stays on proc: ``data="all"`` would put
    raw-only trains into the ledger and make a complete run look incomplete.
    """
    from extra_data import open_run

    return open_run(cfg.proposal, cfg.run, data="raw")


def build_plan(cfg: AgipdSaxsConfig, dc: Any = None, control_dc: Any = None) -> RunPlan:
    """Build the run plan.

    :param dc: an open ``DataCollection``. When ``None``, the proc run named by
        ``cfg`` is opened.
    :param control_dc: where :func:`run_checks` looks for the timeserver, the
        XGM and the quadrant motors. Defaults to the raw location when this
        function opened the run itself, and to ``dc`` otherwise, which is what
        lets a mock run carry both in one collection.

    Trains present in the run but missing from the detector selection get
    ``MISSING_MODULES``; trains with no frames get ``NO_FRAMES``. Neither owns
    any row of the frame table, so the table spans only the detector's trains
    while the ledger spans every train of the run.
    """
    opened_here = dc is None
    if opened_here:
        from extra_data import open_run

        dc = open_run(cfg.proposal, cfg.run, data="proc")

    det = _open_detector(cfg, dc)
    counts = det.frame_counts
    # Computed here rather than via det.train_id_to_ix, which is deprecated.
    first = (counts.cumsum() - counts).astype(np.int64)
    detector_trains = {int(tid): index for index, tid in enumerate(counts.index)}
    modules_present = _modules_present(dc, det)

    records: list[TrainRecord] = []
    row = 0
    for raw_train_id in dc.train_ids:
        train_id = int(raw_train_id)
        index = detector_trains.get(train_id)
        if index is None:
            # No rows of its own; the offset is where the next train starts, so
            # the ledger stays monotonic. A train where *no* module wrote a
            # frame is NO_FRAMES; one where some but too few did is
            # MISSING_MODULES. AGIPD1M cannot tell them apart — it drops both
            # before frame_counts exists — so the per-module counts are
            # consulted directly.
            status = (
                FrameStatus.NO_FRAMES
                if modules_present.get(train_id, 0) == 0
                else FrameStatus.MISSING_MODULES
            )
            records.append(TrainRecord(train_id, 0, row, status))
            continue
        n_frames = int(counts.iloc[index])
        if row != int(first.iloc[index]):
            raise AssertionError(
                f"row offset disagreement for train {train_id}: counted {row}, "
                f"frame_counts says {int(first.iloc[index])}"
            )
        records.append(
            TrainRecord(
                train_id,
                n_frames,
                row,
                FrameStatus.OK if n_frames else FrameStatus.NO_FRAMES,
            )
        )
        row += n_frames

    if control_dc is None:
        control_dc = _control_or_none(cfg) if opened_here else dc

    blocks = build_blocks(records, cfg.trains_per_block)
    n_frames = int(counts.sum())
    log.info(
        "run %d: %d trains, %d frames, %d blocks",
        cfg.run,
        len(records),
        n_frames,
        len(blocks),
    )
    return RunPlan(
        trains=tuple(records),
        blocks=tuple(blocks),
        n_frames=n_frames,
        detector_name=det.detector_name,
        checks=run_checks(control_dc, det, counts, cfg),
    )


def _control_or_none(cfg: AgipdSaxsConfig) -> Any:
    """:func:`_open_control`, or ``None`` when raw is not readable."""
    try:
        return _open_control(cfg)
    except Exception as error:  # noqa: BLE001 - recorded by run_checks
        log.warning("no raw location for the run checks: %r", error)
        return None


def _modules_present(dc: Any, det: Any) -> dict[int, int]:
    """How many detector modules wrote at least one frame, per train.

    ``det.frame_counts`` cannot answer this: ``AGIPD1M`` has already discarded
    every train below ``min_modules`` by the time it exists.
    """
    import pandas as pd

    per_module = (
        pd.DataFrame(
            {src: dc.get_data_counts(src, "image.data") for src in det.source_to_modno}
        )
        .fillna(0)
        .astype(np.uint64)
    )
    present = (per_module > 0).sum(axis=1)
    return {int(tid): int(value) for tid, value in present.items()}


def run_checks(
    dc: Any, det: Any, counts: Any, cfg: AgipdSaxsConfig | None = None
) -> dict[str, Any]:
    """Provenance flags for the run.

    None of these select frames in v1 (integrator I1); each is recorded and a
    disagreement is flagged. Every check is best-effort: a missing source makes
    the check unavailable, which is itself recorded, rather than failing the
    run before any data is read.

    :param dc: a collection carrying the *control* sources — see
        :func:`_open_control`. ``None`` records every check as unavailable.
    :param cfg: when given, the photon-energy check compares the machine's
        nominal energy against ``cfg.photon_energy_kev``.
    """
    checks: dict[str, Any] = {}
    if dc is None:
        return dict.fromkeys(
            ("xray_pulses", "quadrant_motors", "xgm_photon_energy"),
            "unavailable: no control data (the raw location could not be opened)",
        )

    def attempt(name: str, fn) -> None:
        try:
            checks[name] = fn()
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            checks[name] = f"unavailable: {error!r}"
            log.warning("run check %r unavailable: %r", name, error)

    def pulse_check() -> dict[str, Any]:
        from extra.components import XrayPulses

        pulses = XrayPulses(dc)
        pulse_counts = pulses.pulse_counts()
        shared = counts.index.intersection(pulse_counts.index)
        mismatched = int((counts[shared] != pulse_counts[shared]).sum())
        return {
            "constant_pattern": bool(pulses.is_constant_pattern()),
            "trains_compared": int(len(shared)),
            "frames_ne_xray_pulses": mismatched,
        }

    def quadrant_check() -> dict[str, Any]:
        from extra.components import AGIPD1MQuadrantMotors

        positions = AGIPD1MQuadrantMotors(dc).positions(compressed=True)
        moved = len(positions) > 1
        return {"n_positions": int(len(positions)), "quadrants_moved": bool(moved)}

    def energy_check() -> dict[str, Any]:
        """The machine's nominal photon energy, against the configured one.

        ``XGM.photon_energy_by_train`` returns **keV** already (it converts
        ``pulseEnergy.wavelengthUsed``, in nm, and tags the result keV), so
        there is no eV to divide out.

        No tolerance is invented here: the two are
        called equal only within the float32 precision the XGM value carries,
        and any wider gap is reported for a person to settle. It matters
        because q scales with the energy, so a disagreement of x is a
        disagreement of x in every q value this pass writes.
        """
        from extra.components import XGM

        energies = np.asarray(XGM(dc).photon_energy_by_train())
        finite = energies[np.isfinite(energies)]
        if finite.size == 0:
            return {"available": False}
        mean_kev = float(finite.mean())
        check: dict[str, Any] = {
            "available": True,
            "mean_kev": mean_kev,
            "spread_fraction": float(
                (finite.max() - finite.min()) / max(abs(mean_kev), 1e-12)
            ),
        }
        if cfg is not None:
            relative = abs(mean_kev - cfg.photon_energy_kev) / cfg.photon_energy_kev
            check["config_kev"] = cfg.photon_energy_kev
            check["rel_difference"] = relative
            check["agrees"] = bool(relative <= float(np.finfo(np.float32).eps))
            if not check["agrees"]:
                log.warning(
                    "photon energy: config %.4f keV, XGM %.4f keV "
                    "(%.2f %%); q scales with it",
                    cfg.photon_energy_kev,
                    mean_kev,
                    100 * relative,
                )
        return check

    attempt("xray_pulses", pulse_check)
    attempt("quadrant_motors", quadrant_check)
    attempt("xgm_photon_energy", energy_check)
    return checks
