"""Trains, row offsets, blocks and run checks for one JUNGFRAU detector.

The row model of :mod:`analysis.common.plan` carries over, with one thing that
does not transfer from AGIPD and would be a silent off-by-sixteen if assumed:

**``JUNGFRAU.frame_counts`` counts entries, not frames.** ``JUNGFRAU`` is a
``MultimodDetectorBase``, whose ``frame_counts`` is the INDEX entry count — one
per train — while each entry holds ``_frames_per_entry`` memory cells (16 here).
``AGIPD1M.frame_counts`` counts frames directly. So a train's rows are the *lit*
cells of its single entry, not its entry count.

EXtra-data's reader assumes one entry per train throughout — ``buffer_shape`` is
``(modules, trains) + entry_shape`` — so a train with two entries cannot be
represented at all, and :func:`build_plan` refuses the run rather than reading
it wrong.

Nothing here imports DAMNIT, so this module can back the pyBeamtime EuXFEL
reader later.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from analysis.common.plan import Block, RunPlan, TrainRecord, build_blocks
from analysis.common.status import FrameStatus
from analysis.waxs.config import JungfrauWaxsConfig

__all__ = [
    "Block",
    "RunPlan",
    "TrainRecord",
    "build_plan",
    "open_detector",
    "run_checks",
]

log = logging.getLogger(__name__)


class InconsistentEntries(RuntimeError):
    """A train holds a number of detector entries the reader cannot represent."""


def open_detector(cfg: JungfrauWaxsConfig, dc: Any) -> Any:
    """The ``JUNGFRAU`` component for this detector.

    ``detector_name`` is passed through rather than auto-detected:
    ``JUNGFRAU._det_name_pat`` matches both of this experiment's detectors, so
    with two in the run the auto-detection is ambiguous (§6 O1).
    """
    from extra_data.components import JUNGFRAU

    det = JUNGFRAU(
        dc,
        detector_name=cfg.detector_name,
        min_modules=cfg.min_modules,
        first_modno=cfg.first_modno,
    )
    if len(det.source_to_modno) != 1:
        raise ValueError(
            f"{det.detector_name} selects {len(det.source_to_modno)} modules; "
            "this pass integrates one JUNGFRAU-500K module per run because each "
            "carries its own PONI, its own static mask and its own q range "
            "(§3 D2). Point cfg.detector_name at a single detector, or pass "
            "modules= upstream."
        )
    return det


def _open_control(cfg: JungfrauWaxsConfig) -> Any:
    """Open the raw location, which is where the control sources live.

    ``open_run(..., data="proc")`` opens one location and proc holds only the
    corrected detector files: no timeserver, no XGM. The plan itself stays on
    proc, so that trains existing only in raw do not enter the ledger.
    """
    from extra_data import open_run

    return open_run(cfg.proposal, cfg.run, data="raw")


def _control_or_none(cfg: JungfrauWaxsConfig) -> Any:
    try:
        return _open_control(cfg)
    except Exception as error:  # noqa: BLE001 - recorded by run_checks
        log.warning("no raw location for the run checks: %r", error)
        return None


def build_plan(
    cfg: JungfrauWaxsConfig,
    lit_cells: tuple[int, ...],
    dc: Any = None,
    control_dc: Any = None,
    det: Any = None,
) -> RunPlan:
    """Build the run plan.

    :param lit_cells: the memory cells to integrate, measured from the data by
        :class:`analysis.waxs.cells.CellAccumulator`. They set how many rows a
        train owns, which is why the plan cannot be built before them. An empty
        set is allowed and gives a plan of zero rows: a run that saw no beam is
        not a broken run, and 43 of the proposal's are like that. Whether that
        is worth integrating is :func:`analysis.waxs.run.run_jungfrau_waxs`'s
        call, not this function's.
    :param dc: an open ``DataCollection``. When ``None``, the proc run named by
        ``cfg`` is opened.
    :param control_dc: where :func:`run_checks` looks for the timeserver and
        the XGM. Defaults to the raw location when this function opened the run
        itself, and to ``dc`` otherwise, which lets a mock run carry both.
    :param det: an already-open ``JUNGFRAU``; ``None`` opens one.

    :raises InconsistentEntries: a train holds other than one detector entry.
    """
    opened_here = dc is None
    if opened_here:
        from extra_data import open_run

        dc = open_run(cfg.proposal, cfg.run, data="proc")

    if det is None:
        det = open_detector(cfg, dc)
    counts = det.frame_counts  # entries per train, not frames
    rows_per_train = len(lit_cells)

    odd = {int(tid): int(c) for tid, c in counts.items() if int(c) not in (0, 1)}
    if odd:
        raise InconsistentEntries(
            f"{len(odd)} train(s) hold other than one JUNGFRAU entry, e.g. "
            f"{dict(list(odd.items())[:5])}. EXtra-data's reader shapes its "
            "output as (modules, trains, cells, ss, fs), so it cannot represent "
            "them and this pass would store the wrong rows rather than fail."
        )

    detector_trains = {int(tid): int(c) for tid, c in counts.items()}
    modules_present = _modules_present(dc, det)

    records: list[TrainRecord] = []
    row = 0
    for raw_train_id in dc.train_ids:
        train_id = int(raw_train_id)
        entries = detector_trains.get(train_id)
        if not entries:
            # No rows of its own; the offset is where the next train starts, so
            # the ledger stays monotonic. A train where no module wrote an
            # entry is NO_FRAMES; one where some but too few did is
            # MISSING_MODULES. With one module per detector (see
            # :func:`open_detector`) only the first can happen here, but the
            # distinction is kept so the ledger means the same thing as the
            # AGIPD one.
            status = (
                FrameStatus.NO_FRAMES
                if modules_present.get(train_id, 0) == 0
                else FrameStatus.MISSING_MODULES
            )
            records.append(TrainRecord(train_id, 0, row, status))
            continue
        records.append(TrainRecord(train_id, rows_per_train, row, FrameStatus.OK))
        row += rows_per_train

    if control_dc is None:
        control_dc = _control_or_none(cfg) if opened_here else dc

    blocks = build_blocks(records, cfg.trains_per_block)
    log.info(
        "run %d %s: %d trains, %d frames (%d lit cells), %d blocks",
        cfg.run,
        cfg.detector,
        len(records),
        row,
        rows_per_train,
        len(blocks),
    )
    return RunPlan(
        trains=tuple(records),
        blocks=tuple(blocks),
        n_frames=row,
        detector_name=det.detector_name,
        checks=run_checks(control_dc, det, counts, cfg, lit_cells),
    )


def _modules_present(dc: Any, det: Any) -> dict[int, int]:
    """How many detector modules wrote at least one entry, per train.

    ``det.frame_counts`` cannot answer this: ``JUNGFRAU`` has already discarded
    every train below ``min_modules`` by the time it exists.
    """
    import pandas as pd

    per_module = (
        pd.DataFrame(
            {
                src: dc.get_data_counts(src, det._main_data_key)
                for src in det.source_to_modno
            }
        )
        .fillna(0)
        .astype(np.uint64)
    )
    present = (per_module > 0).sum(axis=1)
    return {int(tid): int(value) for tid, value in present.items()}


def run_checks(
    dc: Any,
    det: Any,
    counts: Any,
    cfg: JungfrauWaxsConfig | None = None,
    lit_cells: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Provenance flags for the run.

    None of these select frames; each is recorded and a disagreement flagged.
    Every check is best-effort: a missing source makes the check unavailable,
    which is itself recorded, rather than failing the run before any data is
    read.

    The AGIPD quadrant-motor check has no counterpart here. The X-ray pulse
    check does, but it means something different: the JUNGFRAU records 8 lit
    cells where the machine delivers 155 pulses per train, so a difference is
    the expected state of affairs and is recorded as two numbers rather than
    flagged as a mismatch. Which 8 of the 155 pulses those cells sampled is
    open (§5 R5) and is why no pulse id is stored.
    """
    checks: dict[str, Any] = {
        "entries_per_train": sorted({int(c) for c in counts}),
        "frames_per_entry": int(getattr(det, "_frames_per_entry", 0)),
        "lit_cells": list(lit_cells),
    }
    if dc is None:
        checks.update(
            dict.fromkeys(
                ("xray_pulses", "xgm_photon_energy"),
                "unavailable: no control data (the raw location could not be opened)",
            )
        )
        return checks

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
        return {
            "constant_pattern": bool(pulses.is_constant_pattern()),
            "trains_compared": int(len(shared)),
            "xray_pulses_per_train": sorted({int(v) for v in pulse_counts[shared]})[:8],
            "lit_cells_per_train": len(lit_cells),
            "note": (
                "the JUNGFRAU samples a subset of the train's X-ray pulses; "
                "which subset is unresolved (WAXS context file §5 R5), so no "
                "pulse id is stored"
            ),
        }

    def energy_check() -> dict[str, Any]:
        """The machine's nominal photon energy, against the configured one.

        ``XGM.photon_energy_by_train`` returns **keV** already. No tolerance is
        invented (CLAUDE.md working rule 2): the two are called equal only
        within the float32 precision the XGM value carries, and any wider gap
        is reported for a person to settle, because q scales with it.
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
                    "photon energy: config %.4f keV, XGM %.4f keV (%.2f %%); "
                    "q scales with it",
                    cfg.photon_energy_kev,
                    mean_kev,
                    100 * relative,
                )
        return check

    attempt("xray_pulses", pulse_check)
    attempt("xgm_photon_energy", energy_check)
    return checks
