"""The triage's β checks and optics state, over `analysis.xpcs.snr`'s measurements.

The SNR verdict uses the fixed `analysis.xpcs.snr.REFERENCE_BETA`. `measure_beta`,
the median over whichever runs pass `beta_eligible`, is a cross-check only: it
moves with the run set, because a run that loses contrast before the first lag
still passes eligibility with a low intercept. `reference_beta` re-measures the
fixed table from its own runs.
"""

from __future__ import annotations

import numpy as np

from analysis.xpcs.snr import REFERENCE_BETA_RUNS, Beta, DecayFit, RunMeasurement

__all__ = [
    "BETA_MIN_DECAY_LAGS",
    "BETA_MIN_PRECISION",
    "beta_eligible",
    "crl_state",
    "measure_beta",
    "reference_beta",
]

#: β eligibility: A/σ_A, and 1/(2Γ) in units of the first lag.
BETA_MIN_PRECISION, BETA_MIN_DECAY_LAGS = 10.0, 5.0


def beta_eligible(fit: DecayFit, tau_1: float, tau_max: float) -> bool:
    """A run fixes β when its decay is precise and slow against the first lag."""
    if not fit.converged or not np.isfinite(fit.sigma_amplitude) or fit.rate <= 0:
        return False
    decay = 1.0 / (2.0 * fit.rate)
    return (
        fit.amplitude >= BETA_MIN_PRECISION * fit.sigma_amplitude
        and BETA_MIN_DECAY_LAGS * tau_1 <= decay <= tau_max
    )


def measure_beta(measurements: list[RunMeasurement]) -> dict[int, Beta]:
    """β per ROI index: the median amplitude over the runs eligible at that ROI."""
    out = {}
    n_rois = len(measurements[0].edges_nm) if measurements else 0
    for r in range(n_rois):
        amps, runs = [], []
        for m in measurements:
            if m.n_usable and beta_eligible(m.fits[r], m.tau[0], m.tau[-1]):
                amps.append(m.fits[r].amplitude)
                runs.append(m.run)
        if amps:
            p16, p50, p84 = np.percentile(amps, [16, 50, 84])
            out[r] = Beta(float(p50), len(amps), float(p16), float(p84), tuple(runs))
        else:
            out[r] = Beta(float("nan"), 0, float("nan"), float("nan"), ())
    return out


def reference_beta(
    measurements: list[RunMeasurement], runs: tuple[int, ...] = REFERENCE_BETA_RUNS
) -> dict[int, Beta]:
    """β per ROI index from the reference runs: (median, p16, p84) of converged A.

    No eligibility cut: the reference runs are chosen because their decay is
    complete, and each run counts once whatever its precision.
    """
    out = {}
    chosen = [m for m in measurements if m.run in runs and m.n_usable]
    n_rois = len(measurements[0].edges_nm) if measurements else 0
    for r in range(n_rois):
        amps = [m.fits[r].amplitude for m in chosen if m.fits[r].converged]
        if amps:
            p16, p50, p84 = np.percentile(amps, [16, 50, 84])
            used = tuple(m.run for m in chosen if m.fits[r].converged)
            out[r] = Beta(float(p50), len(amps), float(p16), float(p84), used)
        else:
            out[r] = Beta(float("nan"), 0, float("nan"), float("nan"), ())
    return out


def crl_state(raw) -> str:
    """The CRL lens insertion state of a raw run, ``XTD1:0010100000 XTD6:…``.

    ``~`` marks a lens that moved during the run.
    """
    parts = []
    for name, source in (
        ("XTD1", "SA2_XTD1_CRL/MDL/CRL"),
        ("XTD6", "MID_XTD6_CRL/MDL/CRL"),
    ):
        if source not in raw.control_sources:
            parts.append(f"{name}:absent")
            continue
        lenses = ""
        for i in range(1, 11):
            u = np.unique(raw[source, f"actual.isInsertedLens{i}.value"].ndarray())
            lenses += str(int(u[0])) if u.size == 1 else "~"
        parts.append(f"{name}:{lenses}")
    return " ".join(parts)
