"""The triage's β cross-checks and the CRL state of a raw run."""

from __future__ import annotations

import numpy as np
import pytest
from triage_fakes import hand_run

from analysis.triage.xpcs import (
    beta_eligible,
    crl_state,
    measure_beta,
    reference_beta,
)
from analysis.xpcs.snr import DecayFit


def test_beta_eligibility():
    tau1, tau_max = 1.0, 100.0
    good = DecayFit(0.03, 0.002, 1 / (2 * 10.0), 0.0, 0.0, True)
    assert beta_eligible(good, tau1, tau_max)
    imprecise = DecayFit(0.03, 0.005, 1 / (2 * 10.0), 0.0, 0.0, True)  # A/σ = 6
    fast = DecayFit(0.03, 0.002, 1 / (2 * 2.0), 0.0, 0.0, True)  # 1/(2Γ) = 2 τ₁
    slow = DecayFit(0.03, 0.002, 1 / (2 * 200.0), 0.0, 0.0, True)  # beyond the window
    assert not any(beta_eligible(f, tau1, tau_max) for f in (imprecise, fast, slow))


class _FakeRaw:
    """Just enough of an EXtra-data run for `crl_state`."""

    control_sources = {"SA2_XTD1_CRL/MDL/CRL", "MID_XTD6_CRL/MDL/CRL"}

    def __init__(self, inserted):
        self.inserted = inserted

    def __getitem__(self, key):
        source, k = key
        lens = int(k.split("Lens")[1].split(".")[0])
        values = self.inserted.get((source, lens), [0])

        class _K:
            def ndarray(self_inner):
                return np.asarray(values)

        return _K()


def test_crl_state_reads_lens_insertion_and_marks_movement():
    xtd1, xtd6 = "SA2_XTD1_CRL/MDL/CRL", "MID_XTD6_CRL/MDL/CRL"
    raw = _FakeRaw({(xtd1, 3): [1, 1], (xtd1, 5): [1], (xtd6, 2): [0, 1]})
    assert crl_state(raw) == "XTD1:0010100000 XTD6:0~00000000"


def fit(amplitude, sigma=0.001, decay_lags=10.0, converged=True):
    """A decay fit with 1/(2Γ) of ``decay_lags`` first lags (τ₁ = 1 s)."""
    return DecayFit(amplitude, sigma, 1 / (2 * decay_lags), 0.0, 0.0, converged)


def test_beta_is_the_median_amplitude_over_the_eligible_runs():
    runs = [
        hand_run(1, [fit(0.03), fit(0.01)]),
        hand_run(2, [fit(0.05), fit(0.02)]),
        hand_run(3, [fit(0.04), fit(0.5, sigma=0.1)]),  # ROI 1 imprecise: A/σ = 5
        hand_run(4, [fit(0.9, decay_lags=2.0), fit(0.9, decay_lags=500.0)]),
    ]
    beta = measure_beta(runs)
    assert (beta[0].value, beta[0].n_eligible, beta[0].runs) == (0.04, 3, (1, 2, 3))
    assert (beta[1].value, beta[1].runs) == (pytest.approx(0.015), (1, 2))


def test_reference_beta_is_the_median_over_its_runs_only():
    runs = [
        hand_run(450, [fit(0.03), fit(0.01)]),
        hand_run(493, [fit(0.05), fit(0.02, converged=False)]),
        hand_run(494, [fit(0.04), fit(0.03)]),
        hand_run(999, [fit(0.9), fit(0.9)]),  # not a reference run
    ]
    beta = reference_beta(runs, runs=(450, 493, 494))
    assert (beta[0].value, beta[0].runs) == (0.04, (450, 493, 494))
    assert (beta[1].value, beta[1].runs) == (pytest.approx(0.02), (450, 494))
