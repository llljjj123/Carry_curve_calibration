from __future__ import annotations

import numpy as np
import pytest

from carry_put_pricing import (
    FactorState,
    TwoFactorOUParams,
    exact_forward_price,
    exact_implied_carry,
    integrated_carry_moments,
    ou_integral_loading,
    ou_integral_variance,
)


PARAMS = TwoFactorOUParams(
    kappa_slow=1.2409047966134241,
    kappa_fast=44.32953558819525,
    theta=0.08261737612606601,
    eta_slow=0.0799282328223189,
    eta_fast=2.852079153601385,
)
STATE = FactorState(slow=0.05990951629435136, fast=-0.016141256167290972)


def test_integral_loading_has_correct_zero_and_small_time_limits() -> None:
    assert ou_integral_loading(2.0, 0.0) == 0.0
    tau = 1.0e-10
    assert ou_integral_loading(2.0, tau) == pytest.approx(tau, rel=1.0e-10)


def test_integral_variance_is_stable_at_small_time() -> None:
    kappa, eta, tau = 3.0, 0.7, 1.0e-7
    expected_leading_term = eta**2 * tau**3 / 3.0
    assert ou_integral_variance(kappa, eta, tau) == pytest.approx(expected_leading_term, rel=1.0e-6)


def test_integrated_carry_moments_match_seeded_simulation() -> None:
    tau = 20.0 / 244.0
    moments = integrated_carry_moments(PARAMS, STATE.slow, STATE.fast, tau)
    rng = np.random.default_rng(852)
    paths = 120_000
    steps = 160
    dt = tau / steps
    slow = np.full(paths, STATE.slow)
    fast = np.full(paths, STATE.fast)
    integral = np.zeros(paths)
    for _ in range(steps):
        integral += (PARAMS.theta + slow + fast) * dt
        for values, kappa, eta in (
            (slow, PARAMS.kappa_slow, PARAMS.eta_slow),
            (fast, PARAMS.kappa_fast, PARAMS.eta_fast),
        ):
            decay = np.exp(-kappa * dt)
            variance = eta**2 * (1.0 - decay**2) / (2.0 * kappa)
            values *= decay
            values += np.sqrt(variance) * rng.standard_normal(paths)
    # The mean tolerance includes Monte Carlo sampling error and the left-point
    # discretization error in the independently simulated integral.
    assert integral.mean() == pytest.approx(float(moments.mean), abs=1.0e-4)
    assert integral.var() == pytest.approx(moments.variance, rel=0.035)


def test_agreed_example_exact_forward_check() -> None:
    tau = 20.0 / 244.0
    model_carry = exact_implied_carry(PARAMS, STATE, tau)
    model_futures = exact_forward_price(7601.804, PARAMS, STATE, tau, 0.014)
    assert model_carry == pytest.approx(0.1340041028791029, rel=1.0e-12)
    assert model_futures == pytest.approx(7527.396153560659, rel=1.0e-12)
