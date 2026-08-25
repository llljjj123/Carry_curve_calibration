"""Exact Gaussian integrated-carry and forward-price formulas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import FactorState, TwoFactorOUParams


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IntegratedCarryMoments:
    """Conditional moments of the carry integral over a future interval."""

    mean: FloatArray | np.float64
    variance: float


def ou_integral_loading(kappa: float, tau: float) -> float:
    """Return A(kappa,tau) = (1-exp(-kappa*tau))/kappa stably."""
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if tau < 0.0:
        raise ValueError("tau cannot be negative")
    return float(-np.expm1(-kappa * tau) / kappa)


def _integral_variance_kernel(x: float) -> float:
    """Stable x - 2(1-e^-x) + (1-e^-2x)/2."""
    if abs(x) < 1.0e-3:
        return x**3 / 3.0 - x**4 / 4.0 + 7.0 * x**5 / 60.0 - x**6 / 24.0
    return float(x + 2.0 * np.expm1(-x) - 0.5 * np.expm1(-2.0 * x))


def ou_integral_variance(kappa: float, eta: float, tau: float) -> float:
    """Conditional variance of integral_0^tau x_u du for a centered OU factor."""
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if eta < 0.0:
        raise ValueError("eta cannot be negative")
    if tau < 0.0:
        raise ValueError("tau cannot be negative")
    x = kappa * tau
    value = eta**2 * _integral_variance_kernel(x) / kappa**3
    return max(float(value), 0.0)


def integrated_carry_moments(
    params: TwoFactorOUParams,
    slow_state: ArrayLike,
    fast_state: ArrayLike,
    tau: float,
) -> IntegratedCarryMoments:
    """Moments of I=integral_t^(t+tau) c_u du conditional on current factors."""
    if tau < 0.0:
        raise ValueError("tau cannot be negative")
    slow = np.asarray(slow_state, dtype=float)
    fast = np.asarray(fast_state, dtype=float)
    mean = (
        params.theta * tau
        + ou_integral_loading(params.kappa_slow, tau) * slow
        + ou_integral_loading(params.kappa_fast, tau) * fast
    )
    variance = ou_integral_variance(params.kappa_slow, params.eta_slow, tau)
    variance += ou_integral_variance(params.kappa_fast, params.eta_fast, tau)
    return IntegratedCarryMoments(mean=mean, variance=variance)


def exact_forward_ratio(
    params: TwoFactorOUParams,
    slow_state: ArrayLike,
    fast_state: ArrayLike,
    tau: float,
    risk_free_rate: float,
) -> FloatArray | np.float64:
    """Return F(t,T)/S_t under independent spot and OU shocks."""
    moments = integrated_carry_moments(params, slow_state, fast_state, tau)
    return np.exp(risk_free_rate * tau - moments.mean + 0.5 * moments.variance)


def exact_forward_price(
    spot: float,
    params: TwoFactorOUParams,
    state: FactorState,
    tau: float,
    risk_free_rate: float,
) -> float:
    """Return the exact stochastic-carry futures/forward price."""
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    ratio = exact_forward_ratio(params, state.slow, state.fast, tau, risk_free_rate)
    return float(spot * ratio)


def exact_implied_carry(
    params: TwoFactorOUParams,
    state: FactorState,
    tau: float,
) -> float:
    """Annualized q satisfying F/S=exp((r-q)tau), including convexity."""
    if tau < 0.0:
        raise ValueError("tau cannot be negative")
    if tau == 0.0:
        return params.theta + state.slow + state.fast
    moments = integrated_carry_moments(params, state.slow, state.fast, tau)
    return float((moments.mean - 0.5 * moments.variance) / tau)
