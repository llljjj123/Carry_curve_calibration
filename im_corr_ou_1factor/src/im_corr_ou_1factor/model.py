"""Stable analytical formulas for the correlated one-factor OU model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OUParams:
    """Model parameters; ``sigma`` is fixed separately by configuration."""

    kappa: float
    theta: float
    eta: float
    rho: float
    sigma_epsilon: float
    mu: float = 0.0

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("OU parameters must be finite")
        if self.kappa <= 0 or self.eta <= 0 or self.sigma_epsilon <= 0:
            raise ValueError("kappa, eta, and sigma_epsilon must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly between -1 and 1")


def _array(value: np.ndarray | Iterable[float] | float) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _phi_b(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x)
    small = np.abs(x) < 1e-5
    z = x[small]
    result[small] = 1 - z / 2 + z**2 / 6 - z**3 / 24 + z**4 / 120 - z**5 / 720
    result[~small] = -np.expm1(-x[~small]) / x[~small]
    return result


def integral_b(kappa: float, tau: np.ndarray | Iterable[float] | float) -> np.ndarray:
    """Return ``B(tau) = integral_0^tau exp(-kappa*u) du`` stably."""
    time = _array(tau)
    if kappa <= 0 or np.any(time < 0):
        raise ValueError("kappa must be positive and tau nonnegative")
    return time * _phi_b(kappa * time)


def maturity_loading(kappa: float, tau: np.ndarray | Iterable[float] | float) -> np.ndarray:
    """Return the normalized loading ``B(tau)/tau``, with limit one."""
    time = _array(tau)
    if kappa <= 0 or np.any(time < 0):
        raise ValueError("kappa must be positive and tau nonnegative")
    return _phi_b(kappa * time)


def integral_d(kappa: float, tau: np.ndarray | Iterable[float] | float) -> np.ndarray:
    """Return ``D(tau) = integral_0^tau B(u) du`` stably."""
    time = _array(tau)
    if kappa <= 0 or np.any(time < 0):
        raise ValueError("kappa must be positive and tau nonnegative")
    x = kappa * time
    phi = np.empty_like(x)
    small = np.abs(x) < 1e-4
    z = x[small]
    phi[small] = 0.5 - z / 6 + z**2 / 24 - z**3 / 120 + z**4 / 720 - z**5 / 5040
    phi[~small] = (x[~small] + np.expm1(-x[~small])) / x[~small] ** 2
    return time**2 * phi


def integral_c(kappa: float, tau: np.ndarray | Iterable[float] | float) -> np.ndarray:
    """Return ``C(tau) = integral_0^tau B(u)^2 du`` stably."""
    time = _array(tau)
    if kappa <= 0 or np.any(time < 0):
        raise ValueError("kappa must be positive and tau nonnegative")
    x = kappa * time
    phi = np.empty_like(x)
    small = np.abs(x) < 2e-3
    z = x[small]
    phi[small] = 1 / 3 - z / 4 + 7 * z**2 / 60 - z**3 / 24 + 31 * z**4 / 2520
    xs = x[~small]
    numerator = xs + 2 * np.expm1(-xs) - 0.5 * np.expm1(-2 * xs)
    phi[~small] = numerator / xs**3
    return time**3 * phi


def integral_j(kappa: float, tau: np.ndarray | Iterable[float] | float) -> np.ndarray:
    """Return the transition/return covariance integral ``J = B^2/2``."""
    b = integral_b(kappa, tau)
    return 0.5 * b**2


def transition_moments(params: OUParams, delta: float) -> tuple[float, float]:
    if delta < 0:
        raise ValueError("Observation dates must be nondecreasing")
    a = float(np.exp(-params.kappa * delta))
    q = float(params.eta**2 * (-np.expm1(-2 * params.kappa * delta)) / (2 * params.kappa))
    return a, max(q, 0.0)


def curve_coefficients(
    params: OUParams,
    tau: np.ndarray | Iterable[float] | float,
    sigma: float,
    *,
    variant: str = "exact",
) -> tuple[np.ndarray, np.ndarray]:
    """Return intercept and state loading for the carry observation equation."""
    params.validate()
    time = _array(tau)
    h = maturity_loading(params.kappa, time)
    intercept = params.theta * (1.0 - h)
    if variant == "exact":
        positive = time > 0
        correction = np.zeros_like(time)
        correction[positive] = (
            -0.5 * params.eta**2 * integral_c(params.kappa, time[positive]) / time[positive]
            + params.rho * sigma * params.eta * integral_d(params.kappa, time[positive]) / time[positive]
        )
        intercept += correction
    elif variant != "legacy":
        raise ValueError("variant must be 'exact' or 'legacy'")
    return intercept, h


def fitted_carry(
    state: np.ndarray | float,
    tau: np.ndarray | Iterable[float] | float,
    params: OUParams,
    sigma: float,
    *,
    variant: str = "exact",
) -> np.ndarray:
    intercept, loading = curve_coefficients(params, tau, sigma, variant=variant)
    return intercept + loading * np.asarray(state, dtype=float)


def log_futures_basis(
    state: np.ndarray | float,
    tau: np.ndarray | Iterable[float] | float,
    params: OUParams,
    sigma: float,
    rate: np.ndarray | float,
    *,
    variant: str = "exact",
) -> np.ndarray:
    """Return analytical ``log(F/S)`` for either exact or legacy pricing."""
    time = _array(tau)
    if variant == "legacy":
        return (np.asarray(rate, dtype=float) - fitted_carry(state, time, params, sigma, variant=variant)) * time
    if variant != "exact":
        raise ValueError("variant must be 'exact' or 'legacy'")
    return (
        (np.asarray(rate, dtype=float) - params.theta) * time
        - (np.asarray(state, dtype=float) - params.theta) * integral_b(params.kappa, time)
        + 0.5 * params.eta**2 * integral_c(params.kappa, time)
        - params.rho * sigma * params.eta * integral_d(params.kappa, time)
    )


def joint_interval_moments(
    params: OUParams,
    sigma: float,
    delta: float,
) -> tuple[float, float, float, float, float]:
    """Return ``(a, Q, B, V_R, G)`` for an exact joint interval."""
    if sigma <= 0 or delta <= 0:
        raise ValueError("sigma and delta must be positive")
    a, q = transition_moments(params, delta)
    b = float(integral_b(params.kappa, delta))
    c = float(integral_c(params.kappa, delta))
    d = float(integral_d(params.kappa, delta))
    j = float(integral_j(params.kappa, delta))
    variance_return = sigma**2 * delta + params.eta**2 * c - 2 * params.rho * sigma * params.eta * d
    covariance = params.rho * params.eta * sigma * b - params.eta**2 * j
    covariance_matrix = np.array([[q, covariance], [covariance, variance_return]])
    if variance_return <= 0 or np.linalg.eigvalsh(covariance_matrix).min() < -1e-10:
        raise ValueError("Joint state/return covariance is not positive semidefinite")
    return a, q, b, variance_return, covariance

