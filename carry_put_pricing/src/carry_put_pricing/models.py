"""Validated inputs for the carry-put pricing problem."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class TwoFactorOUParams:
    """Risk-neutral parameters of independent centered slow and fast OU factors."""

    kappa_slow: float
    kappa_fast: float
    theta: float
    eta_slow: float
    eta_fast: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _require_finite(name, value)
        if not 0.0 < self.kappa_slow < self.kappa_fast:
            raise ValueError("Require 0 < kappa_slow < kappa_fast")
        if self.eta_slow < 0.0 or self.eta_fast < 0.0:
            raise ValueError("OU volatilities cannot be negative")


@dataclass(frozen=True)
class FactorState:
    """Current centered slow and fast carry-factor states."""

    slow: float
    fast: float

    def __post_init__(self) -> None:
        _require_finite("slow", self.slow)
        _require_finite("fast", self.fast)


@dataclass(frozen=True)
class GBMParams:
    """Risk-neutral spot parameters.

    Volatility is retained as an explicit input. It cancels from this payoff
    after spot homogeneity and zero spot/carry shock correlation are applied.
    """

    risk_free_rate: float
    volatility: float

    def __post_init__(self) -> None:
        _require_finite("risk_free_rate", self.risk_free_rate)
        _require_finite("volatility", self.volatility)
        if self.volatility < 0.0:
            raise ValueError("volatility cannot be negative")


@dataclass(frozen=True)
class CarryPutContract:
    """Observed inception quote and trading-session maturity."""

    initial_spot: float
    initial_futures: float
    sessions_to_expiry: int
    periods_per_year: int = 244

    def __post_init__(self) -> None:
        _require_finite("initial_spot", self.initial_spot)
        _require_finite("initial_futures", self.initial_futures)
        if self.initial_spot <= 0.0 or self.initial_futures <= 0.0:
            raise ValueError("Spot and futures prices must be positive")
        if isinstance(self.sessions_to_expiry, bool) or self.sessions_to_expiry <= 0:
            raise ValueError("sessions_to_expiry must be a positive integer")
        if int(self.sessions_to_expiry) != self.sessions_to_expiry:
            raise ValueError("sessions_to_expiry must be an integer")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

    @property
    def maturity(self) -> float:
        return self.sessions_to_expiry / float(self.periods_per_year)

    def locked_carry(self, risk_free_rate: float) -> float:
        """Annualized inception carry inferred from the observed futures quote."""
        _require_finite("risk_free_rate", risk_free_rate)
        return risk_free_rate - log(self.initial_futures / self.initial_spot) / self.maturity


@dataclass(frozen=True)
class NumericalConfig:
    """State-grid and Gaussian-quadrature controls."""

    slow_grid_points: int = 301
    fast_grid_points: int = 401
    stationary_stddev_width: float = 6.0
    quadrature_order: int = 43
    minimum_half_width: float = 0.01
    exercise_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.slow_grid_points < 5 or self.fast_grid_points < 5:
            raise ValueError("Each state grid needs at least five points")
        if self.slow_grid_points % 2 == 0 or self.fast_grid_points % 2 == 0:
            raise ValueError("Use odd grid-point counts so zero is a grid node")
        if self.stationary_stddev_width <= 0.0:
            raise ValueError("stationary_stddev_width must be positive")
        if self.quadrature_order < 3:
            raise ValueError("quadrature_order must be at least three")
        if self.minimum_half_width <= 0.0:
            raise ValueError("minimum_half_width must be positive")
        if self.exercise_tolerance < 0.0:
            raise ValueError("exercise_tolerance cannot be negative")
