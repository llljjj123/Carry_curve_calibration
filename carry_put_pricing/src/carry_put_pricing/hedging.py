"""Joint slow/fast factor hedging with two futures maturities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import warnings

import numpy as np

from .analytics import ou_integral_loading
from .models import TwoFactorOUParams


@dataclass(frozen=True)
class HedgeFuturesContract:
    """Observed futures quote and strict trading-session maturity used for hedging."""

    contract: str
    futures_price: float
    sessions_to_expiry: int
    periods_per_year: int = 244

    def __post_init__(self) -> None:
        code = str(self.contract).strip().upper()
        object.__setattr__(self, "contract", code)
        if not code:
            raise ValueError("contract cannot be empty")
        if not isfinite(self.futures_price) or self.futures_price <= 0.0:
            raise ValueError("futures_price must be positive and finite")
        if (
            isinstance(self.sessions_to_expiry, bool)
            or int(self.sessions_to_expiry) != self.sessions_to_expiry
            or self.sessions_to_expiry <= 0
        ):
            raise ValueError("sessions_to_expiry must be a positive integer")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

    @property
    def maturity(self) -> float:
        return self.sessions_to_expiry / float(self.periods_per_year)


@dataclass(frozen=True)
class TwoFuturesHedgeResult:
    """Two-contract delta and hedge positions with matrix diagnostics."""

    contract_1: str
    contract_2: str
    futures_price_1: float
    futures_price_2: float
    sessions_to_expiry_1: int
    sessions_to_expiry_2: int
    slow_loading_1: float
    slow_loading_2: float
    fast_loading_1: float
    fast_loading_2: float
    slow_futures_sensitivity_1: float
    slow_futures_sensitivity_2: float
    fast_futures_sensitivity_1: float
    fast_futures_sensitivity_2: float
    option_slow_factor_sensitivity: float
    option_fast_factor_sensitivity: float
    determinant: float
    condition_number: float
    angular_separation: float
    is_singular: bool
    warning: str | None
    option_delta_1: float | None
    option_delta_2: float | None
    hedge_position_1: float | None
    hedge_position_2: float | None
    residual_slow_exposure: float | None
    residual_fast_exposure: float | None


def calculate_two_futures_hedge(
    *,
    option_slow_factor_sensitivity: float,
    option_fast_factor_sensitivity: float,
    ou_params: TwoFactorOUParams,
    hedge_futures: tuple[HedgeFuturesContract, HedgeFuturesContract],
) -> TwoFuturesHedgeResult:
    """Neutralize the option's two OU-factor sensitivities with two futures.

    ``option_delta_i`` is the option exposure expressed in units of futures
    contract ``i``. The trade needed for a long option is its opposite,
    ``hedge_position_i``. Contract multipliers and integer rounding are outside
    the current scope.
    """
    if len(hedge_futures) != 2:  # pragma: no cover - tuple annotation documents API
        raise ValueError("Exactly two hedge futures are required")
    option_vector = np.array(
        [option_slow_factor_sensitivity, option_fast_factor_sensitivity],
        dtype=float,
    )
    if not np.all(np.isfinite(option_vector)):
        raise ValueError("Option factor sensitivities must be finite")

    first, second = hedge_futures
    slow_loadings = np.array(
        [
            ou_integral_loading(ou_params.kappa_slow, first.maturity),
            ou_integral_loading(ou_params.kappa_slow, second.maturity),
        ]
    )
    fast_loadings = np.array(
        [
            ou_integral_loading(ou_params.kappa_fast, first.maturity),
            ou_integral_loading(ou_params.kappa_fast, second.maturity),
        ]
    )
    prices = np.array([first.futures_price, second.futures_price])
    exposure_matrix = np.vstack((-prices * slow_loadings, -prices * fast_loadings))
    determinant = float(np.linalg.det(exposure_matrix))
    singular_values = np.linalg.svd(exposure_matrix, compute_uv=False)
    largest_singular_value = float(singular_values[0])
    smallest_singular_value = float(singular_values[-1])
    rank_tolerance = (
        np.finfo(float).eps * max(exposure_matrix.shape) * largest_singular_value
    )
    is_singular = smallest_singular_value <= rank_tolerance
    condition_number = (
        float("inf")
        if smallest_singular_value == 0.0
        else largest_singular_value / smallest_singular_value
    )
    column_norms = np.linalg.norm(exposure_matrix, axis=0)
    angular_separation = float(
        abs(determinant) / (float(column_norms[0]) * float(column_norms[1]))
    )

    warning_message: str | None = None
    option_deltas: np.ndarray | None = None
    hedge_positions: np.ndarray | None = None
    residual: np.ndarray | None = None
    if is_singular:
        warning_message = (
            f"The hedge matrix for {first.contract} and {second.contract} is singular; "
            "the joint two-futures delta and hedge positions were not calculated."
        )
        warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
    else:
        option_deltas = np.linalg.solve(exposure_matrix, option_vector)
        hedge_positions = -option_deltas
        residual = option_vector + exposure_matrix @ hedge_positions

    return TwoFuturesHedgeResult(
        contract_1=first.contract,
        contract_2=second.contract,
        futures_price_1=float(first.futures_price),
        futures_price_2=float(second.futures_price),
        sessions_to_expiry_1=int(first.sessions_to_expiry),
        sessions_to_expiry_2=int(second.sessions_to_expiry),
        slow_loading_1=float(slow_loadings[0]),
        slow_loading_2=float(slow_loadings[1]),
        fast_loading_1=float(fast_loadings[0]),
        fast_loading_2=float(fast_loadings[1]),
        slow_futures_sensitivity_1=float(exposure_matrix[0, 0]),
        slow_futures_sensitivity_2=float(exposure_matrix[0, 1]),
        fast_futures_sensitivity_1=float(exposure_matrix[1, 0]),
        fast_futures_sensitivity_2=float(exposure_matrix[1, 1]),
        option_slow_factor_sensitivity=float(option_vector[0]),
        option_fast_factor_sensitivity=float(option_vector[1]),
        determinant=determinant,
        condition_number=float(condition_number),
        angular_separation=angular_separation,
        is_singular=bool(is_singular),
        warning=warning_message,
        option_delta_1=None if option_deltas is None else float(option_deltas[0]),
        option_delta_2=None if option_deltas is None else float(option_deltas[1]),
        hedge_position_1=None if hedge_positions is None else float(hedge_positions[0]),
        hedge_position_2=None if hedge_positions is None else float(hedge_positions[1]),
        residual_slow_exposure=None if residual is None else float(residual[0]),
        residual_fast_exposure=None if residual is None else float(residual[1]),
    )
