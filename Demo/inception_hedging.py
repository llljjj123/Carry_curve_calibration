"""Inception spot-scale hedges for the long carry-put Demo."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import sys


DEMO_ROOT = Path(__file__).resolve().parent
PRICING_SRC = DEMO_ROOT.parent / "carry_put_pricing" / "src"
if str(PRICING_SRC) not in sys.path:
    sys.path.insert(0, str(PRICING_SRC))

from carry_put_pricing import (  # noqa: E402
    HedgeFuturesContract,
    TwoFactorOUParams,
    TwoFuturesHedgeResult,
    calculate_one_futures_hedge,
)


@dataclass(frozen=True)
class InceptionFuturesSpotHedge:
    """Continuous hedge trades per one long option in unit point values."""

    contracts: tuple[str, ...]
    futures_positions: tuple[float, ...] | None
    spot_position: float | None
    scale_residual: float | None
    is_available: bool
    reason: str | None = None


def scale_neutral_spot_position(
    option_price: float,
    initial_spot: float,
    futures_prices: tuple[float, ...],
    futures_positions: tuple[float, ...],
) -> tuple[float, float]:
    """Return H and V + sum(n_i F_i) + H S for a long option."""
    values = (option_price, initial_spot, *futures_prices, *futures_positions)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("Option, spot, futures prices, and positions must be finite")
    if option_price < 0:
        raise ValueError("Long-option price cannot be negative")
    if initial_spot <= 0 or any(price <= 0 for price in futures_prices):
        raise ValueError("Spot and futures prices must be positive")
    if len(futures_prices) != len(futures_positions):
        raise ValueError("Each futures price must have one position")
    spot = -(option_price + sum(n * price for n, price in zip(futures_positions, futures_prices))) / initial_spot
    residual = option_price + sum(n * price for n, price in zip(futures_positions, futures_prices)) + spot * initial_spot
    return float(spot), float(residual)


def calculate_one_futures_spot_hedge(
    *,
    option_price: float,
    initial_spot: float,
    option_slow_factor_sensitivity: float,
    option_fast_factor_sensitivity: float,
    ou_params: TwoFactorOUParams,
    hedge_future: HedgeFuturesContract,
) -> InceptionFuturesSpotHedge:
    """Minimize carry-factor risk with one futures, then neutralize scale with spot."""
    futures_position = calculate_one_futures_hedge(
        option_slow_factor_sensitivity=option_slow_factor_sensitivity,
        option_fast_factor_sensitivity=option_fast_factor_sensitivity,
        ou_params=ou_params,
        hedge_future=hedge_future,
    )
    spot_position, residual = scale_neutral_spot_position(
        option_price,
        initial_spot,
        (hedge_future.futures_price,),
        (futures_position,),
    )
    return InceptionFuturesSpotHedge(
        contracts=(hedge_future.contract,),
        futures_positions=(futures_position,),
        spot_position=spot_position,
        scale_residual=residual,
        is_available=True,
    )


def calculate_two_futures_spot_hedge(
    *,
    option_price: float,
    initial_spot: float,
    joint_hedge: TwoFuturesHedgeResult | None,
) -> InceptionFuturesSpotHedge:
    """Add spot to an available joint long-option hedge without a fallback."""
    if joint_hedge is None:
        return InceptionFuturesSpotHedge((), None, None, None, False, "Two-futures hedge result is unavailable.")
    contracts = (joint_hedge.contract_1, joint_hedge.contract_2)
    if joint_hedge.is_singular or joint_hedge.hedge_position_1 is None or joint_hedge.hedge_position_2 is None:
        return InceptionFuturesSpotHedge(
            contracts, None, None, None, False,
            joint_hedge.warning or "Joint hedge positions are unavailable.",
        )
    positions = (joint_hedge.hedge_position_1, joint_hedge.hedge_position_2)
    try:
        spot_position, residual = scale_neutral_spot_position(
            option_price,
            initial_spot,
            (joint_hedge.futures_price_1, joint_hedge.futures_price_2),
            positions,
        )
    except ValueError as exc:
        return InceptionFuturesSpotHedge(contracts, None, None, None, False, str(exc))
    return InceptionFuturesSpotHedge(
        contracts, positions, spot_position, residual, True,
    )
