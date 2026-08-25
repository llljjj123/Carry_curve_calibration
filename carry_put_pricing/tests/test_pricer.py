from __future__ import annotations

import math

import pytest

from carry_put_pricing import (
    CarryPutContract,
    FactorState,
    GBMParams,
    NumericalConfig,
    TwoFactorOUParams,
    price_american_carry_put,
)


PARAMS = TwoFactorOUParams(
    kappa_slow=1.2409047966134241,
    kappa_fast=44.32953558819525,
    theta=0.08261737612606601,
    eta_slow=0.0799282328223189,
    eta_fast=2.852079153601385,
)
STATE = FactorState(slow=0.05990951629435136, fast=-0.016141256167290972)
CONTRACT = CarryPutContract(initial_spot=7601.804, initial_futures=7527.0, sessions_to_expiry=20)
FAST_CONFIG = NumericalConfig(
    slow_grid_points=41,
    fast_grid_points=61,
    stationary_stddev_width=5.0,
    quadrature_order=5,
)


def test_one_session_contract_has_no_optional_value() -> None:
    contract = CarryPutContract(initial_spot=100.0, initial_futures=99.0, sessions_to_expiry=1)
    result = price_american_carry_put(
        contract,
        PARAMS,
        STATE,
        GBMParams(0.014, 0.25),
        numerical=FAST_CONFIG,
    )
    assert result.price == pytest.approx(0.0, abs=1.0e-14)


def test_spot_volatility_cancels_from_price() -> None:
    low = price_american_carry_put(
        CONTRACT, PARAMS, STATE, GBMParams(0.014, 0.05), numerical=FAST_CONFIG
    )
    high = price_american_carry_put(
        CONTRACT, PARAMS, STATE, GBMParams(0.014, 0.80), numerical=FAST_CONFIG
    )
    assert low.price == high.price
    assert not low.spot_volatility_affects_price


def test_price_scales_with_spot_and_futures_together() -> None:
    base = price_american_carry_put(
        CONTRACT, PARAMS, STATE, GBMParams(0.014, 0.25), numerical=FAST_CONFIG
    )
    scaled_contract = CarryPutContract(
        initial_spot=2.5 * CONTRACT.initial_spot,
        initial_futures=2.5 * CONTRACT.initial_futures,
        sessions_to_expiry=CONTRACT.sessions_to_expiry,
    )
    scaled = price_american_carry_put(
        scaled_contract, PARAMS, STATE, GBMParams(0.014, 0.25), numerical=FAST_CONFIG
    )
    assert scaled.locked_carry == pytest.approx(base.locked_carry)
    assert scaled.price == pytest.approx(2.5 * base.price, rel=1.0e-12)


def test_zero_volatility_flat_carry_has_zero_value() -> None:
    params = TwoFactorOUParams(
        kappa_slow=1.0,
        kappa_fast=20.0,
        theta=0.08,
        eta_slow=0.0,
        eta_fast=0.0,
    )
    spot, r, sessions = 100.0, 0.014, 10
    maturity = sessions / 244.0
    futures = spot * math.exp((r - params.theta) * maturity)
    contract = CarryPutContract(spot, futures, sessions)
    result = price_american_carry_put(
        contract,
        params,
        FactorState(0.0, 0.0),
        GBMParams(r, 0.25),
        numerical=FAST_CONFIG,
    )
    assert result.price == pytest.approx(0.0, abs=1.0e-12)


def test_agreed_example_is_positive_and_model_basis_is_reported() -> None:
    result = price_american_carry_put(
        CONTRACT, PARAMS, STATE, GBMParams(0.014, 0.25), numerical=FAST_CONFIG
    )
    assert result.price > 0.0
    assert result.inception_exercise_value == 0.0
    assert result.initial_futures_model_error == pytest.approx(0.396153560659, abs=1.0e-9)
    assert len(result.exercise_summary) == CONTRACT.sessions_to_expiry - 1
