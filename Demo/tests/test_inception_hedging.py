"""Focused checks for inception futures-plus-spot hedge presentation."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

import inception_hedging as module  # noqa: E402
from inception_hedging import (  # noqa: E402
    calculate_one_futures_spot_hedge,
    calculate_two_futures_spot_hedge,
    scale_neutral_spot_position,
)
from carry_put_pricing import (  # noqa: E402
    HedgeFuturesContract,
    TwoFactorOUParams,
    calculate_one_futures_hedge,
    calculate_two_futures_hedge,
    factor_innovation_covariance,
)
from carry_put_pricing.analytics import ou_integral_loading  # noqa: E402


PARAMS = TwoFactorOUParams(
    kappa_slow=0.4, kappa_fast=12.0, theta=0.08, eta_slow=0.05, eta_fast=1.1,
)
FIRST = HedgeFuturesContract("IM2609", 5950.0, 30)


def test_concrete_long_option_sign_units_and_scale_identity():
    spot, residual = scale_neutral_spot_position(30.0, 6000.0, (5950.0,), (0.30,))
    assert spot == pytest.approx(-0.3025)
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_two_futures_mixed_sign_positions_use_both_terms():
    joint = SimpleNamespace(
        contract_1="IM2609", contract_2="IM2703",
        futures_price_1=5950.0, futures_price_2=5700.0,
        hedge_position_1=0.4, hedge_position_2=-0.15,
        is_singular=False, warning=None,
    )
    result = calculate_two_futures_spot_hedge(
        option_price=30.0, initial_spot=6000.0, joint_hedge=joint,
    )
    expected = -(30.0 + 0.4 * 5950.0 - 0.15 * 5700.0) / 6000.0
    assert result.spot_position == pytest.approx(expected)
    assert result.scale_residual == pytest.approx(0.0, abs=1e-12)


def test_one_futures_helper_minimizes_same_q_carry_variance():
    b = np.array([320.0, 85.0])
    n = calculate_one_futures_hedge(
        option_slow_factor_sensitivity=b[0],
        option_fast_factor_sensitivity=b[1],
        ou_params=PARAMS,
        hedge_future=FIRST,
    )
    g = -FIRST.futures_price * np.array([
        ou_integral_loading(PARAMS.kappa_slow, FIRST.maturity),
        ou_integral_loading(PARAMS.kappa_fast, FIRST.maturity),
    ])
    q = factor_innovation_covariance(PARAMS, 1 / FIRST.periods_per_year)
    variance = lambda position: float((b + position * g) @ q @ (b + position * g))
    directional = (-b[0] / g[0], -b[1] / g[1])
    assert directional[0] != pytest.approx(directional[1])
    for alternative in (*directional, n - 1e-3, n + 1e-3):
        assert variance(n) <= variance(alternative) + 1e-12


def test_demo_wrapper_calls_long_helper_once_without_negation(monkeypatch):
    calls = []

    def fake_helper(**kwargs):
        calls.append(kwargs)
        return 0.30

    monkeypatch.setattr(module, "calculate_one_futures_hedge", fake_helper)
    result = calculate_one_futures_spot_hedge(
        option_price=30.0, initial_spot=6000.0,
        option_slow_factor_sensitivity=320.0,
        option_fast_factor_sensitivity=85.0,
        ou_params=PARAMS, hedge_future=FIRST,
    )
    assert len(calls) == 1
    assert result.futures_positions == (0.30,)
    assert result.spot_position == pytest.approx(-0.3025)


def test_singular_joint_is_unavailable_without_blocking_one_futures():
    duplicate = HedgeFuturesContract("IM2610", 6000.0, 30)
    with pytest.warns(RuntimeWarning, match="singular"):
        joint = calculate_two_futures_hedge(
            option_slow_factor_sensitivity=320.0,
            option_fast_factor_sensitivity=85.0,
            ou_params=PARAMS,
            hedge_futures=(FIRST, duplicate),
        )
    two = calculate_two_futures_spot_hedge(
        option_price=30.0, initial_spot=6000.0, joint_hedge=joint,
    )
    one = calculate_one_futures_spot_hedge(
        option_price=30.0, initial_spot=6000.0,
        option_slow_factor_sensitivity=320.0,
        option_fast_factor_sensitivity=85.0,
        ou_params=PARAMS, hedge_future=FIRST,
    )
    assert not two.is_available and two.spot_position is None and "singular" in two.reason
    assert one.is_available and np.isfinite(one.spot_position)


def test_nonfinite_inputs_are_not_returned_as_numeric_and_zero_risk_is_preserved():
    with pytest.raises(ValueError, match="finite"):
        scale_neutral_spot_position(30.0, 6000.0, (5950.0,), (float("nan"),))
    bad_joint = SimpleNamespace(
        contract_1="IM2609", contract_2="IM2703",
        futures_price_1=5950.0, futures_price_2=5700.0,
        hedge_position_1=0.4, hedge_position_2=float("inf"),
        is_singular=False, warning=None,
    )
    result = calculate_two_futures_spot_hedge(
        option_price=30.0, initial_spot=6000.0, joint_hedge=bad_joint,
    )
    assert not result.is_available and result.spot_position is None
    zero_risk = TwoFactorOUParams(
        kappa_slow=0.4, kappa_fast=12.0, theta=0.08, eta_slow=0.0, eta_fast=0.0,
    )
    assert calculate_one_futures_hedge(
        option_slow_factor_sensitivity=0.0,
        option_fast_factor_sensitivity=0.0,
        ou_params=zero_risk,
        hedge_future=FIRST,
    ) == 0.0


def test_notebook_uses_live_demo_and_presents_one_before_two_futures():
    notebook = json.loads((DEMO_ROOT / "Carry_Put_Demo.ipynb").read_text(encoding="utf-8"))
    cells = {cell.get("id"): cell for cell in notebook["cells"]}
    order = [cell.get("id") for cell in notebook["cells"]]
    assert order.index("curve-delta-table") < order.index("one-futures-spot-table") < order.index("two-futures-hedge-table")
    one_source = "".join(cells["one-futures-spot-table"]["source"])
    two_source = "".join(cells["two-futures-hedge-table"]["source"])
    assert "demo.pricing.hedge_futures[0]" in one_source
    assert "pathwise_option_factor_sensitivity" in one_source
    assert "calculate_one_futures_spot_hedge" in one_source
    assert "calculate_two_futures_spot_hedge" in two_source
    assert "joint_hedge=joint" in two_source
    assert "option_delta_1" not in two_source.split("calculate_two_futures_spot_hedge", 1)[1]
