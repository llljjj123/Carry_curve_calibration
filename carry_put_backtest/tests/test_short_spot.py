"""Focused tests for the short carry-put and funded spot study."""

from dataclasses import replace

import numpy as np
import pytest

from carry_put_backtest.engine import (
    FundedCashLedger,
    SHORT_SCENARIOS,
    _short_signal_targets_from_positions,
)
from carry_put_backtest.tests.test_backtest import CONFIG, PARAMS, run
from carry_put_backtest.analyze_historical import replay


SHORT_CONFIG = replace(CONFIG, study="short_with_spot")


def test_short_study_has_three_primary_scenarios_and_two_controls():
    result = run(config=SHORT_CONFIG)
    assert len(result.summaries) == 5
    assert list(result.summaries.scenario) == [spec.scenario for spec in SHORT_SCENARIOS]
    assert set(result.summaries.loc[result.summaries.scenario.str.endswith("_spot"), "scenario_role"]) == {"primary"}
    assert set(result.summaries.loc[result.summaries.scenario.str.endswith("_only"), "scenario_role"]) == {"control"}
    assert (result.daily.option_side == -1).all()
    assert (result.daily.dividend_cash_flow == 0).all()


def test_short_futures_controls_reverse_legacy_long_positions():
    long = run()
    short = run(config=SHORT_CONFIG)
    for long_name, short_name in (
        ("one_futures", "short_one_futures_only"),
        ("two_futures", "short_two_futures_only"),
    ):
        a = long.daily.query("scenario == @long_name").sort_values("elapsed_sessions")
        b = short.daily.query("scenario == @short_name").sort_values("elapsed_sessions")
        np.testing.assert_allclose(b.contracts_1, -a.contracts_1, atol=1e-12)
        np.testing.assert_allclose(b.contracts_2, -a.contracts_2, atol=1e-12)
        assert short.summaries.loc[short.summaries.scenario == short_name, "total_pnl"].iloc[0] == pytest.approx(
            -long.summaries.loc[long.summaries.scenario == long_name, "total_pnl"].iloc[0]
        )


def test_short_spot_target_uses_actual_futures_and_concrete_funding_case():
    config = replace(SHORT_CONFIG, option_units=1, option_multiplier=1, futures_multiplier=1)
    spec = next(s for s in SHORT_SCENARIOS if s.scenario == "short_one_futures_spot")
    value = type("Value", (), {"price": 30.0})()
    target = _short_signal_targets_from_positions(
        value, np.array([5950.0, 6000.0]), 6000.0, np.array([-.30, 0.0]), spec, config,
    )
    assert target == pytest.approx(.3025)
    ledger = FundedCashLedger(30.0)
    ledger.trade(np.array([-.30, 0.0]), target, 6000.0, config)
    assert ledger.cash == pytest.approx(-1785.0)
    assert ledger.cash + target * 6000.0 - 30.0 == pytest.approx(0.0)


def test_funded_spot_scale_move_is_not_credited_twice():
    config = replace(SHORT_CONFIG, option_units=1, option_multiplier=1, futures_multiplier=1,
                     risk_free_rate=0.0)
    ledger = FundedCashLedger(30.0)
    ledger.trade(np.array([-.30, 0.0]), .3025, 6000.0, config)
    ledger.accrue(np.array([5950.0, 6000.0]), np.array([6009.5, 6060.0]),
                  6000.0, 6060.0, 1 / 244, config)
    # Spot +18.15, futures -17.85, short option mark change -0.30.
    equity = ledger.cash + ledger.spot_units * 6060.0 - 30.3
    assert ledger.cumulative_spot_pnl == pytest.approx(18.15)
    assert ledger.cumulative_futures_pnl == pytest.approx(-17.85)
    assert equity == pytest.approx(0.0)


def test_short_spot_local_residuals_and_controls():
    result = run(config=SHORT_CONFIG)
    active = result.daily.loc[~(result.daily.exercise_now | result.daily.expired)]
    two = active.query("scenario == 'short_two_futures_spot'")
    one = active.query("scenario == 'short_one_futures_spot'")
    one_control = active.query("scenario == 'short_one_futures_only'")
    no = active.query("scenario == 'short_no_hedge'")
    assert np.max(np.abs(two.residual_slow_exposure)) < 1e-10
    assert np.max(np.abs(two.residual_fast_exposure)) < 1e-10
    assert np.max(np.abs(two.spot_delta_residual)) < 1e-10
    assert np.max(np.abs(one.spot_delta_residual)) < 1e-10
    assert (one_control.spot_delta_residual.abs() > 0).all()
    assert (one_control.spot_units == 0).all()
    assert (no.contracts_1 == 0).all() and (no.contracts_2 == 0).all() and (no.spot_units == 0).all()


def test_short_summary_independent_lifetime_identity():
    result = run(config=SHORT_CONFIG)
    for row in result.summaries.itertuples():
        assert row.total_pnl == pytest.approx(
            row.option_pnl + row.futures_pnl + row.spot_pnl + row.financing
            - row.transaction_costs
        )
        assert row.option_side == -1


def test_funded_replay_reproduces_short_engine_with_lag_and_costs():
    config = replace(SHORT_CONFIG, execution_lag_sessions=1,
                     slippage_points=.2, spot_cost_bps=1.0)
    result = run(config=config)
    replayed, _ = replay(result.daily, result.summaries)
    merged = result.daily.merge(
        replayed, on=["cohort", "scenario", "date", "elapsed_sessions"],
        suffixes=("_engine", "_replay"),
    )
    np.testing.assert_allclose(merged.equity_engine, merged.equity_replay, atol=1e-10)
    np.testing.assert_allclose(merged.cash_engine, merged.cash_replay, atol=1e-10)
    np.testing.assert_allclose(merged.spot_units_engine, merged.spot_units_replay, atol=1e-10)
