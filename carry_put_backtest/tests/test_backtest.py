"""Focused synthetic tests only: no cache calibration or historical batch run."""

from dataclasses import replace
import json
from math import exp
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from carry_put_backtest import BacktestConfig, MarketData, load_cached_market, monthly_cohorts, run_backtest
from carry_put_backtest.engine import CashLedger, advance_filter, calibration_window
from carry_put_backtest.market import CohortUnavailable, session_dates
from carry_put_backtest.reporting import aggregate_results, export_result
from calendar_utils import contract_expiry, trading_days_between
from carry_put_pricing import (
    CarryPutContract, ExistingCarryPutResult, FactorState, GBMParams,
    HedgeFuturesContract, NumericalConfig, TwoFactorOUParams,
    calculate_one_futures_hedge, calculate_two_futures_hedge,
    factor_innovation_covariance, ou_integral_loading,
    price_american_carry_put, price_existing_carry_put,
)
from im_2factor_ou_carry.two_factor import TwoFactorParams, two_factor_kalman_filter, transition


PARAMS = TwoFactorOUParams(0.8, 15.0, 0.08, 0.05, 0.5)
FILTER_PARAMS = TwoFactorParams(**PARAMS.__dict__, sigma_epsilon=0.001)
NUMERICAL = NumericalConfig(slow_grid_points=31, fast_grid_points=41, quadrature_order=5)
CONFIG = BacktestConfig(window_dates=10, numerical=NUMERICAL)
ENTRY = pd.Timestamp("2025-01-20")
EXPIRY = pd.Timestamp("2025-02-21")


def estimate_stub(sample, **kwargs):
    return SimpleNamespace(params=FILTER_PARAMS, converged=True, message="synthetic fixture",
                           log_likelihood=0.0, optimizer_runs=pd.DataFrame([{"converged": True}]))


def make_market():
    dates = session_dates("2024-12-20", EXPIRY)
    spot = pd.DataFrame({"date": dates, "spot": 6000. + np.arange(len(dates)) * 3.})
    rows = []
    for i, row in spot.iterrows():
        for code in ("IM2502", "IM2503", "IM2506"):
            tau = trading_days_between(row.date, contract_expiry(code)) / 244
            x = 0.012 * np.sin(i / 4)
            price = row.spot * exp((0.014 - PARAMS.theta) * tau - ou_integral_loading(PARAMS.kappa_slow, tau) * x)
            rows.append({"date": row.date, "contract": code, "futures_price": price})
    return MarketData(spot, pd.DataFrame(rows))


def value_stub(contract, params, state, gbm, *, elapsed_sessions, current_spot, current_futures, numerical):
    remaining = contract.sessions_to_expiry - elapsed_sessions
    expired = remaining == 0
    return ExistingCarryPutResult(
        0. if expired else 20. + elapsed_sessions, 0. if expired else 20. + elapsed_sessions,
        0., False, expired, contract.locked_carry(gbm.risk_free_rate), remaining,
        0. if expired else 180., 0. if expired else 30., current_futures, 0.,
    )


def run(market=None, config=CONFIG, valuer=value_stub, estimator=estimate_stub):
    return run_backtest(market or make_market(), start=ENTRY, end=ENTRY, config=config,
                        estimator=estimator, valuer=valuer)


def test_exact_monthly_schedule_and_no_midmonth_entry():
    schedule = monthly_cohorts("2024-12-01", "2025-03-31")
    assert list(schedule.entry_date) == list(pd.to_datetime([
        "2024-12-23", "2025-01-20", "2025-02-24", "2025-03-24",
    ]))
    assert list(schedule.option_contract) == ["IM2501", "IM2502", "IM2503", "IM2504"]
    assert monthly_cohorts("2025-01-21", "2025-02-20").empty


def test_calibration_window_uses_curve_dates_and_no_future():
    panel, _ = make_market().observations_through(EXPIRY)
    sample = calibration_window(panel, ENTRY, 10)
    assert sample.date.nunique() == 10
    assert sample.date.max() == ENTRY
    assert len(sample) == 30  # three maturities per curve date
    with pytest.raises(CohortUnavailable, match="Need"):
        calibration_window(panel, ENTRY, 488)


def test_one_futures_is_variance_minimizer_and_two_neutralizes_factors():
    hedge = HedgeFuturesContract("IM2502", 6000, 20)
    second = HedgeFuturesContract("IM2506", 5900, 100)
    b = np.array([180., 30.])
    q = factor_innovation_covariance(PARAMS)
    g = -hedge.futures_price * np.array([
        ou_integral_loading(PARAMS.kappa_slow, hedge.maturity),
        ou_integral_loading(PARAMS.kappa_fast, hedge.maturity),
    ])
    kwargs = dict(option_slow_factor_sensitivity=b[0], option_fast_factor_sensitivity=b[1], ou_params=PARAMS)
    n = calculate_one_futures_hedge(**kwargs, hedge_future=hedge)
    residual = b + n * g
    assert abs(g @ q @ residual) < 1e-11
    assert residual @ q @ residual < b @ q @ b
    for trial in (n - 0.1, n + 0.1):
        assert (b + trial * g) @ q @ (b + trial * g) > residual @ q @ residual
    joint = calculate_two_futures_hedge(**kwargs, hedge_futures=(hedge, second))
    assert joint.residual_slow_exposure == pytest.approx(0, abs=1e-10)
    assert joint.residual_fast_exposure == pytest.approx(0, abs=1e-10)
    assert calculate_one_futures_hedge(**{**kwargs, "ou_params": replace(PARAMS, eta_slow=0, eta_fast=0)}, hedge_future=hedge) == 0


def test_existing_price_preserves_inception_and_locked_carry():
    contract = CarryPutContract(6000, 5950, 8)
    state = FactorState(.01, -.02)
    gbm = GBMParams(.014, .25)
    new = price_american_carry_put(contract, PARAMS, state, gbm, numerical=NUMERICAL)
    existing = price_existing_carry_put(contract, PARAMS, state, gbm,
                                      elapsed_sessions=0, current_spot=6000, current_futures=5950,
                                      numerical=NUMERICAL)
    assert existing.price == pytest.approx(new.price)
    assert existing.slow_factor_sensitivity == pytest.approx(new.slow_curve_delta.pathwise_option_factor_sensitivity)
    assert not existing.exercise_now
    later = price_existing_carry_put(contract, PARAMS, state, gbm,
                                   elapsed_sessions=7, current_spot=6100, current_futures=5900,
                                   numerical=NUMERICAL)
    expected = max(6100 * exp((.014 - contract.locked_carry(.014)) / 244) - 5900, 0)
    assert later.locked_carry == contract.locked_carry(.014)
    assert later.price == pytest.approx(expected)
    assert later.continuation_value == 0
    assert later.exercise_now
    # Even a nonzero expiry close basis cannot create a terminal option payoff.
    terminal = price_existing_carry_put(contract, PARAMS, state, gbm,
                                      elapsed_sessions=8, current_spot=6100, current_futures=5900,
                                      numerical=NUMERICAL)
    assert terminal.expired and terminal.price == terminal.exercise_value == 0
    assert terminal.slow_factor_sensitivity == terminal.fast_factor_sensitivity == 0


@pytest.mark.parametrize("elapsed", [-1, 9, 0.5, True])
def test_existing_contract_rejects_invalid_elapsed(elapsed):
    with pytest.raises(ValueError):
        price_existing_carry_put(CarryPutContract(6000, 5950, 8), PARAMS, FactorState(0, 0),
                                GBMParams(.014, .25), elapsed_sessions=elapsed,
                                current_spot=6000, current_futures=5950, numerical=NUMERICAL)


def test_existing_contract_continuation_not_relocked_to_new_quote():
    contract = CarryPutContract(6000, 5950, 8)
    kwargs = dict(elapsed_sessions=2, current_spot=6000, numerical=NUMERICAL)
    # Both quotes are out of the money today; tomorrow's continuation must be
    # identical because q0 and state, not today's noisy quote, specify it.
    a = price_existing_carry_put(contract, PARAMS, FactorState(0, 0), GBMParams(.014, .25),
                                current_futures=6200, **kwargs)
    b = price_existing_carry_put(contract, PARAMS, FactorState(0, 0), GBMParams(.014, .25),
                                current_futures=6300, **kwargs)
    assert not a.exercise_now and not b.exercise_now
    assert a.price == b.price
    assert a.slow_factor_sensitivity == b.slow_factor_sensitivity


def test_cash_ledger_signs_costs_financing_and_no_futures_purchase_outlay():
    cfg = replace(CONFIG, futures_multiplier=200, fee_per_contract=3, slippage_points=.2)
    ledger = CashLedger(-100.)
    turnover, cost = ledger.trade(np.array([2., -1.]), cfg)
    assert turnover == 3 and cost == 129
    assert ledger.cash == -229  # no futures notional debit
    financing, pnl = ledger.accrue(np.array([6000., 5900.]), np.array([6010., 5905.]), 1/244, cfg)
    assert pnl == 3000
    assert financing == pytest.approx(-229 * np.expm1(.014/244))
    ledger.trade(np.zeros(2), cfg)
    assert ledger.cash == pytest.approx(-100 + 3000 + financing - 258)


def test_three_scenarios_have_common_exit_and_reconcile_to_expiry():
    result = run()
    assert len(result.summaries) == 3
    assert set(result.summaries.exit_reason) == {"expiry"}
    assert result.summaries.exit_date.nunique() == 1
    assert result.cohorts.iloc[0].hedge_contract_2 == "IM2506"
    assert result.cohorts.iloc[0].sample_curve_dates == 10
    for _, group in result.daily.groupby("scenario"):
        assert group.iloc[0].daily_futures_pnl == 0
        assert group.iloc[-1].contracts_1 == group.iloc[-1].contracts_2 == 0
        expected_futures = ((group.held_contracts_1 * group.futures_1.diff()
                            + group.held_contracts_2 * group.futures_2.diff()).fillna(0)).sum()
        assert group.iloc[-1].cumulative_futures_pnl == pytest.approx(expected_futures)
        assert group.discounted_pnl_change.sum() == pytest.approx(group.iloc[-1].discounted_equity)
    for row in result.summaries.itertuples():
        assert row.total_pnl == pytest.approx(row.option_pnl + row.futures_pnl + row.financing - row.transaction_costs)
    two = result.daily.query("scenario == 'two_futures' and not expired")
    assert np.max(np.abs(two.residual_slow_exposure)) < 1e-10
    assert np.max(np.abs(two.residual_fast_exposure)) < 1e-10
    assert (two.residual_scale_exposure.abs() > 0).all()


def test_early_exercise_closes_all_scenarios_and_no_replacement_cohort():
    def exercise(*args, **kwargs):
        value = value_stub(*args, **kwargs)
        if kwargs["elapsed_sessions"] == 2:
            return replace(value, price=40., exercise_value=40., exercise_now=True)
        return value
    result = run(valuer=exercise, config=replace(CONFIG, fee_per_contract=1.))
    assert len(result.cohorts) == 1
    assert len(result.daily) == 9
    assert set(result.summaries.exit_reason) == {"exercise"}
    assert set(result.summaries.option_payoff) == {40.}
    last = result.daily.groupby("scenario").tail(1)
    assert (last[["contracts_1", "contracts_2", "option_mark"]] == 0).all().all()
    assert last.query("scenario == 'two_futures'").transaction_cost.iloc[0] > 0


def test_lagged_execution_earns_pnl_only_after_actual_trade():
    result = run(config=replace(CONFIG, execution_lag_sessions=1))
    one = result.daily.query("scenario == 'one_futures'").reset_index(drop=True)
    assert one.loc[0, "contracts_1"] == 0
    assert one.loc[1, "daily_futures_pnl"] == 0
    assert one.loc[1, "contracts_1"] == pytest.approx(one.loc[0, "target_contracts_1"])
    assert one.loc[2, "daily_futures_pnl"] == pytest.approx(
        one.loc[1, "contracts_1"] * (one.loc[2, "futures_1"] - one.loc[1, "futures_1"])
    )


def test_multipliers_and_rounding_applied_to_actual_contracts():
    cfg = replace(CONFIG, option_units=10, option_multiplier=200, futures_multiplier=200,
                  round_contracts=True, fee_per_contract=2)
    result = run(config=cfg)
    positions = result.daily[["contracts_1", "contracts_2"]].to_numpy()
    np.testing.assert_equal(positions, np.rint(positions))
    assert set(result.summaries.initial_premium) == {40000.}
    two = result.daily.query("scenario == 'two_futures' and not expired")
    assert (two.residual_factor_variance > 0).any()
    assert (result.summaries.transaction_costs >= 0).all()


def test_missing_quote_does_not_shift_entry_or_choose_future_surviving_pair():
    market = make_market()
    missing = market.futures.loc[~((market.futures.date == ENTRY) & (market.futures.contract == "IM2502"))]
    result = run(MarketData(market.spot, missing))
    assert result.daily.empty
    assert len(result.cohorts) == 1
    assert "Missing/invalid close" in result.cohorts.iloc[0].reason
    # IM2503 has complete history, but cannot replace the inception-selected IM2506.
    missing = market.futures.loc[~((market.futures.date == EXPIRY) & (market.futures.contract == "IM2506"))]
    result = run(MarketData(market.spot, missing))
    assert result.summaries.empty
    assert "IM2506" in result.cohorts.iloc[0].reason


def test_incomplete_cohort_rejected_before_calibration_even_if_it_would_exercise():
    market = make_market()
    truncated = MarketData(market.spot.iloc[:-1], market.futures)
    def forbidden(*args, **kwargs):
        raise AssertionError("Estimator must not be called for incomplete cohort")
    result = run(truncated, estimator=forbidden)
    assert result.daily.empty
    assert "Incomplete" in result.cohorts.iloc[0].reason


def test_incremental_filter_matches_batch_filter_and_prediction_only():
    panel, _ = make_market().observations_through(EXPIRY)
    dates = sorted(panel.date.unique())
    first = two_factor_kalman_filter(panel.loc[panel.date == dates[0]], FILTER_PARAMS,
                                    gap_function=lambda a, b: trading_days_between(a, b)/244,
                                    observation_noise_model="constant_log_futures")
    from carry_put_backtest.engine import _state_arrays
    mean, cov = _state_arrays(first.states.iloc[-1])
    for prev, date in zip(dates, dates[1:]):
        mean, cov = advance_filter(mean, cov, FILTER_PARAMS, panel.loc[panel.date == date],
                                   trading_days_between(prev, date)/244, "constant_log_futures")
    batch = two_factor_kalman_filter(panel, FILTER_PARAMS,
                                    gap_function=lambda a, b: trading_days_between(a, b)/244,
                                    observation_noise_model="constant_log_futures")
    expected_mean, expected_cov = _state_arrays(batch.states.iloc[-1])
    np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(cov, expected_cov, atol=1e-12)
    a, q = transition(FILTER_PARAMS, 1/244)
    pred, pcov = advance_filter(mean, cov, FILTER_PARAMS, panel.iloc[:0], 1/244, "constant_log_futures")
    np.testing.assert_allclose(pred, a @ mean)
    np.testing.assert_allclose(pcov, a @ cov @ a.T + q)


def test_future_changes_do_not_change_inception_fit_input_or_prior_hedges():
    market = make_market()
    modified = market.futures.copy()
    cutoff = session_dates(ENTRY, EXPIRY)[3]
    modified.loc[modified.date > cutoff, "futures_price"] *= .98
    samples = []
    def capture(sample, **kwargs):
        samples.append(sample.reset_index(drop=True))
        return estimate_stub(sample, **kwargs)
    original = run(market, estimator=capture)
    changed = run(MarketData(market.spot, modified), estimator=capture)
    assert len(samples) == 2  # one calibration per cohort, not a daily refit
    pd.testing.assert_frame_equal(samples[0], samples[1])
    pd.testing.assert_frame_equal(original.daily.loc[original.daily.date <= cutoff].reset_index(drop=True),
                                  changed.daily.loc[changed.daily.date <= cutoff].reset_index(drop=True))


def test_nonconverged_calibration_not_in_performance_comparison():
    def failed(sample, **kwargs):
        value = estimate_stub(sample, **kwargs)
        value.converged = False
        return value
    result = run(estimator=failed)
    assert result.summaries.empty
    assert "did not converge" in result.cohorts.iloc[0].reason


def test_real_pricer_synthetic_cohort_and_export(tmp_path):
    result = run(valuer=price_existing_carry_put)
    assert len(result.summaries) == 3
    assert np.isfinite(result.summaries.total_pnl).all()
    export_result(result, tmp_path, request={"test": "synthetic"})
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["config"]["window_dates"] == 10
    assert manifest["request"] == {"test": "synthetic"}
    assert manifest["sha256"]
    assert len(pd.read_csv(tmp_path / "cohort_pnl.csv")) == 3
    assert len(aggregate_results(result)) == 3


def test_empty_schedule_exports_cleanly(tmp_path):
    result = run_backtest(make_market(), start="2025-01-21", end="2025-01-22", config=CONFIG)
    assert result.daily.empty and result.cohorts.empty
    export_result(result, tmp_path)
    assert json.loads((tmp_path / "run_manifest.json").read_text())["aggregate"] == []
    assert pd.read_csv(tmp_path / "daily_ledger.csv").empty


def test_mixed_completed_and_unavailable_cohorts_export_null_dates(tmp_path):
    result = run_backtest(make_market(), start=ENTRY, end="2025-02-24", config=CONFIG,
                          estimator=estimate_stub, valuer=value_stub)
    assert list(result.cohorts.status) == ["completed", "unavailable"]
    export_result(result, tmp_path)
    records = json.loads((tmp_path / "run_manifest.json").read_text())["cohorts"]
    assert records[1]["exit_date"] is None


def test_near_expiry_raw_quotes_remain_available_for_cash_marks():
    market = make_market()
    panel, audit = market.observations_through(EXPIRY)
    near = audit.loc[(audit.contract == "IM2502") & (audit.sessions_to_expiry <= 5)]
    assert near.excluded.all()
    assert market.quote(EXPIRY, "IM2502") > 0
    result = run(market)
    assert result.cohorts.iloc[0].status == "completed"


def test_singular_pair_is_explicitly_unavailable_not_a_zero_hedge():
    from carry_put_backtest.engine import _targets
    value = ExistingCarryPutResult(20, 20, 0, False, False, .1, 10, 180, 30, 6000, 0)
    pair = (HedgeFuturesContract("IM2502", 6000, 20), HedgeFuturesContract("IM2502", 6000, 20))
    with pytest.warns(RuntimeWarning, match="singular"):
        with pytest.raises(CohortUnavailable, match="Singular"):
            _targets(value, PARAMS, pair, CONFIG)


def test_cached_loader_uses_close_and_never_settlement(tmp_path):
    pd.DataFrame({"date": [ENTRY], "close": [6000], "settle": [0]}).to_csv(tmp_path / "spot_raw.csv", index=False)
    pd.DataFrame({"date": [ENTRY], "contract": ["IM2502"], "close": [5950], "settle": [0]}).to_csv(tmp_path / "futures_raw.csv", index=False)
    market = load_cached_market(tmp_path)
    assert market.quote(ENTRY) == 6000
    assert market.quote(ENTRY, "IM2502") == 5950


def test_causal_stale_flags_do_not_retroactively_change():
    market = make_market()
    dates = sorted(market.futures.date.unique())
    data = market.futures.copy()
    data.loc[data.contract == "IM2506", "futures_price"] = 5900.
    market = MarketData(market.spot, data)
    _, early = market.observations_through(dates[1])
    _, late = market.observations_through(dates[3])
    assert not early.quality_flags.str.contains("stale_price_run").any()
    pd.testing.assert_frame_equal(
        early.reset_index(drop=True), late.loc[late.date <= dates[1]].reset_index(drop=True)
    )
    assert late.loc[(late.contract == "IM2506") & (late.date == dates[2]), "quality_flags"].str.contains("stale_price_run").all()


@pytest.mark.parametrize("kwargs", [
    {"window_dates": 1}, {"execution_lag_sessions": 2}, {"fee_per_contract": -1},
    {"futures_multiplier": 0}, {"risk_free_rate": np.nan}, {"kappa_gap_upper_bound": .001},
])
def test_invalid_config_rejected_before_work(kwargs):
    with pytest.raises(ValueError):
        replace(CONFIG, **kwargs)


def test_independent_replay_reproduces_ledger_and_lag_sensitivity():
    from carry_put_backtest.analyze_historical import replay
    baseline = run()
    reproduced, totals = replay(baseline.daily, baseline.summaries)
    merged = baseline.daily.merge(reproduced, on=["cohort", "scenario", "date", "elapsed_sessions"],
                                  suffixes=("_base", "_replay"))
    np.testing.assert_allclose(merged.equity_base, merged.equity_replay, atol=1e-10)
    lagged = run(config=replace(CONFIG, execution_lag_sessions=1))
    reproduced, totals = replay(baseline.daily, baseline.summaries, lag=1)
    merged = lagged.daily.merge(reproduced, on=["cohort", "scenario", "date", "elapsed_sessions"],
                                suffixes=("_base", "_replay"))
    np.testing.assert_allclose(merged.equity_base, merged.equity_replay, atol=1e-10)
    costed = run(config=replace(CONFIG, slippage_points=.2))
    reproduced, totals = replay(baseline.daily, baseline.summaries, cost_points=.2)
    merged = costed.daily.merge(reproduced, on=["cohort", "scenario", "date", "elapsed_sessions"],
                                suffixes=("_base", "_replay"))
    np.testing.assert_allclose(merged.equity_base, merged.equity_replay, atol=1e-10)


def test_calibration_cache_key_tracks_actual_sample_and_settings(tmp_path, monkeypatch):
    from carry_put_backtest import historical_validation as validation
    calls = []
    def estimate(sample, **kwargs):
        calls.append(kwargs)
        return estimate_stub(sample, **kwargs)
    monkeypatch.setattr(validation, "estimate_two_factor_ou", estimate)
    cached = validation.cached_estimator(tmp_path)
    panel, _ = make_market().observations_through(ENTRY)
    sample = calibration_window(panel, ENTRY, 10)
    first = cached(sample, starts=12)
    second = cached(sample.copy(), starts=12)
    assert first.params == second.params
    assert len(calls) == 1
    changed = sample.copy()
    changed.loc[changed.index[0], "implied_carry"] += .001
    cached(changed, starts=12)
    cached(sample, starts=13)
    assert len(calls) == 3
    assert len(list(tmp_path.glob("*.json"))) == 3
