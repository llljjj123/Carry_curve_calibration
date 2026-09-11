"""Frozen-parameter monthly cohorts with a self-financing daily cash ledger."""

from dataclasses import asdict, dataclass, field
from math import exp
from typing import Callable

import numpy as np
import pandas as pd

from calendar_utils import contract_expiry, trading_days_between
from carry_put_pricing import (
    CarryPutContract, FactorState, GBMParams, HedgeFuturesContract, NumericalConfig,
    TwoFactorOUParams, calculate_one_futures_hedge, calculate_two_futures_hedge,
    factor_innovation_covariance, price_existing_carry_put,
)
from im_2factor_ou_carry.two_factor import two_factor_kalman_filter, transition
from im_2factor_ou_carry.two_factor_estimation import estimate_two_factor_ou

from .market import CohortUnavailable, MarketData, monthly_cohorts, session_dates


SCENARIOS = ("no_hedge", "one_futures", "two_futures")
STUDIES = ("long_futures_only", "short_with_spot")


@dataclass(frozen=True)
class ScenarioSpec:
    """Stable scenario definition used by the explicit short/spot study."""

    scenario: str
    option_side: int
    futures_count: int
    spot_enabled: bool
    role: str
    carry_hedge: str


SHORT_SCENARIOS = (
    ScenarioSpec("short_no_hedge", -1, 0, False, "primary", "none"),
    ScenarioSpec("short_one_futures_spot", -1, 1, True, "primary", "one_futures_min_variance"),
    ScenarioSpec("short_two_futures_spot", -1, 2, True, "primary", "two_factor_neutral"),
    ScenarioSpec("short_one_futures_only", -1, 1, False, "control", "one_futures_min_variance"),
    ScenarioSpec("short_two_futures_only", -1, 2, False, "control", "two_factor_neutral"),
)


def scenario_specs(study: str) -> tuple[ScenarioSpec, ...]:
    """Return the ordered scenario definitions for a configured study."""
    if study == "short_with_spot":
        return SHORT_SCENARIOS
    if study == "long_futures_only":
        return tuple(
            ScenarioSpec(name, 1, i, False, "legacy", name)
            for i, name in enumerate(SCENARIOS)
        )
    raise ValueError(f"Unknown study: {study}")


@dataclass(frozen=True)
class BacktestConfig:
    window_dates: int = 488
    risk_free_rate: float = 0.014
    observation_noise_model: str = "constant_log_futures"
    optimizer_starts: int = 12
    optimizer_maxiter: int = 1500
    seed: int = 852
    kappa_gap_upper_bound: float = 90.0
    eta_fast_upper_bound: float = 6.0
    option_units: float = 1.0
    option_multiplier: float = 1.0
    futures_multiplier: float = 1.0
    fee_per_contract: float = 0.0
    slippage_points: float = 0.0
    round_contracts: bool = False
    execution_lag_sessions: int = 0
    condition_warning: float = 1000.0
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    study: str = "long_futures_only"
    spot_cost_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("window_dates", "optimizer_starts", "optimizer_maxiter"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < (2 if name == "window_dates" else 1):
                raise ValueError(f"Invalid {name}")
        for name in ("option_units", "option_multiplier", "futures_multiplier", "condition_warning"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name, lower in (("kappa_gap_upper_bound", .01), ("eta_fast_upper_bound", 1e-4)):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= lower:
                raise ValueError(f"{name} must be finite and greater than {lower}")
        for name in ("fee_per_contract", "slippage_points"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not np.isfinite(self.spot_cost_bps) or self.spot_cost_bps < 0:
            raise ValueError("spot_cost_bps must be nonnegative and finite")
        if not np.isfinite(self.risk_free_rate):
            raise ValueError("risk_free_rate must be finite")
        if isinstance(self.execution_lag_sessions, bool) or self.execution_lag_sessions not in (0, 1):
            raise ValueError("execution_lag_sessions must be 0 or 1")
        if self.observation_noise_model not in ("constant_carry", "constant_log_futures"):
            raise ValueError("Unknown observation noise model")
        if self.study not in STUDIES:
            raise ValueError(f"Unknown study: {self.study}")

    @property
    def option_point_value(self) -> float:
        return self.option_units * self.option_multiplier


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    summaries: pd.DataFrame
    cohorts: pd.DataFrame
    optimizer_runs: pd.DataFrame
    config: BacktestConfig


def _gap(start: object, end: object) -> float:
    return trading_days_between(start, end) / 244.0


def calibration_window(accepted: pd.DataFrame, entry: object, count: int) -> pd.DataFrame:
    """Exactly count accepted curve dates through entry; never backfill a missed entry."""
    history = accepted.loc[accepted.date <= pd.Timestamp(entry)].copy()
    dates = pd.DatetimeIndex(history.date.unique()).sort_values()
    if len(dates) < count:
        raise CohortUnavailable(f"Need {count} accepted curve dates; found {len(dates)}")
    if dates[-1] != pd.Timestamp(entry):
        raise CohortUnavailable("Inception has no accepted carry curve")
    return history.loc[history.date.isin(dates[-count:])].sort_values(["date", "tau"])


def _state_arrays(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array([row.filtered_slow_state, row.filtered_fast_state])
    covariance = np.array([
        [row.filtered_var_slow, row.filtered_cov_slow_fast],
        [row.filtered_cov_slow_fast, row.filtered_var_fast],
    ])
    return mean, covariance


def advance_filter(mean, covariance, params, observations, dt, noise_model):
    """Predict once, then update with this date only; empty curves are prediction-only."""
    a, q = transition(params, dt)
    mean, covariance = a @ mean, a @ covariance @ a.T + q
    if not observations.empty:
        if observations.date.nunique() != 1:
            raise ValueError("advance_filter accepts exactly one observation date")
        filtered = two_factor_kalman_filter(
            observations, params, gap_function=_gap,
            initial_mean=mean, initial_covariance=covariance,
            observation_noise_model=noise_model,
        )
        mean, covariance = _state_arrays(filtered.states.iloc[-1])
    return mean, covariance


@dataclass
class CashLedger:
    """Positions are contract counts; marks are points; all cash is monetary units."""
    cash: float
    positions: np.ndarray = field(default_factory=lambda: np.zeros(2))
    cumulative_futures_pnl: float = 0.0
    cumulative_financing: float = 0.0
    cumulative_costs: float = 0.0

    def accrue(self, previous_prices, current_prices, dt, config):
        financing = self.cash * np.expm1(config.risk_free_rate * dt)
        futures_pnl = float(self.positions @ (current_prices - previous_prices)) * config.futures_multiplier
        self.cash += financing + futures_pnl
        self.cumulative_financing += financing
        self.cumulative_futures_pnl += futures_pnl
        return float(financing), futures_pnl

    def trade(self, target, config):
        turnover = float(np.abs(target - self.positions).sum())
        costs = turnover * (config.fee_per_contract + config.slippage_points * config.futures_multiplier)
        self.cash -= costs
        self.cumulative_costs += costs
        self.positions = np.asarray(target, dtype=float).copy()
        return turnover, costs


@dataclass
class FundedCashLedger:
    """Self-financing ledger for futures plus continuously sized spot.

    Futures have no principal mark in cash: only their daily settlement P&L is
    credited. Spot is purchased from cash and remains an asset in marked
    equity. Its daily mark is recorded for attribution but is deliberately not
    credited to cash a second time.
    """

    cash: float
    futures_positions: np.ndarray = field(default_factory=lambda: np.zeros(2))
    spot_units: float = 0.0
    cumulative_futures_pnl: float = 0.0
    cumulative_spot_pnl: float = 0.0
    cumulative_financing: float = 0.0
    cumulative_futures_costs: float = 0.0
    cumulative_spot_costs: float = 0.0

    @property
    def cumulative_costs(self) -> float:
        return self.cumulative_futures_costs + self.cumulative_spot_costs

    def accrue(self, previous_prices, current_prices, previous_spot, current_spot, dt, config):
        financing = self.cash * np.expm1(config.risk_free_rate * dt)
        futures_pnl = (
            float(self.futures_positions @ (current_prices - previous_prices))
            * config.futures_multiplier
        )
        spot_pnl = float(self.spot_units * (current_spot - previous_spot))
        self.cash += financing + futures_pnl
        self.cumulative_financing += financing
        self.cumulative_futures_pnl += futures_pnl
        self.cumulative_spot_pnl += spot_pnl
        return float(financing), float(futures_pnl), float(spot_pnl)

    def trade(self, futures_target, spot_target, spot_price, config):
        futures_target = np.asarray(futures_target, dtype=float)
        if futures_target.shape != (2,) or not np.isfinite(futures_target).all():
            raise ValueError("futures_target must contain two finite positions")
        if not np.isfinite(spot_target) or not np.isfinite(spot_price) or spot_price <= 0:
            raise ValueError("spot target and price must be finite; spot price must be positive")
        futures_turnover = float(np.abs(futures_target - self.futures_positions).sum())
        futures_cost = futures_turnover * (
            config.fee_per_contract + config.slippage_points * config.futures_multiplier
        )
        spot_change = float(spot_target - self.spot_units)
        spot_trade_cash_flow = -spot_price * spot_change
        spot_turnover_units = abs(spot_change)
        spot_cost = spot_turnover_units * spot_price * config.spot_cost_bps / 10000.0
        self.cash += spot_trade_cash_flow - futures_cost - spot_cost
        self.cumulative_futures_costs += futures_cost
        self.cumulative_spot_costs += spot_cost
        self.futures_positions = futures_target.copy()
        self.spot_units = float(spot_target)
        return (
            futures_turnover, float(futures_cost), spot_turnover_units,
            float(spot_trade_cash_flow), float(spot_cost),
        )


def _targets(value, params, hedge_contracts, config):
    if config.study == "short_with_spot":
        return _short_targets(value, params, hedge_contracts, config)
    kwargs = dict(
        option_slow_factor_sensitivity=value.slow_factor_sensitivity,
        option_fast_factor_sensitivity=value.fast_factor_sensitivity,
        ou_params=params,
    )
    one = calculate_one_futures_hedge(**kwargs, hedge_future=hedge_contracts[0])
    two = calculate_two_futures_hedge(**kwargs, hedge_futures=hedge_contracts)
    if two.is_singular:
        raise CohortUnavailable("Singular two-futures hedge; no fallback or zero position substituted")
    positions = np.array([[0., 0.], [one, 0.], [two.hedge_position_1, two.hedge_position_2]])
    positions *= config.option_point_value / config.futures_multiplier
    if config.round_contracts:
        positions = np.rint(positions)
    if not np.isfinite(positions).all():
        raise CohortUnavailable("Nonfinite hedge positions")
    g = np.array([
        [two.slow_futures_sensitivity_1, two.slow_futures_sensitivity_2],
        [two.fast_futures_sensitivity_1, two.fast_futures_sensitivity_2],
    ])
    return positions, g, two


def _short_targets(value, params, hedge_contracts, config):
    """Build long-helper-derived short-option futures targets and diagnostics."""
    kwargs = dict(
        option_slow_factor_sensitivity=value.slow_factor_sensitivity,
        option_fast_factor_sensitivity=value.fast_factor_sensitivity,
        ou_params=params,
    )
    long_one = calculate_one_futures_hedge(**kwargs, hedge_future=hedge_contracts[0])
    two = calculate_two_futures_hedge(**kwargs, hedge_futures=hedge_contracts)
    if two.is_singular:
        raise CohortUnavailable(
            "Singular two-futures hedge; no fallback or zero position substituted"
        )
    # The pricing helpers return the hedge for one LONG option with unit
    # monetary value. A short option is the exact opposite, with the monetary
    # option/futures multiplier conversion applied once here.
    scale = config.option_point_value / config.futures_multiplier
    positions = np.array([
        [0.0, 0.0],
        [-long_one * scale, 0.0],
        [-two.hedge_position_1 * scale, -two.hedge_position_2 * scale],
    ])
    if config.round_contracts:
        positions = np.rint(positions)
    if not np.isfinite(positions).all():
        raise CohortUnavailable("Nonfinite hedge positions")
    g = np.array([
        [two.slow_futures_sensitivity_1, two.slow_futures_sensitivity_2],
        [two.fast_futures_sensitivity_1, two.fast_futures_sensitivity_2],
    ])
    return positions, g, two


def run_cohort(market, cohort, config, *, estimator=estimate_two_factor_ou,
               valuer=price_existing_carry_put):
    """One completed scheduled cohort. Injectable estimator/valuer support focused tests."""
    if config.study == "short_with_spot":
        return _run_short_cohort(market, cohort, config, estimator=estimator, valuer=valuer)
    entry, expiry = cohort["entry_date"], cohort["expiry"]
    pair = market.hedge_pair(entry, cohort["option_contract"])
    dates = session_dates(entry, expiry)
    # Require scheduled-life coverage even if the eventual policy exercises early.
    # Selection is already fixed; missing later quotes never trigger a new pair.
    if expiry > min(market.spot.date.max(), market.futures.date.max()):
        raise CohortUnavailable("Incomplete data through scheduled expiry")
    for date in dates:
        market.quote(date)
        for code in pair:
            market.quote(date, code)

    accepted, _ = market.observations_through(entry)
    sample = calibration_window(accepted, entry, config.window_dates)
    estimate = estimator(
        sample, gap_function=_gap, starts=config.optimizer_starts,
        maxiter=config.optimizer_maxiter, seed=config.seed,
        compute_standard_errors=False, eta_fast_upper_bound=config.eta_fast_upper_bound,
        kappa_gap_upper_bound=config.kappa_gap_upper_bound,
        observation_noise_model=config.observation_noise_model,
    )
    if not estimate.converged:
        raise CohortUnavailable(f"Calibration did not converge: {estimate.message}")
    frozen = estimate.params
    frozen.validate()
    pricing_params = TwoFactorOUParams(**{
        k: v for k, v in asdict(frozen).items() if k != "sigma_epsilon"
    })
    filtered = two_factor_kalman_filter(
        sample, frozen, gap_function=_gap, observation_noise_model=config.observation_noise_model,
    )
    mean, covariance = _state_arrays(filtered.states.iloc[-1])
    # Only pointwise quality exclusions affect acceptance; causal stale flags
    # are diagnostic. Each filter update below sees only that day's rows.
    subsequent, _ = market.observations_through(expiry)
    contract = CarryPutContract(market.quote(entry), market.quote(entry, pair[0]), len(dates) - 1)
    gbm = GBMParams(config.risk_free_rate, 0.0)  # cancels under model homogeneity
    q = factor_innovation_covariance(pricing_params)
    rows, ledgers = [], None
    pending = np.zeros((3, 2))
    previous_prices = None
    previous_discounted = np.zeros(3)
    metadata = {
        **cohort, "hedge_contract_1": pair[0], "hedge_contract_2": pair[1],
        "sample_start": sample.date.min(), "sample_end": sample.date.max(),
        "sample_curve_dates": sample.date.nunique(), "optimizer_converged": True,
        "calibration_log_likelihood": estimate.log_likelihood,
        "locked_carry": contract.locked_carry(config.risk_free_rate),
        **asdict(frozen),
        "kappa_gap_at_bound": bool(frozen.kappa_fast - frozen.kappa_slow >= config.kappa_gap_upper_bound - 1e-6),
        "eta_fast_at_bound": bool(frozen.eta_fast >= config.eta_fast_upper_bound - 1e-6),
    }
    for elapsed, date in enumerate(dates):
        observations = subsequent.loc[subsequent.date == date]
        if elapsed:
            mean, covariance = advance_filter(
                mean, covariance, frozen, observations, _gap(dates[elapsed - 1], date),
                config.observation_noise_model,
            )
        prices = np.array([market.quote(date, c) for c in pair])
        spot = market.quote(date)
        value = valuer(
            contract, pricing_params, FactorState(*mean), gbm,
            elapsed_sessions=elapsed, current_spot=spot, current_futures=prices[0],
            numerical=config.numerical,
        )
        if not np.isfinite([value.price, value.continuation_value, value.exercise_value,
                            value.slow_factor_sensitivity, value.fast_factor_sensitivity]).all():
            raise CohortUnavailable("Nonfinite option valuation")
        if ledgers is None:
            premium = config.option_point_value * value.price
            metadata["initial_premium_points"] = value.price
            ledgers = [CashLedger(-premium) for _ in SCENARIOS]
        terminal = value.exercise_now or value.expired
        if terminal:
            targets, g, diagnostic = np.zeros((3, 2)), np.zeros((2, 2)), None
        else:
            hedge_contracts = tuple(HedgeFuturesContract(
                code, prices[i], trading_days_between(date, contract_expiry(code))
            ) for i, code in enumerate(pair))
            targets, g, diagnostic = _targets(value, pricing_params, hedge_contracts, config)
        b = config.option_point_value * np.array([
            value.slow_factor_sensitivity, value.fast_factor_sensitivity,
        ])
        for i, (scenario, ledger) in enumerate(zip(SCENARIOS, ledgers)):
            financing, futures_pnl = (0.0, 0.0) if elapsed == 0 else ledger.accrue(
                previous_prices, prices, _gap(dates[elapsed - 1], date), config
            )
            held = ledger.positions.copy()
            payoff = config.option_point_value * value.exercise_value if terminal else 0.0
            ledger.cash += payoff
            target = (np.zeros(2) if terminal else
                      pending[i] if config.execution_lag_sessions else targets[i])
            turnover, costs = ledger.trade(target, config)
            option_mark = 0.0 if terminal else config.option_point_value * value.price
            equity = ledger.cash + option_mark
            discounted = equity * exp(-config.risk_free_rate * elapsed / 244)
            residual = np.zeros(2) if terminal else b + g @ (ledger.positions * config.futures_multiplier)
            scale = 0.0 if terminal else option_mark + float(ledger.positions @ prices) * config.futures_multiplier
            rows.append({
                "cohort": cohort["cohort"], "date": date, "scenario": scenario,
                "elapsed_sessions": elapsed, "remaining_sessions": value.remaining_sessions,
                "hedge_contract_1": pair[0], "hedge_contract_2": pair[1],
                "spot": spot, "futures_1": prices[0], "futures_2": prices[1],
                "filtered_slow_state": mean[0], "filtered_fast_state": mean[1],
                "filtered_var_slow": covariance[0, 0], "filtered_var_fast": covariance[1, 1],
                "filtered_cov_slow_fast": covariance[0, 1],
                "accepted_contracts": len(observations), "prediction_only": observations.empty,
                "locked_carry": value.locked_carry, "option_price_points": value.price,
                "continuation_points": value.continuation_value,
                "exercise_value_points": value.exercise_value,
                "model_futures": value.model_futures, "futures_model_error": value.futures_model_error,
                "held_contracts_1": held[0], "held_contracts_2": held[1],
                "target_contracts_1": targets[i, 0], "target_contracts_2": targets[i, 1],
                "contracts_1": ledger.positions[0], "contracts_2": ledger.positions[1],
                "turnover_contracts": turnover, "transaction_cost": costs,
                "daily_futures_pnl": futures_pnl, "daily_financing": financing,
                "option_payoff": payoff, "cash": ledger.cash, "option_mark": option_mark,
                "equity": equity, "discounted_equity": discounted,
                "discounted_pnl_change": discounted - previous_discounted[i],
                "cumulative_futures_pnl": ledger.cumulative_futures_pnl,
                "cumulative_financing": ledger.cumulative_financing,
                "cumulative_costs": ledger.cumulative_costs,
                "residual_slow_exposure": residual[0], "residual_fast_exposure": residual[1],
                "residual_factor_variance": float(residual @ q @ residual),
                "unhedged_factor_variance": 0.0 if terminal else float(b @ q @ b),
                "residual_scale_exposure": scale,
                "hedge_condition_number": np.nan if diagnostic is None else diagnostic.condition_number,
                "hedge_angular_separation": np.nan if diagnostic is None else diagnostic.angular_separation,
                "conditioning_warning": False if diagnostic is None else diagnostic.condition_number > config.condition_warning,
                "exercise_now": value.exercise_now, "expired": value.expired,
            })
            previous_discounted[i] = discounted
        pending, previous_prices = targets.copy(), prices
        if terminal:
            metadata.update(status="completed", exit_date=date,
                            exit_reason="exercise" if value.exercise_now else "expiry")
            break
    if metadata.get("status") != "completed":
        raise RuntimeError("Cohort reached expiry without closing")
    daily = pd.DataFrame(rows)
    summaries = summarize_cohort(daily, metadata, config)
    optimizer_runs = estimate.optimizer_runs.copy()
    optimizer_runs.insert(0, "cohort", cohort["cohort"])
    return daily, summaries, metadata, optimizer_runs


def _run_short_cohort(market, cohort, config, *, estimator=estimate_two_factor_ou,
                      valuer=price_existing_carry_put):
    """Run the five-scenario short-option study with funded spot accounting."""
    specs = SHORT_SCENARIOS
    entry, expiry = cohort["entry_date"], cohort["expiry"]
    pair = market.hedge_pair(entry, cohort["option_contract"])
    dates = session_dates(entry, expiry)
    if expiry > min(market.spot.date.max(), market.futures.date.max()):
        raise CohortUnavailable("Incomplete data through scheduled expiry")
    for date in dates:
        market.quote(date)
        for code in pair:
            market.quote(date, code)

    accepted, _ = market.observations_through(entry)
    sample = calibration_window(accepted, entry, config.window_dates)
    estimate = estimator(
        sample, gap_function=_gap, starts=config.optimizer_starts,
        maxiter=config.optimizer_maxiter, seed=config.seed,
        compute_standard_errors=False, eta_fast_upper_bound=config.eta_fast_upper_bound,
        kappa_gap_upper_bound=config.kappa_gap_upper_bound,
        observation_noise_model=config.observation_noise_model,
    )
    if not estimate.converged:
        raise CohortUnavailable(f"Calibration did not converge: {estimate.message}")
    frozen = estimate.params
    frozen.validate()
    pricing_params = TwoFactorOUParams(**{
        k: v for k, v in asdict(frozen).items() if k != "sigma_epsilon"
    })
    filtered = two_factor_kalman_filter(
        sample, frozen, gap_function=_gap,
        observation_noise_model=config.observation_noise_model,
    )
    mean, covariance = _state_arrays(filtered.states.iloc[-1])
    subsequent, _ = market.observations_through(expiry)
    contract = CarryPutContract(
        market.quote(entry), market.quote(entry, pair[0]), len(dates) - 1
    )
    gbm = GBMParams(config.risk_free_rate, 0.0)
    q = factor_innovation_covariance(pricing_params)
    ledgers = [FundedCashLedger(0.0) for _ in specs]
    cumulative_option_cash = np.zeros(len(specs))
    rows = []
    pending_futures = np.zeros((len(specs), 2))
    pending_spot = np.zeros(len(specs))
    previous_prices = None
    previous_spot = None
    previous_discounted = np.zeros(len(specs))
    metadata = {
        **cohort,
        "study": config.study,
        "scenario_count": len(specs),
        "option_side": -1,
        "hedge_contract_1": pair[0], "hedge_contract_2": pair[1],
        "sample_start": sample.date.min(), "sample_end": sample.date.max(),
        "sample_curve_dates": sample.date.nunique(), "optimizer_converged": True,
        "calibration_log_likelihood": estimate.log_likelihood,
        "locked_carry": contract.locked_carry(config.risk_free_rate),
        **asdict(frozen),
        "kappa_gap_at_bound": bool(
            frozen.kappa_fast - frozen.kappa_slow >= config.kappa_gap_upper_bound - 1e-6
        ),
        "eta_fast_at_bound": bool(
            frozen.eta_fast >= config.eta_fast_upper_bound - 1e-6
        ),
    }
    for elapsed, date in enumerate(dates):
        observations = subsequent.loc[subsequent.date == date]
        if elapsed:
            mean, covariance = advance_filter(
                mean, covariance, frozen, observations,
                _gap(dates[elapsed - 1], date), config.observation_noise_model,
            )
        prices = np.array([market.quote(date, c) for c in pair])
        spot = market.quote(date)
        value = valuer(
            contract, pricing_params, FactorState(*mean), gbm,
            elapsed_sessions=elapsed, current_spot=spot,
            current_futures=prices[0], numerical=config.numerical,
        )
        if not np.isfinite([
            value.price, value.continuation_value, value.exercise_value,
            value.slow_factor_sensitivity, value.fast_factor_sensitivity,
        ]).all():
            raise CohortUnavailable("Nonfinite option valuation")
        if elapsed == 0:
            premium = config.option_point_value * value.price
            metadata["initial_premium_points"] = value.price
            for spec, ledger in zip(specs, ledgers):
                ledger.cash = -spec.option_side * premium
        terminal = value.exercise_now or value.expired
        if terminal:
            signal_futures = np.zeros((len(specs), 2))
            signal_spot = np.zeros(len(specs))
            g = np.zeros((2, 2))
            diagnostic = None
        else:
            hedge_contracts = tuple(HedgeFuturesContract(
                code, prices[i], trading_days_between(date, contract_expiry(code))
            ) for i, code in enumerate(pair))
            all_futures, g, diagnostic = _short_targets(
                value, pricing_params, hedge_contracts, config
            )
            signal_futures = np.array([
                all_futures[spec.futures_count] for spec in specs
            ])
            signal_spot = np.array([
                _short_signal_targets_from_positions(
                    value, prices, spot, all_futures[spec.futures_count], spec, config
                )
                for spec in specs
            ])
        b = config.option_point_value * np.array([
            value.slow_factor_sensitivity, value.fast_factor_sensitivity,
        ])
        for i, (spec, ledger) in enumerate(zip(specs, ledgers)):
            if elapsed:
                financing, futures_pnl, spot_pnl = ledger.accrue(
                    previous_prices, prices, previous_spot, spot,
                    _gap(dates[elapsed - 1], date), config,
                )
            else:
                financing = futures_pnl = spot_pnl = 0.0
            held_futures = ledger.futures_positions.copy()
            held_spot = ledger.spot_units
            unsigned_payoff = float(value.exercise_value) if value.exercise_now else 0.0
            option_cash_flow = spec.option_side * config.option_point_value * unsigned_payoff
            ledger.cash += option_cash_flow
            cumulative_option_cash[i] += option_cash_flow
            target_futures = (
                np.zeros(2) if terminal else
                pending_futures[i].copy() if config.execution_lag_sessions else
                signal_futures[i].copy()
            )
            target_spot = (
                0.0 if terminal else
                pending_spot[i] if config.execution_lag_sessions else signal_spot[i]
            )
            (
                turnover, futures_cost, spot_turnover, spot_trade_cash_flow,
                spot_cost,
            ) = ledger.trade(target_futures, target_spot, spot, config)
            option_mark = 0.0 if terminal else spec.option_side * config.option_point_value * value.price
            equity = ledger.cash + ledger.spot_units * spot + option_mark
            discounted = equity * exp(-config.risk_free_rate * elapsed / 244)
            if terminal:
                residual = np.zeros(2)
                scale = 0.0
            else:
                residual = (
                    spec.option_side * b
                    + config.futures_multiplier * g @ ledger.futures_positions
                )
                scale = (
                    option_mark + ledger.spot_units * spot
                    + config.futures_multiplier * float(ledger.futures_positions @ prices)
                )
            rows.append({
                "study": config.study, "scenario": spec.scenario,
                "option_side": spec.option_side, "scenario_role": spec.role,
                "carry_hedge": spec.carry_hedge, "spot_hedge_enabled": spec.spot_enabled,
                "option_point_value": config.option_point_value,
                "futures_multiplier": config.futures_multiplier,
                "fee_per_contract": config.fee_per_contract,
                "slippage_points": config.slippage_points,
                "spot_cost_bps": config.spot_cost_bps,
                "execution_lag_sessions": config.execution_lag_sessions,
                "cohort": cohort["cohort"], "date": date,
                "elapsed_sessions": elapsed, "remaining_sessions": value.remaining_sessions,
                "hedge_contract_1": pair[0], "hedge_contract_2": pair[1],
                "spot": spot, "futures_1": prices[0], "futures_2": prices[1],
                "filtered_slow_state": mean[0], "filtered_fast_state": mean[1],
                "filtered_var_slow": covariance[0, 0], "filtered_var_fast": covariance[1, 1],
                "filtered_cov_slow_fast": covariance[0, 1],
                "accepted_contracts": len(observations), "prediction_only": observations.empty,
                "locked_carry": value.locked_carry,
                "positive_long_option_price": value.price,
                "option_price_points": value.price,
                "signed_option_mark": option_mark, "option_mark": option_mark,
                "continuation_points": value.continuation_value,
                "exercise_value_points": value.exercise_value,
                "unsigned_payoff": unsigned_payoff, "option_payoff": unsigned_payoff,
                "realized_payoff": unsigned_payoff,
                "option_cash_flow": option_cash_flow,
                "signed_option_cash_flow": option_cash_flow,
                "model_futures": value.model_futures,
                "futures_model_error": value.futures_model_error,
                "held_contracts_1": held_futures[0], "held_contracts_2": held_futures[1],
                "target_contracts_1": signal_futures[i, 0],
                "target_contracts_2": signal_futures[i, 1],
                "contracts_1": ledger.futures_positions[0],
                "contracts_2": ledger.futures_positions[1],
                "actual_contracts_1": ledger.futures_positions[0],
                "actual_contracts_2": ledger.futures_positions[1],
                "held_spot_units": held_spot, "target_spot_units": signal_spot[i],
                "spot_units": ledger.spot_units,
                "spot_market_value": ledger.spot_units * spot,
                "spot_notional": abs(ledger.spot_units * spot),
                "turnover_contracts": turnover,
                "spot_turnover_units": spot_turnover,
                "spot_turnover_notional": spot_turnover * spot,
                "spot_trade_cash_flow": spot_trade_cash_flow,
                "transaction_cost": futures_cost + spot_cost,
                "futures_transaction_cost": futures_cost,
                "spot_transaction_cost": spot_cost,
                "daily_futures_pnl": futures_pnl,
                "futures_pnl": futures_pnl,
                "daily_spot_pnl": spot_pnl,
                "spot_pnl": spot_pnl,
                "daily_financing": financing,
                "financing": financing,
                "dividend_cash_flow": 0.0,
                "cumulative_dividend_cash_flow": 0.0,
                "cumulative_futures_pnl": ledger.cumulative_futures_pnl,
                "cumulative_spot_pnl": ledger.cumulative_spot_pnl,
                "cumulative_financing": ledger.cumulative_financing,
                "cumulative_costs": ledger.cumulative_costs,
                "cumulative_futures_costs": ledger.cumulative_futures_costs,
                "cumulative_spot_costs": ledger.cumulative_spot_costs,
                "cumulative_option_cash_flow": cumulative_option_cash[i],
                "cash": ledger.cash, "equity": equity,
                "discounted_equity": discounted,
                "discounted_pnl_change": discounted - previous_discounted[i],
                "borrowing": max(-ledger.cash, 0.0),
                "residual_slow_exposure": residual[0],
                "residual_fast_exposure": residual[1],
                "residual_factor_variance": float(residual @ q @ residual),
                "factor_variance": float(residual @ q @ residual),
                "unhedged_factor_variance": 0.0 if terminal else float(b @ q @ b),
                "residual_scale_exposure": scale,
                "scale_residual": scale,
                "spot_delta_residual": scale / spot,
                "hedge_condition_number": np.nan if diagnostic is None else diagnostic.condition_number,
                "hedge_angular_separation": np.nan if diagnostic is None else diagnostic.angular_separation,
                "conditioning_warning": False if diagnostic is None else diagnostic.condition_number > config.condition_warning,
                "exercise_now": value.exercise_now, "expired": value.expired,
            })
            previous_discounted[i] = discounted
        pending_futures = signal_futures.copy() if not terminal else np.zeros_like(signal_futures)
        pending_spot = signal_spot.copy() if not terminal else np.zeros_like(signal_spot)
        previous_prices, previous_spot = prices, spot
        if terminal:
            metadata.update(
                status="completed", exit_date=date,
                exit_reason="exercise" if value.exercise_now else "expiry",
            )
            break
    if metadata.get("status") != "completed":
        raise RuntimeError("Cohort reached expiry without closing")
    daily = pd.DataFrame(rows)
    summaries = summarize_short_cohort(daily, metadata, config)
    optimizer_runs = estimate.optimizer_runs.copy()
    optimizer_runs.insert(0, "cohort", cohort["cohort"])
    return daily, summaries, metadata, optimizer_runs


def _short_signal_targets_from_positions(value, prices, spot, futures_position, spec, config):
    """Size funded spot from the already rounded actual futures target."""
    signed_option_mark = spec.option_side * config.option_point_value * value.price
    return -(
        signed_option_mark
        + config.futures_multiplier * float(futures_position @ prices)
    ) / spot if spec.spot_enabled else 0.0


def summarize_cohort(daily, metadata, config):
    rows = []
    for scenario, group in daily.groupby("scenario", sort=False):
        last = group.iloc[-1]
        # Entry costs count in total P&L, but aren't a close-to-close daily return.
        changes = group.loc[group.elapsed_sessions > 0, "discounted_pnl_change"]
        variance = float(changes.var(ddof=1)) if len(changes) >= 2 else np.nan
        equity = np.r_[0.0, group.equity.to_numpy()]
        premium = metadata["initial_premium_points"] * config.option_point_value
        payoff = float(group.option_payoff.sum())
        decomposition = payoff - premium + last.cumulative_futures_pnl + last.cumulative_financing - last.cumulative_costs
        if not np.isclose(decomposition, last.equity, rtol=1e-10, atol=1e-8):
            raise AssertionError("Cash ledger does not reconcile with P&L components")
        rows.append({
            "cohort": metadata["cohort"], "scenario": scenario,
            "entry_date": metadata["entry_date"], "exit_date": metadata["exit_date"],
            "exit_reason": metadata["exit_reason"], "initial_premium": premium,
            "option_payoff": payoff, "option_pnl": payoff - premium,
            "futures_pnl": last.cumulative_futures_pnl, "financing": last.cumulative_financing,
            "transaction_costs": last.cumulative_costs, "total_pnl": last.equity,
            "discounted_total_pnl": last.discounted_equity,
            "daily_discounted_pnl_variance": variance,
            "daily_discounted_pnl_std": np.sqrt(variance),
            "daily_volatility": np.sqrt(variance),
            "worst_daily_discounted_pnl": float(changes.min()) if len(changes) else np.nan,
            "max_drawdown": float(np.max(np.maximum.accumulate(equity) - equity)),
            "turnover_contracts": group.turnover_contracts.sum(),
            "peak_abs_contracts": float(group[["contracts_1", "contracts_2"]].abs().sum(axis=1).max()),
            "max_condition_number": group.hedge_condition_number.max(),
            "conditioning_warning_days": int(group.conditioning_warning.sum()),
            "mean_residual_factor_variance": group.loc[~(group.exercise_now | group.expired), "residual_factor_variance"].mean(),
            "rms_residual_scale_exposure": np.sqrt(np.mean(group.residual_scale_exposure**2)),
            "max_abs_futures_model_error": group.futures_model_error.abs().max(),
        })
    result = pd.DataFrame(rows)
    base = result.loc[result.scenario == "no_hedge", "daily_discounted_pnl_variance"].iloc[0]
    result["variance_reduction_vs_no_hedge"] = (
        1 - result.daily_discounted_pnl_variance / base if np.isfinite(base) and base > 0 else np.nan
    )
    return result


def summarize_short_cohort(daily, metadata, config):
    """Summarize signed option, funded spot, futures, and cash components."""
    rows = []
    for scenario, group in daily.groupby("scenario", sort=False):
        last = group.iloc[-1]
        changes = group.loc[group.elapsed_sessions > 0, "discounted_pnl_change"]
        variance = float(changes.var(ddof=1)) if len(changes) >= 2 else np.nan
        equity = np.r_[0.0, group.equity.to_numpy()]
        spec = next(spec for spec in SHORT_SCENARIOS if spec.scenario == scenario)
        premium = metadata["initial_premium_points"] * config.option_point_value
        payoff = float(group.unsigned_payoff.sum())
        option_pnl = spec.option_side * config.option_point_value * (
            payoff - metadata["initial_premium_points"]
        )
        decomposition = (
            option_pnl + last.cumulative_futures_pnl + last.cumulative_spot_pnl
            + last.cumulative_financing - last.cumulative_costs
        )
        if not np.isclose(decomposition, last.equity, rtol=1e-10, atol=1e-8):
            raise AssertionError("Funded cash ledger does not reconcile with P&L components")
        active = group.loc[~(group.exercise_now | group.expired)]
        rows.append({
            "study": metadata["study"], "cohort": metadata["cohort"], "scenario": scenario,
            "option_side": spec.option_side, "scenario_role": spec.role,
            "entry_date": metadata["entry_date"], "exit_date": metadata["exit_date"],
            "exit_reason": metadata["exit_reason"], "initial_premium": premium,
            "option_payoff": payoff,
            "option_cash_flow": float(group.option_cash_flow.sum()),
            "option_pnl": option_pnl,
            "futures_pnl": last.cumulative_futures_pnl,
            "spot_pnl": last.cumulative_spot_pnl,
            "financing": last.cumulative_financing,
            "futures_costs": last.cumulative_futures_costs,
            "spot_costs": last.cumulative_spot_costs,
            "transaction_costs": last.cumulative_costs,
            "total_pnl": last.equity,
            "discounted_total_pnl": last.discounted_equity,
            "daily_discounted_pnl_variance": variance,
            "daily_discounted_pnl_std": np.sqrt(variance),
            "worst_daily_discounted_pnl": float(changes.min()) if len(changes) else np.nan,
            "max_drawdown": float(np.max(np.maximum.accumulate(equity) - equity)),
            "turnover_contracts": group.turnover_contracts.sum(),
            "spot_turnover_units": group.spot_turnover_units.sum(),
            "spot_turnover_notional": group.spot_turnover_notional.sum(),
            "peak_abs_contracts": float(group[["contracts_1", "contracts_2"]].abs().sum(axis=1).max()),
            "peak_borrowing": float(group.borrowing.max()),
            "peak_spot_notional": float(group.spot_notional.max()),
            "max_condition_number": group.hedge_condition_number.max(),
            "conditioning_warning_days": int(group.conditioning_warning.sum()),
            "mean_residual_factor_variance": active.residual_factor_variance.mean(),
            "max_abs_residual_slow_exposure": active.residual_slow_exposure.abs().max() if len(active) else np.nan,
            "max_abs_residual_fast_exposure": active.residual_fast_exposure.abs().max() if len(active) else np.nan,
            "rms_residual_scale_exposure": np.sqrt(np.mean(active.residual_scale_exposure**2)) if len(active) else np.nan,
            "max_abs_residual_scale_exposure": active.residual_scale_exposure.abs().max() if len(active) else np.nan,
            "max_abs_spot_delta_residual": active.spot_delta_residual.abs().max() if len(active) else np.nan,
            "max_abs_futures_model_error": group.futures_model_error.abs().max(),
        })
    result = pd.DataFrame(rows)
    base = result.loc[result.scenario == "short_no_hedge", "daily_discounted_pnl_variance"].iloc[0]
    result["variance_reduction_vs_no_hedge"] = (
        1 - result.daily_discounted_pnl_variance / base
        if np.isfinite(base) and base > 0 else np.nan
    )
    return result


def run_backtest(market: MarketData, *, start: object, end: object,
                 config: BacktestConfig | None = None, max_cohorts: int | None = None,
                 estimator: Callable = estimate_two_factor_ou,
                 valuer: Callable = price_existing_carry_put) -> BacktestResult:
    """Run requested monthly entry dates. No full-history run is triggered implicitly.

    Known eligibility failures are reported in cohorts; unexpected errors raise.
    Failed cohorts never enter the configured scenario performance comparison.
    """
    config = config or BacktestConfig()
    if market.risk_free_rate != config.risk_free_rate:
        raise ValueError("Market carry rate and back-test financing/pricing rate differ")
    schedule = monthly_cohorts(start, end)
    if max_cohorts is not None:
        if isinstance(max_cohorts, bool) or int(max_cohorts) != max_cohorts or max_cohorts < 1:
            raise ValueError("max_cohorts must be a positive integer")
        schedule = schedule.head(max_cohorts)
    daily, summaries, cohorts, runs = [], [], [], []
    for cohort in schedule.to_dict("records"):
        try:
            d, s, m, r = run_cohort(market, cohort, config, estimator=estimator, valuer=valuer)
        except CohortUnavailable as exc:
            cohorts.append({**cohort, "status": "unavailable", "reason": str(exc)})
        else:
            daily.append(d)
            summaries.append(s)
            cohorts.append(m)
            runs.append(r)
    def concat(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return BacktestResult(concat(daily), concat(summaries), pd.DataFrame(cohorts), concat(runs), config)
