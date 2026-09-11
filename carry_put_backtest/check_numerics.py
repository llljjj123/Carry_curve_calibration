"""Selected historical state checks at a finer grid and quadrature order."""

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .historical_validation import DEFAULT_OUTPUT
from carry_put_pricing import (
    CarryPutContract, FactorState, GBMParams, HedgeFuturesContract, NumericalConfig,
    TwoFactorOUParams, calculate_one_futures_hedge, calculate_two_futures_hedge,
    price_existing_carry_put,
)
from calendar_utils import contract_expiry, trading_days_between


def check(output):
    daily = pd.read_csv(output / "baseline/daily_ledger.csv", parse_dates=["date"])
    cohorts = pd.read_csv(output / "baseline/cohort_audit.csv").set_index("cohort")
    active = daily.loc[(daily.scenario == "no_hedge") & ~(daily.exercise_now | daily.expired)]
    # Selection criteria are diagnostics, never trading or model-selection inputs.
    selected = {
        "last_inception": active.loc[active.elapsed_sessions == 0].iloc[-1],
        "largest_premium": active.loc[active.elapsed_sessions == 0].sort_values("option_price_points").iloc[-1],
        "largest_condition_nonzero_option": active.loc[active.option_price_points > .1].sort_values("hedge_condition_number").iloc[-1],
        "largest_fast_state": active.loc[active.filtered_fast_state.abs().idxmax()],
    }
    boundary = daily.loc[(daily.scenario == "no_hedge") & (daily.exercise_value_points > .05)
                         & (daily.continuation_points > .05)].copy()
    if not boundary.empty:
        boundary["gap"] = (boundary.exercise_value_points-boundary.continuation_points).abs()
        selected["nearest_exercise_boundary"] = boundary.sort_values("gap").iloc[0]
    fine = NumericalConfig(slow_grid_points=451, fast_grid_points=601, quadrature_order=61)
    results = []
    path = output / "selected_numerical_checks.csv"
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    for label, row in selected.items():
        if not previous.empty:
            hit = previous.loc[(previous.selection == label) & (previous.cohort == row.cohort)
                               & (previous.date == str(row.date.date()))]
            if not hit.empty:
                saved = hit.iloc[0].to_dict()
                saved["relative_price_difference"] = saved["price_difference"]/abs(saved["base_price"]) if saved["base_price"] else 0.
                results.append(saved)
                print(f"{label}: using existing numerical check", flush=True)
                continue
        metadata = cohorts.loc[row.cohort]
        original = daily.loc[(daily.cohort == row.cohort) & (daily.elapsed_sessions == 0)].iloc[0]
        params = TwoFactorOUParams(**{k: metadata[k] for k in ("kappa_slow", "kappa_fast", "theta", "eta_slow", "eta_fast")})
        contract = CarryPutContract(original.spot, original.futures_1, int(original.remaining_sessions))
        start = time.perf_counter()
        print(f"{label}: {row.cohort} on {row.date.date()}, {row.remaining_sessions} sessions left", flush=True)
        value = price_existing_carry_put(
            contract, params, FactorState(row.filtered_slow_state, row.filtered_fast_state), GBMParams(.014, 0.),
            elapsed_sessions=int(row.elapsed_sessions), current_spot=row.spot, current_futures=row.futures_1,
            numerical=fine,
        )
        pair = tuple(HedgeFuturesContract(code, price, trading_days_between(row.date, contract_expiry(code)))
                     for code, price in [(row.hedge_contract_1, row.futures_1), (row.hedge_contract_2, row.futures_2)])
        kwargs = dict(option_slow_factor_sensitivity=value.slow_factor_sensitivity,
                      option_fast_factor_sensitivity=value.fast_factor_sensitivity, ou_params=params)
        one = calculate_one_futures_hedge(**kwargs, hedge_future=pair[0])
        two = calculate_two_futures_hedge(**kwargs, hedge_futures=pair)
        same_date = daily.loc[(daily.cohort == row.cohort) & (daily.date == row.date)].set_index("scenario")
        # On a stopping date positions are closed, so compare stopping decisions
        # and values, not a new hedge that would never be held.
        comparable = not row.exercise_now and not value.exercise_now
        results.append(dict(
            selection=label, cohort=row.cohort, date=str(row.date.date()),
            base_price=row.option_price_points, fine_price=value.price,
            price_difference=value.price-row.option_price_points,
            relative_price_difference=(value.price-row.option_price_points)/abs(row.option_price_points) if row.option_price_points else 0.,
            base_continuation=row.continuation_points, fine_continuation=value.continuation_value,
            base_exercise=bool(row.exercise_now), fine_exercise=value.exercise_now,
            one_position_difference=one-same_date.loc["one_futures", "target_contracts_1"] if comparable else np.nan,
            two_position_1_difference=two.hedge_position_1-same_date.loc["two_futures", "target_contracts_1"] if comparable else np.nan,
            two_position_2_difference=two.hedge_position_2-same_date.loc["two_futures", "target_contracts_2"] if comparable else np.nan,
            runtime_seconds=time.perf_counter()-start,
            fine_slow_points=fine.slow_grid_points, fine_fast_points=fine.fast_grid_points,
            fine_quadrature=fine.quadrature_order,
        ))
        pd.DataFrame(results).to_csv(path, index=False)
        print(f"  price difference {results[-1]['price_difference']:.6g}; exercise unchanged={value.exercise_now == row.exercise_now}", flush=True)
    pd.DataFrame(results).to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    check(parser.parse_args().output_dir)
