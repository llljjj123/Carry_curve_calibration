"""Run the agreed latest-state IM2609 carry-put example."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "Demo"))

from carry_put_pricing import (  # noqa: E402
    CarryPutContract,
    FactorState,
    GBMParams,
    HedgeFuturesContract,
    NumericalConfig,
    TwoFactorOUParams,
    price_american_carry_put,
)
from calendar_utils import (  # noqa: E402
    calendar_source_for_date,
    contract_expiry,
    trading_days_between,
)


def _load_hedge_futures(
    curves: pd.DataFrame,
    valuation_date: object,
    contract_codes: tuple[str, str],
) -> tuple[
    tuple[HedgeFuturesContract, HedgeFuturesContract],
    list[dict[str, object]],
]:
    """Load observed closes and recompute maturities with the strict Demo calendar."""
    instruments = []
    diagnostics = []
    valuation_date = pd.Timestamp(valuation_date).normalize()
    for raw_code in contract_codes:
        code = str(raw_code).strip().upper()
        rows = curves.loc[
            (pd.to_datetime(curves["date"]) == valuation_date)
            & (curves["contract"].astype(str).str.upper() == code)
            & (~curves["excluded"].astype(bool))
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one accepted {code} hedge quote on "
                f"{valuation_date.date()}, found {len(rows)}"
            )
        quote = rows.iloc[0]
        expiry = contract_expiry(code)
        sessions = trading_days_between(valuation_date, expiry)
        instrument = HedgeFuturesContract(
            contract=code,
            futures_price=float(quote["futures_price"]),
            sessions_to_expiry=sessions,
            periods_per_year=244,
        )
        instruments.append(instrument)
        diagnostics.append(
            {
                "contract": code,
                "valuation_date": str(valuation_date.date()),
                "expiry": str(expiry),
                "sessions_to_expiry": sessions,
                "futures_price": instrument.futures_price,
                "calendar_source": calendar_source_for_date(expiry),
                "weekday_fallback_used": False,
            }
        )
    return (instruments[0], instruments[1]), diagnostics


def _load_agreed_inputs(
    source: Path,
    hedge_contract_codes: tuple[str, str],
) -> tuple[
    CarryPutContract,
    TwoFactorOUParams,
    FactorState,
    GBMParams,
    tuple[HedgeFuturesContract, HedgeFuturesContract],
    dict[str, object],
]:
    source = source.resolve()
    parameter_rows = pd.read_csv(source / "two_factor_parameters.csv")
    estimates = parameter_rows.set_index("parameter")["estimate"].astype(float).to_dict()
    state_row = pd.read_csv(source / "two_factor_filtered_states.csv").iloc[-1]
    curves = pd.read_csv(source / "two_factor_fitted_curves.csv")
    latest_date = curves["date"].max()
    latest = curves.loc[(curves["date"] == latest_date) & (~curves["excluded"].astype(bool))].copy()
    quote = latest.sort_values(["sessions_to_expiry", "volume"], ascending=[True, False]).iloc[0]

    contract = CarryPutContract(
        initial_spot=float(quote["spot"]),
        initial_futures=float(quote["futures_price"]),
        sessions_to_expiry=int(quote["sessions_to_expiry"]),
        periods_per_year=244,
    )
    params = TwoFactorOUParams(
        kappa_slow=estimates["kappa_slow"],
        kappa_fast=estimates["kappa_fast"],
        theta=estimates["theta"],
        eta_slow=estimates["eta_slow"],
        eta_fast=estimates["eta_fast"],
    )
    state = FactorState(
        slow=float(state_row["filtered_slow_state"]),
        fast=float(state_row["filtered_fast_state"]),
    )
    gbm = GBMParams(risk_free_rate=0.014, volatility=0.25)
    hedge_futures, hedge_diagnostics = _load_hedge_futures(
        curves,
        latest_date,
        hedge_contract_codes,
    )
    metadata = {
        "valuation_date": str(latest_date),
        "contract": str(quote["contract"]),
        "expiry": str(quote["expiry"]),
        "source_directory": str(source),
        "hedge_futures": hedge_diagnostics,
    }
    return contract, params, state, gbm, hedge_futures, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Price the carry put from a production calibration snapshot."
    )
    parser.add_argument(
        "--calibration-output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "im_2factor_ou_carry" / "outputs",
        help="Directory containing the production two-factor calibration CSV files.",
    )
    parser.add_argument(
        "--hedge-futures-contracts",
        nargs=2,
        default=("IM2609", "IM2703"),
        metavar=("FIRST", "SECOND"),
        help="Exactly two IM futures used for the joint slow/fast hedge.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for pricing results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract, params, state, gbm, hedge_futures, metadata = _load_agreed_inputs(
        args.calibration_output_dir,
        tuple(args.hedge_futures_contracts),
    )
    configurations = {
        "coarse": NumericalConfig(
            slow_grid_points=201,
            fast_grid_points=281,
        ),
        "base": NumericalConfig(),
        "fine": NumericalConfig(
            slow_grid_points=401,
            fast_grid_points=501,
        ),
    }
    results = {
        name: price_american_carry_put(
            contract,
            params,
            state,
            gbm,
            numerical=config,
            hedge_futures=hedge_futures if name == "base" else None,
        )
        for name, config in configurations.items()
    }
    base = results["base"]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence = pd.DataFrame(
        [
            {
                "configuration": name,
                **asdict(configurations[name]),
                "price": result.price,
                "normalized_price": result.normalized_price,
                "difference_from_fine": result.price - results["fine"].price,
            }
            for name, result in results.items()
        ]
    )
    convergence.to_csv(output_dir / "grid_convergence.csv", index=False)
    quadrature_results = {}
    for order in (39, 41, 43, 45, 47):
        if order == configurations["base"].quadrature_order:
            quadrature_results[order] = base
        else:
            config = NumericalConfig(
                slow_grid_points=configurations["base"].slow_grid_points,
                fast_grid_points=configurations["base"].fast_grid_points,
                stationary_stddev_width=configurations["base"].stationary_stddev_width,
                quadrature_order=order,
            )
            quadrature_results[order] = price_american_carry_put(
                contract, params, state, gbm, numerical=config
            )
    quadrature_convergence = pd.DataFrame(
        [
            {
                "quadrature_order": order,
                "price": result.price,
                "normalized_price": result.normalized_price,
                "difference_from_base_order": result.price - base.price,
            }
            for order, result in quadrature_results.items()
        ]
    )
    quadrature_convergence.to_csv(output_dir / "quadrature_convergence.csv", index=False)
    pd.DataFrame([asdict(row) for row in base.exercise_summary]).to_csv(
        output_dir / "exercise_summary.csv", index=False
    )
    delta_comparison = pd.DataFrame(
        [asdict(base.slow_curve_delta), asdict(base.fast_curve_delta)]
    )
    delta_comparison.to_csv(output_dir / "curve_delta_comparison.csv", index=False)
    if base.two_futures_hedge is None:  # pragma: no cover - base always requests it
        raise AssertionError("Base result is missing the requested two-futures hedge")
    joint_hedge = pd.DataFrame([asdict(base.two_futures_hedge)])
    joint_hedge.to_csv(output_dir / "two_futures_hedge.csv", index=False)
    payload = {
        "example": metadata,
        "contract": asdict(contract),
        "ou_parameters": asdict(params),
        "initial_state": asdict(state),
        "gbm_parameters": asdict(gbm),
        "risk_neutral_assumption": (
            "Historically calibrated OU dynamics are used as risk-neutral dynamics for this prototype."
        ),
        "base_numerical_configuration": asdict(configurations["base"]),
        "numerical_diagnostics": {
            "absolute_base_minus_fine_grid_price": abs(base.price - results["fine"].price),
            "quadrature_price_range_orders_39_to_47": (
                max(result.price for result in quadrature_results.values())
                - min(result.price for result in quadrature_results.values())
            ),
        },
        "result": base.as_dict(include_exercise_summary=False),
    }
    with (output_dir / "example_result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    print("\nGrid convergence")
    print(convergence.to_string(index=False))
    print("\nQuadrature convergence")
    print(quadrature_convergence.to_string(index=False))
    print("\nFutures-equivalent curve deltas")
    print(delta_comparison.to_string(index=False))
    print("\nFixed-carry scale delta (spot and model futures co-scaled)")
    print(base.fixed_carry_scale_delta)
    print("\nJoint two-futures slow/fast hedge")
    print(joint_hedge.to_string(index=False))


if __name__ == "__main__":
    main()
