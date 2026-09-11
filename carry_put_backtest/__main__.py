"""Explicit date-bounded CLI. Importing the package performs no calibration."""

import argparse
from pathlib import Path

from carry_put_pricing import NumericalConfig

from .engine import BacktestConfig, STUDIES, run_backtest, scenario_specs
from .market import load_cached_market
from .reporting import export_result


def main():
    parser = argparse.ArgumentParser(description="Monthly carry-put hedge back-test (explicit entry-date range)")
    parser.add_argument("--start", required=True, help="First eligible entry date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last eligible entry date; quotes must extend to option expiry")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--study", choices=STUDIES, default="long_futures_only")
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[1] / "im_2factor_ou_carry/data/raw")
    parser.add_argument("--window-dates", type=int, default=488)
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--risk-free-rate", type=float, default=0.014)
    parser.add_argument("--observation-noise-model", choices=["constant_log_futures", "constant_carry"], default="constant_log_futures")
    parser.add_argument("--optimizer-starts", type=int, default=12)
    parser.add_argument("--optimizer-maxiter", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=852)
    parser.add_argument("--kappa-gap-upper-bound", type=float, default=90.)
    parser.add_argument("--eta-fast-upper-bound", type=float, default=6.)
    parser.add_argument("--option-units", type=float, default=1.)
    parser.add_argument("--option-multiplier", type=float, default=1.)
    parser.add_argument("--futures-multiplier", type=float, default=1.)
    parser.add_argument("--fee-per-contract", type=float, default=0.)
    parser.add_argument("--slippage-points", type=float, default=0.)
    parser.add_argument("--spot-cost-bps", type=float, default=0.)
    parser.add_argument("--round-contracts", action="store_true")
    parser.add_argument("--execution-lag-sessions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--condition-warning", type=float, default=1000.)
    parser.add_argument("--slow-grid-points", type=int, default=301)
    parser.add_argument("--fast-grid-points", type=int, default=401)
    parser.add_argument("--quadrature-order", type=int, default=43)
    args = parser.parse_args()
    values = vars(args).copy()
    numerical = NumericalConfig(**{k: values.pop(k) for k in (
        "slow_grid_points", "fast_grid_points", "quadrature_order",
    )})
    for key in ("start", "end", "output_dir", "raw_dir", "max_cohorts"):
        values.pop(key)
    config = BacktestConfig(**values, numerical=numerical)
    market = load_cached_market(args.raw_dir, risk_free_rate=config.risk_free_rate)
    result = run_backtest(market, start=args.start, end=args.end, config=config,
                          max_cohorts=args.max_cohorts)
    export_result(result, args.output_dir, request=vars(args), input_paths=[
        args.raw_dir / "spot_raw.csv", args.raw_dir / "futures_raw.csv",
    ])
    scenario_count = len(scenario_specs(config.study))
    print(f"Completed {len(result.summaries) // scenario_count}/{len(result.cohorts)} scheduled cohorts; outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
