# Short spot back-test review handoff

This bundle is ready for independent review. Astra review has not been performed.

## Changed files

- `carry_put_backtest/engine.py`: explicit study/scenario definitions, short-side hedge conversion, funded spot ledger, residuals, costs, lag, and summaries.
- `carry_put_backtest/__main__.py` and `__init__.py`: study and spot-cost configuration/API exposure.
- `carry_put_backtest/reporting.py`: signed-study exports and scenario-count-aware reports.
- `carry_put_backtest/analyze_historical.py`: independent funded replay.
- `carry_put_backtest/analyze_short_spot.py`: independent historical checks, charts, report, and this handoff.
- `carry_put_backtest/historical_validation.py`: short-study selection and sensitivity configuration.
- `carry_put_backtest/tests/test_short_spot.py`: synthetic economic-identity tests.

## Commands and results

Interpreter: `D:\miniforge3\envs\spyder-env\python.exe`
Focused/regression command: `D:\miniforge3\envs\spyder-env\python.exe -B -m pytest carry_put_backtest/tests carry_put_pricing/tests Demo/tests im_2factor_ou_carry/tests -q -p no:cacheprovider`
Historical baseline: `python -B -u -m carry_put_backtest.historical_validation --phase baseline --study short_with_spot --output-dir carry_put_backtest/outputs_short_spot`.
Historical analyzer: `python -B -m carry_put_backtest.analyze_short_spot --output-dir carry_put_backtest/outputs_short_spot`.

Independent replay maximum errors: `{"cash": 1.3642420526593924e-12, "contracts_1": 0.0, "contracts_2": 0.0, "discounted_equity": 9.103828801926284e-13, "discounted_pnl_change": 9.059419880941277e-13, "equity": 9.112710586123285e-13, "spot_units": 0.0}`.
Cohorts: `24` completed of `48` scheduled; exercises `22`, expiries `2`.
Baseline short control sums: `short_no_hedge=-80.082507, short_one_futures_only=-1285.773227, short_two_futures_only=-1263.259184`; compare futures-only controls with the negative legacy long study.

## Sign and unit conventions

The pricer returns positive long-option values and sensitivities. The study uses side `s=-1`; short-option premium receipt is positive, exercise payment is negative, and the live option mark is negative. Futures counts are positive long/negative short. Spot units are continuous hypothetical index units. Point P&L is monetary only when configured multipliers and units are one.

Spot is sized after actual futures rounding. It is funded from cash, and daily spot P&L is attributed through the funded asset rather than credited to cash separately. Dividend cash flow is exactly zero.

## Review points

Check the three primary scenarios against their two futures-only controls, especially residual factor/scale exposure, borrowing, spot notional, and total P&L. Verify the short unhedged and futures-only controls against the negative legacy long study. Inspect singular/near-singular handling and the fine pilot before drawing numerical conclusions.

The market convention combines zero-dividend spot, common financing, and observed nonzero futures carry; it need not be arbitrage-consistent. The historical OU fit remains a provisional risk-neutral input. No strategy optimization, spot-only hedge, ETF, or minimum-total-variance objective was introduced.

## Artifacts

- [Results report](outputs_short_spot/short_spot_results_report.md)
- [Baseline ledger](outputs_short_spot/baseline/daily_ledger.csv)
- [Baseline cohort P&L](outputs_short_spot/baseline/cohort_pnl.csv)
- [Validation checks](outputs_short_spot/validation_checks.json)
- [Execution sensitivities](outputs_short_spot/execution_sensitivities.csv)
- [Fine-grid pilot](outputs_short_spot/fine/cohort_pnl.csv)
- [Test results](outputs_short_spot/test_results.json)

The planned fine-grid pilot was completed for the 2025-07-21 cohort. A full finer-grid historical batch was not run and is not claimed.
