# Carry-put monthly hedge back-test

Implementation and historical validation are complete (2026-09-10). All 24
eligible monthly cohorts ran successfully. The factor hedges neutralized carry
risk but increased total P&L volatility through their spot exposure.

See [the historical validation report](outputs_historical_accelerated/historical_validation_report.md)
for results, execution sensitivities, numerical checks, and limitations.

## Agreed experiment

Exactly one cohort starts on the first exchange trading session after each
monthly IM expiry. No entries occur between those dates, and early exercise
does not start a replacement cohort. The option references the next monthly IM
contract. Inception carry and expiry stay fixed throughout its life.

Each cohort estimates the independent two-factor OU model on the latest **488
accepted curve dates through inception**, including the inception close, and
freezes all parameters. The default is `constant_log_futures` observation noise,
a kappa-gap upper bound of 90, and an eta-fast upper bound of 6. Estimation uses
12 starts, at most 1,500 iterations, and seed 852; Hessian/profile diagnostics
are intentionally omitted. An unconverged estimate is not used for trading.

Factor states update once daily using the exact OU transition and the existing
forward Kalman filter, with no smoothing. Dates with no accepted curve use a
prediction-only state update; raw hedge/spot closes must still exist. The strict
Demo calendar is used for scheduling, maturities, filtering, and financing
(`sessions / 244`); its 2027–2028 extension is provisional.

The three scenarios share the same long optional component, model inception
premium, and exercise/exit date:

| Scenario | Position rule |
|---|---|
| `no_hedge` | No futures |
| `one_futures` | Option's underlying futures, with the local carry-variance-minimizing position |
| `two_futures` | Underlying plus longest-dated later IM maturity quoted at inception; jointly neutralize slow and fast sensitivities |

Contract identities remain fixed; neither rolling nor retrospective replacement
of an unavailable second contract is used. Missing required closes anywhere in
the scheduled option life make the entire cohort unavailable, even if its
eventual exercise policy might have stopped sooner. This ensures a common
comparison sample. Unavailable cohorts and their reasons remain in the audit.

## Hedge mathematics and units

For option point sensitivities `b = (V_s, V_f)` and futures factor loadings
`g_i = -F_i (A_s(tau_i), A_f(tau_i))`, where
`A_j(tau) = (1 - exp(-kappa_j*tau)) / kappa_j`:

- One futures: `n = -(g' Q b) / (g' Q g)`.
- Two futures: solve `G n = -b` with `G = [g_1, g_2]`.
- `Q_jj = eta_j^2 * (1 - exp(-2*kappa_j/244)) / (2*kappa_j)`.

These are hedges for a **long** option. They hedge local carry-factor risk, not
total spot/curve/basis risk. The monetary residual scale sensitivity is
`option_mark + sum(contracts_i * futures_multiplier * F_i)` and is reported
separately. Small carry-factor residuals do not imply sufficient total hedging.

Default multipliers and option units are all 1, so P&L is in points and hedge
positions are continuous units. Monetary contract counts are
`n * option_units * option_multiplier / futures_multiplier`. For example, use
an IM futures multiplier of 200 and the option's actual contractual point value
for monetary results. The option multiplier is a user input, not an assumed
exchange contract specification. Optional rounding uses NumPy nearest-integer
rounding (ties to even); residual exposures are recomputed after rounding.

Singular hedge matrices make the cohort unavailable, never a zero hedge.
Near-singular matrices are retained with condition numbers, angular separation,
and warning flags (default threshold 1,000). There is no position cap in this
unconstrained research benchmark; reported peak positions and turnover expose
unstable pairs.

## Valuation and execution

`price_existing_carry_put` takes the original inception contract, elapsed
sessions, current quotes and current state. It preserves locked carry, values
future exercise using the existing model backward induction, and compares the
resulting continuation with **observed** immediate payoff:

`max(S_t * exp((r-q0)*remaining_sessions/244) - F_t, 0)`.

Exercise occurs when this payoff exceeds continuation by the numerical
tolerance times spot. Inception exercise is zero. Expiry payoff is zero under
contractual convergence; an observed expiry close basis is not an additional
option payoff. Hedge closure always uses the observed futures close. The model
versus observed futures discrepancy is retained in daily output.

The default assumes a signal calculated at close t can trade at that close;
only holdings already established earn the next close-to-close move. This is
an idealized research assumption. `--execution-lag-sessions 1` defers each hedge
target to the next session's close, with no futures position before the first
delayed trade. This sensitivity delays **hedge rebalancing only**: exercise and
forced unwind still happen at the exercise/expiry close, cancelling pending
targets. It is not a complete intraday execution model.

Daily option marks are model values, not observed option market prices. The
historically estimated OU parameters are provisionally treated as risk-neutral,
as in the existing pricing project. Calibration's observation equation and the
pricer's exact convexity-adjusted forward also differ; model errors are not
silently corrected by relocking the contract.

## Cash ledger

At inception, cash is minus option premium and entry hedge costs. Futures have
no purchase-price debit. Each later close:

1. Accrue interest on the preceding cash balance at the configured rate.
2. Credit prior holdings times observed futures close changes and multiplier.
3. If exercised, credit option payoff and remove the option mark.
4. Trade to the current (or delayed) target, or unwind fully at exit.
5. Charge `abs(change in contracts) * (fee_per_contract + slippage_points * futures_multiplier)`.

There is no separate option fee, margin model, asymmetric borrowing rate, or
liquidity model. The source's zero settlement fields are never used. Costs at
closing and all cash financing are included. Every completed cohort must
reconcile:

`total_pnl = payoff - initial_premium + futures_pnl + financing - costs`.

Discounted daily P&L is the change in discounted cash-plus-option equity.
Entry costs are included in lifetime P&L but excluded from the close-to-close
variance statistic. Undefined variance reductions (zero baseline variance or
too few observations) are missing values, not 100% improvements.

## Entry points

From the repository root, inspect the CLI without starting a back-test:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -m carry_put_backtest --help
```

An explicit date-bounded run can be requested as follows:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -m carry_put_backtest `
  --start 2025-01-20 --end 2025-01-20 --max-cohorts 1 `
  --output-dir carry_put_backtest/outputs_single_cohort
```

`--start` and `--end` bound **entry dates**, not liquidation dates. The cache
must extend through each selected option's expiry. `--max-cohorts` limits
scheduled attempts, including unavailable ones. No full-history run starts
on import, without arguments, or during the focused tests.

Python API:

```python
from carry_put_backtest import BacktestConfig, load_cached_market, run_backtest
from carry_put_backtest.reporting import export_result

config = BacktestConfig()
market = load_cached_market(risk_free_rate=config.risk_free_rate)
# Example of an explicit single-cohort request:
# result = run_backtest(market, start="2025-01-20", end="2025-01-20", config=config)
# export_result(result, "carry_put_backtest/outputs_single_cohort")
```

## Outputs

- `daily_ledger.csv`: states, quotes, continuation/payoff, held/target/actual
  positions, cash, costs, financing, P&L, residual exposures, and conditioning.
- `cohort_pnl.csv`: the three scenarios' reconciled lifetime P&L, daily risk,
  drawdown, turnover, position size, and variance reduction for each cohort.
- `cohort_audit.csv`: scheduled entries, unavailable reasons, frozen parameters,
  calibration window, selected pair, parameter-bound flags, and common exit.
- `optimizer_runs.csv`: calibration optimizer diagnostics for completed cohorts.
- `aggregate.csv`: descriptive cohort P&L and pooled daily-risk comparisons.
- `run_manifest.json`: configuration, date request, runtime versions, source and
  raw-input hashes (raw hashes supplied automatically by CLI), and assumptions.
- `report.md`: compact results table with limitations.

The sum of cohort P&L is the sum of independent trades, not a continuously
funded strategy NAV; cash is not rolled between cohorts. Tail statistics are
descriptive and do not imply independent observations or reliable tail estimates
from a small number of cohorts. Historical data coverage, calibration, cash
reconciliation, and hedge performance have been checked. Numerical validation
used a full finer-grid pilot plus selected historical states; it does not assert
exact convergence of every cohort's P&L.

## Short carry-put with funded spot

The explicit `short_with_spot` study preserves the legacy study and compares
these five scenarios: `short_no_hedge`, one- and two-futures hedges with funded
spot, and matching one- and two-futures-only controls. The option side is `-1`.
The pricer still returns positive long-option values and factor sensitivities;
the short conversion is applied once in the hedge target construction.

Spot is a continuously sized hypothetical index instrument with zero dividend
cash flow. It is purchased from the cash account and its daily price change is
recorded for attribution through the funded asset. Carry, theta, locked
inception carry, calibration, and pricing are not changed. Futures have no
principal cash flow. One-session lag delays futures and spot targets together.

The funded ledger reports signed option marks and exercise cash flows, futures
and spot P&L, financing, separate costs, cash/equity, borrowing, spot notional,
turnover, and actual factor/scale/spot-delta residuals. The independent replay
must reconcile each completed cohort's lifetime identity. The original
`outputs_short_spot/` tree is preserved as immutable historical evidence and
its exact execution-source provenance is unresolved. Corrected current-source
results are in `outputs_short_spot_review_fixed/`, with the readable report at
`outputs_short_spot_review_fixed/short_spot_results_report.md` and reviewer
bundle at `short_spot_bugfix_handoff.md`.

Reproduce the short-study historical baseline and independent checks with:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -u -m carry_put_backtest.historical_validation `
  --phase baseline --study short_with_spot --output-dir carry_put_backtest/outputs_short_spot_review_fixed
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -m carry_put_backtest.historical_validation `
  --phase fine --study short_with_spot --output-dir carry_put_backtest/outputs_short_spot_review_fixed --dates 2025-07-21
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -m carry_put_backtest.test_evidence `
  --output-dir carry_put_backtest/outputs_short_spot_review_fixed
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -m carry_put_backtest.analyze_short_spot `
  --output-dir carry_put_backtest/outputs_short_spot_review_fixed --mode full
```

The new historical baseline completed 24 eligible cohorts (24 earlier entries
were excluded for the required 488-date warm-up). Sum P&L was approximately
`-80.083` for the short unhedged baseline, `-276.618` and `-272.485` for the
one-/two-futures-plus-spot primaries, and `-1285.773` and `-1263.259` for the
futures-only controls. Daily discounted P&L standard deviation was approximately
`7.361`, `3.912`, `3.890`, `29.312`, and `28.852`, respectively. The spot
primaries neutralized scale exposure to floating-point tolerance; the two-futures
primary also neutralized both modeled carry factors. The one-session lag and
illustrative 0.2-point futures / 1 bp spot cost sensitivities are in
`execution_sensitivities.csv`. The fine pilot used 451x601 grids and 61-point
quadrature for the 2025-07-21 cohort; it is not a full fine-grid batch.

Execution manifests are immutable. Analysis writes a separate
`analysis_manifest.json`, regenerates and key-validates lag/cost replays, and
requires exact eligible/completed cohort equality in full mode. Subsets require
explicit pilot mode and cohort IDs. Aggregate peak borrowing and spot notional
are maximum individual-cohort full-path values, not sums of cohort peaks or
portfolio requirements. The headline risk metric is pooled discounted daily
P&L standard deviation (`elapsed_sessions > 0`, `ddof=1`).

## Focused tests

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -m pytest `
  carry_put_backtest/tests carry_put_pricing/tests Demo/tests -q -p no:cacheprovider
```

The back-test tests use synthetic prices and a fixed estimator fixture, with a
real-pricer integration test. They cover schedule, no look-ahead, preserved carry
lock, early exercise and expiry, exact incremental filtering, hedge algebra,
cash reconciliation, trade lag, multipliers/rounding, costs, missing quotes,
unconverged calibration, and exports. The existing Demo tests read cached data
but do not run historical cohort calibration. No new packages are required.

The legacy implementation checkpoint recorded **71 passed** across the
back-test, pricing, Demo, and two-factor calibration suites in `spyder-env`.
After the post-review fixes, source-bound evidence records **93 passed tests in
313.23 seconds**, exit code zero. Added regressions cover immutable manifests,
stale/failing test evidence, sensitivity completeness, strict full versus pilot
sets, peak aggregation, pooled risk, residual units, unequal multipliers, and
rounded-futures/continuous-spot sizing.

## Reproducing historical validation

The optional Numba likelihood backend uses an already-installed package and is
explicitly selected only by the historical validation runner. The estimator's
default remains the original NumPy backend. Likelihood/gradient comparisons and
an entire reference pilot fit verified agreement before the accelerated batch.

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -u -m carry_put_backtest.historical_validation `
  --phase baseline --output-dir carry_put_backtest/outputs_historical_accelerated
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -m carry_put_backtest.analyze_historical `
  --output-dir carry_put_backtest/outputs_historical_accelerated
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -u -m carry_put_backtest.check_numerics `
  --output-dir carry_put_backtest/outputs_historical_accelerated
& 'D:\miniforge3\envs\spyder-env\python.exe' -B -m carry_put_backtest.analyze_historical `
  --output-dir carry_put_backtest/outputs_historical_accelerated --report-only
```

The baseline runner reuses checkpoints only when their inputs, core code,
configuration, and runtime signature match. Calibration caches separately key
the actual input sample, estimator settings, and source/runtime versions. The
first cohort's full fine run is in `outputs_historical_accelerated/fine`; its
unaccelerated reference is retained in `outputs_historical/baseline`.
