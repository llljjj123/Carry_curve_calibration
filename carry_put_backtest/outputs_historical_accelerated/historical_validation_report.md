# Historical carry-put hedge validation

The monthly back-test is complete. The proposed futures hedges remove local carry-factor risk, but **do not provide sufficient total hedging** for the long optional component: they introduce large spot exposure and substantially increase total P&L volatility.

## Sample and conventions

The cache ends on 2026-08-21. Of 48 scheduled monthly entries, 24 lacked the required 488-date history. All remaining **24 cohorts**, entered from 2024-08-19 through 2026-07-20, had complete required data and completed successfully. The last actual exit was 2026-08-10. There were 22 early exercises and two expiries (the August 2024 and May 2025 entries), with 245 close-to-close P&L observations per scenario.

Each cohort starts only on the first trading session after monthly expiry, references the next monthly IM contract, calibrates using 488 accepted curve dates through inception, freezes parameters, and updates states with forward filtering. The two-futures pair is the option underlying plus the longest-dated later IM contract quoted at inception. No early-exercise replacement cohort is created.

Results below use one long optional component, continuous hedge units, multipliers of 1, zero transaction costs, same-close execution, and 1.4% financing on the trading-session/244 clock. The separate linear futures leg is excluded. P&L is in option points; sums are across separate cohorts, not a funded strategy NAV.

## Baseline results

| Scenario | Sum P&L | Mean cohort P&L | Cohort P&L std | Daily discounted P&L std | Worst cohort P&L |
|---|---:|---:|---:|---:|---:|
| No hedge | 80.08 | 3.34 | 19.50 | 7.36 | -31.19 |
| One futures | 1285.77 | 53.57 | 96.42 | 29.31 | -193.73 |
| Two futures | 1263.26 | 52.64 | 94.91 | 28.85 | -190.21 |

The one- and two-futures portfolios have roughly 15.86 and 15.36 times the unhedged daily variance, respectively (about four times the standard deviation). Two futures modestly reduce risk relative to one futures, but neither reduces total risk relative to the unhedged option.

![Monthly and cumulative cohort P&L](D:/LuJingjian/Jupyter_files/GuoYuan/Studies/carry_rate/carry_put_backtest/outputs_historical_accelerated/figures/cohort_pnl.png)

## Why factor neutrality does not provide total hedging

The option is homogeneous: `V = S * v(x_s, x_f)`. Its value still responds to spot scaling even though spot volatility cancels from the pricing equation. The long futures positions needed to neutralize carry sensitivity add substantial spot sensitivity.

| Scenario | Reduction in modeled carry-factor variance | RMS spot-scale sensitivity | Approx. first-order P&L for a 1% scale move |
|---|---:|---:|---:|
| No hedge | 0.0000% | 19.74 | 0.20 |
| One futures | 99.9773% | 2015.83 | 20.16 |
| Two futures | 100.0000% | 1984.49 | 19.84 |

The two-futures residual factor sensitivities are below 5.7e-14 in absolute value. Yet its spot-scale exposure is approximately 100 times that of the unhedged option. A first-order diagnostic using preceding holdings shows about 0.991 correlation between hedged daily P&L before financing/costs and the predicted spot-scale component. The residual includes time decay, basis, nonlinearity, and model error; this is an attribution diagnostic, not a separate identification of causal returns.

Higher realized hedge P&L therefore must not be interpreted as better hedge sufficiency or as evidence of carry-option mispricing. Most of the P&L difference comes from the futures positions.

![Hedge sufficiency](D:/LuJingjian/Jupyter_files/GuoYuan/Studies/carry_rate/carry_put_backtest/outputs_historical_accelerated/figures/hedge_sufficiency.png)

## P&L reconciliation

| Scenario | Option payoff less premium | Futures P&L | Financing | Total P&L |
|---|---:|---:|---:|---:|
| No hedge | 80.451316 | 0.000000 | -0.368809 | 80.082507 |
| One futures | 80.451316 | 1205.274702 | 0.047209 | 1285.773227 |
| Two futures | 80.451316 | 1182.768162 | 0.039706 | 1263.259184 |

An independent cash replay reproduces daily equity to less than 2e-13 points. The checks verify prior-position futures P&L, financing, terminal unwind, common exit dates, fixed carry locks, 488-date calibration windows, and complete coverage. All terminal positions and option marks are zero.

## Execution sensitivities

The same option policy and model signals are replayed without refitting or repricing. The one-session lag delays hedge execution only; exercise and forced closing remain immediate. The 0.2-point one-way cost is an illustrative sensitivity, not an observed brokerage fee.

| Variant | Scenario | Sum P&L | Mean cohort P&L | Daily discounted P&L std |
|---|---|---:|---:|---:|
| One-session hedge lag | No hedge | 80.08 | 3.34 | 7.36 |
| One-session hedge lag | One futures | 717.56 | 29.90 | 28.34 |
| One-session hedge lag | Two futures | 702.98 | 29.29 | 27.88 |
| 0.2-point one-way cost | No hedge | 80.08 | 3.34 | 7.36 |
| 0.2-point one-way cost | One futures | 1280.35 | 53.35 | 29.31 |
| 0.2-point one-way cost | Two futures | 1257.85 | 52.41 | 28.85 |

Delaying hedge execution reduces realized hedge P&L materially, while total volatility remains much higher than for the unhedged option. The illustrative cost level does not change the hedge-sufficiency conclusion.

## Calibration and numerical checks

All 288 optimizer starts across 24 cohorts converged. No cohort reached the configured kappa-gap or eta-fast upper bound. The maximum hedge-matrix condition number was 119.94; no singular pair or conditioning warning at the configured 1,000 threshold occurred.

Historical calibration uses an explicitly selected optional Numba likelihood backend, with Numba already installed in spyder-env. Independent observation errors permit sequential scalar Kalman conditioning, which is algebraically equivalent to the dense reference likelihood. Forty historical parameter-set comparisons and synthetic tests verified likelihood agreement; finite-difference gradient discrepancies were below 1e-6. The original NumPy backend remains the estimator default. The full pilot fit changed log likelihood by 2.3e-9 and cohort P&L by less than 0.000031 points compared with the original backend. Pilot calibration runtime fell from 105.8 to 1.7 seconds.

The baseline uses a 301 × 401 factor grid and 43-point quadrature. The finer checks use 451 × 601 and 61-point quadrature. The first cohort was rerun for its entire life:

| Scenario | Fine minus baseline premium | Fine minus baseline lifetime P&L | Fine minus baseline daily P&L std |
|---|---:|---:|---:|
| No hedge | -0.006808 | 0.006816 | -0.001431 |
| One futures | -0.006808 | 0.682540 | -0.057432 |
| Two futures | -0.006808 | 0.692541 | -0.057772 |

The pilot exit date is unchanged. Selected additional historical states check the final inception, largest premium, largest conditioning value with a nonzero option mark, largest absolute fast factor, and nearest exercise boundary:

| Selection | Cohort / valuation date | Fine minus baseline price | Relative difference | Exercise decision unchanged |
|---|---|---:|---:|---|
| last_inception | 2026-07-20 / 2026-07-20 | -0.010542 | -0.030% | True |
| largest_premium | 2025-08-18 / 2025-08-18 | -0.016579 | -0.042% | True |
| largest_condition_nonzero_option | 2024-08-19 / 2024-09-18 | -0.002054 | -0.294% | True |
| largest_fast_state | 2024-09-23 / 2024-09-27 | -0.000016 | -0.456% | True |
| nearest_exercise_boundary | 2025-07-21 / 2025-08-01 | -0.002904 | -0.023% | True |

Across the selected states, option-price changes are below 0.017 points and exercise decisions are unchanged. The largest sampled hedge-position change is about 0.0134 futures units near the exercise boundary. These checks support the broad risk conclusion, but do not establish exact convergence of every hedge ratio or every cohort's P&L. The pilot's roughly 0.69-point hedged P&L change shows that reporting many decimal places would overstate numerical precision. A full finer-grid batch was not run. The regression suite passed **71 tests** in spyder-env.

## Interpretation and next decision

The agreed factor-only futures hedges are effective at the factor objective and insufficient at the total-risk objective. A next research step would be to either include spot risk in a one-/two-futures minimum-total-variance hedge, or introduce an additional spot/ETF hedge alongside the two factor-neutral futures. Neither alternative was substituted into this experiment.

Daily option marks and inception premiums are model values, and historical OU dynamics are provisionally treated as risk-neutral. Observed immediate exercise values are compared with model continuation. Actual fills, settlement cash flows, margin funding, asymmetric borrowing rates, liquidity constraints, and option fees are not modeled. The strict Demo calendar's 2027 maturities use its provisional extension. Twenty-four cohorts are too few for strong tail-risk or expected-return claims. Future exercises are not used to set trading decisions; only scheduled-life coverage is used as a sample eligibility requirement.

## Files

- [Monthly P&L](baseline/cohort_pnl.csv)
- [Daily ledger](baseline/daily_ledger.csv)
- [Cohort calibration and exit audit](baseline/cohort_audit.csv)
- [Data coverage](coverage.csv)
- [Execution sensitivities](execution_sensitivities.csv)
- [Numerical checks](selected_numerical_checks.csv)
- [Independent validation checks](validation_checks.json)
- [Source/input manifest](baseline/run_manifest.json)
