# Carry Curve Calibration — AI Handoff Summary

## Purpose and scope

This repository studies the implied-carry term structure of CSI 1000 index futures (CFFEX `IM` contracts), develops one- and two-factor Ornstein–Uhlenbeck (OU) state-space models for that curve, tests whether stock/carry shock correlation is identifiable, and prices an American-style put on the carry curve.

This handoff was prepared on 2026-08-26 from `session_log.md` and the executable code, tests, configurations, and current generated summaries in the five implementation folders. It was updated on 2026-09-01 after the delta-definition review, pricing-library change, Demo notebook rerun, and user-requested root-README documentation work.

## 2026-09-01 delta update

- The proposed quantity `C_t/F_t,T` was checked carefully. It is not the ordinary partial derivative with respect to futures while spot is fixed; that immediate-exercise derivative is `-1` in the in-the-money region. Instead, `C/F` is the proportional scale sensitivity when spot and futures are multiplied by the same factor and current implied carry is fixed. By homogeneity, the corresponding full-option sensitivity is `V/F`.
- `carry_put_pricing` now exposes `PricingResult.fixed_carry_scale_delta = V/F_model`. The model-implied futures price is used so the denominator is consistent with the slow/fast curve-delta conversions. This scale delta fixes the futures/spot ratio, locked carry, and both OU states; it is distinct from a carry-curve hedge ratio.
- The fixed `IM2609` example has price `36.27943692`, model futures `7527.39615356`, and fixed-carry scale delta `0.00481965293`. The pricing suite still has 10 passing tests, including analytical `V/F`, proportional co-scaling, and finite-difference checks; Ruff passed and the standard example was regenerated.
- `Demo/Carry_Put_Demo.ipynb` now contains an executed futures-equivalent curve-delta subsection immediately below option pricing. It reports slow and fast pathwise/differentiated-backward-induction deltas, local bump-and-value checks, factor sensitivities, futures sensitivities, and the fact that spot is held fixed. The fixed-carry scale delta is intentionally not displayed because the Demo's current focus is changes in carry `q`.
- The current 2026-08-10 / 488-date / `IM2612` Demo rerun gives slow deltas `-0.33235550` pathwise and `-0.33238170` bump-and-value, and fast deltas `-0.01406941` pathwise and `-0.01404718` bump-and-value. The notebook executed end to end and regenerated the Demo CSV/JSON/PNG snapshot.
- The user added a `## Delta部分` section to the root `README.md`. At the user's explicit request, a `### 数值计算方法` subsection was appended without changing the user's prior writing. It documents the derivative conventions, exercise/continuation derivative handling, continuation-side tie convention, exact-model futures conversion, local one-grid-step bump, and directional hedge interpretation.

The five implementation folders are:

1. `im_ou_carry`: baseline one-factor OU calibration and diagnostics.
2. `im_2factor_ou_carry`: preferred two-factor OU calibration, including a like-for-like one-factor comparison.
3. `im_corr_ou_1factor`: exact correlated one-factor extension with curve-only and joint curve/return likelihoods.
4. `carry_put_pricing`: isolated numerical library and fixed example for the American carry-put optional component.
5. `Demo`: configurable end-to-end calibration and carry-put demonstration that reuses the two-factor and pricing engines.

Do not modify or revert the root `README.md` unless the user explicitly asks. It is user-owned writing, including the `## Delta部分` section. Also assume the worktree may already contain user changes, especially in `AGENTS.md`, `README.md`, `Demo/Carry_Put_Demo.ipynb`, and `Demo/outputs`; inspect `git status` before editing anything.

## Repository dependency map

```text
cached CSI 1000 spot + IM futures closes
                  |
                  +--> im_ou_carry (baseline one-factor model)
                  |
                  +--> im_2factor_ou_carry (main slow/fast model)
                  |          |
                  |          +--> carry_put_pricing (reads saved two-factor results)
                  |          |
                  |          +--> Demo/calibration.py (imports the two-factor engine)
                  |
                  +--> im_corr_ou_1factor (isolated correlation experiment)

carry_put_pricing/src
          |
          +--> Demo/option_pricing.py (imports the pricing engine)
```

The three calibration folders intentionally duplicate some data, calendar, quality, one-factor, diagnostic, and plotting code so experiments remain isolated. The Demo avoids duplicating the core two-factor Kalman filter/estimator and carry-put pricer: it adds sibling `src` directories to `sys.path` and imports them.

## Agreed market-data and modelling conventions

These conventions override earlier calendar-day or settlement-price specifications:

- Spot is the CSI 1000 daily close from AkShare symbol `sh000852`.
- Futures are individual `IMYYMM` daily **close** prices, not settlement. In the cache, settlement is zero throughout and is unusable.
- Spot and futures closes are aligned by date. Any timestamp or microstructure mismatch is absorbed by observation noise.
- Carry is a single combined implied-carry yield. It is not decomposed into dividends, funding, basis, hedging demand, or liquidity.
- With spot `S`, futures close `F`, continuously compounded rate `r`, and maturity `tau`, observed carry is

  ```text
  y(t,T) = r - log(F/S) / tau.
  ```

- The default continuously compounded risk-free rate is `0.014`.
- Both maturity and OU observation gaps are exchange trading sessions divided by `244`.
- Sessions are counted over `(observation date, expiry]`; expiry itself has zero remaining sessions.
- Standard CFFEX stock-index-futures expiry is inferred as the third Friday of the contract month, shifted forward if it is not a trading session.
- `2024-02-09` is an explicit exchange closure.
- The shared calibration projects exclude contracts with five or fewer sessions remaining.
- Implied carries with absolute value above `0.50` are excluded. Stale runs are flagged but retained unless configured otherwise.
- Expiry overrides are supported through CSV configuration, though the cached runs use inferred expiries.
- The raw cache is the reproducibility anchor. Use `--refresh` only when intentionally replacing it with a new AkShare snapshot.

Current common cached panel:

- sample: 2022-07-22 through 2026-08-21;
- 991 spot/curve dates;
- 3,964 raw futures rows;
- 52 contracts, `IM2208` through `IM2703`;
- 3,667 accepted futures observations;
- 297 excluded observations: 294 near/after expiry and 3 extreme carries.

The cleaned panel retains exclusions and reason codes in a quality-audit output rather than silently discarding them.

## Shared data and calendar implementation

In the calibration projects:

- `data.py` creates a candidate contract universe, downloads/caches AkShare data, validates the futures price field as `close`, aligns spot/futures, retains volume and open interest (`hold`), and assigns the configurable rate.
- `quality.py` flags missing/nonpositive inputs, duplicates, inconsistent expiries, near-expiry rows, extreme carries, and stale price runs. It returns both the accepted panel and a full audit.
- `calendar.py` implements third-Friday expiry and signed `(start, end]` session counts. When `chinese_calendar` cannot cover a distant year, the shared calendar falls back to ordinary weekdays. This is a known problem for 2027 maturities and must not be forgotten.

The Demo replaces only the calendar/quality layer with `calendar_utils.py` and `demo_quality.py`. It uses `chinese_calendar` through 2026, an explicit company calendar in `Demo/data/china_exchange_calendar_2027_2028.csv` for 2027–2028, and raises `CalendarCoverageError` outside covered years. It never silently uses the weekday fallback. The 2027–2028 calendar is provisional and must be confirmed or replaced once official CFFEX holidays are available.

Example of the difference: from 2026-08-21 to inferred `IM2703` expiry 2027-03-19, the shared weekday fallback gives 144 sessions, while the Demo company calendar gives 138.

## 1. Baseline one-factor project: `im_ou_carry`

### Model

The latent instantaneous carry follows

```text
dc_t = kappa (theta - c_t) dt + eta dW_t.
```

For maturity `tau`, the legacy observation equation is

```text
y_t(tau) = theta + B(kappa,tau) (c_t - theta) + epsilon,
B(kappa,tau) = (1 - exp(-kappa*tau)) / (kappa*tau),
epsilon ~ N(0, sigma_epsilon^2).
```

`kalman.py` implements stable maturity loadings, exact unequal-gap OU transitions, a stationary initial distribution, and a scalar-state Kalman filter that updates on an arbitrary number of contracts per date. `estimation.py` estimates positive parameters in log space using multi-start L-BFGS-B and computes numerical-Hessian standard errors when stable. `fitting.py` reconstructs fitted carry and futures prices. `diagnostics.py` covers train/test errors, benchmarks, maturity buckets, residual ACF, Ljung–Box/ARCH tests, expiry rolls, curve shapes, and maturity dependence. `pipeline.py` orchestrates acquisition, cleaning, calibration, filtering, rolling fits, evaluation, exports, and charts.

The fitted futures reconstruction is

```text
F_hat(t,T) = S_t exp[(r_t - y_hat_t(T)) tau].
```

### Current full-sample result

- `kappa = 7.93313547`
- `theta = 0.09310201`
- `eta = 0.80771883`
- `sigma_epsilon = 0.02078204`
- log likelihood `7950.93624`
- half-life `21.3192` trading sessions
- latest filtered carry `0.16257068`, standard deviation `0.02247358`
- numerical Hessian stable; optimizer converged

The automatic 80/20 evaluation split is 2025-10-29. Out-of-sample results are about 402.70 bp carry RMSE, 297.11 bp carry MAE, 70.76 futures points RMSE, and 51.89 points MAE.

### Structural limitation

A one-factor OU curve is monotonic toward `theta`; it cannot reproduce genuine humps or U-shapes. About 27.75% of observed curve dates were flagged as hump/U-shape dates. This limitation motivated the two-factor model and remains present in the correlated one-factor experiment.

### Entry points and validation

- CLI: `python -m im_ou_carry --config config.yaml`
- script: `analysis/run_workflow.py`
- tests cover calendar conventions, carry construction/auditing, ragged curves, stable transitions/loadings, and seeded parameter recovery.

## 2. Main two-factor project: `im_2factor_ou_carry`

### Model

Instantaneous carry is decomposed into centered independent slow and fast OU factors:

```text
c_t = theta + x_slow,t + x_fast,t
dx_j,t = -kappa_j x_j,t dt + eta_j dW_j,t
0 < kappa_slow < kappa_fast.
```

The observed curve is

```text
y_t(tau) = theta
           + B(kappa_slow,tau) x_slow,t
           + B(kappa_fast,tau) x_fast,t
           + epsilon_t(tau).
```

Opposite-signed slow and fast states can create one meaningful hump or U-shape. The factors are centered around zero so there is only one identified long-run level `theta`. Slow/fast shocks are independent and observation noise is one common Gaussian standard deviation.

`two_factor.py` implements the exact diagonal OU transition, stationary covariance, ragged two-dimensional Kalman filter, posterior states, and whitened sequential innovations. `two_factor_estimation.py` enforces factor ordering by estimating `log(kappa_slow)` and `log(kappa_fast-kappa_slow)`. The mean-reversion gap is capped at 60. `eta_fast_upper_bound` is configurable and defaults to 3.0; this optional argument was added for the Demo without changing the main-project default. `two_factor_fitting.py` attaches prior predictions, posterior fits, futures reconstructions, and both marginal and whitened prediction diagnostics.

The main pipeline also re-estimates the baseline one-factor model on exactly the same data, filter convention, and train/test split. Its model comparisons are therefore like-for-like.

### Current full-sample result

| Parameter | Estimate | Approx. SE |
|---|---:|---:|
| `kappa_slow` | 1.24090480 | 0.06550686 |
| `kappa_fast` | 44.32953559 | 1.34859073 |
| `theta` | 0.08261738 | 0.00183025 |
| `eta_slow` | 0.07992823 | 0.00379335 |
| `eta_fast` | 2.85207915 | 0.10085622 |
| `sigma_epsilon` | 0.00597024 | 0.00009035 |

- log likelihood `10862.80270`, AIC `-21713.60540`, BIC `-21676.36262`;
- slow half-life `136.2940` sessions; fast half-life `3.8152` sessions;
- latest slow state `+0.05990952`, fast state `-0.01614126`;
- latest instantaneous carry `0.12638564`, standard deviation `0.02746968`;
- all 12 full-sample optimizer starts reached the same interior solution and the Hessian was stable.

### Comparison with one factor

The two-factor model materially improves fit and out-of-sample futures pricing:

| Metric | Two factor | One factor |
|---|---:|---:|
| Log likelihood | 10862.80 | 7950.94 |
| AIC | -21713.61 | -15893.87 |
| BIC | -21676.36 | -15869.04 |
| OOS carry RMSE | 359.77 bp | 402.70 bp |
| OOS carry MAE | 213.43 bp | 297.11 bp |
| OOS futures RMSE | 33.16 | 70.76 |
| OOS futures MAE | 25.00 | 51.89 |

It captured 110 of 275 observed hump/U-shape dates (40%). It improved every maturity bucket but did not eliminate residual autocorrelation or volatility clustering.

### Interpretation and cautions

- The slow factor is plausibly a persistent carry regime or structural hedging-demand factor.
- The fast factor mainly controls the front end and may reflect temporary basis, roll, liquidity, or hedging pressure.
- These are statistical interpretations. The model does not establish that DMA activity caused the rise in carry.
- About 21.49% of dates are flagged as weakly observing the instantaneous state, using a nearest-contract/filtered-uncertainty rule. Large historical fast-state spikes on those dates should not be interpreted literally.
- Rolling 488-date estimates reveal more boundary pressure than the full sample: `eta_fast` reaches the default cap of 3 in four of five windows. Rolling fits are diagnostics only and do not overwrite the final full-sample calibration.

### Calibration roles

- Full-sample two-factor result: 991 dates, 12 starts, final reported parameters.
- Rolling diagnostics: five overlapping 488-date windows under current configuration, four two-factor starts per window; used only for stability/boundary analysis.
- Train/test refits: separate evaluation machinery, not final parameter selection.

### Entry points and validation

- CLI: `python -m im_2factor_ou_carry --config config.yaml`
- script: `analysis/run_workflow.py`
- key outputs: `two_factor_parameters.csv`, `two_factor_filtered_states.csv`, `two_factor_fitted_curves.csv`, `calibration_metrics.csv`, `model_information_criteria.csv`, `shape_fit_comparison.csv`, `standardized_innovation_tests.csv`, and `state_observability_diagnostics.csv`.
- tests cover both one- and two-factor formulas, ragged filtering, factor ordering, hump generation, state recovery, and seeded parameter recovery.

`OPTION_PRICING_WITH_OU_CARRY.md` explains downstream option use. For a CSI 1000/MO index option, compute a maturity-specific fitted carry/forward and use forward-form BSM. For an option directly on an IM futures contract, use the observed futures price in Black-76; applying OU carry again would double-count carry. OU factor volatilities are carry-state volatilities, not equity-option volatility.

## 3. Correlated one-factor experiment: `im_corr_ou_1factor`

### Model and exact futures formula

The reduced-form dynamics are

```text
dS/S = (r-c)dt + sigma dW_S
dc   = kappa(theta-c)dt + eta dW_c
corr(dW_S,dW_c) = rho,
```

with annual stock volatility fixed at `sigma = 0.25` rather than estimated.

For `tau=T-t`, the exact formula is

```text
log(F/S) = (r-theta)tau - (c-theta)B(tau)
           + 0.5 eta^2 C(tau) - rho sigma eta D(tau),
```

where `B` is the OU integral, `C` is the integral of `B^2`, and `D` is the integral of `B`. The resulting carry observation equation includes both Gaussian convexity and correlation corrections. `model.py` evaluates `B`, normalized loading, `C`, `D`, and the transition/return covariance integral `J` with small-argument series for numerical stability.

`filtering.py` provides:

- curve-only filtering;
- an exact joint curve/return filter that conditions the next OU prior on the close-to-close stock return using the full state/return covariance;
- a scalar RTS smoother.

The nuisance historical return drift `mu` is estimated in joint mode. Filtered states are used for live and out-of-sample work; smoothed states are exported only for retrospective analysis.

### Five specifications

1. `legacy_curve`: prior one-factor observation equation, no exact convexity/correlation correction.
2. `exact_rho0_curve`: exact formula with convexity and fixed `rho=0`.
3. `exact_corr_curve`: exact curve likelihood with free `rho`.
4. `exact_rho0_joint`: exact curve-plus-return likelihood with fixed `rho=0`.
5. `exact_corr_joint`: exact curve-plus-return likelihood with free `rho`.

The legacy model is not the nested `rho=0` restriction because it lacks the exact convexity term. Likelihood-ratio tests are valid only within exact curve mode and within exact joint mode. Raw curve-only and joint likelihoods are not comparable because the joint likelihood includes an extra return stream.

### Main result: correlation is not supported

For the exact correlated joint model:

- `kappa = 7.96673181`
- `theta = 0.09844654`
- `eta = 0.80731937`
- `rho = -0.03276593` with approximate SE `0.03605`
- `sigma_epsilon = 0.02079370`
- `mu = 0.16228026`
- half-life `21.2293` sessions

Correlation diagnostics:

| Diagnostic | Curve only | Joint curve/return |
|---|---:|---:|
| Point estimate `rho` | -0.6420 | -0.03277 |
| Approx. SE | 2.1303 | 0.03605 |
| 95% profile interval | entire tested `[-0.9,0.9]` | `[-0.10287,0.03783]` |
| LR statistic for `rho=0` | 0.1134 | 0.8245 |
| LR p-value | 0.7363 | 0.3639 |

The curve-only likelihood is nearly flat in `rho`; boundary-like negative rolling estimates are identification warnings, not evidence of large negative economic correlation. The joint likelihood is better identified but includes zero, changes sign across rolling windows, and does not improve out-of-sample performance. The exact correlated joint model has about 407.54 bp OOS carry RMSE and 71.28 points OOS futures RMSE, worse than the legacy and fixed-`rho=0` alternatives.

Conclusion: under fixed 25% stock volatility, nonzero stock/carry shock correlation is not empirically justified in the one-factor model, and correlation does not solve the one-factor maturity-shape problem.

### Entry points and validation

- full workflow: `analysis/run_workflow.py`
- profile-only refresh from saved optima: `analysis/refine_profiles.py`
- key outputs: `parameters.csv`, `likelihood_ratio_tests.csv`, `rho_profile_likelihood.csv`, `rho_profile_confidence_intervals.csv`, `states_filtered_and_smoothed.csv`, `standardized_innovations.csv`, and `calibration_metrics.csv`.
- validation covers analytical integrals versus quadrature, small-argument limits, exact Monte Carlo futures pricing, covariance positive-semidefiniteness, ragged/unequal-gap filters, filtered/smoothed separation, data auditing, and seeded joint recovery.

## 4. Carry-put pricing library: `carry_put_pricing`

### Contract scope

This project prices only the American optional component described in root `put_on_carry.md`:

```text
G_t = S_t [ exp((r-q_0,T)(T-t)) - F_t,T/S_t ]^+.
```

The locked inception carry is inferred from the observed initial spot/futures quote:

```text
q_0,T = r - log(F_0,T/S_0)/T.
```

The separate linear futures leg is deliberately excluded from the reported value.

### Exact stochastic-carry forward

The pricer uses the independent two-factor OU state under a provisional risk-neutral interpretation. With integrated carry `I_t,T = integral_t^T c_u du`, the integral is conditionally Gaussian, so

```text
F_t,T / S_t = exp(r*tau - E[I_t,T] + 0.5 Var[I_t,T]).
```

`analytics.py` implements stable OU integral loadings/variances, integrated-carry moments, exact implied carry, and exact forward ratios/prices. `models.py` validates the contract, OU factors, state, GBM inputs, and numerical grid configuration.

### State reduction and numerical method

Payoff homogeneity gives `V(t,S,x_s,x_f)=S*v(t,x_s,x_f)`. Under zero spot/carry shock correlation, normalized continuation is

```text
C_t/S_t = E[ exp(-integral_t^(t+dt) c_u du)
             * v(t+dt, X_t+dt) ].
```

`pricer.py` performs deterministic backward induction over a rectangular slow/fast factor grid using exact one-session OU transition/integral moments, Gaussian exponential tilting, separable Gauss–Hermite quadrature, and bilinear interpolation. Exercise is permitted once per trading session, so this is a daily Bermudan approximation to continuous American exercise.

Spot volatility remains an explicit `GBMParams` input but cancels from this homogeneous payoff under zero spot/carry correlation. The result scales linearly with spot when spot and futures are scaled together.

At inception, contractual exercise is fixed to zero because locked and prevailing observed futures coincide. A model-implied-versus-observed initial futures difference is reported as a fit diagnostic, not converted into exercise value.

### Futures-equivalent slow and fast curve deltas

The pricer now reports directional deltas for both OU carry factors, converted into option points per IM futures-price point. For `j` equal to slow or fast,

```text
Delta_j^F = (partial V / partial x_j) / (partial F_t,T / partial x_j),
partial F_t,T / partial x_j = -A(kappa_j,T-t) F_t,T,
A(kappa,tau) = (1-exp(-kappa*tau))/kappa.
```

Each direction holds spot and the other carry factor fixed. These are two scenario hedge ratios rather than one unique scalar delta: a single futures quote cannot identify both latent factor states, and one futures position cannot simultaneously neutralize arbitrary slow and fast shocks.

Two methods are implemented and compared:

1. **Differentiated backward induction (`pathwise_delta`).** The derivative follows the exercise policy selected by the original Snell envelope. At an exercise node it uses the exercise-payoff derivative; at a continuation node it uses the differentiated continuation recursion. It does not solve a second optimal-stopping problem for the derivative. Exact exercise/continuation ties use the continuation-side derivative.
2. **Local grid bump-and-value (`bump_and_value_delta`).** The time-zero value grid is evaluated one local grid step above and below the initial state along one factor axis. The option-value change is divided by the change in the exact model futures price over those same states.

The locked inception carry `q_0,T` is frozen under both calculations. Recomputing it from a bumped futures quote would re-strike the contract and would not be a valid Greek. Both conversions use the exact model futures mapping, not the observed/model initial-basis residual.

For the fixed `IM2609` base-grid example:

| Factor direction | Pathwise delta | Bump-and-value delta | Absolute difference |
|---|---:|---:|---:|
| Slow | -0.42744969 | -0.42793891 | 0.00048923 |
| Fast | -0.20766012 | -0.20839806 | 0.00073794 |

Increasing either carry factor raises the carry-put value and lowers the futures price, so both futures-equivalent deltas are negative. For a long carry put, the corresponding one-direction-at-a-time delta hedge is to buy about `0.42745` futures units for a pure slow shock or `0.20766` futures units for a pure fast shock, before applying contract multipliers and position sizes.

Across the coarse/base/fine grids, the pathwise slow deltas are approximately `-0.42831`, `-0.42745`, and `-0.42799`; the fast deltas are approximately `-0.20886`, `-0.20766`, and `-0.20836`. This is a numerical convergence diagnostic, not economic model uncertainty. The base comparison is exported to `outputs/curve_delta_comparison.csv`, and the structured results are also available as `PricingResult.slow_curve_delta` and `PricingResult.fast_curve_delta`.

### Fixed-carry scale delta

The pricer also reports a separate proportional scale sensitivity:

```text
fixed_carry_scale_delta = V / F_model.
```

This derivative follows the path `S(lambda)=lambda*S` and `F(lambda)=lambda*F`, holding the futures/spot ratio and therefore current implied carry fixed. The locked inception carry and both OU factor states are also fixed. It is positive and measures an overall index-level co-move; it is not the futures-only partial derivative with spot fixed and does not replace either slow or fast carry-curve delta. The field is retained in structured pricing results but is intentionally omitted from the current Demo notebook presentation because that presentation focuses on `q` shocks.

### Fixed full-sample example

The script `analysis/run_example.py` reads the latest full-sample two-factor parameters/states and selects the valid nearest contract from saved curves. Current example:

- valuation 2026-08-21, `IM2609`, expiry 2026-09-18, 20 sessions;
- spot `7601.804`, observed futures `7527.0`, locked carry `0.13464618`;
- base grid 301 x 401, width six stationary standard deviations, quadrature order 43;
- optional-component price `36.2794369` points (`0.477248%` of spot);
- model initial futures `7527.39615`, model-minus-observed `+0.39615` points;
- base-minus-fine grid difference about `0.00501` points;
- quadrature orders 39–47 span about `0.00517` points.
- fixed-carry scale delta: `0.00481965293` using the model initial futures price;
- slow futures-equivalent delta: pathwise `-0.42744969`, bump-and-value `-0.42793891`;
- fast futures-equivalent delta: pathwise `-0.20766012`, bump-and-value `-0.20839806`.

Tests cover stable analytical moments, seeded moment simulation, exact-forward regression values, zero one-session optionality, volatility invariance, spot scaling, deterministic flat carry, initial-basis reporting, delta sign and futures conversion, agreement between the two curve-delta methods, zero delta for a one-session zero-value contract, hedge-ratio invariance under proportional spot/futures scaling, and the fixed-carry `V/F_model` scale-delta identity.

### Economic limitations

- Historically estimated OU dynamics are only provisionally treated as risk-neutral.
- Slow/fast and spot/carry shocks are independent.
- Exercise is daily, not continuous.
- Rate is constant.
- Interpolation is clipped at remote grid boundaries.
- The separate linear futures leg and settlement mechanics are not priced.

A production valuation needs risk-neutral OU calibration or explicit carry risk premia. Useful benchmarks/extensions include Longstaff–Schwartz, risk-neutral parameter scenarios, and defensible nonzero correlations.

## 5. Configurable end-to-end Demo: `Demo`

### Purpose and current flow

The Demo accepts:

- valuation date;
- number of accepted curve dates in the calibration sample, including valuation date;
- selected IM futures contract.

It then:

1. loads the shared raw cache only through the requested valuation date;
2. rebuilds expiry/maturity/carry using the strict Demo calendar;
3. selects exactly the requested number of accepted curve dates ending on valuation date;
4. estimates close-to-close historical spot volatility over those dates;
5. recalibrates the independent two-factor OU model;
6. filters the latest state and validates the selected contract quote/expiry;
7. prices the option-only carry put and numerical convergence grids;
8. profiles fixed `eta_fast`, re-optimizing the other five parameters and repricing at each grid point;
9. exports CSV/JSON/PNG results and populates the narrative notebook.

Important code:

- `calibration.py`: sample selection, volatility, 12-start two-factor calibration, state filtering, selected quote, metrics, and exports.
- `calendar_utils.py` and `demo_quality.py`: strict extended calendar and local carry cleaning.
- `option_pricing.py`: converts calibration output into pricing inputs, runs base/coarse/fine and quadrature convergence, and exports results.
- `profile_analysis.py`: fixed-`eta_fast` profile; the other five parameters are optimized at every grid point and the option is repriced.
- `demo_workflow.py`: CLI, orchestration, charts, warnings, and `demo_summary.json`.
- `Carry_Put_Demo.ipynb`: narrative interface; its setup cell deliberately clears Demo-local modules from `sys.modules` before importing, preventing stale Jupyter module-cache errors. The option-pricing section is followed by an executed slow/fast futures-equivalent curve-delta table. It displays only carry-factor deltas and deliberately omits the fixed-carry scale delta.

### Current latest Demo snapshot

The executable defaults in `calibration.py` remain 2026-08-21 / 244 dates / `IM2609`, and `Demo/README.md` contains an older notebook example. However, the actual current notebook inputs and generated `Demo/outputs/demo_summary.json` are:

```text
VALUATION_DATE = 2026-08-10
SAMPLE_SIZE = 488
FUTURES_CONTRACT = IM2612
```

Treat that 2026-08-10 snapshot as the latest generated Demo state unless the user reruns it with new inputs.

Current calibration:

- sample 2024-08-05 through 2026-08-10;
- 488 curve dates, 1,807 accepted observations, 28 contracts, 487 returns;
- historical spot volatility `0.27710856`;
- `kappa_slow = 2.57648937`;
- `kappa_fast = 62.57648937`;
- `theta = 0.10862241`;
- `eta_slow = 0.11495544`;
- `eta_fast = 3.88517353`;
- `sigma_epsilon = 0.00592794`;
- log likelihood `5348.42993`;
- carry RMSE `46.7809` bp and futures RMSE `10.0540` points (posterior/in-sample fit metrics);
- optimizer converged and Hessian was stable.

The Demo passes `eta_fast_upper_bound=6`, so `eta_fast` is interior. Relative to fixing `eta_fast=3`, the cap-6 optimum gains about 12.19 log-likelihood units. The grid-supported 95% profile region is about 3.75–4.00 and maps to option prices about 62.60–65.58.

The mean-reversion gap `kappa_fast-kappa_slow` equals its separate upper bound of 60. Raising the volatility cap solved one boundary but exposed continued pressure toward faster mean reversion. This is a model-risk/identification warning, especially for short-dated pricing, not proof that L-BFGS-B failed.

Current `IM2612` option result:

- expiry 2026-12-18, 88 sessions;
- spot `7733.9`, futures `7413.2`, locked carry `0.13142796`;
- option-only price `63.85753` points (`0.825683%` of spot);
- model initial futures `7421.91640`, model-minus-observed `+8.71640` points;
- slow futures-equivalent delta: pathwise `-0.33235550`, bump-and-value `-0.33238170`;
- fast futures-equivalent delta: pathwise `-0.01406941`, bump-and-value `-0.01404718`;
- base-minus-fine difference `0.04413` points;
- nearby quadrature span `0.00191` points.

These numerical checks do not measure economic model uncertainty. The profile range is conditional on the fitted model and is not a full valuation confidence interval.

### Running the Demo

From `Demo`:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B demo_workflow.py `
  --valuation-date 2026-08-10 `
  --sample-size 488 `
  --futures-contract IM2612
```

The fixed-eta profile can take several minutes because the base calibration uses 12 starts and each fixed-eta point re-optimizes from four starts.

Focused Demo tests cover fixed/flexible sample selection, invalid sample sizes, selected quotes, expiry inference, strict 2027 calendar behavior, historical volatility, and locked-carry construction.

## Environments, dependencies, and validation history

The modelling projects require Python 3.10+ plus NumPy, pandas, SciPy, statsmodels, matplotlib, PyYAML, and AkShare. The calendars also rely on `chinese_calendar` in the working environments even though it is not listed in every `pyproject.toml`. The pricer needs NumPy, pandas, and SciPy.

Two environments appear in the project history:

- `D:\miniforge3\envs\spyder-env\python.exe`: agreed main environment and current Demo command.
- `D:\miniconda3\envs\GuoYuan\python.exe`: used to validate the correlated project and isolated carry-put pricer.

Do not install from Anaconda defaults. If a package is genuinely required, the project convention is conda-forge only. No new package was needed for the final Demo/pricer work.

Historical validation recorded in `session_log.md`:

- main two-factor project: 11 passing tests at the completed-project checkpoint, clean Ruff and compilation checks;
- correlated one-factor project: 16 passing tests, 21 parsed Python files, 22 nonempty CSVs, 10 charts;
- carry-put pricer: 10 passing tests after adding the slow/fast futures-equivalent deltas and fixed-carry scale delta; Ruff passed and the fixed example was regenerated on 2026-09-01;
- Demo: five focused tests plus Ruff historically; the full notebook executed successfully on 2026-09-01 after adding the curve-delta section, and the current generated snapshot was refreshed.

Because code and generated Demo outputs have since changed, rerun the relevant current test suites before claiming a new final validation.

## Generated outputs and source-of-truth hierarchy

Use this order when facts conflict:

1. current executable code and configuration;
2. current generated JSON/CSV outputs for the exact run being discussed;
3. `session_log.md` for decisions, interpretation, and validation history;
4. folder-level READMEs for stable usage context.

Generated outputs are snapshots, not universal constants. In particular, Demo outputs depend on notebook/CLI inputs and can differ from module defaults or README examples.

Several saved JSON fields contain stale absolute directories from earlier workspace locations, including spellings such as `Curry_curve_calibration` and older `D:\LuJingjian\...` paths. Use paths relative to the current repository root instead of trusting those metadata strings. Current numerical values remain useful when the corresponding data/run snapshot is identified.

Do not edit generated outputs merely to normalize paths unless the user asks for a rerun or cleanup.

## Main unresolved modelling issues

1. **Historical versus risk-neutral dynamics.** The pricing prototype uses historically estimated OU parameters as if they were under `Q`. Production pricing requires risk-neutral calibration or explicit carry-factor risk premia.
2. **Fast-factor identification.** Recent rolling windows and the Demo show boundary pressure in `eta_fast` and/or `kappa_fast-kappa_slow`. Very short-dated option values are especially sensitive.
3. **Residual dynamics.** Even two factors leave autocorrelation and volatility clustering; maturity-dependent or time-varying observation noise, stochastic volatility, or regimes may be needed.
4. **Correlation evidence.** The current one-factor data do not support nonzero stock/carry correlation. Curve-only `rho` is essentially unidentified; joint `rho` includes zero and does not improve forecasts.
5. **Calendar authority.** Shared code silently falls back to weekdays beyond holiday-package coverage. The Demo is stricter but its 2027–2028 calendar is provisional.
6. **Market conventions.** The rate is constant, expiries are rule-derived, and closes are treated as synchronized.
7. **Curve flexibility.** One factor cannot create humps/U-shapes; two factors usually support only one meaningful turning point.
8. **Option scope.** The carry-put result excludes the linear futures leg, uses daily exercise, zero correlations, and clipped state-grid interpolation.

## Sensible next steps

The clean next diagnostics discussed in the session are:

- rerun the identical cap-6 two-factor specification over 244-, 488-, 732-, and 991-date windows using the same strict calendar and optimizer-start policy, then compare mean-reversion-gap pressure and `IM2612` option values;
- build an independent Longstaff–Schwartz benchmark for the deterministic-grid carry-put pricer;
- perform risk-neutral OU sensitivity scenarios or calibrate carry risk premia;
- consider maturity-dependent observation noise and/or time-varying volatility;
- replace the provisional 2027–2028 company calendar with the official CFFEX schedule when available;
- if revisiting correlation, test sensitivity to fixed stock volatility and separate historical `(kappa_P, theta_P)` from pricing `(kappa_Q, theta_Q)` rather than interpreting the current curve-only estimate.

## Practical instructions for a future AI

- Begin by reading this file, `session_log.md`, and `git status`.
- Do not read or edit the root `README.md` unless the user explicitly asks; it is user-owned work in progress. In particular, preserve the user's existing `## Delta部分` wording and the appended `### 数值计算方法` explanation unless a future request explicitly targets them.
- Preserve all unrelated changes and generated artifacts.
- Use cached raw data unless the user explicitly wants a refreshed download.
- Keep trading-session carry time, option/volatility time, and discounting time explicit; do not silently switch to calendar-day/365 conventions.
- Never use settlement in place of the IM close.
- Never insert instantaneous carry as one constant rate for every option maturity.
- For an option on an IM futures contract, do not apply carry again to the observed futures price.
- Distinguish posterior fitted residuals from genuine one-step-prior out-of-sample prediction errors.
- Distinguish filtered states (live/forecast use) from smoothed states (historical-only use).
- Treat parameter-bound solutions and flat likelihood profiles as identification/model-risk diagnostics, not automatically as optimizer failures or economic evidence.
- When reporting a Demo value, state the valuation date, sample size, contract, calendar source, OU parameter set, and whether only the optional component is included.
