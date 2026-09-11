# Carry Curve Models, Carry-Put Pricing and Hedging — Project Guide

## Project purpose

This repository models the implied-carry term structure of CSI 1000 index
futures (`IM` contracts), calibrates one- and two-factor Ornstein--Uhlenbeck
(OU/Vasicek) state-space models, studies observation noise and parameter
identification, and prices an American-style put on the carry curve. It also
calculates directional factor deltas, a covariance-weighted one-futures hedge,
and a joint two-futures hedge. The Demo adds inception spot hedges for a LONG
option; the historical funded-spot study uses a SHORT option.

The pricing output covers only the carry-put optional component. The separate
linear futures leg is not included.

## Current scope and reading guide

The current deliverable is [Demo/Carry_Put_Demo.ipynb](Demo/Carry_Put_Demo.ipynb).
It prices the LONG optional component and reports one- and two-futures hedges
with accompanying spot positions at t=0 only. It does not simulate a hedge path.
The user has concluded the back-testing investigation; do not resume its
outstanding validation fixes or historical runs without a new request.

One futures plus spot is the preferred practical candidate among the tested
schemes: it delivers almost the two-futures scheme's measured risk reduction
with one fewer instrument. This is a sample-based simplicity/risk tradeoff,
not proof of global optimality, profitability, or minimum total risk.

This document describes architecture, conventions, equations, current artifacts
and known limitations. Numerical values are identified as run snapshots, not
universal defaults. `coding_plan.md` describes the implemented Demo extension;
read current code before treating it as pending work.

## Repository structure

| Folder | Role |
|---|---|
| `im_ou_carry` | Baseline one-factor OU carry calibration and diagnostics. |
| `im_2factor_ou_carry` | Main slow/fast two-factor calibration, one-factor comparison, filtering, diagnostics, and production-style outputs. |
| `im_corr_ou_1factor` | Exact correlated one-factor experiment using the futures curve alone or jointly with spot returns. |
| `carry_put_pricing` | Reusable American carry-put pricing library, curve deltas, and two-futures hedge calculation. |
| `carry_put_backtest` | Completed monthly cohort studies: legacy long-option futures-only hedges and short-option funded futures-plus-spot hedges, with ledgers and validation artifacts. |
| `fast_factor_boundary_study` | Isolated analysis of fast mean-reversion bounds, sample windows, and short-end exclusions. |
| `maturity_noise_study` | Comparison of constant-carry and maturity-dependent observation-noise specifications with chronological holdouts. |
| `Demo` | Configurable calibration/pricing workflow and notebook showing long-option carry-factor and spot hedge positions at inception. |

Main dependency flow:

```text
cached CSI 1000 spot and IM closes
              |
              +--> im_ou_carry
              +--> im_2factor_ou_carry
              |          |
              |          +--> carry_put_pricing
              |          +--> Demo/calibration.py
              |
              +--> im_corr_ou_1factor

carry_put_pricing/src
              |
              +--> Demo/option_pricing.py
```

The calibration projects intentionally duplicate some data and diagnostic code
to keep experiments isolated. The Demo imports the two-factor calibration and
pricing engines rather than duplicating them.

## Market data and conventions

- Spot is the CSI 1000 daily close from AkShare symbol `sh000852`.
- Futures inputs are individual `IMYYMM` daily **close** prices. Settlement is
  zero in the cache and must not be used.
- Spot and futures closes are aligned by date. Any timing mismatch is absorbed
  by observation noise.
- The continuously compounded risk-free rate is normally `0.014`.
- Observed annualized carry is

  $$
  y_{t,T}=r-\frac{\log(F_{t,T}/S_t)}{\tau},
  \qquad \tau=T-t.
  $$

- Maturity and OU time gaps use exchange trading sessions divided by 244.
- Remaining sessions are counted over `(observation date, expiry]`; expiry has
  zero remaining sessions.
- Standard CFFEX expiry is the third Friday of the contract month, shifted
  forward when that date is not a trading session.
- Contracts with five or fewer remaining sessions and carries with absolute
  value above 0.50 are excluded. Stale runs are flagged and normally retained.
- The raw data cache is the reproducibility anchor. Refresh it only when a new
  market-data snapshot is intentionally required.

The common cached panel runs from 2022-07-22 through 2026-08-21 and contains 991
curve dates, 52 contracts (`IM2208` through `IM2703`), 3,964 raw futures rows,
and 3,667 accepted observations.

### Calendar behavior

The shared calibration calendar uses `chinese_calendar` when covered and falls
back to weekdays for unsupported distant years. This affects 2027 maturities.
The Demo instead uses `chinese_calendar` through 2026, the explicit provisional
calendar in `Demo/data/china_exchange_calendar_2027_2028.csv` for 2027--2028,
and raises `CalendarCoverageError` outside covered years. The Demo never uses a
weekday fallback. Replace the provisional file when the official CFFEX calendar
is available.

## Carry models

### One-factor OU

The baseline instantaneous carry follows

$$
dc_t=\kappa(\theta-c_t)dt+\eta dW_t.
$$

With the legacy carry observation equation,

$$
y_t(\tau)=\theta+
\frac{1-e^{-\kappa\tau}}{\kappa\tau}(c_t-\theta)
+\varepsilon_t(\tau).
$$

The implementation uses exact unequal-gap OU transitions, a stationary initial
distribution, ragged daily curves, multi-start L-BFGS-B estimation, and a
Kalman filter. A one-factor curve is monotonic toward `theta`; it cannot produce
genuine humps or U-shapes. About 27.75% of observed curve dates have been
classified as hump/U-shape dates.

### Independent two-factor OU

The main state is

$$
c_t=\theta+x_{s,t}+x_{f,t},
$$

$$
dx_{j,t}=-\kappa_jx_{j,t}dt+\eta_jdW_{j,t},
\qquad 0<\kappa_s<\kappa_f.
$$

The slow and fast shocks are independent. Opposite-signed factors can generate
one meaningful hump or U-shape. Parameter ordering is imposed by estimating
`log(kappa_slow)` and `log(kappa_fast-kappa_slow)`.

Two observation specifications are implemented:

1. `constant_carry`

   $$
   y_t(\tau)=\theta+B_s(\tau)x_{s,t}+B_f(\tau)x_{f,t}
   +\varepsilon_{t,T},
   $$

   where

   $$
   B_j(\tau)=\frac{1-e^{-\kappa_j\tau}}{\kappa_j\tau}.
   $$

2. `constant_log_futures`

   $$
   z_{t,T}=\log(F_{t,T}/S_t)-r\tau
   =-\theta\tau-A_s(\tau)x_{s,t}-A_f(\tau)x_{f,t}+u_{t,T},
   $$

   $$
   A_j(\tau)=\frac{1-e^{-\kappa_j\tau}}{\kappa_j},
   \qquad u_{t,T}\sim N(0,\sigma_{\log F}^2).
   $$

   In carry units this is equivalent to noise standard deviation
   `sigma_log_futures/tau`, so very short maturities receive less weight. The
   reported likelihood includes the exact `sum(log(tau))` Jacobian when it is
   compared with carry-space models.

`constant_carry` remains the default in `im_2factor_ou_carry/config.yaml`.
`config_log_futures.yaml` is the opt-in candidate and writes to
`outputs_log_futures`.

The production log-futures snapshot estimates approximately
`kappa_slow=0.3413`, `kappa_fast=16.7336`, `eta_slow=0.0469`,
`eta_fast=1.2403`, and `sigma_log_futures=0.0009744`. Its slow and fast
half-lives are roughly 496 and 10 trading sessions.

### Correlated one-factor experiment

`im_corr_ou_1factor` uses

$$
\frac{dS_t}{S_t}=(r-c_t)dt+\sigma dW_t^S,
\qquad
dc_t=\kappa(\theta-c_t)dt+\eta dW_t^c,
\qquad
dW_t^SdW_t^c=\rho dt.
$$

It implements the exact stochastic-carry futures formula, curve-only filtering,
joint curve/return filtering, and an RTS smoother. Spot volatility is fixed at
25%. The joint estimate is approximately `rho=-0.0328` with standard error
0.0361; its profile interval includes zero and it does not improve forecasts.
The current data therefore do not support a nonzero correlation in this
one-factor specification.

## Research conclusions

### Fast-factor boundary study

- Under constant-carry noise, the original fast-minus-slow kappa-gap cap of 60
  is restrictive in the 488-date sample.
- Raising the cap gives an interior gap near 69.7, but improves log likelihood
  by only about 1.3 and changes the option price by about two points.
- Gap estimates vary materially across sample windows.
- Removing short maturities changes the economic model and does not provide a
  clean solution.

### Maturity-noise study

The study compares constant carry noise, two carry-noise buckets, smooth carry
plus log-price noise, and direct constant log-futures noise. Five expanding
calibrations with non-overlapping chronological holdouts support the direct
log-futures specification. Relative to constant carry noise, pooled results
improve carry RMSE by about 5.95%, carry MAE by 4.12%, futures RMSE by 5.59%,
and futures MAE by 7.25%. The log-futures model also produces substantially more
stable and interior fast-factor estimates.

Because option values and hedge ratios change materially, log-futures noise is
implemented as an opt-in candidate rather than the production default.

## Carry-put pricing

### Contract scope

The priced optional payoff is

$$
G_t=S_t\left[
e^{(r-q_{0,T})(T-t)}-\frac{F_{t,T}}{S_t}
\right]^+,
$$

where inception carry is locked from the observed quote:

$$
q_{0,T}=r-\frac{\log(F_{0,T}/S_0)}{T}.
$$

The separate linear payoff `F(t,T)-F(0,T)` is excluded.

### Exact forward and numerical method

For integrated carry

$$
I_{t,T}=\int_t^T c_u\,du,
$$

the OU model makes the integral conditionally Gaussian. The exact model forward
ratio is

$$
\frac{F_{t,T}}{S_t}
=\exp\left(r\tau-E_t[I_{t,T}]+\frac12\operatorname{Var}_t(I_{t,T})\right).
$$

Payoff homogeneity gives `V(t,S,x_s,x_f)=S*v(t,x_s,x_f)`, removing spot from
the state grid. `carry_put_pricing/src/carry_put_pricing/pricer.py` performs
daily backward induction on a two-dimensional factor grid using exact one-step
OU state/integral moments, Gaussian exponential tilting, separable
Gauss--Hermite quadrature, and bilinear interpolation. Exercise is allowed once
per trading session, so this is a daily Bermudan approximation to a continuous
American option.

Spot volatility is retained as an input but cancels under homogeneity and zero
spot/carry correlation. The price scales linearly when spot and futures are
scaled together.

### Deltas

The pricing result includes a fixed-carry scale sensitivity

$$
\Delta_{\mathrm{scale}}=\frac{V}{F_{\mathrm{model}}},
$$

which co-scales spot and futures at fixed current carry. It is distinct from a
futures-only partial derivative.

For each carry factor,

$$
\Delta_j^F=
\frac{\partial V/\partial x_j}{\partial F/\partial x_j},
\qquad
\frac{\partial F}{\partial x_j}=-A_j(\tau)F.
$$

These are one-factor-at-a-time directional ratios. They are calculated through
differentiated backward induction and checked with a local grid bump-and-value
calculation. A single futures contract generally cannot neutralize both factor
exposures simultaneously.

### Joint hedge with two futures

`carry_put_pricing/src/carry_put_pricing/hedging.py` defines
`HedgeFuturesContract`, `TwoFuturesHedgeResult`, and
`calculate_two_futures_hedge`. For two observed futures quotes,

$$
G=
\begin{bmatrix}
-F_1A_s(h_1)&-F_2A_s(h_2)\\
-F_1A_f(h_1)&-F_2A_f(h_2)
\end{bmatrix}.
$$

The option deltas solve

$$
G\Delta=
\begin{bmatrix}V_s\\V_f\end{bmatrix},
$$

and the positions that hedge a long option are

$$
n=-\Delta,
\qquad
Gn=-\begin{bmatrix}V_s\\V_f\end{bmatrix}.
$$

The result reports both deltas and hedge positions, the determinant, condition
number, angular separation, and post-hedge residual factor exposures. Numerical
rank is checked with singular values. If the matrix is singular, a
`RuntimeWarning` is emitted and the deltas and positions are returned as
`None`. Near-singular pairs can still generate unstable positions and should be
judged using the conditioning diagnostics.

Observed hedge-futures closes are used. The standalone example and Demo obtain
their hedge maturities from the strict Demo calendar. The separate
`carry_put_backtest` package implements monthly hedge backtesting, contract
multipliers, optional rounding/costs, and inception pair selection. Its fixed
contract identities require no rolling. Historical runs cover the available
24 cohorts; numerical checks cover a finer-grid pilot and selected states,
not a full finer-grid batch. Validation limitations are documented below.

The baseline pricing example on 2026-08-21 uses an `IM2609` option and
`IM2609`/`IM2703` hedge futures. With strict maturities of 20/138 sessions, the
continuous long-option hedge is approximately +0.153757/+0.055513 futures
units. The result is exported to
`carry_put_pricing/outputs/two_futures_hedge.csv`.

### One futures carry hedge and inception spot hedge

For a LONG unit-point option let b=(V_s,V_f) and
`g_i = -F_i * (A_s(h_i), A_f(h_i))`, using the observed hedge quote and strict
calendar maturity. `calculate_one_futures_hedge` returns the continuous trade

$$n_{carry}=-\frac{g_1^T Q b}{g_1^T Q g_1}.$$

Here Q is the existing helper's one-session OU innovation covariance. The
position minimizes local residual carry-factor variance `(b+n*g_1)^T Q
(b+n*g_1)`. It differs conceptually from either directional hedge `-V_j/g_1j`
and is not their simple average or a total-P&L-variance optimizer. The zero
stochastic-risk fallback is no futures trade.

At fixed locked contractual carry and current factor coordinates, homogeneity
implies `dV/dS=V/S` and local futures scale exposure `F_i/S`. The inception spot
positions for one LONG option are

$$H_{one}=-\frac{V_0+n_{carry}F_1}{S_0},\qquad
H_{two}=-\frac{V_0+n_1F_1+n_2F_2}{S_0}.$$

Thus `V0 + sum(n_i*F_i) + H*S0 = 0`. This is a local exposure identity, not a
market-value or funding identity; futures notionals are not principal payments.
Positive positions mean long/buy; negative positions mean short/sell. Demo
ratios use one option with unit point values, continuous futures and spot units,
and no multipliers or rounding. The spot instrument is hypothetical and has
zero cash dividends; fitted carry and pricing dynamics are unchanged. Spot
hedging does not change factor exposure at fixed spot/factor coordinates.

## Demo

The Demo exposes these inputs through Python, CLI, and the notebook:

- valuation date;
- number of accepted calibration curve dates;
- option futures contract;
- exactly two hedge futures contracts;
- observation-noise model;
- kappa-gap and fast-volatility bounds;
- output directory.

Its flow is:

1. load cached spot/futures data only through the valuation date;
2. rebuild expiries and maturities with the strict calendar;
3. select the requested number of accepted curve dates;
4. estimate historical spot volatility;
5. calibrate and filter the two-factor OU model;
6. validate the option and hedge futures quotes;
7. price the optional component and calculate directional and joint deltas;
8. run grid/quadrature diagnostics;
9. run the fixed-`eta_fast` profile;
10. export CSV, JSON, and chart artifacts.

Important files:

- `Demo/calibration.py`: sample construction, calibration, states, and metrics.
- `Demo/calendar_utils.py`: strict expiry and session calendar.
- `Demo/option_pricing.py`: pricing inputs, convergence, hedge inputs, and
  exports.
- `Demo/profile_analysis.py`: conditional fixed-`eta_fast` likelihood and price
  profile.
- `Demo/demo_workflow.py`: CLI, orchestration, warnings, charts, and summary.
- `Demo/inception_hedging.py`: pure long-option inception spot sizing and wrappers
  around existing one-/two-futures hedge results; finite-input checks and
  unavailable singular-joint handling.
- `Demo/tests/test_inception_hedging.py`: sign, units, covariance objective,
  singularity, finite inputs, and notebook integration checks.
- `Demo/Carry_Put_Demo.ipynb`: narrative interface and live inception hedge tables.

The latest generated notebook configuration is 2026-08-10, 488 dates, option
contract `IM2609`, hedge contracts `IM2609`/`IM2703`,
`constant_log_futures`, kappa-gap cap 90, and `eta_fast` cap 6. Its current
optional-component price is approximately 38.27465 points. The continuous
long-option hedge is approximately +0.221144 `IM2609` and +0.046084 `IM2703`;
the hedge-matrix condition number is about 14.44. Treat these as snapshot values
that change with the selected date, sample, contract, and model.

The fixed-`eta_fast` profile currently runs unconditionally and is expensive.
For an interior log-futures estimate with a stable Hessian, it is mainly a
research diagnostic and could be made optional. The current notebook also has a
known display-path issue: the filtered-state/latest-curve image cell loads
charts from the hard-coded baseline `Demo/outputs` directory instead of the configured
`OUTPUT_DIR`, so a log-futures table can be shown beside an old constant-carry
chart.

The notebook preserves directional slow/fast deltas as diagnostics. It then
reports the combined carry-factor position for the FIRST configured hedge
future, followed by its spot H. The joint section retains both existing
futures positions/conditioning diagnostics and adds H directly. A singular
joint hedge leaves its spot hedge unavailable without a silent fallback.
These notebook calculations use the live `demo.pricing` object; the CLI
workflow alone does not add the notebook spot tables to its summary export.

The saved executed notebook reports inception futures-plus-spot hedge
trades per one long option. With the 2026-08-10 default live run, the first
configured future (`IM2609`) has a combined minimum-carry-variance position of
approximately **+0.271165**, followed by **-0.271672** spot units. The existing
joint positions remain approximately **+0.221144 IM2609** and **+0.046084
IM2703**, with accompanying spot **-0.265492**. Both scale residuals are zero to
displayed precision. These are continuous unit-point inception sensitivities,
not literal exchange contracts, funded trades, or a dynamic hedge strategy.

## Historical hedge studies and economic conclusions

### Shared design

The completed studies use the unchanged cache through 2026-08-21. Exactly one
cohort begins on the first exchange trading session after each monthly IM
expiry; no entries occur between those dates or replace early exercises. The
option references the next monthly IM contract. Each cohort calibrates the
latest 488 accepted curve dates through inception, including that close, then
freezes parameters and forward-filters daily without smoothing/refitting.

Settings include log-futures noise, kappa-gap cap 90, eta-fast cap 6, 12 starts,
seed 852, and maxiter 1500. The first hedge is the option underlying; the second
is the longest-dated later accepted IM maturity at inception. Contract identities
remain fixed. All scenarios in a cohort share the holder's exercise decision,
locked inception carry and zero terminal optional payoff. Baselines use
continuous unit positions, zero costs, 1.4% symmetric funding and same-close
execution. Configurable multipliers, rounding, costs and a joint one-session
lag belong to the back-test, not the inception Demo.

There are 48 scheduled entries: 24 excluded for insufficient warm-up and 24
completed cohorts entered 2024-08-19 through 2026-07-20. There are 22 early
exercises, two expiries and 245 daily P&L intervals per scenario; the final
actual exit is 2026-08-10. Daily risk below is pooled discounted P&L standard
deviation, excluding inception rows. Cohort totals are separate-trade sums,
not annual returns or a continuously funded portfolio NAV.

### Long-option futures-only study

The default `long_futures_only` path remains preserved in
`carry_put_backtest/outputs_historical_accelerated/`; the NumPy reference pilot
is in `outputs_historical/`. The optional Numba likelihood backend is used
explicitly by the historical runner and was checked against the reference;
the generic estimator still defaults to NumPy.

| Long-option scenario | Sum P&L (points) | Daily P&L std (points) |
|---|---:|---:|
| No hedge | 80.08 | 7.361 |
| One futures | 1,285.77 | 29.312 |
| Two futures | 1,263.26 | 28.852 |

The carry-factor hedge introduces large spot-scale exposure. Higher realized
profit here is not evidence of better hedging. One futures removes 99.9773%
of the reported modeled factor variance; two removes it to numerical precision,
but total realized risk rises. Maximum historical condition number is 119.94;
no singular pair or configured fast-factor boundary was encountered.

### Short-option funded futures-plus-spot study

The explicit `short_with_spot` path negates long-helper futures positions once
and applies option/futures monetary conversion and any futures rounding before
sizing continuous spot. For side s=-1 and option point value M, funded spot is
`H = -(s*M*V + MF*sum(N_i*F_i))/S`. Equity is `cash + H*S + s*M*V`.
Cash receives futures variation P&L, financing and signed exercise cash flows;
spot trades change cash, while spot mark-to-market is attribution, not a second
cash credit. Zero dividends do not zero model carry.

Use `carry_put_backtest/outputs_short_spot_review_fixed/` for the fresh
current-source study. Its baseline results are:

| Short-option scenario | Sum P&L (points) | Daily P&L std | RMS active cash-scale exposure |
|---|---:|---:|---:|
| No hedge | -80.082507 | 7.360822 | 19.737801 |
| One futures + spot | -276.617669 | 3.911521 | approximately 0 |
| Two futures + spot | -272.485234 | 3.890142 | approximately 0 |
| One futures only | -1,285.773227 | 29.311837 | 2,015.829258 |
| Two futures only | -1,263.259184 | 28.852239 | 1,984.489503 |

One/two futures plus spot reduce variance versus unhedged by 71.76%/72.07%
(standard-deviation reductions 46.86%/47.15%). The second futures reduces
standard deviation by only about 0.55% relative to one futures plus spot.
This supports the one-futures-plus-spot practical preference, while retaining
two futures as a benchmark. It does not establish statistical superiority.

The one-futures-plus-spot total consists of -80.45 option P&L, -1,205.27 futures
P&L, +1,035.03 spot P&L and -25.92 financing. Spot reduces risk but does not
fully offset futures losses. Both hedged totals and their worst cohort losses
are worse than unhedged in this sample. Peak individual-cohort borrowing/spot
notional are 4,669.81/4,713.35 points for one futures and 4,627.94/4,671.74 for
two; these are maxima over complete paths, not sums of peaks or portfolio
capital requirements.

Cash-scale residuals are at most 4.55e-13; two-futures slow/fast residuals are
at most 5.68e-14/4.26e-14 in their respective units. Independent cash replay
errors are about 1e-12. Short unhedged/futures-only controls exactly negate
legacy long results under the symmetric zero-cost convention.

Joint lag gives funded-spot totals -164.86/-162.17 and daily std 4.317/4.277.
Illustrative one-way costs of 0.2 futures points plus 1 bp spot notional give
-300.48/-296.12. The 2025-07-21 fine pilot (451x601, quadrature 61) changes
funded hedge totals by +0.351467/+0.353215 and preserves the exit outcome.
No full finer-grid historical sweep or exact P&L convergence is claimed.

### Evidence and remaining validation limitations

The original `outputs_short_spot/` is preserved evidence with unresolved
execution-source provenance; old aggregate hashes had been refreshed after
execution. A fresh run in `outputs_short_spot_review_fixed/` provides matching
current-source baseline/fine manifests. Independent review found all 524
preserved artifact hashes intact and all 1,345 shared old/new ledger rows and
fields equal. Numerical agreement does not retroactively repair old provenance.

Corrections include separate analysis manifests, actual subprocess test logs,
fresh sensitivity replay, cohort-set checks, peak maxima and pooled metrics.
The latest review still identified validation weaknesses: NaNs can evade
error tolerances; test identities omit dependencies and changes during a run;
missing source files can be classified as matching; and exit consistency is
not checked fully across artifacts. These are not known errors in the supplied
hedge P&L, but prevent claiming a fully robust validation layer. They remain
out of the user's current work scope.

Read `carry_put_backtest/short_spot_bugfix_astra_review.md` for reproducible
findings, `short_spot_bugfix_handoff.md` for the implementation bundle, and
`outputs_short_spot_review_fixed/short_spot_results_report.md` for results.
`carry_put_backtest/README.md` documents the API and ledger conventions.

## Entry points

From the `im_2factor_ou_carry` folder, run the main calibration with either
configuration:

```powershell
python -m im_2factor_ou_carry --config config.yaml
python -m im_2factor_ou_carry --config config_log_futures.yaml
```

Standalone carry-put example:

```powershell
python -B carry_put_pricing/analysis/run_example.py `
  --hedge-futures-contracts IM2609 IM2703
```

Demo example:

```powershell
python -B Demo/demo_workflow.py `
  --valuation-date 2026-08-10 `
  --sample-size 488 `
  --futures-contract IM2609 `
  --hedge-futures-contracts IM2609 IM2703 `
  --observation-noise-model constant_log_futures `
  --kappa-gap-upper-bound 90 `
  --eta-fast-upper-bound 6 `
  --output-dir Demo/outputs_log_futures
```

The current user-requested and validated interpreter is
`D:\miniforge3\envs\spyder-env\python.exe`. Required packages include NumPy,
pandas, SciPy, statsmodels, matplotlib, PyYAML, AkShare, and
`chinese_calendar`. Do not install packages from Anaconda defaults; use
conda-forge if an installation is genuinely required.

## Validation and source-of-truth hierarchy

Independent checks of the inception Demo extension: **25 Demo/pricing tests
passed** using spyder-env. This includes seven new inception-hedge tests.
Long-option signs, one-futures covariance minimization, mixed-sign two-futures
spot sizing, singular/unavailable results and notebook integration are covered.
Independent recalculation from saved Demo inputs reproduced the displayed
positions and factor/scale identities. The saved notebook has no error outputs;
this review did not rerun the full calibration/profile notebook workflow.

The latest back-test-focused review ran 58 tests successfully. The delivered
cross-project evidence records 93 passed, subject to the test-provenance
limitations above; that full suite was not rerun during the Demo review.
Earlier research-study test counts are historical snapshots rather than a
current cross-project certification.

When facts conflict, use this order:

1. current executable code and configuration;
2. generated JSON/CSV outputs for the exact run;
3. `session_log.md` for historical decisions and validation context;
4. folder READMEs for stable usage guidance.

Generated files are run-specific snapshots, not universal constants. Some
historical JSON metadata contains stale absolute paths; use paths relative to
the repository root.

## Main limitations and future work

1. Historical OU dynamics are provisionally treated as risk-neutral dynamics.
   Production pricing needs risk-neutral calibration or explicit factor risk
   premia.
2. Constant log-futures noise improves forecasts and parameter stability but
   materially changes option values and hedges, so it remains opt-in.
3. Residual autocorrelation and volatility clustering remain after the
   two-factor fit.
4. The 2027--2028 Demo calendar is provisional; shared production code still
   has a weekday fallback outside package coverage.
5. Rates are constant, expiries are rule-derived, and closes are assumed
   synchronized.
6. Pricing uses daily exercise, zero spot/carry and slow/fast correlations, and
   clipped state-grid interpolation.
7. The joint hedge removes only local first-order model-factor exposure. It
   does not remove basis, parameter, nonlinear, liquidity, or execution risk.
8. Hypothetical zero-dividend tradable spot with observed futures carry need not
   be arbitrage-consistent. Funded spot hedges reduce measured historical risk
   but do not demonstrate profitable or risk-free replication. Numerical checks
   are pilots/selected states, not a full fine-grid historical batch.
9. Useful pricing extensions include an independent Longstaff--Schwartz
   benchmark and risk-neutral parameter scenarios.
