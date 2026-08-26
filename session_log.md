# Session Log — 2026-08-24

## Scope completed

Today we developed and evaluated models for the CSI 1000/IM implied-carry term structure, beginning with the requested one-factor Ornstein–Uhlenbeck model and then extending it to a two-factor OU model to better represent humps, inversions, and U-shaped curves.

The two completed project directories are:

- One-factor model: `D:\LuJingjian\Jupyter_files\GuoYuan\Studies\carry_rate\im_ou_carry`
- Two-factor model: `D:\LuJingjian\Jupyter_files\GuoYuan\Studies\carry_rate\im_2factor_ou_carry`

## Agreed data and modelling conventions

We confirmed the following conventions, which override the corresponding initial specifications where they differ:

- Use the IM futures **close**, not settlement.
- Use the CSI 1000 spot close obtained through AkShare.
- Treat carry as one combined implied-carry yield; do not decompose it into dividends, funding, basis, or hedging demand.
- Use the trading-session count in `(observation date, expiry]`.
- Convert maturity and OU observation gaps using trading sessions divided by 244.
- Use `0.014` as the default continuously compounded risk-free rate when no rate series is supplied.
- Use the `spyder-env` Python environment.
- If an additional package is needed, use `conda-forge` only and never Anaconda defaults or another official Anaconda channel.
- Refer to `D:\LuJingjian\Jupyter_files\GuoYuan\Studies\carry_rate\workdays_count.py` for the trading-day-count convention.

## Data panel

Both model versions used the same processed historical panel:

- Sample dates: 2022-07-22 through 2026-08-21
- Accepted observations: 3,667
- Curve dates: 991
- Distinct contracts: 52
- Excluded observations: 297
  - 294 close-to-expiry observations
  - 3 extreme implied-carry observations

The downloaded raw inputs are cached under each project's `data/raw/` directory, and filtering decisions are retained in the generated outputs rather than being silently discarded.

## One-factor OU implementation

The initial project implemented a custom Kalman filter and maximum-likelihood estimator for a latent instantaneous carry state:

$$
dc_t=\kappa(\theta-c_t)dt+\eta\,dW_t.
$$

It includes exact unequal-gap OU transitions, an arbitrary number of contracts per curve date, multiple optimizer starts, filtered-state inference, fitted carry curves, reconstructed futures prices, residual diagnostics, rolling estimation, a train/test split, benchmark comparisons, charts, configuration, documentation, and synthetic recovery tests.

The one-factor results established a useful baseline, but the model could not flexibly reproduce maturity-curve humps and U-shapes. Its remaining maturity-dependent futures-price errors motivated the two-factor extension.

## Two-factor OU implementation

We adopted the following state decomposition:

$$
c_t=\theta+x_{s,t}+x_{f,t},
$$

$$
dx_{j,t}=-\kappa_jx_{j,t}dt+\eta_jdW_{j,t},
\qquad 0<\kappa_s<\kappa_f.
$$

The curve observation equation is

$$
y_t(\tau)
=
\theta
+B(\kappa_s,\tau)x_{s,t}
+B(\kappa_f,\tau)x_{f,t}
+\varepsilon_t,
$$

where

$$
B(\kappa,\tau)
=
\frac{1-e^{-\kappa\tau}}{\kappa\tau}.
$$

The initial two-factor specification uses independent slow and fast state shocks and one common observation-error volatility.

### Full-sample parameter estimates

| Parameter | Estimate | Approximate SE |
|---|---:|---:|
| $\kappa_{slow}$ | 1.2409048 | 0.0655069 |
| $\kappa_{fast}$ | 44.3295356 | 1.3485907 |
| $\theta$ | 0.08261738 | 0.00183025 |
| $\eta_{slow}$ | 0.07992823 | 0.00379335 |
| $\eta_{fast}$ | 2.85207915 | 0.10085622 |
| $\sigma_\varepsilon$ | 0.005970235 | 0.00009035 |

The associated half-lives are:

- Slow factor: 136.29 trading sessions
- Fast factor: 3.82 trading sessions

All 12 optimizer starts converged to the same optimum, and the numerical Hessian was stable.

### Latest filtered state

| Quantity | Estimate |
|---|---:|
| Slow factor | +0.0599095 |
| Fast factor | -0.0161413 |
| Long-run level $\theta$ | 0.0826174 |
| Instantaneous carry | 0.1263856 |
| Filtered-state standard deviation | 0.0274697 |

## One-factor versus two-factor results

The two-factor model improved both in-sample fit and out-of-sample pricing accuracy.

| Metric | Two-factor | One-factor |
|---|---:|---:|
| Log-likelihood | 10,862.80 | 7,950.94 |
| AIC | -21,713.61 | -15,893.87 |
| BIC | -21,676.36 | -15,869.04 |
| Out-of-sample carry RMSE | 359.77 bp | 402.70 bp |
| Out-of-sample carry MAE | 213.43 bp | 297.11 bp |
| Out-of-sample futures RMSE | 33.16 points | 70.76 points |
| Out-of-sample futures MAE | 25.00 points | 51.89 points |

For the latest curve, the two-factor futures residuals were approximately `+0.38`, `-5.03`, and `-4.99` points, compared with `+5.88`, `-35.32`, and `-75.41` points for the one-factor model.

The two-factor model captured 110 of 275 observed hump/U-shape dates, or 40%. It materially improved every maturity bucket and the near-expiry bucket, although it did not eliminate residual autocorrelation or volatility clustering.

## Economic discussion

Rolling estimates showed that the long-run carry level rose from roughly 6.5–7.0% in earlier windows to 10.87% in the latest rolling window. This is consistent with the hypothesis that increased use of IM futures by DMA funds for hedging may have raised observed implied carry in recent years. The analysis supports this as a plausible interpretation, but the statistical model by itself does not establish causality.

The estimated fast-factor volatility reached its configured upper bound of 3.0 in four of five rolling windows. This indicates parameter instability or model pressure in the short end and should be treated as a warning rather than a robust structural result.

## Fast-factor observability caveat

The fast factor can be weakly identified on dates when the nearest listed contract is not sufficiently short-dated. In that situation, the filter may infer a large zero-maturity fast-state movement even while matching the actually traded maturities reasonably well.

- 21.49% of dates were flagged as weakly observed.
- A date is flagged when the nearest contract exceeds 21 trading sessions or the instantaneous-state standard deviation exceeds 4%.
- Detailed flags are stored in `outputs/state_observability_diagnostics.csv`.

Historical instantaneous fast-state spikes on flagged dates should therefore not be interpreted literally.

## Chart walkthrough

We reviewed the nine generated charts and reached these main interpretations:

- The slow-factor loading remains important across the curve, whereas the fast factor mainly controls the front end.
- The latest fitted curve is substantially closer to observed carries and futures prices under the two-factor model.
- The filtered slow state is economically interpretable; large fast-state spikes require the observability warning above.
- Average maturity bias was largely removed, but residual persistence and volatility clustering remain.
- Rolling estimates show a rising long-run carry level and repeated pressure on the fast-volatility bound.
- The out-of-sample innovation autocorrelation chart measures standardized innovation ACF, despite an inherited title referring to a daily mean carry residual.

## Using the model for option pricing

We clarified that the instantaneous state $c_t$ should not be inserted as one constant carry rate for options of every maturity. For a maturity $T$, first calculate the maturity-average fitted carry:

$$
\widehat y_t(T)
=
\theta
+B(\kappa_s,\tau)x_{s,t}
+B(\kappa_f,\tau)x_{f,t}.
$$

For a CSI 1000/MO index option, construct the model-implied forward:

$$
\widehat F_t(T)
=
S_t\exp\left[
\left(r_t-\widehat y_t(T)\right)\tau_{carry}
\right],
$$

then use that forward in forward-form BSM:

$$
C_t
=
D_r(t,T)
\left[
\widehat F_t(T)N(d_1)-KN(d_2)
\right].
$$

For an option directly on an IM futures contract, use Black-76 with the observed futures price. Adding the OU carry to that futures price would double-count carry.

The OU factor volatilities $\eta_{slow}$ and $\eta_{fast}$ are carry-state volatilities, not CSI 1000 option volatility. The BSM volatility input must come separately from an MO implied-volatility surface, a CSI 1000 return-volatility model, or a scenario assumption.

The current practical approach treats the fitted forward as deterministic. A fully stochastic-carry pricing model would additionally require risk-neutral OU parameters, prices of carry risk, spot/carry correlations, convexity treatment, and a Monte Carlo, PDE, or affine pricing method.

The full option-pricing note is available in `OPTION_PRICING_WITH_OU_CARRY.md`.

## Validation and deliverables

The two-factor project was run end-to-end in `spyder-env` and validated with:

- 11 passing tests, including seeded one-factor and two-factor parameter-recovery tests;
- clean Ruff checks;
- successful Python compilation;
- verified source-file hashes after delivery;
- 28 CSV outputs;
- 9 diagnostic charts;
- one JSON run summary.

Important entry points and outputs include:

- `README.md`
- `config.yaml`
- `analysis/run_workflow.py`
- `outputs/run_summary.json`
- `outputs/calibration_metrics.csv`
- `outputs/shape_fit_comparison.csv`
- `outputs/rolling_parameter_comparison.csv`
- `outputs/standardized_innovation_tests.csv`
- `outputs/state_observability_diagnostics.csv`
- `OPTION_PRICING_WITH_OU_CARRY.md`

## Logical next step

The next practical extension is an option-pricing module with two clearly separated routes:

1. MO/CSI 1000 index options: build a maturity-specific OU-implied forward and pass it to forward-form BSM.
2. Options on IM futures: use the observed futures price in Black-76 without applying carry again.

The implementation should keep carry time, volatility time, and discounting time as separate inputs so their respective day-count conventions remain explicit.

---

# Session Update - Correlated One-Factor OU Model

## Scope completed

A new, isolated project was implemented and calibrated at:

- `D:\Jupyter_files\Curry_curve_calibration\im_corr_ou_1factor`

The existing `im_ou_carry` and `im_2factor_ou_carry` projects were left unchanged. The new project extends the one-factor carry model by allowing shocks to the latent carry state to be correlated with CSI 1000 stock-return shocks.

The user confirmed the following choices before implementation:

- Continue using trading-session time divided by 244 for both maturity and OU observation gaps.
- Keep the legacy one-factor observation equation separate from the exact correlated model restricted to `rho=0`.
- Fix annualized CSI 1000 stock volatility at `sigma=0.25` in this first version rather than estimating it.

## Data and cleaning

The new project uses its own copy of the established raw-data cache:

- Spot rows: 991
- Futures rows: 3,964
- Sample: 2022-07-22 through 2026-08-21
- Contracts: 52, from `IM2208` through `IM2703`
- Four raw futures contracts per curve date
- Accepted observations: 3,667 across 991 dates
- Excluded observations: 297
  - 294 near or after expiry
  - 3 extreme implied-carry observations

Futures settlement is zero throughout the cache, so the model continues to use futures close. Spot and futures closes are aligned by daily date. The continuously compounded risk-free rate remains configurable and defaults to `0.014`. Contract expiries continue to use the CFFEX third-Friday rule, holiday shifting, the special 2024-02-09 closure, and the `(observation date, expiry]` session-count convention.

## Correlated model implementation

The reduced-form dynamics are

$$
\frac{dS_t}{S_t}=(r_t-c_t)dt+\sigma dW_t^S,
$$

$$
dc_t=\kappa(\theta-c_t)dt+\eta dW_t^c,
\qquad dW_t^S dW_t^c=\rho dt.
$$

The exact futures-pricing equation implemented is

$$
\log\frac{F(t,T)}{S_t}
=(r_t-\theta)\tau
-(c_t-\theta)B(\tau)
+\frac12\eta^2C(\tau)
-\rho\sigma\eta D(\tau).
$$

The resulting curve observation equation is

$$
y_t(T)=\theta+(c_t-\theta)\frac{B(\tau)}{\tau}
-\frac{\eta^2C(\tau)}{2\tau}
+\frac{\rho\sigma\eta D(\tau)}{\tau}
+\varepsilon_{t,T}.
$$

Stable small-argument implementations were added for `B`, `C`, `D`, and `J`. The exact joint curve-and-return filter conditions the next OU state prior on the close-to-close stock return using the full state/return covariance. The nuisance historical return drift `mu` is estimated in joint mode. Filtered states are used for live and out-of-sample analysis; RTS-smoothed states are exported separately and labeled as historical-only.

## Models compared

Five specifications were estimated:

1. `legacy_curve`: the previous one-factor observation equation, without exact convexity or correlation corrections.
2. `exact_rho0_curve`: exact futures pricing with the convexity correction and `rho=0`.
3. `exact_corr_curve`: exact curve likelihood with free `rho`.
4. `exact_rho0_joint`: exact joint curve/return likelihood with `rho=0`.
5. `exact_corr_joint`: exact joint curve/return likelihood with free `rho`.

The legacy model was not treated as the nested `rho=0` restriction because it omits the exact `eta^2 C / 2` convexity term. Likelihood-ratio tests were therefore performed only between the exact restricted and unrestricted models within each likelihood mode. Curve-only and joint raw likelihoods were not compared directly because joint mode contains an additional stock-return observation stream.

## Full-sample estimates

The exact correlated joint model produced:

| Parameter | Estimate | Approximate SE |
|---|---:|---:|
| $\kappa$ | 7.9667318 | 0.2760828 |
| $\theta$ | 0.0984465 | 0.0012850 |
| $\eta$ | 0.8073194 | 0.0274147 |
| $\rho$ | -0.0327659 | 0.0360505 |
| $\sigma_\varepsilon$ | 0.0207937 | 0.0002880 |
| $\mu$ | 0.1622803 | 0.1240559 |

The stock volatility was fixed at `0.25`. The carry-state half-life was 21.23 trading sessions. The latest filtered carry state on 2026-08-21 was 0.1616429, with filtered standard deviation 0.0225141.

All five full-sample specifications and all optimizer starts reported convergence. Numerical Hessians were stable. The unrestricted curve-only fit nevertheless retained a large gradient norm and an extremely flat likelihood ridge, so its point estimate must not be treated as a reliable calibration of correlation.

## Correlation identification

The main conclusion is that the data do not provide convincing evidence that `rho` differs from zero.

| Diagnostic | Curve-only | Joint curve/return |
|---|---:|---:|
| Point estimate of $\rho$ | -0.6420 | -0.03277 |
| Approximate SE | 2.1303 | 0.03605 |
| 95% profile-likelihood interval | Entire tested `[-0.9, 0.9]` range | `[-0.1029, 0.0378]` |
| LR statistic for $\rho=0$ | 0.1134 | 0.8245 |
| LR p-value | 0.7363 | 0.3639 |

The curve-only likelihood is effectively flat in `rho`. Its training-period estimate was -0.9744, and its four rolling estimates were all approximately -0.994 to -0.995. These are identification warnings, not evidence of a large negative economic correlation.

The joint likelihood is much more concentrated but still includes zero. Its four rolling estimates were approximately -0.377, -0.106, -0.112, and +0.130, showing material sign and sample instability. The empirical Pearson correlation between standardized curve innovations and stock-return residuals was 0.0077 with p-value 0.808; the Spearman correlation was 0.0533 with p-value 0.0938.

## Out-of-sample results

The evaluation split remained 2025-10-29. Parameters were estimated using only the training sample, and test observations were evaluated sequentially from prior filtered states.

| Model | Carry RMSE | Carry MAE | Futures RMSE | Futures MAE |
|---|---:|---:|---:|---:|
| Legacy curve | 402.70 bp | 297.11 bp | 70.76 | 51.89 |
| Exact `rho=0` curve | 402.63 bp | 297.08 bp | 70.89 | 51.95 |
| Correlated curve | 402.54 bp | 296.55 bp | 70.47 | 51.64 |
| Exact `rho=0` joint | 402.98 bp | 297.36 bp | 70.92 | 51.98 |
| Correlated joint | 407.54 bp | 300.93 bp | 71.28 | 52.32 |

The correlated curve model's improvement over the legacy model was economically negligible and accompanied by an unidentified boundary-like correlation estimate. The correlated joint model performed worse than its `rho=0` restriction and the legacy model. Thus out-of-sample evidence does not support adding nonzero correlation in the current one-factor specification.

Strong residual autocorrelation and volatility clustering remain under every specification. The one-factor maturity shape also remains unable to reproduce many observed humps and U-shapes; adding stock/carry correlation does not solve that structural curve limitation.

## Validation and deliverables

The project was run end-to-end in `D:\miniconda3\envs\GuoYuan\python.exe`. `pytest` was installed into that environment with Conda. Final validation produced:

- 16 passing tests in 172.56 seconds;
- analytical `B`, `C`, `D`, and `J` checks against numerical integration;
- stable small-argument tests;
- formula-reduction and exact Monte Carlo futures-pricing tests;
- joint covariance positive-semidefiniteness tests;
- ragged-curve and unequal-gap filter tests;
- filtered/smoothed state-separation tests;
- data-quality and session-count tests;
- seeded exact joint synthetic-recovery testing with fixed `sigma=0.25`;
- 21 parsed Python files;
- 22 readable, nonempty CSV outputs;
- 10 diagnostic charts;
- one JSON run summary.

Important files include:

- `im_corr_ou_1factor/README.md`
- `im_corr_ou_1factor/config.yaml`
- `im_corr_ou_1factor/analysis/run_workflow.py`
- `im_corr_ou_1factor/analysis/refine_profiles.py`
- `im_corr_ou_1factor/outputs/run_summary.json`
- `im_corr_ou_1factor/outputs/parameters.csv`
- `im_corr_ou_1factor/outputs/calibration_metrics.csv`
- `im_corr_ou_1factor/outputs/likelihood_ratio_tests.csv`
- `im_corr_ou_1factor/outputs/rho_profile_likelihood.csv`
- `im_corr_ou_1factor/outputs/rho_profile_confidence_intervals.csv`
- `im_corr_ou_1factor/outputs/rolling_parameters.csv`
- `im_corr_ou_1factor/outputs/standardized_innovations.csv`
- `im_corr_ou_1factor/outputs/charts/rho_profile_likelihood.png`

## Practical interpretation and next steps

Under fixed 25% stock volatility, adding a free stock/carry shock correlation does not materially improve the one-factor carry model. The joint likelihood rules out very large correlations but does not reject zero. The curve-only likelihood cannot identify correlation at all.

Reasonable follow-up work would be sensitivity analysis over fixed stock-volatility assumptions, maturity-dependent observation noise, or a future separation of historical `(kappa_P, theta_P)` from risk-neutral `(kappa_Q, theta_Q)`. The existing one-factor shape limitation should remain explicit when interpreting any such extension.

---

# Session Update - American Put on the Carry Curve

## Scope completed

An isolated pricing project was implemented at:

- `D:\Jupyter_files\Curry_curve_calibration\carry_put_pricing`

It prices the optional component described in `put_on_carry.md`, whose exercise payoff is

$$
G_t
=
S_t\left[
e^{(r-q_{0,T})(T-t)}
-\frac{F_{t,T}}{S_t}
\right]^+.
$$

The separate linear futures leg is not included in the reported option price.

## Agreed modelling choices

The implementation uses the following conventions agreed before coding:

- The carry state follows the calibrated independent two-factor OU model.
- The exact stochastic integrated-carry expression is used for future futures prices.
- The full-sample calibrated two-factor OU parameters are provisionally treated as risk-neutral parameters.
- The initial locked carry is inferred from the observed spot and futures quote rather than the model-fitted futures price.
- The latest filtered slow and fast factor states are used for the example.
- Exercise is allowed on every trading session, with one session equal to `1/244` year.
- The risk-free rate is `r=0.014` and the supplied GBM volatility is `sigma=0.25`.
- Spot and both carry-factor Brownian shocks are mutually independent.

The historically estimated OU dynamics are not automatically risk-neutral. Treating them as risk-neutral is an explicit prototype assumption rather than a resolved calibration result.

## Exact integrated-carry formula

Under

$$
c_t=\theta+x_{s,t}+x_{f,t},
$$

$$
dx_{j,t}=-\kappa_jx_{j,t}dt+\eta_jdW_t^j,
$$

and

$$
\frac{dS_t}{S_t}=(r-c_t)dt+\sigma dW_t^S,
$$

define

$$
I_{t,T}=\int_t^T c_u\,du.
$$

The carry integral is conditionally Gaussian, so the exact futures/forward ratio is

$$
\frac{F_{t,T}}{S_t}
=e^{r(T-t)}E_t^Q[e^{-I_{t,T}}]
=\exp\left(r\tau-m_I+\frac12v_I\right),
$$

where

$$
m_I
=\theta\tau
+A(\kappa_s,\tau)x_{s,t}
+A(\kappa_f,\tau)x_{f,t},
\qquad
A(\kappa,\tau)=\frac{1-e^{-\kappa\tau}}{\kappa},
$$

and

$$
v_I
=\sum_{j\in\{s,f\}}
\frac{\eta_j^2}{\kappa_j^2}
\left[
\tau
-\frac{2(1-e^{-\kappa_j\tau})}{\kappa_j}
+\frac{1-e^{-2\kappa_j\tau}}{2\kappa_j}
\right].
$$

Thus the exact annualized implied carry includes the Gaussian convexity correction:

$$
q_{t,T}^{\mathrm{exact}}
=\frac{m_I-\tfrac12v_I}{T-t}.
$$

## State reduction and pricing method

The payoff is homogeneous in the spot level:

$$
V(t,S,x_s,x_f)=S\,v(t,x_s,x_f).
$$

Zero Brownian-shock correlation alone does not make spot and carry levels independent, because the spot drift contains stochastic carry. The state reduction instead follows from payoff homogeneity. After normalization, the one-step continuation value is

$$
\frac{C_t}{S_t}
=E_t^Q\left[
e^{-\int_t^{t+\Delta t}c_u du}
v(t+\Delta t,X_{t+\Delta t})
\right].
$$

The spot level is therefore only a final scale factor, and the GBM volatility cancels from this particular price under the zero-correlation assumption. Volatility remains an explicit function input for interface clarity and future correlated extensions.

The primary numerical method is a two-dimensional deterministic backward induction over the slow and fast OU states:

1. Construct a rectangular slow/fast factor grid.
2. Use exact one-session OU transition and integrated-carry moments.
3. Apply Gaussian exponential tilting to the carry-weighted continuation expectation.
4. Evaluate the resulting expectation with tensor Gauss-Hermite quadrature and bilinear interpolation.
5. At each trading session, take the maximum of immediate exercise and continuation value.

This is conceptually similar to Longstaff-Schwartz because both solve the stopping problem backward. The difference is that this implementation calculates conditional continuation values from the known Gaussian transition law instead of estimating them with regressions on simulated paths.

The independent Gaussian transition and bilinear interpolation are separable. The implementation therefore applies successive one-dimensional slow and fast quadrature transforms rather than a slower explicit quadrature-order-squared loop.

Exercise at inception is fixed to zero by the contractual identity that the locked and prevailing futures quotes coincide at inception. Any small exact-model-versus-observed initial futures difference is reported as a model-fit diagnostic rather than converted into immediate exercise value.

## Agreed IM2609 example

The latest valid and most liquid near contract was selected from the cached 2026-08-21 curve:

| Input | Value |
|---|---:|
| Valuation date | 2026-08-21 |
| Contract | IM2609 |
| Expiry | 2026-09-18 |
| Trading sessions to expiry | 20 |
| $T$ | 0.0819672131 |
| $S_0$ | 7601.804 |
| Observed $F_{0,T}$ | 7527.0 |
| Locked $q_{0,T}$ | 0.1346461842 |
| $r$ | 0.014 |
| $\sigma$ | 0.25 |
| Latest slow state | +0.0599095163 |
| Latest fast state | -0.0161412562 |

The calibrated OU inputs were:

| Parameter | Value |
|---|---:|
| $\kappa_{slow}$ | 1.2409047966 |
| $\kappa_{fast}$ | 44.3295355882 |
| $\theta$ | 0.0826173761 |
| $\eta_{slow}$ | 0.0799282328 |
| $\eta_{fast}$ | 2.8520791536 |

At the initial filtered state, the exact stochastic-carry formula implies:

- model-implied carry: `0.1340041029`;
- model-implied futures: `7527.3961536`;
- model-minus-observed futures difference: `+0.3961536` points.

The small difference is retained as a diagnostic. The contractual locked carry remains the observed `0.1346461842`.

## Price and numerical convergence

The base numerical configuration uses:

- 301 slow-factor grid nodes;
- 401 fast-factor grid nodes;
- a six-stationary-standard-deviation half-width;
- Gauss-Hermite quadrature order 43.

The resulting American optional-component price is:

$$
\boxed{V_0=36.2794369}
$$

or `0.0047724773` per unit of spot, equal to approximately `0.477248%` of spot.

Grid convergence was:

| Grid | Slow nodes | Fast nodes | Price |
|---|---:|---:|---:|
| Coarse | 201 | 281 | 36.2944617 |
| Base | 301 | 401 | 36.2794369 |
| Fine | 401 | 501 | 36.2744292 |

The absolute base-minus-fine difference was `0.0050077` index points. At the base grid, nearby quadrature orders 39 through 47 spanned `0.0051714` points. These checks address numerical discretization only and do not measure economic model uncertainty.

## Validation and deliverables

Validation in `D:\miniconda3\envs\GuoYuan\python.exe` produced nine passing tests covering:

- stable exact OU integral loadings and variances;
- seeded integrated-carry moment simulation;
- the agreed initial exact-forward calculation;
- zero optional value for a one-session contract;
- invariance to spot volatility under the stated assumptions;
- linear scaling when spot and futures are scaled together;
- zero value under deterministic flat carry locked at the same rate;
- positive value and model-basis reporting for the agreed example.

No new Python packages were installed.

Important files are:

- `carry_put_pricing/README.md`
- `carry_put_pricing/pyproject.toml`
- `carry_put_pricing/src/carry_put_pricing/models.py`
- `carry_put_pricing/src/carry_put_pricing/analytics.py`
- `carry_put_pricing/src/carry_put_pricing/pricer.py`
- `carry_put_pricing/analysis/run_example.py`
- `carry_put_pricing/tests/test_analytics.py`
- `carry_put_pricing/tests/test_pricer.py`
- `carry_put_pricing/outputs/example_result.json`
- `carry_put_pricing/outputs/grid_convergence.csv`
- `carry_put_pricing/outputs/quadrature_convergence.csv`
- `carry_put_pricing/outputs/exercise_summary.csv`

## Remaining limitations and next steps

The largest unresolved issue is the pricing-measure interpretation. The calibrated two-factor OU transition parameters were estimated historically and are only provisionally used under $Q$. A production valuation should calibrate risk-neutral factor dynamics or specify carry-factor risk premia.

Other current limitations are:

- no spot/carry or slow/fast shock correlation;
- daily Bermudan exercise rather than mathematically continuous exercise;
- clipped interpolation at remote factor-grid boundaries;
- a constant risk-free rate;
- no separate treatment of the linear futures leg or its settlement mechanics.

Natural follow-up work is an independent Longstaff-Schwartz benchmark, risk-neutral sensitivity scenarios for the OU parameters, and extension to nonzero correlations if a defensible correlation specification becomes available.

# Session Update — 2026-08-26 — Configurable Carry-Put Demo and Boundary Diagnostics

## Scope completed

Today we turned the carry-put example into a configurable demonstration, investigated the fast-factor parameter boundaries, extended the Demo's calendar handling to 2027-dated contracts, and clarified how the main two-factor project separates final estimation from rolling diagnostics.

The user-facing workflow, calendar, and pricing changes were made under:

- `D:\LuJingjian\Jupyter_files\GuoYuan\Studies\carry_rate\Demo`

The only shared-estimator change was a backward-compatible optional `eta_fast_upper_bound` argument whose default remains 3; the Demo passes 6 explicitly.

The agreed valuation scope remains the American carry-put **optional component only**. The separate linear futures leg is deliberately excluded because the option component is the part of interest for this product.

## Flexible Demo inputs

The Demo now accepts three user inputs:

- valuation date;
- calibration sample size in usable trading dates;
- underlying IM futures contract.

The sample size includes the valuation date. For example, a sample size of 250 selects the 250 accepted curve dates ending on the requested valuation date and supplies 249 close-to-close spot returns.

The inputs are exposed both in `Carry_Put_Demo.ipynb` and through `demo_workflow.py`, for example:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B demo_workflow.py `
  --valuation-date 2026-08-10 `
  --sample-size 250 `
  --futures-contract IM2612
```

The selected CFFEX contract's expiry is inferred from its contract code and checked against the quote used for the locked carry. Calibration carries, historical spot volatility, the filtered state, the locked carry, and option maturity are all aligned to the same requested valuation date and sample.

## Fast-factor volatility recalibration and profile

The earlier Demo calibration placed `eta_fast` at its upper bound of 3. We therefore made the estimator's fast-volatility cap configurable while preserving 3 as the main project's backward-compatible default. The Demo recalibrates with a cap of 6.

After the cap-6 calibration, the Demo runs a fixed-`eta_fast` profile. At every profile value, the other five parameters are re-optimized and the option is repriced. This is a proper profile-likelihood calculation rather than a sensitivity calculation that freezes all other fitted parameters.

For the current 2026-08-10 / 488-date / IM2612 output:

| Item | Result |
|---|---:|
| Sample | 2024-08-05 to 2026-08-10 |
| Curve dates | 488 |
| Accepted observations | 1,807 |
| `kappa_slow` | 2.57648937 |
| `kappa_fast` | 62.57648937 |
| `eta_slow` | 0.11495544 |
| `eta_fast` | 3.88517353 |
| `theta` | 0.10862242 |
| `sigma_epsilon` | 0.00592795 |
| Log likelihood | 5348.42992969 |

The fast volatility is now interior under the cap of 6. Relative to fixing `eta_fast` at 3, the optimized cap-6 result gains approximately 12.19 log-likelihood units. The grid-based 95% profile-supported region is approximately 3.75 to 4.00.

The mean-reversion gap `kappa_fast - kappa_slow`, however, reaches its separate upper bound of 60. Thus increasing the volatility cap resolved the original `eta_fast` boundary but revealed continuing weak identification or pressure toward still-faster mean reversion. This is a model-risk warning, especially for very short-dated options, but it is not by itself evidence that the optimizer failed.

## IM2612 option result

We changed the option example from the short-dated IM2609 contract to IM2612 so that the option is less dominated by the least stable very-fast carry dynamics.

For valuation on 2026-08-10:

| Input or result | Value |
|---|---:|
| Contract | IM2612 |
| Inferred expiry | 2026-12-18 |
| Sessions to expiry | 88 |
| Spot | 7733.9 |
| Observed futures | 7413.2 |
| Locked carry | 0.1314279619 |
| Historical spot volatility | 0.2771085586 |
| Option-only price | 63.8575280 |
| Model-minus-observed initial futures | +8.7164022 |

The fixed-`eta_fast` likelihood-supported grid maps to option prices of approximately 62.60 to 65.58. This interval measures the effect of the profiled fast-volatility uncertainty on the option within the fitted model; it is not a full valuation confidence interval.

The base-versus-fine-grid price difference is approximately 0.04413 index points, and quadrature orders 39 through 47 span approximately 0.00191 points.

## Contract expiry and 2027 calendar handling

The pricing workflow now accepts different IM contracts and automatically infers the standard CFFEX expiry date as the third Friday of the contract month, subject to calendar adjustment and validation against the available quote.

We found that the installed `chinese_calendar` package supports dates only through 2026. The shared `im_2factor_ou_carry` calendar catches this limitation and silently falls back to ordinary weekdays for 2027, which would overcount sessions around Chinese holidays. From 2026-08-21 to IM2703 expiry, that fallback returned 144 sessions.

The Demo no longer uses the weekday fallback. It contains an explicit company exchange calendar for 2027–2028 and raises `CalendarCoverageError` outside covered years. Under this calendar:

- IM2703 expiry is inferred as 2027-03-19;
- there are 138 sessions from 2026-08-21 to expiry;
- there are 147 sessions from 2026-08-10 to expiry;
- the corrected session count is used in both calibration maturities and option pricing.

The 2027–2028 company calendar is provisional and should be confirmed or replaced when the official CFFEX holiday schedule becomes available.

## Notebook module-cache fix

Running the notebook after editing `demo_workflow.py` initially produced:

```text
run_demo() got an unexpected keyword argument 'futures_contract'
```

The function on disk already accepted that argument. The error came from Jupyter retaining an older imported module in the live kernel. The notebook setup cell now removes and reimports all Demo-local modules before calling the workflow, including `demo_workflow`, `profile_analysis`, `option_pricing`, `calibration`, `demo_quality`, and `calendar_utils`.

The current notebook inputs were preserved as:

```python
VALUATION_DATE = "2026-08-10"
SAMPLE_SIZE = 488
FUTURES_CONTRACT = "IM2612"
```

The setup cell and subsequent cells should be rerun to synchronize all embedded notebook outputs with these inputs. A complete fixed-eta profile can take several minutes.

## Main-project calibration comparison

We inspected `im_2factor_ou_carry` because its headline parameter estimates did not appear to hit bounds. The full-sample setup uses 991 curve dates from 2022-07-22 through 2026-08-21, 3,667 accepted observations, and 12 optimizer starts. Its estimates are interior:

- `kappa_slow = 1.24090`;
- `kappa_fast = 44.32954`;
- mean-reversion gap `= 43.08863 < 60`;
- `eta_fast = 2.85208 < 3`.

All 12 optimizer starts converge to the same full-sample solution and the numerical Hessian is stable.

The rolling diagnostics tell a different story. They use five overlapping 488-date windows, spaced by 126 dates with the latest endpoint appended. In four of the five windows, `eta_fast` reaches the original cap of 3. Therefore the boundary pressure is concentrated in more recent rolling samples or regimes rather than being a general failure of the estimator.

The calibration roles were clarified as follows:

1. The **full-sample calibration** produces the main project's final reported parameters. Its two-factor optimization uses 12 starting guesses.
2. The **rolling-window calibrations** are diagnostics for stability and boundary behavior. Each rolling window uses four optimizer starts and retains the best likelihood solution; those estimates do not overwrite or influence the full-sample parameters.

With five windows and four starts per window, the rolling two-factor diagnostic performs 20 complete optimization searches. The pipeline also contains train/test benchmark refits, but these belong to model evaluation rather than final parameter selection.

## Validation and current status

The Demo's five focused tests pass, Ruff passes, and the notebook structure is valid. Earlier established checks also included 12 calibration tests and nine pricing tests. No new Python package was installed.

The main Demo outputs are stored under `Demo\outputs`, including the JSON summary, parameter tables, fixed-eta likelihood/price profile, convergence diagnostics, and charts.

The clean next diagnostic, if needed, is to rerun an identical cap-6 specification over 244-, 488-, 732-, and 991-date windows using the same calendar and optimizer-start policy. This would show directly whether the remaining mean-reversion-gap boundary relaxes as the estimation history lengthens and how much that changes the IM2612 option price.
