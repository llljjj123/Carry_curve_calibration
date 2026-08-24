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
