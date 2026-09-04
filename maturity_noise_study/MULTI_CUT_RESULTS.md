# Multiple-cut-date robustness results

## Decision

The multiple-cut-date test **supports advancing constant log-futures noise to
production integration behind a configuration switch**.

The candidate wins futures RMSE and predictive log score in all five holdout
windows, improves every pooled error metric relative to constant carry noise,
and keeps both the kappa gap and `eta_fast` comfortably away from their bounds.
The evidence is strong enough to proceed with implementation, although the
existing production default should remain available for parallel comparison
during the first rollout.

## Design

- Complete panel: 991 curve dates through 2026-08-21.
- Models: constant carry, two carry-noise buckets, and constant log-futures.
- Five expanding training samples.
- Five consecutive, non-overlapping holdouts of 120 curve dates each.
- The holdouts cover the final 600 curve dates exactly once, from 2024-03-05
  through 2026-08-21.
- Parameters are frozen within each holdout.
- Each day's curve is predicted and scored before it updates the Kalman state.
- Every fit uses the same observations, 12 optimizer starts, kappa-gap cap 120,
  and `eta_fast` cap 6.
- Daily paired comparisons use Newey-West/HAC standard errors with five lags.

| Window | Training cut | Test period | Training dates |
|---:|---:|---:|---:|
| 1 | 2024-03-04 | 2024-03-05 to 2024-08-27 | 391 |
| 2 | 2024-08-27 | 2024-08-28 to 2025-03-03 | 511 |
| 3 | 2025-03-03 | 2025-03-04 to 2025-08-25 | 631 |
| 4 | 2025-08-25 | 2025-08-26 to 2026-02-27 | 751 |
| 5 | 2026-02-27 | 2026-03-02 to 2026-08-21 | 871 |

## Pooled out-of-sample results

The pooled sample contains 600 test dates and 2,219 curve observations for
each model.

| Model | Carry RMSE (bp) | Carry MAE (bp) | Futures RMSE | Futures MAE | Mean log score |
|---|---:|---:|---:|---:|---:|
| Constant carry | 385.473 | 207.341 | 28.437 | 20.360 | 2.786522 |
| Two buckets | 365.545 | 201.148 | 27.389 | 19.383 | 2.910991 |
| Constant log-futures | **362.546** | **198.800** | **26.846** | **18.885** | **3.097327** |

Relative to constant carry, the candidate improves:

- carry RMSE by 5.95%;
- carry MAE by 4.12%;
- futures RMSE by 5.59%;
- futures MAE by 7.25%;
- mean predictive log score by 0.311 per observation.

Relative to two buckets, it improves carry RMSE by 0.82%, carry MAE by 1.17%,
futures RMSE by 1.98%, futures MAE by 2.57%, and mean log score by 0.186 per
observation.

## Consistency across windows

| Window | Model | Carry RMSE (bp) | Futures RMSE | Mean log score | Kappa gap |
|---:|---|---:|---:|---:|---:|
| 1 | Constant carry | **277.934** | 19.608 | 2.5528 | 35.620 |
| 1 | Two buckets | 280.840 | 17.377 | 2.6323 | 21.541 |
| 1 | Constant log-futures | 278.600 | **16.677** | **3.0228** | **18.452** |
| 2 | Constant carry | 411.716 | 27.018 | 2.8935 | 37.379 |
| 2 | Two buckets | 374.388 | 26.871 | 2.9124 | 17.820 |
| 2 | Constant log-futures | **368.627** | **26.183** | **3.0174** | **14.788** |
| 3 | Constant carry | 458.978 | 26.786 | 2.7875 | 40.888 |
| 3 | Two buckets | 433.508 | 25.664 | 2.9683 | 19.716 |
| 3 | Constant log-futures | **425.858** | **25.279** | **3.1089** | **16.469** |
| 4 | Constant carry | 401.295 | 34.835 | 2.7302 | 44.232 |
| 4 | Two buckets | **381.757** | 33.259 | 3.0336 | 17.430 |
| 4 | Constant log-futures | 384.968 | **33.143** | **3.1574** | **15.447** |
| 5 | Constant carry | 352.902 | 31.585 | 2.9688 | 41.789 |
| 5 | Two buckets | 339.742 | 31.014 | 3.0084 | 16.801 |
| 5 | Constant log-futures | **337.859** | **30.038** | **3.1800** | **15.838** |

Constant log-futures wins:

- carry RMSE in 3 of 5 windows;
- futures RMSE in 5 of 5 windows;
- predictive log score in 5 of 5 windows.

Its one carry-RMSE loss to constant carry is only 0.67 bp in window 1. In
window 4, two buckets beats it by 3.21 bp of carry RMSE, while constant
log-futures still has the better carry MAE, futures RMSE/MAE, and log score.

## Paired daily inference

Positive gains mean that the candidate performs better. The table reports
two-sided Newey-West/HAC tests across the 600 daily paired observations.

| Reference | Candidate metric | Candidate win rate | HAC p-value | Interpretation |
|---|---|---:|---:|---|
| Constant carry | Carry squared error | 53.3% | 0.0505 | Borderline at 5% |
| Constant carry | Carry absolute error | 54.2% | 0.000038 | Significant gain |
| Constant carry | Futures squared error | 67.8% | 0.000010 | Significant gain |
| Constant carry | Futures absolute error | 64.7% | <0.000001 | Significant gain |
| Constant carry | Mean log score | 80.8% | <0.000001 | Significant gain |
| Two buckets | Carry squared error | 52.5% | 0.1013 | Not statistically decisive |
| Two buckets | Carry absolute error | 55.3% | 0.0269 | Significant gain |
| Two buckets | Futures squared error | 61.2% | 0.0030 | Significant gain |
| Two buckets | Futures absolute error | 58.7% | 0.000069 | Significant gain |
| Two buckets | Mean log score | 79.5% | 0.000006 | Significant gain |

The RMSE-related carry evidence is weaker because a few very large short-end
errors dominate squared carry loss. The candidate's advantage is much clearer
for absolute carry error, both futures losses, and the full predictive density.

## Parameter stability

| Model | Mean gap | Minimum | Maximum | Gap-bound hits | `eta_fast`-bound hits |
|---|---:|---:|---:|---:|---:|
| Constant carry | 39.982 | 35.620 | 44.232 | 0 | 0 |
| Two buckets | 18.662 | 16.801 | 21.541 | 0 | 0 |
| Constant log-futures | **16.199** | **14.788** | **18.452** | **0** | **0** |

The candidate's `eta_fast` estimates range from 1.171 to 1.515, far below the
cap of 6. Its mean kappa gap of 16.199 is essentially identical to the earlier
991-date full-sample estimate of 16.203. The stabilization is therefore not
specific to the original 2025-10-29 cut date.

All 12 starts in all 15 fits reported optimizer convergence. The selected
likelihood was reproduced within 0.01 by 7-9 starts in every fit, so no selected
solution depends on one isolated start.

## Recommendation

The predefined robustness requirements are met:

- predictive log score improves in every window;
- futures RMSE improves in every window;
- all pooled error metrics improve;
- there is no severe adverse window;
- the kappa gap is stable and never approaches its bound;
- `eta_fast` remains interior;
- the direct model remains more parsimonious than smooth carry-plus-log noise.

The next implementation step is to add constant log-futures observation noise
to the production calibration behind an explicit configuration option. First
run the existing constant-carry and new log-futures versions side by side,
compare downstream option values and deltas, and retain an immediate fallback.
After that parallel validation, make log-futures noise the default if no
integration discrepancy appears.

## Files

- `multi_cut_study.py`: reproducible runner.
- `outputs/multi_cut/model_summary.csv`: pooled results and window wins.
- `outputs/multi_cut/window_metrics.csv`: per-window errors and parameters.
- `outputs/multi_cut/paired_daily_comparisons.csv`: paired HAC results.
- `outputs/multi_cut/metrics_by_maturity.csv`: maturity-bucket results.
- `outputs/multi_cut/daily_metrics.csv`: daily loss and score series.
- `outputs/multi_cut/oos_predictions.csv`: raw holdout predictions.
- `outputs/multi_cut/fits/`: checkpoints and per-start optimizer audits.
- `outputs/multi_cut/charts/`: performance and parameter-stability charts.

## Reproduce

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B `
  maturity_noise_study\multi_cut_study.py
```
