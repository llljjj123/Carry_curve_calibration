# Maturity-dependent observation-noise results

> Follow-up: the recommended multiple-cut-date test has now been completed.
> It supports advancing constant log-futures noise to integration behind a
> configuration switch. See `MULTI_CUT_RESULTS.md` for the five-window results.

## Conclusion

Use **constant log-futures noise** as the leading specification for the next
stage of validation.

It removes the extreme fast-factor estimate seen under constant carry noise,
has the best out-of-sample predictive log score and full-sample BIC, improves
both carry and futures forecast errors, and uses one fewer parameter than the
numerically equivalent smooth model. The two-bucket model remains a useful
diagnostic, but its small advantage in carry RMSE does not outweigh the direct
model's results on the other metrics.

This study is deliberately isolated from the production calibration. Nothing
under `Demo`, `im_2factor_ou_carry`, or `carry_put_pricing` was changed.

## Test design

- Sample: 991 curve dates and 3,667 observations from 2022-07-22 through
  2026-08-21.
- Training period: 792 dates and 2,931 observations through 2025-10-29.
- Holdout period: 199 dates and 736 observations after 2025-10-29.
- All models use exactly the same dates and contracts.
- Parameters are estimated only on the training sample and then frozen.
- The Kalman state is updated sequentially, but each test day's predictions
  are recorded before that day's observations update the state.
- Full-sample fits are separate and are used for AIC/BIC and IM2612 valuation
  sensitivity.
- The direct log-futures likelihood includes the exact `sum(log(tau))`
  Jacobian adjustment, putting its likelihood and predictive score in the
  same carry-observation units as the other models.

## Out-of-sample performance

Lower is better for RMSE and MAE; higher is better for the log predictive
score.

| Model | Carry RMSE (bp) | Carry MAE (bp) | Futures RMSE (points) | Futures MAE (points) | Mean log score |
|---|---:|---:|---:|---:|---:|
| Constant carry | 359.891 | 213.856 | 33.354 | 25.135 | 2.890140 |
| Two buckets | **344.504** | 207.520 | 32.245 | 23.992 | 3.059324 |
| Smooth carry + log-price | 346.070 | 205.611 | **31.801** | 23.426 | 3.219316 |
| Constant log-futures | 346.067 | **205.610** | 31.801 | **23.426** | **3.219320** |

Relative to the constant-carry baseline, constant log-futures noise improves:

- carry RMSE by 3.84%;
- futures RMSE by 4.66%;
- carry MAE by 3.86%;
- futures MAE by 6.80%;
- total holdout log predictive score by 242.277.

The two-bucket model's overall carry RMSE is 1.563 bp lower than the direct
log-futures model, a difference of only 0.45%. The direct model performs better
on carry MAE, futures RMSE/MAE, and predictive log score.

The maturity buckets explain why the specifications differ. Under constant
carry noise, the fitted noise is 59.85 bp at every maturity, which is too small
for the shortest contracts and becomes an increasingly large futures-price
error allowance at long maturities. Constant log-futures noise instead implies
a carry-noise standard deviation proportional to `1 / tau` while keeping the
approximate futures-price noise nearly constant at 7.29 points.

For example, the full-sample constant log-futures fit implies:

| Sessions to expiry | Carry-noise SD (bp) | Approx. futures-noise SD at F=7,500 |
|---:|---:|---:|
| 10 | 237.07 | 7.29 points |
| 21 | 112.89 | 7.29 points |
| 63 | 37.63 | 7.29 points |
| 126 | 18.81 | 7.29 points |
| 252 | 9.41 | 7.29 points |

## Full-sample fit and boundary diagnosis

| Model | Parameters | Kappa gap | Log likelihood | BIC | IM2612 option price |
|---|---:|---:|---:|---:|---:|
| Constant carry | 6 | 42.9034 | 10,857.383 | -21,665.524 | 74.679 |
| Two buckets | 7 | 17.7756 | 11,285.011 | -22,512.573 | 86.163 |
| Smooth carry + log-price | 7 | 16.2033 | 11,673.612 | -23,289.775 | 86.764 |
| Constant log-futures | 6 | **16.2033** | **11,673.612** | **-23,297.982** | **86.764** |

All four solutions are below the widened kappa-gap cap of 120 and the
eta-fast cap of 6. More importantly, allowing maturity-dependent noise reduces
the kappa gap by 59%-62% relative to constant carry noise. Under the recommended
model:

$$
\kappa_{slow}=0.2886,\qquad
\kappa_{fast}=16.4918,\qquad
\kappa_{fast}-\kappa_{slow}=16.2033.
$$

The full-sample log-likelihood rises by 816.229 relative to constant carry
noise. This is far too large to be explained by model complexity, and BIC
strongly prefers constant log-futures noise.

The smooth model estimates

$$
\sigma_{floor}=8.58\times10^{-7},\qquad
\sigma_{logF}=9.716\times10^{-4}.
$$

Its floor is effectively zero. Its OU parameters, likelihood, forecasts, and
option value therefore coincide with the simpler direct log-futures model.
This nested-model result is a strong internal consistency check.

The observation-noise choice is economically material. The IM2612 optional
component rises from 74.679 under constant carry noise to 86.764 under constant
log-futures noise, an increase of 12.085, or about 16.2%. Fast-factor pathwise
delta also changes from -0.0443 to -0.1475.

## Optimizer audit

Every one of the 12 starts for every train/full fit reported convergence. The
best likelihood was reproduced within 0.01 by 7-10 starts depending on the
fit. Some other starts converged to clearly inferior local modes, confirming
that the likelihood is multimodal and that multi-start estimation remains
necessary. The selected optimum is nevertheless repeatedly recovered rather
than coming from a single lucky start.

## Recommendation and next check

Promote constant log-futures noise to a candidate production calibration, but
do not replace the baseline yet. The next useful check is a rolling or multiple
cut-date out-of-sample exercise. The present result uses one contiguous
199-date holdout, so repeated cut dates will show whether the improvement and
the kappa-gap stabilization persist across market regimes.

If that check passes, integrate the direct log-futures observation equation
behind a configuration switch, rerun downstream option valuation and hedge
sensitivity, and only then change the production default.

## Reproduce

From the repository root in `spyder-env`:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B `
  maturity_noise_study\study.py
```

Completed train/full fits are checkpointed under `outputs/fits`, so the command
resumes rather than repeating finished optimization.
