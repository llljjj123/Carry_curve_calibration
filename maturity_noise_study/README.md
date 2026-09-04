# Maturity-dependent observation-noise study

This folder compares four two-factor OU observation specifications while
leaving the production calibration, Demo, and root documentation unchanged.

## Compared models

1. `constant_carry`: the current baseline

   $$
   \sigma_q(\tau)=\sigma_q.
   $$

2. `two_bucket_carry`: a simple diagnostic split

   $$
   \sigma_q(\tau)=
   \begin{cases}
   \sigma_{short}, & \text{sessions}\le 15,\\
   \sigma_{long}, & \text{sessions}>15.
   \end{cases}
   $$

3. `smooth_carry_log`: a carry-noise floor plus amplified log-price noise

   $$
   \sigma_q(\tau)=
   \sqrt{\sigma_{floor}^2+
   \left(\frac{\sigma_{logF}}{\tau}\right)^2}.
   $$

4. `constant_log_futures`: direct filtering of
   `log(F/S) - r*tau` with constant log-futures noise. In carry units this is

   $$
   \sigma_q(\tau)=\frac{\sigma_{logF}}{\tau}.
   $$

The direct log-futures likelihood is adjusted by the exact change-of-variables
Jacobian, `sum(log(tau))`, before comparing likelihoods, AIC, BIC, or predictive
scores with the carry-space models.

## Fair out-of-sample design

The default run uses all 991 accepted curve dates through 2026-08-21. Every
model has the same split and observations:

- first 792 dates: parameter estimation;
- final 199 dates: sequential one-day-ahead testing;
- the OU parameters remain frozen throughout the test;
- each test curve updates the states only after that day's predictions have
  been recorded.

The study reports aggregate and maturity-bucket carry/futures RMSE, MAE, bias,
standardized innovation RMS, and carry-unit log predictive scores. Full-sample
fits are used separately for information criteria and `IM2612` option-price
sensitivity.

## Run in `spyder-env`

From the repository root:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B `
  maturity_noise_study\study.py
```

Each train/full fit is checkpointed, so an interrupted run resumes safely.

## Multiple-cut-date robustness test

The production-candidate check compares constant carry, two-bucket carry, and
constant log-futures noise over five expanding calibrations followed by
non-overlapping 120-date holdouts. The final 600 curve dates are covered once,
so pooled results do not double-count test observations.

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B `
  maturity_noise_study\multi_cut_study.py
```

Fits resume from `outputs/multi_cut/fits`. The runner exports per-window,
pooled, maturity-bucket, daily paired/HAC, parameter-stability, and raw
prediction results under `outputs/multi_cut`.

The completed interpretation and production recommendation are in
`MULTI_CUT_RESULTS.md`.

Quick mechanical smoke run:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B `
  maturity_noise_study\study.py `
  --sample-dates 80 --optimizer-starts 2 --optimizer-maxiter 100 `
  --skip-pricing --output-dir maturity_noise_study\smoke_outputs
```

## Outputs

The interpreted findings and recommendation are in `RESULTS.md`.

- `oos_aggregate_metrics.csv`
- `oos_metrics_by_maturity.csv`
- `oos_predictions.csv`
- `full_sample_model_comparison.csv`
- `fitted_noise_curves.csv`
- `study_summary.json`
- per-start optimizer audits and resumable fit checkpoints under `fits/`
- comparison charts under `charts/`
