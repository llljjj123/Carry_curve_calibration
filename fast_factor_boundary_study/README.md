# Fast-factor boundary study

This folder diagnoses why `kappa_fast - kappa_slow` reaches its upper bound in
the two-factor OU carry calibration. It leaves the production model, root
README, Demo, and generated Demo outputs unchanged.

The study reuses:

- the strict Demo exchange calendar and cached market-data preparation;
- the validated two-factor Kalman likelihood and filtering implementation;
- the validated American carry-put pricing engine and both curve deltas.

The only duplicated modelling code is the optimizer wrapper needed to replace
the production estimator's hard-coded gap cap of 60 and total-fast-kappa guard
of 80 with explicit diagnostic settings.

## Default design

All comparisons use valuation date `2026-08-21`, contract `IM2612`,
`eta_fast <= 6`, 12 optimizer starts, and cached raw market data.

The runner performs 10 unique free-gap fits:

1. Gap-cap sensitivity at 488 dates: caps 60, 90, 120, and 180.
2. Window sensitivity at cap 180: 244, 488, 732, and 991 dates.
3. Short-end sensitivity at 488 dates and cap 180: exclude observations with
   sessions to expiry less than or equal to 5, 10, 15, and 20.

It then profiles fixed kappa gaps from 20 through 240, with a five-unit grid
around the likelihood peak. At each profile point, the other five parameters
are re-optimized and the `IM2612` option is repriced.
Every fit is checkpointed as JSON and CSV, so an interrupted run resumes.

Hard clipping or winsorizing observed carry is deliberately not included in
this first study because it could create artificial mean reversion. The
short-end exclusion diagnostic tells us whether a richer observation-noise
model should be the next implementation.

## Run in `spyder-env`

From the repository root:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B `
  fast_factor_boundary_study\study.py
```

Add `--with-oos` to estimate parameters on the first 80% of each scenario's
dates and calculate sequential one-step forecast errors on the last 20%. This
roughly doubles calibration work and is therefore separate from the initial
boundary diagnosis.

For a quick mechanical smoke run without option pricing:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B `
  fast_factor_boundary_study\study.py `
  --windows 244 --gap-caps 60 90 --short-end-cutoffs 5 `
  --profile-gaps 40 60 90 --optimizer-starts 2 --profile-starts 1 `
  --optimizer-maxiter 100 --skip-pricing `
  --output-dir fast_factor_boundary_study\smoke_outputs
```

## Outputs

The default `outputs` folder contains:

- `all_scenarios.csv`;
- `cap_sensitivity.csv`;
- `window_sensitivity.csv`;
- `short_end_sensitivity.csv`;
- `fixed_gap_profile.csv`;
- `study_summary.json`;
- optimizer audits and resumable checkpoints under `fits` and `profile_fits`;
- four diagnostic charts under `charts`.

`posterior_*` errors are same-date filtered fit diagnostics.
`recursive_prior_*` errors use each day's prior state but parameters estimated
from the whole scenario. Only `oos_*` columns from `--with-oos` are strict
chronological holdout diagnostics.
