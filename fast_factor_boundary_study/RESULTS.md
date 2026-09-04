# Initial fast-factor boundary results

## Run definition

- Environment: `D:\miniforge3\envs\spyder-env\python.exe`
- Valuation date: 2026-08-21
- Priced contract: IM2612, 79 trading sessions to expiry
- Raw data: existing cache only; no refresh
- Calendar: strict Demo calendar
- Fast-volatility bound: `eta_fast <= 6`
- Optimizer policy: 12 starts for each free fit and 4 starts for each fixed-gap profile point
- Pricing: optional carry-put component only, using the validated base grid

The production calibration, Demo, root README, and their existing outputs were
not modified.

## Main answer

The original `kappa_fast - kappa_slow <= 60` constraint is restrictive for the
recent 488-date sample, but the likelihood does **not** keep improving as the
bound is raised indefinitely.

| Gap cap | Estimated gap | `eta_fast` | Log likelihood | Option price |
|---:|---:|---:|---:|---:|
| 60 | 60.0000 | 3.9274 | 5349.9250 | 69.8248 |
| 90 | 69.6799 | 4.5069 | 5351.2262 | 67.7558 |
| 120 | 69.6798 | 4.5069 | 5351.2262 | 67.7558 |
| 180 | 69.6798 | 4.5069 | 5351.2262 | 67.7559 |

All 12 starts converged to the same solution in the cap-90, cap-120, and
cap-180 reference fits. The cap-60 restriction lowers log likelihood by about
1.30 and increases the option price by about 2.07 points. A cap of 90 is enough
for this sample; 120 provides more diagnostic headroom without changing the
optimum.

At the cap-180 optimum:

- `kappa_slow = 2.6790`;
- `kappa_fast = 72.3588`;
- fast half-life is about 2.34 trading sessions;
- option price is 67.7559 points;
- slow futures-equivalent delta is -0.40020;
- fast futures-equivalent delta is -0.01829.

## Fixed-gap profile

The fixed-gap likelihood peaks at 70, matching the free estimate of 69.68.
The grid-supported 95% likelihood-ratio region is 60–90. Linear interpolation
between the adjacent profile points gives an approximate region of 58.8–92.1.

Across the supported grid points:

- option price ranges from 64.9067 to 69.8248 points;
- slow delta ranges from -0.40145 to -0.39736;
- fast delta ranges from -0.02405 to -0.01046.

At fixed gaps of 95 and above, `eta_fast` reaches its separate cap of 6. The
gap-95 point is already outside the 95% region, so the local gap conclusion is
not created by that second cap. Conclusions about extreme gaps remain
conditional on `eta_fast <= 6`.

## Sample-window sensitivity

| Curve dates | Estimated gap | `kappa_fast` | `eta_fast` | Option price | Fast delta |
|---:|---:|---:|---:|---:|---:|
| 244 | 68.2322 | 70.0166 | 4.0074 | 61.5142 | -0.01423 |
| 488 | 69.6798 | 72.3588 | 4.5069 | 67.7559 | -0.01829 |
| 732 | 42.5596 | 43.6474 | 3.1334 | 83.0786 | -0.04396 |
| 991 | 42.9034 | 44.1260 | 2.8405 | 74.6792 | -0.04432 |

Recent 244/488-date samples consistently prefer a gap near 69, while the
732/991-date samples prefer a gap near 43. The fast-factor timescale is
therefore regime/sample dependent. The resulting option-price range of
61.51–83.08 points is much wider than the roughly two-point cap effect.

Raw likelihoods are not compared across windows because the observation counts
differ.

## Short-end sensitivity

| Exclude sessions <= | Observations | Estimated gap | `eta_fast` | Option price | Fast delta |
|---:|---:|---:|---:|---:|---:|
| 5 | 1,806 | 69.6798 | 4.5069 | 67.7559 | -0.01829 |
| 10 | 1,687 | 92.3176 | 6.0000 | 64.9676 | -0.00942 |
| 15 | 1,567 | 16.8295 | 1.3229 | 85.5001 | -0.12845 |
| 20 | 1,446 | 15.2622 | 1.2132 | 85.7487 | -0.14722 |

Removing the nearest observations does not produce a monotonic stabilization.
The cutoff-10 case is weakly multimodal: one of 12 starts found the reported
gap-92/eta-cap solution, while 11 found a gap near 38.06 whose likelihood was
only 0.579 lower. All starts agreed at cutoffs 5, 15, and 20.

This sharp change means contracts around 11–15 sessions to expiry contain
substantial information about the fast factor. Deleting them changes the
economic model rather than merely removing harmless outliers. Hard clipping or
wholesale exclusion is therefore not recommended as the permanent solution.

Raw likelihoods are not compared across cutoffs because each cutoff uses a
different number of observations.

## Recommendation

For experimentation, use a gap cap of 120 rather than 60. Do not yet change the
production calibration solely on that basis, because sample selection and
front-end treatment dominate the option-value uncertainty.

The next model change should be a maturity-dependent observation-noise model
that keeps all currently accepted observations. A useful first comparison is:

1. current common `sigma_epsilon`;
2. two noise buckets, with a separate standard deviation for contracts with
   15 or fewer sessions remaining;
3. a smooth maturity-dependent noise function;
4. optionally, a robust likelihood for isolated observation outliers.

These specifications should be compared on identical observations using
chronological out-of-sample forecasts. The runner already supports the current
model's stricter 80/20 diagnostic through `--with-oos`; the same forecast
convention should be used for the alternative noise models.

## Result files

- `outputs/cap_sensitivity.csv`
- `outputs/window_sensitivity.csv`
- `outputs/short_end_sensitivity.csv`
- `outputs/fixed_gap_profile.csv`
- `outputs/study_summary.json`
- `outputs/charts/`
- Per-start optimizer audits under `outputs/fits/` and `outputs/profile_fits/`
