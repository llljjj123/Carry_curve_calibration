# Constant log-futures production integration

## Decision

The validated `constant_log_futures` observation equation is now available in
the production calibration, Demo, and pricing-example workflow behind an
explicit configuration switch. `constant_carry` remains the default.

The implementation passes the full production workflow and downstream pricing
checks. The candidate continues to improve out-of-sample curve errors and
removes the fast-factor boundary problem, but it also changes option values and
hedge ratios materially. Keep both configurations available and review the
economic pricing impact before changing the default.

## Configuration

The historical production behavior remains:

```yaml
estimation:
  observation_noise_model: constant_carry
  kappa_gap_upper_bound: 60.0
  eta_fast_upper_bound: 3.0
```

The candidate is in `config_log_futures.yaml` and writes to
`outputs_log_futures/`:

```yaml
estimation:
  observation_noise_model: constant_log_futures
  kappa_gap_upper_bound: 120.0
  eta_fast_upper_bound: 6.0
```

The pricing engine itself is unchanged. Only the calibrated OU parameters and
filtered states passed into it differ.

## Production calibration result

The full run used 3,667 accepted observations on 991 curve dates through
2026-08-21.

| Parameter | Constant carry | Constant log-futures |
|---|---:|---:|
| `kappa_slow` | 1.240905 | 0.341280 |
| `kappa_fast` | 44.329536 | 16.733586 |
| Kappa gap | 43.088631 | 16.392307 |
| `theta` | 0.082617 | 0.034420 |
| `eta_slow` | 0.079928 | 0.046913 |
| `eta_fast` | 2.852079 | 1.240254 |
| Native observation SD | `sigma_epsilon = 0.005970` | `sigma_log_futures = 0.000974` |

The candidate gap and `eta_fast` are comfortably inside the relaxed bounds.
The numerical Hessian is stable and the model-specific standard errors are
exported successfully.

On the production 20% evaluation holdout, the candidate changes the two-factor
errors as follows:

| Metric | Constant carry | Constant log-futures | Improvement |
|---|---:|---:|---:|
| Carry RMSE (bp) | 359.771 | 346.168 | 3.78% |
| Carry MAE (bp) | 213.435 | 205.416 | 3.76% |
| Futures RMSE (points) | 33.160 | 31.802 | 4.09% |
| Futures MAE (points) | 25.003 | 23.434 | 6.27% |

These results are consistent with the stronger five-window evidence in
`../maturity_noise_study/MULTI_CUT_RESULTS.md`.

## Calendar reconciliation

The production full-sample optimum is slightly different from the research
study optimum (`kappa_slow = 0.288552`, `kappa_fast = 16.491817`). This is not
an implementation or optimizer discrepancy.

The study uses the Demo's explicit provisional 2027--2028 company calendar,
while production retains its documented weekday fallback beyond the installed
`chinese_calendar` coverage. This changes the maturity of 25 far-dated 2027
contract observations by as much as six sessions. Starting the optimizer at
the research parameters under production maturities converges to the production
result and likelihood reported above.

## Downstream pricing comparison

### Production pricing adapter

For the unchanged 2026-08-21 IM2609 example with 20 sessions to expiry:

| Quantity | Constant carry | Constant log-futures |
|---|---:|---:|
| Option-only value | 36.2794 | 29.6609 |
| Change | -- | -18.24% |
| Slow futures-equivalent delta | -0.427450 | -0.437348 |
| Fast futures-equivalent delta | -0.207660 | -0.351647 |
| Model-minus-observed initial futures | +0.3962 | -5.4716 |

The candidate numerical checks remain controlled: base-minus-fine grid price
is 0.0091 points and the quadrature-order range is 0.0173 points.

### Demo snapshot

For the unchanged 2026-08-10, 488-date, IM2612 Demo:

| Quantity | Constant carry | Constant log-futures |
|---|---:|---:|
| Kappa gap | 60.0000 (bound) | 18.8481 (interior) |
| `eta_fast` | 3.8852 | 1.3668 |
| In-sample futures RMSE (points) | 10.0540 | 5.3671 |
| Option-only value | 63.8575 | 83.0422 |
| Change | -- | +30.04% |
| Slow futures-equivalent delta | -0.332355 | -0.416663 |
| Fast futures-equivalent delta | -0.014069 | -0.112344 |

The Demo's complete fixed-`eta_fast` profile, convergence tables, charts, and
summary are in `../Demo/outputs_log_futures/`.

## Validation

- Full production `config_log_futures.yaml` workflow: passed end to end.
- Candidate Demo calibration, pricing, fixed-eta profile, and exports: passed.
- Pricing adapter reading `outputs_log_futures`: passed.
- Production tests: 17 passed after adding the rolling-chart regression.
- Demo tests: 5 passed.
- Pricing tests: 10 passed.
- Maturity-noise study tests: 8 passed.
- Fast-factor boundary study tests: 4 passed.
- Ruff and notebook JSON validation: passed.

## Recommendation

Use `constant_log_futures` as an opt-in parallel production candidate. Its
forecasting and parameter-stability evidence is better than constant carry,
but the option-price and hedge changes are large enough to require business and
risk review. Do not silently change the default until those differences have
been accepted and the 2027 production calendar policy has been resolved.
