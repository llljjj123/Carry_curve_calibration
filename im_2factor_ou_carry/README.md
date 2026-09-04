# CSI 1000 / IM two-factor OU implied-carry model

This project estimates slow and fast latent factors from the full historical IM
futures curve. It also re-estimates the original one-factor model on the same
observations, filters, time convention, and train/test split so every comparison
is like-for-like.

## Confirmed data conventions

- CSI 1000 spot: daily close from `ak.stock_zh_index_daily("sh000852")`.
- IM futures: individual-contract **close**, not settlement, from
  `ak.futures_zh_daily_sina("IMYYMM")`.
- Time: exchange trading sessions divided by 244 for maturity and state gaps.
- Sessions to expiry are counted over `(observation date, expiry]`.
- Risk-free rate: continuously compounded `0.014` unless configured otherwise.
- Expiry: CFFEX third Friday inferred from the contract code, shifted to the next
  exchange session for a public holiday; an override CSV is supported.
- Carry remains the combined implied carry yield and is not economically decomposed.

## Model

The instantaneous carry is

```text
c_t = theta + x_slow,t + x_fast,t
```

with independent centered OU factors

```text
dx_j,t = -kappa_j x_j,t dt + eta_j dW_j,t
0 < kappa_slow < kappa_fast.
```

The backward-compatible `constant_carry` observation equation is

```text
y_t(tau) = theta
           + B(kappa_slow,tau) x_slow,t
           + B(kappa_fast,tau) x_fast,t
           + epsilon_t(tau)

B(kappa,tau) = (1-exp(-kappa*tau))/(kappa*tau)
epsilon ~ N(0, sigma_epsilon^2).
```

The production-candidate `constant_log_futures` equation instead filters the
native log-futures observation

```text
z_t(tau) = log(F_t,T/S_t) - r_t tau
         = -theta tau
           - tau B(kappa_slow,tau) x_slow,t
           - tau B(kappa_fast,tau) x_fast,t
           + epsilon_logF,tau

epsilon_logF ~ N(0, sigma_log_futures^2).
```

In annualized-carry units this implies

```text
sigma_carry(tau) = sigma_log_futures / tau.
```

The log-futures likelihood includes the exact `sum(log(tau))` Jacobian when
reported, so likelihoods, AIC, BIC, and predictive scores remain comparable
with carry-space runs. Both the one- and two-factor models use the selected
observation equation, preserving their like-for-like comparison.

Opposite-signed slow and fast states allow a fitted curve to contain a hump or
U-shape. Centered factors and a single `theta` avoid the unidentifiable separate
factor means. The initial version fixes factor-shock correlation at zero and
retains one common observation-error volatility to isolate the value of the
second factor.

Exact irregular-gap transitions and stationary initial covariance are used.
The custom two-dimensional Kalman filter performs a vector update for every
ragged daily curve and exports whitened one-step innovations. Positivity and
factor ordering are enforced through transformed parameters. Maximum likelihood
uses multiple starts and a numerical Hessian when stable.

## Running

In PowerShell from this directory:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
& 'D:\miniforge3\envs\spyder-env\python.exe' -m im_2factor_ou_carry --config .\config.yaml
```

Use `--refresh` to download a new AkShare snapshot. Without it, the reproducible
CSV cache under `data/raw/` is used. The equivalent analysis entry point is:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' .\analysis\run_workflow.py
```

Full-sample, latest rolling-window, and explicit train/test structural estimation
are selected through `estimation.mode`. Diagnostic rolling estimates and all
optimizer start counts are separately configurable.

`config.yaml` retains `constant_carry` as the default. The validated candidate
configuration writes to a separate output directory and can be run with:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -m im_2factor_ou_carry `
  --config .\config_log_futures.yaml
```

The relevant settings are:

```yaml
estimation:
  observation_noise_model: constant_log_futures
  kappa_gap_upper_bound: 120.0
  eta_fast_upper_bound: 6.0
```

## Primary outputs

`outputs/` contains:

- `two_factor_parameters.csv`, optimizer starts, filtered slow/fast states,
  sequential whitened innovations, fitted curves, and residuals;
- independently estimated one-factor parameters, curves, and residuals;
- `model_information_criteria.csv` with log likelihood, AIC, and BIC;
- `calibration_metrics.csv` with in-sample filtered and genuine out-of-sample
  one-step prediction errors for both OU models and three benchmarks;
- standardized innovation time series, ACFs, Ljung–Box, squared-residual, and
  ARCH tests by in-sample/out-of-sample period;
- direct maturity-bucket and expiry/roll comparisons;
- `shape_fit_comparison.csv`, showing whether observed humps/U-shapes and
  inversions are reproduced on every date;
- `state_observability_diagnostics.csv`, relating fast-state magnitude and
  instantaneous-state uncertainty to the nearest available contract;
- rolling one- and two-factor parameter estimates;
- `run_summary.json` and diagnostic PNG charts.

The latest instantaneous state is reported as

```text
c_hat_t = theta + x_hat_slow,t + x_hat_fast,t
```

with uncertainty computed from both state variances and their filtered covariance.

## Interpretation and diagnostics

The factors are statistical. A natural working interpretation is:

- slow factor: persistent carry regime or structural IM hedging demand;
- fast factor: temporary front-end basis, roll, liquidity, or hedging pressure.

This project does not claim DMA causality. The raw futures schema retains volume
and open interest (`hold`) so the factor series can later be compared with market
activity and policy/event dates.

Standardized one-step Kalman innovations—not only posterior fitted residuals—are
tested for autocorrelation and volatility clustering. A second factor can address
omitted slope/curvature dynamics, but it does not itself create stochastic
volatility. Persistent ARCH effects would motivate time-varying or
maturity-dependent observation noise.

Near-expiry filtering remains fixed at five sessions for a fair model comparison.
Persistent near-expiry errors may be microstructure or timing effects rather than
evidence for another state factor.

The zero-maturity fast state is an extrapolation. Dates with no short contract or
high filtered instantaneous uncertainty are explicitly flagged; curve fits can
remain reliable even when the instantaneous fast state is weakly observed.

## Tests

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -m pytest
```

Tests cover the trading-session calendar, expiry shifts, carry calculation,
quality auditing, exact one- and two-factor transitions, ragged Kalman filtering,
hump generation, combined-state recovery, and seeded approximate recovery of
known one- and two-factor OU parameters.

## Current model limitations

- State shocks are independent; correlation can be added after identification is
  demonstrated.
- Observation noise is Gaussian and constant through time. Under
  `constant_carry` it is common in annualized-carry units; under
  `constant_log_futures` it is common in log-futures units and therefore scales
  as `1/tau` when expressed as annualized carry.
- The factors are stationary. A permanent structural carry trend may require a
  time-varying mean or regime model rather than merely a slow OU factor.
- Two factors generally support one economically meaningful turning point, not an
  arbitrary sequence of multiple humps.
- Rule-derived expiries should be overridden if an authoritative contract master
  identifies an exception.
