# CSI 1000 / IM implied carry: one-factor OU model

This project estimates the latent instantaneous combined implied carry yield of
the CSI 1000 index from the complete available curve of IM futures contracts.
It uses historical curves for structural parameter estimation and the latest
curve to update the current filtered state.

## Confirmed conventions

- Spot: CSI 1000 daily close from `ak.stock_zh_index_daily("sh000852")`.
- Futures: individual IM contract **close**, not settlement, from
  `ak.futures_zh_daily_sina("IMYYMM")`.
- Interest rate: continuously compounded constant `0.014` unless changed in
  `config.yaml`.
- Time: number of exchange sessions divided by 244, for both maturity `tau` and
  state-transition gaps `Delta`. Sessions are counted over `(t, T]`, so `tau`
  is zero on expiry itself.
- Expiry: CFFEX third Friday inferred from the contract code, shifted to the
  next exchange session for a public holiday. The expiry source is retained.
- Carry is a combined implied carry yield. It is not decomposed into dividends,
  funding, or other basis components.

These trading-session conventions intentionally supersede the calendar-day/365
convention in the initial specification.

## Raw and normalized schemas

AkShare spot fields are `date, open, high, low, close, volume`. Futures fields
are `date, open, high, low, close, volume, hold, settle`. The normalized model
panel contains:

`date, spot, contract, futures_price, price_source, expiry, risk_free_rate,
sessions_to_expiry, tau, implied_carry`.

Every monthly contract code from the IM launch period is attempted; current
and next monthly contracts and the next two quarterly maturities are included.
Provider errors and empty contracts are written to `download_log.csv` rather
than suppressed.

## Model

The latent state follows

```text
dc_t = kappa (theta - c_t) dt + eta dW_t
```

with its exact irregular-gap transition. For maturity `tau`, the observed carry
is

```text
y(t,T) = theta + (c_t-theta) * (1-exp(-kappa*tau))/(kappa*tau) + epsilon
epsilon ~ N(0, sigma_epsilon^2).
```

The custom scalar-state Kalman filter performs a vector update for all available
contracts on each date. Parameters `kappa`, `eta`, and `sigma_epsilon` are
optimized in logs, with multiple starts. The initial state uses the stationary
OU distribution. A numerical Hessian supplies standard errors when it is
positive definite and well conditioned.

The fitted futures price is

```text
F_hat(t,T) = S_t * exp((r_t-y_hat(t,T))*tau).
```

## Running

From this directory in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
& 'D:\miniforge3\envs\spyder-env\python.exe' -m im_ou_carry --config .\config.yaml --refresh
```

After the first successful run, omit `--refresh` to use the reproducible raw
CSV cache. The same workflow can be run as an analysis script:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' .\analysis\run_workflow.py
```

Select `full`, `rolling`, or `split` under `estimation.mode`. The rolling window,
explicit train end date, automatic test fraction, close-to-expiry threshold,
stale handling, extreme-carry bound, rate, and paths are configurable.

## Outputs

`outputs/` contains:

- `implied_carries.csv` and `quality_audit.csv`;
- `parameters.csv`, `optimizer_runs.csv`, and `filtered_states.csv`;
- `fitted_curves.csv` and `residuals.csv` in carry and futures-price units;
- `evaluation_fits.csv`, `calibration_metrics.csv`, and benchmark fits;
- maturity, time-series, ACF, heteroskedasticity, expiry/roll, curve-shape, and
  rolling-parameter diagnostic tables;
- `run_summary.json` with the latest state and headline results;
- `charts/*.png` for the latest, representative, and worst curves, states,
  residuals, autocorrelation, and parameter stability.

Out-of-sample OU metrics use one-step-prior predictions after estimating
parameters only on the training period. The flat same-day benchmark is a
descriptive cross-sectional fit; previous-day and EWMA levels are lagged and
therefore feasible forecasts.

## Quality control

Missing fields, nonpositive prices, duplicate keys, inconsistent expiries,
near/expired contracts, stale price runs, and extreme carries are explicitly
flagged. Hard-invalid and configured exclusions remain in `quality_audit.csv`
with reason codes. Stale runs are flagged but retained by default.

## Tests

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -m pytest
```

Tests cover the trading-session convention, holiday-shifted expiry, implied
carry formula, audit behavior, ragged-curve filtering, stable OU formulas, and
seeded approximate recovery of known synthetic parameters.

## Limitations

- Futures close and index close are treated as date-aligned daily observations;
  any small timestamp/microstructure mismatch enters observation noise.
- A constant risk-free rate is not a full funding curve.
- Expiries are rule-derived rather than read from an authoritative contract
  master, though exceptions can be added to the calendar configuration.
- The initial model has one common observation-error volatility and ignores
  futures convexity.
- A one-factor OU curve is monotonic toward `theta`; it cannot reproduce
  systematic humps or U-shapes. The project flags such dates and tests residual
  maturity dependence rather than hiding this misspecification.

