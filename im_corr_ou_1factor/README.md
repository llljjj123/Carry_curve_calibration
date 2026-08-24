# CSI 1000 / IM correlated one-factor OU carry model

This project calibrates a one-factor latent IM implied-carry process and allows
its shocks to be correlated with CSI 1000 stock-return shocks. It is isolated
from the legacy `im_ou_carry` and two-factor projects.

## Confirmed data conventions

- CSI 1000 spot and individual IM futures **close** prices are aligned by date.
- Carry is one combined implied-carry yield; it is not decomposed.
- The continuously compounded risk-free rate defaults to configurable `0.014`.
- Maturity and observation gaps use exchange sessions divided by 244. Sessions
  are counted over `(date, expiry]`.
- CFFEX expiries are inferred as the third Friday, shifted to the next exchange
  session for a holiday. Exceptions can be supplied through an override CSV.
- Contracts with five or fewer sessions to expiry are excluded and audited.
- Stock volatility is fixed at **25% annualized** in this first version. It is a
  configuration value and is not optimized.

The cached sample spans 2022-07-22 through 2026-08-21. It contains 991 spot
closes and 3,964 futures observations from 52 contracts. Quality filtering
accepts 3,667 observations and reports all 297 exclusions.

## Model and pricing

The reduced-form dynamics are

```text
dS/S = (r-c) dt + sigma dW_S
dc   = kappa(theta-c) dt + eta dW_c
corr(dW_S,dW_c) = rho
```

with fixed `sigma=0.25`. For `tau=T-t`, define

```text
B(tau) = (1-exp(-kappa*tau))/kappa
C(tau) = integral B(u)^2 du
D(tau) = integral B(u) du
```

The exact futures relation is

```text
log(F/S) = (r-theta)tau - (c-theta)B
           + 0.5 eta^2 C - rho sigma eta D.
```

Consequently, the Kalman observation equation is

```text
y(t,T) = theta + (c-theta)B/tau
         - eta^2 C/(2 tau) + rho sigma eta D/tau + epsilon.
```

`B`, `C`, `D`, and the joint-return integral `J` use stable series for small
`kappa*tau`. Unequal exchange-session gaps and ragged daily curves are handled
directly.

## Models compared

Five fits are kept distinct:

1. `legacy_curve`: the prior observation equation, with neither correlation
   nor the exact futures convexity correction;
2. `exact_rho0_curve`: exact pricing with the convexity correction and `rho=0`;
3. `exact_corr_curve`: exact pricing with free `rho` using curve likelihood;
4. `exact_rho0_joint`: exact curve-and-return likelihood with `rho=0`;
5. `exact_corr_joint`: exact curve-and-return likelihood with free `rho`.

The legacy model is not mislabeled as the nested `rho=0` restriction. Formal
likelihood-ratio tests compare models 2 versus 3 and models 4 versus 5. Raw
curve-only and joint likelihoods are not directly compared because the joint
models include an additional return observation stream.

## Exact joint timing

After the date-`t` curve update, the next close-to-close stock return and OU
state transition are treated as a joint Gaussian pair. Conditioning on the
return creates the prior state for the next date; that date's futures curve
then updates it. The historical return drift `mu` is estimated as a nuisance
parameter instead of imposing the risk-neutral drift on realized returns.

Filtered states are used for one-step predictions and out-of-sample metrics.
Smoothed states are exported only for retrospective interpretation and are
explicitly labeled.

## Running

From this directory in PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
& 'D:\miniconda3\envs\GuoYuan\python.exe' -B .\analysis\run_workflow.py
```

Or use the module entry point with `src` on `PYTHONPATH`. Set `data.refresh` or
pass `--refresh` only when a fresh AkShare download is wanted.

## Outputs

`outputs/` contains cleaned carries and the quality audit; model parameters,
optimizer audits, filtered and smoothed states; fitted carries and futures;
train/test errors and benchmarks; rho profiles and likelihood-ratio tests;
standardized curve/return innovations; maturity, time-series, ACF, ARCH, and
rolling-stability diagnostics. `outputs/charts/` contains the corresponding
curve, futures, state, profile, innovation, residual, and stability charts.

## Tests

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
& 'D:\miniconda3\envs\GuoYuan\python.exe' -B -m pytest
```

Tests cover analytical integrals against numerical quadrature, small-argument
stability, formula reductions, Monte Carlo futures pricing, positive
semidefiniteness, ragged and unequal-gap filtering, filtered/smoothed
separation, data auditing, and seeded joint synthetic recovery.

## Limitations

- The same `(kappa, theta)` governs historical dynamics and risk-neutral
  futures pricing. This restricts the market price of carry risk. The code
  separates transition and pricing functions so P/Q parameters can be split
  later, but that extension is not estimated here.
- A constant rate is not a funding curve, and constant 25% stock volatility is
  only a first-pass assumption.
- Daily spot and futures closes are treated as synchronized; microstructure
  timing differences enter observation noise.
- A single observation-error volatility ignores maturity heteroskedasticity.
- Curve-only `rho` may be weakly identified because it enters through a smooth
  maturity correction. Identification is assessed through profile likelihood,
  stability, the joint likelihood, and out-of-sample performance—not by the
  optimizer estimate alone.
- A one-factor curve remains structurally limited in fitting humps and U-shapes.

