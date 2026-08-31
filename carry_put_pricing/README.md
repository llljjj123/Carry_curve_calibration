# American put on the CSI 1000 / IM carry curve

This isolated project prices the early-exercise payoff described in
`../put_on_carry.md`:

$$
G_t = S_t\left[
e^{(r-q_{0,T})(T-t)}-
\frac{F_{t,T}}{S_t}
\right]^+.
$$

The locked inception carry is inferred from the observed spot and futures quote,

$$
q_{0,T}=r-\frac{\log(F_{0,T}/S_0)}{T}.
$$

Future futures prices use the exact integrated-carry expression rather than the
deterministic maturity-average approximation.

## Risk-neutral model

The implementation assumes

$$
\frac{dS_t}{S_t}=(r-c_t)dt+\sigma dW_t^S,
\qquad
c_t=\theta+x_{s,t}+x_{f,t},
$$

$$
dx_{j,t}=-\kappa_jx_{j,t}dt+\eta_jdW_t^j,
$$

with independent slow, fast, and spot Brownian shocks. For

$$
I_{t,T}=\int_t^T c_u\,du,
$$

the OU model makes $I_{t,T}$ conditionally Gaussian. Therefore,

$$
\frac{F_{t,T}}{S_t}
=e^{r(T-t)}E_t^Q[e^{-I_{t,T}}]
=\exp\left(r\tau-m_I+\frac12v_I\right).
$$

The conditional mean and variance are

$$
m_I=\theta\tau+A(\kappa_s,\tau)x_s+A(\kappa_f,\tau)x_f,
\qquad
A(\kappa,\tau)=\frac{1-e^{-\kappa\tau}}{\kappa},
$$

$$
v_I=\sum_{j\in\{s,f\}}
\frac{\eta_j^2}{\kappa_j^2}
\left[
\tau-\frac{2(1-e^{-\kappa_j\tau})}{\kappa_j}
+\frac{1-e^{-2\kappa_j\tau}}{2\kappa_j}
\right].
$$

The calibrated historical OU parameters are used as risk-neutral parameters in
this prototype. A production calibration would need to identify risk-neutral
dynamics or carry-factor risk premia.

## Numerical method

Payoff homogeneity gives

$$
V(t,S,x_s,x_f)=S\,v(t,x_s,x_f).
$$

Over one exercise interval, the normalized continuation value is

$$
C_t/S_t = E_t^Q\left[
e^{-\int_t^{t+\Delta t}c_u du}
v(t+\Delta t,X_{t+\Delta t})
\right].
$$

The pricer performs backward induction on a two-dimensional slow/fast factor
grid. It evaluates this conditional expectation with exact OU transition moments,
Gaussian exponential tilting, and tensor Gauss-Hermite quadrature. Linear
interpolation connects quadrature points to the state grid. Exercise is allowed
on every trading session, so the result is a daily Bermudan approximation to the
continuous-time American contract.

This resembles Longstaff-Schwartz backward induction, but continuation values
are computed deterministically from the known Gaussian transition rather than
estimated by regression on simulated paths.

The spot volatility remains an explicit `GBMParams` input, as requested, but it
does not affect this payoff under zero spot/carry shock correlation. The spot
level scales the normalized price, while its Brownian volatility cancels.

## Futures-equivalent curve deltas

The option state contains separate slow and fast carry factors, while one IM
futures quote cannot identify both factors by itself. The pricer therefore
reports two directional hedge ratios. For factor $j\in\{s,f\}$,

$$
\Delta_j^F
=\frac{\partial V/\partial x_j}{\partial F_{t,T}/\partial x_j},
\qquad
\frac{\partial F_{t,T}}{\partial x_j}
=-A(\kappa_j,T-t)F_{t,T},
$$

holding spot and the other carry factor fixed. Each number is expressed in
option points per futures-price point and is calculated in two ways:

1. a differentiated backward induction that follows the exercise policy of the
   original option value; and
2. a local grid bump-and-value ratio using nearby factor states and their exact
   model futures prices.

The derivative recursion never solves a separate stopping problem. At an
exercise node it uses the exercise-payoff derivative; at a continuation node it
uses the differentiated continuation value. The locked inception carry remains
fixed under both factor bumps, so the calculation measures the existing
contract rather than restriking it.

## Main API

```python
from carry_put_pricing import (
    CarryPutContract,
    FactorState,
    GBMParams,
    TwoFactorOUParams,
    price_american_carry_put,
)

result = price_american_carry_put(
    CarryPutContract(
        initial_spot=7601.804,
        initial_futures=7527.0,
        sessions_to_expiry=20,
    ),
    TwoFactorOUParams(
        kappa_slow=1.2409047966134241,
        kappa_fast=44.32953558819525,
        theta=0.08261737612606601,
        eta_slow=0.0799282328223189,
        eta_fast=2.852079153601385,
    ),
    FactorState(
        slow=0.05990951629435136,
        fast=-0.016141256167290972,
    ),
    GBMParams(risk_free_rate=0.014, volatility=0.25),
)

print(result.price)
```

The pricer fixes inception exercise value to zero by the contractual identity
that the locked and prevailing futures quotes coincide at inception. It reports
the exact-model-versus-observed initial futures difference separately instead of
turning a curve-fit residual into an exercise payoff.

## Agreed example

The example script reads the latest calibrated parameters, filtered factors, and
valid nearest futures quote from `../im_2factor_ou_carry/outputs`. With the current
snapshot this selects IM2609 on 2026-08-21, expiring 2026-09-18.

From this directory in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
& 'D:\miniconda3\envs\GuoYuan\python.exe' -B .\analysis\run_example.py
```

Generated files are written to `outputs/`:

- `example_result.json`: inputs, assumptions, diagnostics, and base price;
- `grid_convergence.csv`: coarse/base/fine numerical comparison;
- `quadrature_convergence.csv`: quadrature-order stability at the base grid;
- `exercise_summary.csv`: exercise-region diagnostics by exercise date.
- `curve_delta_comparison.csv`: slow and fast futures-equivalent deltas from
  pathwise backward induction and local bump-and-value.

For the cached 2026-08-21 inputs, the base-grid result is **36.2794 index
points**, or **0.477248% of spot**. The base-versus-fine grid difference is
0.0050 points, and prices over nearby quadrature orders 39 through 47 span
0.0052 points. These are numerical diagnostics only; they do not address the
larger economic uncertainty from treating historical OU estimates as
risk-neutral parameters.

## Tests

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\miniconda3\envs\GuoYuan\python.exe' -B -m pytest
```

Tests cover stable exact-integral formulas, a seeded moment check, the agreed
initial forward calculation, zero terminal optionality, spot-volatility
invariance, spot scaling, deterministic flat-carry behavior, and output
diagnostics. They also verify delta sign/conversion, agreement between the two
delta methods, zero delta for a one-session zero-value contract, and invariance
of the futures-equivalent hedge ratios under proportional spot/futures scaling.

## Scope limitations

- OU parameters are provisionally treated as risk-neutral.
- Slow and fast shocks are independent, as in the calibrated two-factor model.
- Spot/carry shock correlations are zero.
- Exercise is daily rather than mathematically continuous.
- Observed initial futures and filtered states need not fit the exact stochastic
  forward formula perfectly; that initial basis is reported.
- The state grid uses clipped boundary interpolation. The example therefore
  reports coarse/base/fine convergence rather than relying on one grid silently.
