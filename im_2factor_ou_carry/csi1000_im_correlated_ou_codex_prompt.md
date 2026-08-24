# Codex Task: Calibrate a One-Factor Correlated OU Model for CSI 1000 / IM Implied Carry

Build a clean, reproducible Python research project that calibrates a one-factor stochastic implied-carry model to CSI 1000 spot data and IM futures curves. I will provide the historical spot and futures files.

Inspect the supplied files before coding, summarize the detected schemas, and adapt the data loader to them. Ask me only if an essential field—especially contract expiry—is missing or ambiguous.

## 1. Objective

Model the combined IM implied carry yield. Do not try to separate dividends, funding, shorting costs, or other basis components.

Use a single latent OU carry factor and allow its shocks to be correlated with CSI 1000 returns. Assume constant stock volatility in this first version.

The implementation must:

1. construct historical implied-carry curves from spot and futures data;
2. estimate the one-factor OU parameters and latent daily carry states;
3. estimate the stock/carry shock correlation;
4. compare the correlated model with the restricted model \(\rho=0\);
5. assess in-sample fit, parameter identification, stability, and out-of-sample performance.

## 2. Model

For the first version, use the reduced-form shared-dynamics specification

\[
\frac{dS_t}{S_t}=(r_t-c_t)\,dt+\sigma\,dW_t^S,
\]

\[
dc_t=\kappa(\theta-c_t)\,dt+\eta\,dW_t^c,
\]

with

\[
dW_t^S dW_t^c=\rho\,dt.
\]

Here:

- \(c_t\) is the latent instantaneous implied carry;
- \(\kappa>0\) is the mean-reversion speed;
- \(\theta\) is the long-run carry level;
- \(\eta>0\) is carry volatility;
- \(\sigma>0\) is constant CSI 1000 volatility;
- \(-1<\rho<1\) is the instantaneous stock/carry shock correlation.

This is a reduced-form baseline. It imposes the same OU drift parameters for historical state evolution and risk-neutral futures pricing, which amounts to restricting the market price of carry risk. State this limitation clearly in the README. Design the code so separate \((\kappa_P,\theta_P)\) and \((\kappa_Q,\theta_Q)\) can be added later, but do not implement a two-factor model in this task.

The correct log-stock equation is

\[
d\log S_t=\left(r_t-c_t-\frac12\sigma^2\right)dt+\sigma dW_t^S.
\]

## 3. Exact futures-pricing formula

For \(\tau=T-t\), define

\[
B(\tau)=\frac{1-e^{-\kappa\tau}}{\kappa},
\]

\[
C(\tau)=\int_0^\tau B(u)^2du
=\frac{1}{\kappa^2}\left[
\tau-\frac{2(1-e^{-\kappa\tau})}{\kappa}
+\frac{1-e^{-2\kappa\tau}}{2\kappa}
\right],
\]

and

\[
D(\tau)=\int_0^\tau B(u)du
=\frac{\tau}{\kappa}-\frac{1-e^{-\kappa\tau}}{\kappa^2}.
\]

Under the model,

\[
\log\frac{F(t,T)}{S_t}
=(r_t-\theta)\tau
-(c_t-\theta)B(\tau)
+\frac12\eta^2C(\tau)
-\rho\sigma\eta D(\tau).
\]

Calculate the observed implied carry as

\[
y_t(T)=r_t-\frac{1}{\tau}\log\frac{F(t,T)}{S_t}.
\]

The Kalman-filter observation equation is therefore

\[
y_t(T)=
\theta
+(c_t-\theta)\frac{B(\tau)}{\tau}
-\frac{\eta^2C(\tau)}{2\tau}
+\frac{\rho\sigma\eta D(\tau)}{\tau}
+\varepsilon_{t,T}.
\]

This equation is affine in \(c_t\). Implement it directly and also reconstruct fitted futures prices from the fitted carry curves.

Use numerically stable small-\(\kappa\tau\) evaluations for \(B\), \(C\), and \(D\), using `expm1` or appropriate series expansions.

## 4. Exact OU state transition

For an observation interval \(\Delta_t\), use

\[
c_{t+\Delta_t}
=\theta+e^{-\kappa\Delta_t}(c_t-\theta)+w_t,
\]

where

\[
w_t\sim N(0,Q_t),
\qquad
Q_t=\frac{\eta^2}{2\kappa}
\left(1-e^{-2\kappa\Delta_t}\right).
\]

Allow unequal gaps between dates and a changing number of available futures contracts.

## 5. Input data and cleaning

The supplied data should contain or allow the loader to derive:

- trading date or timestamp;
- CSI 1000 spot level \(S_t\);
- IM contract code;
- IM settlement or closing price \(F(t,T)\);
- contract expiry date \(T\);
- optionally, a risk-free rate \(r_t\).

If rates are unavailable, make the continuously compounded annual rate a configuration input. Do not silently hard-code it.

Use actual calendar time:

\[
\tau=\frac{T-t}{365}.
\]

Align spot and futures observations to the same timestamp convention. Preserve actual contract expiries and do not build a continuous rolled futures series before calculating carry.

Make the near-expiry exclusion threshold configurable, with a default of five trading days. Report, rather than silently discard, records removed because of missing values, duplicates, nonpositive prices, invalid expiries, stale prices, or extreme implied carries.

## 6. Calibration modes

Implement both modes below.

### Mode A: futures-curve Kalman likelihood

Estimate

\[
(\kappa,\theta,\eta,\rho,\sigma_\varepsilon)
\]

from the historical panel of IM implied-carry curves, conditional on constant \(\sigma\).

For each candidate parameter vector, run the Kalman filter over all dates and maximize the Gaussian innovation likelihood. On a date with several contracts, assimilate all maturities as one observation vector.

Initially estimate \(\sigma\) independently from CSI 1000 log returns, using a configurable sample window and annualization convention. Also allow the user to supply or fix \(\sigma\).

Because the futures curve may identify \(\rho\) weakly, produce a profile likelihood over a configurable grid of \(\rho\) values and report a likelihood-based confidence interval when possible.

### Mode B: joint curve-and-return calibration

Add CSI 1000 log returns to the likelihood so that \(\rho\) is informed by the co-movement between stock-return shocks and innovations in the latent carry state.

Use constant \(\sigma\). Treat the historical stock-return mean as a configurable or estimated nuisance parameter instead of assuming that realized returns have risk-neutral drift exactly equal to \(r-c_t\).

Implement the daily stock-return/state transition as a joint Gaussian system. At minimum, provide a documented Euler version. Prefer the exact discretization below.

For interval \(\Delta\), the conditional log-return mean is

\[
m_R(c_t)=
\left(\mu-\theta-\frac12\sigma^2\right)\Delta
-(c_t-\theta)B(\Delta).
\]

Its conditional variance is

\[
V_R(\Delta)=
\sigma^2\Delta+\eta^2C(\Delta)
-2\rho\sigma\eta D(\Delta).
\]

The covariance between the OU transition innovation \(w_t\) and the return innovation is

\[
G(\Delta)=
\rho\eta\sigma B(\Delta)-\eta^2J(\Delta),
\]

where

\[
J(\Delta)=
\frac{1}{\kappa}\left[
\frac{1-e^{-\kappa\Delta}}{\kappa}
-\frac{1-e^{-2\kappa\Delta}}{2\kappa}
\right].
\]

Thus, conditional on \(c_t\), the pair consisting of the next state and the interval stock return has covariance

\[
\begin{pmatrix}
Q(\Delta) & G(\Delta)\\
G(\Delta) & V_R(\Delta)
\end{pmatrix}.
\]

Use a generalized Kalman-filter formulation, an equivalent augmented-state formulation, or direct Gaussian conditioning. Document the timing convention carefully and test that the covariance matrix remains positive semidefinite for valid parameters.

As an additional diagnostic—not as the final estimator—calculate the empirical correlation between standardized filtered carry innovations and standardized stock-return residuals. Explain that state-estimation noise can bias this two-step correlation toward zero.

## 7. Optimization and parameter constraints

Use maximum likelihood with multiple starting points. Apply transformations:

- optimize \(\log\kappa\), \(\log\eta\), \(\log\sigma\), and \(\log\sigma_\varepsilon\);
- parameterize \(\rho=\tanh(a_\rho)\);
- leave \(\theta\) and the nuisance return mean unconstrained unless sensible configurable bounds are required.

Report optimizer convergence, gradient norm if available, parameter estimates, approximate standard errors from a numerical Hessian if stable, log-likelihood, AIC, BIC, and the mean-reversion half-life

\[
t_{1/2}=\frac{\log 2}{\kappa}.
\]

Use several economically plausible initial values. Do not present a boundary solution or failed optimization as a valid calibration.

## 8. Model comparisons and diagnostics

Estimate and compare:

1. the original uncorrelated model with \(\rho=0\);
2. the correlated curve-only model;
3. the correlated joint curve-and-return model.

Produce:

- in-sample and out-of-sample RMSE and MAE in carry percentage points, basis points, and futures index points;
- observed-versus-fitted carry curves for the latest date, representative dates, and worst-fit dates;
- observed-versus-fitted futures prices;
- filtered latent carry states and their uncertainty bands;
- carry innovations and stock-return residuals;
- a scatter plot and rolling correlation of standardized innovations;
- residuals by maturity and over time;
- residual autocorrelation diagnostics;
- rolling or expanding-window parameter stability;
- profile likelihood for \(\rho\);
- likelihood-ratio comparison of \(\rho=0\) versus free \(\rho\);
- out-of-sample comparison against a flat carry curve, previous-day curve, and EWMA carry benchmark.

Explicitly discuss whether \(\rho\) is economically and statistically identified. A nonzero optimizer output is not sufficient evidence: require a reasonably concentrated profile likelihood, stability across samples, and/or material out-of-sample improvement.

Use filtered states for live or out-of-sample analysis. Smoothed states may be shown for historical interpretation, but label them clearly because they contain future information.

## 9. Deliverables

Create:

- modular Python source code;
- a configuration file for data paths, column mappings, rates, volatility handling, filters, and train/test windows;
- a command-line entry point;
- a notebook or analysis script demonstrating the full workflow;
- CSV or Parquet outputs containing cleaned implied carries, parameter estimates, filtered states, fitted curves, fitted futures prices, and residuals;
- diagnostic plots;
- a concise README with equations, assumptions, commands, outputs, and limitations;
- unit and synthetic-data tests.

Use standard packages such as pandas, NumPy, SciPy, statsmodels, matplotlib, and seaborn. Avoid unnecessary dependencies.

## 10. Required tests

Include tests that verify:

1. \(B\), \(C\), \(D\), and \(J\) against numerical integration;
2. stable evaluation when \(\kappa\tau\) is small;
3. the futures formula reduces correctly when \(\rho=0\) and/or \(\eta=0\);
4. Monte Carlo simulation agrees with the analytical futures formula within simulation error;
5. the joint state/return covariance matrix is positive semidefinite for valid test parameters;
6. the Kalman filter handles missing maturities and unequal date gaps;
7. synthetic data approximately recover known \(\kappa,\theta,\eta,\rho\), and \(\sigma\), with tolerances appropriate to sample size;
8. filtered and smoothed states are not accidentally mixed in out-of-sample evaluation.

## 11. Working procedure

Before implementation:

1. inspect the supplied data;
2. summarize the detected schema, date range, available contracts, missing fields, and assumptions;
3. state how \(r_t\), \(\sigma\), expiries, and timestamps will be handled.

Then implement and run the project. Afterward, report:

- all estimated parameters;
- calibration and optimizer status;
- whether \(\rho\) appears identified;
- comparison with \(\rho=0\);
- in-sample and out-of-sample errors;
- major residual patterns and model limitations;
- exact locations of code, configuration, outputs, and charts.

Do not proceed to a two-factor model unless I explicitly request it after reviewing the one-factor results.
