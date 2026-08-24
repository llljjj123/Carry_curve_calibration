# Using the Two-Factor OU Carry Model in Option Pricing

## Short answer

For an option on the CSI 1000 spot index, the two-factor OU carry model can be used to construct the maturity-specific forward input to a BSM-style pricing function. However, the instantaneous carry state should not be inserted directly as a constant carry rate for every option maturity.

For an option directly on an IM futures contract, use Black-76 with the observed futures price. Do not add the OU carry again, because the futures price already incorporates the market-implied carry.

## 1. Two-factor carry specification

The instantaneous carry state is

$$
c_t = \theta + x_{s,t} + x_{f,t},
$$

where the slow and fast factors follow

$$
dx_{j,t} = -\kappa_j x_{j,t}\,dt + \eta_j\,dW_{j,t},
\qquad j \in \{s,f\}.
$$

The calibrated instantaneous state, $c_t$, describes the carry at zero maturity. An option expiring at a positive maturity depends instead on the expected average carry accumulated between today and expiry.

Define the OU loading

$$
B(\kappa,\tau)
=
\frac{1-e^{-\kappa\tau}}{\kappa\tau}.
$$

The fitted maturity-average carry for expiry $T$ is

$$
\widehat y_t(T)
=
\theta
+ B(\kappa_s,\tau_{\text{carry}})x_{s,t}
+ B(\kappa_f,\tau_{\text{carry}})x_{f,t}.
$$

Under the convention used in this project,

$$
\tau_{\text{carry}}
=
\frac{N_{\text{trading sessions in }(t,T]}}{244}.
$$

This maturity averaging is important. The fast factor has a large effect at the front of the curve, but its loading decays rapidly as maturity increases. Using $c_t$ unchanged for all expiries would therefore overstate the fast factor's influence on longer-dated options.

## 2. Construct the forward first

For a CSI 1000 spot-index option, use the fitted carry to construct an OU-implied forward:

$$
\widehat F_t(T)
=
S_t
\exp\left[
\left(r_t-\widehat y_t(T)\right)
\tau_{\text{carry}}
\right].
$$

Here:

- $S_t$ is the CSI 1000 spot index level;
- $r_t$ is the continuously compounded risk-free rate;
- $\widehat y_t(T)$ is the two-factor OU maturity-average carry;
- $\tau_{\text{carry}}$ follows the trading-sessions-over-244 convention.

The cleanest implementation is then a forward-form BSM function. For a European call,

$$
C_t
=
D_r(t,T)
\left[
\widehat F_t(T)N(d_1)-KN(d_2)
\right],
$$

and for a European put,

$$
P_t
=
D_r(t,T)
\left[
KN(-d_2)-\widehat F_t(T)N(-d_1)
\right],
$$

with

$$
d_1
=
\frac{
\ln\left(\widehat F_t(T)/K\right)
+\tfrac{1}{2}\sigma^2\tau_{\text{vol}}
}{
\sigma\sqrt{\tau_{\text{vol}}}
},
$$

$$
d_2=d_1-\sigma\sqrt{\tau_{\text{vol}}}.
$$

The discount factor can be written as

$$
D_r(t,T)=\exp\left(-r_t\tau_{\text{discount}}\right).
$$

This formulation makes the roles of the inputs explicit: the OU model supplies the forward curve, while the option-volatility model supplies $\sigma$.

## 3. Do not use the OU factor volatilities as option volatility

The calibrated parameters $\eta_s$ and $\eta_f$ describe changes in the latent carry factors. They are not the volatility of the CSI 1000 index return and should not be passed into BSM as $\sigma$.

The BSM volatility input should come from an appropriate source, such as:

- the MO option implied-volatility surface;
- a separately estimated CSI 1000 return-volatility model;
- a chosen scenario volatility.

Thus, in the practical deterministic-carry implementation:

$$
\text{OU states and kappas}
\longrightarrow
\widehat y_t(T)
\longrightarrow
\widehat F_t(T),
$$

while

$$
\text{equity option-volatility model}
\longrightarrow
\sigma(K,T).
$$

These inputs meet only inside the option-pricing formula.

## 4. MO option versus an option on IM futures

### CSI 1000/MO spot-index option

For an MO option whose underlying is the CSI 1000 index, using the OU-implied forward is natural:

$$
\text{spot and OU carry}
\longrightarrow
\widehat F_t(T)
\longrightarrow
\text{forward BSM price}.
$$

You may also use an observed or interpolated IM futures price as the forward input. That gives a market-forward price rather than a model-forward price. Comparing the two can be useful for valuation and diagnostics.

### Option on an IM futures contract

For an option whose underlying is an IM futures contract, use Black-76 directly:

$$
C_t
=
D_r(t,T_o)
\left[
F_t^{\mathrm{IM}}N(d_1)-KN(d_2)
\right],
$$

where $T_o$ is the option expiry and $F_t^{\mathrm{IM}}$ is the relevant observed futures price.

Do not first adjust $F_t^{\mathrm{IM}}$ by the OU carry. The observed futures price already embeds the carry; applying it again would double-count carry.

## 5. Relevant time conventions

It is useful to keep three time variables separate in code:

1. Carry time:

$$
\tau_{\text{carry}}
=
\frac{\text{trading sessions}}{244}.
$$

2. Volatility time, $\tau_{\text{vol}}$, following the convention of the volatility input or implied-volatility surface.

3. Discount time, $\tau_{\text{discount}}$, following the interest-rate curve's day-count convention.

They may be numerically similar, but silently treating them as identical can create avoidable pricing inconsistencies.

## 6. Numerical illustration from the latest filtered state

The latest estimated state was approximately

$$
\theta=8.2617\%,
\qquad
x_{s,t}=+5.9910\%,
\qquad
x_{f,t}=-1.6141\%.
$$

Therefore, the instantaneous carry was

$$
c_t
=
\theta+x_{s,t}+x_{f,t}
\approx 12.6386\%.
$$

The fitted maturity-average carries were approximately:

| Maturity | Fitted carry |
|---:|---:|
| 20 trading sessions | 13.53% |
| 79 trading sessions | 13.08% |
| 144 trading sessions | 12.45% |

The maturity-specific values differ from the instantaneous carry because each factor is averaged over the option's life. This is exactly why the pricing function should receive $\widehat y_t(T)$ or the resulting $\widehat F_t(T)$, rather than one common $c_t$ for all maturities.

## 7. What this practical method assumes

Using the fitted carry as a deterministic input is a pragmatic first step. It conditions on today's filtered factors and uses their expected mean-reverting paths to form the forward. In this version, future random carry shocks do not directly affect the option payoff distribution.

A fully stochastic-carry option model would require additional assumptions, including:

- risk-neutral, rather than historical, OU dynamics;
- market prices of carry-factor risk;
- correlations between CSI 1000 returns and the two carry shocks;
- possible correlation between the slow and fast factors;
- a futures convexity adjustment;
- Monte Carlo, PDE, or an affine pricing derivation.

The current OU parameters were estimated from historical futures curves. They should not automatically be interpreted as risk-neutral parameters. Consequently, the deterministic-forward approach is suitable as a transparent pricing input or scenario framework, but it is not yet a complete stochastic-carry derivatives model.

## Recommended first implementation

For an MO index option:

1. Infer the latest $x_{s,t}$ and $x_{f,t}$ from the IM futures curve.
2. Calculate $\widehat y_t(T)$ for the option expiry using trading sessions divided by 244.
3. Construct $\widehat F_t(T)$ from spot, the risk-free rate, and fitted carry.
4. Supply that forward to a forward-form BSM function.
5. Supply option volatility separately from an MO implied-volatility surface or another equity-volatility model.
6. Discount using the chosen interest-rate curve and its day-count convention.

For an option on IM futures, skip steps 1–3 for the base price and use the observed IM futures price in Black-76. The OU model can still be used for scenario analysis or to assess whether the traded futures curve is rich or cheap relative to the fitted curve.
