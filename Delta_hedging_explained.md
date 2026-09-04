# Two-factor delta hedging with multiple futures maturities

## 1. Why one futures contract is insufficient in general

The two-factor carry model has slow and fast state variables:

$$
c_t=\theta+x_{s,t}+x_{f,t}.
$$

An option value therefore has two local factor sensitivities:

$$
V_s=\frac{\partial V}{\partial x_s},
\qquad
V_f=\frac{\partial V}{\partial x_f}.
$$

One futures contract has exposure to both factors, but its two exposures come
in one fixed maturity-dependent proportion. A position in that one contract
therefore supplies only one hedge direction. It cannot, in general, neutralize
two arbitrary factor exposures independently.

This is the meaning of the two directional futures-equivalent deltas currently
reported by the pricing code. They answer two separate questions:

$$
\Delta_s^F
=
\frac{\partial V/\partial x_s}{\partial F/\partial x_s},
\qquad
\Delta_f^F
=
\frac{\partial V/\partial x_f}{\partial F/\partial x_f}.
$$

Each ratio considers one factor direction at a time. They are not two positions
that can simply be applied simultaneously to the same futures contract, and
averaging them has no general factor-neutrality justification.

## 2. Factor exposure of a futures maturity

For a futures contract with remaining maturity

$$
h_i=T_i-t,
$$

the two-factor model gives

$$
F_i
=
S_t\exp\left(
r h_i-\theta h_i
-A_s(h_i)x_s
-A_f(h_i)x_f
\right),
$$

where

$$
A_j(h)
=
\frac{1-e^{-\kappa_jh}}{\kappa_j}.
$$

The factor sensitivities of futures contract $i$ are therefore

$$
\frac{\partial F_i}{\partial x_s}
=
-F_iA_s(h_i),
\qquad
\frac{\partial F_i}{\partial x_f}
=
-F_iA_f(h_i).
$$

Its exposure vector is

$$
\mathbf g_i
=
\begin{bmatrix}
-F_iA_s(h_i)\\
-F_iA_f(h_i)
\end{bmatrix}.
$$

Different maturities have different slow/fast mixtures, so their exposure
vectors generally point in different directions.

## 3. Hedging both factors with two futures

Let $n_1$ and $n_2$ be positions in two futures contracts. Ignoring contract
multipliers temporarily, the hedged portfolio's first-order factor exposure is

$$
\begin{bmatrix}
V_s\\
V_f
\end{bmatrix}
+n_1\mathbf g_1
+n_2\mathbf g_2.
$$

To neutralize both factors, solve

$$
\begin{bmatrix}
-F_1A_s(h_1) & -F_2A_s(h_2)\\
-F_1A_f(h_1) & -F_2A_f(h_2)
\end{bmatrix}
\begin{bmatrix}
n_1\\
n_2
\end{bmatrix}
=
-
\begin{bmatrix}
V_s\\
V_f
\end{bmatrix}.
$$

If the two futures have contract multipliers $M_1$ and $M_2$, each column must
also be multiplied by its corresponding multiplier. The option notional must
be applied to the right-hand side.

## 4. What makes two maturities appropriately chosen

The exposure matrix has a unique solution when its determinant is nonzero:

$$
\det(G)
=
F_1F_2
\left[
A_s(h_1)A_f(h_2)
-A_s(h_2)A_f(h_1)
\right].
$$

Equivalently, the two futures should have meaningfully different fast-to-slow
loading ratios:

$$
R(h)
=
\frac{A_f(h)}{A_s(h)},
$$

so that

$$
R(h_1)\not\approx R(h_2).
$$

If the maturities are identical or very close, the two exposure vectors are
nearly parallel. The matrix then becomes nearly singular: hedge quantities can
become very large, unstable, and highly sensitive to small parameter or price
errors.

A scale-free way to measure separation is the sine of the angle between the
two exposure vectors:

$$
s
=
\frac{|\det(G)|}
{\|\mathbf g_1\|\,\|\mathbf g_2\|}.
$$

Values near zero indicate a poorly conditioned pair. Larger values indicate
better separation, although liquidity and transaction costs must also be
considered.

## 5. Why a short and a longer future can work

At very short maturities,

$$
A_s(h)\approx h,
\qquad
A_f(h)\approx h.
$$

The short future therefore has relatively strong fast-factor exposure. As
maturity increases, the fast loading quickly saturates near $1/\kappa_f$,
while the slow loading continues increasing because $\kappa_s$ is much lower.
The longer future consequently has relatively more slow-factor exposure.

Under the current production log-futures parameters,

$$
\kappa_s=0.3413,
\qquad
\kappa_f=16.7336,
$$

the factor half-lives are approximately

$$
H_s=495.6\text{ trading sessions},
\qquad
H_f=10.1\text{ trading sessions}.
$$

An indicative starting combination could therefore be:

- one liquid future with roughly 10--30 sessions remaining;
- one liquid future with roughly 80--150 sessions remaining.

These are not fixed optimal ranges. The actual choice must use the maturities
and liquidity of listed IM contracts at the hedge date.

## 6. Illustrative calculation

The current constant-log-futures IM2609 pricing example reports approximate
option factor sensitivities

$$
V_s=265.897,
\qquad
V_f=117.961.
$$

For illustration, suppose two futures are both around 7,500 index points and
have 20 and 100 trading sessions remaining. Using the current production
parameters gives

$$
G
\approx
\begin{bmatrix}
-606.235 & -2868.490\\
-334.492 & -447.729
\end{bmatrix}.
$$

Their fast-to-slow loading ratios are approximately

$$
R(20)=0.552,
\qquad
R(100)=0.156,
$$

so their risk directions are meaningfully different. Solving

$$
G
\begin{bmatrix}
n_1\\
n_2
\end{bmatrix}
=
-
\begin{bmatrix}
265.897\\
117.961
\end{bmatrix}
$$

gives

$$
n_1\approx0.319,
\qquad
n_2\approx0.025.
$$

In this simplified example, long positions of about 0.319 units of the
20-session future and 0.025 units of the 100-session future offset the option's
first-order slow and fast exposures. These are illustrative continuous units,
not executable contract recommendations. Actual quantities require observed
futures prices, contract multipliers, the option notional, and integer-contract
constraints.

## 7. Practical maturity-pair selection

For each hedge date:

1. List the liquid IM futures contracts and calculate their remaining trading
   sessions under the agreed exchange calendar.
2. Calculate $A_s(h_i)$ and $A_f(h_i)$ for every contract.
3. Build the two-factor exposure matrix for every candidate pair.
4. Reject pairs with very small determinant, small angular separation, or a
   large condition number.
5. Solve for the two hedge positions using current option sensitivities.
6. Reject or penalize solutions that require large gross notionals, illiquid
   contracts, or excessive turnover.
7. Backtest hedge P&L using realistic bid/ask spreads, daily or threshold-based
   rebalancing, contract expiry, and rolling.

The shortest and longest available contracts are not automatically the best
pair. Very short contracts may be noisy or about to expire, while very long
contracts may be illiquid. The objective is a compromise among factor
separation, absolute sensitivity, liquidity, basis risk, and trading cost.

## 8. More than two futures

With more than two liquid futures, exact inversion is no longer necessary. A
constrained weighted least-squares hedge can minimize residual factor exposure:

$$
\min_{\mathbf n}
\left\|
\mathbf v+G\mathbf n
\right\|_W^2
+\lambda_{turn}\|\mathbf n-\mathbf n_{old}\|^2
+\lambda_{gross}\|\mathbf n\|^2.
$$

This can incorporate:

- different importance weights for slow and fast risk;
- liquidity and bid/ask costs;
- turnover penalties;
- gross-position limits;
- integer or contract-count constraints;
- restrictions on using near-expiry contracts.

This approach is generally more stable than selecting exactly two contracts
when several liquid maturities are available.

## 9. Rebalancing and limitations

The hedge is dynamic. It should be recalculated because:

- futures maturities decrease each day;
- factor loadings change with maturity;
- the option's exercise boundary and factor sensitivities change;
- filtered factor states change as new curves arrive;
- contracts lose liquidity and must eventually be rolled.

The hedge removes only the model's local first-order exposure to $x_s$ and
$x_f$. It does not eliminate:

- observation-model and parameter-estimation risk;
- errors from treating historical OU parameters as risk-neutral parameters;
- calendar, expiry, basis, and asynchronous-close risk;
- nonlinear gamma and exercise-boundary effects;
- volatility, correlation, liquidity, and transaction-cost risk.

## 10. Current implementation status

The current pricing code reports separate slow and fast directional deltas and
their bump-and-value checks. It does not yet select two listed futures, construct
the cross-maturity exposure matrix, or output a joint two-contract hedge.

Implementing that extension would require current quotes and maturities for all
eligible hedge contracts, a pair-selection or constrained-optimization rule,
contract multipliers and integer rounding, and a historical hedge backtest.
