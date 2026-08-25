# Scenario explained
Consider such a scenario, at time $t = 0$, the spot price is $S_0$ and futures price $F_{0,T}$, where $S_0 > F_{0,T}$. A client buys a product that promises to deliver at maturity $t = T$, the change of the underlying spot $S_T - S_0$ plus the initially locked spot-futures gap $S_0 - F_{0,T}$. We define notation $q_{0,T}$ as the prevailing implied carry rate, where $F_{0,T} = \exp\left[\left(r-q_{0,T}\right)T\right]S_0$. Given a stochastic carry rate $c_t$, we know $q_{0,T} = \frac{1}{T}\int_0^T c_udu$

On top of that he also receives a American put option that allows him to exit the position at any time $\tau$ using the pre-defined carry rate $q_{0,T}$ instead of the time $\tau$ carry rate $q_{\tau,T}$ if $q_{0,T} < q_{\tau,T}$

So putting everything together, the client buys such a product that pays if exercised at any time $\tau$ 
$$
\max\left\{\exp\left[\left(r-q_{0,T}\right)\left(T - \tau\right)\right]S_\tau, F_{\tau,T}\right\} - F_{0,T}
$$

# Option property

Following the above formula, we have


$$
\max\left\{\exp\left[\left(r-q_{0,T}\right)\left(T - \tau\right)\right]S_\tau, F_{\tau,T}\right\} - F_{0,T}
= \max\left\{\exp\left[\left(r-q_{0,T}\right)\left(T - \tau\right)\right]S_\tau - F_{\tau,T}, 0\right\} + F_{\tau,T}  - F_{0,T} 
$$

The $ F_{\tau,T}  - F_{0,T} $ doesn't matter for it vanishes under expectation and it's stationarily hedgable. So the problem lands at how to evaluate such an American put option that pays

$$
\max\left\{\exp\left[\left(r-q_{0,T}\right)\left(T - \tau\right)\right]S_\tau - F_{\tau,T}, 0\right\} = \max\left\{\exp\left[\left(r-q_{0,T}\right)\left(T - \tau\right)\right]S_\tau - \exp\left[\left(r-q_{\tau,T}\right)\left(T - \tau\right)\right]S_\tau, 0\right\}
$$

when exercised.

# Pricing scheme

In the above equation, we have 2 stochastic sources. The one in $q_{\tau,T}$ and the one in $S_\tau$. We assume the correlation between the brownian motions behind $c_t$ and $S_t$ is $0$. For $c_t$, we use a 2-factor OU process. For $S_t$, we first assume a GBM with constant vol $\sigma$. We propose a binomial tree scheme for the pricing of such American option.