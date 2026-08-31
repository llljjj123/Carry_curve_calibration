Context: Read `summary_for_AI.md` for a summary of the project. This time we would mainly focus on `carry_put_pricing` part.

Goal: Derive formula and algorithm for pricing the `delta` of the `carry_put_pricing`. Here by `delta` I mean the partial derivative to the current underlying futures price $F_{\tau,T}$

Todo: I would propose 2 methods down below, examine the methods and formula derivations. Report back to me after reading. After discussion and making sure that we are on the common ground, you can go coding. **DON'T** jump right into coding.

We have an American put option with exercise value
$$
\max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)} - e^{\left(r - q_{\tau,T}\right)\left(T-\tau\right)},0\right\}S_\tau
$$

Let's call the value of this put $V$. I wish to find the `delta` defined by $\Delta_\tau = \frac{\partial V}{\partial F_{\tau,T}}$.

I propose 2 methods:

1. hump-and-value. 

The most rudimentary method and I cheapest in calculation I believe. Since we are pricing on a whole grid, we just need to find near-by values of the nodes around the starting node and use them as the hump value. We just need to convert these values into $F_{0,T}$ based not $x_i$ and $x_j$ based.

2. path-wise derivation.

We have the exercise value, denote by $C_\tau$

$$
\begin{align}
C_\tau &= max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)} - e^{\left(r - q_{\tau,T}\right)\left(T-\tau\right)},0\right\}S_\tau \\
        &= max\left\{e^{\left(r - q_{0,T}\right)\left(T-\tau\right)}S_\tau - F_{\tau,T},0\right\}
\end{align}
$$
We take derivative to $F_{\tau,T}$ and we have
$$
\frac{\partial C_\tau}{\partial F_{\tau,T}} = \mathbb{\Theta}\left(e^{\left(r - q_{0,T}\right)\left(T-\tau\right)}S_\tau > F_{\tau,T}\right)\left(\exp\left[\left(q_{\tau,T}-q_{0,T}\right)\left(T-\tau\right)\right] - 1\right)
$$

Where $\mathbb{\Theta}$ is the Heaviside function.

So we use the same backward induction method and grid to price the above path-wise delta.

Make a comparison of the delta calculated by these 2 means at the end.

You may use environment `GUOYUAN`. If you need to install any package, use `conda install -c conda-forge`.