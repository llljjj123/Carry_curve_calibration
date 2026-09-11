"""Optional compiled scalar-observation Kalman likelihood (no filtering outputs).

Independent observation errors allow sequential conditioning within each date.
This is the same joint Gaussian likelihood as the dense reference filter.
Numba is imported only when this optional backend is explicitly requested.
"""

import numpy as np


def _kernel(k_s, k_f, theta, eta_s, eta_f, sigma, offsets, gaps, tau, observed, log_model):
    m_s, m_f = 0.0, 0.0
    p_ss, p_ff, p_sf = eta_s**2 / (2*k_s), eta_f**2 / (2*k_f), 0.0
    noise = sigma**2
    likelihood = 0.0
    for day in range(len(gaps)):
        if day:
            ds, df = np.exp(-k_s*gaps[day]), np.exp(-k_f*gaps[day])
            m_s, m_f = ds*m_s, df*m_f
            p_ss = ds*ds*p_ss + eta_s**2 * (-np.expm1(-2*k_s*gaps[day])) / (2*k_s)
            p_ff = df*df*p_ff + eta_f**2 * (-np.expm1(-2*k_f*gaps[day])) / (2*k_f)
            p_sf *= ds*df
        for i in range(offsets[day], offsets[day+1]):
            a_s, a_f = -np.expm1(-k_s*tau[i])/k_s, -np.expm1(-k_f*tau[i])/k_f
            if log_model:
                h_s, h_f, offset = -a_s, -a_f, -theta*tau[i]
            else:
                h_s, h_f, offset = a_s/tau[i], a_f/tau[i], theta
            innovation = observed[i] - offset - h_s*m_s - h_f*m_f
            v_s, v_f = p_ss*h_s + p_sf*h_f, p_sf*h_s + p_ff*h_f
            variance = h_s*v_s + h_f*v_f + noise
            if not np.isfinite(variance) or variance <= 0:
                return -np.inf
            likelihood += -.5 * (np.log(2*np.pi) + np.log(variance) + innovation**2/variance)
            if log_model:
                likelihood += np.log(tau[i])
            m_s += v_s*innovation/variance
            m_f += v_f*innovation/variance
            p_ss -= v_s*v_s/variance
            p_sf -= v_s*v_f/variance
            p_ff -= v_f*v_f/variance
    return likelihood


_compiled = None


def make_fast_log_likelihood(dataset):
    global _compiled
    if _compiled is None:
        from numba import njit
        _compiled = njit(cache=False)(_kernel)
    offsets = np.r_[0, np.cumsum([len(g.tau) for g in dataset.groups])].astype(np.int64)
    tau = np.concatenate([g.tau for g in dataset.groups])
    log_model = dataset.observation_noise_model == "constant_log_futures"
    observed = np.concatenate([
        g.log_futures_observation if log_model else g.carry for g in dataset.groups
    ])
    if np.any(tau <= 0) or not np.isfinite(tau).all():
        raise ValueError("Maturities must be positive and finite")

    def evaluate(params):
        params.validate()
        return float(_compiled(
            params.kappa_slow, params.kappa_fast, params.theta,
            params.eta_slow, params.eta_fast, params.sigma_epsilon,
            offsets, dataset.deltas, tau, observed, log_model,
        ))
    return evaluate
