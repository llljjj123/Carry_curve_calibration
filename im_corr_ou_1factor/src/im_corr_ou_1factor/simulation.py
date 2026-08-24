"""Exact synthetic curve/return simulation for recovery tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import OUParams, fitted_carry, joint_interval_moments


def simulate_joint_panel(
    params: OUParams,
    *,
    sigma: float = 0.25,
    n_dates: int = 700,
    maturity_sessions: tuple[int, ...] = (15, 35, 70, 140, 250),
    periods_per_year: int = 244,
    seed: int = 852,
    missing_probability: float = 0.08,
    variant: str = "exact",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params.validate()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    state = rng.normal(params.theta, params.eta / np.sqrt(2 * params.kappa))
    spot = 5000.0
    state_rows = []
    observations = []
    delta = 1 / periods_per_year
    a, q, b, variance_return, covariance = joint_interval_moments(params, sigma, delta)
    covariance_matrix = np.array([[q, covariance], [covariance, variance_return]])
    for index, day in enumerate(dates):
        if index:
            mean_return = (params.mu - params.theta - 0.5 * sigma**2) * delta - (state - params.theta) * b
            innovation_state, innovation_return = rng.multivariate_normal([0.0, 0.0], covariance_matrix)
            return_value = mean_return + innovation_return
            state = params.theta + a * (state - params.theta) + innovation_state
            spot *= np.exp(return_value)
        state_rows.append({"date": day, "true_state": state, "spot": spot})
        for sessions in maturity_sessions:
            if rng.random() < missing_probability and len(maturity_sessions) > 1:
                continue
            tau = sessions / periods_per_year
            carry = float(fitted_carry(state, tau, params, sigma, variant=variant))
            carry += rng.normal(0, params.sigma_epsilon)
            observations.append(
                {
                    "date": day,
                    "contract": f"SYN{sessions:03d}",
                    "sessions_to_expiry": sessions,
                    "tau": tau,
                    "implied_carry": carry,
                    "spot": spot,
                    "risk_free_rate": 0.014,
                    "futures_price": spot * np.exp((0.014 - carry) * tau),
                }
            )
    return pd.DataFrame(observations), pd.DataFrame(state_rows)

