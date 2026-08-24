"""Seeded synthetic two-factor OU curve generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kalman import maturity_loading
from .two_factor import TwoFactorParams, transition


def simulate_two_factor_panel(
    params: TwoFactorParams,
    *,
    n_dates: int = 900,
    maturity_sessions: tuple[int, ...] = (15, 35, 70, 130, 220),
    periods_per_year: int = 244,
    seed: int = 852,
    missing_probability: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params.validate()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_dates)
    state = rng.multivariate_normal(np.zeros(2), np.diag([
        params.eta_slow**2 / (2 * params.kappa_slow),
        params.eta_fast**2 / (2 * params.kappa_fast),
    ]))
    a, q = transition(params, 1.0 / periods_per_year)
    observations = []
    states = []
    for index, day in enumerate(dates):
        if index:
            state = a @ state + rng.multivariate_normal(np.zeros(2), q)
        states.append({"date": day, "true_slow_state": state[0], "true_fast_state": state[1]})
        for sessions in maturity_sessions:
            if rng.random() < missing_probability:
                continue
            tau = sessions / periods_per_year
            slow = float(maturity_loading(params.kappa_slow, tau))
            fast = float(maturity_loading(params.kappa_fast, tau))
            carry = params.theta + slow * state[0] + fast * state[1] + rng.normal(0, params.sigma_epsilon)
            observations.append(
                {
                    "date": day,
                    "contract": f"SYN{sessions:03d}",
                    "sessions_to_expiry": sessions,
                    "tau": tau,
                    "implied_carry": carry,
                }
            )
    return pd.DataFrame(observations), pd.DataFrame(states)

