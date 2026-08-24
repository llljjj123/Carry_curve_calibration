"""Synthetic OU curve-panel generation for tests and examples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kalman import OUParams, maturity_loading, transition_moments


def simulate_panel(
    params: OUParams,
    *,
    n_dates: int = 600,
    maturity_sessions: tuple[int, ...] = (20, 45, 90, 180, 270),
    periods_per_year: int = 244,
    seed: int = 852,
    missing_probability: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a seeded daily latent state and ragged observed carry curves."""
    params.validate()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    state = rng.normal(params.theta, params.eta / np.sqrt(2.0 * params.kappa))
    state_rows = []
    observations = []
    a, q = transition_moments(params, 1.0 / periods_per_year)
    for index, day in enumerate(dates):
        if index:
            state = params.theta + a * (state - params.theta) + rng.normal(0.0, np.sqrt(q))
        state_rows.append({"date": day, "true_state": state})
        for sessions in maturity_sessions:
            if rng.random() < missing_probability and len(maturity_sessions) > 1:
                continue
            tau = sessions / periods_per_year
            loading = float(maturity_loading(params.kappa, tau))
            carry = params.theta + (state - params.theta) * loading + rng.normal(0.0, params.sigma_epsilon)
            observations.append(
                {
                    "date": day,
                    "contract": f"SYN{sessions:03d}",
                    "sessions_to_expiry": sessions,
                    "tau": tau,
                    "implied_carry": carry,
                }
            )
    return pd.DataFrame(observations), pd.DataFrame(state_rows)

