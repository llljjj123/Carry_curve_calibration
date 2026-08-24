"""Two-factor carry curves, futures reconstruction, and residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kalman import maturity_loading
from .two_factor import TwoFactorParams


def attach_two_factor_fits(panel: pd.DataFrame, states: pd.DataFrame, params: TwoFactorParams) -> pd.DataFrame:
    result = panel.merge(states, on="date", how="left", validate="many_to_one")
    slow = maturity_loading(params.kappa_slow, result["tau"].to_numpy(dtype=float))
    fast = maturity_loading(params.kappa_fast, result["tau"].to_numpy(dtype=float))
    result["slow_loading"] = slow
    result["fast_loading"] = fast
    result["predicted_carry"] = (
        params.theta
        + slow * result["predicted_slow_state"]
        + fast * result["predicted_fast_state"]
    )
    result["fitted_carry"] = (
        params.theta
        + slow * result["filtered_slow_state"]
        + fast * result["filtered_fast_state"]
    )
    result["predicted_futures_price"] = result["spot"] * np.exp(
        (result["risk_free_rate"] - result["predicted_carry"]) * result["tau"]
    )
    result["fitted_futures_price"] = result["spot"] * np.exp(
        (result["risk_free_rate"] - result["fitted_carry"]) * result["tau"]
    )
    result["carry_residual"] = result["implied_carry"] - result["fitted_carry"]
    result["carry_prediction_error"] = result["implied_carry"] - result["predicted_carry"]
    result["futures_residual"] = result["futures_price"] - result["fitted_futures_price"]
    result["futures_prediction_error"] = result["futures_price"] - result["predicted_futures_price"]
    predicted_variance = (
        slow**2 * result["predicted_var_slow"]
        + fast**2 * result["predicted_var_fast"]
        + 2.0 * slow * fast * result["predicted_cov_slow_fast"]
        + params.sigma_epsilon**2
    )
    result["marginal_prediction_variance"] = predicted_variance
    result["standardized_marginal_prediction_error"] = result["carry_prediction_error"] / np.sqrt(predicted_variance)
    return result

