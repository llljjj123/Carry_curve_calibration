"""Curve fitting, futures-price reconstruction, and residual exports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kalman import OUParams, maturity_loading


def attach_model_fits(panel: pd.DataFrame, states: pd.DataFrame, params: OUParams) -> pd.DataFrame:
    """Attach prior and filtered carry/price fits to every curve observation."""
    result = panel.merge(states, on="date", how="left", validate="many_to_one")
    b = maturity_loading(params.kappa, result["tau"].to_numpy(dtype=float))
    result["maturity_loading"] = b
    result["predicted_carry"] = params.theta + (result["predicted_state"] - params.theta) * b
    result["fitted_carry"] = params.theta + (result["filtered_state"] - params.theta) * b
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
    predicted_variance = b**2 * result["predicted_variance"] + params.sigma_epsilon**2
    result["marginal_prediction_variance"] = predicted_variance
    result["standardized_marginal_prediction_error"] = (
        result["carry_prediction_error"] / np.sqrt(predicted_variance)
    )
    return result
