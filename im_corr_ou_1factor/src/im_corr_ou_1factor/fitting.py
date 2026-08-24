"""Curve fits and analytical futures-price reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import OUParams, fitted_carry, log_futures_basis, maturity_loading


def attach_model_fits(
    panel: pd.DataFrame,
    states: pd.DataFrame,
    params: OUParams,
    sigma: float,
    *,
    model: str,
    variant: str,
) -> pd.DataFrame:
    result = panel.merge(states, on="date", how="left", validate="many_to_one", suffixes=("", "_state"))
    tau = result["tau"].to_numpy(dtype=float)
    result["model"] = model
    result["maturity_loading"] = maturity_loading(params.kappa, tau)
    result["predicted_carry"] = fitted_carry(result["predicted_state"], tau, params, sigma, variant=variant)
    result["fitted_carry"] = fitted_carry(result["filtered_state"], tau, params, sigma, variant=variant)
    result["predicted_futures_price"] = result["spot"] * np.exp(
        log_futures_basis(result["predicted_state"], tau, params, sigma, result["risk_free_rate"], variant=variant)
    )
    result["fitted_futures_price"] = result["spot"] * np.exp(
        log_futures_basis(result["filtered_state"], tau, params, sigma, result["risk_free_rate"], variant=variant)
    )
    result["carry_residual"] = result["implied_carry"] - result["fitted_carry"]
    result["carry_prediction_error"] = result["implied_carry"] - result["predicted_carry"]
    result["futures_residual"] = result["futures_price"] - result["fitted_futures_price"]
    result["futures_prediction_error"] = result["futures_price"] - result["predicted_futures_price"]
    return result

