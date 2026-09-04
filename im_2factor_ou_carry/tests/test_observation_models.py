"""Regression tests for the configurable calibration observation equation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from im_2factor_ou_carry.estimation import EstimationResult, parameter_table
from im_2factor_ou_carry.kalman import OUParams, kalman_filter
from im_2factor_ou_carry.observation import (
    carry_noise_sd,
    normalize_observation_noise_model,
)
from im_2factor_ou_carry.plots import plot_rolling_parameters, plot_two_factor_rolling
from im_2factor_ou_carry.simulation import simulate_panel
from im_2factor_ou_carry.two_factor import TwoFactorParams, two_factor_kalman_filter
from im_2factor_ou_carry.two_factor_estimation import (
    TwoFactorEstimationResult,
    two_factor_parameter_table,
)
from im_2factor_ou_carry.two_factor_simulation import simulate_two_factor_panel


TAU = 40.0 / 244.0


def _gap(start, end):
    return np.busday_count(np.datetime64(start.date()), np.datetime64(end.date())) / 244


def _add_futures_columns(panel):
    result = panel.copy()
    result["spot"] = 7000.0
    result["risk_free_rate"] = 0.014
    result["futures_price"] = result["spot"] * np.exp(
        (result["risk_free_rate"] - result["implied_carry"]) * result["tau"]
    )
    return result


def test_two_factor_log_futures_matches_scaled_carry_filter() -> None:
    carry_params = TwoFactorParams(0.8, 14.0, 0.05, 0.08, 0.40, 0.003)
    panel, _ = simulate_two_factor_panel(
        carry_params,
        n_dates=80,
        maturity_sessions=(40,),
        missing_probability=0.0,
        seed=13,
    )
    panel = _add_futures_columns(panel)
    log_params = TwoFactorParams(
        carry_params.kappa_slow,
        carry_params.kappa_fast,
        carry_params.theta,
        carry_params.eta_slow,
        carry_params.eta_fast,
        carry_params.sigma_epsilon * TAU,
    )
    carry = two_factor_kalman_filter(panel, carry_params, gap_function=_gap)
    direct = two_factor_kalman_filter(
        panel,
        log_params,
        gap_function=_gap,
        observation_noise_model="constant_log_futures",
    )
    assert direct.log_likelihood == pytest.approx(carry.log_likelihood, abs=1e-8)
    assert np.allclose(
        direct.states[["filtered_slow_state", "filtered_fast_state"]],
        carry.states[["filtered_slow_state", "filtered_fast_state"]],
        atol=1e-10,
    )


def test_one_factor_log_futures_matches_scaled_carry_filter() -> None:
    carry_params = OUParams(2.0, 0.04, 0.08, 0.003)
    panel, _ = simulate_panel(
        carry_params,
        n_dates=80,
        maturity_sessions=(40,),
        missing_probability=0.0,
        seed=21,
    )
    panel = _add_futures_columns(panel)
    log_params = OUParams(
        carry_params.kappa,
        carry_params.theta,
        carry_params.eta,
        carry_params.sigma_epsilon * TAU,
    )
    carry = kalman_filter(panel, carry_params, gap_function=_gap)
    direct = kalman_filter(
        panel,
        log_params,
        gap_function=_gap,
        observation_noise_model="constant_log_futures",
    )
    assert direct.log_likelihood == pytest.approx(carry.log_likelihood, abs=1e-8)
    assert np.allclose(
        direct.states["filtered_state"],
        carry.states["filtered_state"],
        atol=1e-10,
    )


def test_log_futures_noise_converts_to_inverse_maturity_carry_noise() -> None:
    tau = np.array([10.0, 80.0]) / 244.0
    converted = carry_noise_sd("constant_log_futures", 0.001, tau)
    assert converted[0] / converted[1] == pytest.approx(8.0)
    with pytest.raises(ValueError):
        normalize_observation_noise_model("unsupported")


def test_parameter_tables_use_model_specific_noise_name() -> None:
    one_params = OUParams(2.0, 0.04, 0.08, 0.001)
    one = EstimationResult(
        params=one_params,
        log_likelihood=1.0,
        converged=True,
        message="ok",
        transformed_optimum=np.zeros(4),
        standard_errors={key: 0.1 for key in vars(one_params)},
        hessian_stable=True,
        optimizer_runs=None,
        observation_noise_model="constant_log_futures",
    )
    two_params = TwoFactorParams(0.8, 14.0, 0.05, 0.08, 0.40, 0.001)
    two = TwoFactorEstimationResult(
        params=two_params,
        log_likelihood=1.0,
        converged=True,
        message="ok",
        transformed_optimum=np.zeros(6),
        standard_errors={key: 0.1 for key in vars(two_params)},
        hessian_stable=True,
        optimizer_runs=None,
        observation_noise_model="constant_log_futures",
    )
    assert parameter_table(one)["parameter"].iloc[-1] == "sigma_log_futures"
    assert two_factor_parameter_table(two)["parameter"].iloc[-1] == "sigma_log_futures"


def test_rolling_plots_accept_log_futures_noise_column(tmp_path) -> None:
    dates = pd.to_datetime(["2026-01-01", "2026-02-01"])
    one = pd.DataFrame(
        {
            "window_end": dates,
            "kappa": [1.0, 1.1],
            "theta": [0.04, 0.05],
            "eta": [0.10, 0.11],
            "sigma_log_futures": [0.0010, 0.0011],
        }
    )
    two = pd.DataFrame(
        {
            "window_end": dates,
            "kappa_slow": [0.3, 0.4],
            "kappa_fast": [15.0, 16.0],
            "theta": [0.04, 0.05],
            "eta_slow": [0.05, 0.06],
            "eta_fast": [1.2, 1.3],
            "sigma_log_futures": [0.0010, 0.0011],
        }
    )
    one_path = tmp_path / "one.png"
    two_path = tmp_path / "two.png"
    plot_rolling_parameters(one, one_path)
    plot_two_factor_rolling(two, two_path)
    assert one_path.exists()
    assert two_path.exists()
