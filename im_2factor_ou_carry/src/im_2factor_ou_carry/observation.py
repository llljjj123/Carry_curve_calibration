"""Observation-space conventions shared by the OU calibration models."""

from __future__ import annotations

from typing import Literal

import numpy as np


ObservationNoiseModel = Literal["constant_carry", "constant_log_futures"]
OBSERVATION_NOISE_MODELS: tuple[ObservationNoiseModel, ...] = (
    "constant_carry",
    "constant_log_futures",
)


def normalize_observation_noise_model(value: object) -> ObservationNoiseModel:
    model = str(value).strip().lower()
    if model not in OBSERVATION_NOISE_MODELS:
        choices = ", ".join(OBSERVATION_NOISE_MODELS)
        raise ValueError(f"observation_noise_model must be one of: {choices}")
    return model  # type: ignore[return-value]


def noise_parameter_name(model: ObservationNoiseModel) -> str:
    return "sigma_epsilon" if model == "constant_carry" else "sigma_log_futures"


def native_noise_units(model: ObservationNoiseModel) -> str:
    return "annualized_carry" if model == "constant_carry" else "log_futures_price"


def carry_noise_sd(
    model: ObservationNoiseModel,
    sigma_observation: float,
    tau: np.ndarray,
) -> np.ndarray:
    """Return native observation noise expressed as annualized-carry SD."""
    maturity = np.asarray(tau, dtype=float)
    if np.any(maturity <= 0.0):
        raise ValueError("Maturities must be positive")
    if model == "constant_carry":
        return np.full_like(maturity, sigma_observation)
    return np.full_like(maturity, sigma_observation) / maturity


def comparison_log_jacobian(
    model: ObservationNoiseModel,
    tau: np.ndarray,
) -> float:
    """Adjustment expressing a native log-price density in carry units."""
    if model == "constant_carry":
        return 0.0
    maturity = np.asarray(tau, dtype=float)
    if np.any(maturity <= 0.0):
        raise ValueError("Maturities must be positive")
    return float(np.log(maturity).sum())
