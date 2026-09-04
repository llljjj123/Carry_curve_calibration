"""Custom scalar-state Kalman filter for the one-factor OU carry model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, pi
from typing import Iterable

import numpy as np
import pandas as pd

from .observation import (
    ObservationNoiseModel,
    comparison_log_jacobian,
    normalize_observation_noise_model,
)


@dataclass(frozen=True)
class OUParams:
    """OU parameters, with time measured in 244-session years."""

    kappa: float
    theta: float
    eta: float
    sigma_epsilon: float

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("OU parameters must be finite")
        if self.kappa <= 0 or self.eta <= 0 or self.sigma_epsilon <= 0:
            raise ValueError("kappa, eta, and sigma_epsilon must be positive")


@dataclass
class FilterResult:
    log_likelihood: float
    states: pd.DataFrame
    raw_log_likelihood: float | None = None
    observation_noise_model: ObservationNoiseModel = "constant_carry"


def maturity_loading(kappa: float, tau: np.ndarray | Iterable[float] | float) -> np.ndarray:
    """Stable evaluation of ``(1-exp(-kappa*tau))/(kappa*tau)``."""
    x = kappa * np.asarray(tau, dtype=float)
    result = np.ones_like(x)
    nonzero = np.abs(x) > 1e-10
    result[nonzero] = -np.expm1(-x[nonzero]) / x[nonzero]
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("Non-finite maturity loading")
    return result


def transition_moments(params: OUParams, delta: float) -> tuple[float, float]:
    """Return exact OU autoregressive loading and innovation variance."""
    if delta < 0:
        raise ValueError("Observation dates must be nondecreasing")
    a = float(np.exp(-params.kappa * delta))
    q = float(params.eta**2 * (-np.expm1(-2.0 * params.kappa * delta)) / (2.0 * params.kappa))
    return a, max(q, 0.0)


def _session_gaps(dates: pd.Series, periods_per_year: int) -> np.ndarray:
    """Observation gaps based on dates present in the accepted market panel."""
    normalized = pd.to_datetime(dates).dt.normalize()
    gaps = normalized.diff().dt.days.to_numpy(dtype=float)
    # The pipeline supplies an exchange-session ordinal for exact gaps when available.
    return gaps / (365.25 if periods_per_year == 365 else periods_per_year * 365.25 / 244.0)


def kalman_filter(
    panel: pd.DataFrame,
    params: OUParams,
    *,
    periods_per_year: int = 244,
    initial_mean: float | None = None,
    initial_variance: float | None = None,
    initial_date: object | None = None,
    gap_function=None,
    observation_noise_model: ObservationNoiseModel = "constant_carry",
) -> FilterResult:
    """Filter a ragged panel of carry observations and return its log likelihood.

    ``gap_function(previous_date, current_date)`` should return the transition
    year fraction.  The pipeline supplies the same trading-day/244 convention
    used for maturity.  A weekday-scaled fallback is retained for standalone use.
    """
    params.validate()
    observation_noise_model = normalize_observation_noise_model(observation_noise_model)
    required = {"date", "tau", "implied_carry"}
    if observation_noise_model == "constant_log_futures":
        required.update({"futures_price", "spot", "risk_free_rate"})
    if not required.issubset(panel.columns):
        raise ValueError(f"Panel missing columns: {sorted(required - set(panel.columns))}")
    work = panel.dropna(subset=list(required)).sort_values(["date", "tau"]).copy()
    if work.empty:
        raise ValueError("No observations to filter")

    mean = params.theta if initial_mean is None else float(initial_mean)
    variance = params.eta**2 / (2.0 * params.kappa) if initial_variance is None else float(initial_variance)
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("Initial variance must be finite and positive")
    previous_date = pd.Timestamp(initial_date).normalize() if initial_date is not None else None
    raw_log_likelihood = 0.0
    log_likelihood = 0.0
    rows: list[dict[str, object]] = []
    obs_variance = params.sigma_epsilon**2

    for date_value, group in work.groupby("date", sort=True):
        current_date = pd.Timestamp(date_value).normalize()
        if previous_date is not None:
            if gap_function is not None:
                delta = float(gap_function(previous_date, current_date))
            else:
                delta = max((current_date - previous_date).days / 365.25, 0.0)
            a, q = transition_moments(params, delta)
            predicted_mean = params.theta + a * (mean - params.theta)
            predicted_variance = a * a * variance + q
        else:
            delta = np.nan
            predicted_mean, predicted_variance = mean, variance

        tau = group["tau"].to_numpy(dtype=float)
        b = maturity_loading(params.kappa, tau)
        if observation_noise_model == "constant_carry":
            observed = group["implied_carry"].to_numpy(dtype=float)
            observation_loading = b
            offset = params.theta * (1.0 - b)
        else:
            observed = (
                np.log(
                    group["futures_price"].to_numpy(dtype=float)
                    / group["spot"].to_numpy(dtype=float)
                )
                - group["risk_free_rate"].to_numpy(dtype=float) * tau
            )
            observation_loading = -tau * b
            offset = -tau * params.theta * (1.0 - b)
        centered_y = observed - offset
        residual = observed - (offset + predicted_mean * observation_loading)
        loading_square = float(observation_loading @ observation_loading)
        loading_residual = float(observation_loading @ residual)
        denom = 1.0 + predicted_variance * loading_square / obs_variance
        logdet = len(observed) * log(obs_variance) + log(denom)
        quad = float(residual @ residual) / obs_variance
        quad -= (
            predicted_variance
            * loading_residual
            * loading_residual
            / (obs_variance * obs_variance * denom)
        )
        date_raw_log_likelihood = -0.5 * (
            len(observed) * log(2.0 * pi) + logdet + quad
        )
        raw_log_likelihood += date_raw_log_likelihood
        log_likelihood += date_raw_log_likelihood + comparison_log_jacobian(
            observation_noise_model,
            tau,
        )

        precision = 1.0 / predicted_variance + loading_square / obs_variance
        filtered_variance = 1.0 / precision
        filtered_mean = filtered_variance * (
            predicted_mean / predicted_variance
            + float(observation_loading @ centered_y) / obs_variance
        )
        rows.append(
            {
                "date": current_date,
                "delta": delta,
                "n_contracts": len(group),
                "predicted_state": predicted_mean,
                "predicted_variance": predicted_variance,
                "predicted_std": np.sqrt(predicted_variance),
                "filtered_state": filtered_mean,
                "filtered_variance": filtered_variance,
                "filtered_std": np.sqrt(filtered_variance),
            }
        )
        mean, variance, previous_date = filtered_mean, filtered_variance, current_date

    return FilterResult(
        log_likelihood=float(log_likelihood),
        states=pd.DataFrame(rows),
        raw_log_likelihood=float(raw_log_likelihood),
        observation_noise_model=observation_noise_model,
    )
