"""Curve-only and exact joint curve/return Kalman filters."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Callable

import numpy as np
import pandas as pd

from .model import OUParams, curve_coefficients, joint_interval_moments, transition_moments


@dataclass
class FilterResult:
    log_likelihood: float
    curve_log_likelihood: float
    return_log_likelihood: float
    states: pd.DataFrame


def _prepared_groups(panel: pd.DataFrame, gap_function) -> list[tuple[pd.Timestamp, float, np.ndarray, np.ndarray, float]]:
    """Cache immutable daily arrays and internal gaps for repeated likelihood calls."""
    dates = pd.to_datetime(panel["date"])
    signature = (
        len(panel),
        int(panel["date"].nunique()),
        str(dates.min()),
        str(dates.max()),
        float(pd.to_numeric(panel["tau"], errors="coerce").sum()),
    )
    if panel.attrs.get("_corr_ou_filter_signature") == signature:
        cached = panel.attrs.get("_corr_ou_filter_groups")
        if cached is not None:
            return cached
    work = panel.dropna(subset=["date", "tau", "implied_carry", "spot"]).sort_values(["date", "tau"])
    groups: list[tuple[pd.Timestamp, float, np.ndarray, np.ndarray, float]] = []
    previous = None
    for date_value, group in work.groupby("date", sort=True):
        current = pd.Timestamp(date_value).normalize()
        spot_values = group["spot"].dropna().unique()
        if len(spot_values) != 1:
            raise ValueError(f"Expected one spot value on {current.date()}")
        delta = np.nan if previous is None else float(gap_function(previous, current))
        groups.append((
            current,
            float(spot_values[0]),
            group["implied_carry"].to_numpy(dtype=float),
            group["tau"].to_numpy(dtype=float),
            delta,
        ))
        previous = current
    panel.attrs["_corr_ou_filter_signature"] = signature
    panel.attrs["_corr_ou_filter_groups"] = groups
    return groups


def _curve_update(
    y: np.ndarray,
    tau: np.ndarray,
    predicted_mean: float,
    predicted_variance: float,
    params: OUParams,
    sigma: float,
    variant: str,
) -> tuple[float, float, float, float, float]:
    intercept, h = curve_coefficients(params, tau, sigma, variant=variant)
    innovation = y - intercept - h * predicted_mean
    observation_variance = params.sigma_epsilon**2
    h2 = float(h @ h)
    h_innovation = float(h @ innovation)
    denominator = 1.0 + predicted_variance * h2 / observation_variance
    logdet = len(y) * log(observation_variance) + log(denominator)
    quadratic = float(innovation @ innovation) / observation_variance
    quadratic -= (
        predicted_variance * h_innovation**2
        / (observation_variance**2 * denominator)
    )
    ll = -0.5 * (len(y) * log(2 * pi) + logdet + quadratic)
    filtered_variance = 1.0 / (1.0 / predicted_variance + h2 / observation_variance)
    filtered_mean = predicted_mean + filtered_variance * h_innovation / observation_variance
    standardized = float(
        np.mean(innovation / np.sqrt(observation_variance + predicted_variance * h**2))
    )
    return filtered_mean, max(filtered_variance, 1e-14), float(ll), float(np.mean(innovation)), standardized


def kalman_filter(
    panel: pd.DataFrame,
    params: OUParams,
    *,
    sigma: float,
    gap_function: Callable[[object, object], float],
    mode: str = "curve",
    variant: str = "exact",
    initial_mean: float | None = None,
    initial_variance: float | None = None,
    initial_date: object | None = None,
    initial_spot: float | None = None,
) -> FilterResult:
    """Filter a ragged curve panel, optionally adding exact close-to-close returns."""
    params.validate()
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if mode not in {"curve", "joint"}:
        raise ValueError("mode must be 'curve' or 'joint'")
    required = {"date", "tau", "implied_carry", "spot"}
    if not required.issubset(panel.columns):
        raise ValueError(f"Panel missing columns: {sorted(required - set(panel.columns))}")
    groups = _prepared_groups(panel, gap_function)
    if not groups:
        raise ValueError("No observations to filter")

    mean = params.theta if initial_mean is None else float(initial_mean)
    variance = params.eta**2 / (2 * params.kappa) if initial_variance is None else float(initial_variance)
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("Initial variance must be finite and positive")
    previous_date = pd.Timestamp(initial_date).normalize() if initial_date is not None else None
    previous_spot = float(initial_spot) if initial_spot is not None else None
    curve_ll = 0.0
    return_ll = 0.0
    rows: list[dict[str, object]] = []

    for group_index, (current_date, current_spot, y, tau, internal_delta) in enumerate(groups):
        return_value = np.nan
        return_residual = np.nan
        standardized_return = np.nan
        interval_return_ll = 0.0
        transition_cross = np.nan

        if previous_date is None:
            delta = np.nan
            predicted_mean, predicted_variance = mean, variance
        else:
            delta = (
                float(internal_delta)
                if group_index > 0 and np.isfinite(internal_delta)
                else float(gap_function(previous_date, current_date))
            )
            if delta <= 0:
                raise ValueError("Observation gaps must be positive")
            a, q = transition_moments(params, delta)
            unconditional_mean = params.theta + a * (mean - params.theta)
            unconditional_variance = a * a * variance + q
            transition_cross = a * variance
            predicted_mean, predicted_variance = unconditional_mean, unconditional_variance

            if mode == "joint":
                if previous_spot is None or previous_spot <= 0:
                    raise ValueError("A positive previous spot is required in joint mode")
                a, q, b, variance_return, covariance_wr = joint_interval_moments(params, sigma, delta)
                return_value = log(current_spot / previous_spot)
                h_return = -b
                intercept_return = (params.mu - params.theta - 0.5 * sigma**2) * delta + params.theta * b
                expected_return = intercept_return + h_return * mean
                return_residual = return_value - expected_return
                total_return_variance = h_return**2 * variance + variance_return
                state_return_covariance = a * variance * h_return + covariance_wr
                interval_return_ll = -0.5 * (
                    log(2 * pi * total_return_variance) + return_residual**2 / total_return_variance
                )
                standardized_return = return_residual / np.sqrt(total_return_variance)
                predicted_mean = unconditional_mean + state_return_covariance / total_return_variance * return_residual
                predicted_variance = unconditional_variance - state_return_covariance**2 / total_return_variance
                transition_cross = a * variance - (
                    variance * h_return * state_return_covariance / total_return_variance
                )
                if predicted_variance <= 0:
                    raise ValueError("Conditional state variance is not positive")
                return_ll += interval_return_ll

        filtered_mean, filtered_variance, interval_curve_ll, curve_innovation, standardized_curve = _curve_update(
            y,
            tau,
            predicted_mean,
            predicted_variance,
            params,
            sigma,
            variant,
        )
        curve_ll += interval_curve_ll
        rows.append(
            {
                "date": current_date,
                "delta": delta,
                "spot": current_spot,
                "n_contracts": len(y),
                "predicted_state": predicted_mean,
                "predicted_variance": predicted_variance,
                "predicted_std": np.sqrt(predicted_variance),
                "filtered_state": filtered_mean,
                "filtered_variance": filtered_variance,
                "filtered_std": np.sqrt(filtered_variance),
                "transition_cross_covariance": transition_cross,
                "curve_log_likelihood": interval_curve_ll,
                "return_log_likelihood": interval_return_ll,
                "curve_innovation_mean": curve_innovation,
                "standardized_curve_innovation": standardized_curve,
                "log_return": return_value,
                "return_residual": return_residual,
                "standardized_return_residual": standardized_return,
            }
        )
        mean, variance = filtered_mean, filtered_variance
        previous_date, previous_spot = current_date, current_spot

    states = pd.DataFrame(rows)
    return FilterResult(float(curve_ll + return_ll), float(curve_ll), float(return_ll), states)


def smooth_states(states: pd.DataFrame) -> pd.DataFrame:
    """Apply a scalar RTS smoother using stored return-conditioned transitions."""
    result = states.copy().sort_values("date").reset_index(drop=True)
    result["smoothed_state"] = result["filtered_state"]
    result["smoothed_variance"] = result["filtered_variance"]
    for index in range(len(result) - 2, -1, -1):
        next_row = result.iloc[index + 1]
        gain = float(next_row["transition_cross_covariance"] / next_row["predicted_variance"])
        result.loc[index, "smoothed_state"] = result.loc[index, "filtered_state"] + gain * (
            result.loc[index + 1, "smoothed_state"] - next_row["predicted_state"]
        )
        result.loc[index, "smoothed_variance"] = result.loc[index, "filtered_variance"] + gain**2 * (
            result.loc[index + 1, "smoothed_variance"] - next_row["predicted_variance"]
        )
    result["smoothed_variance"] = result["smoothed_variance"].clip(lower=0)
    result["smoothed_std"] = np.sqrt(result["smoothed_variance"])
    return result
