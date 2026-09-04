"""Two-factor OU filters with alternative maturity-dependent observation noise."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, pi
from typing import Callable, Literal

import numpy as np
import pandas as pd

from im_2factor_ou_carry.kalman import maturity_loading


NoiseModelName = Literal[
    "constant_carry",
    "two_bucket_carry",
    "smooth_carry_log",
    "constant_log_futures",
]

MODEL_NAMES: tuple[NoiseModelName, ...] = (
    "constant_carry",
    "two_bucket_carry",
    "smooth_carry_log",
    "constant_log_futures",
)

MODEL_LABELS: dict[NoiseModelName, str] = {
    "constant_carry": "Constant carry noise",
    "two_bucket_carry": "Two carry-noise buckets",
    "smooth_carry_log": "Smooth carry + log-price noise",
    "constant_log_futures": "Constant log-futures noise",
}


@dataclass(frozen=True)
class OUParams:
    """Independent centered slow/fast OU factors and their common level."""

    kappa_slow: float
    kappa_fast: float
    theta: float
    eta_slow: float
    eta_fast: float

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("OU parameters must be finite")
        if self.kappa_slow <= 0.0 or self.kappa_fast <= self.kappa_slow:
            raise ValueError("Require 0 < kappa_slow < kappa_fast")
        if self.eta_slow <= 0.0 or self.eta_fast <= 0.0:
            raise ValueError("OU volatilities must be positive")


@dataclass(frozen=True)
class ModelParams:
    """OU parameters plus observation-noise parameters for one specification."""

    model: NoiseModelName
    ou: OUParams
    noise: tuple[float, ...]

    def validate(self) -> None:
        if self.model not in MODEL_NAMES:
            raise ValueError(f"Unknown observation-noise model: {self.model}")
        self.ou.validate()
        expected = len(noise_parameter_names(self.model))
        if len(self.noise) != expected:
            raise ValueError(f"{self.model} needs {expected} noise parameters")
        values = np.asarray(self.noise, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Observation-noise parameters must be positive and finite")

    def as_dict(self) -> dict[str, float | str]:
        result: dict[str, float | str] = {"model": self.model, **asdict(self.ou)}
        result.update(dict(zip(noise_parameter_names(self.model), self.noise, strict=True)))
        return result


@dataclass(frozen=True)
class ObservationGroup:
    date: pd.Timestamp
    tau: np.ndarray
    sessions: np.ndarray
    carry: np.ndarray
    futures: np.ndarray
    spot: np.ndarray
    rate: np.ndarray
    row_index: np.ndarray


@dataclass(frozen=True)
class ObservationDataset:
    groups: tuple[ObservationGroup, ...]
    deltas: np.ndarray
    n_observations: int


@dataclass
class NoiseFilterResult:
    raw_log_likelihood: float
    comparison_log_likelihood: float
    states: pd.DataFrame
    predictions: pd.DataFrame
    date_likelihoods: pd.DataFrame


def noise_parameter_names(model: NoiseModelName) -> tuple[str, ...]:
    """Return named noise parameters in optimizer-vector order."""
    names: dict[NoiseModelName, tuple[str, ...]] = {
        "constant_carry": ("sigma_carry",),
        "two_bucket_carry": ("sigma_short", "sigma_long"),
        "smooth_carry_log": ("sigma_carry_floor", "sigma_log_futures"),
        "constant_log_futures": ("sigma_log_futures",),
    }
    return names[model]


def parameter_count(model: NoiseModelName) -> int:
    return 5 + len(noise_parameter_names(model))


def make_dataset(
    panel: pd.DataFrame,
    gap_function: Callable[[object, object], float],
) -> ObservationDataset:
    """Build a reusable ragged dataset with carry and log-futures observations."""
    required = {
        "date",
        "tau",
        "sessions_to_expiry",
        "implied_carry",
        "futures_price",
        "spot",
        "risk_free_rate",
    }
    if not required.issubset(panel.columns):
        raise ValueError(f"Panel missing columns: {sorted(required - set(panel.columns))}")
    work = panel.dropna(subset=list(required)).sort_values(["date", "tau"]).copy()
    if work.empty:
        raise ValueError("No observations to filter")
    groups: list[ObservationGroup] = []
    deltas: list[float] = []
    previous_date: pd.Timestamp | None = None
    for date_value, group in work.groupby("date", sort=True):
        current = pd.Timestamp(date_value).normalize()
        deltas.append(
            np.nan if previous_date is None else float(gap_function(previous_date, current))
        )
        groups.append(
            ObservationGroup(
                date=current,
                tau=group["tau"].to_numpy(dtype=float),
                sessions=group["sessions_to_expiry"].to_numpy(dtype=float),
                carry=group["implied_carry"].to_numpy(dtype=float),
                futures=group["futures_price"].to_numpy(dtype=float),
                spot=group["spot"].to_numpy(dtype=float),
                rate=group["risk_free_rate"].to_numpy(dtype=float),
                row_index=group.index.to_numpy(),
            )
        )
        previous_date = current
    return ObservationDataset(tuple(groups), np.asarray(deltas), len(work))


def stationary_covariance(params: OUParams) -> np.ndarray:
    return np.diag(
        [
            params.eta_slow**2 / (2.0 * params.kappa_slow),
            params.eta_fast**2 / (2.0 * params.kappa_fast),
        ]
    )


def transition(params: OUParams, delta: float) -> tuple[np.ndarray, np.ndarray]:
    if delta < 0.0:
        raise ValueError("Observation dates must be nondecreasing")
    kappas = np.array([params.kappa_slow, params.kappa_fast])
    etas = np.array([params.eta_slow, params.eta_fast])
    decay = np.exp(-kappas * delta)
    variances = etas**2 * (-np.expm1(-2.0 * kappas * delta)) / (2.0 * kappas)
    return np.diag(decay), np.diag(np.maximum(variances, 0.0))


def carry_noise_sd(
    model: NoiseModelName,
    noise: tuple[float, ...],
    tau: np.ndarray,
    sessions: np.ndarray,
) -> np.ndarray:
    """Observation-noise standard deviation expressed in annualized carry."""
    if model == "constant_carry":
        return np.full_like(tau, noise[0], dtype=float)
    if model == "two_bucket_carry":
        sigma_short, sigma_long = noise
        return np.where(sessions <= 15.0, sigma_short, sigma_long)
    if model == "smooth_carry_log":
        sigma_floor, sigma_log = noise
        return np.sqrt(sigma_floor**2 + (sigma_log / tau) ** 2)
    if model == "constant_log_futures":
        return noise[0] / tau
    raise ValueError(f"Unknown model: {model}")


def _observation_components(
    group: ObservationGroup,
    params: ModelParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return observation, offset, H, R diagonal, and carry-scale Jacobian."""
    slow_b = maturity_loading(params.ou.kappa_slow, group.tau)
    fast_b = maturity_loading(params.ou.kappa_fast, group.tau)
    if params.model == "constant_log_futures":
        observed = np.log(group.futures / group.spot) - group.rate * group.tau
        offset = -params.ou.theta * group.tau
        matrix = np.column_stack([-group.tau * slow_b, -group.tau * fast_b])
        variance = np.full_like(group.tau, params.noise[0] ** 2)
        comparison_adjustment = float(np.log(group.tau).sum())
        return observed, offset, matrix, variance, comparison_adjustment

    observed = group.carry
    offset = np.full_like(group.tau, params.ou.theta)
    matrix = np.column_stack([slow_b, fast_b])
    sigma = carry_noise_sd(params.model, params.noise, group.tau, group.sessions)
    return observed, offset, matrix, sigma**2, 0.0


def filter_dataset(
    dataset: ObservationDataset,
    params: ModelParams,
    *,
    collect: bool = True,
) -> NoiseFilterResult:
    """Evaluate likelihood and optionally collect states and native-unit predictions."""
    params.validate()
    mean = np.zeros(2)
    covariance = stationary_covariance(params.ou)
    raw_log_likelihood = 0.0
    comparison_log_likelihood = 0.0
    state_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    likelihood_rows: list[dict[str, object]] = []

    for group_number, (group, delta) in enumerate(zip(dataset.groups, dataset.deltas, strict=True)):
        if group_number:
            transition_matrix, process_covariance = transition(params.ou, float(delta))
            predicted_mean = transition_matrix @ mean
            predicted_covariance = (
                transition_matrix @ covariance @ transition_matrix.T + process_covariance
            )
        else:
            predicted_mean = mean.copy()
            predicted_covariance = covariance.copy()

        observed, offset, observation_matrix, noise_variance, adjustment = _observation_components(
            group, params
        )
        innovation = observed - offset - observation_matrix @ predicted_mean
        innovation_covariance = (
            observation_matrix @ predicted_covariance @ observation_matrix.T
            + np.diag(noise_variance)
        )
        cholesky = np.linalg.cholesky(innovation_covariance)
        whitened = np.linalg.solve(cholesky, innovation)
        log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
        date_raw = -0.5 * (
            len(observed) * log(2.0 * pi) + log_determinant + float(whitened @ whitened)
        )
        date_comparison = date_raw + adjustment
        raw_log_likelihood += date_raw
        comparison_log_likelihood += date_comparison

        solved_innovation = np.linalg.solve(cholesky.T, whitened)
        solved_observation = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, observation_matrix),
        )
        mean = predicted_mean + predicted_covariance @ observation_matrix.T @ solved_innovation
        covariance = (
            predicted_covariance
            - predicted_covariance
            @ observation_matrix.T
            @ solved_observation
            @ predicted_covariance
        )
        covariance = 0.5 * (covariance + covariance.T)

        if collect:
            slow_b = maturity_loading(params.ou.kappa_slow, group.tau)
            fast_b = maturity_loading(params.ou.kappa_fast, group.tau)
            predicted_carry = (
                params.ou.theta + slow_b * predicted_mean[0] + fast_b * predicted_mean[1]
            )
            fitted_carry = params.ou.theta + slow_b * mean[0] + fast_b * mean[1]
            predicted_futures = group.spot * np.exp((group.rate - predicted_carry) * group.tau)
            fitted_futures = group.spot * np.exp((group.rate - fitted_carry) * group.tau)
            sigma_carry = carry_noise_sd(
                params.model,
                params.noise,
                group.tau,
                group.sessions,
            )
            marginal_standardized = innovation / np.sqrt(np.diag(innovation_covariance))
            for position, row_index in enumerate(group.row_index):
                prediction_rows.append(
                    {
                        "date": group.date,
                        "row_index": row_index,
                        "within_date_order": position,
                        "predicted_carry": predicted_carry[position],
                        "fitted_carry": fitted_carry[position],
                        "carry_prediction_error": (
                            group.carry[position] - predicted_carry[position]
                        ),
                        "carry_residual": group.carry[position] - fitted_carry[position],
                        "predicted_futures_price": predicted_futures[position],
                        "fitted_futures_price": fitted_futures[position],
                        "futures_prediction_error": (
                            group.futures[position] - predicted_futures[position]
                        ),
                        "futures_residual": group.futures[position] - fitted_futures[position],
                        "observation_noise_carry_sd": sigma_carry[position],
                        "innovation_model_units": innovation[position],
                        "marginal_innovation_variance_model_units": (
                            innovation_covariance[position, position]
                        ),
                        "marginal_standardized_innovation": (marginal_standardized[position]),
                    }
                )
            state_rows.append(
                {
                    "date": group.date,
                    "delta": delta,
                    "n_contracts": len(group.carry),
                    "predicted_slow_state": predicted_mean[0],
                    "predicted_fast_state": predicted_mean[1],
                    "predicted_var_slow": predicted_covariance[0, 0],
                    "predicted_var_fast": predicted_covariance[1, 1],
                    "predicted_cov_slow_fast": predicted_covariance[0, 1],
                    "filtered_slow_state": mean[0],
                    "filtered_fast_state": mean[1],
                    "filtered_var_slow": covariance[0, 0],
                    "filtered_var_fast": covariance[1, 1],
                    "filtered_cov_slow_fast": covariance[0, 1],
                    "filtered_instantaneous_carry": params.ou.theta + mean.sum(),
                    "filtered_instantaneous_std": np.sqrt(max(covariance.sum(), 0.0)),
                }
            )
            likelihood_rows.append(
                {
                    "date": group.date,
                    "n_observations": len(group.carry),
                    "raw_log_likelihood": date_raw,
                    "comparison_log_likelihood": date_comparison,
                    "log_jacobian_to_carry_units": adjustment,
                }
            )

    return NoiseFilterResult(
        raw_log_likelihood=raw_log_likelihood,
        comparison_log_likelihood=comparison_log_likelihood,
        states=pd.DataFrame(state_rows),
        predictions=pd.DataFrame(prediction_rows),
        date_likelihoods=pd.DataFrame(likelihood_rows),
    )


def filter_panel(
    panel: pd.DataFrame,
    params: ModelParams,
    *,
    gap_function: Callable[[object, object], float],
) -> NoiseFilterResult:
    return filter_dataset(make_dataset(panel, gap_function), params, collect=True)


def log_likelihood(dataset: ObservationDataset, params: ModelParams) -> float:
    """Comparable likelihood expressed relative to annualized-carry observations."""
    return filter_dataset(dataset, params, collect=False).comparison_log_likelihood
