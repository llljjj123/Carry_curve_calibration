"""Exact two-factor OU Kalman filter with ragged daily carry curves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, pi
from typing import Callable

import numpy as np
import pandas as pd

from .kalman import maturity_loading


@dataclass(frozen=True)
class TwoFactorParams:
    """Identified slow/fast OU parameters with independent state shocks."""

    kappa_slow: float
    kappa_fast: float
    theta: float
    eta_slow: float
    eta_fast: float
    sigma_epsilon: float

    def validate(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Two-factor parameters must be finite")
        if self.kappa_slow <= 0 or self.kappa_fast <= self.kappa_slow:
            raise ValueError("Require 0 < kappa_slow < kappa_fast")
        if self.eta_slow <= 0 or self.eta_fast <= 0 or self.sigma_epsilon <= 0:
            raise ValueError("Volatilities must be positive")


@dataclass(frozen=True)
class CurveGroup:
    date: pd.Timestamp
    tau: np.ndarray
    carry: np.ndarray
    row_index: np.ndarray


@dataclass
class TwoFactorDataset:
    groups: list[CurveGroup]
    deltas: np.ndarray
    n_observations: int


@dataclass
class TwoFactorFilterResult:
    log_likelihood: float
    states: pd.DataFrame
    innovations: pd.DataFrame


def make_dataset(panel: pd.DataFrame, gap_function: Callable[[object, object], float]) -> TwoFactorDataset:
    """Pre-group a ragged panel once so optimizer likelihood calls are fast."""
    required = {"date", "tau", "implied_carry"}
    if not required.issubset(panel.columns):
        raise ValueError(f"Panel missing columns: {sorted(required - set(panel.columns))}")
    work = panel.dropna(subset=list(required)).sort_values(["date", "tau"]).copy()
    if work.empty:
        raise ValueError("No observations to filter")
    groups: list[CurveGroup] = []
    deltas = []
    previous_date = None
    for date_value, group in work.groupby("date", sort=True):
        current = pd.Timestamp(date_value).normalize()
        deltas.append(np.nan if previous_date is None else float(gap_function(previous_date, current)))
        groups.append(
            CurveGroup(
                date=current,
                tau=group["tau"].to_numpy(dtype=float),
                carry=group["implied_carry"].to_numpy(dtype=float),
                row_index=group.index.to_numpy(),
            )
        )
        previous_date = current
    return TwoFactorDataset(groups, np.asarray(deltas, dtype=float), len(work))


def stationary_covariance(params: TwoFactorParams) -> np.ndarray:
    """Stationary covariance for independent slow and fast OU shocks."""
    return np.diag(
        [
            params.eta_slow**2 / (2.0 * params.kappa_slow),
            params.eta_fast**2 / (2.0 * params.kappa_fast),
        ]
    )


def transition(params: TwoFactorParams, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact two-factor transition matrix and innovation covariance."""
    if delta < 0:
        raise ValueError("Observation dates must be nondecreasing")
    kappas = np.array([params.kappa_slow, params.kappa_fast])
    etas = np.array([params.eta_slow, params.eta_fast])
    loadings = np.exp(-kappas * delta)
    variances = etas**2 * (-np.expm1(-2.0 * kappas * delta)) / (2.0 * kappas)
    return np.diag(loadings), np.diag(np.maximum(variances, 0.0))


def _filter_dataset(
    dataset: TwoFactorDataset,
    params: TwoFactorParams,
    *,
    initial_mean: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
    collect: bool = True,
) -> TwoFactorFilterResult:
    params.validate()
    mean = np.zeros(2) if initial_mean is None else np.asarray(initial_mean, dtype=float).copy()
    covariance = stationary_covariance(params) if initial_covariance is None else np.asarray(initial_covariance, dtype=float).copy()
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("Two-factor initial state dimensions are invalid")
    observation_variance = params.sigma_epsilon**2
    log_likelihood = 0.0
    state_rows: list[dict[str, object]] = []
    innovation_rows: list[dict[str, object]] = []

    for group_number, (group, delta) in enumerate(zip(dataset.groups, dataset.deltas, strict=True)):
        if group_number:
            transition_matrix, process_covariance = transition(params, float(delta))
            predicted_mean = transition_matrix @ mean
            predicted_covariance = transition_matrix @ covariance @ transition_matrix.T + process_covariance
        else:
            predicted_mean, predicted_covariance = mean.copy(), covariance.copy()
        slow_loading = maturity_loading(params.kappa_slow, group.tau)
        fast_loading = maturity_loading(params.kappa_fast, group.tau)
        observation_matrix = np.column_stack([slow_loading, fast_loading])
        innovation = group.carry - params.theta - observation_matrix @ predicted_mean
        innovation_covariance = (
            observation_matrix @ predicted_covariance @ observation_matrix.T
            + observation_variance * np.eye(len(group.carry))
        )
        cholesky = np.linalg.cholesky(innovation_covariance)
        whitened_innovation = np.linalg.solve(cholesky, innovation)
        log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
        log_likelihood += -0.5 * (
            len(group.carry) * log(2.0 * pi)
            + log_determinant
            + float(whitened_innovation @ whitened_innovation)
        )
        solved_innovation = np.linalg.solve(cholesky.T, whitened_innovation)
        solved_observation = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, observation_matrix),
        )
        mean = predicted_mean + predicted_covariance @ observation_matrix.T @ solved_innovation
        covariance = (
            predicted_covariance
            - predicted_covariance @ observation_matrix.T @ solved_observation @ predicted_covariance
        )
        covariance = 0.5 * (covariance + covariance.T)
        if collect:
            for position, row_index in enumerate(group.row_index):
                innovation_rows.append(
                    {
                        "date": group.date,
                        "row_index": row_index,
                        "within_date_order": position,
                        "innovation": innovation[position],
                        "innovation_variance": innovation_covariance[position, position],
                        "standardized_innovation": whitened_innovation[position],
                    }
                )
        if collect:
            state_rows.append(
                {
                    "date": group.date,
                    "delta": delta,
                    "n_contracts": len(group.carry),
                    "predicted_slow_state": predicted_mean[0],
                    "predicted_fast_state": predicted_mean[1],
                    "predicted_instantaneous_carry": params.theta + predicted_mean.sum(),
                    "predicted_var_slow": predicted_covariance[0, 0],
                    "predicted_var_fast": predicted_covariance[1, 1],
                    "predicted_cov_slow_fast": predicted_covariance[0, 1],
                    "filtered_slow_state": mean[0],
                    "filtered_fast_state": mean[1],
                    "filtered_instantaneous_carry": params.theta + mean.sum(),
                    "filtered_var_slow": covariance[0, 0],
                    "filtered_var_fast": covariance[1, 1],
                    "filtered_cov_slow_fast": covariance[0, 1],
                    "filtered_instantaneous_variance": covariance.sum(),
                    "filtered_instantaneous_std": np.sqrt(max(covariance.sum(), 0.0)),
                }
            )
    return TwoFactorFilterResult(log_likelihood, pd.DataFrame(state_rows), pd.DataFrame(innovation_rows))


def two_factor_log_likelihood(dataset: TwoFactorDataset, params: TwoFactorParams) -> float:
    return _filter_dataset(dataset, params, collect=False).log_likelihood


def two_factor_kalman_filter(
    panel: pd.DataFrame,
    params: TwoFactorParams,
    *,
    gap_function: Callable[[object, object], float],
    initial_mean: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
) -> TwoFactorFilterResult:
    dataset = make_dataset(panel, gap_function)
    return _filter_dataset(
        dataset,
        params,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        collect=True,
    )
