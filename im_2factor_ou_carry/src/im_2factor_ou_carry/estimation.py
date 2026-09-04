"""Multi-start maximum-likelihood estimation and numerical uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .kalman import OUParams, kalman_filter
from .observation import (
    ObservationNoiseModel,
    native_noise_units,
    noise_parameter_name,
    normalize_observation_noise_model,
)


@dataclass
class EstimationResult:
    params: OUParams
    log_likelihood: float
    converged: bool
    message: str
    transformed_optimum: np.ndarray
    standard_errors: dict[str, float]
    hessian_stable: bool
    optimizer_runs: pd.DataFrame
    observation_noise_model: ObservationNoiseModel = "constant_carry"


def pack(params: OUParams) -> np.ndarray:
    return np.array([np.log(params.kappa), params.theta, np.log(params.eta), np.log(params.sigma_epsilon)])


def unpack(values: np.ndarray) -> OUParams:
    return OUParams(float(np.exp(values[0])), float(values[1]), float(np.exp(values[2])), float(np.exp(values[3])))


def initial_guesses(
    panel: pd.DataFrame,
    count: int,
    seed: int = 852,
    observation_noise_model: ObservationNoiseModel = "constant_carry",
) -> list[OUParams]:
    """Construct deterministic, data-scaled starts plus seeded perturbations."""
    observation_noise_model = normalize_observation_noise_model(observation_noise_model)
    carries = panel["implied_carry"].dropna().to_numpy(dtype=float)
    theta = float(np.clip(np.median(carries), -0.25, 0.25))
    grouped = panel.groupby("date")["implied_carry"].mean().sort_index()
    daily_diff = grouped.diff().dropna().std()
    eta = float(np.clip((daily_diff if np.isfinite(daily_diff) else 0.005) * np.sqrt(244), 0.01, 0.30))
    cross = panel.assign(_mean=panel.groupby("date")["implied_carry"].transform("mean"))
    sigma = float(np.clip((cross["implied_carry"] - cross["_mean"]).std(), 0.001, 0.10))
    if observation_noise_model == "constant_log_futures":
        sigma = float(np.clip(sigma * panel["tau"].median(), 1e-6, 0.02))
    kappas = [0.25, 0.75, 1.5, 3.0, 6.0, 12.0]
    starts = [OUParams(k, theta, eta, sigma) for k in kappas[: max(1, min(count, len(kappas)))]]
    rng = np.random.default_rng(seed)
    while len(starts) < count:
        starts.append(
            OUParams(
                float(np.exp(rng.uniform(np.log(0.15), np.log(15.0)))),
                float(theta + rng.normal(0, 0.02)),
                float(eta * np.exp(rng.normal(0, 0.5))),
                float(sigma * np.exp(rng.normal(0, 0.4))),
            )
        )
    return starts


def numerical_hessian(function: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """Central finite-difference Hessian with parameter-scaled steps."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    steps = 2e-4 * np.maximum(1.0, np.abs(x))
    hessian = np.empty((n, n), dtype=float)
    f0 = function(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        hessian[i, i] = (function(x + ei) - 2.0 * f0 + function(x - ei)) / steps[i] ** 2
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = steps[j]
            value = (
                function(x + ei + ej)
                - function(x + ei - ej)
                - function(x - ei + ej)
                + function(x - ei - ej)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = hessian[j, i] = value
    return hessian


def estimate_ou(
    panel: pd.DataFrame,
    *,
    gap_function,
    starts: int = 8,
    maxiter: int = 1200,
    seed: int = 852,
    compute_standard_errors: bool = True,
    observation_noise_model: ObservationNoiseModel = "constant_carry",
) -> EstimationResult:
    """Estimate OU parameters using bounded log-parameterization and multi-starts."""
    observation_noise_model = normalize_observation_noise_model(observation_noise_model)
    noise_bound = (
        (np.log(1e-5), np.log(0.50))
        if observation_noise_model == "constant_carry"
        else (np.log(1e-8), np.log(0.05))
    )
    bounds = [
        (np.log(0.01), np.log(50.0)),
        (-0.50, 0.50),
        (np.log(1e-4), np.log(1.0)),
        noise_bound,
    ]

    def objective(x: np.ndarray) -> float:
        try:
            value = -kalman_filter(
                panel,
                unpack(x),
                gap_function=gap_function,
                observation_noise_model=observation_noise_model,
            ).log_likelihood
            return value if np.isfinite(value) else 1e100
        except (ValueError, FloatingPointError, OverflowError):
            return 1e100

    results = []
    audit: list[dict[str, object]] = []
    for index, guess in enumerate(
        initial_guesses(panel, starts, seed, observation_noise_model)
    ):
        result = minimize(
            objective,
            pack(guess),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-7, "maxls": 50},
        )
        results.append(result)
        fitted = unpack(result.x)
        audit.append(
            {
                "start_id": index,
                "observation_noise_model": observation_noise_model,
                "noise_parameter_name": noise_parameter_name(observation_noise_model),
                **{f"start_{k}": v for k, v in asdict(guess).items()},
                **{f"estimate_{k}": v for k, v in asdict(fitted).items()},
                "log_likelihood": -float(result.fun),
                "converged": bool(result.success),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )
    finite = [result for result in results if np.isfinite(result.fun)]
    if not finite:
        raise RuntimeError("All optimizer starts failed")
    successful = [result for result in finite if result.success]
    best = min(successful or finite, key=lambda result: result.fun)
    optimum = np.asarray(best.x, dtype=float)
    params = unpack(optimum)

    standard_errors = {key: np.nan for key in asdict(params)}
    stable = False
    if compute_standard_errors:
        try:
            hessian = numerical_hessian(objective, optimum)
            eigenvalues = np.linalg.eigvalsh(hessian)
            covariance_transformed = np.linalg.inv(hessian)
            stable = bool(np.all(eigenvalues > 1e-7) and np.all(np.diag(covariance_transformed) >= 0))
            if stable:
                jacobian = np.diag([params.kappa, 1.0, params.eta, params.sigma_epsilon])
                covariance = jacobian @ covariance_transformed @ jacobian.T
                standard_errors = dict(zip(asdict(params), np.sqrt(np.diag(covariance)), strict=True))
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            stable = False
    return EstimationResult(
        params=params,
        observation_noise_model=observation_noise_model,
        log_likelihood=-float(best.fun),
        converged=bool(best.success),
        message=str(best.message),
        transformed_optimum=optimum,
        standard_errors=standard_errors,
        hessian_stable=stable,
        optimizer_runs=pd.DataFrame(audit).sort_values("log_likelihood", ascending=False),
    )


def parameter_table(result: EstimationResult) -> pd.DataFrame:
    values = asdict(result.params)
    sigma = values.pop("sigma_epsilon")
    sigma_se = result.standard_errors["sigma_epsilon"]
    parameter_name = noise_parameter_name(result.observation_noise_model)
    values[parameter_name] = sigma
    standard_errors = {
        key: result.standard_errors[key]
        for key in values
        if key != parameter_name
    }
    standard_errors[parameter_name] = sigma_se
    table = pd.DataFrame(
        {
            "parameter": list(values),
            "estimate": list(values.values()),
            "standard_error": [standard_errors[key] for key in values],
        }
    )
    table["observation_noise_model"] = result.observation_noise_model
    table["native_observation_units"] = native_noise_units(
        result.observation_noise_model
    )
    table["log_likelihood"] = result.log_likelihood
    table["optimizer_converged"] = result.converged
    table["optimizer_message"] = result.message
    table["hessian_stable"] = result.hessian_stable
    table["half_life_years"] = np.log(2.0) / result.params.kappa
    table["half_life_sessions"] = 244.0 * np.log(2.0) / result.params.kappa
    return table
