"""Multi-start maximum likelihood for the identified two-factor OU model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .estimation import numerical_hessian
from .observation import (
    ObservationNoiseModel,
    native_noise_units,
    noise_parameter_name,
    normalize_observation_noise_model,
)
from .two_factor import TwoFactorDataset, TwoFactorParams, make_dataset, two_factor_log_likelihood


@dataclass
class TwoFactorEstimationResult:
    params: TwoFactorParams
    log_likelihood: float
    converged: bool
    message: str
    transformed_optimum: np.ndarray
    standard_errors: dict[str, float]
    hessian_stable: bool
    optimizer_runs: pd.DataFrame
    observation_noise_model: ObservationNoiseModel = "constant_carry"


def pack(params: TwoFactorParams) -> np.ndarray:
    gap = params.kappa_fast - params.kappa_slow
    return np.array(
        [
            np.log(params.kappa_slow),
            np.log(gap),
            params.theta,
            np.log(params.eta_slow),
            np.log(params.eta_fast),
            np.log(params.sigma_epsilon),
        ]
    )


def unpack(values: np.ndarray) -> TwoFactorParams:
    slow = float(np.exp(values[0]))
    return TwoFactorParams(
        kappa_slow=slow,
        kappa_fast=slow + float(np.exp(values[1])),
        theta=float(values[2]),
        eta_slow=float(np.exp(values[3])),
        eta_fast=float(np.exp(values[4])),
        sigma_epsilon=float(np.exp(values[5])),
    )


def _parameter_bounds(
    eta_fast_upper_bound: float,
    kappa_gap_upper_bound: float = 60.0,
    observation_noise_model: ObservationNoiseModel = "constant_carry",
) -> list[tuple[float, float]]:
    """Optimizer bounds, with a separately configurable fast-factor volatility."""
    if not np.isfinite(eta_fast_upper_bound) or eta_fast_upper_bound <= 1e-4:
        raise ValueError("eta_fast_upper_bound must be finite and greater than 1e-4")
    if not np.isfinite(kappa_gap_upper_bound) or kappa_gap_upper_bound <= 0.01:
        raise ValueError("kappa_gap_upper_bound must be finite and greater than 0.01")
    observation_noise_model = normalize_observation_noise_model(observation_noise_model)
    noise_bound = (
        (np.log(1e-5), np.log(0.50))
        if observation_noise_model == "constant_carry"
        else (np.log(1e-8), np.log(0.05))
    )
    return [
        (np.log(0.01), np.log(20.0)),  # kappa_slow
        (np.log(0.01), np.log(kappa_gap_upper_bound)),  # kappa_fast - kappa_slow
        (-0.50, 0.50),  # theta
        (np.log(1e-4), np.log(3.0)),  # eta_slow
        (np.log(1e-4), np.log(eta_fast_upper_bound)),  # eta_fast
        noise_bound,
    ]


def initial_guesses(
    panel: pd.DataFrame,
    count: int,
    seed: int = 852,
    observation_noise_model: ObservationNoiseModel = "constant_carry",
) -> list[TwoFactorParams]:
    observation_noise_model = normalize_observation_noise_model(observation_noise_model)
    carry = panel["implied_carry"].dropna().to_numpy(dtype=float)
    theta = float(np.clip(np.median(carry), -0.25, 0.25))
    daily = panel.groupby("date")["implied_carry"].mean().sort_index()
    annualized_change = float(np.clip(daily.diff().dropna().std() * np.sqrt(244), 0.05, 1.5))
    cross = panel.assign(_daily=panel.groupby("date")["implied_carry"].transform("mean"))
    sigma = float(np.clip((cross["implied_carry"] - cross["_daily"]).std(), 0.001, 0.10))
    if observation_noise_model == "constant_log_futures":
        sigma = float(np.clip(sigma * panel["tau"].median(), 1e-6, 0.02))
    templates = [
        (0.15, 4.0, 0.25, 0.75),
        (0.35, 8.0, 0.35, 1.00),
        (0.75, 12.0, 0.50, 1.25),
        (1.50, 18.0, 0.65, 1.50),
        (3.00, 25.0, 0.80, 1.75),
        (5.00, 35.0, 1.00, 2.00),
    ]
    starts = [
        TwoFactorParams(slow, fast, theta, annualized_change * slow_scale, annualized_change * fast_scale, sigma)
        for slow, fast, slow_scale, fast_scale in templates[: min(count, len(templates))]
    ]
    rng = np.random.default_rng(seed)
    while len(starts) < count:
        slow = float(np.exp(rng.uniform(np.log(0.08), np.log(6.0))))
        fast = slow + float(np.exp(rng.uniform(np.log(1.0), np.log(40.0))))
        starts.append(
            TwoFactorParams(
                slow,
                min(fast, 60.0),
                float(theta + rng.normal(0, 0.02)),
                float(np.clip(annualized_change * np.exp(rng.normal(-0.5, 0.6)), 0.01, 2.5)),
                float(np.clip(annualized_change * np.exp(rng.normal(0.2, 0.6)), 0.01, 2.5)),
                float(np.clip(sigma * np.exp(rng.normal(0, 0.4)), 1e-4, 0.3)),
            )
        )
    return starts


def estimate_two_factor_ou(
    panel: pd.DataFrame,
    *,
    gap_function,
    starts: int = 12,
    maxiter: int = 1500,
    seed: int = 852,
    compute_standard_errors: bool = True,
    eta_fast_upper_bound: float = 3.0,
    kappa_gap_upper_bound: float = 60.0,
    observation_noise_model: ObservationNoiseModel = "constant_carry",
) -> TwoFactorEstimationResult:
    """Estimate independent slow/fast OU factors with enforced ordering."""
    observation_noise_model = normalize_observation_noise_model(observation_noise_model)
    dataset: TwoFactorDataset = make_dataset(
        panel,
        gap_function,
        observation_noise_model,
    )
    bounds = _parameter_bounds(
        eta_fast_upper_bound,
        kappa_gap_upper_bound,
        observation_noise_model,
    )

    def objective(values: np.ndarray) -> float:
        try:
            params = unpack(values)
            result = -two_factor_log_likelihood(dataset, params)
            return result if np.isfinite(result) else 1e100
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return 1e100

    optimizer_results = []
    audit = []
    for start_id, guess in enumerate(
        initial_guesses(panel, starts, seed, observation_noise_model)
    ):
        result = minimize(
            objective,
            pack(guess),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-7, "maxls": 50},
        )
        optimizer_results.append(result)
        fitted = unpack(result.x)
        audit.append(
            {
                "start_id": start_id,
                "observation_noise_model": observation_noise_model,
                "noise_parameter_name": noise_parameter_name(observation_noise_model),
                **{f"start_{key}": value for key, value in asdict(guess).items()},
                **{f"estimate_{key}": value for key, value in asdict(fitted).items()},
                "log_likelihood": -float(result.fun),
                "converged": bool(result.success),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )
    finite = [result for result in optimizer_results if np.isfinite(result.fun) and result.fun < 1e99]
    if not finite:
        raise RuntimeError("All two-factor optimizer starts failed")
    successful = [result for result in finite if result.success]
    best = min(successful or finite, key=lambda result: result.fun)
    optimum = np.asarray(best.x, dtype=float)
    params = unpack(optimum)
    standard_errors = {key: np.nan for key in asdict(params)}
    stable = False
    if compute_standard_errors:
        try:
            hessian = numerical_hessian(objective, optimum)
            covariance_transformed = np.linalg.inv(hessian)
            stable = bool(
                np.all(np.linalg.eigvalsh(hessian) > 1e-7)
                and np.all(np.diag(covariance_transformed) >= 0)
            )
            if stable:
                slow, gap = params.kappa_slow, params.kappa_fast - params.kappa_slow
                jacobian = np.zeros((6, 6))
                jacobian[0, 0] = slow
                jacobian[1, 0] = slow
                jacobian[1, 1] = gap
                jacobian[2, 2] = 1.0
                jacobian[3, 3] = params.eta_slow
                jacobian[4, 4] = params.eta_fast
                jacobian[5, 5] = params.sigma_epsilon
                covariance = jacobian @ covariance_transformed @ jacobian.T
                standard_errors = dict(zip(asdict(params), np.sqrt(np.diag(covariance)), strict=True))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            stable = False
    return TwoFactorEstimationResult(
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


def two_factor_parameter_table(result: TwoFactorEstimationResult) -> pd.DataFrame:
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
    table["slow_half_life_sessions"] = 244.0 * np.log(2.0) / result.params.kappa_slow
    table["fast_half_life_sessions"] = 244.0 * np.log(2.0) / result.params.kappa_fast
    return table
