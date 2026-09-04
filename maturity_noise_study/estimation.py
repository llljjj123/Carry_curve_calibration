"""Multi-start maximum likelihood for all observation-noise specifications."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

from im_2factor_ou_carry.two_factor_estimation import initial_guesses

from noise_models import (
    ModelParams,
    NoiseModelName,
    OUParams,
    log_likelihood,
    make_dataset,
)


@dataclass
class EstimationResult:
    params: ModelParams
    comparison_log_likelihood: float
    converged: bool
    message: str
    transformed_optimum: np.ndarray
    optimizer_runs: pd.DataFrame


def parameter_bounds(
    model: NoiseModelName,
    *,
    kappa_gap_upper_bound: float,
    eta_fast_upper_bound: float,
) -> list[tuple[float, float]]:
    if kappa_gap_upper_bound <= 0.01 or not np.isfinite(kappa_gap_upper_bound):
        raise ValueError("kappa_gap_upper_bound must be finite and greater than 0.01")
    if eta_fast_upper_bound <= 1e-4 or not np.isfinite(eta_fast_upper_bound):
        raise ValueError("eta_fast_upper_bound must be finite and greater than 1e-4")
    core = [
        (np.log(0.01), np.log(20.0)),
        (np.log(0.01), np.log(kappa_gap_upper_bound)),
        (-0.50, 0.50),
        (np.log(1e-4), np.log(3.0)),
        (np.log(1e-4), np.log(eta_fast_upper_bound)),
    ]
    carry_bound = (np.log(1e-7), np.log(0.50))
    log_price_bound = (np.log(1e-8), np.log(0.05))
    noise_bounds = {
        "constant_carry": [carry_bound],
        "two_bucket_carry": [carry_bound, carry_bound],
        "smooth_carry_log": [carry_bound, log_price_bound],
        "constant_log_futures": [log_price_bound],
    }
    return [*core, *noise_bounds[model]]


def pack(params: ModelParams) -> np.ndarray:
    gap = params.ou.kappa_fast - params.ou.kappa_slow
    return np.array(
        [
            np.log(params.ou.kappa_slow),
            np.log(gap),
            params.ou.theta,
            np.log(params.ou.eta_slow),
            np.log(params.ou.eta_fast),
            *(np.log(value) for value in params.noise),
        ]
    )


def unpack(values: np.ndarray, model: NoiseModelName) -> ModelParams:
    slow = float(np.exp(values[0]))
    return ModelParams(
        model=model,
        ou=OUParams(
            kappa_slow=slow,
            kappa_fast=slow + float(np.exp(values[1])),
            theta=float(values[2]),
            eta_slow=float(np.exp(values[3])),
            eta_fast=float(np.exp(values[4])),
        ),
        noise=tuple(float(np.exp(value)) for value in values[5:]),
    )


def _noise_from_carry_scale(
    model: NoiseModelName,
    sigma_carry: float,
    median_tau: float,
) -> tuple[float, ...]:
    if model == "constant_carry":
        return (sigma_carry,)
    if model == "two_bucket_carry":
        return (1.5 * sigma_carry, 0.8 * sigma_carry)
    if model == "smooth_carry_log":
        return (0.7 * sigma_carry, 0.7 * sigma_carry * median_tau)
    if model == "constant_log_futures":
        return (sigma_carry * median_tau,)
    raise ValueError(f"Unknown model: {model}")


def _convert_core_guess(
    guess: object,
    model: NoiseModelName,
    median_tau: float,
) -> ModelParams:
    return ModelParams(
        model=model,
        ou=OUParams(
            kappa_slow=float(guess.kappa_slow),
            kappa_fast=float(guess.kappa_fast),
            theta=float(guess.theta),
            eta_slow=float(guess.eta_slow),
            eta_fast=float(guess.eta_fast),
        ),
        noise=_noise_from_carry_scale(
            model,
            float(guess.sigma_epsilon),
            median_tau,
        ),
    )


def convert_warm_params(
    source: ModelParams,
    target_model: NoiseModelName,
    median_tau: float,
) -> ModelParams:
    """Map a fitted model into a defensible starting point for another model."""
    tau = np.array([median_tau])
    sessions = np.array([median_tau * 244.0])
    from noise_models import carry_noise_sd

    sigma = float(carry_noise_sd(source.model, source.noise, tau, sessions)[0])
    return ModelParams(
        model=target_model,
        ou=source.ou,
        noise=_noise_from_carry_scale(target_model, sigma, median_tau),
    )


def _clip_to_bounds(
    params: ModelParams,
    bounds: list[tuple[float, float]],
) -> ModelParams:
    values = pack(params)
    clipped = np.array(
        [
            np.clip(value, lower + 1e-10, upper - 1e-10)
            for value, (lower, upper) in zip(values, bounds, strict=True)
        ]
    )
    return unpack(clipped, params.model)


def _best_result(results: list[OptimizeResult]) -> OptimizeResult:
    finite = [result for result in results if np.isfinite(result.fun) and result.fun < 1e99]
    if not finite:
        raise RuntimeError("All optimizer starts failed")
    successful = [result for result in finite if result.success]
    return min(successful or finite, key=lambda result: result.fun)


def estimate_model(
    panel: pd.DataFrame,
    model: NoiseModelName,
    *,
    gap_function,
    starts: int = 12,
    maxiter: int = 1500,
    seed: int = 852,
    kappa_gap_upper_bound: float = 120.0,
    eta_fast_upper_bound: float = 6.0,
    warm_params: ModelParams | None = None,
) -> EstimationResult:
    """Estimate one observation-noise model with identical OU parameter bounds."""
    if starts < 1:
        raise ValueError("starts must be at least one")
    dataset = make_dataset(panel, gap_function)
    bounds = parameter_bounds(
        model,
        kappa_gap_upper_bound=kappa_gap_upper_bound,
        eta_fast_upper_bound=eta_fast_upper_bound,
    )
    median_tau = float(panel["tau"].median())

    def objective(values: np.ndarray) -> float:
        try:
            params = unpack(values, model)
            value = -log_likelihood(dataset, params)
            return float(value) if np.isfinite(value) else 1e100
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return 1e100

    guesses: list[ModelParams] = []
    if warm_params is not None:
        guesses.append(convert_warm_params(warm_params, model, median_tau))
    core_guesses = initial_guesses(panel, max(starts - len(guesses), 0), seed)
    guesses.extend(_convert_core_guess(guess, model, median_tau) for guess in core_guesses)
    guesses = [_clip_to_bounds(guess, bounds) for guess in guesses[:starts]]

    optimizer_results: list[OptimizeResult] = []
    audit_rows: list[dict[str, object]] = []
    for start_id, guess in enumerate(guesses):
        result = minimize(
            objective,
            pack(guess),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-7, "maxls": 50},
        )
        fitted = unpack(result.x, model)
        optimizer_results.append(result)
        audit_rows.append(
            {
                "model": model,
                "start_id": start_id,
                **{f"start_{key}": value for key, value in guess.as_dict().items()},
                **{f"estimate_{key}": value for key, value in fitted.as_dict().items()},
                "estimate_kappa_gap": fitted.ou.kappa_fast - fitted.ou.kappa_slow,
                "comparison_log_likelihood": -float(result.fun),
                "converged": bool(result.success),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )

    best = _best_result(optimizer_results)
    optimum = np.asarray(best.x, dtype=float)
    return EstimationResult(
        params=unpack(optimum, model),
        comparison_log_likelihood=-float(best.fun),
        converged=bool(best.success),
        message=str(best.message),
        transformed_optimum=optimum,
        optimizer_runs=pd.DataFrame(audit_rows).sort_values(
            "comparison_log_likelihood", ascending=False
        ),
    )
