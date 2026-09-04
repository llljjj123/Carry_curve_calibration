"""Configurable estimation tools for the fast-factor boundary study.

The production two-factor implementation deliberately keeps its original
parameter bounds.  This module reuses its validated likelihood, dataset, and
filter code while making the fast/slow kappa gap configurable for diagnosis.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

from im_2factor_ou_carry.estimation import numerical_hessian
from im_2factor_ou_carry.two_factor import (
    TwoFactorDataset,
    TwoFactorParams,
    make_dataset,
    two_factor_log_likelihood,
)
from im_2factor_ou_carry.two_factor_estimation import (
    TwoFactorEstimationResult,
    initial_guesses,
)


def pack(params: TwoFactorParams) -> np.ndarray:
    """Transform an ordered two-factor parameter vector to optimizer space."""
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
    """Transform optimizer coordinates to ordered model parameters."""
    slow = float(np.exp(values[0]))
    return TwoFactorParams(
        kappa_slow=slow,
        kappa_fast=slow + float(np.exp(values[1])),
        theta=float(values[2]),
        eta_slow=float(np.exp(values[3])),
        eta_fast=float(np.exp(values[4])),
        sigma_epsilon=float(np.exp(values[5])),
    )


def parameter_bounds(
    eta_fast_upper_bound: float,
    kappa_gap_upper_bound: float,
) -> list[tuple[float, float]]:
    """Return bounds without the production estimator's hidden fast-kappa cap."""
    if not np.isfinite(eta_fast_upper_bound) or eta_fast_upper_bound <= 1e-4:
        raise ValueError("eta_fast_upper_bound must be finite and greater than 1e-4")
    if not np.isfinite(kappa_gap_upper_bound) or kappa_gap_upper_bound <= 0.01:
        raise ValueError("kappa_gap_upper_bound must be finite and greater than 0.01")
    return [
        (np.log(0.01), np.log(20.0)),
        (np.log(0.01), np.log(kappa_gap_upper_bound)),
        (-0.50, 0.50),
        (np.log(1e-4), np.log(3.0)),
        (np.log(1e-4), np.log(eta_fast_upper_bound)),
        (np.log(1e-5), np.log(0.50)),
    ]


def _clip_guess_to_bounds(
    guess: TwoFactorParams,
    bounds: list[tuple[float, float]],
) -> TwoFactorParams:
    values = pack(guess)
    clipped = np.array(
        [
            np.clip(value, lower + 1e-10, upper - 1e-10)
            for value, (lower, upper) in zip(values, bounds, strict=True)
        ]
    )
    return unpack(clipped)


def _best_result(results: list[OptimizeResult]) -> OptimizeResult:
    finite = [result for result in results if np.isfinite(result.fun) and result.fun < 1e99]
    if not finite:
        raise RuntimeError("All two-factor optimizer starts failed")
    successful = [result for result in finite if result.success]
    return min(successful or finite, key=lambda result: result.fun)


def estimate_two_factor_configurable(
    panel: pd.DataFrame,
    *,
    gap_function,
    starts: int = 12,
    maxiter: int = 1500,
    seed: int = 852,
    compute_standard_errors: bool = False,
    eta_fast_upper_bound: float = 6.0,
    kappa_gap_upper_bound: float = 60.0,
    warm_starts: Iterable[TwoFactorParams] = (),
) -> TwoFactorEstimationResult:
    """Estimate the model with an explicit fast-minus-slow kappa bound."""
    if starts < 1:
        raise ValueError("starts must be at least one")
    dataset: TwoFactorDataset = make_dataset(panel, gap_function)
    bounds = parameter_bounds(eta_fast_upper_bound, kappa_gap_upper_bound)

    def objective(values: np.ndarray) -> float:
        try:
            params = unpack(values)
            result = -two_factor_log_likelihood(dataset, params)
            return float(result) if np.isfinite(result) else 1e100
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return 1e100

    guesses = list(warm_starts)
    guesses.extend(initial_guesses(panel, max(starts - len(guesses), 0), seed))
    guesses = [_clip_guess_to_bounds(guess, bounds) for guess in guesses[:starts]]
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
        optimizer_results.append(result)
        fitted = unpack(result.x)
        audit_rows.append(
            {
                "start_id": start_id,
                "kappa_gap_upper_bound": kappa_gap_upper_bound,
                "eta_fast_upper_bound": eta_fast_upper_bound,
                **{f"start_{key}": value for key, value in asdict(guess).items()},
                **{f"estimate_{key}": value for key, value in asdict(fitted).items()},
                "estimate_kappa_gap": fitted.kappa_fast - fitted.kappa_slow,
                "log_likelihood": -float(result.fun),
                "converged": bool(result.success),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )

    best = _best_result(optimizer_results)
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
                slow = params.kappa_slow
                gap = params.kappa_fast - params.kappa_slow
                jacobian = np.zeros((6, 6))
                jacobian[0, 0] = slow
                jacobian[1, 0] = slow
                jacobian[1, 1] = gap
                jacobian[2, 2] = 1.0
                jacobian[3, 3] = params.eta_slow
                jacobian[4, 4] = params.eta_fast
                jacobian[5, 5] = params.sigma_epsilon
                covariance = jacobian @ covariance_transformed @ jacobian.T
                standard_errors = dict(
                    zip(asdict(params), np.sqrt(np.diag(covariance)), strict=True)
                )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            stable = False

    return TwoFactorEstimationResult(
        params=params,
        log_likelihood=-float(best.fun),
        converged=bool(best.success),
        message=str(best.message),
        transformed_optimum=optimum,
        standard_errors=standard_errors,
        hessian_stable=stable,
        optimizer_runs=pd.DataFrame(audit_rows).sort_values("log_likelihood", ascending=False),
    )


def _pack_fixed_gap(params: TwoFactorParams) -> np.ndarray:
    return np.array(
        [
            np.log(params.kappa_slow),
            params.theta,
            np.log(params.eta_slow),
            np.log(params.eta_fast),
            np.log(params.sigma_epsilon),
        ]
    )


def _unpack_fixed_gap(values: np.ndarray, fixed_gap: float) -> TwoFactorParams:
    slow = float(np.exp(values[0]))
    return TwoFactorParams(
        kappa_slow=slow,
        kappa_fast=slow + fixed_gap,
        theta=float(values[1]),
        eta_slow=float(np.exp(values[2])),
        eta_fast=float(np.exp(values[3])),
        sigma_epsilon=float(np.exp(values[4])),
    )


def estimate_with_fixed_kappa_gap(
    panel: pd.DataFrame,
    fixed_gap: float,
    *,
    gap_function,
    starts: int = 4,
    maxiter: int = 1500,
    seed: int = 852,
    eta_fast_upper_bound: float = 6.0,
    warm_params: TwoFactorParams | None = None,
) -> TwoFactorEstimationResult:
    """Profile the kappa gap while re-optimizing the other five parameters."""
    if not np.isfinite(fixed_gap) or fixed_gap <= 0.0:
        raise ValueError("fixed_gap must be positive and finite")
    if starts < 1:
        raise ValueError("starts must be at least one")
    parameter_bounds(eta_fast_upper_bound, max(fixed_gap, 0.02))
    dataset = make_dataset(panel, gap_function)
    bounds = [
        (np.log(0.01), np.log(20.0)),
        (-0.50, 0.50),
        (np.log(1e-4), np.log(3.0)),
        (np.log(1e-4), np.log(eta_fast_upper_bound)),
        (np.log(1e-5), np.log(0.50)),
    ]

    def objective(values: np.ndarray) -> float:
        try:
            params = _unpack_fixed_gap(values, fixed_gap)
            result = -two_factor_log_likelihood(dataset, params)
            return float(result) if np.isfinite(result) else 1e100
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return 1e100

    raw_guesses: list[TwoFactorParams] = []
    if warm_params is not None:
        raw_guesses.append(warm_params)
    raw_guesses.extend(initial_guesses(panel, max(starts - len(raw_guesses), 0), seed))
    optimizer_results: list[OptimizeResult] = []
    audit_rows: list[dict[str, object]] = []
    for start_id, raw_guess in enumerate(raw_guesses[:starts]):
        guess = TwoFactorParams(
            kappa_slow=raw_guess.kappa_slow,
            kappa_fast=raw_guess.kappa_slow + fixed_gap,
            theta=raw_guess.theta,
            eta_slow=raw_guess.eta_slow,
            eta_fast=min(raw_guess.eta_fast, eta_fast_upper_bound * (1.0 - 1e-10)),
            sigma_epsilon=raw_guess.sigma_epsilon,
        )
        result = minimize(
            objective,
            _pack_fixed_gap(guess),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-7, "maxls": 50},
        )
        optimizer_results.append(result)
        fitted = _unpack_fixed_gap(result.x, fixed_gap)
        audit_rows.append(
            {
                "fixed_kappa_gap": fixed_gap,
                "start_id": start_id,
                **{f"start_{key}": value for key, value in asdict(guess).items()},
                **{f"estimate_{key}": value for key, value in asdict(fitted).items()},
                "log_likelihood": -float(result.fun),
                "converged": bool(result.success),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )

    best = _best_result(optimizer_results)
    optimum = np.asarray(best.x, dtype=float)
    params = _unpack_fixed_gap(optimum, fixed_gap)
    return TwoFactorEstimationResult(
        params=params,
        log_likelihood=-float(best.fun),
        converged=bool(best.success),
        message=str(best.message),
        transformed_optimum=optimum,
        standard_errors={key: np.nan for key in asdict(params)},
        hessian_stable=False,
        optimizer_runs=pd.DataFrame(audit_rows).sort_values("log_likelihood", ascending=False),
    )
