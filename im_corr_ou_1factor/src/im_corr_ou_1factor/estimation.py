"""Maximum-likelihood estimation, uncertainty, and rho profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from .filtering import kalman_filter
from .model import OUParams


@dataclass
class EstimationResult:
    name: str
    mode: str
    variant: str
    free_rho: bool
    params: OUParams
    sigma: float
    log_likelihood: float
    curve_log_likelihood: float
    return_log_likelihood: float
    converged: bool
    message: str
    gradient_norm: float
    transformed_optimum: np.ndarray
    parameter_names: list[str]
    standard_errors: dict[str, float]
    hessian_stable: bool
    optimizer_runs: pd.DataFrame


def _layout(mode: str, free_rho: bool) -> list[str]:
    names = ["log_kappa", "theta", "log_eta"]
    if free_rho:
        names.append("atanh_rho")
    names.append("log_sigma_epsilon")
    if mode == "joint":
        names.append("mu")
    return names


def pack(params: OUParams, mode: str, free_rho: bool) -> np.ndarray:
    values = [np.log(params.kappa), params.theta, np.log(params.eta)]
    if free_rho:
        values.append(np.arctanh(np.clip(params.rho, -0.995, 0.995)))
    values.append(np.log(params.sigma_epsilon))
    if mode == "joint":
        values.append(params.mu)
    return np.asarray(values, dtype=float)


def unpack(values: np.ndarray, mode: str, free_rho: bool, fixed_rho: float) -> OUParams:
    cursor = 0
    kappa = float(np.exp(values[cursor])); cursor += 1
    theta = float(values[cursor]); cursor += 1
    eta = float(np.exp(values[cursor])); cursor += 1
    rho = float(np.tanh(values[cursor])) if free_rho else float(fixed_rho)
    if free_rho:
        cursor += 1
    sigma_epsilon = float(np.exp(values[cursor])); cursor += 1
    mu = float(values[cursor]) if mode == "joint" else 0.0
    return OUParams(kappa, theta, eta, rho, sigma_epsilon, mu)


def _bounds(mode: str, free_rho: bool) -> list[tuple[float, float]]:
    result = [(np.log(0.01), np.log(50.0)), (-0.50, 0.50), (np.log(1e-4), np.log(3.0))]
    if free_rho:
        limit = float(np.arctanh(0.995))
        result.append((-limit, limit))
    result.append((np.log(1e-5), np.log(0.50)))
    if mode == "joint":
        result.append((-1.0, 1.0))
    return result


def initial_guesses(panel: pd.DataFrame, count: int, mode: str, free_rho: bool, seed: int) -> list[OUParams]:
    carries = panel["implied_carry"].dropna().to_numpy(dtype=float)
    theta = float(np.clip(np.median(carries), -0.25, 0.25))
    daily = panel.groupby("date")["implied_carry"].mean().sort_index()
    daily_diff = float(daily.diff().std())
    eta = float(np.clip((daily_diff if np.isfinite(daily_diff) else 0.005) * np.sqrt(244), 0.02, 1.5))
    centered = panel["implied_carry"] - panel.groupby("date")["implied_carry"].transform("mean")
    noise = float(np.clip(centered.std(), 0.001, 0.10))
    spot = panel.groupby("date")["spot"].first().sort_index()
    mu = float(np.log(spot).diff().mean() * 244 + theta + 0.5 * 0.25**2)
    kappas = [0.5, 2.0, 6.0, 12.0, 25.0]
    rhos = [0.0, -0.4, 0.4, -0.75, 0.75] if free_rho else [0.0] * len(kappas)
    starts = [OUParams(k, theta, eta, rhos[i], noise, mu) for i, k in enumerate(kappas[:count])]
    rng = np.random.default_rng(seed)
    while len(starts) < count:
        starts.append(
            OUParams(
                float(np.exp(rng.uniform(np.log(0.15), np.log(30.0)))),
                float(theta + rng.normal(0, 0.025)),
                float(np.clip(eta * np.exp(rng.normal(0, 0.5)), 1e-3, 2.5)),
                float(rng.uniform(-0.85, 0.85)) if free_rho else 0.0,
                float(np.clip(noise * np.exp(rng.normal(0, 0.4)), 1e-4, 0.25)),
                float(mu + rng.normal(0, 0.10)) if mode == "joint" else 0.0,
            )
        )
    return starts


def numerical_hessian(function: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    steps = 2e-4 * np.maximum(1.0, np.abs(x))
    result = np.empty((len(x), len(x)), dtype=float)
    f0 = function(x)
    for i in range(len(x)):
        ei = np.zeros_like(x); ei[i] = steps[i]
        result[i, i] = (function(x + ei) - 2 * f0 + function(x - ei)) / steps[i] ** 2
        for j in range(i):
            ej = np.zeros_like(x); ej[j] = steps[j]
            value = (
                function(x + ei + ej) - function(x + ei - ej)
                - function(x - ei + ej) + function(x - ei - ej)
            ) / (4 * steps[i] * steps[j])
            result[i, j] = result[j, i] = value
    return result


def estimate_model(
    panel: pd.DataFrame,
    *,
    name: str,
    mode: str,
    variant: str,
    sigma: float,
    gap_function,
    free_rho: bool,
    fixed_rho: float = 0.0,
    starts: int = 8,
    maxiter: int = 1200,
    seed: int = 852,
    compute_standard_errors: bool = True,
    supplied_starts: Iterable[OUParams] | None = None,
) -> EstimationResult:
    """Estimate one model specification with transformed, bounded parameters."""
    if variant == "legacy" and (free_rho or fixed_rho != 0):
        raise ValueError("The legacy model is defined only for rho=0")
    names = _layout(mode, free_rho)

    def objective(x: np.ndarray) -> float:
        try:
            params = unpack(x, mode, free_rho, fixed_rho)
            value = -kalman_filter(
                panel, params, sigma=sigma, gap_function=gap_function, mode=mode, variant=variant
            ).log_likelihood
            return float(value) if np.isfinite(value) else 1e100
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return 1e100

    guesses = list(supplied_starts or [])
    if len(guesses) < starts:
        guesses.extend(initial_guesses(panel, starts - len(guesses), mode, free_rho, seed + len(guesses)))
    results = []
    audit: list[dict[str, object]] = []
    for index, guess in enumerate(guesses[:starts]):
        if not free_rho:
            guess = replace(guess, rho=fixed_rho)
        result = minimize(
            objective,
            pack(guess, mode, free_rho),
            method="L-BFGS-B",
            bounds=_bounds(mode, free_rho),
            options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-7, "maxls": 50},
        )
        results.append(result)
        fitted = unpack(result.x, mode, free_rho, fixed_rho)
        audit.append(
            {
                "model": name,
                "start_id": index,
                **{f"start_{key}": value for key, value in asdict(guess).items()},
                **{f"estimate_{key}": value for key, value in asdict(fitted).items()},
                "log_likelihood": -float(result.fun),
                "converged": bool(result.success),
                "iterations": int(result.nit),
                "gradient_norm": float(np.linalg.norm(result.jac)) if result.jac is not None else np.nan,
                "message": str(result.message),
            }
        )
    finite = [item for item in results if np.isfinite(item.fun) and item.fun < 1e99]
    if not finite:
        raise RuntimeError(f"All optimizer starts failed for {name}")
    successful = [item for item in finite if item.success]
    best = min(successful or finite, key=lambda item: item.fun)
    optimum = np.asarray(best.x, dtype=float)
    params = unpack(optimum, mode, free_rho, fixed_rho)
    filtered = kalman_filter(panel, params, sigma=sigma, gap_function=gap_function, mode=mode, variant=variant)

    reported_names = ["kappa", "theta", "eta"] + (["rho"] if free_rho else []) + ["sigma_epsilon"]
    if mode == "joint":
        reported_names.append("mu")
    standard_errors = {key: np.nan for key in asdict(params)}
    stable = False
    if compute_standard_errors:
        try:
            hessian = numerical_hessian(objective, optimum)
            covariance_x = np.linalg.inv(hessian)
            stable = bool(np.linalg.eigvalsh(hessian).min() > 1e-7 and np.diag(covariance_x).min() >= 0)
            if stable:
                derivatives = [params.kappa, 1.0, params.eta]
                if free_rho:
                    derivatives.append(1.0 - params.rho**2)
                derivatives.append(params.sigma_epsilon)
                if mode == "joint":
                    derivatives.append(1.0)
                covariance = np.diag(derivatives) @ covariance_x @ np.diag(derivatives)
                for key, value in zip(reported_names, np.sqrt(np.diag(covariance)), strict=True):
                    standard_errors[key] = float(value)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            stable = False

    return EstimationResult(
        name=name,
        mode=mode,
        variant=variant,
        free_rho=free_rho,
        params=params,
        sigma=float(sigma),
        log_likelihood=filtered.log_likelihood,
        curve_log_likelihood=filtered.curve_log_likelihood,
        return_log_likelihood=filtered.return_log_likelihood,
        converged=bool(best.success),
        message=str(best.message),
        gradient_norm=float(np.linalg.norm(best.jac)) if best.jac is not None else np.nan,
        transformed_optimum=optimum,
        parameter_names=reported_names,
        standard_errors=standard_errors,
        hessian_stable=stable,
        optimizer_runs=pd.DataFrame(audit).sort_values("log_likelihood", ascending=False),
    )


def result_table(result: EstimationResult, n_curve_observations: int, n_dates: int) -> pd.DataFrame:
    n_likelihood = n_curve_observations + (n_dates - 1 if result.mode == "joint" else 0)
    k = len(result.transformed_optimum)
    rows = []
    for key in result.parameter_names:
        rows.append(
            {
                "model": result.name,
                "mode": result.mode,
                "variant": result.variant,
                "parameter": key,
                "estimate": getattr(result.params, key),
                "standard_error": result.standard_errors.get(key, np.nan),
                "fixed": False,
            }
        )
    if not result.free_rho:
        rows.append({"model": result.name, "mode": result.mode, "variant": result.variant, "parameter": "rho", "estimate": result.params.rho, "standard_error": np.nan, "fixed": True})
    rows.append({"model": result.name, "mode": result.mode, "variant": result.variant, "parameter": "sigma", "estimate": result.sigma, "standard_error": np.nan, "fixed": True})
    table = pd.DataFrame(rows)
    table["log_likelihood"] = result.log_likelihood
    table["curve_log_likelihood"] = result.curve_log_likelihood
    table["return_log_likelihood"] = result.return_log_likelihood
    table["n_likelihood_observations"] = n_likelihood
    table["n_estimated_parameters"] = k
    table["aic"] = 2 * k - 2 * result.log_likelihood
    table["bic"] = np.log(n_likelihood) * k - 2 * result.log_likelihood
    table["optimizer_converged"] = result.converged
    table["optimizer_message"] = result.message
    table["gradient_norm"] = result.gradient_norm
    table["hessian_stable"] = result.hessian_stable
    table["half_life_sessions"] = 244 * np.log(2) / result.params.kappa
    return table


def profile_rho(
    panel: pd.DataFrame,
    unrestricted: EstimationResult,
    grid: Iterable[float],
    *,
    gap_function,
    starts: int,
    maxiter: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    """Optimize nuisance parameters at fixed rho values and form a 95% LR set."""
    rows = []
    warm = unrestricted.params
    for index, rho in enumerate(grid):
        fitted = estimate_model(
            panel,
            name=f"{unrestricted.name}_profile",
            mode=unrestricted.mode,
            variant=unrestricted.variant,
            sigma=unrestricted.sigma,
            gap_function=gap_function,
            free_rho=False,
            fixed_rho=float(rho),
            starts=starts,
            maxiter=maxiter,
            seed=seed + index,
            compute_standard_errors=False,
            supplied_starts=[replace(warm, rho=float(rho))],
        )
        warm = fitted.params
        rows.append({"rho": float(rho), "log_likelihood": fitted.log_likelihood, "converged": fitted.converged})
    profile = pd.DataFrame(rows).sort_values("rho").reset_index(drop=True)
    profile["lr_statistic"] = 2 * (unrestricted.log_likelihood - profile["log_likelihood"])
    cutoff = float(chi2.ppf(0.95, 1))
    profile["inside_95pct_lr_set"] = profile["lr_statistic"] <= cutoff
    accepted_index = profile.index[profile["inside_95pct_lr_set"]].to_numpy()
    accepted = profile.loc[accepted_index, "rho"]
    lower_hit = bool(len(accepted_index) and accepted_index[0] == 0)
    upper_hit = bool(len(accepted_index) and accepted_index[-1] == len(profile) - 1)
    ci_low = float(accepted.min()) if not accepted.empty else np.nan
    ci_high = float(accepted.max()) if not accepted.empty else np.nan
    if len(accepted_index) and not lower_hit:
        outside = profile.loc[accepted_index[0] - 1]
        inside = profile.loc[accepted_index[0]]
        ci_low = float(outside.rho + (cutoff - outside.lr_statistic) * (inside.rho - outside.rho) / (inside.lr_statistic - outside.lr_statistic))
    if len(accepted_index) and not upper_hit:
        inside = profile.loc[accepted_index[-1]]
        outside = profile.loc[accepted_index[-1] + 1]
        ci_high = float(inside.rho + (cutoff - inside.lr_statistic) * (outside.rho - inside.rho) / (outside.lr_statistic - inside.lr_statistic))
    summary: dict[str, float | bool] = {
        "cutoff": cutoff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_hits_lower_grid": lower_hit,
        "ci_hits_upper_grid": upper_hit,
    }
    return profile, summary
