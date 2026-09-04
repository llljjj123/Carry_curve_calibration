"""Fixed-fast-volatility likelihood and carry-put price profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from calibration import CalibrationResult, _gap_function
from option_pricing import price_from_parameters

from im_2factor_ou_carry.two_factor import (
    TwoFactorParams,
    make_dataset,
    two_factor_kalman_filter,
    two_factor_log_likelihood,
)
from im_2factor_ou_carry.two_factor_estimation import initial_guesses
from im_2factor_ou_carry.two_factor_fitting import attach_two_factor_fits
from im_2factor_ou_carry.observation import (
    ObservationNoiseModel,
    noise_parameter_name,
    normalize_observation_noise_model,
)


ETA_FAST_PROFILE = (2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 5.0, 6.0, 8.0, 10.0)


@dataclass
class FixedEtaEstimate:
    params: TwoFactorParams
    log_likelihood: float
    converged: bool
    message: str
    optimizer_runs: pd.DataFrame
    observation_noise_model: ObservationNoiseModel = "constant_carry"


@dataclass
class FixedEtaProfile:
    results: pd.DataFrame
    optimizer_runs: pd.DataFrame


def _pack_fixed(params: TwoFactorParams) -> np.ndarray:
    return np.array(
        [
            np.log(params.kappa_slow),
            np.log(params.kappa_fast - params.kappa_slow),
            params.theta,
            np.log(params.eta_slow),
            np.log(params.sigma_epsilon),
        ]
    )


def _unpack_fixed(values: np.ndarray, eta_fast: float) -> TwoFactorParams:
    slow = float(np.exp(values[0]))
    return TwoFactorParams(
        kappa_slow=slow,
        kappa_fast=slow + float(np.exp(values[1])),
        theta=float(values[2]),
        eta_slow=float(np.exp(values[3])),
        eta_fast=float(eta_fast),
        sigma_epsilon=float(np.exp(values[4])),
    )


def estimate_with_fixed_eta_fast(
    panel: pd.DataFrame,
    eta_fast: float,
    warm_params: TwoFactorParams,
    *,
    starts: int = 4,
    maxiter: int = 1500,
    seed: int = 852,
    observation_noise_model: ObservationNoiseModel = "constant_carry",
    kappa_gap_upper_bound: float = 60.0,
) -> FixedEtaEstimate:
    """Re-optimize the other five parameters conditional on fixed eta_fast."""
    if not np.isfinite(eta_fast) or eta_fast <= 0.0:
        raise ValueError("eta_fast must be positive and finite")
    if starts < 1:
        raise ValueError("starts must be at least one")
    observation_noise_model = normalize_observation_noise_model(
        observation_noise_model
    )
    if not np.isfinite(kappa_gap_upper_bound) or kappa_gap_upper_bound <= 0.01:
        raise ValueError("kappa_gap_upper_bound must be finite and greater than 0.01")
    dataset = make_dataset(panel, _gap_function, observation_noise_model)
    noise_bound = (
        (np.log(1e-5), np.log(0.50))
        if observation_noise_model == "constant_carry"
        else (np.log(1e-8), np.log(0.05))
    )
    bounds = [
        (np.log(0.01), np.log(20.0)),
        (np.log(0.01), np.log(kappa_gap_upper_bound)),
        (-0.50, 0.50),
        (np.log(1e-4), np.log(3.0)),
        noise_bound,
    ]

    def objective(values: np.ndarray) -> float:
        try:
            params = _unpack_fixed(values, eta_fast)
            value = -two_factor_log_likelihood(dataset, params)
            return float(value) if np.isfinite(value) else 1e100
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            return 1e100

    warm = TwoFactorParams(
        kappa_slow=warm_params.kappa_slow,
        kappa_fast=warm_params.kappa_fast,
        theta=warm_params.theta,
        eta_slow=warm_params.eta_slow,
        eta_fast=eta_fast,
        sigma_epsilon=warm_params.sigma_epsilon,
    )
    guesses = [warm]
    if starts > 1:
        guesses.extend(
            initial_guesses(
                panel,
                starts - 1,
                seed,
                observation_noise_model,
            )
        )

    optimizer_results = []
    audit_rows = []
    for start_id, guess in enumerate(guesses[:starts]):
        result = minimize(
            objective,
            _pack_fixed(guess),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-11, "gtol": 1e-7, "maxls": 50},
        )
        optimizer_results.append(result)
        fitted = _unpack_fixed(result.x, eta_fast)
        audit_rows.append(
            {
                "eta_fast_fixed": eta_fast,
                "start_id": start_id,
                "observation_noise_model": observation_noise_model,
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
        raise RuntimeError(f"All optimizer starts failed for eta_fast={eta_fast}")
    successful = [result for result in finite if result.success]
    best = min(successful or finite, key=lambda result: result.fun)
    return FixedEtaEstimate(
        params=_unpack_fixed(best.x, eta_fast),
        log_likelihood=-float(best.fun),
        converged=bool(best.success),
        message=str(best.message),
        optimizer_runs=pd.DataFrame(audit_rows).sort_values("log_likelihood", ascending=False),
        observation_noise_model=observation_noise_model,
    )


def run_fixed_eta_profile(
    calibration: CalibrationResult,
    eta_values: tuple[float, ...] | None = None,
    *,
    optimizer_starts: int = 4,
) -> FixedEtaProfile:
    """Profile likelihood and option value, re-estimating all other parameters."""
    if eta_values is None:
        eta_values = tuple(
            sorted({*ETA_FAST_PROFILE, float(calibration.estimate.params.eta_fast)})
        )
    rows: list[dict[str, object]] = []
    audits: list[pd.DataFrame] = []
    warm = calibration.estimate.params
    observation_noise_model = calibration.estimate.observation_noise_model
    kappa_gap_upper_bound = float(
        calibration.metrics["kappa_fast_minus_slow_upper_bound"]
    )
    for eta_fast in eta_values:
        estimate = estimate_with_fixed_eta_fast(
            calibration.sample,
            eta_fast,
            warm,
            starts=optimizer_starts,
            observation_noise_model=observation_noise_model,
            kappa_gap_upper_bound=kappa_gap_upper_bound,
        )
        filtered = two_factor_kalman_filter(
            calibration.sample,
            estimate.params,
            gap_function=_gap_function,
            observation_noise_model=observation_noise_model,
        )
        fitted = attach_two_factor_fits(
            calibration.sample,
            filtered.states,
            estimate.params,
            observation_noise_model,
        )
        latest_state = filtered.states.iloc[-1]
        option = price_from_parameters(calibration, estimate.params, latest_state)
        carry_residual = fitted["carry_residual"].to_numpy(dtype=float)
        futures_residual = fitted["futures_residual"].to_numpy(dtype=float)
        rows.append(
            {
                "eta_fast_fixed": eta_fast,
                "kappa_slow": estimate.params.kappa_slow,
                "kappa_fast": estimate.params.kappa_fast,
                "kappa_fast_minus_slow": (
                    estimate.params.kappa_fast - estimate.params.kappa_slow
                ),
                "kappa_fast_minus_slow_at_upper_bound": bool(
                    estimate.params.kappa_fast - estimate.params.kappa_slow
                    >= kappa_gap_upper_bound - 1.0e-8
                ),
                "theta": estimate.params.theta,
                "eta_slow": estimate.params.eta_slow,
                "observation_noise_model": observation_noise_model,
                noise_parameter_name(observation_noise_model): (
                    estimate.params.sigma_epsilon
                ),
                "log_likelihood": estimate.log_likelihood,
                "converged": estimate.converged,
                "carry_rmse_bps": np.sqrt(np.mean(carry_residual**2)) * 10_000,
                "futures_rmse_points": np.sqrt(np.mean(futures_residual**2)),
                "latest_slow_state": latest_state["filtered_slow_state"],
                "latest_fast_state": latest_state["filtered_fast_state"],
                "latest_instantaneous_carry": latest_state["filtered_instantaneous_carry"],
                "option_price": option.price,
                "normalized_option_price": option.normalized_price,
                "model_initial_carry": option.model_initial_carry,
                "model_initial_futures": option.model_initial_futures,
                "initial_futures_model_error": option.initial_futures_model_error,
            }
        )
        audits.append(estimate.optimizer_runs)

    profile = pd.DataFrame(rows).sort_values("eta_fast_fixed").reset_index(drop=True)
    maximum = float(profile["log_likelihood"].max())
    cap_log_likelihood = float(calibration.estimate.log_likelihood)
    profile["log_likelihood_below_profile_max"] = profile["log_likelihood"] - maximum
    profile["likelihood_ratio_vs_profile_max"] = 2.0 * (maximum - profile["log_likelihood"])
    profile["relative_likelihood"] = np.exp(
        np.maximum(profile["log_likelihood"] - maximum, -745.0)
    )
    profile["log_likelihood_difference_from_cap6_fit"] = (
        profile["log_likelihood"] - cap_log_likelihood
    )
    return FixedEtaProfile(
        results=profile,
        optimizer_runs=pd.concat(audits, ignore_index=True),
    )
