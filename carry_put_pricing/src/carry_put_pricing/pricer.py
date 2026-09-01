"""Deterministic backward induction for the American carry put."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, sqrt

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import RegularGridInterpolator

from .analytics import (
    exact_forward_price,
    exact_forward_ratio,
    exact_implied_carry,
    ou_integral_loading,
    ou_integral_variance,
)
from .models import (
    CarryPutContract,
    FactorState,
    GBMParams,
    NumericalConfig,
    TwoFactorOUParams,
)


@dataclass(frozen=True)
class ExerciseStepSummary:
    """Compact description of the exercise region on one state grid."""

    elapsed_sessions: int
    remaining_sessions: int
    exercise_grid_fraction: float
    maximum_normalized_exercise_value: float
    maximum_normalized_continuation_value: float


@dataclass(frozen=True)
class FuturesEquivalentCurveDelta:
    """One factor-state sensitivity expressed per futures-price point."""

    factor: str
    pathwise_delta: float
    bump_and_value_delta: float
    absolute_method_difference: float
    pathwise_option_factor_sensitivity: float
    model_futures_factor_sensitivity: float
    bump_down_factor_state: float
    bump_up_factor_state: float
    bump_down_option_value: float
    bump_up_option_value: float
    bump_down_model_futures: float
    bump_up_model_futures: float


@dataclass(frozen=True)
class PricingResult:
    """Price, model checks, grid information, and exercise diagnostics."""

    price: float
    normalized_price: float
    continuation_value: float
    inception_exercise_value: float
    locked_carry: float
    model_initial_carry: float
    model_initial_futures: float
    observed_initial_futures: float
    initial_futures_model_error: float
    maturity: float
    exercise_steps: int
    slow_grid_min: float
    slow_grid_max: float
    fast_grid_min: float
    fast_grid_max: float
    spot_volatility_input: float
    spot_volatility_affects_price: bool
    fixed_carry_scale_delta: float
    slow_curve_delta: FuturesEquivalentCurveDelta
    fast_curve_delta: FuturesEquivalentCurveDelta
    exercise_summary: tuple[ExerciseStepSummary, ...]

    def as_dict(self, *, include_exercise_summary: bool = True) -> dict[str, object]:
        result = asdict(self)
        if not include_exercise_summary:
            result.pop("exercise_summary")
        return result


@dataclass(frozen=True)
class _StepMoments:
    slow_decay: float
    fast_decay: float
    slow_state_variance: float
    fast_state_variance: float
    slow_integral_covariance: float
    fast_integral_covariance: float
    integral_variance: float
    slow_integral_loading: float
    fast_integral_loading: float


def _step_moments(params: TwoFactorOUParams, dt: float) -> _StepMoments:
    slow_decay = exp(-params.kappa_slow * dt)
    fast_decay = exp(-params.kappa_fast * dt)
    slow_state_variance = params.eta_slow**2 * (-np.expm1(-2.0 * params.kappa_slow * dt)) / (
        2.0 * params.kappa_slow
    )
    fast_state_variance = params.eta_fast**2 * (-np.expm1(-2.0 * params.kappa_fast * dt)) / (
        2.0 * params.kappa_fast
    )
    slow_integral_covariance = (
        params.eta_slow**2 * (-np.expm1(-params.kappa_slow * dt)) ** 2
        / (2.0 * params.kappa_slow**2)
    )
    fast_integral_covariance = (
        params.eta_fast**2 * (-np.expm1(-params.kappa_fast * dt)) ** 2
        / (2.0 * params.kappa_fast**2)
    )
    integral_variance = ou_integral_variance(params.kappa_slow, params.eta_slow, dt)
    integral_variance += ou_integral_variance(params.kappa_fast, params.eta_fast, dt)
    return _StepMoments(
        slow_decay=slow_decay,
        fast_decay=fast_decay,
        slow_state_variance=float(slow_state_variance),
        fast_state_variance=float(fast_state_variance),
        slow_integral_covariance=float(slow_integral_covariance),
        fast_integral_covariance=float(fast_integral_covariance),
        integral_variance=integral_variance,
        slow_integral_loading=ou_integral_loading(params.kappa_slow, dt),
        fast_integral_loading=ou_integral_loading(params.kappa_fast, dt),
    )


def _state_grid(
    kappa: float,
    eta: float,
    initial_state: float,
    points: int,
    config: NumericalConfig,
    dt: float,
) -> np.ndarray:
    stationary_std = eta / sqrt(2.0 * kappa)
    one_step_variance = eta**2 * (-np.expm1(-2.0 * kappa * dt)) / (2.0 * kappa)
    one_step_std = sqrt(max(float(one_step_variance), 0.0))
    half_width = max(
        config.minimum_half_width,
        config.stationary_stddev_width * stationary_std,
        1.20 * abs(initial_state) + 2.0 * one_step_std,
    )
    return np.linspace(-half_width, half_width, points)


def _clipped_interpolator(
    slow_grid: np.ndarray,
    fast_grid: np.ndarray,
    values: np.ndarray,
) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (slow_grid, fast_grid),
        values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def _linear_indices(grid: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices and upper-node weights for clipped one-dimensional interpolation."""
    clipped = np.clip(points, grid[0], grid[-1])
    upper = np.searchsorted(grid, clipped, side="right")
    upper = np.clip(upper, 1, len(grid) - 1)
    lower = upper - 1
    fraction = (clipped - grid[lower]) / (grid[upper] - grid[lower])
    return lower, fraction


def _continuation_on_grid(
    next_values: np.ndarray,
    slow_grid: np.ndarray,
    fast_grid: np.ndarray,
    params: TwoFactorOUParams,
    step: _StepMoments,
    dt: float,
    quadrature_nodes: np.ndarray,
    quadrature_weights: np.ndarray,
    *,
    enforce_nonnegative: bool = True,
) -> np.ndarray:
    """Evaluate E[e^-integral(c) next_value(X')] on the current grid.

    Exponential tilting shifts the conditional mean of each next factor by
    -Cov(X'_j, integral(c)); the Gaussian transition covariance is unchanged.
    """
    tilted_slow = step.slow_decay * slow_grid - step.slow_integral_covariance
    tilted_fast = step.fast_decay * fast_grid - step.fast_integral_covariance
    slow_scale = sqrt(2.0 * step.slow_state_variance)
    fast_scale = sqrt(2.0 * step.fast_state_variance)

    # Bilinear interpolation and the independent Gaussian measure are both
    # separable. Applying the fast and slow quadrature transforms successively
    # is algebraically identical to a q-by-q tensor loop and much faster.
    fast_expectation = np.zeros_like(next_values)
    for fast_node, fast_weight in zip(quadrature_nodes, quadrature_weights, strict=True):
        lower, fraction = _linear_indices(fast_grid, tilted_fast + fast_scale * fast_node)
        interpolated = (
            next_values[:, lower] * (1.0 - fraction)[None, :]
            + next_values[:, lower + 1] * fraction[None, :]
        )
        fast_expectation += fast_weight * interpolated

    expectation = np.zeros_like(next_values)
    for slow_node, slow_weight in zip(quadrature_nodes, quadrature_weights, strict=True):
        lower, fraction = _linear_indices(slow_grid, tilted_slow + slow_scale * slow_node)
        interpolated = (
            fast_expectation[lower, :] * (1.0 - fraction)[:, None]
            + fast_expectation[lower + 1, :] * fraction[:, None]
        )
        expectation += slow_weight * interpolated
    expectation /= np.pi

    mean_integral = (
        params.theta * dt
        + step.slow_integral_loading * slow_grid[:, None]
        + step.fast_integral_loading * fast_grid[None, :]
    )
    carry_discount = np.exp(-mean_integral + 0.5 * step.integral_variance)
    result = carry_discount * expectation
    if enforce_nonnegative:
        return np.maximum(result, 0.0)
    return result


def _exercise_value_and_factor_derivatives(
    slow_grid: np.ndarray,
    fast_grid: np.ndarray,
    remaining_time: float,
    locked_carry: float,
    params: TwoFactorOUParams,
    risk_free_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if remaining_time == 0.0:
        zeros = np.zeros((len(slow_grid), len(fast_grid)))
        return zeros, zeros.copy(), zeros.copy()
    locked_ratio = exp((risk_free_rate - locked_carry) * remaining_time)
    market_ratio = exact_forward_ratio(
        params,
        slow_grid[:, None],
        fast_grid[None, :],
        remaining_time,
        risk_free_rate,
    )
    in_the_money = locked_ratio > market_ratio
    exercise = np.maximum(locked_ratio - market_ratio, 0.0)
    slow_derivative = np.where(
        in_the_money,
        ou_integral_loading(params.kappa_slow, remaining_time) * market_ratio,
        0.0,
    )
    fast_derivative = np.where(
        in_the_money,
        ou_integral_loading(params.kappa_fast, remaining_time) * market_ratio,
        0.0,
    )
    return exercise, slow_derivative, fast_derivative


def _futures_equivalent_curve_delta(
    *,
    factor: str,
    option_factor_sensitivity: float,
    futures_factor_sensitivity: float,
    time_zero_interpolator: RegularGridInterpolator,
    slow_grid: np.ndarray,
    fast_grid: np.ndarray,
    contract: CarryPutContract,
    params: TwoFactorOUParams,
    initial_state: FactorState,
    risk_free_rate: float,
) -> FuturesEquivalentCurveDelta:
    """Compare a tangent delta with a local grid bump along one factor axis."""
    if factor == "slow":
        grid = slow_grid
        initial_factor = initial_state.slow
        other_factor = initial_state.fast
    elif factor == "fast":
        grid = fast_grid
        initial_factor = initial_state.fast
        other_factor = initial_state.slow
    else:  # pragma: no cover - internal programming guard
        raise ValueError(f"Unknown factor: {factor}")

    grid_step = float(grid[1] - grid[0])
    down_factor = max(float(grid[0]), initial_factor - grid_step)
    up_factor = min(float(grid[-1]), initial_factor + grid_step)
    if not down_factor < up_factor:  # pragma: no cover - grid construction protects this
        raise ValueError(f"Cannot form a local {factor}-factor bump")

    if factor == "slow":
        down_state = FactorState(down_factor, other_factor)
        up_state = FactorState(up_factor, other_factor)
        interpolation_points = np.array(
            [[down_factor, other_factor], [up_factor, other_factor]]
        )
    else:
        down_state = FactorState(other_factor, down_factor)
        up_state = FactorState(other_factor, up_factor)
        interpolation_points = np.array(
            [[other_factor, down_factor], [other_factor, up_factor]]
        )

    normalized_values = time_zero_interpolator(interpolation_points)
    down_option = contract.initial_spot * float(normalized_values[0])
    up_option = contract.initial_spot * float(normalized_values[1])
    down_futures = exact_forward_price(
        contract.initial_spot,
        params,
        down_state,
        contract.maturity,
        risk_free_rate,
    )
    up_futures = exact_forward_price(
        contract.initial_spot,
        params,
        up_state,
        contract.maturity,
        risk_free_rate,
    )
    bump_delta = (up_option - down_option) / (up_futures - down_futures)
    pathwise_delta = option_factor_sensitivity / futures_factor_sensitivity
    return FuturesEquivalentCurveDelta(
        factor=factor,
        pathwise_delta=float(pathwise_delta),
        bump_and_value_delta=float(bump_delta),
        absolute_method_difference=abs(float(pathwise_delta - bump_delta)),
        pathwise_option_factor_sensitivity=float(option_factor_sensitivity),
        model_futures_factor_sensitivity=float(futures_factor_sensitivity),
        bump_down_factor_state=float(down_factor),
        bump_up_factor_state=float(up_factor),
        bump_down_option_value=float(down_option),
        bump_up_option_value=float(up_option),
        bump_down_model_futures=float(down_futures),
        bump_up_model_futures=float(up_futures),
    )


def price_american_carry_put(
    contract: CarryPutContract,
    ou_params: TwoFactorOUParams,
    initial_state: FactorState,
    gbm_params: GBMParams,
    *,
    numerical: NumericalConfig | None = None,
) -> PricingResult:
    """Price the daily-exercisable carry put by exact-transition backward induction.

    The calibrated OU parameters are interpreted as risk-neutral inputs. The
    inception locked carry comes from the observed futures quote. Exercise at
    inception is fixed to zero by the contract identity F_(0,T)=F_(0,T), rather
    than allowing a small model-versus-market curve-fit residual to create value.
    """
    config = numerical or NumericalConfig()
    dt = 1.0 / contract.periods_per_year
    slow_grid = _state_grid(
        ou_params.kappa_slow,
        ou_params.eta_slow,
        initial_state.slow,
        config.slow_grid_points,
        config,
        dt,
    )
    fast_grid = _state_grid(
        ou_params.kappa_fast,
        ou_params.eta_fast,
        initial_state.fast,
        config.fast_grid_points,
        config,
        dt,
    )
    nodes, weights = hermgauss(config.quadrature_order)
    step = _step_moments(ou_params, dt)
    locked_carry = contract.locked_carry(gbm_params.risk_free_rate)

    # At maturity both forward-growth terms are one, so the payoff is zero.
    next_values = np.zeros((len(slow_grid), len(fast_grid)))
    next_slow_derivative = np.zeros_like(next_values)
    next_fast_derivative = np.zeros_like(next_values)
    summaries: list[ExerciseStepSummary] = []
    for elapsed_sessions in range(contract.sessions_to_expiry - 1, 0, -1):
        continuation = _continuation_on_grid(
            next_values,
            slow_grid,
            fast_grid,
            ou_params,
            step,
            dt,
            nodes,
            weights,
        )
        slow_derivative_expectation = _continuation_on_grid(
            next_slow_derivative,
            slow_grid,
            fast_grid,
            ou_params,
            step,
            dt,
            nodes,
            weights,
            enforce_nonnegative=False,
        )
        fast_derivative_expectation = _continuation_on_grid(
            next_fast_derivative,
            slow_grid,
            fast_grid,
            ou_params,
            step,
            dt,
            nodes,
            weights,
            enforce_nonnegative=False,
        )
        slow_continuation_derivative = (
            step.slow_decay * slow_derivative_expectation
            - step.slow_integral_loading * continuation
        )
        fast_continuation_derivative = (
            step.fast_decay * fast_derivative_expectation
            - step.fast_integral_loading * continuation
        )
        remaining_sessions = contract.sessions_to_expiry - elapsed_sessions
        exercise, slow_exercise_derivative, fast_exercise_derivative = (
            _exercise_value_and_factor_derivatives(
                slow_grid,
                fast_grid,
                remaining_sessions / contract.periods_per_year,
                locked_carry,
                ou_params,
                gbm_params.risk_free_rate,
            )
        )
        exercise_region = exercise > continuation + config.exercise_tolerance
        summaries.append(
            ExerciseStepSummary(
                elapsed_sessions=elapsed_sessions,
                remaining_sessions=remaining_sessions,
                exercise_grid_fraction=float(exercise_region.mean()),
                maximum_normalized_exercise_value=float(exercise.max()),
                maximum_normalized_continuation_value=float(continuation.max()),
            )
        )
        # Differentiate the original Snell envelope. The derivative does not
        # solve a second optimal-stopping problem: it follows the exercise
        # policy selected by the primal option value. At exact ties we use the
        # continuation-side derivative.
        exercise_policy = exercise > continuation
        next_slow_derivative = np.where(
            exercise_policy,
            slow_exercise_derivative,
            slow_continuation_derivative,
        )
        next_fast_derivative = np.where(
            exercise_policy,
            fast_exercise_derivative,
            fast_continuation_derivative,
        )
        next_values = np.maximum(exercise, continuation)

    time_zero_continuation_grid = _continuation_on_grid(
        next_values,
        slow_grid,
        fast_grid,
        ou_params,
        step,
        dt,
        nodes,
        weights,
    )
    time_zero_slow_derivative_expectation = _continuation_on_grid(
        next_slow_derivative,
        slow_grid,
        fast_grid,
        ou_params,
        step,
        dt,
        nodes,
        weights,
        enforce_nonnegative=False,
    )
    time_zero_fast_derivative_expectation = _continuation_on_grid(
        next_fast_derivative,
        slow_grid,
        fast_grid,
        ou_params,
        step,
        dt,
        nodes,
        weights,
        enforce_nonnegative=False,
    )
    time_zero_slow_derivative_grid = (
        step.slow_decay * time_zero_slow_derivative_expectation
        - step.slow_integral_loading * time_zero_continuation_grid
    )
    time_zero_fast_derivative_grid = (
        step.fast_decay * time_zero_fast_derivative_expectation
        - step.fast_integral_loading * time_zero_continuation_grid
    )
    time_zero_interpolator = _clipped_interpolator(
        slow_grid,
        fast_grid,
        time_zero_continuation_grid,
    )
    time_zero_slow_derivative_interpolator = _clipped_interpolator(
        slow_grid,
        fast_grid,
        time_zero_slow_derivative_grid,
    )
    time_zero_fast_derivative_interpolator = _clipped_interpolator(
        slow_grid,
        fast_grid,
        time_zero_fast_derivative_grid,
    )
    initial_point = np.array([[initial_state.slow, initial_state.fast]])
    normalized_continuation = float(
        time_zero_interpolator(initial_point)[0]
    )
    normalized_continuation = max(normalized_continuation, 0.0)
    normalized_slow_derivative = float(
        time_zero_slow_derivative_interpolator(initial_point)[0]
    )
    normalized_fast_derivative = float(
        time_zero_fast_derivative_interpolator(initial_point)[0]
    )

    # The contractual exercise value is exactly zero at inception. A curve-fit
    # residual between the exact OU formula and the observed quote is diagnostic.
    inception_exercise_value = 0.0
    normalized_price = max(inception_exercise_value, normalized_continuation)
    model_initial_futures = exact_forward_price(
        contract.initial_spot,
        ou_params,
        initial_state,
        contract.maturity,
        gbm_params.risk_free_rate,
    )
    model_initial_carry = exact_implied_carry(ou_params, initial_state, contract.maturity)
    price = contract.initial_spot * normalized_price
    # Scale spot and futures by the same positive factor while holding their
    # ratio, the locked carry, and both OU factor states fixed. Homogeneity then
    # gives dV/dF = V/F along this co-scaling direction. This is distinct from a
    # futures-only delta with spot fixed and from either factor curve delta.
    fixed_carry_scale_delta = price / model_initial_futures
    slow_futures_factor_sensitivity = -ou_integral_loading(
        ou_params.kappa_slow, contract.maturity
    ) * model_initial_futures
    fast_futures_factor_sensitivity = -ou_integral_loading(
        ou_params.kappa_fast, contract.maturity
    ) * model_initial_futures
    slow_curve_delta = _futures_equivalent_curve_delta(
        factor="slow",
        option_factor_sensitivity=contract.initial_spot * normalized_slow_derivative,
        futures_factor_sensitivity=slow_futures_factor_sensitivity,
        time_zero_interpolator=time_zero_interpolator,
        slow_grid=slow_grid,
        fast_grid=fast_grid,
        contract=contract,
        params=ou_params,
        initial_state=initial_state,
        risk_free_rate=gbm_params.risk_free_rate,
    )
    fast_curve_delta = _futures_equivalent_curve_delta(
        factor="fast",
        option_factor_sensitivity=contract.initial_spot * normalized_fast_derivative,
        futures_factor_sensitivity=fast_futures_factor_sensitivity,
        time_zero_interpolator=time_zero_interpolator,
        slow_grid=slow_grid,
        fast_grid=fast_grid,
        contract=contract,
        params=ou_params,
        initial_state=initial_state,
        risk_free_rate=gbm_params.risk_free_rate,
    )
    summaries.sort(key=lambda row: row.elapsed_sessions)
    return PricingResult(
        price=price,
        normalized_price=normalized_price,
        continuation_value=contract.initial_spot * normalized_continuation,
        inception_exercise_value=inception_exercise_value,
        locked_carry=locked_carry,
        model_initial_carry=model_initial_carry,
        model_initial_futures=model_initial_futures,
        observed_initial_futures=contract.initial_futures,
        initial_futures_model_error=model_initial_futures - contract.initial_futures,
        maturity=contract.maturity,
        exercise_steps=contract.sessions_to_expiry,
        slow_grid_min=float(slow_grid[0]),
        slow_grid_max=float(slow_grid[-1]),
        fast_grid_min=float(fast_grid[0]),
        fast_grid_max=float(fast_grid[-1]),
        spot_volatility_input=gbm_params.volatility,
        spot_volatility_affects_price=False,
        fixed_carry_scale_delta=fixed_carry_scale_delta,
        slow_curve_delta=slow_curve_delta,
        fast_curve_delta=fast_curve_delta,
        exercise_summary=tuple(summaries),
    )
