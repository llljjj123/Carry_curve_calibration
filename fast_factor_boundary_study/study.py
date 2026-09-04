"""Run the fast-factor identification and boundary sensitivity study."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = STUDY_ROOT.parent
DEMO_ROOT = WORKSPACE_ROOT / "Demo"
CALIBRATION_SRC = WORKSPACE_ROOT / "im_2factor_ou_carry" / "src"
PRICING_SRC = WORKSPACE_ROOT / "carry_put_pricing" / "src"
for path in (STUDY_ROOT, DEMO_ROOT, CALIBRATION_SRC, PRICING_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibration import (  # noqa: E402
    PERIODS_PER_YEAR,
    RISK_FREE_RATE,
    _gap_function,
    estimate_historical_volatility,
    load_calibration_sample,
)
from carry_put_pricing import (  # noqa: E402
    CarryPutContract,
    FactorState,
    GBMParams,
    NumericalConfig,
    TwoFactorOUParams,
    price_american_carry_put,
)
from configurable_estimation import (  # noqa: E402
    estimate_two_factor_configurable,
    estimate_with_fixed_kappa_gap,
)
from im_2factor_ou_carry.two_factor import (  # noqa: E402
    TwoFactorParams,
    two_factor_kalman_filter,
)
from im_2factor_ou_carry.two_factor_fitting import (  # noqa: E402
    attach_two_factor_fits,
)


DEFAULT_EVALUATION_DATE = "2026-08-21"
DEFAULT_CONTRACT = "IM2612"
DEFAULT_WINDOWS = (244, 488, 732, 991)
DEFAULT_GAP_CAPS = (60.0, 90.0, 120.0, 180.0)
DEFAULT_SHORT_END_CUTOFFS = (5, 10, 15, 20)
DEFAULT_PROFILE_GAPS = (
    20.0,
    40.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
    95.0,
    100.0,
    120.0,
    150.0,
    180.0,
    240.0,
)


@dataclass(frozen=True)
class ScenarioSpec:
    """One free-gap calibration specification."""

    window_dates: int
    kappa_gap_upper_bound: float
    min_sessions_exclusive: int

    @property
    def scenario_id(self) -> str:
        cap = f"{self.kappa_gap_upper_bound:g}".replace(".", "p")
        return f"w{self.window_dates}_cap{cap}_min{self.min_sessions_exclusive}"


@dataclass
class StudySettings:
    evaluation_date: str = DEFAULT_EVALUATION_DATE
    contract: str = DEFAULT_CONTRACT
    reference_window: int = 488
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    gap_caps: tuple[float, ...] = DEFAULT_GAP_CAPS
    short_end_cutoffs: tuple[int, ...] = DEFAULT_SHORT_END_CUTOFFS
    profile_gaps: tuple[float, ...] = DEFAULT_PROFILE_GAPS
    eta_fast_upper_bound: float = 6.0
    optimizer_starts: int = 12
    profile_starts: int = 4
    optimizer_maxiter: int = 1500
    random_seed: int = 852
    with_oos: bool = False
    price_options: bool = True
    resume: bool = True


def build_scenario_specs(settings: StudySettings) -> list[ScenarioSpec]:
    """Construct the three one-dimensional diagnostics and reuse overlaps."""
    baseline_cutoff = min(settings.short_end_cutoffs)
    widest_cap = max(settings.gap_caps)
    candidates = [
        *(
            ScenarioSpec(settings.reference_window, cap, baseline_cutoff)
            for cap in settings.gap_caps
        ),
        *(ScenarioSpec(window, widest_cap, baseline_cutoff) for window in settings.windows),
        *(
            ScenarioSpec(settings.reference_window, widest_cap, cutoff)
            for cutoff in settings.short_end_cutoffs
        ),
    ]
    unique: dict[str, ScenarioSpec] = {}
    for spec in candidates:
        unique.setdefault(spec.scenario_id, spec)
    return list(unique.values())


def prepare_sample(
    evaluation_date: str,
    window_dates: int,
    min_sessions_exclusive: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a strict-calendar sample and apply the diagnostic short-end cutoff."""
    sample, audit, spot_history = load_calibration_sample(
        evaluation_date=evaluation_date,
        window_dates=window_dates,
    )
    sample = sample.loc[sample["sessions_to_expiry"] > min_sessions_exclusive].copy()
    sample = sample.sort_values(["date", "tau", "contract"]).reset_index(drop=True)
    actual_dates = int(sample["date"].nunique())
    if actual_dates != window_dates:
        raise ValueError(
            f"Short-end cutoff leaves {actual_dates} curve dates; expected {window_dates}"
        )
    return sample, audit, spot_history


def _selected_quote(
    fitted_panel: pd.DataFrame,
    evaluation_date: str,
    contract: str,
) -> pd.Series:
    rows = fitted_panel.loc[
        (fitted_panel["date"] == pd.Timestamp(evaluation_date))
        & (fitted_panel["contract"] == contract.upper())
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one accepted {contract.upper()} quote on {evaluation_date}, "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def _price(
    params: TwoFactorParams,
    latest_state: pd.Series,
    quote: pd.Series,
    historical_volatility: float,
) -> dict[str, float]:
    contract = CarryPutContract(
        initial_spot=float(quote["spot"]),
        initial_futures=float(quote["futures_price"]),
        sessions_to_expiry=int(quote["sessions_to_expiry"]),
        periods_per_year=PERIODS_PER_YEAR,
    )
    ou_params = TwoFactorOUParams(
        kappa_slow=params.kappa_slow,
        kappa_fast=params.kappa_fast,
        theta=params.theta,
        eta_slow=params.eta_slow,
        eta_fast=params.eta_fast,
    )
    state = FactorState(
        slow=float(latest_state["filtered_slow_state"]),
        fast=float(latest_state["filtered_fast_state"]),
    )
    gbm = GBMParams(
        risk_free_rate=RISK_FREE_RATE,
        volatility=historical_volatility,
    )
    result = price_american_carry_put(
        contract,
        ou_params,
        state,
        gbm,
        numerical=NumericalConfig(),
    )
    return {
        "option_price": result.price,
        "normalized_option_price": result.normalized_price,
        "locked_carry": result.locked_carry,
        "model_initial_carry": result.model_initial_carry,
        "model_initial_futures": result.model_initial_futures,
        "initial_futures_model_error": result.initial_futures_model_error,
        "slow_delta_pathwise": result.slow_curve_delta.pathwise_delta,
        "slow_delta_bump": result.slow_curve_delta.bump_and_value_delta,
        "fast_delta_pathwise": result.fast_curve_delta.pathwise_delta,
        "fast_delta_bump": result.fast_curve_delta.bump_and_value_delta,
        "fixed_carry_scale_delta": result.fixed_carry_scale_delta,
    }


def _error_metrics(fitted: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    carry_fit = fitted["carry_residual"].to_numpy(dtype=float)
    futures_fit = fitted["futures_residual"].to_numpy(dtype=float)
    carry_prediction = fitted["carry_prediction_error"].to_numpy(dtype=float)
    futures_prediction = fitted["futures_prediction_error"].to_numpy(dtype=float)
    return {
        f"{prefix}posterior_carry_rmse_bps": float(np.sqrt(np.mean(carry_fit**2)) * 10_000),
        f"{prefix}posterior_futures_rmse_points": float(np.sqrt(np.mean(futures_fit**2))),
        f"{prefix}recursive_prior_carry_rmse_bps": float(
            np.sqrt(np.mean(carry_prediction**2)) * 10_000
        ),
        f"{prefix}recursive_prior_futures_rmse_points": float(
            np.sqrt(np.mean(futures_prediction**2))
        ),
    }


def _strict_oos_metrics(
    sample: pd.DataFrame,
    settings: StudySettings,
    spec: ScenarioSpec,
    output_dir: Path,
) -> dict[str, float | int | bool]:
    dates = pd.Index(pd.to_datetime(sample["date"].unique())).sort_values()
    split_index = int(np.floor(0.8 * len(dates)))
    training_dates = dates[:split_index]
    test_dates = dates[split_index:]
    training = sample.loc[sample["date"].isin(training_dates)].copy()
    estimate = estimate_two_factor_configurable(
        training,
        gap_function=_gap_function,
        starts=settings.optimizer_starts,
        maxiter=settings.optimizer_maxiter,
        seed=settings.random_seed,
        eta_fast_upper_bound=settings.eta_fast_upper_bound,
        kappa_gap_upper_bound=spec.kappa_gap_upper_bound,
    )
    estimate.optimizer_runs.to_csv(
        output_dir / f"{spec.scenario_id}_oos_optimizer_runs.csv", index=False
    )
    filtered = two_factor_kalman_filter(
        sample,
        estimate.params,
        gap_function=_gap_function,
    )
    fitted = attach_two_factor_fits(sample, filtered.states, estimate.params)
    test = fitted.loc[fitted["date"].isin(test_dates)]
    carry = test["carry_prediction_error"].to_numpy(dtype=float)
    futures = test["futures_prediction_error"].to_numpy(dtype=float)
    gap = estimate.params.kappa_fast - estimate.params.kappa_slow
    return {
        "oos_training_dates": len(training_dates),
        "oos_test_dates": len(test_dates),
        "oos_carry_rmse_bps": float(np.sqrt(np.mean(carry**2)) * 10_000),
        "oos_futures_rmse_points": float(np.sqrt(np.mean(futures**2))),
        "oos_train_kappa_gap": gap,
        "oos_train_gap_at_upper_bound": bool(gap >= spec.kappa_gap_upper_bound - 1e-6),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def _load_cached(path: Path, settings: StudySettings, spec: ScenarioSpec) -> dict[str, Any] | None:
    if not settings.resume or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "evaluation_date": settings.evaluation_date,
        "contract": settings.contract.upper(),
        "window_dates": spec.window_dates,
        "kappa_gap_upper_bound": spec.kappa_gap_upper_bound,
        "min_sessions_exclusive": spec.min_sessions_exclusive,
        "eta_fast_upper_bound": settings.eta_fast_upper_bound,
        "optimizer_starts": settings.optimizer_starts,
        "optimizer_maxiter": settings.optimizer_maxiter,
        "with_oos": settings.with_oos,
        "price_options": settings.price_options,
    }
    if all(payload.get(key) == value for key, value in expected.items()):
        print(f"[cache] {spec.scenario_id}", flush=True)
        return payload
    return None


def run_scenario(
    settings: StudySettings,
    spec: ScenarioSpec,
    fit_dir: Path,
) -> dict[str, Any]:
    """Calibrate, filter, diagnose, price, and checkpoint one specification."""
    fit_dir.mkdir(parents=True, exist_ok=True)
    json_path = fit_dir / f"{spec.scenario_id}.json"
    cached = _load_cached(json_path, settings, spec)
    if cached is not None:
        return cached

    started = time.perf_counter()
    print(f"[fit] {spec.scenario_id}", flush=True)
    sample, _, spot_history = prepare_sample(
        settings.evaluation_date,
        spec.window_dates,
        spec.min_sessions_exclusive,
    )
    estimate = estimate_two_factor_configurable(
        sample,
        gap_function=_gap_function,
        starts=settings.optimizer_starts,
        maxiter=settings.optimizer_maxiter,
        seed=settings.random_seed,
        eta_fast_upper_bound=settings.eta_fast_upper_bound,
        kappa_gap_upper_bound=spec.kappa_gap_upper_bound,
    )
    estimate.optimizer_runs.to_csv(fit_dir / f"{spec.scenario_id}_optimizer_runs.csv", index=False)
    filtered = two_factor_kalman_filter(
        sample,
        estimate.params,
        gap_function=_gap_function,
    )
    fitted = attach_two_factor_fits(sample, filtered.states, estimate.params)
    quote = _selected_quote(fitted, settings.evaluation_date, settings.contract)
    latest_state = filtered.states.iloc[-1]
    historical_volatility = estimate_historical_volatility(spot_history)
    gap = estimate.params.kappa_fast - estimate.params.kappa_slow
    row: dict[str, Any] = {
        "scenario_id": spec.scenario_id,
        "evaluation_date": settings.evaluation_date,
        "contract": settings.contract.upper(),
        "window_dates": spec.window_dates,
        "sample_start": str(pd.Timestamp(sample["date"].min()).date()),
        "sample_end": str(pd.Timestamp(sample["date"].max()).date()),
        "accepted_observations": len(sample),
        "distinct_contracts": int(sample["contract"].nunique()),
        "min_sessions_exclusive": spec.min_sessions_exclusive,
        "kappa_gap_upper_bound": spec.kappa_gap_upper_bound,
        "eta_fast_upper_bound": settings.eta_fast_upper_bound,
        "optimizer_starts": settings.optimizer_starts,
        "optimizer_maxiter": settings.optimizer_maxiter,
        "with_oos": settings.with_oos,
        "price_options": settings.price_options,
        **asdict(estimate.params),
        "kappa_gap": gap,
        "slow_half_life_sessions": PERIODS_PER_YEAR * np.log(2.0) / estimate.params.kappa_slow,
        "fast_half_life_sessions": PERIODS_PER_YEAR * np.log(2.0) / estimate.params.kappa_fast,
        "gap_at_upper_bound": bool(gap >= spec.kappa_gap_upper_bound - 1e-6),
        "eta_fast_at_upper_bound": bool(
            estimate.params.eta_fast >= settings.eta_fast_upper_bound - 1e-6
        ),
        "log_likelihood": estimate.log_likelihood,
        "aic": 12.0 - 2.0 * estimate.log_likelihood,
        "bic": 6.0 * np.log(len(sample)) - 2.0 * estimate.log_likelihood,
        "optimizer_converged": estimate.converged,
        "optimizer_message": estimate.message,
        "latest_slow_state": float(latest_state["filtered_slow_state"]),
        "latest_fast_state": float(latest_state["filtered_fast_state"]),
        "latest_instantaneous_carry": float(latest_state["filtered_instantaneous_carry"]),
        "latest_instantaneous_std": float(latest_state["filtered_instantaneous_std"]),
        "historical_spot_volatility": historical_volatility,
        "observed_initial_spot": float(quote["spot"]),
        "observed_initial_futures": float(quote["futures_price"]),
        "contract_sessions_to_expiry": int(quote["sessions_to_expiry"]),
        **_error_metrics(fitted),
    }
    if settings.price_options:
        row.update(_price(estimate.params, latest_state, quote, historical_volatility))
    if settings.with_oos:
        row.update(_strict_oos_metrics(sample, settings, spec, fit_dir))
    row["elapsed_seconds"] = time.perf_counter() - started
    json_path.write_text(
        json.dumps(_jsonable(row), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[done] {spec.scenario_id}: gap={gap:.6f}, "
        f"ll={estimate.log_likelihood:.3f}, elapsed={row['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return row


def _params_from_row(row: dict[str, Any]) -> TwoFactorParams:
    return TwoFactorParams(
        kappa_slow=float(row["kappa_slow"]),
        kappa_fast=float(row["kappa_fast"]),
        theta=float(row["theta"]),
        eta_slow=float(row["eta_slow"]),
        eta_fast=float(row["eta_fast"]),
        sigma_epsilon=float(row["sigma_epsilon"]),
    )


def run_gap_profile(
    settings: StudySettings,
    warm_params: TwoFactorParams,
    profile_dir: Path,
) -> pd.DataFrame:
    """Profile fixed gap values and map them into prices and deltas."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    baseline_cutoff = min(settings.short_end_cutoffs)
    sample, _, spot_history = prepare_sample(
        settings.evaluation_date,
        settings.reference_window,
        baseline_cutoff,
    )
    historical_volatility = estimate_historical_volatility(spot_history)
    rows: list[dict[str, Any]] = []
    current_warm = warm_params
    for fixed_gap in settings.profile_gaps:
        label = f"gap{fixed_gap:g}".replace(".", "p")
        json_path = profile_dir / f"{label}.json"
        if settings.resume and json_path.exists():
            row = json.loads(json_path.read_text(encoding="utf-8"))
            expected = (
                row.get("evaluation_date") == settings.evaluation_date
                and row.get("contract") == settings.contract.upper()
                and row.get("window_dates") == settings.reference_window
                and row.get("min_sessions_exclusive") == baseline_cutoff
                and row.get("fixed_kappa_gap") == fixed_gap
                and row.get("eta_fast_upper_bound") == settings.eta_fast_upper_bound
                and row.get("optimizer_starts") == settings.profile_starts
                and row.get("price_options") == settings.price_options
            )
            if expected:
                print(f"[cache] profile {label}", flush=True)
                rows.append(row)
                current_warm = _params_from_row(row)
                continue

        print(f"[profile] fixed gap={fixed_gap:g}", flush=True)
        started = time.perf_counter()
        estimate = estimate_with_fixed_kappa_gap(
            sample,
            fixed_gap,
            gap_function=_gap_function,
            starts=settings.profile_starts,
            maxiter=settings.optimizer_maxiter,
            seed=settings.random_seed,
            eta_fast_upper_bound=settings.eta_fast_upper_bound,
            warm_params=current_warm,
        )
        estimate.optimizer_runs.to_csv(profile_dir / f"{label}_optimizer_runs.csv", index=False)
        filtered = two_factor_kalman_filter(
            sample,
            estimate.params,
            gap_function=_gap_function,
        )
        fitted = attach_two_factor_fits(sample, filtered.states, estimate.params)
        quote = _selected_quote(fitted, settings.evaluation_date, settings.contract)
        latest_state = filtered.states.iloc[-1]
        row = {
            "profile_id": label,
            "evaluation_date": settings.evaluation_date,
            "contract": settings.contract.upper(),
            "window_dates": settings.reference_window,
            "min_sessions_exclusive": baseline_cutoff,
            "fixed_kappa_gap": fixed_gap,
            "eta_fast_upper_bound": settings.eta_fast_upper_bound,
            "optimizer_starts": settings.profile_starts,
            "price_options": settings.price_options,
            **asdict(estimate.params),
            "log_likelihood": estimate.log_likelihood,
            "optimizer_converged": estimate.converged,
            "eta_fast_at_upper_bound": bool(
                estimate.params.eta_fast >= settings.eta_fast_upper_bound - 1e-6
            ),
            "latest_slow_state": float(latest_state["filtered_slow_state"]),
            "latest_fast_state": float(latest_state["filtered_fast_state"]),
            **_error_metrics(fitted),
        }
        if settings.price_options:
            row.update(_price(estimate.params, latest_state, quote, historical_volatility))
        row["elapsed_seconds"] = time.perf_counter() - started
        json_path.write_text(
            json.dumps(_jsonable(row), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        rows.append(row)
        current_warm = estimate.params
        print(
            f"[done] profile {label}: ll={estimate.log_likelihood:.3f}, "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )

    profile = pd.DataFrame(rows).sort_values("fixed_kappa_gap").reset_index(drop=True)
    maximum = float(profile["log_likelihood"].max())
    profile["log_likelihood_below_profile_max"] = profile["log_likelihood"] - maximum
    profile["likelihood_ratio_vs_profile_max"] = 2.0 * (maximum - profile["log_likelihood"])
    profile["relative_likelihood"] = np.exp(np.maximum(profile["log_likelihood"] - maximum, -745.0))
    return profile


def _subset_tables(
    results: pd.DataFrame,
    settings: StudySettings,
) -> dict[str, pd.DataFrame]:
    baseline_cutoff = min(settings.short_end_cutoffs)
    widest_cap = max(settings.gap_caps)
    return {
        "cap_sensitivity": results.loc[
            (results["window_dates"] == settings.reference_window)
            & (results["min_sessions_exclusive"] == baseline_cutoff)
            & (results["kappa_gap_upper_bound"].isin(settings.gap_caps))
        ].sort_values("kappa_gap_upper_bound"),
        "window_sensitivity": results.loc[
            (results["kappa_gap_upper_bound"] == widest_cap)
            & (results["min_sessions_exclusive"] == baseline_cutoff)
            & (results["window_dates"].isin(settings.windows))
        ].sort_values("window_dates"),
        "short_end_sensitivity": results.loc[
            (results["window_dates"] == settings.reference_window)
            & (results["kappa_gap_upper_bound"] == widest_cap)
            & (results["min_sessions_exclusive"].isin(settings.short_end_cutoffs))
        ].sort_values("min_sessions_exclusive"),
    }


def _plot_pair(
    table: pd.DataFrame,
    x: str,
    xlabel: str,
    title: str,
    output_path: Path,
    price_options: bool,
    diagnostic_column: str,
    diagnostic_label: str,
) -> None:
    columns = 3 if price_options else 2
    figure, axes = plt.subplots(1, columns, figsize=(5.2 * columns, 4.2))
    axes[0].plot(table[x], table["kappa_gap"], "o-", label="Estimated gap")
    axes[0].plot(table[x], table["kappa_gap_upper_bound"], "--", label="Gap cap")
    axes[0].set(ylabel="Annualized kappa gap", xlabel=xlabel)
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(table[x], table[diagnostic_column], "o-")
    axes[1].set(ylabel=diagnostic_label, xlabel=xlabel)
    axes[1].grid(alpha=0.25)
    if price_options:
        axes[2].plot(table[x], table["option_price"], "o-", label="Option price")
        delta_axis = axes[2].twinx()
        delta_axis.plot(
            table[x], table["slow_delta_pathwise"], "s--", color="tab:orange", label="Slow delta"
        )
        delta_axis.plot(
            table[x], table["fast_delta_pathwise"], "^--", color="tab:green", label="Fast delta"
        )
        axes[2].set(xlabel=xlabel, ylabel="Option points")
        delta_axis.set_ylabel("Futures-equivalent delta")
        handles_a, labels_a = axes[2].get_legend_handles_labels()
        handles_b, labels_b = delta_axis.get_legend_handles_labels()
        axes[2].legend(handles_a + handles_b, labels_a + labels_b, loc="best")
        axes[2].grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def make_charts(
    tables: dict[str, pd.DataFrame],
    profile: pd.DataFrame,
    settings: StudySettings,
    chart_dir: Path,
) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_specs = {
        "cap_sensitivity": (
            "kappa_gap_upper_bound",
            "Kappa-gap upper bound",
            "Boundary sensitivity",
            "log_likelihood",
            "Log likelihood (common data)",
        ),
        "window_sensitivity": (
            "window_dates",
            "Calibration curve dates",
            "Sample-window sensitivity",
            "recursive_prior_futures_rmse_points",
            "Recursive-prior futures RMSE (points)",
        ),
        "short_end_sensitivity": (
            "min_sessions_exclusive",
            "Exclude sessions-to-expiry <=",
            "Short-end data sensitivity",
            "recursive_prior_futures_rmse_points",
            "Recursive-prior futures RMSE (points)",
        ),
    }
    for name, table in tables.items():
        x, xlabel, title, diagnostic_column, diagnostic_label = chart_specs[name]
        _plot_pair(
            table,
            x,
            xlabel,
            title,
            chart_dir / f"{name}.png",
            settings.price_options,
            diagnostic_column,
            diagnostic_label,
        )

    columns = 3 if settings.price_options else 2
    figure, axes = plt.subplots(1, columns, figsize=(5.2 * columns, 4.2))
    axes[0].plot(
        profile["fixed_kappa_gap"],
        profile["log_likelihood_below_profile_max"],
        "o-",
    )
    axes[0].axhline(-1.920729, color="tab:red", linestyle="--", label="95% LR cutoff")
    axes[0].set(xlabel="Fixed kappa gap", ylabel="Log likelihood below profile max")
    axes[0].set_ylim(-15.0, 1.0)
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(profile["fixed_kappa_gap"], profile["eta_fast"], "o-")
    axes[1].axhline(
        settings.eta_fast_upper_bound,
        color="tab:red",
        linestyle="--",
        label="eta-fast cap",
    )
    axes[1].set(xlabel="Fixed kappa gap", ylabel="Re-optimized eta-fast")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    if settings.price_options:
        axes[2].plot(
            profile["fixed_kappa_gap"], profile["option_price"], "o-", label="Option price"
        )
        delta_axis = axes[2].twinx()
        delta_axis.plot(
            profile["fixed_kappa_gap"],
            profile["slow_delta_pathwise"],
            "s--",
            color="tab:orange",
            label="Slow delta",
        )
        delta_axis.plot(
            profile["fixed_kappa_gap"],
            profile["fast_delta_pathwise"],
            "^--",
            color="tab:green",
            label="Fast delta",
        )
        axes[2].set(xlabel="Fixed kappa gap", ylabel="Option points")
        delta_axis.set_ylabel("Futures-equivalent delta")
        handles_a, labels_a = axes[2].get_legend_handles_labels()
        handles_b, labels_b = delta_axis.get_legend_handles_labels()
        axes[2].legend(handles_a + handles_b, labels_a + labels_b, loc="best")
        axes[2].grid(alpha=0.25)
    figure.suptitle("Fixed kappa-gap profile")
    figure.tight_layout()
    figure.savefig(chart_dir / "fixed_gap_profile.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def summarize_findings(
    tables: dict[str, pd.DataFrame],
    profile: pd.DataFrame,
    settings: StudySettings,
) -> dict[str, Any]:
    cap = tables["cap_sensitivity"]
    price_column = "option_price" if settings.price_options else None
    findings: dict[str, Any] = {
        "all_cap_solutions_on_boundary": bool(cap["gap_at_upper_bound"].all()),
        "largest_cap_estimated_gap": float(cap.iloc[-1]["kappa_gap"]),
        "log_likelihood_gain_from_smallest_to_largest_cap": float(
            cap.iloc[-1]["log_likelihood"] - cap.iloc[0]["log_likelihood"]
        ),
        "profile_best_fixed_gap": float(
            profile.loc[profile["log_likelihood"].idxmax(), "fixed_kappa_gap"]
        ),
        "profile_95pct_supported_min": None,
        "profile_95pct_supported_max": None,
    }
    supported = profile.loc[profile["likelihood_ratio_vs_profile_max"] <= 3.841459]
    if not supported.empty:
        findings["profile_95pct_supported_min"] = float(supported["fixed_kappa_gap"].min())
        findings["profile_95pct_supported_max"] = float(supported["fixed_kappa_gap"].max())
    if price_column:
        findings.update(
            cap_option_price_min=float(cap[price_column].min()),
            cap_option_price_max=float(cap[price_column].max()),
            profile_option_price_min=float(profile[price_column].min()),
            profile_option_price_max=float(profile[price_column].max()),
        )
    return findings


def run_study(settings: StudySettings, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = output_dir / "fits"
    profile_dir = output_dir / "profile_fits"
    specs = build_scenario_specs(settings)
    print(f"Running {len(specs)} unique free-gap scenarios", flush=True)
    rows = [run_scenario(settings, spec, fit_dir) for spec in specs]
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "all_scenarios.csv", index=False)
    tables = _subset_tables(results, settings)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    widest_cap = max(settings.gap_caps)
    baseline_cutoff = min(settings.short_end_cutoffs)
    warm_row = next(
        row
        for row in rows
        if row["window_dates"] == settings.reference_window
        and row["kappa_gap_upper_bound"] == widest_cap
        and row["min_sessions_exclusive"] == baseline_cutoff
    )
    profile = run_gap_profile(settings, _params_from_row(warm_row), profile_dir)
    profile.to_csv(output_dir / "fixed_gap_profile.csv", index=False)
    make_charts(tables, profile, settings, output_dir / "charts")
    findings = summarize_findings(tables, profile, settings)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "settings": asdict(settings),
        "unique_free_gap_scenarios": len(specs),
        "profile_points": len(profile),
        "findings": findings,
        "metric_note": (
            "recursive_prior_* errors are one-step-prior errors conditional on parameters "
            "estimated from the full scenario sample; oos_* metrics, when requested, use "
            "parameters estimated on the first 80% of dates."
        ),
    }
    (output_dir / "study_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _tuple_int(values: Iterable[str]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _tuple_float(values: Iterable[str]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-date", default=DEFAULT_EVALUATION_DATE)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--reference-window", type=int, default=488)
    parser.add_argument("--windows", nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--gap-caps", nargs="+", default=list(DEFAULT_GAP_CAPS))
    parser.add_argument("--short-end-cutoffs", nargs="+", default=list(DEFAULT_SHORT_END_CUTOFFS))
    parser.add_argument("--profile-gaps", nargs="+", default=list(DEFAULT_PROFILE_GAPS))
    parser.add_argument("--eta-fast-upper-bound", type=float, default=6.0)
    parser.add_argument("--optimizer-starts", type=int, default=12)
    parser.add_argument("--profile-starts", type=int, default=4)
    parser.add_argument("--optimizer-maxiter", type=int, default=1500)
    parser.add_argument("--random-seed", type=int, default=852)
    parser.add_argument("--with-oos", action="store_true")
    parser.add_argument("--skip-pricing", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STUDY_ROOT / "outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = StudySettings(
        evaluation_date=str(pd.Timestamp(args.evaluation_date).date()),
        contract=args.contract.strip().upper(),
        reference_window=args.reference_window,
        windows=_tuple_int(args.windows),
        gap_caps=_tuple_float(args.gap_caps),
        short_end_cutoffs=_tuple_int(args.short_end_cutoffs),
        profile_gaps=_tuple_float(args.profile_gaps),
        eta_fast_upper_bound=args.eta_fast_upper_bound,
        optimizer_starts=args.optimizer_starts,
        profile_starts=args.profile_starts,
        optimizer_maxiter=args.optimizer_maxiter,
        random_seed=args.random_seed,
        with_oos=args.with_oos,
        price_options=not args.skip_pricing,
        resume=not args.no_resume,
    )
    summary = run_study(settings, args.output_dir.resolve())
    print(json.dumps(_jsonable(summary["findings"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
