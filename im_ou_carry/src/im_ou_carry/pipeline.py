"""End-to-end acquisition, estimation, evaluation, export, and chart pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calendar import trading_days_between
from .config import ensure_directories
from .data import acquire_data, normalized_market_panel
from .diagnostics import (
    attach_benchmarks,
    benchmark_metrics,
    curve_shape_flags,
    expiry_roll_diagnostics,
    maturity_dependence_test,
    model_metrics,
    residual_by_maturity,
    residual_statistical_tests,
    residual_time_series,
)
from .estimation import EstimationResult, estimate_ou, parameter_table
from .fitting import attach_model_fits
from .kalman import kalman_filter
from .plots import (
    plot_acf,
    plot_curve,
    plot_residuals,
    plot_rolling_parameters,
    plot_selected_curves,
    plot_states,
)
from .quality import prepare_implied_carry


def gap_function(config: dict[str, Any]):
    periods = int(config["calendar"].get("periods_per_year", 244))
    closures = config["calendar"].get("special_exchange_closures", [])

    def gap(start: object, end: object) -> float:
        return trading_days_between(start, end, closures) / float(periods)

    return gap


def evaluation_split_date(panel: pd.DataFrame, fraction: float) -> pd.Timestamp:
    dates = pd.Series(pd.to_datetime(panel["date"].unique())).sort_values().reset_index(drop=True)
    if len(dates) < 20:
        raise ValueError("At least 20 curve dates are required for train/test evaluation")
    test_count = max(1, int(np.ceil(len(dates) * float(fraction))))
    return pd.Timestamp(dates.iloc[-test_count - 1])


def select_main_training_panel(panel: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cfg = config["estimation"]
    mode = str(cfg.get("mode", "full")).lower()
    dates = pd.Series(pd.to_datetime(panel["date"].unique())).sort_values().reset_index(drop=True)
    if mode == "full":
        return panel.copy(), "full"
    if mode == "rolling":
        window = int(cfg.get("rolling_window_dates", 488))
        selected = dates.iloc[-min(window, len(dates)) :]
        return panel.loc[panel["date"].isin(selected)].copy(), f"rolling_{len(selected)}_dates"
    if mode == "split":
        train_end = cfg.get("train_end_date")
        split = pd.Timestamp(train_end) if train_end else evaluation_split_date(panel, float(cfg.get("test_fraction", 0.2)))
        return panel.loc[panel["date"] <= split].copy(), f"split_through_{split.date()}"
    raise ValueError("estimation.mode must be full, rolling, or split")


def _fit(
    panel: pd.DataFrame,
    config: dict[str, Any],
    *,
    starts: int | None = None,
    standard_errors: bool = True,
) -> EstimationResult:
    cfg = config["estimation"]
    return estimate_ou(
        panel,
        gap_function=gap_function(config),
        starts=int(starts if starts is not None else cfg.get("optimizer_starts", 8)),
        maxiter=int(cfg.get("optimizer_maxiter", 1200)),
        seed=int(cfg.get("random_seed", 852)),
        compute_standard_errors=standard_errors,
    )


def train_test_evaluation(
    panel: pd.DataFrame,
    config: dict[str, Any],
    split_date: pd.Timestamp,
) -> tuple[EstimationResult, pd.DataFrame, pd.DataFrame]:
    train = panel.loc[panel["date"] <= split_date].copy()
    test = panel.loc[panel["date"] > split_date].copy()
    starts = max(3, min(5, int(config["estimation"].get("optimizer_starts", 8))))
    result = _fit(train, config, starts=starts, standard_errors=False)
    gap = gap_function(config)
    train_filter = kalman_filter(train, result.params, gap_function=gap)
    train_fits = attach_model_fits(train, train_filter.states, result.params)
    if test.empty:
        return result, train_fits, train_filter.states
    terminal = train_filter.states.iloc[-1]
    test_filter = kalman_filter(
        test,
        result.params,
        initial_mean=float(terminal["filtered_state"]),
        initial_variance=float(terminal["filtered_variance"]),
        initial_date=terminal["date"],
        gap_function=gap,
    )
    test_fits = attach_model_fits(test, test_filter.states, result.params)
    fits = pd.concat([train_fits, test_fits], ignore_index=True).sort_values(["date", "tau"])
    states = pd.concat([train_filter.states, test_filter.states], ignore_index=True).sort_values("date")
    return result, fits, states


def rolling_parameter_estimates(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config["diagnostics"]
    if not bool(cfg.get("rolling_enabled", True)):
        return pd.DataFrame()
    dates = pd.Series(pd.to_datetime(panel["date"].unique())).sort_values().reset_index(drop=True)
    window = min(int(cfg.get("rolling_window_dates", 488)), len(dates))
    step = max(1, int(cfg.get("rolling_step_dates", 63)))
    end_positions = list(range(window - 1, len(dates), step))
    if end_positions and end_positions[-1] != len(dates) - 1:
        end_positions.append(len(dates) - 1)
    rows = []
    for position in end_positions:
        selected = dates.iloc[position - window + 1 : position + 1]
        sample = panel.loc[panel["date"].isin(selected)]
        try:
            fit = _fit(sample, config, starts=int(cfg.get("rolling_optimizer_starts", 3)), standard_errors=False)
            rows.append(
                {
                    "window_start": selected.iloc[0],
                    "window_end": selected.iloc[-1],
                    "n_dates": len(selected),
                    "n_observations": len(sample),
                    **asdict(fit.params),
                    "half_life_sessions": 244.0 * np.log(2.0) / fit.params.kappa,
                    "log_likelihood": fit.log_likelihood,
                    "converged": fit.converged,
                    "message": fit.message,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "window_start": selected.iloc[0],
                    "window_end": selected.iloc[-1],
                    "n_dates": len(selected),
                    "n_observations": len(sample),
                    "converged": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def _representative_dates(fits: pd.DataFrame, count: int) -> list[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(fits["date"].unique())).sort_values().reset_index(drop=True)
    if dates.empty:
        return []
    positions = np.unique(np.linspace(0, len(dates) - 1, min(count, len(dates)), dtype=int))
    return [pd.Timestamp(dates.iloc[position]) for position in positions]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the complete workflow and return a concise summary."""
    ensure_directories(config)
    output_dir = Path(config["project"]["output_dir"])
    chart_dir = Path(config["project"]["chart_dir"])

    spot, futures, download_log = acquire_data(config)
    market = normalized_market_panel(spot, futures, config)
    panel, quality_audit = prepare_implied_carry(market, config)
    if panel["date"].nunique() < 20:
        raise RuntimeError("Too few accepted curve dates after quality filtering")
    _write_csv(panel, output_dir / "implied_carries.csv")
    _write_csv(quality_audit, output_dir / "quality_audit.csv")
    _write_csv(download_log, output_dir / "download_log.csv")

    training_panel, training_label = select_main_training_panel(panel, config)
    main_estimate = _fit(training_panel, config)
    main_filter = kalman_filter(panel, main_estimate.params, gap_function=gap_function(config))
    fits = attach_model_fits(panel, main_filter.states, main_estimate.params)

    params = parameter_table(main_estimate)
    params["training_scheme"] = training_label
    params["training_start"] = training_panel["date"].min()
    params["training_end"] = training_panel["date"].max()
    params["n_training_dates"] = training_panel["date"].nunique()
    params["n_training_observations"] = len(training_panel)
    _write_csv(params, output_dir / "parameters.csv")
    _write_csv(main_estimate.optimizer_runs, output_dir / "optimizer_runs.csv")
    _write_csv(main_filter.states, output_dir / "filtered_states.csv")
    _write_csv(fits, output_dir / "fitted_curves.csv")
    residual_columns = [
        "date", "contract", "expiry", "sessions_to_expiry", "tau", "spot", "futures_price",
        "implied_carry", "predicted_carry", "fitted_carry", "carry_prediction_error", "carry_residual",
        "predicted_futures_price", "fitted_futures_price", "futures_prediction_error", "futures_residual",
    ]
    _write_csv(fits[residual_columns], output_dir / "residuals.csv")

    split = evaluation_split_date(panel, float(config["diagnostics"].get("evaluation_test_fraction", 0.2)))
    evaluation_estimate, evaluation_fits, evaluation_states = train_test_evaluation(panel, config, split)
    _write_csv(parameter_table(evaluation_estimate), output_dir / "evaluation_train_parameters.csv")
    _write_csv(evaluation_fits, output_dir / "evaluation_fits.csv")
    metrics = model_metrics(evaluation_fits, split)
    benchmark_panel = attach_benchmarks(panel)
    benchmark_table = benchmark_metrics(benchmark_panel, split)
    metrics = pd.concat([metrics, benchmark_table], ignore_index=True)
    _write_csv(metrics, output_dir / "calibration_metrics.csv")
    _write_csv(benchmark_panel, output_dir / "benchmark_fits.csv")

    maturity = residual_by_maturity(fits)
    residual_ts = residual_time_series(fits)
    acf_table, statistical_tests = residual_statistical_tests(
        residual_ts, int(config["diagnostics"].get("acf_lags", 20))
    )
    expiry_roll = expiry_roll_diagnostics(fits)
    shape_flags = curve_shape_flags(panel, float(config["diagnostics"].get("shape_tolerance_bps", 5.0)))
    maturity_test = maturity_dependence_test(fits)
    statistical_tests = pd.concat([statistical_tests, maturity_test], ignore_index=True)
    rolling = rolling_parameter_estimates(panel, config)
    for frame, filename in (
        (maturity, "residuals_by_maturity.csv"),
        (residual_ts, "residual_time_series.csv"),
        (acf_table, "residual_acf.csv"),
        (statistical_tests, "residual_statistical_tests.csv"),
        (expiry_roll, "expiry_roll_diagnostics.csv"),
        (shape_flags, "curve_shape_flags.csv"),
        (rolling, "rolling_parameters.csv"),
    ):
        _write_csv(frame, output_dir / filename)

    latest_date = pd.Timestamp(fits["date"].max())
    plot_curve(fits, latest_date, chart_dir / "latest_curve.png", "Latest curve: ")
    representatives = _representative_dates(fits, int(config["diagnostics"].get("representative_dates", 4)))
    plot_selected_curves(fits, representatives, chart_dir / "representative_curves.png", "Representative observed and fitted curves")
    worst_count = int(config["diagnostics"].get("worst_dates", 4))
    worst_dates = residual_ts.nlargest(worst_count, "carry_rmse")["date"].tolist()
    plot_selected_curves(fits, worst_dates, chart_dir / "worst_fit_curves.png", "Dates with largest carry RMSE")
    plot_residuals(fits, residual_ts, maturity, chart_dir)
    plot_acf(acf_table, chart_dir / "residual_acf.png")
    plot_states(main_filter.states, chart_dir / "filtered_state.png")
    plot_rolling_parameters(rolling, chart_dir / "rolling_parameters.png")

    latest_state = main_filter.states.iloc[-1]
    shape_problem_rate = float(shape_flags["one_factor_shape_limitation"].mean())
    summary = {
        "data_start": str(pd.Timestamp(panel["date"].min()).date()),
        "data_end": str(latest_date.date()),
        "accepted_observations": int(len(panel)),
        "accepted_curve_dates": int(panel["date"].nunique()),
        "excluded_observations": int(quality_audit["excluded"].sum()),
        "training_scheme": training_label,
        "parameters": asdict(main_estimate.params),
        "standard_errors": main_estimate.standard_errors,
        "hessian_stable": main_estimate.hessian_stable,
        "log_likelihood": main_estimate.log_likelihood,
        "optimizer_converged": main_estimate.converged,
        "half_life_sessions": float(244.0 * np.log(2.0) / main_estimate.params.kappa),
        "current_filtered_state": float(latest_state["filtered_state"]),
        "current_filtered_std": float(latest_state["filtered_std"]),
        "evaluation_split_date": str(split.date()),
        "one_factor_shape_limitation_date_fraction": shape_problem_rate,
        "output_dir": str(output_dir),
        "chart_dir": str(chart_dir),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
