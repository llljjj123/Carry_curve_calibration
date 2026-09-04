"""End-to-end two-factor workflow with a directly comparable one-factor baseline."""

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
    expiry_roll_diagnostics,
    maturity_dependence_test,
    model_metrics,
    residual_by_maturity,
    residual_statistical_tests,
    residual_time_series,
    shape_fit_comparison,
    standardized_innovation_diagnostics,
)
from .estimation import EstimationResult, estimate_ou, parameter_table
from .fitting import attach_model_fits
from .kalman import kalman_filter
from .observation import noise_parameter_name, normalize_observation_noise_model
from .plots import (
    plot_acf,
    plot_curve_comparison,
    plot_factor_loadings,
    plot_factor_states,
    plot_residuals,
    plot_selected_curves,
    plot_two_factor_rolling,
)
from .quality import prepare_implied_carry
from .two_factor import two_factor_kalman_filter
from .two_factor_estimation import (
    TwoFactorEstimationResult,
    estimate_two_factor_ou,
    two_factor_parameter_table,
)
from .two_factor_fitting import attach_two_factor_fits


def gap_function(config: dict[str, Any]):
    periods = int(config["calendar"].get("periods_per_year", 244))
    closures = config["calendar"].get("special_exchange_closures", [])

    def gap(start: object, end: object) -> float:
        return trading_days_between(start, end, closures) / float(periods)

    return gap


def evaluation_split_date(panel: pd.DataFrame, fraction: float) -> pd.Timestamp:
    dates = pd.Series(pd.to_datetime(panel["date"].unique())).sort_values().reset_index(drop=True)
    if len(dates) < 20:
        raise ValueError("At least 20 curve dates are required")
    test_count = max(1, int(np.ceil(len(dates) * float(fraction))))
    return pd.Timestamp(dates.iloc[-test_count - 1])


def select_main_training_panel(panel: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cfg = config["estimation"]
    mode = str(cfg.get("mode", "full")).lower()
    dates = pd.Series(pd.to_datetime(panel["date"].unique())).sort_values().reset_index(drop=True)
    if mode == "full":
        return panel.copy(), "full"
    if mode == "rolling":
        selected = dates.iloc[-min(int(cfg.get("rolling_window_dates", 488)), len(dates)) :]
        return panel.loc[panel["date"].isin(selected)].copy(), f"rolling_{len(selected)}_dates"
    if mode == "split":
        explicit = cfg.get("train_end_date")
        split = pd.Timestamp(explicit) if explicit else evaluation_split_date(
            panel, float(cfg.get("test_fraction", 0.2))
        )
        return panel.loc[panel["date"] <= split].copy(), f"split_through_{split.date()}"
    raise ValueError("estimation.mode must be full, rolling, or split")


def _estimate_two(
    panel: pd.DataFrame,
    config: dict[str, Any],
    *,
    starts: int | None = None,
    standard_errors: bool = True,
) -> TwoFactorEstimationResult:
    cfg = config["estimation"]
    observation_model = normalize_observation_noise_model(
        cfg.get("observation_noise_model", "constant_carry")
    )
    return estimate_two_factor_ou(
        panel,
        gap_function=gap_function(config),
        starts=int(starts if starts is not None else cfg.get("two_factor_optimizer_starts", 12)),
        maxiter=int(cfg.get("optimizer_maxiter", 1500)),
        seed=int(cfg.get("random_seed", 852)),
        compute_standard_errors=standard_errors,
        eta_fast_upper_bound=float(cfg.get("eta_fast_upper_bound", 3.0)),
        kappa_gap_upper_bound=float(cfg.get("kappa_gap_upper_bound", 60.0)),
        observation_noise_model=observation_model,
    )


def _estimate_one(
    panel: pd.DataFrame,
    config: dict[str, Any],
    *,
    starts: int | None = None,
    standard_errors: bool = True,
) -> EstimationResult:
    cfg = config["estimation"]
    observation_model = normalize_observation_noise_model(
        cfg.get("observation_noise_model", "constant_carry")
    )
    return estimate_ou(
        panel,
        gap_function=gap_function(config),
        starts=int(starts if starts is not None else cfg.get("one_factor_optimizer_starts", 5)),
        maxiter=int(cfg.get("optimizer_maxiter", 1500)),
        seed=int(cfg.get("random_seed", 852)),
        compute_standard_errors=standard_errors,
        observation_noise_model=observation_model,
    )


def _information_criteria(log_likelihood: float, parameters: int, observations: int) -> tuple[float, float]:
    aic = 2 * parameters - 2 * log_likelihood
    bic = np.log(observations) * parameters - 2 * log_likelihood
    return aic, bic


def _rolling_estimates(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cfg = config["diagnostics"]
    if not bool(cfg.get("rolling_enabled", True)):
        return pd.DataFrame()
    dates = pd.Series(pd.to_datetime(panel["date"].unique())).sort_values().reset_index(drop=True)
    window = min(int(cfg.get("rolling_window_dates", 488)), len(dates))
    step = max(1, int(cfg.get("rolling_step_dates", 126)))
    positions = list(range(window - 1, len(dates), step))
    if positions[-1] != len(dates) - 1:
        positions.append(len(dates) - 1)
    rows: list[dict[str, object]] = []
    for position in positions:
        selected = dates.iloc[position - window + 1 : position + 1]
        sample = panel.loc[panel["date"].isin(selected)]
        common = {
            "window_start": selected.iloc[0],
            "window_end": selected.iloc[-1],
            "n_dates": len(selected),
            "n_observations": len(sample),
        }
        try:
            two = _estimate_two(
                sample,
                config,
                starts=int(cfg.get("rolling_two_factor_starts", 4)),
                standard_errors=False,
            )
            two_values = asdict(two.params)
            two_sigma = two_values.pop("sigma_epsilon")
            two_values[noise_parameter_name(two.observation_noise_model)] = two_sigma
            rows.append(
                {
                    **common,
                    "model": "two_factor_ou",
                    "observation_noise_model": two.observation_noise_model,
                    **two_values,
                    "slow_half_life_sessions": 244 * np.log(2) / two.params.kappa_slow,
                    "fast_half_life_sessions": 244 * np.log(2) / two.params.kappa_fast,
                    "log_likelihood": two.log_likelihood,
                    "converged": two.converged,
                    "message": two.message,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **common,
                    "model": "two_factor_ou",
                    "converged": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        try:
            one = _estimate_one(
                sample,
                config,
                starts=int(cfg.get("rolling_one_factor_starts", 2)),
                standard_errors=False,
            )
            one_values = asdict(one.params)
            one_sigma = one_values.pop("sigma_epsilon")
            one_values[noise_parameter_name(one.observation_noise_model)] = one_sigma
            rows.append(
                {
                    **common,
                    "model": "one_factor_ou",
                    "observation_noise_model": one.observation_noise_model,
                    **one_values,
                    "half_life_sessions": 244 * np.log(2) / one.params.kappa,
                    "log_likelihood": one.log_likelihood,
                    "converged": one.converged,
                    "message": one.message,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **common,
                    "model": "one_factor_ou",
                    "converged": False,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def _innovation_outputs(
    innovations: pd.DataFrame,
    split: pd.Timestamp,
    model: str,
    lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_daily = []
    all_acf = []
    all_tests = []
    for sample, mask in (
        ("full", pd.Series(True, index=innovations.index)),
        ("in_sample", innovations["date"] <= split),
        ("out_of_sample", innovations["date"] > split),
    ):
        daily, acf_table, tests = standardized_innovation_diagnostics(innovations.loc[mask], lags)
        daily["model"], daily["sample"] = model, sample
        acf_table["model"], acf_table["sample"] = model, sample
        tests["model"], tests["sample"] = model, sample
        all_daily.append(daily)
        all_acf.append(acf_table)
        all_tests.append(tests)
    return pd.concat(all_daily), pd.concat(all_acf), pd.concat(all_tests)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    ensure_directories(config)
    output_dir = Path(config["project"]["output_dir"])
    chart_dir = Path(config["project"]["chart_dir"])

    spot, futures, download_log = acquire_data(config)
    market = normalized_market_panel(spot, futures, config)
    panel, quality_audit = prepare_implied_carry(market, config)
    if panel["date"].nunique() < 20:
        raise RuntimeError("Too few accepted curve dates")
    _write_csv(panel, output_dir / "implied_carries.csv")
    _write_csv(quality_audit, output_dir / "quality_audit.csv")
    _write_csv(download_log, output_dir / "download_log.csv")

    training, training_label = select_main_training_panel(panel, config)
    observation_model = normalize_observation_noise_model(
        config["estimation"].get("observation_noise_model", "constant_carry")
    )
    compute_se = bool(config["estimation"].get("compute_standard_errors", True))
    two_estimate = _estimate_two(training, config, standard_errors=compute_se)
    one_estimate = _estimate_one(training, config, standard_errors=compute_se)
    gap = gap_function(config)
    two_filter = two_factor_kalman_filter(
        panel,
        two_estimate.params,
        gap_function=gap,
        observation_noise_model=observation_model,
    )
    one_filter = kalman_filter(
        panel,
        one_estimate.params,
        gap_function=gap,
        observation_noise_model=observation_model,
    )
    nearest_sessions = panel.groupby("date")["sessions_to_expiry"].min().rename("nearest_contract_sessions")
    two_filter.states = two_filter.states.merge(nearest_sessions, on="date", how="left", validate="one_to_one")
    two_filter.states["weak_instantaneous_observability"] = (
        (two_filter.states["nearest_contract_sessions"] > 21)
        | (two_filter.states["filtered_instantaneous_std"] > 0.04)
    )
    two_fits = attach_two_factor_fits(
        panel,
        two_filter.states,
        two_estimate.params,
        observation_model,
    )
    one_fits = attach_model_fits(
        panel,
        one_filter.states,
        one_estimate.params,
        observation_model,
    )

    two_parameters = two_factor_parameter_table(two_estimate)
    one_parameters = parameter_table(one_estimate)
    for table in (two_parameters, one_parameters):
        table["training_scheme"] = training_label
        table["training_start"] = training["date"].min()
        table["training_end"] = training["date"].max()
        table["n_training_dates"] = training["date"].nunique()
        table["n_training_observations"] = len(training)
    _write_csv(two_parameters, output_dir / "two_factor_parameters.csv")
    _write_csv(one_parameters, output_dir / "one_factor_parameters.csv")
    _write_csv(two_estimate.optimizer_runs, output_dir / "two_factor_optimizer_runs.csv")
    _write_csv(one_estimate.optimizer_runs, output_dir / "one_factor_optimizer_runs.csv")
    _write_csv(two_filter.states, output_dir / "two_factor_filtered_states.csv")
    observability = (
        two_filter.states.assign(
            nearest_maturity_bucket=pd.cut(
                two_filter.states["nearest_contract_sessions"],
                [0, 10, 21, 42, 63, np.inf],
                labels=["1-10", "11-21", "22-42", "43-63", "64+"],
            )
        )
        .groupby("nearest_maturity_bucket", observed=True)
        .agg(
            n_dates=("date", "size"),
            mean_abs_fast_state=("filtered_fast_state", lambda x: np.mean(np.abs(x))),
            max_abs_fast_state=("filtered_fast_state", lambda x: np.max(np.abs(x))),
            mean_instantaneous_std=("filtered_instantaneous_std", "mean"),
            weak_observability_fraction=("weak_instantaneous_observability", "mean"),
        )
        .reset_index()
    )
    _write_csv(observability, output_dir / "state_observability_diagnostics.csv")
    _write_csv(two_filter.innovations, output_dir / "two_factor_sequential_innovations.csv")
    _write_csv(two_fits, output_dir / "two_factor_fitted_curves.csv")
    _write_csv(one_fits, output_dir / "one_factor_fitted_curves.csv")

    residual_columns = [
        "date", "contract", "expiry", "sessions_to_expiry", "tau", "spot", "futures_price",
        "implied_carry", "predicted_carry", "fitted_carry", "carry_prediction_error", "carry_residual",
        "predicted_futures_price", "fitted_futures_price", "futures_prediction_error", "futures_residual",
        "standardized_marginal_prediction_error",
        "observation_noise_model", "observation_noise_carry_sd",
    ]
    _write_csv(two_fits[residual_columns], output_dir / "two_factor_residuals.csv")
    _write_csv(one_fits[residual_columns], output_dir / "one_factor_residuals.csv")

    n_observations = len(training)
    two_aic, two_bic = _information_criteria(two_estimate.log_likelihood, 6, n_observations)
    one_aic, one_bic = _information_criteria(one_estimate.log_likelihood, 4, n_observations)
    model_comparison = pd.DataFrame(
        [
            {"model": "two_factor_ou", "observation_noise_model": observation_model, "parameters": 6, "log_likelihood": two_estimate.log_likelihood, "aic": two_aic, "bic": two_bic},
            {"model": "one_factor_ou", "observation_noise_model": observation_model, "parameters": 4, "log_likelihood": one_estimate.log_likelihood, "aic": one_aic, "bic": one_bic},
        ]
    )
    _write_csv(model_comparison, output_dir / "model_information_criteria.csv")

    split = evaluation_split_date(panel, float(config["diagnostics"].get("evaluation_test_fraction", 0.2)))
    train = panel.loc[panel["date"] <= split]
    eval_two = _estimate_two(
        train,
        config,
        starts=int(config["diagnostics"].get("evaluation_two_factor_starts", 6)),
        standard_errors=False,
    )
    eval_one = _estimate_one(
        train,
        config,
        starts=int(config["diagnostics"].get("evaluation_one_factor_starts", 4)),
        standard_errors=False,
    )
    eval_two_filter = two_factor_kalman_filter(
        panel,
        eval_two.params,
        gap_function=gap,
        observation_noise_model=observation_model,
    )
    eval_one_filter = kalman_filter(
        panel,
        eval_one.params,
        gap_function=gap,
        observation_noise_model=observation_model,
    )
    eval_two_fits = attach_two_factor_fits(
        panel,
        eval_two_filter.states,
        eval_two.params,
        observation_model,
    )
    eval_one_fits = attach_model_fits(
        panel,
        eval_one_filter.states,
        eval_one.params,
        observation_model,
    )
    _write_csv(two_factor_parameter_table(eval_two), output_dir / "evaluation_two_factor_train_parameters.csv")
    _write_csv(parameter_table(eval_one), output_dir / "evaluation_one_factor_train_parameters.csv")

    benchmark_panel = attach_benchmarks(panel)
    metrics = pd.concat(
        [
            model_metrics(eval_two_fits, split, "two_factor_ou"),
            model_metrics(eval_one_fits, split, "one_factor_ou"),
            benchmark_metrics(benchmark_panel, split),
        ],
        ignore_index=True,
    )
    metrics["observation_noise_model"] = observation_model
    _write_csv(metrics, output_dir / "calibration_metrics.csv")
    _write_csv(benchmark_panel, output_dir / "benchmark_fits.csv")

    two_innovations = eval_two_filter.innovations.copy()
    one_innovations = eval_one_fits[["date", "standardized_marginal_prediction_error"]].rename(
        columns={"standardized_marginal_prediction_error": "standardized_innovation"}
    )
    lags = int(config["diagnostics"].get("acf_lags", 20))
    two_daily, two_acf, two_tests = _innovation_outputs(two_innovations, split, "two_factor_ou", lags)
    one_daily, one_acf, one_tests = _innovation_outputs(one_innovations, split, "one_factor_ou", lags)
    innovation_daily = pd.concat([two_daily, one_daily], ignore_index=True)
    innovation_acf = pd.concat([two_acf, one_acf], ignore_index=True)
    innovation_tests = pd.concat([two_tests, one_tests], ignore_index=True)
    _write_csv(innovation_daily, output_dir / "standardized_innovation_daily.csv")
    _write_csv(innovation_acf, output_dir / "standardized_innovation_acf.csv")
    _write_csv(innovation_tests, output_dir / "standardized_innovation_tests.csv")

    maturity_tables = []
    expiry_tables = []
    statistical_tables = []
    for model, fits in (("two_factor_ou", two_fits), ("one_factor_ou", one_fits)):
        maturity = residual_by_maturity(fits)
        maturity["model"] = model
        maturity_tables.append(maturity)
        expiry = expiry_roll_diagnostics(fits)
        expiry["model"] = model
        expiry_tables.append(expiry)
        residual_ts = residual_time_series(fits)
        _, filtered_tests = residual_statistical_tests(residual_ts, lags)
        filtered_tests["model"] = model
        maturity_test = maturity_dependence_test(fits)
        maturity_test["model"] = model
        statistical_tables.extend([filtered_tests, maturity_test])
        if model == "two_factor_ou":
            two_residual_ts = residual_ts
            two_maturity = maturity
    maturity_comparison = pd.concat(maturity_tables, ignore_index=True)
    expiry_comparison = pd.concat(expiry_tables, ignore_index=True)
    filtered_residual_tests = pd.concat(statistical_tables, ignore_index=True)
    shape_comparison = shape_fit_comparison(
        panel,
        one_fits,
        two_fits,
        float(config["diagnostics"].get("shape_tolerance_bps", 5.0)),
    )
    rolling = _rolling_estimates(panel, config)
    _write_csv(maturity_comparison, output_dir / "residuals_by_maturity_comparison.csv")
    _write_csv(expiry_comparison, output_dir / "expiry_roll_comparison.csv")
    _write_csv(filtered_residual_tests, output_dir / "filtered_residual_tests.csv")
    _write_csv(shape_comparison, output_dir / "shape_fit_comparison.csv")
    _write_csv(rolling, output_dir / "rolling_parameter_comparison.csv")
    _write_csv(two_residual_ts, output_dir / "two_factor_residual_time_series.csv")

    latest = pd.Timestamp(panel["date"].max())
    plot_curve_comparison(
        panel,
        one_fits,
        two_fits,
        latest,
        chart_dir / "latest_curve_comparison.png",
        "Latest curve: ",
    )
    dates = pd.Series(pd.to_datetime(panel["date"].unique())).sort_values().reset_index(drop=True)
    count = min(int(config["diagnostics"].get("representative_dates", 4)), len(dates))
    positions = np.unique(np.linspace(0, len(dates) - 1, count, dtype=int))
    representative = [dates.iloc[position] for position in positions]
    plot_selected_curves(
        two_fits,
        representative,
        chart_dir / "two_factor_representative_curves.png",
        "Two-factor representative curves",
    )
    worst_count = int(config["diagnostics"].get("worst_dates", 4))
    worst = two_residual_ts.nlargest(worst_count, "carry_rmse")["date"].tolist()
    plot_selected_curves(
        two_fits,
        worst,
        chart_dir / "two_factor_worst_curves.png",
        "Two-factor worst-fit curves",
    )
    plot_residuals(two_fits, two_residual_ts, two_maturity, chart_dir)
    test_acf = innovation_acf.loc[
        (innovation_acf["model"] == "two_factor_ou")
        & (innovation_acf["sample"] == "out_of_sample")
    ]
    plot_acf(test_acf, chart_dir / "two_factor_out_of_sample_innovation_acf.png")
    plot_factor_states(two_filter.states, two_estimate.params.theta, chart_dir / "two_factor_states.png")
    plot_factor_loadings(
        two_estimate.params.kappa_slow,
        two_estimate.params.kappa_fast,
        chart_dir / "factor_loadings.png",
    )
    if not rolling.empty:
        plot_two_factor_rolling(
            rolling.loc[rolling["model"] == "two_factor_ou"],
            chart_dir / "rolling_two_factor_parameters.png",
        )

    latest_state = two_filter.states.iloc[-1]
    observed_humps = shape_comparison["observed_hump_or_u"]
    hump_capture = float(
        shape_comparison.loc[observed_humps, "two_factor_hump_or_u"].mean()
        if observed_humps.any()
        else np.nan
    )
    two_parameter_values = asdict(two_estimate.params)
    two_sigma = two_parameter_values.pop("sigma_epsilon")
    two_parameter_values[noise_parameter_name(observation_model)] = two_sigma
    two_standard_errors = dict(two_estimate.standard_errors)
    two_sigma_se = two_standard_errors.pop("sigma_epsilon")
    two_standard_errors[noise_parameter_name(observation_model)] = two_sigma_se
    summary = {
        "data_start": str(pd.Timestamp(panel["date"].min()).date()),
        "data_end": str(latest.date()),
        "accepted_observations": int(len(panel)),
        "accepted_curve_dates": int(panel["date"].nunique()),
        "excluded_observations": int(quality_audit["excluded"].sum()),
        "training_scheme": training_label,
        "observation_noise_model": observation_model,
        "two_factor_parameters": two_parameter_values,
        "two_factor_standard_errors": two_standard_errors,
        "two_factor_hessian_stable": two_estimate.hessian_stable,
        "two_factor_log_likelihood": two_estimate.log_likelihood,
        "one_factor_log_likelihood": one_estimate.log_likelihood,
        "two_factor_aic": two_aic,
        "one_factor_aic": one_aic,
        "two_factor_bic": two_bic,
        "one_factor_bic": one_bic,
        "slow_half_life_sessions": float(244 * np.log(2) / two_estimate.params.kappa_slow),
        "fast_half_life_sessions": float(244 * np.log(2) / two_estimate.params.kappa_fast),
        "current_slow_state": float(latest_state["filtered_slow_state"]),
        "current_fast_state": float(latest_state["filtered_fast_state"]),
        "current_filtered_carry": float(latest_state["filtered_instantaneous_carry"]),
        "current_filtered_std": float(latest_state["filtered_instantaneous_std"]),
        "evaluation_split_date": str(split.date()),
        "observed_hump_or_u_date_fraction": float(observed_humps.mean()),
        "two_factor_hump_capture_rate": hump_capture,
        "weak_instantaneous_observability_date_fraction": float(
            two_filter.states["weak_instantaneous_observability"].mean()
        ),
        "output_dir": str(output_dir),
        "chart_dir": str(chart_dir),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
