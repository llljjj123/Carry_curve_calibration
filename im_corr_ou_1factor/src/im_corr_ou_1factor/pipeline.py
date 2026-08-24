"""End-to-end correlated one-factor calibration and reporting workflow."""

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
    innovation_diagnostics,
    likelihood_ratio_table,
    model_metrics,
    residual_by_maturity,
    residual_tests,
    residual_time_series,
)
from .estimation import EstimationResult, estimate_model, profile_rho, result_table
from .filtering import kalman_filter, smooth_states
from .fitting import attach_model_fits
from .plots import (
    plot_acf,
    plot_innovations,
    plot_latest_curves,
    plot_latest_futures,
    plot_profiles,
    plot_residuals,
    plot_rolling,
    plot_selected_curves,
    plot_states,
)
from .quality import prepare_implied_carry


MODEL_SPECS = (
    {"name": "legacy_curve", "mode": "curve", "variant": "legacy", "free_rho": False},
    {"name": "exact_rho0_curve", "mode": "curve", "variant": "exact", "free_rho": False},
    {"name": "exact_corr_curve", "mode": "curve", "variant": "exact", "free_rho": True},
    {"name": "exact_rho0_joint", "mode": "joint", "variant": "exact", "free_rho": False},
    {"name": "exact_corr_joint", "mode": "joint", "variant": "exact", "free_rho": True},
)


def gap_function(config: dict[str, Any]):
    periods = int(config["calendar"].get("periods_per_year", 244))
    closures = config["calendar"].get("special_exchange_closures", [])

    def gap(start: object, end: object) -> float:
        return trading_days_between(start, end, closures) / float(periods)

    return gap


def evaluation_split_date(panel: pd.DataFrame, fraction: float) -> pd.Timestamp:
    dates = pd.Series(pd.to_datetime(panel["date"].unique())).sort_values().reset_index(drop=True)
    if len(dates) < 20:
        raise ValueError("At least 20 dates are required")
    test_count = max(1, int(np.ceil(len(dates) * float(fraction))))
    return pd.Timestamp(dates.iloc[-test_count - 1])


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _estimate_spec(
    panel: pd.DataFrame,
    spec: dict[str, object],
    config: dict[str, Any],
    *,
    starts: int | None = None,
    standard_errors: bool = True,
    supplied_starts=None,
) -> EstimationResult:
    cfg = config["estimation"]
    return estimate_model(
        panel,
        name=str(spec["name"]),
        mode=str(spec["mode"]),
        variant=str(spec["variant"]),
        sigma=float(config["model"]["stock_volatility"]),
        gap_function=gap_function(config),
        free_rho=bool(spec["free_rho"]),
        fixed_rho=0.0,
        starts=int(starts if starts is not None else cfg.get("optimizer_starts", 8)),
        maxiter=int(cfg.get("optimizer_maxiter", 1200)),
        seed=int(cfg.get("random_seed", 852)),
        compute_standard_errors=standard_errors,
        supplied_starts=supplied_starts,
    )


def fit_all_models(panel: pd.DataFrame, config: dict[str, Any], *, standard_errors: bool = True, starts: int | None = None) -> dict[str, EstimationResult]:
    results = {}
    for spec in MODEL_SPECS:
        print(f"Estimating {spec['name']}...", flush=True)
        results[str(spec["name"])] = _estimate_spec(panel, spec, config, starts=starts, standard_errors=standard_errors)
    return results


def filter_and_fit(panel: pd.DataFrame, estimates: dict[str, EstimationResult], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = []
    fits = []
    gap = gap_function(config)
    for result in estimates.values():
        filtered = kalman_filter(panel, result.params, sigma=result.sigma, gap_function=gap, mode=result.mode, variant=result.variant)
        state = smooth_states(filtered.states)
        state["model"] = result.name
        states.append(state)
        fits.append(attach_model_fits(panel, state, result.params, result.sigma, model=result.name, variant=result.variant))
    return pd.concat(states, ignore_index=True), pd.concat(fits, ignore_index=True)


def train_test_evaluation(panel: pd.DataFrame, config: dict[str, Any], split: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, EstimationResult]]:
    train = panel.loc[panel["date"] <= split].copy()
    test = panel.loc[panel["date"] > split].copy()
    starts = min(3, max(1, int(config["estimation"].get("optimizer_starts", 5))))
    estimates = fit_all_models(train, config, standard_errors=False, starts=starts)
    all_fits = []
    all_states = []
    gap = gap_function(config)
    for result in estimates.values():
        train_filter = kalman_filter(train, result.params, sigma=result.sigma, gap_function=gap, mode=result.mode, variant=result.variant)
        train_states = train_filter.states.copy(); train_states["sample"] = "in_sample"; train_states["model"] = result.name
        train_fits = attach_model_fits(train, train_states, result.params, result.sigma, model=result.name, variant=result.variant)
        train_fits["sample"] = "in_sample"
        terminal = train_filter.states.iloc[-1]
        test_filter = kalman_filter(
            test,
            result.params,
            sigma=result.sigma,
            gap_function=gap,
            mode=result.mode,
            variant=result.variant,
            initial_mean=float(terminal["filtered_state"]),
            initial_variance=float(terminal["filtered_variance"]),
            initial_date=terminal["date"],
            initial_spot=float(terminal["spot"]),
        )
        test_states = test_filter.states.copy(); test_states["sample"] = "out_of_sample"; test_states["model"] = result.name
        test_fits = attach_model_fits(test, test_states, result.params, result.sigma, model=result.name, variant=result.variant)
        test_fits["sample"] = "out_of_sample"
        all_states.extend([train_states, test_states])
        all_fits.extend([train_fits, test_fits])
    return pd.concat(all_fits, ignore_index=True), pd.concat(all_states, ignore_index=True), estimates


def rolling_estimates(panel: pd.DataFrame, config: dict[str, Any], full: dict[str, EstimationResult]) -> pd.DataFrame:
    cfg = config["diagnostics"]
    if not bool(cfg.get("rolling_enabled", True)):
        return pd.DataFrame()
    dates = pd.Series(pd.to_datetime(panel["date"].unique())).sort_values().reset_index(drop=True)
    window = min(int(cfg.get("rolling_window_dates", 488)), len(dates))
    step = max(1, int(cfg.get("rolling_step_dates", 126)))
    endpoints = list(range(window - 1, len(dates), step))
    if endpoints[-1] != len(dates) - 1:
        endpoints.append(len(dates) - 1)
    specs = [spec for spec in MODEL_SPECS if spec["name"] in {"exact_corr_curve", "exact_corr_joint"}]
    rows = []
    warm = {name: full[name].params for name in ("exact_corr_curve", "exact_corr_joint")}
    for endpoint in endpoints:
        selected = dates.iloc[endpoint - window + 1:endpoint + 1]
        sample = panel.loc[panel["date"].isin(selected)]
        for spec in specs:
            name = str(spec["name"])
            print(f"Rolling fit {name} through {pd.Timestamp(selected.iloc[-1]).date()}...", flush=True)
            try:
                result = _estimate_spec(
                    sample,
                    spec,
                    config,
                    starts=int(cfg.get("rolling_optimizer_starts", 2)),
                    standard_errors=False,
                    supplied_starts=[warm[name]],
                )
                warm[name] = result.params
                rows.append({"model": name, "window_start": selected.iloc[0], "window_end": selected.iloc[-1], "n_dates": len(selected), "n_observations": len(sample), **asdict(result.params), "sigma": result.sigma, "half_life_sessions": 244 * np.log(2) / result.params.kappa, "log_likelihood": result.log_likelihood, "converged": result.converged})
            except Exception as exc:
                rows.append({"model": name, "window_start": selected.iloc[0], "window_end": selected.iloc[-1], "n_dates": len(selected), "n_observations": len(sample), "converged": False, "message": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Run acquisition, five model fits, profiles, diagnostics, and exports."""
    ensure_directories(config)
    output_dir = Path(config["project"]["output_dir"])
    chart_dir = Path(config["project"]["chart_dir"])
    sigma = float(config["model"]["stock_volatility"])
    if sigma <= 0:
        raise ValueError("model.stock_volatility must be positive")

    spot, futures, download_log = acquire_data(config)
    market = normalized_market_panel(spot, futures, config)
    panel, audit = prepare_implied_carry(market, config)
    _write_csv(panel, output_dir / "implied_carries.csv")
    _write_csv(audit, output_dir / "quality_audit.csv")
    _write_csv(download_log, output_dir / "download_log.csv")

    print(f"Accepted {len(panel)} observations across {panel['date'].nunique()} curve dates.", flush=True)
    estimates = fit_all_models(panel, config)
    states, fits = filter_and_fit(panel, estimates, config)
    parameter_tables = [result_table(result, len(panel), panel["date"].nunique()) for result in estimates.values()]
    parameters = pd.concat(parameter_tables, ignore_index=True)
    optimizer_runs = pd.concat([result.optimizer_runs for result in estimates.values()], ignore_index=True)
    _write_csv(parameters, output_dir / "parameters.csv")
    _write_csv(optimizer_runs, output_dir / "optimizer_runs.csv")
    _write_csv(states, output_dir / "states_filtered_and_smoothed.csv")
    _write_csv(fits, output_dir / "fitted_curves.csv")

    profile_cfg = config["estimation"]["rho_profile_grid"]
    grid = np.linspace(float(profile_cfg["start"]), float(profile_cfg["stop"]), int(profile_cfg["points"]))
    profile_frames = []
    profile_summaries = []
    for model in ("exact_corr_curve", "exact_corr_joint"):
        print(f"Profiling rho for {model}...", flush=True)
        model_grid = grid
        if model == "exact_corr_joint" and "joint_rho_refinement_grid" in config["estimation"]:
            refine_cfg = config["estimation"]["joint_rho_refinement_grid"]
            refine = np.linspace(float(refine_cfg["start"]), float(refine_cfg["stop"]), int(refine_cfg["points"]))
            model_grid = np.unique(np.concatenate([grid, refine]))
        profile, summary = profile_rho(
            panel,
            estimates[model],
            model_grid,
            gap_function=gap_function(config),
            starts=int(config["estimation"].get("profile_optimizer_starts", 1)),
            maxiter=int(config["estimation"].get("optimizer_maxiter", 1200)),
            seed=int(config["estimation"].get("random_seed", 852)),
        )
        profile["model"] = model; profile["mode"] = estimates[model].mode
        profile_frames.append(profile)
        profile_summaries.append({"model": model, "mode": estimates[model].mode, **summary})
    profiles = pd.concat(profile_frames, ignore_index=True)
    profile_summary = pd.DataFrame(profile_summaries)
    _write_csv(profiles, output_dir / "rho_profile_likelihood.csv")
    _write_csv(profile_summary, output_dir / "rho_profile_confidence_intervals.csv")

    lr = pd.concat([
        likelihood_ratio_table(estimates["exact_rho0_curve"], estimates["exact_corr_curve"]),
        likelihood_ratio_table(estimates["exact_rho0_joint"], estimates["exact_corr_joint"]),
    ], ignore_index=True)
    _write_csv(lr, output_dir / "likelihood_ratio_tests.csv")

    split = evaluation_split_date(panel, float(config["diagnostics"].get("evaluation_test_fraction", 0.2)))
    print(f"Running train/test evaluation through split {split.date()}...", flush=True)
    evaluation_fits, evaluation_states, evaluation_estimates = train_test_evaluation(panel, config, split)
    train_metrics = model_metrics(evaluation_fits.loc[evaluation_fits["sample"] == "in_sample"], "in_sample", "filtered")
    test_metrics = model_metrics(evaluation_fits.loc[evaluation_fits["sample"] == "out_of_sample"], "out_of_sample", "predicted")
    benchmark_fits = attach_benchmarks(panel)
    metrics = pd.concat([train_metrics, test_metrics, benchmark_metrics(benchmark_fits, split)], ignore_index=True)
    _write_csv(evaluation_fits, output_dir / "evaluation_fits.csv")
    _write_csv(evaluation_states, output_dir / "evaluation_states.csv")
    _write_csv(metrics, output_dir / "calibration_metrics.csv")
    _write_csv(benchmark_fits, output_dir / "benchmark_fits.csv")
    evaluation_parameters = pd.concat([result_table(result, len(panel.loc[panel["date"] <= split]), panel.loc[panel["date"] <= split, "date"].nunique()) for result in evaluation_estimates.values()], ignore_index=True)
    _write_csv(evaluation_parameters, output_dir / "evaluation_train_parameters.csv")

    maturity = residual_by_maturity(fits)
    residual_ts = residual_time_series(fits)
    acf_table, statistical_tests = residual_tests(residual_ts, int(config["diagnostics"].get("acf_lags", 20)))
    joint_states = states.loc[states["model"] == "exact_corr_joint"]
    innovations, innovation_summary = innovation_diagnostics(joint_states, int(config["diagnostics"].get("innovation_correlation_window", 63)))
    print("Running rolling stability fits...", flush=True)
    rolling = rolling_estimates(panel, config, estimates)
    for frame, filename in (
        (maturity, "residuals_by_maturity.csv"),
        (residual_ts, "residual_time_series.csv"),
        (acf_table, "residual_acf.csv"),
        (statistical_tests, "residual_statistical_tests.csv"),
        (innovations, "standardized_innovations.csv"),
        (innovation_summary, "innovation_correlation_summary.csv"),
        (rolling, "rolling_parameters.csv"),
    ):
        _write_csv(frame, output_dir / filename)

    plot_latest_curves(fits, chart_dir / "latest_carry_curves.png")
    plot_latest_futures(fits, chart_dir / "latest_futures_prices.png")
    dates = pd.Series(pd.to_datetime(panel["date"].unique())).sort_values().reset_index(drop=True)
    representative_count = int(config["diagnostics"].get("representative_dates", 4))
    positions = np.unique(np.linspace(0, len(dates) - 1, min(representative_count, len(dates)), dtype=int))
    representative_dates = [dates.iloc[position] for position in positions]
    plot_selected_curves(fits, representative_dates, chart_dir / "representative_carry_curves.png", "Representative observed and fitted curves")
    joint_daily = residual_ts.loc[residual_ts["model"] == "exact_corr_joint"]
    worst_dates = joint_daily.nlargest(int(config["diagnostics"].get("worst_dates", 4)), "carry_rmse")["date"].tolist()
    plot_selected_curves(fits, worst_dates, chart_dir / "worst_fit_carry_curves.png", "Worst exact-joint fit dates")
    plot_states(states.loc[states["model"].isin(["legacy_curve", "exact_corr_curve", "exact_corr_joint"])], chart_dir / "filtered_and_smoothed_states.png")
    plot_profiles(profiles, chart_dir / "rho_profile_likelihood.png")
    plot_innovations(innovations, chart_dir / "standardized_innovations.png")
    plot_residuals(residual_ts, chart_dir / "residual_time_series.png")
    plot_acf(acf_table, chart_dir / "residual_acf.png")
    plot_rolling(rolling, chart_dir / "rolling_parameters.png")

    headline = {}
    for name, result in estimates.items():
        headline[name] = {
            "parameters": asdict(result.params),
            "fixed_sigma": result.sigma,
            "log_likelihood": result.log_likelihood,
            "curve_log_likelihood": result.curve_log_likelihood,
            "return_log_likelihood": result.return_log_likelihood,
            "converged": result.converged,
            "hessian_stable": result.hessian_stable,
            "half_life_sessions": float(244 * np.log(2) / result.params.kappa),
        }
    summary = {
        "data_start": str(pd.Timestamp(panel["date"].min()).date()),
        "data_end": str(pd.Timestamp(panel["date"].max()).date()),
        "accepted_observations": int(len(panel)),
        "curve_dates": int(panel["date"].nunique()),
        "excluded_observations": int(audit["excluded"].sum()),
        "fixed_stock_volatility": sigma,
        "time_convention": "trading sessions / 244",
        "evaluation_split_date": str(split.date()),
        "models": headline,
        "profile_confidence_intervals": profile_summaries,
        "likelihood_ratio_tests": lr.to_dict(orient="records"),
        "output_dir": str(output_dir),
        "chart_dir": str(chart_dir),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
