"""Expanding-window, multiple-cut-date tests for observation-noise models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


STUDY_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = STUDY_ROOT.parent
DEMO_ROOT = WORKSPACE_ROOT / "Demo"
CALIBRATION_SRC = WORKSPACE_ROOT / "im_2factor_ou_carry" / "src"
for path in (STUDY_ROOT, DEMO_ROOT, CALIBRATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibration import _gap_function, load_calibration_sample  # noqa: E402
from noise_models import MODEL_LABELS, ModelParams, NoiseModelName, filter_panel  # noqa: E402
from study import (  # noqa: E402
    StudySettings,
    _jsonable,
    _ordinary_error_metrics,
    attach_predictions,
    fit_or_load,
    oos_metrics_by_maturity,
)


COMPARISON_MODELS: tuple[NoiseModelName, ...] = (
    "constant_carry",
    "two_bucket_carry",
    "constant_log_futures",
)


@dataclass
class MultiCutSettings:
    evaluation_date: str = "2026-08-21"
    sample_dates: int = 991
    test_dates_per_window: int = 120
    windows: int = 5
    minimum_training_dates: int = 300
    kappa_gap_upper_bound: float = 120.0
    eta_fast_upper_bound: float = 6.0
    optimizer_starts: int = 12
    optimizer_maxiter: int = 1500
    random_seed: int = 852
    resume: bool = True


@dataclass(frozen=True)
class CutWindow:
    window: int
    training_dates: int
    cut_date: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_nonoverlapping_cut_windows(
    dates: pd.Index,
    *,
    windows: int,
    test_dates_per_window: int,
    minimum_training_dates: int,
) -> list[CutWindow]:
    """Use consecutive equal-length holdouts ending at the final sample date."""
    ordered = pd.Index(pd.to_datetime(dates)).drop_duplicates().sort_values()
    if windows < 2:
        raise ValueError("At least two cut windows are required")
    if test_dates_per_window < 1:
        raise ValueError("test_dates_per_window must be positive")
    first_training_count = len(ordered) - windows * test_dates_per_window
    if first_training_count < minimum_training_dates:
        raise ValueError(
            "Not enough dates for the requested windows, horizon, and minimum training sample"
        )

    result: list[CutWindow] = []
    for number in range(windows):
        training_count = first_training_count + number * test_dates_per_window
        test_start_index = training_count
        test_end_index = test_start_index + test_dates_per_window - 1
        result.append(
            CutWindow(
                window=number + 1,
                training_dates=training_count,
                cut_date=pd.Timestamp(ordered[training_count - 1]),
                test_start=pd.Timestamp(ordered[test_start_index]),
                test_end=pd.Timestamp(ordered[test_end_index]),
            )
        )
    return result


def _window_metrics(
    predictions: pd.DataFrame,
    likelihoods: pd.DataFrame,
    window: CutWindow,
    model: NoiseModelName,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "window": window.window,
        "model": model,
        "model_label": MODEL_LABELS[model],
        "training_dates": window.training_dates,
        "cut_date": window.cut_date,
        "test_start": window.test_start,
        "test_end": window.test_end,
    }
    metrics.update(_ordinary_error_metrics(predictions))
    score = float(likelihoods["comparison_log_likelihood"].sum())
    metrics["comparison_log_predictive_score"] = score
    metrics["mean_log_predictive_score"] = score / len(predictions)
    return metrics


def _daily_metrics(
    predictions: pd.DataFrame,
    likelihoods: pd.DataFrame,
    window: CutWindow,
    model: NoiseModelName,
) -> pd.DataFrame:
    work = predictions.copy()
    work["carry_squared_error"] = work["carry_prediction_error"] ** 2
    work["carry_absolute_error"] = work["carry_prediction_error"].abs()
    work["futures_squared_error"] = work["futures_prediction_error"] ** 2
    work["futures_absolute_error"] = work["futures_prediction_error"].abs()
    daily = (
        work.groupby("date", as_index=False)
        .agg(
            n_observations=("row_index", "size"),
            carry_squared_error=("carry_squared_error", "mean"),
            carry_absolute_error=("carry_absolute_error", "mean"),
            futures_squared_error=("futures_squared_error", "mean"),
            futures_absolute_error=("futures_absolute_error", "mean"),
        )
        .merge(
            likelihoods[["date", "comparison_log_likelihood"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
    )
    daily["mean_log_predictive_score"] = (
        daily["comparison_log_likelihood"] / daily["n_observations"]
    )
    daily.insert(0, "model_label", MODEL_LABELS[model])
    daily.insert(0, "model", model)
    daily.insert(0, "window", window.window)
    return daily


def newey_west_mean_test(values: np.ndarray, lag: int | None = None) -> dict[str, float | int]:
    """Return a two-sided HAC test of whether the series mean equals zero."""
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    count = len(series)
    if count < 3:
        return {
            "n_dates": count,
            "hac_lag": 0,
            "mean_gain": np.nan,
            "hac_standard_error": np.nan,
            "z_statistic": np.nan,
            "two_sided_p_value": np.nan,
        }
    if lag is None:
        lag = int(np.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(int(lag), count - 1))
    mean = float(series.mean())
    centered = series - mean
    long_run_variance = float(centered @ centered) / count
    for offset in range(1, lag + 1):
        covariance = float(centered[offset:] @ centered[:-offset]) / count
        long_run_variance += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    variance_of_mean = max(long_run_variance, 0.0) / count
    standard_error = float(np.sqrt(variance_of_mean))
    z_statistic = mean / standard_error if standard_error > 0.0 else np.nan
    p_value = 2.0 * norm.sf(abs(z_statistic)) if np.isfinite(z_statistic) else np.nan
    return {
        "n_dates": count,
        "hac_lag": lag,
        "mean_gain": mean,
        "hac_standard_error": standard_error,
        "z_statistic": float(z_statistic),
        "two_sided_p_value": float(p_value),
    }


def paired_daily_comparisons(daily: pd.DataFrame) -> pd.DataFrame:
    """Compare model pairs; positive gain always favors the candidate."""
    metrics = (
        "carry_squared_error",
        "carry_absolute_error",
        "futures_squared_error",
        "futures_absolute_error",
        "mean_log_predictive_score",
    )
    rows: list[dict[str, Any]] = []
    comparisons: tuple[tuple[NoiseModelName, NoiseModelName], ...] = (
        ("constant_carry", "two_bucket_carry"),
        ("constant_carry", "constant_log_futures"),
        ("two_bucket_carry", "constant_log_futures"),
    )
    for reference_model, candidate_model in comparisons:
        reference = daily.loc[
            daily["model"] == reference_model, ["date", *metrics]
        ].copy()
        candidate = daily.loc[
            daily["model"] == candidate_model, ["date", *metrics]
        ].copy()
        paired = reference.merge(
            candidate,
            on="date",
            suffixes=("_reference", "_candidate"),
            validate="one_to_one",
        )
        for metric in metrics:
            if metric == "mean_log_predictive_score":
                gain = (
                    paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
                ).to_numpy()
            else:
                gain = (
                    paired[f"{metric}_reference"] - paired[f"{metric}_candidate"]
                ).to_numpy()
            result = newey_west_mean_test(gain)
            rows.append(
                {
                    "reference_model": reference_model,
                    "reference_model_label": MODEL_LABELS[reference_model],
                    "candidate_model": candidate_model,
                    "candidate_model_label": MODEL_LABELS[candidate_model],
                    "metric": metric,
                    **result,
                    "median_gain": float(np.median(gain)),
                    "candidate_win_dates": int(np.sum(gain > 0.0)),
                    "candidate_win_rate": float(np.mean(gain > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def summarize_models(
    window_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    carry_winners = window_metrics.loc[
        window_metrics.groupby("window")["carry_rmse_bps"].idxmin(), "model"
    ].value_counts()
    futures_winners = window_metrics.loc[
        window_metrics.groupby("window")["futures_rmse_points"].idxmin(), "model"
    ].value_counts()
    score_winners = window_metrics.loc[
        window_metrics.groupby("window")["mean_log_predictive_score"].idxmax(), "model"
    ].value_counts()

    for model in COMPARISON_MODELS:
        model_predictions = predictions.loc[predictions["model"] == model]
        model_daily = daily.loc[daily["model"] == model]
        model_windows = window_metrics.loc[window_metrics["model"] == model]
        row: dict[str, Any] = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "windows": len(model_windows),
            "test_dates": model_daily["date"].nunique(),
            "carry_rmse_window_wins": int(carry_winners.get(model, 0)),
            "futures_rmse_window_wins": int(futures_winners.get(model, 0)),
            "log_score_window_wins": int(score_winners.get(model, 0)),
            "mean_kappa_gap": float(model_windows["kappa_gap"].mean()),
            "median_kappa_gap": float(model_windows["kappa_gap"].median()),
            "min_kappa_gap": float(model_windows["kappa_gap"].min()),
            "max_kappa_gap": float(model_windows["kappa_gap"].max()),
            "gap_upper_bound_hits": int(model_windows["gap_at_upper_bound"].sum()),
            "eta_fast_upper_bound_hits": int(model_windows["eta_fast_at_upper_bound"].sum()),
        }
        row.update(_ordinary_error_metrics(model_predictions))
        score = float(model_daily["comparison_log_likelihood"].sum())
        row["comparison_log_predictive_score"] = score
        row["mean_log_predictive_score"] = score / len(model_predictions)
        rows.append(row)

    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"] == "constant_carry"].iloc[0]
    for metric in (
        "carry_rmse_bps",
        "carry_mae_bps",
        "futures_rmse_points",
        "futures_mae_points",
    ):
        result[f"{metric}_improvement_vs_constant_pct"] = (
            1.0 - result[metric] / baseline[metric]
        ) * 100.0
    result["mean_log_score_gain_vs_constant"] = (
        result["mean_log_predictive_score"] - baseline["mean_log_predictive_score"]
    )
    return result


def make_charts(
    window_metrics: pd.DataFrame,
    model_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(COMPARISON_MODELS, plt.cm.tab10(np.linspace(0.0, 0.5, 3)), strict=True))

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for model in COMPARISON_MODELS:
        data = window_metrics.loc[window_metrics["model"] == model]
        label = MODEL_LABELS[model]
        axes[0].plot(
            data["test_start"],
            data["carry_rmse_bps"],
            "o-",
            label=label,
            color=colors[model],
        )
        axes[1].plot(
            data["test_start"],
            data["futures_rmse_points"],
            "o-",
            label=label,
            color=colors[model],
        )
        axes[2].plot(
            data["test_start"],
            data["mean_log_predictive_score"],
            "o-",
            label=label,
            color=colors[model],
        )
    axes[0].set_ylabel("Carry RMSE (bp)")
    axes[1].set_ylabel("Futures RMSE (points)")
    axes[2].set_ylabel("Mean log predictive score")
    for axis in axes:
        axis.set_xlabel("Test-window start")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.suptitle("Multiple-cut-date out-of-sample performance")
    figure.tight_layout()
    figure.savefig(output_dir / "window_performance.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for model in COMPARISON_MODELS:
        data = window_metrics.loc[window_metrics["model"] == model]
        axes[0].plot(
            data["cut_date"],
            data["kappa_gap"],
            "o-",
            label=MODEL_LABELS[model],
            color=colors[model],
        )
        axes[1].plot(
            data["cut_date"],
            data["eta_fast"],
            "o-",
            label=MODEL_LABELS[model],
            color=colors[model],
        )
    axes[0].set_ylabel("Estimated kappa gap")
    axes[1].set_ylabel("Estimated eta-fast")
    for axis in axes:
        axis.set_xlabel("Calibration cut date")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Expanding-window parameter stability")
    figure.tight_layout()
    figure.savefig(output_dir / "parameter_stability.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    positions = np.arange(len(model_summary))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(positions, model_summary["carry_rmse_bps"], color=list(colors.values()))
    axes[0].set_ylabel("Pooled carry RMSE (bp)")
    axes[1].bar(
        positions,
        model_summary["futures_rmse_points"],
        color=list(colors.values()),
    )
    axes[1].set_ylabel("Pooled futures RMSE (points)")
    axes[2].bar(
        positions,
        model_summary["mean_log_predictive_score"],
        color=list(colors.values()),
    )
    axes[2].set_ylabel("Pooled mean log score")
    for axis in axes:
        axis.set_xticks(positions, model_summary["model_label"], rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Pooled performance across non-overlapping holdouts")
    figure.tight_layout()
    figure.savefig(output_dir / "pooled_performance.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_multi_cut_study(settings: MultiCutSettings, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = output_dir / "fits"
    sample, _, _ = load_calibration_sample(
        evaluation_date=settings.evaluation_date,
        window_dates=settings.sample_dates,
    )
    dates = pd.Index(pd.to_datetime(sample["date"].unique())).sort_values()
    windows = make_nonoverlapping_cut_windows(
        dates,
        windows=settings.windows,
        test_dates_per_window=settings.test_dates_per_window,
        minimum_training_dates=settings.minimum_training_dates,
    )
    fit_settings = StudySettings(
        evaluation_date=settings.evaluation_date,
        sample_dates=settings.sample_dates,
        kappa_gap_upper_bound=settings.kappa_gap_upper_bound,
        eta_fast_upper_bound=settings.eta_fast_upper_bound,
        optimizer_starts=settings.optimizer_starts,
        optimizer_maxiter=settings.optimizer_maxiter,
        random_seed=settings.random_seed,
        price_options=False,
        resume=settings.resume,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    maturity_tables: list[pd.DataFrame] = []
    daily_tables: list[pd.DataFrame] = []
    prior_fits: dict[NoiseModelName, ModelParams] = {}

    for window in windows:
        training = sample.loc[sample["date"] <= window.cut_date].copy()
        evaluation = sample.loc[sample["date"] <= window.test_end].copy()
        print(
            f"[window {window.window}/{len(windows)}] cut={window.cut_date.date()}, "
            f"test={window.test_start.date()}..{window.test_end.date()}, "
            f"train={training['date'].nunique()} dates/{len(training)} rows",
            flush=True,
        )
        current_fits: dict[NoiseModelName, ModelParams] = {}
        for model in COMPARISON_MODELS:
            warm = prior_fits.get(model)
            if warm is None and model != "constant_carry":
                warm = current_fits.get("constant_carry")
            stage = f"window_{window.window:02d}_cut_{window.cut_date:%Y%m%d}"
            artifact = fit_or_load(
                training,
                model,
                stage,
                fit_settings,
                fit_dir,
                warm_params=warm,
            )
            current_fits[model] = artifact.params

            filtered = filter_panel(evaluation, artifact.params, gap_function=_gap_function)
            attached = attach_predictions(evaluation, filtered.predictions)
            test_predictions = attached.loc[
                (attached["date"] >= window.test_start)
                & (attached["date"] <= window.test_end)
            ].copy()
            test_likelihoods = filtered.date_likelihoods.loc[
                (filtered.date_likelihoods["date"] >= window.test_start)
                & (filtered.date_likelihoods["date"] <= window.test_end)
            ].copy()
            test_predictions["window"] = window.window
            test_predictions["model"] = model
            test_predictions["model_label"] = MODEL_LABELS[model]
            prediction_tables.append(test_predictions)

            metrics = _window_metrics(test_predictions, test_likelihoods, window, model)
            params = artifact.params
            gap = params.ou.kappa_fast - params.ou.kappa_slow
            metrics.update(params.as_dict())
            metrics["kappa_gap"] = gap
            metrics["gap_at_upper_bound"] = bool(
                gap >= settings.kappa_gap_upper_bound - 1e-6
            )
            metrics["eta_fast_at_upper_bound"] = bool(
                params.ou.eta_fast >= settings.eta_fast_upper_bound - 1e-6
            )
            metrics["optimizer_converged"] = artifact.converged
            metric_rows.append(metrics)

            maturity = oos_metrics_by_maturity(attached, window.cut_date, model)
            maturity.insert(0, "test_end", window.test_end)
            maturity.insert(0, "test_start", window.test_start)
            maturity.insert(0, "cut_date", window.cut_date)
            maturity.insert(0, "window", window.window)
            maturity_tables.append(maturity)
            daily_tables.append(
                _daily_metrics(test_predictions, test_likelihoods, window, model)
            )
        prior_fits.update(current_fits)

    window_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    maturity = pd.concat(maturity_tables, ignore_index=True)
    daily = pd.concat(daily_tables, ignore_index=True)
    model_summary = summarize_models(window_metrics, predictions, daily)
    paired = paired_daily_comparisons(daily)

    window_metrics.to_csv(output_dir / "window_metrics.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_daily_comparisons.csv", index=False)
    maturity.to_csv(output_dir / "metrics_by_maturity.csv", index=False)
    daily.to_csv(output_dir / "daily_metrics.csv", index=False)
    predictions.to_csv(output_dir / "oos_predictions.csv", index=False)
    make_charts(window_metrics, model_summary, output_dir / "charts")

    candidate = model_summary.loc[
        model_summary["model"] == "constant_log_futures"
    ].iloc[0]
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "settings": asdict(settings),
        "sample_start": str(pd.Timestamp(sample["date"].min()).date()),
        "sample_end": str(pd.Timestamp(sample["date"].max()).date()),
        "windows": [asdict(window) for window in windows],
        "candidate_model": "constant_log_futures",
        "candidate_carry_rmse_improvement_vs_constant_pct": candidate[
            "carry_rmse_bps_improvement_vs_constant_pct"
        ],
        "candidate_futures_rmse_improvement_vs_constant_pct": candidate[
            "futures_rmse_points_improvement_vs_constant_pct"
        ],
        "candidate_mean_log_score_gain_vs_constant": candidate[
            "mean_log_score_gain_vs_constant"
        ],
        "candidate_gap_upper_bound_hits": candidate["gap_upper_bound_hits"],
        "candidate_eta_fast_upper_bound_hits": candidate["eta_fast_upper_bound_hits"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-date", default="2026-08-21")
    parser.add_argument("--sample-dates", type=int, default=991)
    parser.add_argument("--test-dates-per-window", type=int, default=120)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--minimum-training-dates", type=int, default=300)
    parser.add_argument("--kappa-gap-upper-bound", type=float, default=120.0)
    parser.add_argument("--eta-fast-upper-bound", type=float, default=6.0)
    parser.add_argument("--optimizer-starts", type=int, default=12)
    parser.add_argument("--optimizer-maxiter", type=int, default=1500)
    parser.add_argument("--random-seed", type=int, default=852)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STUDY_ROOT / "outputs" / "multi_cut",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = MultiCutSettings(
        evaluation_date=str(pd.Timestamp(args.evaluation_date).date()),
        sample_dates=args.sample_dates,
        test_dates_per_window=args.test_dates_per_window,
        windows=args.windows,
        minimum_training_dates=args.minimum_training_dates,
        kappa_gap_upper_bound=args.kappa_gap_upper_bound,
        eta_fast_upper_bound=args.eta_fast_upper_bound,
        optimizer_starts=args.optimizer_starts,
        optimizer_maxiter=args.optimizer_maxiter,
        random_seed=args.random_seed,
        resume=not args.no_resume,
    )
    run_multi_cut_study(settings, args.output_dir.resolve())


if __name__ == "__main__":
    main()
