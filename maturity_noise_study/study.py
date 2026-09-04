"""Compare maturity-dependent observation-noise models on a common holdout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

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
from estimation import EstimationResult, estimate_model  # noqa: E402
from noise_models import (  # noqa: E402
    MODEL_LABELS,
    MODEL_NAMES,
    ModelParams,
    NoiseModelName,
    OUParams,
    carry_noise_sd,
    filter_panel,
    noise_parameter_names,
    parameter_count,
)


@dataclass
class StudySettings:
    evaluation_date: str = "2026-08-21"
    sample_dates: int = 991
    test_fraction: float = 0.20
    contract: str = "IM2612"
    kappa_gap_upper_bound: float = 120.0
    eta_fast_upper_bound: float = 6.0
    optimizer_starts: int = 12
    optimizer_maxiter: int = 1500
    random_seed: int = 852
    price_options: bool = True
    resume: bool = True


@dataclass
class FitArtifact:
    params: ModelParams
    comparison_log_likelihood: float
    converged: bool
    message: str


def common_split_date(panel: pd.DataFrame, test_fraction: float) -> pd.Timestamp:
    dates = pd.Index(pd.to_datetime(panel["date"].unique())).sort_values()
    if len(dates) < 20:
        raise ValueError("At least 20 curve dates are required")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between zero and one")
    test_count = max(1, int(np.ceil(len(dates) * test_fraction)))
    return pd.Timestamp(dates[-test_count - 1])


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


def _params_from_dict(model: NoiseModelName, payload: dict[str, Any]) -> ModelParams:
    return ModelParams(
        model=model,
        ou=OUParams(
            kappa_slow=float(payload["kappa_slow"]),
            kappa_fast=float(payload["kappa_fast"]),
            theta=float(payload["theta"]),
            eta_slow=float(payload["eta_slow"]),
            eta_fast=float(payload["eta_fast"]),
        ),
        noise=tuple(float(payload[name]) for name in noise_parameter_names(model)),
    )


def _fit_metadata(
    panel: pd.DataFrame,
    model: NoiseModelName,
    stage: str,
    settings: StudySettings,
) -> dict[str, Any]:
    return {
        "model": model,
        "stage": stage,
        "sample_start": str(pd.Timestamp(panel["date"].min()).date()),
        "sample_end": str(pd.Timestamp(panel["date"].max()).date()),
        "sample_dates": int(panel["date"].nunique()),
        "sample_observations": len(panel),
        "kappa_gap_upper_bound": settings.kappa_gap_upper_bound,
        "eta_fast_upper_bound": settings.eta_fast_upper_bound,
        "optimizer_starts": settings.optimizer_starts,
        "optimizer_maxiter": settings.optimizer_maxiter,
        "random_seed": settings.random_seed,
    }


def fit_or_load(
    panel: pd.DataFrame,
    model: NoiseModelName,
    stage: str,
    settings: StudySettings,
    fit_dir: Path,
    *,
    warm_params: ModelParams | None,
) -> FitArtifact:
    """Estimate and checkpoint one train/full model fit."""
    fit_dir.mkdir(parents=True, exist_ok=True)
    json_path = fit_dir / f"{model}_{stage}.json"
    expected = _fit_metadata(panel, model, stage, settings)
    if settings.resume and json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if all(payload.get(key) == value for key, value in expected.items()):
            print(f"[cache] {model} {stage}", flush=True)
            return FitArtifact(
                params=_params_from_dict(model, payload["parameters"]),
                comparison_log_likelihood=float(payload["comparison_log_likelihood"]),
                converged=bool(payload["converged"]),
                message=str(payload["message"]),
            )

    print(f"[fit] {model} {stage}", flush=True)
    started = time.perf_counter()
    estimate: EstimationResult = estimate_model(
        panel,
        model,
        gap_function=_gap_function,
        starts=settings.optimizer_starts,
        maxiter=settings.optimizer_maxiter,
        seed=settings.random_seed,
        kappa_gap_upper_bound=settings.kappa_gap_upper_bound,
        eta_fast_upper_bound=settings.eta_fast_upper_bound,
        warm_params=warm_params,
    )
    estimate.optimizer_runs.to_csv(fit_dir / f"{model}_{stage}_optimizer_runs.csv", index=False)
    payload = {
        **expected,
        "parameters": estimate.params.as_dict(),
        "comparison_log_likelihood": estimate.comparison_log_likelihood,
        "converged": estimate.converged,
        "message": estimate.message,
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[done] {model} {stage}: gap="
        f"{estimate.params.ou.kappa_fast - estimate.params.ou.kappa_slow:.4f}, "
        f"ll={estimate.comparison_log_likelihood:.3f}, "
        f"elapsed={payload['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return FitArtifact(
        params=estimate.params,
        comparison_log_likelihood=estimate.comparison_log_likelihood,
        converged=estimate.converged,
        message=estimate.message,
    )


def attach_predictions(panel: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    base = panel.reset_index().rename(columns={"index": "row_index"})
    extra = predictions.drop(columns="date")
    return base.merge(extra, on="row_index", how="left", validate="one_to_one")


def _ordinary_error_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    carry = frame["carry_prediction_error"].to_numpy(dtype=float)
    futures = frame["futures_prediction_error"].to_numpy(dtype=float)
    return {
        "n_observations": len(frame),
        "carry_rmse_bps": float(np.sqrt(np.mean(carry**2)) * 10_000),
        "carry_mae_bps": float(np.mean(np.abs(carry)) * 10_000),
        "carry_mean_bias_bps": float(np.mean(carry) * 10_000),
        "futures_rmse_points": float(np.sqrt(np.mean(futures**2))),
        "futures_mae_points": float(np.mean(np.abs(futures))),
        "futures_mean_bias_points": float(np.mean(futures)),
        "marginal_standardized_innovation_rms": float(
            np.sqrt(np.mean(frame["marginal_standardized_innovation"] ** 2))
        ),
    }


def oos_metrics(
    predictions: pd.DataFrame,
    date_likelihoods: pd.DataFrame,
    split_date: pd.Timestamp,
    model: NoiseModelName,
) -> dict[str, Any]:
    test = predictions.loc[predictions["date"] > split_date]
    test_likelihoods = date_likelihoods.loc[date_likelihoods["date"] > split_date]
    result: dict[str, Any] = {"model": model, "model_label": MODEL_LABELS[model]}
    result.update(_ordinary_error_metrics(test))
    total_score = float(test_likelihoods["comparison_log_likelihood"].sum())
    result["comparison_log_predictive_score"] = total_score
    result["mean_log_predictive_score"] = total_score / len(test)
    return result


MATURITY_BINS = [5, 10, 15, 21, 42, 63, 126, 252, np.inf]
MATURITY_LABELS = ["6-10", "11-15", "16-21", "22-42", "43-63", "64-126", "127-252", "253+"]


def oos_metrics_by_maturity(
    predictions: pd.DataFrame,
    split_date: pd.Timestamp,
    model: NoiseModelName,
) -> pd.DataFrame:
    test = predictions.loc[predictions["date"] > split_date].copy()
    test["maturity_bucket"] = pd.cut(
        test["sessions_to_expiry"],
        bins=MATURITY_BINS,
        labels=MATURITY_LABELS,
        right=True,
    )
    rows = []
    for bucket, group in test.groupby("maturity_bucket", observed=True, sort=True):
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "maturity_bucket": str(bucket),
                **_ordinary_error_metrics(group),
                "mean_observation_noise_carry_bps": float(
                    group["observation_noise_carry_sd"].mean() * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def price_model(
    params: ModelParams,
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
    ou = TwoFactorOUParams(
        kappa_slow=params.ou.kappa_slow,
        kappa_fast=params.ou.kappa_fast,
        theta=params.ou.theta,
        eta_slow=params.ou.eta_slow,
        eta_fast=params.ou.eta_fast,
    )
    state = FactorState(
        slow=float(latest_state["filtered_slow_state"]),
        fast=float(latest_state["filtered_fast_state"]),
    )
    result = price_american_carry_put(
        contract,
        ou,
        state,
        GBMParams(RISK_FREE_RATE, historical_volatility),
        numerical=NumericalConfig(),
    )
    return {
        "option_price": result.price,
        "model_initial_futures": result.model_initial_futures,
        "initial_futures_model_error": result.initial_futures_model_error,
        "slow_delta_pathwise": result.slow_curve_delta.pathwise_delta,
        "fast_delta_pathwise": result.fast_curve_delta.pathwise_delta,
    }


def full_sample_row(
    model: NoiseModelName,
    artifact: FitArtifact,
    sample: pd.DataFrame,
    full_filter,
    quote: pd.Series,
    historical_volatility: float,
    settings: StudySettings,
) -> dict[str, Any]:
    params = artifact.params
    count = parameter_count(model)
    gap = params.ou.kappa_fast - params.ou.kappa_slow
    row: dict[str, Any] = {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "parameter_count": count,
        **params.as_dict(),
        "kappa_gap": gap,
        "gap_at_upper_bound": bool(gap >= settings.kappa_gap_upper_bound - 1e-6),
        "eta_fast_at_upper_bound": bool(params.ou.eta_fast >= settings.eta_fast_upper_bound - 1e-6),
        "comparison_log_likelihood": artifact.comparison_log_likelihood,
        "aic": 2.0 * count - 2.0 * artifact.comparison_log_likelihood,
        "bic": count * np.log(len(sample)) - 2.0 * artifact.comparison_log_likelihood,
        "optimizer_converged": artifact.converged,
    }
    if settings.price_options:
        row.update(
            price_model(
                params,
                full_filter.states.iloc[-1],
                quote,
                historical_volatility,
            )
        )
    return row


def noise_curve_table(
    artifacts: dict[NoiseModelName, FitArtifact],
) -> pd.DataFrame:
    sessions = np.arange(6, 253, dtype=float)
    tau = sessions / PERIODS_PER_YEAR
    rows = []
    representative_futures = 7500.0
    for model, artifact in artifacts.items():
        sigma_carry = carry_noise_sd(model, artifact.params.noise, tau, sessions)
        approximate_points = representative_futures * tau * sigma_carry
        rows.extend(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "sessions_to_expiry": int(session),
                "observation_noise_carry_bps": carry_sd * 10_000,
                "approximate_futures_noise_points_at_7500": point_sd,
            }
            for session, carry_sd, point_sd in zip(
                sessions, sigma_carry, approximate_points, strict=True
            )
        )
    return pd.DataFrame(rows)


def make_charts(
    aggregate: pd.DataFrame,
    maturity: pd.DataFrame,
    full: pd.DataFrame,
    noise_curves: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0.0, 0.7, len(MODEL_NAMES)))

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    positions = np.arange(len(aggregate))
    axes[0].bar(positions, aggregate["carry_rmse_bps"], color=colors)
    axes[0].set_ylabel("OOS carry RMSE (bp)")
    axes[1].bar(positions, aggregate["futures_rmse_points"], color=colors)
    axes[1].set_ylabel("OOS futures RMSE (points)")
    axes[2].bar(positions, aggregate["mean_log_predictive_score"], color=colors)
    axes[2].set_ylabel("Mean OOS log predictive score")
    for axis in axes:
        axis.set_xticks(positions, aggregate["model_label"], rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Common-period out-of-sample performance")
    figure.tight_layout()
    figure.savefig(output_dir / "oos_aggregate_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for model in MODEL_NAMES:
        data = maturity.loc[maturity["model"] == model]
        axes[0].plot(
            data["maturity_bucket"],
            data["carry_rmse_bps"],
            "o-",
            label=MODEL_LABELS[model],
        )
        axes[1].plot(
            data["maturity_bucket"],
            data["futures_rmse_points"],
            "o-",
            label=MODEL_LABELS[model],
        )
    axes[0].set_ylabel("OOS carry RMSE (bp)")
    axes[1].set_ylabel("OOS futures RMSE (points)")
    for axis in axes:
        axis.set_xlabel("Sessions-to-expiry bucket")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Out-of-sample errors by maturity")
    figure.tight_layout()
    figure.savefig(output_dir / "oos_by_maturity.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in MODEL_NAMES:
        data = noise_curves.loc[noise_curves["model"] == model]
        axes[0].plot(
            data["sessions_to_expiry"],
            data["observation_noise_carry_bps"],
            label=MODEL_LABELS[model],
        )
        axes[1].plot(
            data["sessions_to_expiry"],
            data["approximate_futures_noise_points_at_7500"],
            label=MODEL_LABELS[model],
        )
    axes[0].set(xlabel="Sessions to expiry", ylabel="Carry noise SD (bp)")
    axes[1].set(
        xlabel="Sessions to expiry",
        ylabel="Approximate futures noise SD at F=7500",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Full-sample fitted observation noise")
    figure.tight_layout()
    figure.savefig(output_dir / "fitted_noise_curves.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    if "option_price" in full:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].bar(positions, full["kappa_gap"], color=colors)
        axes[0].set_ylabel("Full-sample kappa gap")
        axes[1].bar(positions, full["eta_fast"], color=colors)
        axes[1].set_ylabel("Full-sample eta-fast")
        axes[2].bar(positions, full["option_price"], color=colors)
        axes[2].set_ylabel("IM2612 optional-component price")
        for axis in axes:
            axis.set_xticks(positions, full["model_label"], rotation=25, ha="right")
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle("Economic parameter and valuation sensitivity")
        figure.tight_layout()
        figure.savefig(
            output_dir / "parameter_and_price_comparison.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)


def run_study(settings: StudySettings, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = output_dir / "fits"
    sample, _, spot_history = load_calibration_sample(
        evaluation_date=settings.evaluation_date,
        window_dates=settings.sample_dates,
    )
    split_date = common_split_date(sample, settings.test_fraction)
    training = sample.loc[sample["date"] <= split_date].copy()
    test = sample.loc[sample["date"] > split_date].copy()
    print(
        f"Common split: {split_date.date()}, train={training['date'].nunique()} dates/"
        f"{len(training)} rows, test={test['date'].nunique()} dates/{len(test)} rows",
        flush=True,
    )

    train_artifacts: dict[NoiseModelName, FitArtifact] = {}
    full_artifacts: dict[NoiseModelName, FitArtifact] = {}
    aggregate_rows: list[dict[str, Any]] = []
    maturity_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    full_rows: list[dict[str, Any]] = []
    historical_volatility = estimate_historical_volatility(spot_history)
    quote_rows = sample.loc[
        (sample["date"] == pd.Timestamp(settings.evaluation_date))
        & (sample["contract"] == settings.contract.upper())
    ]
    if len(quote_rows) != 1:
        raise ValueError(f"Expected one {settings.contract} quote, found {len(quote_rows)}")
    quote = quote_rows.iloc[0]

    for model in MODEL_NAMES:
        train_warm = train_artifacts.get("constant_carry")
        full_warm = full_artifacts.get("constant_carry")
        train_artifact = fit_or_load(
            training,
            model,
            "train",
            settings,
            fit_dir,
            warm_params=None if train_warm is None else train_warm.params,
        )
        full_artifact = fit_or_load(
            sample,
            model,
            "full",
            settings,
            fit_dir,
            warm_params=None if full_warm is None else full_warm.params,
        )
        train_artifacts[model] = train_artifact
        full_artifacts[model] = full_artifact

        oos_filter = filter_panel(sample, train_artifact.params, gap_function=_gap_function)
        predictions = attach_predictions(sample, oos_filter.predictions)
        predictions["model"] = model
        predictions["model_label"] = MODEL_LABELS[model]
        prediction_tables.append(predictions.loc[predictions["date"] > split_date])
        aggregate_rows.append(
            oos_metrics(predictions, oos_filter.date_likelihoods, split_date, model)
        )
        maturity_tables.append(oos_metrics_by_maturity(predictions, split_date, model))

        full_filter = filter_panel(sample, full_artifact.params, gap_function=_gap_function)
        full_rows.append(
            full_sample_row(
                model,
                full_artifact,
                sample,
                full_filter,
                quote,
                historical_volatility,
                settings,
            )
        )

    aggregate = pd.DataFrame(aggregate_rows)
    maturity = pd.concat(maturity_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    full = pd.DataFrame(full_rows)
    noise_curves = noise_curve_table(full_artifacts)
    aggregate.to_csv(output_dir / "oos_aggregate_metrics.csv", index=False)
    maturity.to_csv(output_dir / "oos_metrics_by_maturity.csv", index=False)
    predictions.to_csv(output_dir / "oos_predictions.csv", index=False)
    full.to_csv(output_dir / "full_sample_model_comparison.csv", index=False)
    noise_curves.to_csv(output_dir / "fitted_noise_curves.csv", index=False)
    make_charts(aggregate, maturity, full, noise_curves, output_dir / "charts")

    best_carry = aggregate.loc[aggregate["carry_rmse_bps"].idxmin()]
    best_futures = aggregate.loc[aggregate["futures_rmse_points"].idxmin()]
    best_score = aggregate.loc[aggregate["mean_log_predictive_score"].idxmax()]
    best_bic = full.loc[full["bic"].idxmin()]
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "settings": asdict(settings),
        "sample_start": str(pd.Timestamp(sample["date"].min()).date()),
        "sample_end": str(pd.Timestamp(sample["date"].max()).date()),
        "split_date": str(split_date.date()),
        "training_dates": int(training["date"].nunique()),
        "training_observations": len(training),
        "test_dates": int(test["date"].nunique()),
        "test_observations": len(test),
        "best_oos_carry_rmse_model": best_carry["model"],
        "best_oos_futures_rmse_model": best_futures["model"],
        "best_oos_log_score_model": best_score["model"],
        "best_full_sample_bic_model": best_bic["model"],
        "likelihood_comparison_note": (
            "The direct log-futures likelihood includes the exact sum(log(tau)) "
            "Jacobian adjustment and is therefore expressed in carry-observation units."
        ),
    }
    (output_dir / "study_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-date", default="2026-08-21")
    parser.add_argument("--sample-dates", type=int, default=991)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--contract", default="IM2612")
    parser.add_argument("--kappa-gap-upper-bound", type=float, default=120.0)
    parser.add_argument("--eta-fast-upper-bound", type=float, default=6.0)
    parser.add_argument("--optimizer-starts", type=int, default=12)
    parser.add_argument("--optimizer-maxiter", type=int, default=1500)
    parser.add_argument("--random-seed", type=int, default=852)
    parser.add_argument("--skip-pricing", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=STUDY_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = StudySettings(
        evaluation_date=str(pd.Timestamp(args.evaluation_date).date()),
        sample_dates=args.sample_dates,
        test_fraction=args.test_fraction,
        contract=args.contract.strip().upper(),
        kappa_gap_upper_bound=args.kappa_gap_upper_bound,
        eta_fast_upper_bound=args.eta_fast_upper_bound,
        optimizer_starts=args.optimizer_starts,
        optimizer_maxiter=args.optimizer_maxiter,
        random_seed=args.random_seed,
        price_options=not args.skip_pricing,
        resume=not args.no_resume,
    )
    run_study(settings, args.output_dir.resolve())


if __name__ == "__main__":
    main()
