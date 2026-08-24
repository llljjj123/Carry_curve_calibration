"""Model comparisons, forecast metrics, residual tests, and innovation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, pearsonr, spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf


def error_metrics(carry_error: pd.Series, price_error: pd.Series) -> dict[str, float | int]:
    carry = pd.to_numeric(carry_error, errors="coerce").dropna().to_numpy(dtype=float)
    price = pd.to_numeric(price_error, errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "n_carry": len(carry),
        "carry_rmse_percentage_points": float(100 * np.sqrt(np.mean(carry**2))) if len(carry) else np.nan,
        "carry_mae_percentage_points": float(100 * np.mean(np.abs(carry))) if len(carry) else np.nan,
        "carry_rmse_bps": float(1e4 * np.sqrt(np.mean(carry**2))) if len(carry) else np.nan,
        "carry_mae_bps": float(1e4 * np.mean(np.abs(carry))) if len(carry) else np.nan,
        "n_price": len(price),
        "futures_rmse_points": float(np.sqrt(np.mean(price**2))) if len(price) else np.nan,
        "futures_mae_points": float(np.mean(np.abs(price))) if len(price) else np.nan,
        "carry_mean_bias_bps": float(1e4 * np.mean(carry)) if len(carry) else np.nan,
        "futures_mean_bias_points": float(np.mean(price)) if len(price) else np.nan,
    }


def model_metrics(fits: pd.DataFrame, sample: str, error_kind: str) -> pd.DataFrame:
    carry_col = "carry_residual" if error_kind == "filtered" else "carry_prediction_error"
    price_col = "futures_residual" if error_kind == "filtered" else "futures_prediction_error"
    return pd.DataFrame([
        {"model": model, "sample": sample, "error_kind": error_kind, **error_metrics(group[carry_col], group[price_col])}
        for model, group in fits.groupby("model", sort=False)
    ])


def attach_benchmarks(panel: pd.DataFrame, ewma_span: int = 20) -> pd.DataFrame:
    result = panel.copy()
    levels = result.groupby("date", sort=True)["implied_carry"].mean().rename("daily_level").to_frame()
    levels["flat_same_day"] = levels["daily_level"]
    levels["previous_day"] = levels["daily_level"].shift(1)
    levels["ewma"] = levels["daily_level"].ewm(span=ewma_span, adjust=False).mean().shift(1)
    result = result.merge(levels.drop(columns="daily_level"), on="date", how="left")
    for name in ("flat_same_day", "previous_day", "ewma"):
        result[f"{name}_carry_error"] = result["implied_carry"] - result[name]
        price = result["spot"] * np.exp((result["risk_free_rate"] - result[name]) * result["tau"])
        result[f"{name}_futures_error"] = result["futures_price"] - price
    return result


def benchmark_metrics(panel: pd.DataFrame, split_date: object) -> pd.DataFrame:
    rows = []
    split = pd.Timestamp(split_date)
    for name in ("flat_same_day", "previous_day", "ewma"):
        for sample, mask in (("in_sample", panel["date"] <= split), ("out_of_sample", panel["date"] > split)):
            rows.append({"model": name, "sample": sample, "error_kind": "benchmark", **error_metrics(panel.loc[mask, f"{name}_carry_error"], panel.loc[mask, f"{name}_futures_error"])})
    return pd.DataFrame(rows)


def likelihood_ratio_table(restricted, unrestricted) -> pd.DataFrame:
    if restricted.mode != unrestricted.mode:
        raise ValueError("Likelihood-ratio models must use the same data stream")
    statistic = max(0.0, 2 * (unrestricted.log_likelihood - restricted.log_likelihood))
    return pd.DataFrame([{
        "mode": unrestricted.mode,
        "restricted_model": restricted.name,
        "unrestricted_model": unrestricted.name,
        "restricted_log_likelihood": restricted.log_likelihood,
        "unrestricted_log_likelihood": unrestricted.log_likelihood,
        "lr_statistic": statistic,
        "degrees_of_freedom": 1,
        "p_value": float(chi2.sf(statistic, 1)),
    }])


def innovation_diagnostics(states: pd.DataFrame, rolling_window: int = 63) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["date", "standardized_curve_innovation", "standardized_return_residual"]
    work = states[columns].dropna().copy()
    work["rolling_correlation"] = work["standardized_curve_innovation"].rolling(
        int(rolling_window), min_periods=max(10, int(rolling_window) // 3)
    ).corr(work["standardized_return_residual"])
    if len(work) >= 3:
        pearson = pearsonr(work["standardized_curve_innovation"], work["standardized_return_residual"])
        spearman = spearmanr(work["standardized_curve_innovation"], work["standardized_return_residual"])
        summary = pd.DataFrame([
            {"test": "pearson", "correlation": pearson.statistic, "p_value": pearson.pvalue, "n": len(work)},
            {"test": "spearman", "correlation": spearman.statistic, "p_value": spearman.pvalue, "n": len(work)},
        ])
    else:
        summary = pd.DataFrame([{"test": "pearson", "correlation": np.nan, "p_value": np.nan, "n": len(work)}])
    return work, summary


def residual_by_maturity(fits: pd.DataFrame) -> pd.DataFrame:
    bins = [5, 21, 42, 63, 126, 189, 252, np.inf]
    labels = ["6-21", "22-42", "43-63", "64-126", "127-189", "190-252", "253+"]
    work = fits.copy()
    work["maturity_bucket"] = pd.cut(work["sessions_to_expiry"], bins=bins, labels=labels)
    return (
        work.groupby(["model", "maturity_bucket"], observed=True)
        .agg(
            n=("carry_residual", "size"),
            mean_carry_residual_bps=("carry_residual", lambda x: 1e4 * np.mean(x)),
            carry_rmse_bps=("carry_residual", lambda x: 1e4 * np.sqrt(np.mean(np.square(x)))),
            mean_futures_residual=("futures_residual", "mean"),
            futures_rmse=("futures_residual", lambda x: np.sqrt(np.mean(np.square(x)))),
        ).reset_index()
    )


def residual_time_series(fits: pd.DataFrame) -> pd.DataFrame:
    return (
        fits.groupby(["model", "date"], sort=True)
        .agg(
            mean_carry_residual=("carry_residual", "mean"),
            carry_rmse=("carry_residual", lambda x: np.sqrt(np.mean(np.square(x)))),
            mean_futures_residual=("futures_residual", "mean"),
            futures_rmse=("futures_residual", lambda x: np.sqrt(np.mean(np.square(x)))),
            n_contracts=("contract", "size"),
        ).reset_index()
    )


def residual_tests(time_series: pd.DataFrame, lags: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    acf_rows = []
    test_rows = []
    for model, group in time_series.groupby("model", sort=False):
        values = group["mean_carry_residual"].dropna().to_numpy(dtype=float)
        usable = min(int(lags), max(1, len(values) // 5), max(1, len(values) - 2))
        for lag, value in enumerate(acf(values, nlags=usable, fft=True, missing="drop")):
            acf_rows.append({"model": model, "lag": lag, "acf": value})
        if len(values) > usable + 5:
            lb = acorr_ljungbox(values, lags=[usable], return_df=True).iloc[0]
            test_rows.append({"model": model, "test": "ljung_box", "statistic": lb.lb_stat, "p_value": lb.lb_pvalue, "lags": usable})
            squared = acorr_ljungbox(values**2, lags=[usable], return_df=True).iloc[0]
            test_rows.append({"model": model, "test": "ljung_box_squared", "statistic": squared.lb_stat, "p_value": squared.lb_pvalue, "lags": usable})
            try:
                lm, pvalue, _, _ = het_arch(values, nlags=min(10, usable))
                test_rows.append({"model": model, "test": "arch_lm", "statistic": lm, "p_value": pvalue, "lags": min(10, usable)})
            except (ValueError, np.linalg.LinAlgError):
                pass
    return pd.DataFrame(acf_rows), pd.DataFrame(test_rows)

