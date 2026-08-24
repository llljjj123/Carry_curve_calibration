"""Calibration metrics, benchmarks, residual tests, and curve-shape flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf


def _error_metrics(carry_error: pd.Series, price_error: pd.Series) -> dict[str, float | int]:
    carry = pd.to_numeric(carry_error, errors="coerce").dropna().to_numpy(dtype=float)
    price = pd.to_numeric(price_error, errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "n_carry": len(carry),
        "carry_rmse": float(np.sqrt(np.mean(carry**2))) if len(carry) else np.nan,
        "carry_mae": float(np.mean(np.abs(carry))) if len(carry) else np.nan,
        "carry_rmse_bps": float(1e4 * np.sqrt(np.mean(carry**2))) if len(carry) else np.nan,
        "carry_mae_bps": float(1e4 * np.mean(np.abs(carry))) if len(carry) else np.nan,
        "n_price": len(price),
        "futures_rmse_points": float(np.sqrt(np.mean(price**2))) if len(price) else np.nan,
        "futures_mae_points": float(np.mean(np.abs(price))) if len(price) else np.nan,
        "carry_mean_bias": float(np.mean(carry)) if len(carry) else np.nan,
        "futures_mean_bias": float(np.mean(price)) if len(price) else np.nan,
    }


def model_metrics(fits: pd.DataFrame, split_date: object, model_name: str = "ou_one_factor") -> pd.DataFrame:
    """Filtered in-sample and one-step-prior out-of-sample model metrics."""
    split = pd.Timestamp(split_date)
    rows = []
    for sample, mask, carry_col, price_col in (
        ("in_sample", fits["date"] <= split, "carry_residual", "futures_residual"),
        ("out_of_sample", fits["date"] > split, "carry_prediction_error", "futures_prediction_error"),
    ):
        rows.append({"model": model_name, "sample": sample, **_error_metrics(fits.loc[mask, carry_col], fits.loc[mask, price_col])})
    return pd.DataFrame(rows)


def attach_benchmarks(panel: pd.DataFrame, ewma_span: int = 20) -> pd.DataFrame:
    """Attach flat, previous-day level, and lagged EWMA carry benchmarks."""
    result = panel.copy()
    levels = result.groupby("date", sort=True)["implied_carry"].mean().rename("daily_level").to_frame()
    levels["flat_same_day"] = levels["daily_level"]
    levels["previous_day_level"] = levels["daily_level"].shift(1)
    levels["ewma_level"] = levels["daily_level"].ewm(span=ewma_span, adjust=False).mean().shift(1)
    result = result.merge(levels.drop(columns="daily_level"), on="date", how="left")
    for name in ("flat_same_day", "previous_day_level", "ewma_level"):
        result[f"{name}_carry_error"] = result["implied_carry"] - result[name]
        fitted_price = result["spot"] * np.exp((result["risk_free_rate"] - result[name]) * result["tau"])
        result[f"{name}_futures_error"] = result["futures_price"] - fitted_price
    return result


def benchmark_metrics(panel: pd.DataFrame, split_date: object) -> pd.DataFrame:
    split = pd.Timestamp(split_date)
    rows = []
    for name in ("flat_same_day", "previous_day_level", "ewma_level"):
        for sample, mask in (("in_sample", panel["date"] <= split), ("out_of_sample", panel["date"] > split)):
            rows.append(
                {
                    "model": name,
                    "sample": sample,
                    **_error_metrics(panel.loc[mask, f"{name}_carry_error"], panel.loc[mask, f"{name}_futures_error"]),
                }
            )
    return pd.DataFrame(rows)


def residual_by_maturity(fits: pd.DataFrame) -> pd.DataFrame:
    """Aggregate filtered residuals into trading-session maturity buckets."""
    bins = [5, 21, 42, 63, 126, 189, 252, np.inf]
    labels = ["6-21", "22-42", "43-63", "64-126", "127-189", "190-252", "253+"]
    work = fits.copy()
    work["maturity_bucket"] = pd.cut(work["sessions_to_expiry"], bins=bins, labels=labels, right=True)
    return (
        work.groupby("maturity_bucket", observed=True)
        .agg(
            n=("carry_residual", "size"),
            mean_carry_residual=("carry_residual", "mean"),
            carry_rmse=("carry_residual", lambda x: np.sqrt(np.mean(np.square(x)))),
            mean_futures_residual=("futures_residual", "mean"),
            futures_rmse=("futures_residual", lambda x: np.sqrt(np.mean(np.square(x)))),
        )
        .reset_index()
    )


def residual_time_series(fits: pd.DataFrame) -> pd.DataFrame:
    return (
        fits.groupby("date", sort=True)
        .agg(
            mean_carry_residual=("carry_residual", "mean"),
            mean_abs_carry_residual=("carry_residual", lambda x: np.mean(np.abs(x))),
            carry_rmse=("carry_residual", lambda x: np.sqrt(np.mean(np.square(x)))),
            mean_futures_residual=("futures_residual", "mean"),
            futures_rmse=("futures_residual", lambda x: np.sqrt(np.mean(np.square(x)))),
            n_contracts=("contract", "size"),
        )
        .reset_index()
    )


def residual_statistical_tests(time_series: pd.DataFrame, lags: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = time_series["mean_carry_residual"].dropna().to_numpy(dtype=float)
    usable_lags = min(int(lags), max(1, len(values) // 5), max(1, len(values) - 2))
    acf_values = acf(values, nlags=usable_lags, fft=True, missing="drop")
    acf_table = pd.DataFrame({"lag": np.arange(len(acf_values)), "acf": acf_values})
    rows: list[dict[str, object]] = []
    if len(values) > usable_lags + 5:
        lb = acorr_ljungbox(values, lags=[usable_lags], return_df=True).iloc[0]
        rows.append({"test": "ljung_box_residual", "statistic": lb["lb_stat"], "p_value": lb["lb_pvalue"], "lags": usable_lags})
        lb2 = acorr_ljungbox(values**2, lags=[usable_lags], return_df=True).iloc[0]
        rows.append({"test": "ljung_box_squared_residual", "statistic": lb2["lb_stat"], "p_value": lb2["lb_pvalue"], "lags": usable_lags})
        arch_lags = min(10, usable_lags)
        try:
            lm, pvalue, _, _ = het_arch(values, nlags=arch_lags)
            rows.append({"test": "arch_lm", "statistic": lm, "p_value": pvalue, "lags": arch_lags})
        except (ValueError, np.linalg.LinAlgError):
            rows.append({"test": "arch_lm", "statistic": np.nan, "p_value": np.nan, "lags": arch_lags})
    return acf_table, pd.DataFrame(rows)


def expiry_roll_diagnostics(fits: pd.DataFrame) -> pd.DataFrame:
    work = fits.copy().sort_values(["date", "contract"])
    contract_sets = work.groupby("date")["contract"].agg(lambda x: tuple(sorted(x)))
    roll_dates = contract_sets.index[contract_sets.ne(contract_sets.shift(1))]
    work["roll_change_date"] = work["date"].isin(roll_dates)
    work["expiry_zone"] = pd.cut(
        work["sessions_to_expiry"], [5, 10, 21, np.inf], labels=["6-10_sessions", "11-21_sessions", "22+_sessions"]
    )
    return (
        work.groupby(["expiry_zone", "roll_change_date"], observed=True)
        .agg(
            n=("carry_residual", "size"),
            mean_carry_residual=("carry_residual", "mean"),
            carry_rmse=("carry_residual", lambda x: np.sqrt(np.mean(np.square(x)))),
            futures_rmse=("futures_residual", lambda x: np.sqrt(np.mean(np.square(x)))),
        )
        .reset_index()
    )


def curve_shape_flags(panel: pd.DataFrame, tolerance_bps: float = 5.0) -> pd.DataFrame:
    """Flag inversions and non-monotonic curves beyond a noise tolerance."""
    tolerance = float(tolerance_bps) / 1e4
    rows = []
    for date, group in panel.groupby("date", sort=True):
        curve = group.sort_values("tau")
        values = curve["implied_carry"].to_numpy(dtype=float)
        diffs = np.diff(values)
        material = diffs[np.abs(diffs) > tolerance]
        signs = np.sign(material)
        sign_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        inversion = bool(len(values) > 1 and values[-1] < values[0] - tolerance)
        hump_or_u = sign_changes > 0
        rows.append(
            {
                "date": date,
                "n_contracts": len(curve),
                "front_carry": values[0] if len(values) else np.nan,
                "back_carry": values[-1] if len(values) else np.nan,
                "inversion": inversion,
                "hump_or_u_shape": hump_or_u,
                "material_slope_sign_changes": sign_changes,
                "one_factor_shape_limitation": hump_or_u,
            }
        )
    return pd.DataFrame(rows)


def maturity_dependence_test(fits: pd.DataFrame) -> pd.DataFrame:
    valid = fits[["tau", "carry_residual"]].dropna()
    correlation, pvalue = spearmanr(valid["tau"], valid["carry_residual"])
    return pd.DataFrame(
        [{"test": "spearman_residual_vs_maturity", "statistic": correlation, "p_value": pvalue, "n": len(valid)}]
    )


def standardized_innovation_diagnostics(
    innovations: pd.DataFrame,
    lags: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test daily aggregated standardized one-step Kalman innovations."""
    daily = (
        innovations.groupby("date", sort=True)["standardized_innovation"]
        .agg([("mean_standardized_innovation", "mean"), ("innovation_rms", lambda x: np.sqrt(np.mean(np.square(x)))), ("n", "size")])
        .reset_index()
    )
    values = daily["mean_standardized_innovation"].dropna().to_numpy(dtype=float)
    usable_lags = min(int(lags), max(1, len(values) // 5), max(1, len(values) - 2))
    acf_values = acf(values, nlags=usable_lags, fft=True, missing="drop")
    acf_table = pd.DataFrame({"lag": np.arange(len(acf_values)), "acf": acf_values})
    tests = []
    if len(values) > usable_lags + 5:
        for label, tested in (("innovation_ljung_box", values), ("innovation_squared_ljung_box", values**2)):
            row = acorr_ljungbox(tested, lags=[usable_lags], return_df=True).iloc[0]
            tests.append({"test": label, "statistic": row["lb_stat"], "p_value": row["lb_pvalue"], "lags": usable_lags})
        try:
            lm, pvalue, _, _ = het_arch(values, nlags=min(10, usable_lags))
            tests.append({"test": "innovation_arch_lm", "statistic": lm, "p_value": pvalue, "lags": min(10, usable_lags)})
        except (ValueError, np.linalg.LinAlgError):
            pass
    return daily, acf_table, pd.DataFrame(tests)


def shape_fit_comparison(
    observed_panel: pd.DataFrame,
    one_factor_fits: pd.DataFrame,
    two_factor_fits: pd.DataFrame,
    tolerance_bps: float = 5.0,
) -> pd.DataFrame:
    """Compare observed and fitted turning-point classifications date by date."""
    tolerance = float(tolerance_bps) / 1e4

    def classification(values: np.ndarray) -> tuple[bool, bool, int]:
        differences = np.diff(values)
        material = differences[np.abs(differences) > tolerance]
        signs = np.sign(material)
        changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        inversion = bool(len(values) > 1 and values[-1] < values[0] - tolerance)
        return changes > 0, inversion, changes

    one_by_date = {date: group.sort_values("tau") for date, group in one_factor_fits.groupby("date")}
    two_by_date = {date: group.sort_values("tau") for date, group in two_factor_fits.groupby("date")}
    rows = []
    for date, observed in observed_panel.groupby("date", sort=True):
        observed = observed.sort_values("tau")
        one = one_by_date[date]
        two = two_by_date[date]
        observed_hump, observed_inversion, observed_changes = classification(observed["implied_carry"].to_numpy())
        one_hump, one_inversion, one_changes = classification(one["fitted_carry"].to_numpy())
        two_hump, two_inversion, two_changes = classification(two["fitted_carry"].to_numpy())
        rows.append(
            {
                "date": date,
                "n_contracts": len(observed),
                "observed_hump_or_u": observed_hump,
                "one_factor_hump_or_u": one_hump,
                "two_factor_hump_or_u": two_hump,
                "two_factor_correctly_reproduced_shape": observed_hump == two_hump,
                "observed_inversion": observed_inversion,
                "one_factor_inversion": one_inversion,
                "two_factor_inversion": two_inversion,
                "observed_sign_changes": observed_changes,
                "one_factor_sign_changes": one_changes,
                "two_factor_sign_changes": two_changes,
            }
        )
    return pd.DataFrame(rows)
