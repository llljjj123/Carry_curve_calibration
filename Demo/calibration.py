"""Flexible data preparation and two-factor OU calibration for the demo.

The numerical implementation is deliberately reused from the validated
``im_2factor_ou_carry`` project.  This module fixes the demo sample and exposes
small, notebook-friendly functions without duplicating the Kalman filter or
maximum-likelihood code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


DEMO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEMO_ROOT.parent
CALIBRATION_PROJECT = WORKSPACE_ROOT / "im_2factor_ou_carry"
CALIBRATION_SRC = CALIBRATION_PROJECT / "src"
if str(CALIBRATION_SRC) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SRC))

from im_2factor_ou_carry.data import normalized_market_panel  # noqa: E402
from im_2factor_ou_carry.two_factor import (  # noqa: E402
    TwoFactorFilterResult,
    two_factor_kalman_filter,
)
from im_2factor_ou_carry.two_factor_estimation import (  # noqa: E402
    TwoFactorEstimationResult,
    estimate_two_factor_ou,
    two_factor_parameter_table,
)
from im_2factor_ou_carry.two_factor_fitting import attach_two_factor_fits  # noqa: E402

from calendar_utils import (  # noqa: E402
    calendar_source_for_date,
    contract_expiry,
    trading_days_between,
)
from demo_quality import prepare_implied_carry  # noqa: E402


EVALUATION_DATE = pd.Timestamp("2026-08-21")
CALIBRATION_DATES = 244
PERIODS_PER_YEAR = 244
RISK_FREE_RATE = 0.014
CONTRACT_CODE = "IM2609"
FAST_ETA_UPPER_BOUND = 6.0
KAPPA_GAP_UPPER_BOUND = 60.0


@dataclass
class CalibrationResult:
    """All inputs and outputs needed by the demonstration notebook."""

    sample: pd.DataFrame
    quality_audit: pd.DataFrame
    spot_history: pd.DataFrame
    estimate: TwoFactorEstimationResult
    filter_result: TwoFactorFilterResult
    fitted_panel: pd.DataFrame
    parameter_table: pd.DataFrame
    quote: pd.Series
    historical_volatility: float
    metrics: dict[str, Any]

    @property
    def latest_state(self) -> pd.Series:
        return self.filter_result.states.iloc[-1]


def demo_config() -> dict[str, Any]:
    """Return the conventions agreed for the demonstration."""
    return {
        "data": {
            "risk_free_rate": RISK_FREE_RATE,
            "futures_price_field": "close",
        },
        "calendar": {
            "periods_per_year": PERIODS_PER_YEAR,
            "min_sessions_to_expiry": 5,
            "special_exchange_closures": ["2024-02-09"],
        },
        "quality": {
            "max_abs_implied_carry": 0.50,
            "stale_run_length": 3,
            "exclude_stale": False,
        },
    }


def _gap_function(start: object, end: object) -> float:
    return trading_days_between(start, end, ["2024-02-09"]) / PERIODS_PER_YEAR


def load_calibration_sample(
    evaluation_date: object = EVALUATION_DATE,
    window_dates: int = CALIBRATION_DATES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, clean, and select the final ``window_dates`` accepted curve dates.

    Selection is by distinct curve date, not by individual futures observation.
    The evaluation date is included and no observation after it is visible to
    either the quality rules or the estimator.
    """
    evaluation_date = pd.Timestamp(evaluation_date).normalize()
    if isinstance(window_dates, bool) or int(window_dates) != window_dates or window_dates < 2:
        raise ValueError("window_dates must be an integer of at least 2")
    window_dates = int(window_dates)
    raw_dir = CALIBRATION_PROJECT / "data" / "raw"
    spot = pd.read_csv(raw_dir / "spot_raw.csv", parse_dates=["date"])
    futures = pd.read_csv(raw_dir / "futures_raw.csv", parse_dates=["date", "expiry"])
    spot = spot.loc[spot["date"] <= evaluation_date].copy()
    futures = futures.loc[futures["date"] <= evaluation_date].copy()

    market = normalized_market_panel(spot, futures, demo_config())
    accepted, audit = prepare_implied_carry(market, demo_config())
    available_dates = pd.Index(pd.to_datetime(accepted["date"].unique())).sort_values()
    if evaluation_date not in available_dates:
        raise ValueError(f"Evaluation date {evaluation_date.date()} has no accepted carry curve")
    if len(available_dates) < window_dates:
        raise ValueError(f"Need {window_dates} curve dates, found only {len(available_dates)}")

    selected_dates = available_dates[-window_dates:]
    sample = accepted.loc[accepted["date"].isin(selected_dates)].copy()
    sample = sample.sort_values(["date", "tau", "contract"]).reset_index(drop=True)
    sample_audit = audit.loc[audit["date"].isin(selected_dates)].copy()
    sample_audit = sample_audit.sort_values(["date", "contract"]).reset_index(drop=True)

    spot_history = (
        sample[["date", "spot"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(spot_history) != window_dates:
        raise AssertionError("Spot and accepted-curve date counts are not aligned")
    spot_history["log_return"] = np.log(spot_history["spot"]).diff()
    return sample, sample_audit, spot_history


def estimate_historical_volatility(spot_history: pd.DataFrame) -> float:
    """Annualized close-to-close log-return volatility using 244 sessions/year."""
    returns = spot_history["log_return"].dropna().to_numpy(dtype=float)
    if len(returns) < 2:
        raise ValueError("At least two returns are required")
    return float(np.std(returns, ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def calibrate_two_factor(
    *,
    evaluation_date: object = EVALUATION_DATE,
    window_dates: int = CALIBRATION_DATES,
    optimizer_starts: int = 12,
    optimizer_maxiter: int = 1500,
    compute_standard_errors: bool = True,
    random_seed: int = 852,
    eta_fast_upper_bound: float = FAST_ETA_UPPER_BOUND,
    contract_code: str = CONTRACT_CODE,
) -> CalibrationResult:
    """Calibrate the independent slow/fast OU model and filter current states."""
    sample, audit, spot_history = load_calibration_sample(evaluation_date, window_dates)
    estimate = estimate_two_factor_ou(
        sample,
        gap_function=_gap_function,
        starts=optimizer_starts,
        maxiter=optimizer_maxiter,
        seed=random_seed,
        compute_standard_errors=compute_standard_errors,
        eta_fast_upper_bound=eta_fast_upper_bound,
    )
    filtered = two_factor_kalman_filter(
        sample,
        estimate.params,
        gap_function=_gap_function,
    )
    fitted = attach_two_factor_fits(sample, filtered.states, estimate.params)
    parameters = two_factor_parameter_table(estimate)
    historical_volatility = estimate_historical_volatility(spot_history)

    contract_code = str(contract_code).strip().upper()
    inferred_expiry = pd.Timestamp(contract_expiry(contract_code))
    quote_rows = fitted.loc[
        (fitted["date"] == pd.Timestamp(evaluation_date).normalize())
        & (fitted["contract"] == contract_code)
    ]
    if len(quote_rows) != 1:
        raise ValueError(f"Expected one accepted {contract_code} quote, found {len(quote_rows)}")
    quote = quote_rows.iloc[0]
    if pd.Timestamp(quote["expiry"]) != inferred_expiry:
        raise AssertionError(
            f"{contract_code} quote expiry {quote['expiry']} differs from inferred {inferred_expiry.date()}"
        )

    residual = fitted["carry_residual"].to_numpy(dtype=float)
    futures_residual = fitted["futures_residual"].to_numpy(dtype=float)
    n_observations = len(fitted)
    parameter_count = 6
    latest_state = filtered.states.iloc[-1]
    nearest_sessions = int(
        fitted.loc[fitted["date"] == pd.Timestamp(evaluation_date), "sessions_to_expiry"].min()
    )
    metrics: dict[str, Any] = {
        "evaluation_date": str(pd.Timestamp(evaluation_date).date()),
        "requested_sample_size": int(window_dates),
        "selected_futures_contract": contract_code,
        "selected_contract_inferred_expiry": str(inferred_expiry.date()),
        "selected_contract_expiry_calendar_source": calendar_source_for_date(inferred_expiry),
        "selected_contract_sessions_to_expiry": int(quote["sessions_to_expiry"]),
        "sample_start": str(pd.Timestamp(sample["date"].min()).date()),
        "sample_end": str(pd.Timestamp(sample["date"].max()).date()),
        "curve_dates": int(sample["date"].nunique()),
        "accepted_observations": n_observations,
        "excluded_observations_within_window": int(audit["excluded"].sum()),
        "distinct_contracts": int(sample["contract"].nunique()),
        "return_observations": int(spot_history["log_return"].notna().sum()),
        "historical_volatility": historical_volatility,
        "eta_fast_upper_bound": float(eta_fast_upper_bound),
        "kappa_fast_minus_slow_upper_bound": KAPPA_GAP_UPPER_BOUND,
        "log_likelihood": float(estimate.log_likelihood),
        "aic": float(2 * parameter_count - 2 * estimate.log_likelihood),
        "bic": float(parameter_count * np.log(n_observations) - 2 * estimate.log_likelihood),
        "carry_rmse_bps": float(np.sqrt(np.mean(residual**2)) * 10_000),
        "carry_mae_bps": float(np.mean(np.abs(residual)) * 10_000),
        "futures_rmse_points": float(np.sqrt(np.mean(futures_residual**2))),
        "futures_mae_points": float(np.mean(np.abs(futures_residual))),
        "optimizer_converged": bool(estimate.converged),
        "eta_fast_at_upper_bound": bool(
            estimate.params.eta_fast >= eta_fast_upper_bound - 1.0e-8
        ),
        "kappa_fast_minus_slow_at_upper_bound": bool(
            estimate.params.kappa_fast - estimate.params.kappa_slow
            >= KAPPA_GAP_UPPER_BOUND - 1.0e-8
        ),
        "hessian_stable": bool(estimate.hessian_stable),
        "latest_filtered_slow_state": float(latest_state["filtered_slow_state"]),
        "latest_filtered_fast_state": float(latest_state["filtered_fast_state"]),
        "latest_filtered_instantaneous_carry": float(
            latest_state["filtered_instantaneous_carry"]
        ),
        "latest_filtered_instantaneous_std": float(
            latest_state["filtered_instantaneous_std"]
        ),
        "nearest_contract_sessions": nearest_sessions,
        "weak_latest_instantaneous_observability": bool(
            nearest_sessions > 21 or latest_state["filtered_instantaneous_std"] > 0.04
        ),
    }
    return CalibrationResult(
        sample=sample,
        quality_audit=audit,
        spot_history=spot_history,
        estimate=estimate,
        filter_result=filtered,
        fitted_panel=fitted,
        parameter_table=parameters,
        quote=quote,
        historical_volatility=historical_volatility,
        metrics=metrics,
    )


def export_calibration(result: CalibrationResult, output_dir: Path) -> None:
    """Write compact, reproducible calibration artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.sample.to_csv(output_dir / "calibration_sample.csv", index=False)
    result.quality_audit.to_csv(output_dir / "quality_audit.csv", index=False)
    result.spot_history.to_csv(output_dir / "spot_history.csv", index=False)
    result.parameter_table.to_csv(output_dir / "two_factor_parameters.csv", index=False)
    result.estimate.optimizer_runs.to_csv(output_dir / "optimizer_runs.csv", index=False)
    result.filter_result.states.to_csv(output_dir / "filtered_states.csv", index=False)
    result.fitted_panel.to_csv(output_dir / "fitted_curves.csv", index=False)


def parameter_estimates(result: CalibrationResult) -> dict[str, float]:
    """Return calibrated parameters as an ordinary dictionary."""
    return {key: float(value) for key, value in asdict(result.estimate.params).items()}
