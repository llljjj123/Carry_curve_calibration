"""End-to-end orchestration, exports, and charts for the carry-put demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from calibration import (
    CALIBRATION_DATES,
    EVALUATION_DATE,
    RISK_FREE_RATE,
    CalibrationResult,
    calibrate_two_factor,
    export_calibration,
    parameter_estimates,
)
from calendar_utils import EXPLICIT_CALENDAR_PATH, explicit_calendar_years, maturity_diagnostics
from option_pricing import OptionPricingBundle, export_pricing, price_optional_component
from profile_analysis import FixedEtaProfile, run_fixed_eta_profile


DEMO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = DEMO_ROOT / "outputs"


@dataclass
class DemoResult:
    calibration: CalibrationResult
    pricing: OptionPricingBundle
    eta_fast_profile: FixedEtaProfile
    summary: dict[str, Any]


def _make_charts(
    calibration: CalibrationResult,
    pricing: OptionPricingBundle,
    eta_fast_profile: FixedEtaProfile,
    chart_dir: Path,
) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    latest = calibration.fitted_panel.loc[
        calibration.fitted_panel["date"] == calibration.fitted_panel["date"].max()
    ].sort_values("sessions_to_expiry")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        latest["sessions_to_expiry"], latest["implied_carry"] * 100,
        "o-", label="Observed implied carry",
    )
    axes[0].plot(
        latest["sessions_to_expiry"], latest["fitted_carry"] * 100,
        "s--", label="Filtered two-factor fit",
    )
    axes[0].set(xlabel="Trading sessions to expiry", ylabel="Annualized carry (%)")
    axes[0].set_title("Valuation-date carry curve")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        latest["sessions_to_expiry"], latest["futures_price"],
        "o-", label="Observed IM close",
    )
    axes[1].plot(
        latest["sessions_to_expiry"], latest["fitted_futures_price"],
        "s--", label="Filtered model fit",
    )
    axes[1].set(xlabel="Trading sessions to expiry", ylabel="Index points")
    axes[1].set_title("Valuation-date futures curve")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(chart_dir / "latest_curve.png", dpi=160)
    plt.close(figure)

    states = calibration.filter_result.states
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    axes[0].plot(states["date"], states["filtered_slow_state"], label="Slow factor")
    axes[0].plot(states["date"], states["filtered_fast_state"], label="Fast factor", alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Centered factor")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        states["date"], states["filtered_instantaneous_carry"] * 100,
        label="Filtered instantaneous carry",
    )
    axes[1].axhline(
        calibration.estimate.params.theta * 100,
        color="tab:red", linestyle="--", label="Long-run level theta",
    )
    axes[1].set(ylabel="Annualized carry (%)", xlabel="Date")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.suptitle(
        "Filtered two-factor states: "
        f"{calibration.metrics['curve_dates']}-date calibration window"
    )
    figure.tight_layout()
    figure.savefig(chart_dir / "filtered_states.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    grid = pricing.grid_convergence
    axes[0].plot(grid["configuration"], grid["price"], "o-")
    axes[0].set(ylabel="Option price (index points)", xlabel="State-grid configuration")
    axes[0].set_title("Grid convergence")
    axes[0].grid(alpha=0.25)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    quad = pricing.quadrature_convergence
    axes[1].plot(quad["quadrature_order"], quad["price"], "o-")
    axes[1].set(ylabel="Option price (index points)", xlabel="Gauss-Hermite order")
    axes[1].set_title("Quadrature convergence")
    axes[1].grid(alpha=0.25)
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    figure.tight_layout()
    figure.savefig(chart_dir / "numerical_convergence.png", dpi=160)
    plt.close(figure)

    exercise = pricing.exercise_summary.sort_values("elapsed_sessions")
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(
        exercise["elapsed_sessions"], exercise["exercise_grid_fraction"] * 100,
        "o-", markersize=3,
    )
    axis.set(
        xlabel="Elapsed trading sessions",
        ylabel="Exercise share of state grid (%)",
        title="Daily early-exercise region",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(chart_dir / "exercise_region.png", dpi=160)
    plt.close(figure)

    profile = eta_fast_profile.results
    eta_estimate = float(calibration.estimate.params.eta_fast)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(
        profile["eta_fast_fixed"],
        profile["log_likelihood_below_profile_max"],
        "o-",
    )
    axes[0].axvline(6.0, color="tab:red", linestyle="--", label="Revised cap")
    axes[0].axvline(
        eta_estimate, color="tab:green", linestyle=":", label="Cap-6 estimate"
    )
    axes[0].set(
        xlabel="Fixed fast-factor volatility",
        ylabel="Log likelihood below profile maximum",
        title="Fixed-eta likelihood profile",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(profile["eta_fast_fixed"], profile["option_price"], "o-")
    axes[1].axvline(6.0, color="tab:red", linestyle="--")
    axes[1].axvline(eta_estimate, color="tab:green", linestyle=":")
    axes[1].set(
        xlabel="Fixed fast-factor volatility",
        ylabel="Option price (index points)",
        title="Carry-put price profile",
    )
    axes[1].grid(alpha=0.25)
    axes[2].plot(
        profile["eta_fast_fixed"], profile["kappa_fast"], "o-", label="kappa fast"
    )
    axes[2].axvline(eta_estimate, color="tab:green", linestyle=":")
    axes[2].set(
        xlabel="Fixed fast-factor volatility",
        ylabel="Fast mean-reversion speed",
        title="Kappa/eta interaction",
    )
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(chart_dir / "eta_fast_profile.png", dpi=160)
    plt.close(figure)


def _summary(
    calibration: CalibrationResult,
    pricing: OptionPricingBundle,
    eta_fast_profile: FixedEtaProfile,
) -> dict[str, Any]:
    quote = calibration.quote
    base = pricing.base_result
    warnings = [
        "Historical two-factor OU parameters are provisionally treated as risk-neutral parameters.",
        "Exercise is available once per trading session, so the calculation is a daily Bermudan approximation to an American option.",
        "The linear IM futures leg is deliberately excluded; the reported value is the carry-put optional component only.",
        "Spot volatility is estimated and reported but cancels from this homogeneous payoff under zero spot/carry shock correlation.",
        "Maturities entering 2027 use the Demo's explicit company calendar rather than the shared weekday fallback; this calendar is provisional until the official exchange schedule is available.",
    ]
    eta_cap = float(calibration.metrics["eta_fast_upper_bound"])
    if calibration.metrics["eta_fast_at_upper_bound"]:
        warnings.append(
            f"The fast-factor volatility is at the revised upper bound of {eta_cap:g}; the larger cap did not produce an interior optimum."
        )
    if calibration.metrics["kappa_fast_minus_slow_at_upper_bound"]:
        warnings.append(
            "The fast-minus-slow mean-reversion gap is at its upper bound of 60; the eta boundary is resolved, but fast mean reversion remains constrained."
        )
    if calibration.metrics["weak_latest_instantaneous_observability"]:
        warnings.append("The latest instantaneous state meets the established weak-observability flag.")
    profile = eta_fast_profile.results
    best_profile = profile.loc[profile["log_likelihood"].idxmax()]
    profile_95 = profile.loc[profile["likelihood_ratio_vs_profile_max"] <= 3.841459]
    eta3_profile = profile.loc[profile["eta_fast_fixed"] == 3.0].iloc[0]
    profile_summary = {
        "eta_grid": profile["eta_fast_fixed"].tolist(),
        "best_grid_eta_fast": float(best_profile["eta_fast_fixed"]),
        "best_grid_log_likelihood": float(best_profile["log_likelihood"]),
        "log_likelihood_gain_over_cap6_fit": float(
            best_profile["log_likelihood"] - calibration.estimate.log_likelihood
        ),
        "cap6_option_price": float(base.price),
        "eta3_log_likelihood": float(eta3_profile["log_likelihood"]),
        "log_likelihood_gain_over_eta3": float(
            calibration.estimate.log_likelihood - eta3_profile["log_likelihood"]
        ),
        "likelihood_ratio_vs_eta3": float(
            2.0 * (calibration.estimate.log_likelihood - eta3_profile["log_likelihood"])
        ),
        "eta3_option_price": float(eta3_profile["option_price"]),
        "cap6_price_difference_from_eta3": float(
            base.price - eta3_profile["option_price"]
        ),
        "minimum_profile_option_price": float(profile["option_price"].min()),
        "maximum_profile_option_price": float(profile["option_price"].max()),
        "profile_grid_95pct_eta_min": float(profile_95["eta_fast_fixed"].min()),
        "profile_grid_95pct_eta_max": float(profile_95["eta_fast_fixed"].max()),
        "profile_grid_95pct_option_price_min": float(profile_95["option_price"].min()),
        "profile_grid_95pct_option_price_max": float(profile_95["option_price"].max()),
    }
    return {
        "scope": "American carry-put optional component only; linear futures leg excluded",
        "calibration": calibration.metrics,
        "two_factor_parameters": parameter_estimates(calibration),
        "contract": {
            "valuation_date": str(pd.Timestamp(quote["date"]).date()),
            "contract": str(quote["contract"]),
            "expiry": str(pd.Timestamp(quote["expiry"]).date()),
            "sessions_to_expiry": int(quote["sessions_to_expiry"]),
            "initial_spot": float(quote["spot"]),
            "initial_futures": float(quote["futures_price"]),
            "risk_free_rate": RISK_FREE_RATE,
            "historical_spot_volatility": calibration.historical_volatility,
            "locked_carry": base.locked_carry,
        },
        "calendar": {
            "explicit_calendar_path": str(EXPLICIT_CALENDAR_PATH),
            "explicit_calendar_years": list(explicit_calendar_years()),
            "unsupported_year_policy": "raise CalendarCoverageError; never use weekday fallback",
            "im2703_diagnostic": maturity_diagnostics("IM2703", quote["date"]),
        },
        "option_result": base.as_dict(include_exercise_summary=False),
        "fixed_eta_fast_profile": profile_summary,
        "numerical_diagnostics": {
            "absolute_base_minus_fine_grid_price": float(
                abs(grid_value(pricing, "base") - grid_value(pricing, "fine"))
            ),
            "quadrature_price_range_orders_39_to_47": float(
                pricing.quadrature_convergence["price"].max()
                - pricing.quadrature_convergence["price"].min()
            ),
        },
        "warnings": warnings,
    }


def grid_value(pricing: OptionPricingBundle, configuration: str) -> float:
    row = pricing.grid_convergence.loc[
        pricing.grid_convergence["configuration"] == configuration, "price"
    ]
    return float(row.iloc[0])


def run_demo(
    *,
    valuation_date: object = EVALUATION_DATE,
    sample_size: int = CALIBRATION_DATES,
    futures_contract: str = "IM2609",
    output_dir: str | Path = OUTPUT_DIR,
) -> DemoResult:
    """Run calibration, price the option-only leg, and persist all demo artifacts."""
    output_dir = Path(output_dir).resolve()
    chart_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = calibrate_two_factor(
        evaluation_date=valuation_date,
        window_dates=sample_size,
        contract_code=futures_contract,
    )
    export_calibration(calibration, output_dir)
    pricing = price_optional_component(calibration)
    export_pricing(pricing, output_dir)
    eta_fast_profile = run_fixed_eta_profile(calibration)
    eta_fast_profile.results["option_price_difference_from_cap6_fit"] = (
        eta_fast_profile.results["option_price"] - pricing.base_result.price
    )
    eta_fast_profile.results.to_csv(output_dir / "eta_fast_profile.csv", index=False)
    eta_fast_profile.optimizer_runs.to_csv(
        output_dir / "eta_fast_profile_optimizer_runs.csv", index=False
    )
    _make_charts(calibration, pricing, eta_fast_profile, chart_dir)
    summary = _summary(calibration, pricing, eta_fast_profile)
    with (output_dir / "demo_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    pd.DataFrame([calibration.metrics]).to_csv(
        output_dir / "calibration_metrics.csv", index=False
    )
    return DemoResult(
        calibration=calibration,
        pricing=pricing,
        eta_fast_profile=eta_fast_profile,
        summary=summary,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--valuation-date",
        default=str(EVALUATION_DATE.date()),
        help="Required market-data date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=CALIBRATION_DATES,
        help="Number of usable trading dates, including the valuation date",
    )
    parser.add_argument(
        "--futures-contract",
        default="IM2609",
        help="CFFEX index-futures contract code, for example IM2612",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated CSV, JSON, and chart outputs",
    )
    arguments = parser.parse_args()
    result = run_demo(
        valuation_date=arguments.valuation_date,
        sample_size=arguments.sample_size,
        futures_contract=arguments.futures_contract,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
