"""Carry-put optional-component pricing and convergence helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import pandas as pd

from calibration import CalibrationResult, PERIODS_PER_YEAR, RISK_FREE_RATE
from calendar_utils import contract_expiry, trading_days_between


DEMO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEMO_ROOT.parent
PRICING_SRC = WORKSPACE_ROOT / "carry_put_pricing" / "src"
if str(PRICING_SRC) not in sys.path:
    sys.path.insert(0, str(PRICING_SRC))

from carry_put_pricing import (  # noqa: E402
    CarryPutContract,
    FactorState,
    GBMParams,
    HedgeFuturesContract,
    NumericalConfig,
    PricingResult,
    TwoFactorOUParams,
    price_american_carry_put,
)


@dataclass
class OptionPricingBundle:
    """Base option price plus numerical diagnostics."""

    contract: CarryPutContract
    ou_params: TwoFactorOUParams
    initial_state: FactorState
    gbm_params: GBMParams
    hedge_futures: tuple[HedgeFuturesContract, HedgeFuturesContract]
    base_result: PricingResult
    grid_convergence: pd.DataFrame
    quadrature_convergence: pd.DataFrame
    exercise_summary: pd.DataFrame


def hedge_futures_inputs(
    calibration: CalibrationResult,
    contract_codes: tuple[str, str],
) -> tuple[HedgeFuturesContract, HedgeFuturesContract]:
    """Select observed hedge quotes and apply the strict Demo maturity calendar."""
    if len(contract_codes) != 2:
        raise ValueError("Exactly two hedge futures contracts are required")
    valuation_date = pd.Timestamp(calibration.quote["date"]).normalize()
    instruments = []
    for raw_code in contract_codes:
        code = str(raw_code).strip().upper()
        rows = calibration.fitted_panel.loc[
            (pd.to_datetime(calibration.fitted_panel["date"]) == valuation_date)
            & (calibration.fitted_panel["contract"].astype(str).str.upper() == code)
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one accepted {code} hedge quote on "
                f"{valuation_date.date()}, found {len(rows)}"
            )
        quote = rows.iloc[0]
        expiry = pd.Timestamp(contract_expiry(code))
        sessions = trading_days_between(valuation_date, expiry)
        if pd.Timestamp(quote["expiry"]) != expiry:
            raise AssertionError(
                f"{code} quote expiry {quote['expiry']} differs from strict-calendar "
                f"expiry {expiry.date()}"
            )
        if int(quote["sessions_to_expiry"]) != sessions:
            raise AssertionError(
                f"{code} quote maturity {quote['sessions_to_expiry']} differs from "
                f"strict-calendar maturity {sessions}"
            )
        instruments.append(
            HedgeFuturesContract(
                contract=code,
                futures_price=float(quote["futures_price"]),
                sessions_to_expiry=sessions,
                periods_per_year=PERIODS_PER_YEAR,
            )
        )
    return instruments[0], instruments[1]


def pricing_inputs(
    calibration: CalibrationResult,
) -> tuple[CarryPutContract, TwoFactorOUParams, FactorState, GBMParams]:
    quote = calibration.quote
    fitted = calibration.estimate.params
    state = calibration.latest_state
    contract = CarryPutContract(
        initial_spot=float(quote["spot"]),
        initial_futures=float(quote["futures_price"]),
        sessions_to_expiry=int(quote["sessions_to_expiry"]),
        periods_per_year=PERIODS_PER_YEAR,
    )
    ou_params = TwoFactorOUParams(
        kappa_slow=float(fitted.kappa_slow),
        kappa_fast=float(fitted.kappa_fast),
        theta=float(fitted.theta),
        eta_slow=float(fitted.eta_slow),
        eta_fast=float(fitted.eta_fast),
    )
    initial_state = FactorState(
        slow=float(state["filtered_slow_state"]),
        fast=float(state["filtered_fast_state"]),
    )
    gbm_params = GBMParams(
        risk_free_rate=RISK_FREE_RATE,
        volatility=calibration.historical_volatility,
    )
    return contract, ou_params, initial_state, gbm_params


def price_from_parameters(
    calibration: CalibrationResult,
    fitted_params: object,
    latest_state: pd.Series,
    *,
    numerical: NumericalConfig | None = None,
) -> PricingResult:
    """Price from an alternative calibration, used by the fixed-eta profile."""
    quote = calibration.quote
    contract = CarryPutContract(
        initial_spot=float(quote["spot"]),
        initial_futures=float(quote["futures_price"]),
        sessions_to_expiry=int(quote["sessions_to_expiry"]),
        periods_per_year=PERIODS_PER_YEAR,
    )
    params = TwoFactorOUParams(
        kappa_slow=float(fitted_params.kappa_slow),
        kappa_fast=float(fitted_params.kappa_fast),
        theta=float(fitted_params.theta),
        eta_slow=float(fitted_params.eta_slow),
        eta_fast=float(fitted_params.eta_fast),
    )
    state = FactorState(
        slow=float(latest_state["filtered_slow_state"]),
        fast=float(latest_state["filtered_fast_state"]),
    )
    gbm = GBMParams(
        risk_free_rate=RISK_FREE_RATE,
        volatility=calibration.historical_volatility,
    )
    return price_american_carry_put(
        contract,
        params,
        state,
        gbm,
        numerical=numerical or NumericalConfig(),
    )


def price_optional_component(
    calibration: CalibrationResult,
    hedge_futures_contracts: tuple[str, str] = ("IM2609", "IM2703"),
) -> OptionPricingBundle:
    """Price only the American carry-put option, excluding the linear IM leg."""
    contract, params, state, gbm = pricing_inputs(calibration)
    hedge_futures = hedge_futures_inputs(calibration, hedge_futures_contracts)
    configurations = {
        "coarse": NumericalConfig(slow_grid_points=201, fast_grid_points=281),
        "base": NumericalConfig(),
        "fine": NumericalConfig(slow_grid_points=401, fast_grid_points=501),
    }
    results = {
        name: price_american_carry_put(
            contract,
            params,
            state,
            gbm,
            numerical=config,
            hedge_futures=hedge_futures if name == "base" else None,
        )
        for name, config in configurations.items()
    }
    base = results["base"]
    grid_rows = []
    for name, result in results.items():
        row = {"configuration": name, **asdict(configurations[name])}
        row.update(
            price=result.price,
            normalized_price=result.normalized_price,
            difference_from_fine=result.price - results["fine"].price,
        )
        grid_rows.append(row)
    grid_convergence = pd.DataFrame(grid_rows)

    quadrature_results: dict[int, PricingResult] = {43: base}
    for order in (39, 41, 45, 47):
        numerical = NumericalConfig(quadrature_order=order)
        quadrature_results[order] = price_american_carry_put(
            contract, params, state, gbm, numerical=numerical
        )
    quadrature_convergence = pd.DataFrame(
        [
            {
                "quadrature_order": order,
                "price": result.price,
                "normalized_price": result.normalized_price,
                "difference_from_base_order": result.price - base.price,
            }
            for order, result in sorted(quadrature_results.items())
        ]
    )
    exercise_summary = pd.DataFrame([asdict(row) for row in base.exercise_summary])
    return OptionPricingBundle(
        contract=contract,
        ou_params=params,
        initial_state=state,
        gbm_params=gbm,
        hedge_futures=hedge_futures,
        base_result=base,
        grid_convergence=grid_convergence,
        quadrature_convergence=quadrature_convergence,
        exercise_summary=exercise_summary,
    )


def export_pricing(bundle: OptionPricingBundle, output_dir: Path) -> None:
    """Write option-only pricing and convergence outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([bundle.base_result.as_dict(include_exercise_summary=False)]).to_csv(
        output_dir / "option_price.csv", index=False
    )
    bundle.grid_convergence.to_csv(output_dir / "grid_convergence.csv", index=False)
    bundle.quadrature_convergence.to_csv(
        output_dir / "quadrature_convergence.csv", index=False
    )
    bundle.exercise_summary.to_csv(output_dir / "exercise_summary.csv", index=False)
    joint_hedge = bundle.base_result.two_futures_hedge
    if joint_hedge is None:  # pragma: no cover - base pricing always requests it
        raise AssertionError("Base result is missing the requested two-futures hedge")
    pd.DataFrame([asdict(joint_hedge)]).to_csv(
        output_dir / "two_futures_hedge.csv", index=False
    )
