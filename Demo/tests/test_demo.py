from __future__ import annotations

from math import isclose, log, sqrt
from pathlib import Path
import sys

import numpy as np


DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from calibration import (  # noqa: E402
    CALIBRATION_DATES,
    CONTRACT_CODE,
    EVALUATION_DATE,
    PERIODS_PER_YEAR,
    RISK_FREE_RATE,
    estimate_historical_volatility,
    load_calibration_sample,
)
from calendar_utils import (  # noqa: E402
    CalendarCoverageError,
    contract_expiry,
    maturity_diagnostics,
    trading_days_between,
)


def test_fixed_244_date_sample_and_im2609_quote() -> None:
    sample, audit, spot = load_calibration_sample()
    assert sample["date"].nunique() == CALIBRATION_DATES
    assert len(spot) == CALIBRATION_DATES
    assert sample["date"].max() == EVALUATION_DATE
    quote = sample.loc[
        (sample["date"] == EVALUATION_DATE) & (sample["contract"] == CONTRACT_CODE)
    ].iloc[0]
    assert quote["spot"] == 7601.804
    assert quote["futures_price"] == 7527.0
    assert quote["sessions_to_expiry"] == 20
    assert audit["excluded"].sum() == 72


def test_flexible_valuation_date_and_250_date_sample() -> None:
    evaluation_date = np.datetime64("2026-08-10")
    sample, _, spot = load_calibration_sample(evaluation_date, 250)
    assert sample["date"].nunique() == 250
    assert len(spot) == 250
    assert sample["date"].max() == evaluation_date
    assert spot["log_return"].notna().sum() == 249
    quote = sample.loc[
        (sample["date"] == evaluation_date) & (sample["contract"] == CONTRACT_CODE)
    ]
    assert len(quote) == 1


def test_invalid_sample_size_is_rejected() -> None:
    for invalid in (0, 1, 2.5, True):
        try:
            load_calibration_sample(EVALUATION_DATE, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for sample size {invalid!r}")


def test_contract_expiry_and_explicit_2027_calendar() -> None:
    assert str(contract_expiry("IM2612")) == "2026-12-18"
    assert str(contract_expiry("im2703")) == "2027-03-19"
    diagnostics = maturity_diagnostics("IM2703", "2026-08-21")
    assert diagnostics["sessions_to_expiry"] == 138
    assert diagnostics["expiry_calendar_source"] == "demo_company_calendar_2027_2028"
    assert diagnostics["weekday_fallback_used"] is False
    assert trading_days_between("2026-08-21", "2027-03-19") == 138
    with np.testing.assert_raises(CalendarCoverageError):
        contract_expiry("IM2903")


def test_historical_volatility_and_locked_carry() -> None:
    sample, _, spot = load_calibration_sample()
    sigma = estimate_historical_volatility(spot)
    manual_sigma = spot["log_return"].dropna().std(ddof=1) * sqrt(PERIODS_PER_YEAR)
    assert isclose(sigma, manual_sigma, rel_tol=0.0, abs_tol=1.0e-14)
    assert isclose(sigma, 0.25997649376789345, rel_tol=0.0, abs_tol=1.0e-12)

    quote = sample.loc[
        (sample["date"] == EVALUATION_DATE) & (sample["contract"] == CONTRACT_CODE)
    ].iloc[0]
    maturity = quote["sessions_to_expiry"] / PERIODS_PER_YEAR
    locked = RISK_FREE_RATE - log(quote["futures_price"] / quote["spot"]) / maturity
    assert isclose(locked, 0.13464618422090385, rel_tol=0.0, abs_tol=1.0e-12)
    assert np.isfinite(locked)
