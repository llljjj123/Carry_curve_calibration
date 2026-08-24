from datetime import date

import numpy as np
import pandas as pd
import pytest

from im_corr_ou_1factor.calendar import contract_expiry, trading_days_between
from im_corr_ou_1factor.quality import prepare_implied_carry


def config() -> dict:
    return {
        "calendar": {"periods_per_year": 244, "min_sessions_to_expiry": 5, "special_exchange_closures": ["2024-02-09"]},
        "quality": {"max_abs_implied_carry": 0.50, "stale_run_length": 3, "exclude_stale": False},
    }


def test_trading_session_and_expiry_conventions() -> None:
    assert trading_days_between("2024-03-14", "2024-03-15") == 1
    assert trading_days_between("2024-03-15", "2024-03-15") == 0
    assert contract_expiry("IM2402") == date(2024, 2, 19)


def test_implied_carry_uses_close_formula_and_audits_exclusions() -> None:
    target, spot, rate, tau = 0.035, 5000.0, 0.014, 10 / 244
    futures = spot * np.exp((rate - target) * tau)
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-01", "2024-03-14"]),
        "contract": ["IM2403", "IM2403"],
        "expiry": pd.to_datetime(["2024-03-15", "2024-03-15"]),
        "spot": [spot, spot],
        "futures_price": [futures, futures],
        "risk_free_rate": [rate, rate],
    })
    accepted, audit = prepare_implied_carry(raw, config())
    assert len(accepted) == 1
    assert accepted.iloc[0]["implied_carry"] == pytest.approx(target)
    assert audit["excluded"].sum() == 1
    assert "near_or_after_expiry" in audit.loc[audit["excluded"], "exclusion_reasons"].iloc[0]

