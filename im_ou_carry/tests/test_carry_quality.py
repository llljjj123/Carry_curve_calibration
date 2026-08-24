import numpy as np
import pandas as pd
import pytest

from im_ou_carry.quality import prepare_implied_carry


def _config() -> dict:
    return {
        "calendar": {
            "periods_per_year": 244,
            "min_sessions_to_expiry": 5,
            "special_exchange_closures": ["2024-02-09"],
        },
        "quality": {"max_abs_implied_carry": 0.50, "stale_run_length": 3, "exclude_stale": False},
    }


def test_implied_carry_uses_trading_day_tau_and_close_price() -> None:
    target_carry = 0.035
    tau = 10 / 244
    spot = 5000.0
    rate = 0.014
    futures = spot * np.exp((rate - target_carry) * tau)
    raw = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-03-01")],
            "contract": ["IM2403"],
            "expiry": [pd.Timestamp("2024-03-15")],
            "spot": [spot],
            "futures_price": [futures],
            "risk_free_rate": [rate],
        }
    )
    accepted, audit = prepare_implied_carry(raw, _config())
    assert len(accepted) == 1
    assert accepted.iloc[0]["sessions_to_expiry"] == 10
    assert accepted.iloc[0]["implied_carry"] == pytest.approx(target_carry)
    assert not bool(audit.iloc[0]["excluded"])


def test_quality_exclusions_are_reported() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-14", "2024-03-14"]),
            "contract": ["IM2403", "IM2403"],
            "expiry": pd.to_datetime(["2024-03-15", "2024-03-15"]),
            "spot": [5000.0, 5000.0],
            "futures_price": [4999.0, 4999.0],
            "risk_free_rate": [0.014, 0.014],
        }
    )
    accepted, audit = prepare_implied_carry(raw, _config())
    assert accepted.empty
    assert audit["excluded"].all()
    assert audit["quality_flags"].str.contains("duplicate_key").all()
    assert audit["exclusion_reasons"].str.contains("near_or_after_expiry").all()

