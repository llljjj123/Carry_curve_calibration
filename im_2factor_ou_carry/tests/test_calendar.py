from datetime import date

import pytest

from im_2factor_ou_carry.calendar import contract_expiry, third_friday, trading_days_between, year_fraction


def test_third_friday_and_holiday_shift() -> None:
    assert third_friday(2024, 3) == date(2024, 3, 15)
    # 2024-02-16 was Spring Festival; CFFEX expiry moved to the next session.
    assert contract_expiry("IM2402") == date(2024, 2, 19)


def test_trading_day_interval_is_start_exclusive() -> None:
    assert trading_days_between("2024-03-14", "2024-03-15") == 1
    assert trading_days_between("2024-03-15", "2024-03-15") == 0
    assert trading_days_between("2024-03-15", "2024-03-18") == 1
    assert year_fraction("2024-03-14", "2024-03-15") == pytest.approx(1 / 244)
