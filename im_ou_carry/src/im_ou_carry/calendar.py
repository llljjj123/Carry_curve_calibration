"""CFFEX-oriented trading-session and contract-expiry utilities.

The supplied ``workdays_count.py`` counts Chinese weekdays that are not public
holidays and makes a special correction for 2024-02-09.  This module follows
that convention while using the mathematically useful interval ``(start, end]``:
there are zero sessions remaining on the expiry date itself.
"""

from __future__ import annotations

import calendar as _calendar
from datetime import date, timedelta
from functools import lru_cache
from typing import Iterable

import pandas as pd

try:
    from chinese_calendar import is_holiday
except ImportError:  # pragma: no cover - exercised only in minimal environments
    is_holiday = None


DEFAULT_SPECIAL_CLOSURES = {date(2024, 2, 9)}


def _as_date(value: object) -> date:
    return pd.Timestamp(value).date()


@lru_cache(maxsize=None)
def is_trading_session(day: date, special_closures: tuple[date, ...] = ()) -> bool:
    """Return whether *day* is an exchange session under the configured rule."""
    closures = DEFAULT_SPECIAL_CLOSURES.union(special_closures)
    if day in closures or day.weekday() >= 5:
        return False
    if is_holiday is None:
        return True
    try:
        return not bool(is_holiday(day))
    except NotImplementedError:
        # The holiday package may not yet cover distant future years.
        return True


def trading_days_between(
    start: object,
    end: object,
    special_closures: Iterable[object] = (),
) -> int:
    """Count trading sessions in ``(start, end]``; return a signed count."""
    start_day, end_day = _as_date(start), _as_date(end)
    if start_day == end_day:
        return 0
    sign = 1
    if start_day > end_day:
        start_day, end_day = end_day, start_day
        sign = -1
    closures = tuple(sorted(_as_date(x) for x in special_closures))
    count = 0
    cursor = start_day + timedelta(days=1)
    while cursor <= end_day:
        if is_trading_session(cursor, closures):
            count += 1
        cursor += timedelta(days=1)
    return sign * count


def third_friday(year: int, month: int) -> date:
    """Return the calendar third Friday of a month."""
    month_calendar = _calendar.monthcalendar(year, month)
    fridays = [week[_calendar.FRIDAY] for week in month_calendar if week[_calendar.FRIDAY]]
    return date(year, month, fridays[2])


def contract_expiry(contract: str, special_closures: Iterable[object] = ()) -> date:
    """Infer an IM/IF/IC/IH expiry and shift a holiday to the next session."""
    code = contract.strip().upper()
    if len(code) < 6 or not code[-4:].isdigit():
        raise ValueError(f"Cannot infer expiry from contract code: {contract!r}")
    year = 2000 + int(code[-4:-2])
    month = int(code[-2:])
    expiry = third_friday(year, month)
    closures = tuple(sorted(_as_date(x) for x in special_closures))
    while not is_trading_session(expiry, closures):
        expiry += timedelta(days=1)
    return expiry


def year_fraction(start: object, end: object, periods_per_year: int = 244, **kwargs: object) -> float:
    """Trading-session year fraction under the configured 244-day scheme."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return trading_days_between(start, end, **kwargs) / float(periods_per_year)

