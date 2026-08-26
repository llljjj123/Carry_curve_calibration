"""Strict CFFEX expiry and trading-session utilities for the Demo.

The installed ``chinese_calendar`` release covers 2004--2026.  The Demo ships
an explicit company calendar for 2027--2028 and refuses to silently treat an
unsupported future year as ordinary weekdays.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
from chinese_calendar import is_holiday


DEMO_ROOT = Path(__file__).resolve().parent
EXPLICIT_CALENDAR_PATH = DEMO_ROOT / "data" / "china_exchange_calendar_2027_2028.csv"
DEFAULT_SPECIAL_CLOSURES = {date(2024, 2, 9)}
CONTRACT_PATTERN = re.compile(r"^(IM|IF|IC|IH)(\d{2})(0[1-9]|1[0-2])$")


class CalendarCoverageError(ValueError):
    """Raised when no verified calendar covers a requested date."""


def _as_date(value: object) -> date:
    return pd.Timestamp(value).date()


@lru_cache(maxsize=1)
def _explicit_calendar() -> dict[date, bool]:
    frame = pd.read_csv(EXPLICIT_CALENDAR_PATH)
    required = {"date", "is_trading_day", "source"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Calendar missing columns: {sorted(required - set(frame.columns))}")
    dates = pd.to_datetime(frame["date"], errors="raise").dt.date
    values = frame["is_trading_day"].astype(str).str.lower().map({"true": True, "false": False})
    if values.isna().any() or dates.duplicated().any():
        raise ValueError("Explicit calendar contains invalid or duplicate rows")
    return dict(zip(dates, values.astype(bool), strict=True))


def explicit_calendar_years() -> tuple[int, ...]:
    return tuple(sorted({day.year for day in _explicit_calendar()}))


@lru_cache(maxsize=None)
def is_trading_session(day: date, special_closures: tuple[date, ...] = ()) -> bool:
    closures = DEFAULT_SPECIAL_CLOSURES.union(special_closures)
    if day in closures or day.weekday() >= 5:
        return False
    explicit = _explicit_calendar()
    if day in explicit:
        return explicit[day]
    if day.year in explicit_calendar_years():
        raise CalendarCoverageError(f"Explicit calendar has no row for {day}")
    try:
        return not bool(is_holiday(day))
    except NotImplementedError as exc:
        raise CalendarCoverageError(
            f"No verified trading calendar covers {day.year}; weekday fallback is disabled"
        ) from exc


def trading_days_between(
    start: object,
    end: object,
    special_closures: Iterable[object] = (),
) -> int:
    """Count verified trading sessions over ``(start, end]``."""
    start_day, end_day = _as_date(start), _as_date(end)
    if start_day == end_day:
        return 0
    sign = 1
    if start_day > end_day:
        start_day, end_day = end_day, start_day
        sign = -1
    closures = tuple(sorted(_as_date(value) for value in special_closures))
    count = 0
    cursor = start_day + timedelta(days=1)
    while cursor <= end_day:
        count += int(is_trading_session(cursor, closures))
        cursor += timedelta(days=1)
    return sign * count


def third_friday(year: int, month: int) -> date:
    month_calendar = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in month_calendar if week[calendar.FRIDAY]]
    return date(year, month, fridays[2])


def contract_expiry(contract: str, special_closures: Iterable[object] = ()) -> date:
    """Infer CFFEX stock-index-futures expiry and holiday-shift it strictly."""
    code = str(contract).strip().upper()
    match = CONTRACT_PATTERN.fullmatch(code)
    if match is None:
        raise ValueError(f"Invalid CFFEX stock-index futures contract: {contract!r}")
    year = 2000 + int(match.group(2))
    month = int(match.group(3))
    expiry = third_friday(year, month)
    closures = tuple(sorted(_as_date(value) for value in special_closures))
    while not is_trading_session(expiry, closures):
        expiry += timedelta(days=1)
    return expiry


def calendar_source_for_date(value: object) -> str:
    day = _as_date(value)
    if day in _explicit_calendar():
        return "demo_company_calendar_2027_2028"
    try:
        is_holiday(day)
    except NotImplementedError:
        return "unsupported"
    return "chinese_calendar"


def maturity_diagnostics(contract: str, valuation_date: object) -> dict[str, object]:
    expiry = contract_expiry(contract)
    return {
        "contract": str(contract).strip().upper(),
        "valuation_date": str(pd.Timestamp(valuation_date).date()),
        "inferred_expiry": str(expiry),
        "sessions_to_expiry": trading_days_between(valuation_date, expiry),
        "expiry_calendar_source": calendar_source_for_date(expiry),
        "weekday_fallback_used": False,
    }
