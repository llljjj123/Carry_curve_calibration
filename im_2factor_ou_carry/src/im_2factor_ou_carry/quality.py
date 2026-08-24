"""Data-quality flags, exclusions, and implied-carry construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .calendar import trading_days_between


def _append_flag(flags: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    result = flags.copy()
    result.loc[mask] = result.loc[mask].map(lambda x: f"{x};{label}" if x else label)
    return result


def prepare_implied_carry(
    market_panel: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate carry, returning accepted observations and a complete audit."""
    panel = market_panel.copy()
    panel["quality_flags"] = ""
    panel["exclusion_reasons"] = ""
    hard_exclude = pd.Series(False, index=panel.index)

    if "spot_duplicate" in panel:
        spot_duplicate = panel["spot_duplicate"].fillna(False).astype(bool)
        panel["quality_flags"] = _append_flag(panel["quality_flags"], spot_duplicate, "duplicate_spot_date")

    missing = panel[["date", "contract", "expiry", "spot", "futures_price", "risk_free_rate"]].isna().any(axis=1)
    panel["exclusion_reasons"] = _append_flag(panel["exclusion_reasons"], missing, "missing_required")
    hard_exclude |= missing

    nonpositive = (panel["spot"] <= 0) | (panel["futures_price"] <= 0)
    nonpositive = nonpositive.fillna(False)
    panel["exclusion_reasons"] = _append_flag(panel["exclusion_reasons"], nonpositive, "nonpositive_price")
    hard_exclude |= nonpositive

    duplicates = panel.duplicated(["date", "contract"], keep=False)
    drop_duplicate = panel.duplicated(["date", "contract"], keep="first")
    panel["quality_flags"] = _append_flag(panel["quality_flags"], duplicates, "duplicate_key")
    panel["exclusion_reasons"] = _append_flag(panel["exclusion_reasons"], drop_duplicate, "duplicate_extra")
    hard_exclude |= drop_duplicate

    inconsistent_contracts = panel.groupby("contract", dropna=False)["expiry"].nunique(dropna=True)
    inconsistent_codes = inconsistent_contracts[inconsistent_contracts > 1].index
    inconsistent = panel["contract"].isin(inconsistent_codes)
    panel["exclusion_reasons"] = _append_flag(panel["exclusion_reasons"], inconsistent, "inconsistent_expiry")
    hard_exclude |= inconsistent

    calendar_cfg = config["calendar"]
    closures = calendar_cfg.get("special_exchange_closures", [])
    periods = int(calendar_cfg.get("periods_per_year", 244))
    valid_dates = panel["date"].notna() & panel["expiry"].notna()
    panel["sessions_to_expiry"] = np.nan
    panel.loc[valid_dates, "sessions_to_expiry"] = [
        trading_days_between(start, end, closures)
        for start, end in zip(panel.loc[valid_dates, "date"], panel.loc[valid_dates, "expiry"], strict=True)
    ]
    panel["tau"] = panel["sessions_to_expiry"] / periods
    min_sessions = int(calendar_cfg.get("min_sessions_to_expiry", 5))
    too_close = panel["sessions_to_expiry"].notna() & (panel["sessions_to_expiry"] <= min_sessions)
    panel["exclusion_reasons"] = _append_flag(panel["exclusion_reasons"], too_close, "near_or_after_expiry")
    hard_exclude |= too_close

    calculation_ok = ~(missing | nonpositive | too_close)
    panel["implied_carry"] = np.nan
    panel.loc[calculation_ok, "implied_carry"] = (
        panel.loc[calculation_ok, "risk_free_rate"]
        - np.log(panel.loc[calculation_ok, "futures_price"] / panel.loc[calculation_ok, "spot"])
        / panel.loc[calculation_ok, "tau"]
    )
    extreme_limit = float(config["quality"].get("max_abs_implied_carry", 0.50))
    extreme = panel["implied_carry"].abs() > extreme_limit
    extreme = extreme.fillna(False)
    panel["quality_flags"] = _append_flag(panel["quality_flags"], extreme, "extreme_implied_carry")
    panel["exclusion_reasons"] = _append_flag(panel["exclusion_reasons"], extreme, "extreme_implied_carry")
    hard_exclude |= extreme

    panel["_hard_exclude"] = hard_exclude
    panel = panel.sort_values(["contract", "date"]).reset_index(drop=True)
    unchanged = panel.groupby("contract", dropna=False)["futures_price"].diff().eq(0)
    groups = (~unchanged).groupby(panel["contract"], dropna=False).cumsum()
    run_length = unchanged.groupby([panel["contract"], groups], dropna=False).transform("sum") + 1
    stale_threshold = int(config["quality"].get("stale_run_length", 3))
    stale = unchanged & (run_length >= stale_threshold)
    panel["quality_flags"] = _append_flag(panel["quality_flags"], stale, "stale_price_run")
    if bool(config["quality"].get("exclude_stale", False)):
        panel["exclusion_reasons"] = _append_flag(panel["exclusion_reasons"], stale, "stale_price_run")
        panel["_hard_exclude"] |= stale

    panel["excluded"] = panel.pop("_hard_exclude") | panel["implied_carry"].isna()
    panel = panel.sort_values(["date", "contract"]).reset_index(drop=True)
    accepted = panel.loc[~panel["excluded"]].copy()
    return accepted, panel
