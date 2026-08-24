"""AkShare data acquisition, caching, and schema normalization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .calendar import contract_expiry


SPOT_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
FUTURES_COLUMNS = ["date", "open", "high", "low", "close", "volume", "hold", "settle"]


def _month_add(value: pd.Timestamp, months: int) -> pd.Timestamp:
    return value + pd.DateOffset(months=months)


def contract_universe(prefix: str, start: object, end: object) -> list[str]:
    """Generate all historical monthly codes plus currently listed quarters."""
    first = pd.Timestamp(start).to_period("M").to_timestamp()
    last = pd.Timestamp(end).to_period("M").to_timestamp()
    months: set[tuple[int, int]] = set()
    cursor = first
    while cursor <= _month_add(last, 1):
        months.add((cursor.year, cursor.month))
        cursor = _month_add(cursor, 1)
    cursor = last
    while cursor <= _month_add(last, 12):
        if cursor.month in (3, 6, 9, 12):
            months.add((cursor.year, cursor.month))
        cursor = _month_add(cursor, 1)
    return [f"{prefix.upper()}{year % 100:02d}{month:02d}" for year, month in sorted(months)]


def _retry(call: Callable[[], pd.DataFrame], retries: int) -> pd.DataFrame:
    error: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            return call()
        except Exception as exc:  # network/provider failures are recorded by caller
            error = exc
            if attempt < retries:
                time.sleep(min(2.0, 0.25 * (2**attempt)))
    assert error is not None
    raise error


def download_akshare(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download spot and every candidate IM contract, returning an audit log."""
    import akshare as ak

    data_cfg = config["data"]
    retries = int(data_cfg.get("request_retries", 2))
    spot = _retry(lambda: ak.stock_zh_index_daily(symbol=data_cfg["index_symbol"]), retries)
    spot["date"] = pd.to_datetime(spot["date"], errors="coerce")
    start = pd.Timestamp(data_cfg["start_date"])
    configured_end = data_cfg.get("end_date")
    available_end = spot["date"].max()
    end = min(pd.Timestamp(configured_end), available_end) if configured_end else available_end
    spot = spot.loc[spot["date"].between(start, end)].copy()

    frames: list[pd.DataFrame] = []
    logs: list[dict[str, object]] = []
    closures = config["calendar"].get("special_exchange_closures", [])
    for code in contract_universe(data_cfg["futures_prefix"], start, end):
        try:
            frame = _retry(lambda code=code: ak.futures_zh_daily_sina(symbol=code), retries)
            if frame is None or frame.empty:
                logs.append({"contract": code, "status": "empty", "rows": 0, "message": ""})
                continue
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["contract"] = code
            frame["expiry"] = pd.Timestamp(contract_expiry(code, closures))
            frame["expiry_source"] = "contract_code_cffex_rule"
            frames.append(frame)
            logs.append({"contract": code, "status": "ok", "rows": len(frame), "message": ""})
        except Exception as exc:
            logs.append({"contract": code, "status": "error", "rows": 0, "message": f"{type(exc).__name__}: {exc}"})
    futures = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    override_path = data_cfg.get("expiry_override_path")
    if override_path and not futures.empty:
        overrides = pd.read_csv(override_path)
        required = {"contract", "expiry"}
        if not required.issubset(overrides.columns):
            raise ValueError("Expiry override CSV must contain contract and expiry columns")
        mapping = overrides.assign(
            contract=overrides["contract"].astype(str).str.upper().str.strip(),
            expiry=pd.to_datetime(overrides["expiry"], errors="raise").dt.normalize(),
        ).set_index("contract")["expiry"]
        mask = futures["contract"].isin(mapping.index)
        futures.loc[mask, "expiry"] = futures.loc[mask, "contract"].map(mapping)
        futures.loc[mask, "expiry_source"] = "override_csv"
    log = pd.DataFrame(logs)
    return spot, futures, log


def _read_cached(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spot = pd.read_csv(raw_dir / "spot_raw.csv", parse_dates=["date"])
    futures = pd.read_csv(raw_dir / "futures_raw.csv", parse_dates=["date", "expiry"])
    log_path = raw_dir / "download_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    return spot, futures, log


def acquire_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the cache unless refresh is requested; otherwise download and snapshot."""
    raw_dir = Path(config["project"]["raw_dir"])
    spot_path, futures_path = raw_dir / "spot_raw.csv", raw_dir / "futures_raw.csv"
    refresh = bool(config["data"].get("refresh", False))
    if not refresh and spot_path.exists() and futures_path.exists():
        return _read_cached(raw_dir)
    spot, futures, log = download_akshare(config)
    if spot.empty or futures.empty:
        raise RuntimeError("AkShare returned no usable spot or futures data; see download log")
    raw_dir.mkdir(parents=True, exist_ok=True)
    spot.to_csv(spot_path, index=False, encoding="utf-8-sig")
    futures.to_csv(futures_path, index=False, encoding="utf-8-sig")
    log.to_csv(raw_dir / "download_log.csv", index=False, encoding="utf-8-sig")
    return spot, futures, log


def normalized_market_panel(
    spot: pd.DataFrame,
    futures: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Normalize the supplied notebook schemas without silently dropping rows."""
    price_field = config["data"].get("futures_price_field", "close")
    if price_field != "close":
        raise ValueError("This project is configured to use futures close prices")
    required_spot = {"date", "close"}
    required_futures = {"date", "contract", "expiry", price_field}
    if not required_spot.issubset(spot.columns):
        raise ValueError(f"Spot data missing columns: {sorted(required_spot - set(spot.columns))}")
    if not required_futures.issubset(futures.columns):
        raise ValueError(f"Futures data missing columns: {sorted(required_futures - set(futures.columns))}")

    spot_norm = spot[["date", "close"]].rename(columns={"close": "spot"}).copy()
    spot_norm["date"] = pd.to_datetime(spot_norm["date"], errors="coerce").dt.normalize()
    spot_norm["spot_duplicate"] = spot_norm.duplicated("date", keep=False)
    spot_norm = spot_norm.drop_duplicates("date", keep="last")

    keep = [c for c in ["date", "contract", "expiry", price_field, "volume", "hold"] if c in futures]
    fut = futures[keep].rename(columns={price_field: "futures_price"}).copy()
    fut["date"] = pd.to_datetime(fut["date"], errors="coerce").dt.normalize()
    fut["expiry"] = pd.to_datetime(fut["expiry"], errors="coerce").dt.normalize()
    fut["contract"] = fut["contract"].astype("string").str.upper().str.strip()
    fut["price_source"] = "close"
    panel = fut.merge(spot_norm, on="date", how="left", validate="many_to_one")
    panel["risk_free_rate"] = float(config["data"].get("risk_free_rate", 0.014))
    return panel.sort_values(["date", "contract"], na_position="last").reset_index(drop=True)
