"""Strict-calendar cohort scheduling and immutable cached close inputs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_utils import contract_expiry, is_trading_session
from calibration import demo_config
from demo_quality import prepare_implied_carry


class CohortUnavailable(ValueError):
    """An explicit data/eligibility failure, never a synthetic zero return."""


def session_dates(start: object, end: object) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([
        d for d in pd.date_range(start, end) if is_trading_session(d.date())
    ])


def monthly_cohorts(start: object, end: object) -> pd.DataFrame:
    """Exactly the first session after each monthly expiry, never a catch-up date."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start > end:
        raise ValueError("start must not be after end")
    rows = []
    for month in pd.period_range(start.to_period("M") - 1, end.to_period("M"), freq="M"):
        previous_contract = "IM" + month.strftime("%y%m")
        expiry = pd.Timestamp(contract_expiry(previous_contract))
        entry = expiry + pd.Timedelta(days=1)
        while not is_trading_session(entry.date()):
            entry += pd.Timedelta(days=1)
        if start <= entry <= end:
            option_contract = "IM" + (month + 1).strftime("%y%m")
            rows.append({
                "cohort": entry.strftime("%Y-%m-%d"), "entry_date": entry,
                "previous_expiry": expiry, "option_contract": option_contract,
                "expiry": pd.Timestamp(contract_expiry(option_contract)),
            })
    return pd.DataFrame(rows, columns=[
        "cohort", "entry_date", "previous_expiry", "option_contract", "expiry",
    ])


@dataclass
class MarketData:
    """Raw marks remain separate from observations accepted for calibration.

    Required columns: spot (date, spot), futures (date, contract, futures_price).
    Missing/invalid prices are retained and cause explicit cohort failures when
    needed. No interpolation, forward filling, or settlement substitution.
    """

    spot: pd.DataFrame
    futures: pd.DataFrame
    risk_free_rate: float = 0.014

    def __post_init__(self) -> None:
        if not np.isfinite(self.risk_free_rate):
            raise ValueError("risk_free_rate must be finite")
        self.spot = self.spot[["date", "spot"]].copy()
        self.futures = self.futures[["date", "contract", "futures_price"]].copy()
        for frame in (self.spot, self.futures):
            frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
            if frame["date"].isna().any():
                raise ValueError("Missing market date")
        self.futures["contract"] = self.futures["contract"].astype(str).str.strip().str.upper()
        self.futures = self.futures.loc[self.futures["contract"].str.fullmatch(r"IM\d{4}")].copy()
        if self.spot.duplicated("date").any() or self.futures.duplicated(["date", "contract"]).any():
            raise ValueError("Duplicate market keys; resolve the raw input explicitly")
        self.spot["spot"] = pd.to_numeric(self.spot["spot"], errors="coerce")
        self.futures["futures_price"] = pd.to_numeric(self.futures["futures_price"], errors="coerce")
        self.spot = self.spot.sort_values("date").reset_index(drop=True)
        self.futures = self.futures.sort_values(["date", "contract"]).reset_index(drop=True)
        if self.spot.empty or self.futures.empty:
            raise ValueError("Empty market data")

    def observations_through(self, date: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply existing quality conventions only to the visible history."""
        date = pd.Timestamp(date)
        spot = self.spot.loc[self.spot.date <= date]
        futures = self.futures.loc[self.futures.date <= date].copy()
        futures["expiry"] = futures.contract.map(lambda c: pd.Timestamp(contract_expiry(c)))
        market = futures.merge(spot, on="date", how="left", validate="many_to_one")
        market["risk_free_rate"] = self.risk_free_rate
        config = demo_config()
        config["data"]["risk_free_rate"] = self.risk_free_rate
        _, audit = prepare_implied_carry(market, config)
        # Stale flags are diagnostic only; reconstruct them causally so later
        # unchanged prices cannot retroactively label an earlier observation.
        audit = audit.sort_values(["contract", "date"]).copy()
        unchanged = audit.groupby("contract").futures_price.diff().eq(0)
        groups = (~unchanged).groupby(audit.contract).cumsum()
        runs = audit.groupby([audit.contract, groups]).cumcount() + 1
        audit["quality_flags"] = audit.quality_flags.str.replace(
            r"(^|;)stale_price_run(?=;|$)", "", regex=True
        ).str.strip(";")
        stale = runs >= config["quality"]["stale_run_length"]
        audit.loc[stale, "quality_flags"] = audit.loc[stale, "quality_flags"].map(
            lambda f: f + ";stale_price_run" if f else "stale_price_run"
        )
        valid = np.isfinite(audit.spot) & np.isfinite(audit.futures_price)
        non_session = ~audit.date.map(lambda d: is_trading_session(d.date()))
        bad = ~valid | non_session
        audit.loc[bad, "excluded"] = True
        audit.loc[bad, "exclusion_reasons"] += ";nonfinite_price_or_non_session"
        audit = audit.sort_values(["date", "contract"]).reset_index(drop=True)
        return audit.loc[~audit.excluded].copy(), audit

    def quote(self, date: object, contract: str | None = None) -> float:
        date = pd.Timestamp(date)
        if contract is None:
            values = self.spot.loc[self.spot.date == date, "spot"]
        else:
            values = self.futures.loc[
                (self.futures.date == date) & (self.futures.contract == contract), "futures_price"
            ]
        if len(values) != 1 or not np.isfinite(values.iloc[0]) or values.iloc[0] <= 0:
            raise CohortUnavailable(f"Missing/invalid close: {date.date()} {contract or 'spot'}")
        return float(values.iloc[0])

    def hedge_pair(self, entry: object, underlying: str) -> tuple[str, str]:
        """Selection sees inception quotes only, never future quote availability."""
        self.quote(entry, underlying)
        expiry = contract_expiry(underlying)
        candidates = self.futures.loc[self.futures.date == pd.Timestamp(entry)]
        eligible = [c for c, p in zip(candidates.contract, candidates.futures_price)
                    if c != underlying and np.isfinite(p) and p > 0
                    and contract_expiry(c) > expiry]
        if not eligible:
            raise CohortUnavailable("No later-maturity hedge quoted at inception")
        return underlying, max(eligible, key=contract_expiry)


def load_cached_market(raw_dir: Path | str | None = None, *, risk_free_rate: float = 0.014) -> MarketData:
    raw_dir = (Path(raw_dir) if raw_dir is not None else
               Path(__file__).resolve().parents[1] / "im_2factor_ou_carry" / "data" / "raw")
    spot = pd.read_csv(raw_dir / "spot_raw.csv").rename(columns={"close": "spot"})
    futures = pd.read_csv(raw_dir / "futures_raw.csv").rename(columns={"close": "futures_price"})
    return MarketData(spot, futures, risk_free_rate)
