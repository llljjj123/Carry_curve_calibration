"""Monthly carry-put cohorts; importing this package never runs a back-test."""

from pathlib import Path
import sys

# Reuse the repository's standalone calibration, calendar, and pricing projects.
_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "Demo", _ROOT / "carry_put_pricing" / "src",
              _ROOT / "im_2factor_ou_carry" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from .engine import (
    BacktestConfig, BacktestResult, SHORT_SCENARIOS, STUDIES, ScenarioSpec,
    run_backtest, scenario_specs,
)
from .market import MarketData, load_cached_market, monthly_cohorts

__all__ = ["BacktestConfig", "BacktestResult", "MarketData", "load_cached_market",
           "monthly_cohorts", "run_backtest", "ScenarioSpec", "STUDIES",
           "SHORT_SCENARIOS", "scenario_specs"]
