"""Reproducible exports; no aggregate performance is claimed before a run."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from .engine import scenario_specs


def aggregate_results(result):
    if result.summaries.empty:
        return pd.DataFrame()
    rows = []
    for scenario, summary in result.summaries.groupby("scenario", sort=False):
        daily = result.daily.loc[(result.daily.scenario == scenario) & (result.daily.elapsed_sessions > 0)]
        pnl = summary.total_pnl
        cutoff = pnl.quantile(0.05)
        row = {
            "scenario": scenario, "completed_cohorts": len(summary),
            "sum_cohort_pnl": pnl.sum(), "mean_cohort_pnl": pnl.mean(),
            "median_cohort_pnl": pnl.median(), "std_cohort_pnl": pnl.std(ddof=1),
            "worst_cohort_pnl": pnl.min(), "fifth_percentile_cohort_pnl": cutoff,
            "mean_pnl_below_fifth_percentile": pnl.loc[pnl <= cutoff].mean(),
            "daily_discounted_pnl_variance": daily.discounted_pnl_change.var(ddof=1),
            "pooled_daily_discounted_pnl_std": daily.discounted_pnl_change.std(ddof=1),
            "mean_cohort_daily_discounted_pnl_std": summary.daily_discounted_pnl_std.mean(),
            "total_costs": summary.transaction_costs.sum(),
            "total_turnover_contracts": summary.turnover_contracts.sum(),
        }
        for column in ("spot_pnl", "futures_costs", "spot_costs", "spot_turnover_notional"):
            if column in summary:
                row[f"total_{column}" if column.endswith("costs") else column] = summary[column].sum()
        for column in ("peak_borrowing", "peak_spot_notional"):
            if column in summary:
                row[column] = summary[column].max()
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    baseline = "short_no_hedge" if "short_no_hedge" in set(aggregate.scenario) else "no_hedge"
    base = aggregate.loc[aggregate.scenario == baseline, "daily_discounted_pnl_variance"].iloc[0]
    aggregate["variance_reduction_vs_no_hedge"] = (
        1 - aggregate.daily_discounted_pnl_variance / base if np.isfinite(base) and base > 0 else np.nan
    )
    return aggregate


def _json_value(value):
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def export_result(result, output_dir, *, input_paths=(), request=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_results(result)
    tables = {
        "daily_ledger": result.daily, "cohort_pnl": result.summaries,
        "cohort_audit": result.cohorts, "optimizer_runs": result.optimizer_runs,
        "aggregate": aggregate,
    }
    for name, table in tables.items():
        if not len(table.columns):
            # Header-only exports remain readable with pandas when no cohort
            # completes; the audit/manifest carries the eligibility reasons.
            columns = ["scenario"] if name == "aggregate" else ["cohort"]
            if name in ("daily_ledger", "cohort_pnl"):
                columns += ["scenario"]
            if name == "daily_ledger":
                columns += ["date"]
            table = pd.DataFrame(columns=columns)
        table.to_csv(output / f"{name}.csv", index=False)
    root = Path(__file__).resolve().parents[1]
    source_paths = list(Path(__file__).parent.glob("*.py"))
    source_paths += list((root / "carry_put_pricing" / "src" / "carry_put_pricing").glob("*.py"))
    source_paths += list((root / "im_2factor_ou_carry" / "src" / "im_2factor_ou_carry").glob("*.py"))
    source_paths += [root / "Demo" / f for f in ("calendar_utils.py", "demo_quality.py", "calibration.py",
                                                "data/china_exchange_calendar_2027_2028.csv")]
    hashes = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in [*input_paths, *source_paths]}
    metadata = {
        "config": asdict(result.config), "request": request,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__},
        "sha256": hashes,
        "assumptions": [
            ("Short optional component with five scenarios; funded directly tradable spot and "
             "zero spot dividends." if result.config.study == "short_with_spot" else
             "Long optional component only; synthetic model inception premium."),
            "Frozen inception parameters provisionally treated as risk-neutral.",
            "Close-based marks; settlement, liquidity, and margin funding are not modeled.",
            "Execution lag applies to hedge rebalancing, not to the daily exercise decision/unwind.",
            "Historical option marks are model values; exercise cash flows use observed futures.",
            "Terminal optional payoff is zero; futures unwind uses observed close.",
            ("Spot is fully funded through the cash account; spot mark-to-market is attribution only "
             "and dividend cash flow is exactly zero." if result.config.study == "short_with_spot" else
             "Futures have no principal purchase outlay; only daily settlement P&L is recorded."),
            "Sum of cohort P&L is a sum of separate trades, not a continuously funded portfolio return.",
            "Tail statistics are descriptive; no independent-sample confidence claim.",
        ],
        "cohorts": result.cohorts.to_dict("records"),
        "aggregate": aggregate.to_dict("records"),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(_json_value(metadata), indent=2, allow_nan=False), encoding="utf-8"
    )
    completed = len(result.summaries) // len(scenario_specs(result.config.study))
    if result.config.study == "short_with_spot":
        title = "# Short carry-put with funded spot hedge — monthly cohort back-test"
        scenario_note = "The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date."
    else:
        title = "# Carry-put monthly cohort back-test"
        scenario_note = "Each cohort has the same option and exit date in all three scenarios."
    lines = [title, "",
              f"Completed cohorts: {completed}. Scheduled cohorts: {len(result.cohorts)}.", "",
              scenario_note, "",
              "| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |",
              "|---|---:|---:|---:|"]
    for row in aggregate.to_dict("records"):
        reduction = row["variance_reduction_vs_no_hedge"]
        reduction_text = f"{reduction:.2%}" if np.isfinite(reduction) else "undefined"
        lines.append(f"| {row['scenario']} | {row['mean_cohort_pnl']:.6f} | {row['sum_cohort_pnl']:.6f} | {reduction_text} |")
    lines += ["", "P&L is in configured monetary units (points when both multipliers are 1).",
              "Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.",
              "Daily variance uses model option marks. Tail statistics can be unstable in a small sample.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
