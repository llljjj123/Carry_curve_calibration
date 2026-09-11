"""Regression tests for the post-review validation and reporting fixes."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from carry_put_backtest.analyze_short_spot import (
    SCENARIOS, _compare_historical_outputs, _validate_variant, _write_analysis_manifest, validate_batch,
)
from carry_put_backtest.engine import SHORT_SCENARIOS, _short_signal_targets_from_positions
from carry_put_backtest.reporting import aggregate_results
from carry_put_backtest.test_evidence import classify_test_evidence, run_test_evidence, source_identity
from carry_put_backtest.tests.test_backtest import CONFIG
from carry_put_backtest.tests.test_backtest import run


def validation_frames(cohorts=("2025-01-20",)):
    coverage = pd.DataFrame({"cohort": list(cohorts), "eligible": True})
    audit = pd.DataFrame({
        "cohort": cohorts, "status": "completed", "sample_curve_dates": 488,
        "entry_date": pd.to_datetime(cohorts), "sample_end": pd.to_datetime(cohorts),
    })
    summary_rows, daily_rows = [], []
    for cohort in cohorts:
        for scenario in SCENARIOS:
            summary_rows.append({"cohort": cohort, "scenario": scenario,
                                 "exit_date": pd.Timestamp("2025-02-21"), "exit_reason": "expiry"})
            for i, date in enumerate(pd.to_datetime(["2025-01-20", "2025-02-21"])):
                daily_rows.append({
                    "cohort": cohort, "scenario": scenario, "date": date,
                    "contracts_1": 0.0, "contracts_2": 0.0, "spot_units": 0.0,
                    "option_mark": 0.0 if i else 1.0,
                    "exercise_now": False, "expired": i == 1,
                })
    return pd.DataFrame(daily_rows), pd.DataFrame(summary_rows), audit, coverage


@pytest.mark.parametrize("mutation,match", [
    ("empty", "empty"), ("missing", "differ"), ("extra", "differ"),
    ("duplicate", "duplicate"), ("scenario", "scenario set"), ("daily_ids", "differ"),
])
def test_full_validation_rejects_bad_cohort_and_scenario_sets(mutation, match):
    daily, summary, audit, coverage = validation_frames(("2025-01-20", "2025-02-24"))
    if mutation == "empty":
        daily = daily.iloc[:0]
    elif mutation == "missing":
        audit = audit.iloc[:1]
        summary = summary.query("cohort == '2025-01-20'")
        daily = daily.query("cohort == '2025-01-20'")
    elif mutation == "extra":
        coverage = coverage.iloc[:1]
    elif mutation == "duplicate":
        summary = pd.concat([summary, summary.iloc[[0]]], ignore_index=True)
    elif mutation == "scenario":
        summary = summary.loc[~((summary.cohort == "2025-01-20") & (summary.scenario == SCENARIOS[-1]))]
    elif mutation == "daily_ids":
        daily = daily.query("cohort == '2025-01-20'")
    with pytest.raises(RuntimeError, match=match):
        validate_batch(daily, summary, audit, coverage)


def test_explicit_pilot_subset_is_labeled_and_full_mode_rejects_it():
    daily, summary, audit, coverage = validation_frames(("2025-01-20", "2025-02-24"))
    pilot = "2025-01-20"
    subset = (daily.query("cohort == @pilot"), summary.query("cohort == @pilot"),
              audit.query("cohort == @pilot"), coverage)
    with pytest.raises(RuntimeError, match="differ"):
        validate_batch(*subset)
    result = validate_batch(*subset, mode="pilot", requested_cohorts=[pilot])
    assert result == {"mode": "pilot", "requested_cohorts": [pilot], "is_full_historical_batch": False}


def test_peak_aggregation_uses_max_flows_sum_and_terminal_borrowing_counts():
    summaries = pd.DataFrame({
        "scenario": ["short_no_hedge", "short_no_hedge"], "total_pnl": [1.0, 2.0],
        "transaction_costs": [3.0, 4.0], "turnover_contracts": [5.0, 6.0],
        "spot_pnl": [7.0, 8.0], "peak_borrowing": [10.0, 40.0],
        "peak_spot_notional": [20.0, 30.0], "spot_turnover_notional": [11.0, 12.0],
        "daily_discounted_pnl_std": [1.0, 9.0],
    })
    daily = pd.DataFrame({
        "scenario": ["short_no_hedge"] * 4, "elapsed_sessions": [0, 1, 0, 1],
        "discounted_pnl_change": [0.0, 1.0, 0.0, 3.0],
    })
    row = aggregate_results(SimpleNamespace(summaries=summaries, daily=daily)).iloc[0]
    assert row.peak_borrowing == 40.0 and row.peak_spot_notional == 30.0
    assert row.spot_pnl == 15.0 and row.total_costs == 7.0
    assert row.pooled_daily_discounted_pnl_std == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert row.pooled_daily_discounted_pnl_std != row.mean_cohort_daily_discounted_pnl_std


def test_test_evidence_missing_stale_and_actual_failure(tmp_path):
    root = Path(__file__).resolve().parents[2]
    assert classify_test_evidence(tmp_path, root=root)["status"] == "not_run"
    stale = {"status": "passed", "exit_code": 0, "source_identity": {}}
    (tmp_path / "test_results.json").write_text(json.dumps(stale), encoding="utf-8")
    assert classify_test_evidence(tmp_path, root=root)["status"] == "stale"
    failure = tmp_path / "intentional_failure.py"
    failure.write_text("def test_failure():\n    assert False\n", encoding="utf-8")
    evidence = run_test_evidence(tmp_path, [sys.executable, "-B", "-m", "pytest", failure.name, "-q"], root=tmp_path)
    assert evidence["status"] == "failed" and evidence["exit_code"] != 0
    assert "FAILED" in (tmp_path / "test_results.log").read_text(encoding="utf-8")


def test_unequal_multipliers_exact_once_and_rounded_futures_drive_spot():
    config = replace(CONFIG, study="short_with_spot", option_units=3,
                     option_multiplier=100, futures_multiplier=200, round_contracts=True)
    spec = next(x for x in SHORT_SCENARIOS if x.scenario == "short_two_futures_spot")
    value = SimpleNamespace(price=30.0)
    mathematical = np.array([-0.61, 0.37]) * config.option_point_value / config.futures_multiplier
    executed = np.round(mathematical)
    spot = _short_signal_targets_from_positions(
        value, np.array([5950.0, 6100.0]), 6000.0, executed, spec, config,
    )
    cash_scale = -config.option_point_value * value.price + config.futures_multiplier * executed @ np.array([5950.0, 6100.0]) + spot * 6000.0
    assert cash_scale == pytest.approx(0.0, abs=1e-10)
    assert not np.allclose(executed, mathematical)


def test_source_identity_is_content_bound(tmp_path):
    root = Path(__file__).resolve().parents[2]
    identity = source_identity(root)
    assert "carry_put_backtest/analyze_short_spot.py" in identity


def test_analysis_manifest_does_not_relabel_execution_source(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    source = Path(__file__).resolve().parents[1] / "analyze_short_spot.py"
    manifest = {"sha256": {str(source): "0" * 64}}
    execution_path = baseline / "run_manifest.json"
    execution_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in ("daily_ledger.csv", "cohort_pnl.csv", "cohort_audit.csv"):
        (baseline / name).write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "coverage.csv").write_text("cohort,eligible\n2025-01-20,true\n", encoding="utf-8")
    before = execution_path.read_bytes()
    _write_analysis_manifest(tmp_path, {"ok": True}, {"lag1": {"lag": 1}})
    assert execution_path.read_bytes() == before
    analysis = json.loads((tmp_path / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert analysis["exact_execution_source_status"] == "unresolved"
    assert analysis["execution_source_comparison"]["baseline"][0]["status"] == "mismatch"
    assert analysis["test_evidence"]["status"] == "not_run"


def test_incomplete_sensitivity_variant_is_rejected():
    daily, _, _, _ = validation_frames()
    baseline = daily.assign(elapsed_sessions=np.tile([0, 1], len(daily) // 2))
    replayed = baseline[["cohort", "scenario", "date", "elapsed_sessions",
                         "contracts_1", "contracts_2", "spot_units"]].copy()
    summary = baseline[["cohort", "scenario"]].drop_duplicates()
    _validate_variant(replayed, summary, baseline, "complete")
    with pytest.raises(RuntimeError, match="incomplete"):
        _validate_variant(replayed.iloc[:-1], summary, baseline, "partial")


def test_rounded_engine_neutralizes_cash_scale_but_can_leave_factor_residuals():
    config = replace(CONFIG, study="short_with_spot", option_units=3,
                     option_multiplier=100, futures_multiplier=200, round_contracts=True)
    result = run(config=config)
    active = result.daily.loc[~(result.daily.exercise_now | result.daily.expired)]
    hedged = active.query("scenario == 'short_two_futures_spot'")
    assert hedged.residual_scale_exposure.abs().max() < 1e-8
    assert hedged.spot_delta_residual.abs().max() < 1e-10
    assert (hedged.residual_slow_exposure.abs() > 1e-8).any() or (hedged.residual_fast_exposure.abs() > 1e-8).any()
    required = {
        "max_abs_residual_slow_exposure", "max_abs_residual_fast_exposure",
        "max_abs_residual_scale_exposure", "max_abs_spot_delta_residual",
    }
    assert required <= set(result.summaries.columns)


def test_old_new_comparison_handles_boolean_decisions(tmp_path, monkeypatch):
    old_root = tmp_path / "carry_put_backtest/outputs_short_spot"
    new_root = tmp_path / "new"
    (old_root / "baseline").mkdir(parents=True)
    (new_root / "baseline").mkdir(parents=True)
    frame = pd.DataFrame({
        "cohort": ["2025-01-20"], "scenario": [SCENARIOS[0]],
        "date": ["2025-01-20"], "elapsed_sessions": [0],
        "exercise_now": [False], "expired": [False], "cash": [1.0],
    })
    frame.to_csv(old_root / "baseline/daily_ledger.csv", index=False)
    frame.to_csv(new_root / "baseline/daily_ledger.csv", index=False)
    coverage = pd.DataFrame({"cohort": ["2025-01-20"], "eligible": [True]})
    coverage.to_csv(old_root / "coverage.csv", index=False)
    coverage.to_csv(new_root / "coverage.csv", index=False)
    monkeypatch.setattr("carry_put_backtest.analyze_short_spot.ROOT", tmp_path)
    result = _compare_historical_outputs(new_root)
    assert result["maximum_absolute_numeric_or_mismatch_count_by_field"]["exercise_now"] == 0
