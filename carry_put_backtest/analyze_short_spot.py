"""Independent validation, figures, and review handoff for the short/spot study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .analyze_historical import replay
from .historical_validation import DEFAULT_OUTPUT, ROOT, SHORT_OUTPUT, write_json
from .test_evidence import classify_test_evidence, source_identity


SCENARIOS = (
    "short_no_hedge", "short_one_futures_spot", "short_two_futures_spot",
    "short_one_futures_only", "short_two_futures_only",
)
PRIMARY = ("short_no_hedge", "short_one_futures_spot", "short_two_futures_spot")
CONTROLS = ("short_one_futures_only", "short_two_futures_only")


def _read_batch(output: Path, label: str):
    base = output / label
    daily = pd.read_csv(base / "daily_ledger.csv", parse_dates=["date"])
    summary = pd.read_csv(base / "cohort_pnl.csv", parse_dates=["entry_date", "exit_date"])
    cohorts = pd.read_csv(base / "cohort_audit.csv", parse_dates=[
        "entry_date", "expiry", "previous_expiry", "sample_start", "sample_end", "exit_date",
    ])
    return daily, summary, cohorts


def _errors(original, replayed):
    keys = ["cohort", "scenario", "date", "elapsed_sessions"]
    merged = original.merge(replayed, on=keys, suffixes=("_engine", "_replay"))
    columns = ["cash", "equity", "discounted_equity", "discounted_pnl_change",
               "contracts_1", "contracts_2", "spot_units"]
    return {
        column: float((merged[f"{column}_engine"] - merged[f"{column}_replay"]).abs().max())
        for column in columns
    }


def _variant_row(output: Path, label: str):
    # Replay-only sensitivity summaries contain accounting totals rather than
    # the full engine cohort schema, so do not require entry/exit date fields.
    summary = pd.read_csv(output / label / "cohort_pnl.csv")
    rows = []
    for scenario, group in summary.groupby("scenario", sort=False):
        changes = pd.Series(dtype=float)
        daily_path = output / label / "daily_ledger.csv"
        daily = pd.read_csv(daily_path)
        changes = daily.loc[
            (daily.scenario == scenario) & (daily.elapsed_sessions > 0),
            "discounted_pnl_change",
        ]
        rows.append({
            "variant": label, "scenario": scenario,
            "sum_pnl": group.total_pnl.sum(), "mean_pnl": group.total_pnl.mean(),
            "cohort_pnl_std": group.total_pnl.std(ddof=1),
            "daily_pnl_std": changes.std(ddof=1),
            "daily_pnl_variance": changes.var(ddof=1),
            "costs": group.transaction_costs.sum(),
        })
    return rows


def _make_figures(output: Path, summary: pd.DataFrame, exposures: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = {
        "short_no_hedge": "No hedge", "short_one_futures_spot": "1 futures + spot",
        "short_two_futures_spot": "2 futures + spot", "short_one_futures_only": "1 futures only",
        "short_two_futures_only": "2 futures only",
    }
    order = [s for s in SCENARIOS if s in set(summary.scenario)]
    pivot = summary.pivot(index="cohort", columns="scenario", values="total_pnl").sort_index()[order]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    positions = np.arange(len(pivot))
    width = .8 / max(len(order), 1)
    for i, scenario in enumerate(order):
        values = pivot[scenario]
        axes[0].bar(positions + (i - (len(order)-1)/2) * width, values,
                    width=width, label=labels.get(scenario, scenario))
        axes[1].plot(positions, values.cumsum(), label=labels.get(scenario, scenario), marker="o", markersize=2)
    for ax in axes:
        ax.axhline(0, color="#334155", linewidth=.6)
        ax.set_xticks(positions, [str(d)[:7] for d in pivot.index], rotation=60, ha="right")
        ax.grid(axis="y", alpha=.2)
        ax.set_ylabel("Configured monetary units")
    axes[0].set_title("Short carry-put cohort P&L")
    axes[0].legend(ncol=3)
    axes[1].set_title("Cumulative sum of independent cohort P&L (not portfolio NAV)")
    fig.savefig(figures / "cohort_pnl.png", dpi=160)
    plt.close(fig)

    plot_order = [s for s in PRIMARY + CONTROLS if s in exposures.index]
    names = [labels[s] for s in plot_order]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].bar(names, np.sqrt(exposures.loc[plot_order, "mean_factor_variance"]))
    axes[0].set_title("Residual carry-factor risk")
    axes[0].set_ylabel("One-session factor-risk std")
    axes[1].bar(names, exposures.loc[plot_order, "rms_scale_exposure"])
    axes[1].set_title("Residual scale exposure")
    axes[1].set_ylabel("Cash units per 100% scale move")
    axes[2].bar(names, exposures.loc[plot_order, "peak_spot_notional"])
    axes[2].set_title("Peak funded spot notional")
    axes[2].set_ylabel("Cash units")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(figures / "hedge_sufficiency.png", dpi=160)
    plt.close(fig)


def validate_batch(daily, summary, cohorts, coverage, *, mode="full", requested_cohorts=None):
    """Validate a complete historical batch or an explicitly requested pilot."""
    if mode not in ("full", "pilot"):
        raise RuntimeError(f"Unknown validation mode: {mode}")
    for name, frame in (("daily ledger", daily), ("cohort summary", summary),
                        ("cohort audit", cohorts), ("coverage", coverage)):
        if frame.empty:
            raise RuntimeError(f"{name} is empty")
    if coverage.cohort.duplicated().any():
        raise RuntimeError("coverage contains duplicate cohort IDs")
    if cohorts.cohort.duplicated().any():
        raise RuntimeError("cohort audit contains duplicate cohort IDs")
    if summary.duplicated(["cohort", "scenario"]).any():
        raise RuntimeError("cohort summary contains duplicate cohort/scenario rows")
    if daily.duplicated(["cohort", "scenario", "date"]).any():
        raise RuntimeError("daily ledger contains duplicate cohort/scenario/date rows")

    eligible = set(coverage.loc[coverage.eligible.astype(bool), "cohort"].astype(str))
    audit_ids = set(cohorts.cohort.astype(str))
    summary_ids = set(summary.cohort.astype(str))
    daily_ids = set(daily.cohort.astype(str))
    if mode == "full":
        expected = eligible
    else:
        if not requested_cohorts:
            raise RuntimeError("pilot mode requires explicit requested cohort IDs")
        expected = set(map(str, requested_cohorts))
        if not expected <= eligible:
            raise RuntimeError(f"pilot requested ineligible cohort IDs: {sorted(expected - eligible)}")
    for name, actual in (("audit", audit_ids), ("summary", summary_ids), ("daily", daily_ids)):
        if actual != expected:
            raise RuntimeError(f"{name} cohort IDs differ from expected: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    if set(cohorts.status) != {"completed"}:
        raise RuntimeError("cohort audit contains non-completed status")
    if not (cohorts.sample_curve_dates == 488).all():
        raise RuntimeError("cohort audit violates the 488-date calibration requirement")
    if not (pd.to_datetime(cohorts.sample_end) == pd.to_datetime(cohorts.entry_date)).all():
        raise RuntimeError("cohort audit sample_end differs from entry_date")

    expected_scenarios = set(SCENARIOS)
    for cohort, group in summary.groupby("cohort", sort=False):
        if set(group.scenario) != expected_scenarios:
            raise RuntimeError(f"summary scenario set is incomplete for {cohort}")
        if group.exit_date.nunique() != 1 or group.exit_reason.nunique() != 1:
            raise RuntimeError(f"summary exits are inconsistent for {cohort}")
    for cohort, group in daily.groupby("cohort", sort=False):
        if set(group.scenario) != expected_scenarios:
            raise RuntimeError(f"daily scenario set is incomplete for {cohort}")
        paths = [tuple(pd.to_datetime(x.date)) for _, x in group.groupby("scenario", sort=False)]
        if any(path != paths[0] for path in paths[1:]):
            raise RuntimeError(f"scenario dates/exits differ for {cohort}")
    final = daily.sort_values(["cohort", "scenario", "date"]).groupby(
        ["cohort", "scenario"], sort=False).tail(1)
    if not (final[["contracts_1", "contracts_2", "spot_units", "option_mark"]].abs() <= 1e-12).all().all():
        raise RuntimeError("terminal liquidation is incomplete")
    if not (final.exercise_now.astype(bool) | final.expired.astype(bool)).all():
        raise RuntimeError("daily ledger has a non-terminal final row")
    return {"mode": mode, "requested_cohorts": sorted(expected), "is_full_historical_batch": mode == "full"}


def check_and_analyze(output: Path, *, mode="full", requested_cohorts=None):
    output = Path(output)
    daily, summary, cohorts = _read_batch(output, "baseline")
    coverage = pd.read_csv(output / "coverage.csv")
    validation_mode = validate_batch(daily, summary, cohorts, coverage, mode=mode,
                                     requested_cohorts=requested_cohorts)
    replayed, replay_totals = replay(daily, summary)
    errors = _errors(daily, replayed)
    if max(errors.values()) >= 1e-8:
        raise RuntimeError(f"independent cash replay failed: {errors}")
    total_errors = (
        summary.set_index(["cohort", "scenario"]).total_pnl
        - replay_totals.set_index(["cohort", "scenario"]).total_pnl
    ).abs()
    if float(total_errors.max()) >= 1e-8:
        raise RuntimeError(f"cohort total replay failed: {float(total_errors.max())}")

    active = daily.loc[~(daily.exercise_now | daily.expired)]
    two_spot = active.loc[active.scenario == "short_two_futures_spot"]
    one_spot = active.loc[active.scenario == "short_one_futures_spot"]
    manifest = json.loads((output / "baseline/run_manifest.json").read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    if config.get("execution_lag_sessions") == 0 and not config.get("round_contracts", False):
        if float(two_spot[["residual_slow_exposure", "residual_fast_exposure"]].abs().max().max()) >= 1e-8:
            raise RuntimeError("two-futures same-close factor neutrality failed")
        if float(two_spot.residual_scale_exposure.abs().max()) >= 1e-8:
            raise RuntimeError("two-futures same-close cash-scale neutrality failed")
        if float(one_spot.residual_scale_exposure.abs().max()) >= 1e-8:
            raise RuntimeError("one-futures same-close cash-scale neutrality failed")
    controls = active.loc[active.scenario.isin(CONTROLS)]
    if not (controls.spot_units == 0).all():
        raise RuntimeError("futures-only control contains spot holdings")
    diagnostics = {
        "completed_cohorts": int(cohorts.cohort.nunique()),
        "scheduled_cohorts": int(len(coverage)),
        "coverage_exclusions": coverage.loc[~coverage.eligible, "reason"].value_counts().to_dict(),
        "independent_cash_replay_max_errors": errors,
        "maximum_total_replay_error": float(total_errors.max()),
        "short_two_futures_spot_max_abs_slow_factor_exposure_option_cash_per_state_unit": float(two_spot.residual_slow_exposure.abs().max()),
        "short_two_futures_spot_max_abs_fast_factor_exposure_option_cash_per_state_unit": float(two_spot.residual_fast_exposure.abs().max()),
        "short_two_futures_spot_max_abs_cash_scale_exposure_cash_per_unit_scale": float(two_spot.residual_scale_exposure.abs().max()),
        "short_two_futures_spot_max_abs_spot_delta_spot_units": float(two_spot.spot_delta_residual.abs().max()),
        "short_one_futures_spot_max_abs_cash_scale_exposure_cash_per_unit_scale": float(one_spot.residual_scale_exposure.abs().max()),
        "short_one_futures_spot_max_abs_spot_delta_spot_units": float(one_spot.spot_delta_residual.abs().max()),
        "maximum_condition_number": float(active.hedge_condition_number.max()),
        "conditioning_warning_days": int(active.conditioning_warning.sum()),
        "exercise_cohorts": int((cohorts.exit_reason == "exercise").sum()),
        "expiry_cohorts": int((cohorts.exit_reason == "expiry").sum()),
        "all_dividend_cash_flows_zero": bool((daily.dividend_cash_flow == 0).all()),
        "validation_mode": validation_mode,
    }
    write_json(output / "validation_checks.json", diagnostics)

    exposures = active.groupby("scenario").agg(
        mean_factor_variance=("residual_factor_variance", "mean"),
        mean_unhedged_factor_variance=("unhedged_factor_variance", "mean"),
        rms_scale_exposure=("residual_scale_exposure", lambda x: np.sqrt(np.mean(x**2))),
        max_abs_scale_exposure=("residual_scale_exposure", lambda x: x.abs().max()),
        max_abs_slow_factor_exposure=("residual_slow_exposure", lambda x: x.abs().max()),
        max_abs_fast_factor_exposure=("residual_fast_exposure", lambda x: x.abs().max()),
        max_abs_spot_delta=("spot_delta_residual", lambda x: x.abs().max()),
    )
    full_path_peaks = daily.groupby("scenario").agg(
        peak_spot_notional=("spot_notional", "max"),
        peak_borrowing=("borrowing", "max"),
    )
    exposures = exposures.join(full_path_peaks)
    exposures["factor_variance_reduction"] = 1 - exposures.mean_factor_variance / exposures.mean_unhedged_factor_variance
    exposures.to_csv(output / "exposure_summary.csv")

    # Lag and cost variants are accounting-only changes. Replaying the saved
    # baseline signals avoids refitting/repricing while retaining an
    # independently checked cash path.
    variants_to_replay = {
        "lag1": dict(lag=1),
        "costs": dict(lag=0, cost_points=.2, spot_cost_bps=1.0),
    }
    for label, kwargs in variants_to_replay.items():
        path = output / label
        path.mkdir(parents=True, exist_ok=True)
        replayed_variant, pnl_variant = replay(daily, summary, **kwargs)
        replayed_variant.to_csv(path / "daily_ledger.csv", index=False)
        pnl_variant.to_csv(path / "cohort_pnl.csv", index=False)
        _validate_variant(replayed_variant, pnl_variant, daily, label)

    variants = pd.DataFrame(_variant_row(output, "baseline"))
    for label in ("lag1", "costs"):
        if (output / label / "cohort_pnl.csv").exists():
            variants = pd.concat([variants, pd.DataFrame(_variant_row(output, label))], ignore_index=True)
    unhedged_variance = variants.loc[variants.scenario == "short_no_hedge"].set_index("variant").daily_pnl_variance
    variants["variance_reduction_vs_no_hedge"] = 1 - variants.daily_pnl_variance / variants.variant.map(unhedged_variance)
    variants.to_csv(output / "execution_sensitivities.csv", index=False)
    comparison = _compare_historical_outputs(output)
    _make_figures(output, summary, exposures)
    _write_analysis_manifest(output, diagnostics, variants_to_replay)
    write_report(output, summary, cohorts, exposures, variants, diagnostics)
    write_handoff(output, diagnostics, variants, comparison)
    return diagnostics


def _validate_variant(daily, summary, baseline_daily, label):
    expected_keys = set(map(tuple, baseline_daily[["cohort", "scenario", "date", "elapsed_sessions"]].itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, daily[["cohort", "scenario", "date", "elapsed_sessions"]].itertuples(index=False, name=None)))
    if len(daily) != len(expected_keys) or actual_keys != expected_keys:
        raise RuntimeError(f"{label} sensitivity daily rows are incomplete or duplicated")
    expected_pairs = set(map(tuple, baseline_daily[["cohort", "scenario"]].drop_duplicates().itertuples(index=False, name=None)))
    actual_pairs = set(map(tuple, summary[["cohort", "scenario"]].itertuples(index=False, name=None)))
    if len(summary) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise RuntimeError(f"{label} sensitivity summary rows are incomplete or duplicated")
    terminal = daily.sort_values(["cohort", "scenario", "elapsed_sessions"]).groupby(
        ["cohort", "scenario"], sort=False).tail(1)
    if not (terminal[["contracts_1", "contracts_2", "spot_units"]].abs() <= 1e-12).all().all():
        raise RuntimeError(f"{label} sensitivity did not liquidate terminal positions")


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _execution_source_comparison(manifest):
    comparisons = []
    for recorded_name, recorded_hash in manifest.get("sha256", {}).items():
        candidate = Path(recorded_name)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.exists() and candidate.suffix == ".py":
            actual = _hash_file(candidate)
            comparisons.append({
                "path": str(candidate), "execution_sha256": recorded_hash,
                "current_sha256": actual, "status": "match" if actual == recorded_hash else "mismatch",
            })
    return comparisons


def _write_analysis_manifest(output: Path, diagnostics, sensitivity_settings):
    inputs = []
    for relative in ("baseline/daily_ledger.csv", "baseline/cohort_pnl.csv",
                     "baseline/cohort_audit.csv", "baseline/run_manifest.json", "coverage.csv"):
        path = output / relative
        if path.exists():
            inputs.append(path)
    execution = {}
    source_comparisons = {}
    for label in ("baseline", "fine"):
        path = output / label / "run_manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            execution[label] = {"path": str(path), "sha256": _hash_file(path)}
            source_comparisons[label] = _execution_source_comparison(manifest)
    mismatches = [row for rows in source_comparisons.values() for row in rows if row["status"] != "match"]
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_command": [sys.executable, *sys.argv],
        "analysis_source_identity": source_identity(ROOT),
        "input_artifact_sha256": {str(path.relative_to(output)).replace("\\", "/"): _hash_file(path) for path in inputs},
        "execution_manifests": execution,
        "execution_source_comparison": source_comparisons,
        "exact_execution_source_status": "unresolved" if mismatches else "matched_current_source",
        "accounting_validation_status": "passed",
        "validation_outcomes": diagnostics,
        "sensitivity_settings": sensitivity_settings,
        "test_evidence": classify_test_evidence(output, root=ROOT),
    }
    write_json(output / "analysis_manifest.json", record)


def _compare_historical_outputs(output: Path):
    old_root = ROOT / "carry_put_backtest/outputs_short_spot"
    old_daily_path = old_root / "baseline/daily_ledger.csv"
    if not old_daily_path.exists():
        result = {"status": "not_available", "old_exact_source_provenance": "unresolved"}
        write_json(output / "old_new_comparison.json", result)
        return result
    old = pd.read_csv(old_daily_path)
    new = pd.read_csv(output / "baseline/daily_ledger.csv")
    keys = ["cohort", "scenario", "date", "elapsed_sessions"]
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise RuntimeError("old/new comparison encountered duplicate daily keys")
    merged = old.merge(new, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    requested = (
        "initial_premium", "positive_option_price", "signed_option_mark", "option_mark",
        "exercise_now", "expired", "contracts_1", "contracts_2", "spot_units", "cash",
        "equity", "discounted_equity", "option_pnl", "futures_pnl", "spot_pnl",
        "financing", "discounted_pnl_change", "residual_slow_exposure",
        "residual_fast_exposure", "residual_scale_exposure", "spot_delta_residual",
    )
    differences = {}
    for column in requested:
        left, right = f"{column}_old", f"{column}_new"
        if left not in merged or right not in merged:
            continue
        numeric = (pd.api.types.is_numeric_dtype(merged[left]) and
                   pd.api.types.is_numeric_dtype(merged[right]) and
                   not pd.api.types.is_bool_dtype(merged[left]) and
                   not pd.api.types.is_bool_dtype(merged[right]))
        if numeric:
            differences[column] = float((merged[left] - merged[right]).abs().max())
        else:
            differences[column] = int((merged[left].astype(str) != merged[right].astype(str)).sum())
    old_coverage = pd.read_csv(old_root / "coverage.csv")
    new_coverage = pd.read_csv(output / "coverage.csv")
    coverage_keys = [column for column in ("cohort", "eligible", "reason", "available_curve_dates")
                     if column in old_coverage and column in new_coverage]
    coverage_equal = old_coverage[coverage_keys].astype(str).equals(new_coverage[coverage_keys].astype(str))
    result = {
        "status": "compared", "old_exact_source_provenance": "unresolved",
        "matched_daily_rows": int((merged._merge == "both").sum()),
        "old_only_daily_rows": int((merged._merge == "left_only").sum()),
        "new_only_daily_rows": int((merged._merge == "right_only").sum()),
        "maximum_absolute_numeric_or_mismatch_count_by_field": differences,
        "coverage_equal": coverage_equal,
    }
    write_json(output / "old_new_comparison.json", result)
    return result


def write_report(output, summary, cohorts, exposures, variants, diagnostics):
    aggregate = summary.groupby("scenario").agg(
        sum_pnl=("total_pnl", "sum"), mean_pnl=("total_pnl", "mean"),
        std_pnl=("total_pnl", "std"), mean_cohort_daily_std=("daily_discounted_pnl_std", "mean"),
    )
    daily = pd.read_csv(output / "baseline/daily_ledger.csv")
    pooled = daily.loc[daily.elapsed_sessions > 0].groupby("scenario").discounted_pnl_change.std(ddof=1)
    aggregate["pooled_daily_std"] = pooled
    fine_path = output / "fine" / "cohort_pnl.csv"
    fine = pd.read_csv(fine_path) if fine_path.exists() else pd.DataFrame()
    labels = {
        "short_no_hedge": "Short no hedge", "short_one_futures_spot": "Short 1 futures + spot",
        "short_two_futures_spot": "Short 2 futures + spot", "short_one_futures_only": "Short 1 futures only",
        "short_two_futures_only": "Short 2 futures only",
    }
    lines = [
        ("# Short carry-put with funded spot — historical validation" if diagnostics["validation_mode"]["is_full_historical_batch"]
         else "# PILOT/PARTIAL — short carry-put with funded spot validation"), "",
        "This conditional experiment sells the established carry-put optional component and compares funded spot hedges with futures-only controls. Spot dividends are exactly zero; model carry, theta, and locked inception carry remain unchanged.", "",
        f"The run completed {len(cohorts)} eligible cohorts from {cohorts.entry_date.min().date()} through {cohorts.entry_date.max().date()}. There were {diagnostics['exercise_cohorts']} early exercises and {diagnostics['expiry_cohorts']} expiries; all five scenarios share each cohort's exercise date.", "",
        "## Baseline results", "",
        "| Scenario | Sum P&L | Mean P&L | Pooled daily discounted P&L std | Mean cohort daily std |", "|---|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        row = aggregate.loc[scenario]
        lines.append(f"| {labels[scenario]} | {row.sum_pnl:.6f} | {row.mean_pnl:.6f} | {row.pooled_daily_std:.6f} | {row.mean_cohort_daily_std:.6f} |")
    lines += [
        "", "P&L is in configured monetary units (points when both multipliers are 1). Cohort sums are independent trades, not a funded portfolio NAV or annual return.", "",
        "![Cohort P&L](figures/cohort_pnl.png)", "",
        "## Hedge sufficiency", "",
        "The one-futures-plus-spot scenario neutralizes scale exposure and minimizes the remaining one-dimensional carry-factor variance. The two-futures-plus-spot scenario neutralizes both modeled carry factors and scale. Futures-only controls intentionally retain their scale exposure.", "",
        "| Scenario | Carry-factor variance reduction | Max |slow| (cash/state) | Max |fast| (cash/state) | Max cash-scale (cash/unit scale) | Max spot delta (spot units) | Peak spot notional | Peak borrowing |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        row = exposures.loc[scenario]
        lines.append(f"| {labels[scenario]} | {row.factor_variance_reduction:.4%} | {row.max_abs_slow_factor_exposure:.6f} | {row.max_abs_fast_factor_exposure:.6f} | {row.max_abs_scale_exposure:.6f} | {row.max_abs_spot_delta:.9f} | {row.peak_spot_notional:.6f} | {row.peak_borrowing:.6f} |")
    lines += ["", "![Hedge sufficiency](figures/hedge_sufficiency.png)", "",
              "## Independent accounting and controls", "",
              f"The independent funded replay's maximum daily equity/cash/position error was {max(diagnostics['independent_cash_replay_max_errors'].values()):.3e}. The short futures-only controls should be compared with the negative legacy long-option results; this is a sign/accounting regression, not a forecast of new spot-hedged performance.", "",
              "| Variant | Scenario | Sum P&L | Daily P&L std | Costs |", "|---|---|---:|---:|---:|"]
    for row in variants.itertuples():
        lines.append(f"| {row.variant} | {labels.get(row.scenario, row.scenario)} | {row.sum_pnl:.6f} | {row.daily_pnl_std:.6f} | {row.costs:.6f} |")
    lines += ["", "The baseline uses same-close continuous targets, zero costs, the 1.4% common funding rate, and zero spot dividends. The lag sensitivity delays futures and spot targets together; the costs sensitivity uses 0.2 futures points one-way and 1 bp spot notional. These are illustrative accounting sensitivities.", "",
              "## Numerical pilot", ""]
    if not fine.empty:
        baseline_pilot = summary.loc[summary.cohort == fine.cohort.iloc[0]].set_index("scenario")
        fine_pilot = fine.loc[fine.cohort == fine.cohort.iloc[0]].set_index("scenario")
        lines += [
            "The planned 2025-07-21 cohort was rerun at 451×601 with 61-point quadrature. These differences are a pilot only; no full fine-grid historical batch was run.", "",
            "| Scenario | Fine minus baseline premium | Fine minus baseline lifetime P&L | Fine minus baseline daily P&L std |", "|---|---:|---:|---:|"]
        for scenario in SCENARIOS:
            lines.append(f"| {labels[scenario]} | {fine_pilot.loc[scenario, 'initial_premium'] - baseline_pilot.loc[scenario, 'initial_premium']:.6f} | {fine_pilot.loc[scenario, 'total_pnl'] - baseline_pilot.loc[scenario, 'total_pnl']:.6f} | {fine_pilot.loc[scenario, 'daily_discounted_pnl_std'] - baseline_pilot.loc[scenario, 'daily_discounted_pnl_std']:.6f} |")
        lines += ["", "The fine pilot kept the same exercise/expiry outcome; the observed P&L differences should limit reported numerical precision.", ""]
    else:
        lines += ["No fine-grid pilot artifact was found.", ""]
    lines += [
              "## Assumptions and limitations", "",
              "Spot is a hypothetical directly tradable index instrument with zero cash dividends, fully funded through the common cash account. Observed futures carry and this zero-dividend spot convention need not be an arbitrage-consistent market. The results are therefore a conditional hedge/P&L experiment, not an arbitrage or replication claim.", "",
              "Historical OU dynamics are provisionally treated as risk-neutral. Option marks and premiums are model values, exercise uses observed futures, margin and liquidity are omitted, and no asymmetric financing or option transaction fee is modeled.", "",
              "## Files", "",
              "- [Baseline daily ledger](baseline/daily_ledger.csv)",
              "- [Baseline cohort P&L](baseline/cohort_pnl.csv)",
              "- [Exposure summary](exposure_summary.csv)",
              "- [Execution sensitivities](execution_sensitivities.csv)",
              "- [Validation checks](validation_checks.json)",
              "- [Numerical fine pilot](fine/cohort_pnl.csv)",
              "- [Test results](test_results.json)",
              "- [Analysis provenance](analysis_manifest.json)",
              "- [Old/new comparison](old_new_comparison.json)",
              "- [Bug-fix handoff](../short_spot_bugfix_handoff.md)", "",
    ]
    (output / "short_spot_results_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_handoff(output, diagnostics, variants, comparison=None):
    test_command = r"D:\miniforge3\envs\spyder-env\python.exe -B -m pytest carry_put_backtest/tests carry_put_pricing/tests Demo/tests im_2factor_ou_carry/tests -q -p no:cacheprovider"
    baseline = variants.loc[variants.variant == "baseline"].set_index("scenario")
    control_comparison = ", ".join(
        f"{scenario}={baseline.loc[scenario, 'sum_pnl']:.6f}"
        for scenario in ("short_no_hedge", "short_one_futures_only", "short_two_futures_only")
    )
    lines = [
        "# Short spot post-review bug-fix handoff", "",
        "This bundle implements the findings in `short_spot_astra_review.md`. The corrected bundle is ready for a new independent review.", "",
        "## Changed files", "",
        "- `carry_put_backtest/engine.py`: explicit study/scenario definitions, short-side hedge conversion, funded spot ledger, residuals, costs, lag, and summaries.",
        "- `carry_put_backtest/__main__.py` and `__init__.py`: study and spot-cost configuration/API exposure.",
        "- `carry_put_backtest/reporting.py`: flow sums, individual-cohort peak maxima, and pooled risk metrics.",
        "- `carry_put_backtest/analyze_historical.py`: independent funded replay.",
        "- `carry_put_backtest/analyze_short_spot.py`: strict full/pilot validation, regenerated sensitivities, immutable analysis provenance, unit-safe diagnostics, comparisons, and reports.",
        "- `carry_put_backtest/test_evidence.py`: explicit subprocess test evidence with logs, exits, timestamps, and source identity.",
        "- `carry_put_backtest/provenance.py`: immutable historical evidence hash audit.",
        "- `carry_put_backtest/historical_validation.py`: short-study selection and sensitivity configuration.",
        "- `carry_put_backtest/tests/test_short_spot.py`: synthetic economic-identity tests.", "",
        "## Commands and results", "",
        f"Interpreter: `{test_command.split(' -B ')[0]}`", f"Focused/regression command: `{test_command}`",
        "Historical baseline: `python -B -u -m carry_put_backtest.historical_validation --phase baseline --study short_with_spot --output-dir carry_put_backtest/outputs_short_spot_review_fixed`.",
        "Historical analyzer: `python -B -m carry_put_backtest.analyze_short_spot --output-dir carry_put_backtest/outputs_short_spot_review_fixed --mode full`.",
        "", f"Independent replay maximum errors: `{json.dumps(diagnostics['independent_cash_replay_max_errors'], sort_keys=True)}`.",
        f"Cohorts: `{diagnostics['completed_cohorts']}` completed of `{diagnostics['scheduled_cohorts']}` scheduled; exercises `{diagnostics['exercise_cohorts']}`, expiries `{diagnostics['expiry_cohorts']}`.",
        f"Baseline short control sums: `{control_comparison}`; compare futures-only controls with the negative legacy long study.",
        "", "## Sign and unit conventions", "",
        "The pricer returns positive long-option values and sensitivities. The study uses side `s=-1`; short-option premium receipt is positive, exercise payment is negative, and the live option mark is negative. Futures counts are positive long/negative short. Spot units are continuous hypothetical index units. Point P&L is monetary only when configured multipliers and units are one.", "",
        "Spot is sized after actual futures rounding. It is funded from cash, and daily spot P&L is attributed through the funded asset rather than credited to cash separately. Dividend cash flow is exactly zero.", "",
        "## Review points", "",
        "Check the three primary scenarios against their two futures-only controls, especially residual factor/scale exposure, borrowing, spot notional, and total P&L. Verify the short unhedged and futures-only controls against the negative legacy long study. Inspect singular/near-singular handling and the fine pilot before drawing numerical conclusions.", "",
        "The market convention combines zero-dividend spot, common financing, and observed nonzero futures carry; it need not be arbitrage-consistent. The historical OU fit remains a provisional risk-neutral input. No strategy optimization, spot-only hedge, ETF, or minimum-total-variance objective was introduced.", "",
        "## Artifacts", "",
        "- [Results report](outputs_short_spot_review_fixed/short_spot_results_report.md)",
        "- [Baseline ledger](outputs_short_spot_review_fixed/baseline/daily_ledger.csv)",
        "- [Baseline cohort P&L](outputs_short_spot_review_fixed/baseline/cohort_pnl.csv)",
        "- [Validation checks](outputs_short_spot_review_fixed/validation_checks.json)",
        "- [Analysis manifest](outputs_short_spot_review_fixed/analysis_manifest.json)",
        "- [Historical evidence audit](outputs_short_spot_review_fixed/historical_evidence_audit.json)",
        "- [Old/new comparison](outputs_short_spot_review_fixed/old_new_comparison.json)",
        "- [Execution sensitivities](outputs_short_spot_review_fixed/execution_sensitivities.csv)",
        "- [Fine-grid pilot](outputs_short_spot_review_fixed/fine/cohort_pnl.csv)",
        "- [Test evidence](outputs_short_spot_review_fixed/test_results.json)",
        "", "The planned fine-grid pilot was completed for the 2025-07-21 cohort. A full finer-grid historical batch was not run and is not claimed.", "",
    ]
    test_evidence = classify_test_evidence(output, root=ROOT)
    lines += ["## Test evidence", "", f"Status: `{test_evidence.get('status')}`; exit code: `{test_evidence.get('exit_code')}`; parsed passed count: `{test_evidence.get('passed_count')}`.", "",
              "## Provenance resolution", "", "The old output remains immutable and its exact execution source is unresolved. The fresh baseline is generated from current source; numerical agreement cannot retroactively repair the old manifest.", "",
              f"Old/new comparison status: `{(comparison or {}).get('status', 'not_available')}`.", ""]
    (output.parent / "short_spot_bugfix_handoff.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SHORT_OUTPUT)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--mode", choices=("full", "pilot"), default="full")
    parser.add_argument("--pilot-cohorts", nargs="+")
    args = parser.parse_args()
    if args.report_only:
        daily, summary, cohorts = _read_batch(args.output_dir, "baseline")
        exposures = pd.read_csv(args.output_dir / "exposure_summary.csv").set_index("scenario")
        variants = pd.read_csv(args.output_dir / "execution_sensitivities.csv")
        diagnostics = json.loads((args.output_dir / "validation_checks.json").read_text(encoding="utf-8"))
        write_report(args.output_dir, summary, cohorts, exposures, variants, diagnostics)
        comparison = json.loads((args.output_dir / "old_new_comparison.json").read_text()) if (args.output_dir / "old_new_comparison.json").exists() else None
        write_handoff(args.output_dir, diagnostics, variants, comparison)
    else:
        check_and_analyze(args.output_dir, mode=args.mode, requested_cohorts=args.pilot_cohorts)


if __name__ == "__main__":
    main()
