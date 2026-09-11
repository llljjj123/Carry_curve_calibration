"""Independent ledger reconciliation, execution sensitivities, and historical figures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .historical_validation import DEFAULT_OUTPUT, write_json


def replay(daily, summaries, *, lag=None, cost_points=0.0, rate=.014,
           spot_cost_bps=None, futures_multiplier=None, fee_per_contract=None,
           slippage_points=None):
    """Re-account identical option policy/signals without refitting or repricing.

    Deliberately independent of CashLedger. Defaults are the agreed point units.
    Exercise cancels any pending target and closes all positions on that close.
    """
    if "spot_units" in daily.columns:
        return replay_funded(
            daily, summaries, lag=lag, cost_points=cost_points, rate=rate,
            spot_cost_bps=spot_cost_bps, futures_multiplier=futures_multiplier,
            fee_per_contract=fee_per_contract, slippage_points=slippage_points,
        )
    if lag is None:
        lag = 0
    rows, totals = [], []
    for (cohort, scenario), group in daily.groupby(["cohort", "scenario"], sort=False):
        group = group.sort_values("elapsed_sessions")
        premium = summaries.loc[(summaries.cohort == cohort) & (summaries.scenario == scenario), "initial_premium"].iloc[0]
        cash, cumulative_futures, cumulative_costs, cumulative_finance = -premium, 0., 0., 0.
        held, pending, previous_price = np.zeros(2), np.zeros(2), None
        previous_elapsed, previous_discounted = 0, 0.
        for row in group.itertuples():
            prices = np.array([row.futures_1, row.futures_2])
            if previous_price is not None:
                dt = (row.elapsed_sessions-previous_elapsed)/244
                finance = cash*np.expm1(rate*dt)
                futures_pnl = held @ (prices-previous_price)
                cumulative_finance += finance
                cumulative_futures += futures_pnl
                cash += finance+futures_pnl
            terminal = row.exercise_now or row.expired
            cash += row.option_payoff
            signal = np.array([row.target_contracts_1, row.target_contracts_2])
            new = np.zeros(2) if terminal else pending.copy() if lag else signal.copy()
            cost = np.abs(new-held).sum()*cost_points
            cash -= cost
            cumulative_costs += cost
            equity = cash+row.option_mark
            discounted = equity*np.exp(-rate*row.elapsed_sessions/244)
            rows.append(dict(cohort=cohort, scenario=scenario, date=row.date, elapsed_sessions=row.elapsed_sessions,
                             cash=cash, equity=equity, discounted_equity=discounted,
                             discounted_pnl_change=discounted-previous_discounted,
                             contracts_1=new[0], contracts_2=new[1]))
            held, pending = new, signal
            previous_price, previous_elapsed, previous_discounted = prices, row.elapsed_sessions, discounted
        totals.append(dict(cohort=cohort, scenario=scenario, total_pnl=equity,
                           futures_pnl=cumulative_futures, financing=cumulative_finance,
                           transaction_costs=cumulative_costs))
    return pd.DataFrame(rows), pd.DataFrame(totals)


def replay_funded(daily, summaries, *, lag=None, cost_points=0.0, rate=.014,
                  spot_cost_bps=None, futures_multiplier=None,
                  fee_per_contract=None, slippage_points=None):
    """Independently replay a funded spot ledger from saved signals and marks."""
    rows, totals = [], []
    for (cohort, scenario), group in daily.groupby(["cohort", "scenario"], sort=False):
        group = group.sort_values("elapsed_sessions")
        summary = summaries.loc[
            (summaries.cohort == cohort) & (summaries.scenario == scenario)
        ].iloc[0]
        first = group.iloc[0]
        side = int(first.option_side)
        cash = -side * float(summary.initial_premium)
        effective_lag = int(first.execution_lag_sessions) if lag is None else lag
        fm = float(futures_multiplier if futures_multiplier is not None else first.futures_multiplier)
        fee = float(fee_per_contract if fee_per_contract is not None else first.fee_per_contract)
        slip = float(slippage_points if slippage_points is not None else first.slippage_points)
        bps = float(first.spot_cost_bps if spot_cost_bps is None else spot_cost_bps)
        held, pending = np.zeros(2), np.zeros(2)
        held_spot, pending_spot = 0.0, 0.0
        previous_prices = previous_spot = None
        previous_elapsed, previous_discounted = 0, 0.0
        cumulative_futures = cumulative_spot = cumulative_finance = 0.0
        cumulative_futures_costs = cumulative_spot_costs = 0.0
        for row in group.itertuples():
            prices = np.array([row.futures_1, row.futures_2], dtype=float)
            spot = float(row.spot)
            if previous_prices is not None:
                dt = (row.elapsed_sessions - previous_elapsed) / 244
                finance = cash * np.expm1(rate * dt)
                futures_pnl = float(held @ (prices - previous_prices)) * fm
                spot_pnl = held_spot * (spot - previous_spot)
                cash += finance + futures_pnl
                cumulative_finance += finance
                cumulative_futures += futures_pnl
                cumulative_spot += spot_pnl
            else:
                finance = futures_pnl = spot_pnl = 0.0
            terminal = row.exercise_now or row.expired
            option_cash_flow = float(row.option_cash_flow)
            cash += option_cash_flow
            signal = np.array([row.target_contracts_1, row.target_contracts_2], dtype=float)
            signal_spot = float(row.target_spot_units)
            new = np.zeros(2) if terminal else pending.copy() if effective_lag else signal.copy()
            new_spot = 0.0 if terminal else pending_spot if effective_lag else signal_spot
            if cost_points:
                futures_cost = float(np.abs(new - held).sum() * cost_points)
            else:
                futures_cost = float(np.abs(new - held).sum() * (fee + slip * fm))
            spot_trade_cash_flow = -spot * (new_spot - held_spot)
            spot_cost = abs(new_spot - held_spot) * spot * bps / 10000.0
            cash += spot_trade_cash_flow - futures_cost - spot_cost
            cumulative_futures_costs += futures_cost
            cumulative_spot_costs += spot_cost
            signed_mark = 0.0 if terminal else float(row.signed_option_mark)
            equity = cash + new_spot * spot + signed_mark
            discounted = equity * np.exp(-rate * row.elapsed_sessions / 244)
            rows.append({
                "cohort": cohort, "scenario": scenario, "date": row.date,
                "elapsed_sessions": row.elapsed_sessions, "cash": cash,
                "equity": equity, "discounted_equity": discounted,
                "discounted_pnl_change": discounted - previous_discounted,
                "contracts_1": new[0], "contracts_2": new[1],
                "spot_units": new_spot,
            })
            held, pending = new, signal
            held_spot, pending_spot = new_spot, signal_spot
            previous_prices, previous_spot = prices, spot
            previous_elapsed, previous_discounted = row.elapsed_sessions, discounted
        totals.append({
            "cohort": cohort, "scenario": scenario,
            "total_pnl": equity,
            "option_pnl": float(summary.option_pnl),
            "futures_pnl": cumulative_futures, "spot_pnl": cumulative_spot,
            "financing": cumulative_finance,
            "futures_costs": cumulative_futures_costs,
            "spot_costs": cumulative_spot_costs,
            "transaction_costs": cumulative_futures_costs + cumulative_spot_costs,
        })
    return pd.DataFrame(rows), pd.DataFrame(totals)


def check_and_analyze(output):
    base = output / "baseline"
    daily = pd.read_csv(base / "daily_ledger.csv", parse_dates=["date"])
    summary = pd.read_csv(base / "cohort_pnl.csv")
    cohorts = pd.read_csv(base / "cohort_audit.csv")
    coverage = pd.read_csv(output / "coverage.csv")
    assert cohorts.cohort.nunique() == int(coverage.eligible.sum()), "Historical batch is incomplete"
    assert (cohorts.status == "completed").all(), "Investigate unavailable calibration/hedge results"
    assert (cohorts.sample_curve_dates == 488).all()
    assert (pd.to_datetime(cohorts.sample_end) == pd.to_datetime(cohorts.entry_date)).all()
    assert (cohorts.optimizer_converged).all()
    assert (summary.groupby("cohort").scenario.nunique() == 3).all()
    assert (summary.groupby("cohort").exit_date.nunique() == 1).all()
    assert (daily.groupby(["cohort", "scenario"]).locked_carry.nunique() == 1).all()
    assert np.isfinite(daily[["cash", "equity", "option_mark", "contracts_1", "contracts_2"]]).all().all()
    final = daily.groupby(["cohort", "scenario"]).tail(1)
    assert (final[["contracts_1", "contracts_2", "option_mark"]] == 0).all().all()
    assert (final.exercise_now | final.expired).all()
    recalculated, totals = replay(daily, summary)
    comparison = daily.merge(recalculated, on=["cohort", "scenario", "date", "elapsed_sessions"], suffixes=("_original", "_replayed"))
    errors = {column: float((comparison[column+"_original"]-comparison[column+"_replayed"]).abs().max())
              for column in ("cash", "equity", "discounted_equity", "discounted_pnl_change", "contracts_1", "contracts_2")}
    assert max(errors.values()) < 1e-8, errors
    active = daily.loc[~(daily.exercise_now | daily.expired)]
    two = active.loc[active.scenario == "two_futures"]
    neutrality = float(two[["residual_slow_exposure", "residual_fast_exposure"]].abs().max().max())
    assert neutrality < 1e-8
    one = active.loc[active.scenario == "one_futures"]
    assert (one.residual_factor_variance <= one.unhedged_factor_variance+1e-8).all()
    diagnostics = {
        "completed_cohorts": len(cohorts), "active_dates": len(daily)//3,
        "coverage_exclusions": coverage.loc[~coverage.eligible].reason.value_counts().to_dict(),
        "independent_cash_replay_max_errors": errors,
        "two_futures_max_absolute_factor_residual": neutrality,
        "maximum_condition_number": float(active.hedge_condition_number.max()),
        "conditioning_warning_days": int(two.conditioning_warning.sum()),
        "kappa_gap_bound_cohorts": int(cohorts.kappa_gap_at_bound.sum()),
        "eta_fast_bound_cohorts": int(cohorts.eta_fast_at_bound.sum()),
        "exercise_cohorts": int((cohorts.exit_reason == "exercise").sum()),
        "expiry_cohorts": int((cohorts.exit_reason == "expiry").sum()),
    }
    write_json(output / "validation_checks.json", diagnostics)
    variants = []
    for label, lag, cost in [("same_close_zero_cost", 0, 0.), ("lag_one_session", 1, 0.),
                             ("cost_0.2_points_one_way", 0, .2)]:
        replayed, pnl = replay(daily, summary, lag=lag, cost_points=cost)
        if label != "same_close_zero_cost":
            path = output / label
            path.mkdir(exist_ok=True)
            replayed.to_csv(path / "daily_ledger.csv", index=False)
            pnl.to_csv(path / "cohort_pnl.csv", index=False)
        for scenario in ("no_hedge", "one_futures", "two_futures"):
            x = pnl.loc[pnl.scenario == scenario]
            changes = replayed.loc[(replayed.scenario == scenario) & (replayed.elapsed_sessions > 0), "discounted_pnl_change"]
            variants.append(dict(variant=label, scenario=scenario, sum_pnl=x.total_pnl.sum(),
                                 mean_pnl=x.total_pnl.mean(), cohort_pnl_std=x.total_pnl.std(ddof=1),
                                 daily_pnl_std=changes.std(ddof=1), daily_pnl_variance=changes.var(ddof=1),
                                 costs=x.transaction_costs.sum()))
    variants = pd.DataFrame(variants)
    unhedged = variants.loc[variants.scenario == "no_hedge"].set_index("variant").daily_pnl_variance
    variants["variance_reduction"] = 1-variants.daily_pnl_variance/variants.variant.map(unhedged)
    variants.to_csv(output / "execution_sensitivities.csv", index=False)
    exposures = active.groupby("scenario").agg(
        mean_factor_variance=("residual_factor_variance", "mean"),
        mean_unhedged_factor_variance=("unhedged_factor_variance", "mean"),
        rms_scale_exposure=("residual_scale_exposure", lambda x: np.sqrt(np.mean(x**2))),
        max_abs_scale_exposure=("residual_scale_exposure", lambda x: x.abs().max()),
    )
    exposures["factor_variance_reduction"] = 1-exposures.mean_factor_variance/exposures.mean_unhedged_factor_variance
    exposures.to_csv(output / "exposure_summary.csv")
    # A diagnostic first-order attribution using preceding holdings/sensitivities.
    # The remainder includes time decay, nonlinearity, basis, and model error.
    attribution = []
    for (cohort, scenario), group in daily.groupby(["cohort", "scenario"]):
        group = group.sort_values("elapsed_sessions")
        raw_change = group.equity.diff()-group.daily_financing+group.transaction_cost
        scale_move = group.residual_scale_exposure.shift(1)*group.spot.pct_change()
        factor_move = (group.residual_slow_exposure.shift(1)*group.filtered_slow_state.diff()
                       +group.residual_fast_exposure.shift(1)*group.filtered_fast_state.diff())
        for date, actual, scale, factor in zip(group.date.iloc[1:], raw_change.iloc[1:], scale_move.iloc[1:], factor_move.iloc[1:]):
            attribution.append(dict(cohort=cohort, scenario=scenario, date=date, pnl_before_financing_costs=actual,
                                    first_order_spot_move=scale, first_order_factor_move=factor,
                                    remainder=actual-scale-factor))
    attribution = pd.DataFrame(attribution)
    attribution.to_csv(output / "first_order_attribution.csv", index=False)
    correlations = {scenario: float(g.pnl_before_financing_costs.corr(g.first_order_spot_move))
                    for scenario, g in attribution.groupby("scenario")}
    write_json(output / "spot_attribution_correlations.json", correlations)
    print(pd.DataFrame(variants).to_string(index=False), flush=True)
    print(diagnostics, flush=True)
    make_figures(output, daily, summary, cohorts, variants, exposures)


def make_figures(output, daily, summary, cohorts, variants, exposures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["No hedge", "One futures", "Two futures"]
    names = ["no_hedge", "one_futures", "two_futures"]
    colors = ["#64748b", "#dc7830", "#24779b"]
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    pivot = summary.pivot(index="cohort", columns="scenario", values="total_pnl").sort_index()[names]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    positions = np.arange(len(pivot))
    for i, (name, label, color) in enumerate(zip(names, labels, colors)):
        axes[0].bar(positions+(i-1)*.25, pivot[name], width=.25, label=label, color=color)
        axes[1].plot(positions, pivot[name].cumsum(), label=label, color=color, marker="o", markersize=3)
    for ax in axes:
        ax.axhline(0, color="#334155", linewidth=.6)
        ax.set_xticks(positions, [d[:7] for d in pivot.index], rotation=60, ha="right")
        ax.grid(axis="y", alpha=.2)
        ax.set_ylabel("Option points")
    axes[0].set_title("Monthly cohort P&L — long optional component, zero costs")
    axes[0].legend(ncol=3)
    axes[1].set_title("Cumulative sum of separate cohort P&L (not a funded strategy NAV)")
    fig.savefig(figures / "cohort_pnl.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    baseline = variants.loc[variants.variant == "same_close_zero_cost"].set_index("scenario").loc[names]
    for ax, values, title, ylabel in [
        (axes[0], baseline.daily_pnl_std, "Daily total P&L risk", "Discounted P&L std, points"),
        (axes[1], np.sqrt(exposures.loc[names].mean_factor_variance), "Residual carry-factor risk", "Model one-session std, points"),
        (axes[2], exposures.loc[names].rms_scale_exposure, "Residual spot-scale exposure", "Points per 100% scale move"),
    ]:
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(figures / "hedge_sufficiency.png", dpi=160)
    plt.close(fig)


def write_report(output):
    summary = pd.read_csv(output / "baseline/cohort_pnl.csv")
    aggregate = pd.read_csv(output / "baseline/aggregate.csv").set_index("scenario")
    cohorts = pd.read_csv(output / "baseline/cohort_audit.csv")
    exposures = pd.read_csv(output / "exposure_summary.csv").set_index("scenario")
    sensitivity = pd.read_csv(output / "execution_sensitivities.csv")
    checks = json.loads((output / "validation_checks.json").read_text())
    numerical = pd.read_csv(output / "selected_numerical_checks.csv")
    fine = pd.read_csv(output / "fine/cohort_pnl.csv").set_index("scenario")
    pilot = summary.loc[summary.cohort == fine.cohort.iloc[0]].set_index("scenario")
    names = {"no_hedge": "No hedge", "one_futures": "One futures", "two_futures": "Two futures"}
    lines = [
        "# Historical carry-put hedge validation", "",
        "The monthly back-test is complete. The proposed futures hedges remove local carry-factor risk, "
        "but **do not provide sufficient total hedging** for the long optional component: they introduce "
        "large spot exposure and substantially increase total P&L volatility.", "",
        "## Sample and conventions", "",
        f"The cache ends on 2026-08-21. Of 48 scheduled monthly entries, 24 lacked the required 488-date history. "
        f"All remaining **24 cohorts**, entered from {cohorts.entry_date.min()} through {cohorts.entry_date.max()}, "
        f"had complete required data and completed successfully. The last actual exit was {cohorts.exit_date.max()}. "
        "There were 22 early exercises and two expiries (the August 2024 and May 2025 entries), "
        "with 245 close-to-close P&L observations per scenario.", "",
        "Each cohort starts only on the first trading session after monthly expiry, references the next monthly "
        "IM contract, calibrates using 488 accepted curve dates through inception, freezes parameters, and updates "
        "states with forward filtering. The two-futures pair is the option underlying plus the longest-dated later "
        "IM contract quoted at inception. No early-exercise replacement cohort is created.", "",
        "Results below use one long optional component, continuous hedge units, multipliers of 1, zero transaction "
        "costs, same-close execution, and 1.4% financing on the trading-session/244 clock. The separate linear "
        "futures leg is excluded. P&L is in option points; sums are across separate cohorts, not a funded strategy NAV.", "",
        "## Baseline results", "",
        "| Scenario | Sum P&L | Mean cohort P&L | Cohort P&L std | Daily discounted P&L std | Worst cohort P&L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in names.items():
        row = aggregate.loc[name]
        lines.append(f"| {label} | {row.sum_cohort_pnl:.2f} | {row.mean_cohort_pnl:.2f} | {row.std_cohort_pnl:.2f} | "
                     f"{np.sqrt(row.daily_discounted_pnl_variance):.2f} | {row.worst_cohort_pnl:.2f} |")
    lines += ["", "The one- and two-futures portfolios have roughly 15.86 and 15.36 times the unhedged daily "
              "variance, respectively (about four times the standard deviation). Two futures modestly reduce risk "
              "relative to one futures, but neither reduces total risk relative to the unhedged option.", "",
              f"![Monthly and cumulative cohort P&L]({(output / 'figures/cohort_pnl.png').resolve().as_posix()})", "",
              "## Why factor neutrality does not provide total hedging", "",
              "The option is homogeneous: `V = S * v(x_s, x_f)`. Its value still responds to spot scaling even "
              "though spot volatility cancels from the pricing equation. The long futures positions needed to "
              "neutralize carry sensitivity add substantial spot sensitivity.", "",
              "| Scenario | Reduction in modeled carry-factor variance | RMS spot-scale sensitivity | Approx. first-order P&L for a 1% scale move |",
              "|---|---:|---:|---:|"]
    for name, label in names.items():
        row = exposures.loc[name]
        lines.append(f"| {label} | {row.factor_variance_reduction:.4%} | {row.rms_scale_exposure:.2f} | {row.rms_scale_exposure*.01:.2f} |")
    lines += ["", "The two-futures residual factor sensitivities are below 5.7e-14 in absolute value. "
              "Yet its spot-scale exposure is approximately 100 times that of the unhedged option. "
              "A first-order diagnostic using preceding holdings shows about 0.991 correlation between "
              "hedged daily P&L before financing/costs and the predicted spot-scale component. "
              "The residual includes time decay, basis, nonlinearity, and model error; this is an attribution "
              "diagnostic, not a separate identification of causal returns.", "",
              "Higher realized hedge P&L therefore must not be interpreted as better hedge sufficiency or as "
              "evidence of carry-option mispricing. Most of the P&L difference comes from the futures positions.", "",
              f"![Hedge sufficiency]({(output / 'figures/hedge_sufficiency.png').resolve().as_posix()})", "",
              "## P&L reconciliation", "",
              "| Scenario | Option payoff less premium | Futures P&L | Financing | Total P&L |",
              "|---|---:|---:|---:|---:|"]
    components = summary.groupby("scenario")[["option_pnl", "futures_pnl", "financing", "total_pnl"]].sum()
    for name, label in names.items():
        row = components.loc[name]
        lines.append(f"| {label} | {row.option_pnl:.6f} | {row.futures_pnl:.6f} | {row.financing:.6f} | {row.total_pnl:.6f} |")
    lines += ["", "An independent cash replay reproduces daily equity to less than 2e-13 points. "
              "The checks verify prior-position futures P&L, financing, terminal unwind, common exit dates, "
              "fixed carry locks, 488-date calibration windows, and complete coverage. All terminal positions "
              "and option marks are zero.", "",
              "## Execution sensitivities", "",
              "The same option policy and model signals are replayed without refitting or repricing. "
              "The one-session lag delays hedge execution only; exercise and forced closing remain immediate. "
              "The 0.2-point one-way cost is an illustrative sensitivity, not an observed brokerage fee.", "",
              "| Variant | Scenario | Sum P&L | Mean cohort P&L | Daily discounted P&L std |",
              "|---|---|---:|---:|---:|"]
    for row in sensitivity.loc[sensitivity.variant != "same_close_zero_cost"].itertuples():
        label = {"lag_one_session": "One-session hedge lag", "cost_0.2_points_one_way": "0.2-point one-way cost"}[row.variant]
        lines.append(f"| {label} | {names[row.scenario]} | {row.sum_pnl:.2f} | {row.mean_pnl:.2f} | {row.daily_pnl_std:.2f} |")
    lines += ["", "Delaying hedge execution reduces realized hedge P&L materially, while total volatility "
              "remains much higher than for the unhedged option. The illustrative cost level does not change "
              "the hedge-sufficiency conclusion.", "",
              "## Calibration and numerical checks", "",
              "All 288 optimizer starts across 24 cohorts converged. No cohort reached the configured "
              "kappa-gap or eta-fast upper bound. The maximum hedge-matrix condition number was 119.94; "
              "no singular pair or conditioning warning at the configured 1,000 threshold occurred.", "",
              "Historical calibration uses an explicitly selected optional Numba likelihood backend, with "
              "Numba already installed in spyder-env. Independent observation errors permit sequential scalar "
              "Kalman conditioning, which is algebraically equivalent to the dense reference likelihood. "
              "Forty historical parameter-set comparisons and synthetic tests verified likelihood agreement; "
              "finite-difference gradient discrepancies were below 1e-6. The original NumPy backend remains "
              "the estimator default. The full pilot fit changed log likelihood by 2.3e-9 and cohort P&L by "
              "less than 0.000031 points compared with the original backend. Pilot calibration runtime fell "
              "from 105.8 to 1.7 seconds.", "",
              "The baseline uses a 301 × 401 factor grid and 43-point quadrature. The finer checks use "
              "451 × 601 and 61-point quadrature. The first cohort was rerun for its entire life:", "",
              "| Scenario | Fine minus baseline premium | Fine minus baseline lifetime P&L | Fine minus baseline daily P&L std |",
              "|---|---:|---:|---:|"]
    for name, label in names.items():
        a, b = fine.loc[name], pilot.loc[name]
        lines.append(f"| {label} | {a.initial_premium-b.initial_premium:.6f} | {a.total_pnl-b.total_pnl:.6f} | "
                     f"{a.daily_discounted_pnl_std-b.daily_discounted_pnl_std:.6f} |")
    lines += ["", "The pilot exit date is unchanged. Selected additional historical states check the final "
              "inception, largest premium, largest conditioning value with a nonzero option mark, largest absolute fast factor, and "
              "nearest exercise boundary:", "",
              "| Selection | Cohort / valuation date | Fine minus baseline price | Relative difference | Exercise decision unchanged |",
              "|---|---|---:|---:|---|"]
    for row in numerical.itertuples():
        lines.append(f"| {row.selection} | {row.cohort} / {row.date} | {row.price_difference:.6f} | "
                     f"{row.relative_price_difference:.3%} | {row.base_exercise == row.fine_exercise} |")
    lines += ["", "Across the selected states, option-price changes are below 0.017 points and exercise decisions "
              "are unchanged. The largest sampled hedge-position change is about 0.0134 futures units near "
              "the exercise boundary. These checks support the broad risk conclusion, but do not establish exact convergence "
              "of every hedge ratio or every cohort's P&L. The pilot's roughly 0.69-point hedged P&L change "
              "shows that reporting many decimal places would overstate numerical precision. A full finer-grid "
              "batch was not run. The regression suite passed **71 tests** in spyder-env.", "",
              "## Interpretation and next decision", "",
              "The agreed factor-only futures hedges are effective at the factor objective and insufficient "
              "at the total-risk objective. A next research step would be to either include spot risk in a "
              "one-/two-futures minimum-total-variance hedge, or introduce an additional spot/ETF hedge alongside "
              "the two factor-neutral futures. Neither alternative was substituted into this experiment.", "",
              "Daily option marks and inception premiums are model values, and historical OU dynamics are "
              "provisionally treated as risk-neutral. Observed immediate exercise values are compared with model "
              "continuation. Actual fills, settlement cash flows, margin funding, asymmetric borrowing rates, "
              "liquidity constraints, and option fees are not modeled. The strict Demo calendar's 2027 maturities "
              "use its provisional extension. Twenty-four cohorts are too few for strong tail-risk or expected-return "
              "claims. Future exercises are not used to set trading decisions; only scheduled-life coverage is "
              "used as a sample eligibility requirement.", "",
              "## Files", "",
              "- [Monthly P&L](baseline/cohort_pnl.csv)",
              "- [Daily ledger](baseline/daily_ledger.csv)",
              "- [Cohort calibration and exit audit](baseline/cohort_audit.csv)",
              "- [Data coverage](coverage.csv)",
              "- [Execution sensitivities](execution_sensitivities.csv)",
              "- [Numerical checks](selected_numerical_checks.csv)",
              "- [Independent validation checks](validation_checks.json)",
              "- [Source/input manifest](baseline/run_manifest.json)", ""]
    (output / "historical_validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    base_manifest = json.loads((output / "baseline/run_manifest.json").read_text())
    protected = {}
    for name, expected in base_manifest["sha256"].items():
        normalized = name.replace("\\", "/")
        if ("/data/raw/" in normalized or "/src/" in normalized
                or Path(name).name in ("engine.py", "market.py", "calendar_utils.py", "demo_quality.py")):
            actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Input/core source changed after the historical run: {name}")
            protected[name] = actual
    artifacts = list(output.glob("*.csv")) + list(output.glob("*.json")) + list(output.glob("*.md"))
    artifacts += list((output / "figures").glob("*.png"))
    artifacts += list((output / "baseline").glob("*.csv")) + list((output / "fine").glob("*.csv"))
    artifacts = [p for p in artifacts if p.name != "validation_manifest.json"]
    source_paths = list(Path(__file__).parent.glob("*.py"))
    write_json(output / "validation_manifest.json", {
        "status": "completed_baseline_with_pilot_and_selected_state_numerical_checks",
        "baseline_cohorts": len(cohorts), "selected_numerical_checks": len(numerical),
        "regression_tests_passed": 71, "full_finer_grid_batch_run": False,
        "unchanged_raw_inputs_and_core_sources_sha256": protected,
        "validation_scripts_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "artifacts_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        write_report(args.output_dir)
    else:
        check_and_analyze(args.output_dir)


if __name__ == "__main__":
    main()
