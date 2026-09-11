# Short spot post-review bug-fix handoff

This bundle implements the findings in `short_spot_astra_review.md`. The corrected bundle is ready for a new independent review.

## Changed files

- `carry_put_backtest/engine.py`: explicit study/scenario definitions, short-side hedge conversion, funded spot ledger, residuals, costs, lag, and summaries.
- `carry_put_backtest/__main__.py` and `__init__.py`: study and spot-cost configuration/API exposure.
- `carry_put_backtest/reporting.py`: flow sums, individual-cohort peak maxima, and pooled risk metrics.
- `carry_put_backtest/analyze_historical.py`: independent funded replay.
- `carry_put_backtest/analyze_short_spot.py`: strict full/pilot validation, regenerated sensitivities, immutable analysis provenance, unit-safe diagnostics, comparisons, and reports.
- `carry_put_backtest/test_evidence.py`: explicit subprocess test evidence with logs, exits, timestamps, and source identity.
- `carry_put_backtest/provenance.py`: immutable historical evidence hash audit.
- `carry_put_backtest/historical_validation.py`: short-study selection and sensitivity configuration.
- `carry_put_backtest/tests/test_short_spot.py`: synthetic economic-identity tests.

## Commands and results

Interpreter: `D:\miniforge3\envs\spyder-env\python.exe`
Focused/regression command: `D:\miniforge3\envs\spyder-env\python.exe -B -m pytest carry_put_backtest/tests carry_put_pricing/tests Demo/tests im_2factor_ou_carry/tests -q -p no:cacheprovider`
Historical baseline: `python -B -u -m carry_put_backtest.historical_validation --phase baseline --study short_with_spot --output-dir carry_put_backtest/outputs_short_spot_review_fixed`.
Historical analyzer: `python -B -m carry_put_backtest.analyze_short_spot --output-dir carry_put_backtest/outputs_short_spot_review_fixed --mode full`.

Full regression result: **93 passed in 313.23 seconds**, exit code 0. The full
combined stdout/stderr is preserved in `test_results.log`; this count is parsed
from that execution and is not hard-coded.

Independent replay maximum errors: `{"cash": 1.3642420526593924e-12, "contracts_1": 0.0, "contracts_2": 0.0, "discounted_equity": 9.103828801926284e-13, "discounted_pnl_change": 9.059419880941277e-13, "equity": 9.112710586123285e-13, "spot_units": 0.0}`.
Cohorts: `24` completed of `48` scheduled; exercises `22`, expiries `2`.
Baseline short control sums: `short_no_hedge=-80.082507, short_one_futures_only=-1285.773227, short_two_futures_only=-1263.259184`; compare futures-only controls with the negative legacy long study.

## Review findings and fixes

1. Execution manifests are never rewritten by analysis. `analysis_manifest.json`
   separately records commands, source/input hashes, outcomes, and current-source
   comparisons. The preserved old tree has 524 artifact hashes and 72 exact
   checkpoint mismatches: 24 each for `engine.py`, `analyze_historical.py`, and
   `analyze_short_spot.py`; `reporting.py` matches.
2. Test evidence comes from an explicit subprocess and classifies missing,
   stale, failed, and passed states. A failing-subprocess regression verifies
   that nonzero exits are persisted as failures.
3. Lag and cost variants are regenerated on every analyzer invocation and
   checked for exact baseline row/pair keys plus terminal liquidation.
4. Full mode enforces exact eligible cohort equality across coverage, audit,
   summaries, and daily ledgers. Pilot mode requires explicit cohort IDs and is
   labeled partial.
5. Flow fields sum, while peak borrowing and peak spot notional use the maximum
   individual cohort over the complete path, including terminal rows. These are
   not portfolio funding requirements.
6. Slow-factor, fast-factor, cash-scale, and spot-delta residuals are reported
   separately with units. Headline risk is pooled discounted daily P&L standard
   deviation; mean cohort standard deviation is separately labeled.

## Corrected current-source results

| Scenario | Sum P&L | Pooled daily std |
|---|---:|---:|
| Short no hedge | -80.082507 | 7.360822 |
| Short 1 futures + spot | -276.617669 | 3.911521 |
| Short 2 futures + spot | -272.485234 | 3.890142 |
| Short 1 futures only | -1285.773227 | 29.311837 |
| Short 2 futures only | -1263.259184 | 28.852239 |

The funded-spot scenarios' maximum individual-cohort borrowing/spot notional
are 4,669.808/4,713.348 and 4,627.937/4,671.740 points. Maximum cash-scale
residual is `4.55e-13`; the two-futures slow/fast residual maxima are
`5.68e-14`/`4.26e-14`. One-session lag gives P&L -164.858/-162.174 and pooled
daily standard deviations 4.317/4.277. Illustrative costs give
-300.483/-296.124 and 3.951/3.930.

All 1,345 old/new cohort-scenario-date rows match exactly for available marks,
exercise decisions, holdings, cash/equity, P&L changes, and residuals; coverage
also matches. Independent comparison against the legacy long study gives exact
negative symmetry for short no-hedge and both futures-only controls across 269
rows per scenario. The 2025-07-21 fine pilot changes funded-spot P&L by
+0.351467/+0.353215 points and preserves the exit outcome.

## Sign and unit conventions

The pricer returns positive long-option values and sensitivities. The study uses side `s=-1`; short-option premium receipt is positive, exercise payment is negative, and the live option mark is negative. Futures counts are positive long/negative short. Spot units are continuous hypothetical index units. Point P&L is monetary only when configured multipliers and units are one.

Spot is sized after actual futures rounding. It is funded from cash, and daily spot P&L is attributed through the funded asset rather than credited to cash separately. Dividend cash flow is exactly zero.

## Review points

Check the three primary scenarios against their two futures-only controls, especially residual factor/scale exposure, borrowing, spot notional, and total P&L. Verify the short unhedged and futures-only controls against the negative legacy long study. Inspect singular/near-singular handling and the fine pilot before drawing numerical conclusions.

The market convention combines zero-dividend spot, common financing, and observed nonzero futures carry; it need not be arbitrage-consistent. The historical OU fit remains a provisional risk-neutral input. No strategy optimization, spot-only hedge, ETF, or minimum-total-variance objective was introduced.

## Artifacts

- [Results report](outputs_short_spot_review_fixed/short_spot_results_report.md)
- [Baseline ledger](outputs_short_spot_review_fixed/baseline/daily_ledger.csv)
- [Baseline cohort P&L](outputs_short_spot_review_fixed/baseline/cohort_pnl.csv)
- [Validation checks](outputs_short_spot_review_fixed/validation_checks.json)
- [Analysis manifest](outputs_short_spot_review_fixed/analysis_manifest.json)
- [Historical evidence audit](outputs_short_spot_review_fixed/historical_evidence_audit.json)
- [Old/new comparison](outputs_short_spot_review_fixed/old_new_comparison.json)
- [Execution sensitivities](outputs_short_spot_review_fixed/execution_sensitivities.csv)
- [Fine-grid pilot](outputs_short_spot_review_fixed/fine/cohort_pnl.csv)
- [Test evidence](outputs_short_spot_review_fixed/test_results.json)

The planned fine-grid pilot was completed for the 2025-07-21 cohort. A full finer-grid historical batch was not run and is not claimed.

## Test evidence

Status: `passed`; exit code: `0`; parsed passed count: `93`.

## Provenance resolution

The old output remains immutable and its exact execution source is unresolved. The fresh baseline is generated from current source; numerical agreement cannot retroactively repair the old manifest.

Old/new comparison status: `compared`.
