# Independent review after bug fixes

Date: 2026-09-11. Review only: no implementation or historical output changes.

The original fixes substantially improve the implementation. Peak aggregation,
pooled risk metrics, regenerated sensitivities, explicit full/pilot cohort sets,
and separate execution/analysis manifests are present. Four remaining issues
were reproduced using in-memory data or temporary directories.

## Remaining findings

### 1. P2 — NaN errors bypass accounting validation

Location: `analyze_short_spot.py:193-200` and analogous neutrality checks.

Replacing every saved cash value with NaN gives `_errors` a cash error of NaN.
The condition `max(errors.values()) >= 1e-8` evaluates false for this result,
so the accounting gate does not reject it. Pandas maxima also skip isolated
NaNs, allowing missing summary totals or diagnostics to disappear from checks.
The same problem affects `total_errors.max()` and neutrality comparisons.

Require finite values in economically required columns before replay, reject
nonfinite error arrays explicitly, and only then apply tolerance checks. Permit
NaN only in specifically optional diagnostics, such as terminal conditioning
fields. Add all-NaN and single-NaN cash, total-P&L and residual regressions.
The supplied baseline core cash/position/mark columns contain no nonfinite values.

### 2. P2 — Test evidence does not identify all tested source, or source changes during execution

Location: `test_evidence.py:15-21,24-39,52`.

`source_identity` hashes only back-test Python files and back-test tests. The
recorded full suite also tests pricing, Demo and the OU estimator; changes to
those sources/tests leave evidence classified as passed. A temporary fixture
with changed pricing source reproduced this false-current classification.

Identity is also captured only AFTER subprocess completion. A subprocess that
changes a back-test source file while running receives evidence bound to the
changed file, and classification reports passed. It cannot establish that the
changed source was what the subprocess loaded/tested.

Hash all relevant tested source/test dependencies, including required local
calendar/config data, before execution. Compare identities after execution and
report changed-during-run evidence as invalid/stale. Keep the actual execution
log and exit status distinct from source-validity status. Record the interpreter
used by the command rather than assuming the wrapper's interpreter for arbitrary
commands. Add dependency-change and during-run-change regressions.

### 3. P2 — Missing execution sources are silently treated as matching provenance

Location: `analyze_short_spot.py:304-315,334-342`.

`_execution_source_comparison` only appends existing .py files. A manifest naming
a missing source returns an empty comparison list; the downstream absence of
mismatches is classified as `matched_current_source`. Empty/missing hash maps
have the same issue. Thus unavailable source is incorrectly stronger evidence
than an explicit mismatch.

Represent missing sources, absent required hashes and unavailable manifests as
explicit unresolved statuses. Require a nonempty, complete expected dependency
set before claiming a match. Distinguish execution-source comparisons from raw
input comparisons; check relevant recorded non-Python dependencies separately.
Add missing-file and empty-map cases. Both delivered baseline/fine manifests
currently have 39 existing Python source comparisons and zero mismatches, so
this defect does not undermine that observed current-source match.

### 4. P2 — Exit consistency is checked within summaries, not across artifacts

Location: `analyze_short_spot.py:164-183`.

Changing every summary exit_date to 2099-01-01 still passes `validate_batch` in
full mode. Summaries agree with one another, and ledger final rows are terminal,
but neither is compared with audit exits or ledger dates/reasons. Reports can
therefore pass with summaries from a different exercise path. Matching scenario
date lists also does not validate a common omitted session.

Join summary and audit entry/exit fields to each ledger's first/last rows;
validate exit reason against terminal exercise/expiry flags, require no earlier
terminal row, and validate the session sequence/elapsed counts against the
calendar. Add mismatched exit date/reason and commonly missing-session tests.

## Lower-priority reporting/comparison issues

- `analyze_short_spot.py:365` requests `positive_option_price`, but the ledger
  column is `positive_long_option_price`. `initial_premium` and `option_pnl` are
  summary fields rather than daily fields. These comparisons are silently
  skipped, so the automatic audit does not cover all the quantities the handoff
  requests. Compare summaries separately and require expected field names;
  report intentionally unavailable fields explicitly.
- `analyze_short_spot.py:432` uses unescaped pipes in `Max |slow|` and
  `Max |fast|` Markdown headers, creating extra table columns. Use `Max abs slow`
  and `Max abs fast`, or escape the embedded pipes.

## Checks performed and interpretation

- Independent targeted regression command:
  `D:\miniforge3\envs\spyder-env\python.exe -B -m pytest carry_put_backtest/tests -q -p no:cacheprovider`
  Result: **58 passed in 10.78 seconds**, exit code 0. The full 93-test suite
  was not rerun in this review; its supplied evidence currently classifies as
  passed under the existing checker, subject to finding 2.
- All **524** historical artifact hashes match the preserved audit snapshot.
- Fresh baseline and fine aggregate execution manifests: **39/39 Python source
  hashes match current files** in each. No execution manifest was rewritten.
- All **1,345** old/new daily keys match. Every shared column agrees exactly;
  the new ledger adds six aliases/attribution columns. This was checked directly
  after sorting by cohort/scenario/date/elapsed sessions, independently of the
  automatic comparison's incomplete field list.
- Current baseline totals remain -80.082507 unhedged, -276.617669 one futures
  plus spot, -272.485234 two futures plus spot, and -1285.773227/-1263.259184
  for the futures-only controls.

No new hedge-sign or funded-cash-accounting bug was found. The economic
conclusions remain unchanged: one futures plus spot is the simpler near-equal
risk-reduction candidate; two futures has slightly lower measured daily risk.
The findings concern validation accepting invalid inputs/evidence, and should
be fixed before the validation layer is considered complete. The original
historical manifest discrepancy remains preserved rather than retroactively
repaired; the new matching-source run supplies fresh evidence.
