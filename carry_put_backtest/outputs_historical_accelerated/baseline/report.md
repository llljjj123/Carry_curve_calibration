# Carry-put monthly cohort back-test

Completed cohorts: 24. Scheduled cohorts: 24.

Each cohort has the same option and exit date in all three scenarios.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| no_hedge | 3.336771 | 80.082507 | 0.00% |
| one_futures | 53.573884 | 1285.773227 | -1485.75% |
| two_futures | 52.635799 | 1263.259184 | -1436.41% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
