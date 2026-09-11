# Carry-put monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

Each cohort has the same option and exit date in all three scenarios.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| no_hedge | -10.514159 | -10.514159 | 0.00% |
| one_futures | 94.237241 | 94.237241 | -1115.98% |
| two_futures | 92.673768 | 92.673768 | -1078.81% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
