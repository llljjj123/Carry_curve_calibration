# Carry-put monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

Each cohort has the same option and exit date in all three scenarios.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| no_hedge | 3.816268 | 3.816268 | 0.00% |
| one_futures | -18.933094 | -18.933094 | -2174.59% |
| two_futures | -19.310913 | -19.310913 | -2103.75% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
