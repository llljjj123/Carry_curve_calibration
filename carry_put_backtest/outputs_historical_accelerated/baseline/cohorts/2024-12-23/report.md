# Carry-put monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

Each cohort has the same option and exit date in all three scenarios.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| no_hedge | -12.484425 | -12.484425 | 0.00% |
| one_futures | -154.482004 | -154.482004 | -9129.76% |
| two_futures | -151.832597 | -151.832597 | -8782.14% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
