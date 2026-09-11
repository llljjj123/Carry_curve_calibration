# Carry-put monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

Each cohort has the same option and exit date in all three scenarios.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| no_hedge | 31.013713 | 31.013713 | 0.00% |
| one_futures | 145.795322 | 145.795322 | 99.50% |
| two_futures | 143.887363 | 143.887363 | 99.55% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
