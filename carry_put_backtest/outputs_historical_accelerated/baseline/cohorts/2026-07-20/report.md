# Carry-put monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

Each cohort has the same option and exit date in all three scenarios.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| no_hedge | -7.194631 | -7.194631 | 0.00% |
| one_futures | 232.184361 | 232.184361 | -5268.76% |
| two_futures | 229.087641 | 229.087641 | -5082.97% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
