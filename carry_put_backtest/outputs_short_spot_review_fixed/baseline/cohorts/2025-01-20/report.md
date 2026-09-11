# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | 9.230977 | 9.230977 | 0.00% |
| short_one_futures_spot | 3.720374 | 3.720374 | 94.42% |
| short_two_futures_spot | 3.752634 | 3.752634 | 94.64% |
| short_one_futures_only | -101.894671 | -101.894671 | -3675.59% |
| short_two_futures_only | -100.403215 | -100.403215 | -3550.12% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
