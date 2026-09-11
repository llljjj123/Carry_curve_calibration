# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -11.574681 | -11.574681 | 0.00% |
| short_one_futures_spot | -2.082650 | -2.082650 | 89.57% |
| short_two_futures_spot | -1.619703 | -1.619703 | 91.64% |
| short_one_futures_only | -18.553569 | -18.553569 | -2127.11% |
| short_two_futures_only | -17.783280 | -17.783280 | -2039.94% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
