# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -13.006749 | -13.006749 | 0.00% |
| short_one_futures_spot | -52.133846 | -52.133846 | 31.82% |
| short_two_futures_spot | -51.424448 | -51.424448 | 32.11% |
| short_one_futures_only | 193.727733 | 193.727733 | -2188.20% |
| short_two_futures_only | 190.206480 | 190.206480 | -2118.78% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
