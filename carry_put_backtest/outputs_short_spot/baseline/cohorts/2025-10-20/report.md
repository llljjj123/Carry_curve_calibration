# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | 10.514159 | 10.514159 | 0.00% |
| short_one_futures_spot | -7.613826 | -7.613826 | 83.15% |
| short_two_futures_spot | -7.621006 | -7.621006 | 83.46% |
| short_one_futures_only | -94.237241 | -94.237241 | -1115.98% |
| short_two_futures_only | -92.673768 | -92.673768 | -1078.81% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
