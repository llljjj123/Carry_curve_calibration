# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | 16.686839 | 16.686839 | 0.00% |
| short_one_futures_spot | 0.277154 | 0.277154 | 79.79% |
| short_two_futures_spot | 0.497880 | 0.497880 | 80.78% |
| short_one_futures_only | -60.940997 | -60.940997 | -1283.38% |
| short_two_futures_only | -59.761683 | -59.761683 | -1237.90% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
