# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -23.306208 | -23.306208 | 0.00% |
| short_one_futures_spot | -2.511778 | -2.511778 | 41.74% |
| short_two_futures_spot | -2.618743 | -2.618743 | 49.92% |
| short_one_futures_only | -96.041546 | -96.041546 | -2436.06% |
| short_two_futures_only | -93.889879 | -93.889879 | -2311.09% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
