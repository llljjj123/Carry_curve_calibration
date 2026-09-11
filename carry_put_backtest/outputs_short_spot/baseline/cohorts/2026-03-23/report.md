# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -31.013713 | -31.013713 | 0.00% |
| short_one_futures_spot | -7.948323 | -7.948323 | 86.15% |
| short_two_futures_spot | -8.549973 | -8.549973 | 86.65% |
| short_one_futures_only | -145.795322 | -145.795322 | 99.50% |
| short_two_futures_only | -143.887363 | -143.887363 | 99.55% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
