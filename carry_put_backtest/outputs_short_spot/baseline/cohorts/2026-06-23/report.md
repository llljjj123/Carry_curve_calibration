# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -20.639334 | -20.639334 | undefined |
| short_one_futures_spot | -7.152542 | -7.152542 | undefined |
| short_two_futures_spot | -7.447666 | -7.447666 | undefined |
| short_one_futures_only | -41.600457 | -41.600457 | undefined |
| short_two_futures_only | -41.126345 | -41.126345 | undefined |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
