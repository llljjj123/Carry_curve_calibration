# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -3.368363 | -3.368363 | 0.00% |
| short_one_futures_spot | -20.518696 | -20.518696 | 38.50% |
| short_two_futures_spot | -20.533036 | -20.533036 | 38.73% |
| short_one_futures_only | -102.427606 | -102.427606 | -960.79% |
| short_two_futures_only | -101.584123 | -101.584123 | -932.94% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
