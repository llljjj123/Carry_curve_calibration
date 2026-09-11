# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -0.082925 | -0.082925 | 0.00% |
| short_one_futures_spot | -19.284122 | -19.284122 | 59.07% |
| short_two_futures_spot | -18.938863 | -18.938863 | 59.10% |
| short_one_futures_only | -29.426356 | -29.426356 | -1308.77% |
| short_two_futures_only | -29.050248 | -29.050248 | -1265.62% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
