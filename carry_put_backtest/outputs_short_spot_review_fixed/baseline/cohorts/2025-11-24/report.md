# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -19.813406 | -19.813406 | 0.00% |
| short_one_futures_spot | -6.626962 | -6.626962 | 88.05% |
| short_two_futures_spot | -6.381138 | -6.381138 | 87.25% |
| short_one_futures_only | -96.922583 | -96.922583 | -365.17% |
| short_two_futures_only | -95.321345 | -95.321345 | -354.10% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
