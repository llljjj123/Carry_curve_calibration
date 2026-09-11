# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -7.997283 | -7.997283 | 0.00% |
| short_one_futures_spot | -29.276904 | -29.276904 | 27.71% |
| short_two_futures_spot | -28.987967 | -28.987967 | 28.41% |
| short_one_futures_only | -96.725719 | -96.725719 | -1233.54% |
| short_two_futures_only | -94.586278 | -94.586278 | -1170.75% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
