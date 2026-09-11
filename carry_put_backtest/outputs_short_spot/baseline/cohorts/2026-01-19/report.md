# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -11.292107 | -11.292107 | 0.00% |
| short_one_futures_spot | -29.991143 | -29.991143 | 69.12% |
| short_two_futures_spot | -30.477473 | -30.477473 | 69.22% |
| short_one_futures_only | 4.297212 | 4.297212 | -38.06% |
| short_two_futures_only | 3.252473 | 3.252473 | -33.48% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
