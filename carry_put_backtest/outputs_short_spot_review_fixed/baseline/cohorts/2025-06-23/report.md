# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -6.369244 | -6.369244 | 0.00% |
| short_one_futures_spot | -4.584733 | -4.584733 | 85.81% |
| short_two_futures_spot | -5.084114 | -5.084114 | 87.90% |
| short_one_futures_only | -81.712714 | -81.712714 | -299.08% |
| short_two_futures_only | -80.345613 | -80.345613 | -284.44% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
