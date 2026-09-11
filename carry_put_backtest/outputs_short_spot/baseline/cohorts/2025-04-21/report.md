# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | 13.420469 | 13.420469 | 0.00% |
| short_one_futures_spot | -12.709806 | -12.709806 | 74.31% |
| short_two_futures_spot | -12.376750 | -12.376750 | 75.90% |
| short_one_futures_only | -99.929244 | -99.929244 | -11394.19% |
| short_two_futures_only | -98.970762 | -98.970762 | -11204.27% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
