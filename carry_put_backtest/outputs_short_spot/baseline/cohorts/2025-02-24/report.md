# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -3.816268 | -3.816268 | 0.00% |
| short_one_futures_spot | -14.572538 | -14.572538 | 85.76% |
| short_two_futures_spot | -13.758626 | -13.758626 | 86.54% |
| short_one_futures_only | 18.933094 | 18.933094 | -2174.59% |
| short_two_futures_only | 19.310913 | 19.310913 | -2103.75% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
