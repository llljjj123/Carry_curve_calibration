# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | 12.484425 | 12.484425 | 0.00% |
| short_one_futures_spot | 3.054578 | 3.054578 | 84.38% |
| short_two_futures_spot | 3.168763 | 3.168763 | 86.19% |
| short_one_futures_only | 154.482004 | 154.482004 | -9129.76% |
| short_two_futures_only | 151.832597 | 151.832597 | -8782.14% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
