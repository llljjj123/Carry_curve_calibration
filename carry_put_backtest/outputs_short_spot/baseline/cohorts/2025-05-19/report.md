# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | 31.194059 | 31.194059 | 0.00% |
| short_one_futures_spot | -18.027291 | -18.027291 | 76.17% |
| short_two_futures_spot | -16.492043 | -16.492043 | 79.37% |
| short_one_futures_only | -4.176543 | -4.176543 | -5348.43% |
| short_two_futures_only | -2.504440 | -2.504440 | -5185.23% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
