# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | 20.110798 | 20.110798 | 0.00% |
| short_one_futures_spot | -3.321802 | -3.321802 | 45.43% |
| short_two_futures_spot | -3.030304 | -3.030304 | 48.10% |
| short_one_futures_only | 68.952312 | 68.952312 | -10088.77% |
| short_two_futures_only | 67.971407 | 67.971407 | -9781.98% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
