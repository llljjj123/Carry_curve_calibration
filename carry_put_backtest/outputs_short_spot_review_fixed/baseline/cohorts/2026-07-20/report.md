# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | 7.194631 | 7.194631 | 0.00% |
| short_one_futures_spot | -0.245534 | -0.245534 | 92.52% |
| short_two_futures_spot | -0.787370 | -0.787370 | 93.66% |
| short_one_futures_only | -232.184361 | -232.184361 | -5268.76% |
| short_two_futures_only | -229.087641 | -229.087641 | -5082.97% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
