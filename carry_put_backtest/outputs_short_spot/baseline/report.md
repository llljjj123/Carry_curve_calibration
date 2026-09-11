# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 24. Scheduled cohorts: 24.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -3.336771 | -80.082507 | 0.00% |
| short_one_futures_spot | -11.525736 | -276.617669 | 71.76% |
| short_two_futures_spot | -11.353551 | -272.485234 | 72.07% |
| short_one_futures_only | -53.573884 | -1285.773227 | -1485.75% |
| short_two_futures_only | -52.635799 | -1263.259184 | -1436.41% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
