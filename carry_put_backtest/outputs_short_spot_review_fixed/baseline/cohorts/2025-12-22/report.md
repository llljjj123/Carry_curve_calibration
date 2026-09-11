# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | 19.057746 | 19.057746 | 0.00% |
| short_one_futures_spot | -10.008924 | -10.008924 | 22.52% |
| short_two_futures_spot | -10.073118 | -10.073118 | 24.45% |
| short_one_futures_only | -194.462794 | -194.462794 | -2317.05% |
| short_two_futures_only | -191.205742 | -191.205742 | -2247.01% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
