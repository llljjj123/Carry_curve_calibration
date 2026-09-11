# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -31.452460 | -31.452460 | 0.00% |
| short_one_futures_spot | -14.623741 | -14.623741 | 86.46% |
| short_two_futures_spot | -14.151082 | -14.151082 | 85.80% |
| short_one_futures_only | -21.616912 | -21.616912 | -2539.26% |
| short_two_futures_only | -20.683545 | -20.683545 | -2452.90% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
