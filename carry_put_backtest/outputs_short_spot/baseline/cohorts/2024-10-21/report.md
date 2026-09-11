# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | 12.519668 | 12.519668 | 0.00% |
| short_one_futures_spot | 2.444121 | 2.444121 | 81.10% |
| short_two_futures_spot | 2.847473 | 2.847473 | 81.65% |
| short_one_futures_only | -92.575678 | -92.575678 | -7319.92% |
| short_two_futures_only | -90.249771 | -90.249771 | -7110.52% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
