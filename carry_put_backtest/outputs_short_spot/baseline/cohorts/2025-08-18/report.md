# Short carry-put with funded spot hedge — monthly cohort back-test

Completed cohorts: 1. Scheduled cohorts: 1.

The three primary scenarios are shown with two futures-only controls. All five share the same exercise policy and exit date.

| Scenario | Mean cohort P&L | Sum cohort P&L | Daily variance reduction |
|---|---:|---:|---:|
| short_no_hedge | -48.774371 | -48.774371 | 0.00% |
| short_one_futures_spot | -22.527268 | -22.527268 | 92.93% |
| short_two_futures_spot | -22.045346 | -22.045346 | 92.99% |
| short_one_futures_only | -113.870054 | -113.870054 | -337.66% |
| short_two_futures_only | -111.639126 | -111.639126 | -327.39% |

P&L is in configured monetary units (points when both multipliers are 1).
Read cohort_audit.csv for unavailable cohorts and run_manifest.json for assumptions.
Daily variance uses model option marks. Tail statistics can be unstable in a small sample.
