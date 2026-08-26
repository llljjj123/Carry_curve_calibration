# CSI 1000 / IM carry-put demonstration

This folder contains the requested readable demonstration with configurable
valuation date and calibration sample size. It recalibrates the independent
two-factor OU carry model on the requested number of usable trading dates,
estimates CSI 1000 historical volatility over the same dates, derives the locked
carry from the selected observed IM close, and prices only the American carry-put
**optional component**.

The futures contract is configurable. Its CFFEX expiry is inferred from the
contract code and validated against the selected quote.

The linear payoff `F(t,T) - F(0,T)` is deliberately excluded.

## Files

- `Carry_Put_Demo.ipynb`: executed narrative demonstration.
- `calibration.py`: sample selection, volatility, calibration, filtering, and exports.
- `option_pricing.py`: option-only pricing and convergence checks.
- `profile_analysis.py`: fixed-fast-volatility likelihood and option-price profile.
- `demo_workflow.py`: orchestration, charts, and summary output.
- `outputs/`: generated CSV, JSON, and PNG results.
- `tests/`: fast checks of the fixed sample and locked-carry construction.

The demo reuses the validated numerical engines in sibling folders
`im_2factor_ou_carry/src` and `carry_put_pricing/src`. The calibration estimator
now accepts a backward-compatible configurable `eta_fast_upper_bound`; its
default remains `3.0`, while this demo uses `6.0`.

## Run

From this directory:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B demo_workflow.py
```

For example, to use exactly 250 trading dates ending on 2026-08-10:

```powershell
& 'D:\miniforge3\envs\spyder-env\python.exe' -B demo_workflow.py `
  --valuation-date 2026-08-10 `
  --sample-size 250 `
  --futures-contract IM2612
```

The sample size includes the valuation date. Thus `--sample-size 250` selects
250 usable curve dates and produces 249 close-to-close spot returns. The
valuation date must itself have an accepted carry curve and a quote for the
selected contract.

The notebook exposes the same inputs in its first code cell:

```python
VALUATION_DATE = "2026-08-21"
SAMPLE_SIZE = 244
FUTURES_CONTRACT = "IM2612"
```

## Calendar coverage

The installed `chinese_calendar` package supports dates only through 2026.
The shared project calendar therefore falls back to ordinary weekdays in 2027.
The Demo does **not** use that fallback: it contains an explicit company
exchange calendar for 2027--2028 and fails clearly for any uncovered future
year. Under the Demo calendar, `IM2703` expires on 2027-03-19 and has 138
sessions remaining from 2026-08-21, versus 144 under the old weekday fallback.
The company calendar is a planning input rather than a final official CFFEX
schedule and should be replaced or confirmed when the official 2027 calendar
is published.

The base calibration uses 12 optimizer starts. The fixed-eta profile re-estimates
the other five parameters from four starts at each grid point and reprices the
option, so the complete run can take several minutes. No new packages are required.

## Important interpretation

Historical OU parameters are provisionally used as risk-neutral parameters.
The calculation permits exercise on each trading session and is therefore a
daily Bermudan approximation to a continuous-time American option. With zero
spot/carry shock correlation, spot homogeneity removes spot volatility from
this particular option value; the requested historical volatility is still
estimated, reported, and passed through the pricing interface.

The cap-6 calibration produces an interior fast-volatility estimate, but the
fast-minus-slow mean-reversion gap reaches its existing upper bound of `60`.
This second boundary is retained and reported as a model-risk diagnostic.
