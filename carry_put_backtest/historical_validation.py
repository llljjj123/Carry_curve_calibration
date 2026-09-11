"""Explicit historical validation with keyed calibration caches and cohort checkpoints."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy

from .engine import BacktestConfig, BacktestResult, STUDIES, run_backtest, scenario_specs
from .engine import _gap, calibration_window
from .market import CohortUnavailable, load_cached_market, monthly_cohorts, session_dates
from .reporting import _json_value, export_result
from carry_put_pricing import NumericalConfig
from im_2factor_ou_carry.two_factor import TwoFactorParams
from im_2factor_ou_carry.two_factor_estimation import estimate_two_factor_ou


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "im_2factor_ou_carry/data/raw"
DEFAULT_OUTPUT = ROOT / "carry_put_backtest/outputs_historical"
SHORT_OUTPUT = ROOT / "carry_put_backtest/outputs_short_spot"


def write_json(path, value):
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False), encoding="utf-8")


def digest(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def cached_estimator(directory, backend="numba"):
    directory.mkdir(parents=True, exist_ok=True)
    sources = digest(list((ROOT / "im_2factor_ou_carry/src/im_2factor_ou_carry").glob("*.py"))
                     + [ROOT / "Demo/calendar_utils.py", ROOT / "Demo/data/china_exchange_calendar_2027_2028.csv"])

    def estimate(sample, **kwargs):
        kwargs["likelihood_backend"] = backend
        date = sample.date.max().strftime("%Y-%m-%d")
        columns = ["date", "contract", "tau", "implied_carry", "spot", "futures_price", "risk_free_rate"]
        specification = {
            "sample_sha256": hashlib.sha256(sample[columns].to_csv(index=False, float_format="%.17g").encode()).hexdigest(),
            "settings": {k: v for k, v in kwargs.items() if k != "gap_function"},
            "sources": sources, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
        }
        key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:20]
        path = directory / f"{date}_{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["specification"] != specification:
                raise ValueError("Calibration cache specification mismatch")
            print(f"{date}: reusing matching frozen calibration", flush=True)
        else:
            print(f"{date}: calibrating {sample.date.nunique()} curves, {kwargs['starts']} starts", flush=True)
            start = time.perf_counter()
            result = estimate_two_factor_ou(sample, **kwargs)
            record = {"specification": specification, "params": asdict(result.params),
                      "converged": result.converged, "message": result.message,
                      "log_likelihood": result.log_likelihood,
                      "optimizer_runs": result.optimizer_runs.to_dict("records"),
                      "runtime_seconds": time.perf_counter() - start}
            write_json(path, record)
            print(f"{date}: calibration finished in {record['runtime_seconds']:.1f}s; converged={result.converged}", flush=True)
        params = TwoFactorParams(**record["params"])
        params.validate()
        return SimpleNamespace(params=params, converged=record["converged"], message=record["message"],
                               log_likelihood=record["log_likelihood"],
                               optimizer_runs=pd.DataFrame(record["optimizer_runs"]))
    return estimate


def coverage(market, output):
    accepted, _ = market.observations_through(market.spot.date.max())
    curve_dates = accepted.date.drop_duplicates().sort_values()
    schedule = monthly_cohorts(market.spot.date.min(), market.spot.date.max())
    rows = []
    for cohort in schedule.to_dict("records"):
        entry, expiry = cohort["entry_date"], cohort["expiry"]
        count = int((curve_dates <= entry).sum())
        row = {**cohort, "available_curve_dates": count}
        try:
            if count < 488:
                raise CohortUnavailable("insufficient_488_date_history")
            if entry not in set(curve_dates):
                raise CohortUnavailable("no_accepted_inception_curve")
            if expiry > min(market.spot.date.max(), market.futures.date.max()):
                raise CohortUnavailable("incomplete_scheduled_life")
            pair = market.hedge_pair(entry, cohort["option_contract"])
            row.update(hedge_contract_1=pair[0], hedge_contract_2=pair[1])
            for date in session_dates(entry, expiry):
                market.quote(date)
                for code in pair:
                    market.quote(date, code)
        except CohortUnavailable as exc:
            row.update(eligible=False, reason=str(exc))
        else:
            row.update(eligible=True, reason="")
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output / "coverage.csv", index=False)
    print(f"Coverage: {len(result)} scheduled entries, {int(result.eligible.sum())} eligible", flush=True)
    return result


def validate_backend(output):
    from im_2factor_ou_carry.fast_likelihood import make_fast_log_likelihood
    from im_2factor_ou_carry.two_factor import make_dataset, two_factor_log_likelihood
    from im_2factor_ou_carry.two_factor_estimation import pack, unpack
    market = load_cached_market()
    rows = []
    rng = np.random.default_rng(852)
    for date in ("2024-08-19", "2026-07-20"):
        accepted, _ = market.observations_through(date)
        sample = calibration_window(accepted, date, 488)
        dataset = make_dataset(sample, _gap, "constant_log_futures")
        fast = make_fast_log_likelihood(dataset)
        for i in range(20):
            slow = rng.uniform(.02, 2.)
            params = TwoFactorParams(slow, slow+rng.uniform(2, 90), rng.uniform(-.1, .2),
                                     rng.uniform(.02, .2), rng.uniform(.1, 2.), rng.uniform(.0005, .005))
            reference, compiled = two_factor_log_likelihood(dataset, params), fast(params)
            error = abs(reference-compiled)
            assert np.isclose(reference, compiled, rtol=1e-10, atol=1e-7), (date, params, error)
            rows.append(dict(date=date, parameter_set=i, reference=reference, compiled=compiled, absolute_error=error))
        point = pack(TwoFactorParams(.34, 16.7, .08, .05, 1.2, .001))
        grad_errors = []
        for j in range(6):
            bump = np.zeros(6)
            bump[j] = 1e-5
            high, low = unpack(point+bump), unpack(point-bump)
            ref = (two_factor_log_likelihood(dataset, high)-two_factor_log_likelihood(dataset, low))/2e-5
            acc = (fast(high)-fast(low))/2e-5
            grad_errors.append(abs(ref-acc))
        assert max(grad_errors) < 1e-4, grad_errors
        start = time.perf_counter()
        for _ in range(5):
            two_factor_log_likelihood(dataset, params)
        dense_seconds = (time.perf_counter()-start)/5
        start = time.perf_counter()
        for _ in range(100):
            fast(params)
        fast_seconds = (time.perf_counter()-start)/100
        print(f"{date}: dense {dense_seconds:.6f}s, compiled {fast_seconds:.6f}s, speedup {dense_seconds/fast_seconds:.0f}x; gradient max error {max(grad_errors):.2g}", flush=True)
    pd.DataFrame(rows).to_csv(output / "likelihood_backend_equivalence.csv", index=False)


def load_result(path, config):
    def read(name, date_columns=()):
        frame = pd.read_csv(path / f"{name}.csv")
        for column in date_columns:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column])
        return frame
    return BacktestResult(
        read("daily_ledger", ["date"]), read("cohort_pnl", ["entry_date", "exit_date"]),
        read("cohort_audit", ["entry_date", "expiry", "previous_expiry", "sample_start", "sample_end", "exit_date"]),
        read("optimizer_runs"), config,
    )


def run_batch(output, config, label, *, max_cohorts=None, dates=None):
    output.mkdir(parents=True, exist_ok=True)
    market = load_cached_market()
    audit = coverage(market, output)
    eligible = audit.loc[audit.eligible].copy()
    if dates is not None:
        eligible = eligible.loc[eligible.cohort.isin(dates)]
    if max_cohorts is not None:
        eligible = eligible.head(max_cohorts)
    estimator = cached_estimator(output / "calibrations")
    paths = [RAW / "spot_raw.csv", RAW / "futures_raw.csv"]
    code_paths = list((ROOT / "carry_put_pricing/src/carry_put_pricing").glob("*.py"))
    code_paths += [ROOT / "carry_put_backtest" / f for f in (
        "engine.py", "market.py", "reporting.py", "analyze_historical.py",
        "analyze_short_spot.py",
    )]
    code_paths += list((ROOT / "im_2factor_ou_carry/src/im_2factor_ou_carry").glob("*.py"))
    code_paths += [ROOT / "Demo" / f for f in ("calendar_utils.py", "demo_quality.py",
                                               "data/china_exchange_calendar_2027_2028.csv")]
    signature = {"config": asdict(config),
                 "scenario_definitions": [asdict(spec) for spec in scenario_specs(config.study)],
                 "inputs_and_code": digest(paths + code_paths),
                 "backend": "numba", "python": platform.python_version(),
                 "numpy": np.__version__, "scipy": scipy.__version__}
    results = []
    for index, cohort in enumerate(eligible.to_dict("records"), 1):
        date = cohort["cohort"]
        path = output / label / "cohorts" / date
        stamp = path / "checkpoint.json"
        print(f"[{label}] {index}/{len(eligible)} {date}", flush=True)
        started = time.perf_counter()
        if stamp.exists() and json.loads(stamp.read_text())["signature"] == signature:
            result = load_result(path, config)
            print("  matching completed checkpoint loaded", flush=True)
        else:
            result = run_backtest(market, start=date, end=date, config=config, estimator=estimator)
            export_result(result, path, input_paths=paths, request={"entry": date, "validation_label": label})
            write_json(stamp, {"signature": signature, "runtime_seconds": time.perf_counter() - started})
            print(f"  cohort finished in {time.perf_counter()-started:.1f}s; {result.cohorts.status.iloc[0]}", flush=True)
        results.append(result)
        # Persist the aggregate after every cohort; it is explicitly a partial
        # batch until every eligible cohort appears in the audit.
        combined = BacktestResult(*[
            pd.concat([getattr(r, name) for r in results], ignore_index=True)
            for name in ("daily", "summaries", "cohorts", "optimizer_runs")
        ], config)
        export_result(combined, output / label, input_paths=paths,
                      request={"validation_label": label, "completed_batch_entries": len(results),
                               "requested_batch_entries": len(eligible)})
    return combined if results else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["coverage", "backendcheck", "baseline", "fine", "lag1", "costs"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--study", choices=STUDIES, default="long_futures_only")
    parser.add_argument("--max-cohorts", type=int)
    parser.add_argument("--dates", nargs="+")
    args = parser.parse_args()
    output_dir = args.output_dir or (SHORT_OUTPUT if args.study == "short_with_spot" else DEFAULT_OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "backendcheck":
        validate_backend(output_dir)
        return
    if args.phase == "coverage":
        coverage(load_cached_market(), output_dir)
        return
    config = BacktestConfig(study=args.study)
    if args.phase == "fine":
        config = replace(config, numerical=NumericalConfig(slow_grid_points=451, fast_grid_points=601, quadrature_order=61))
    elif args.phase == "lag1":
        config = replace(config, execution_lag_sessions=1)
    elif args.phase == "costs":
        # Explicit illustrative point cost, not an assumed actual brokerage fee.
        config = replace(config, slippage_points=0.2,
                         spot_cost_bps=1.0 if args.study == "short_with_spot" else 0.0)
    run_batch(output_dir, config, args.phase, max_cohorts=args.max_cohorts, dates=args.dates)


if __name__ == "__main__":
    main()
