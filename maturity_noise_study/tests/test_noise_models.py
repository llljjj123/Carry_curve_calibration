"""Focused tests for observation transformations, likelihoods, and splitting."""

from __future__ import annotations

import numpy as np

from calibration import _gap_function, load_calibration_sample
from im_2factor_ou_carry.two_factor import (
    TwoFactorParams,
    make_dataset as make_original_dataset,
    two_factor_log_likelihood,
)

from noise_models import (
    ModelParams,
    OUParams,
    carry_noise_sd,
    filter_panel,
    log_likelihood,
    make_dataset,
)
from study import common_split_date


def small_panel():
    sample, _, _ = load_calibration_sample("2026-08-21", 40)
    return sample


def test_constant_carry_matches_validated_baseline_likelihood() -> None:
    panel = small_panel()
    original = TwoFactorParams(1.5, 40.0, 0.10, 0.08, 2.0, 0.01)
    alternative = ModelParams(
        "constant_carry",
        OUParams(1.5, 40.0, 0.10, 0.08, 2.0),
        (0.01,),
    )
    expected = two_factor_log_likelihood(make_original_dataset(panel, _gap_function), original)
    actual = log_likelihood(make_dataset(panel, _gap_function), alternative)
    assert np.isclose(actual, expected, rtol=0.0, atol=1e-9)


def test_log_futures_matches_zero_floor_smooth_model_after_jacobian() -> None:
    panel = small_panel()
    ou = OUParams(1.5, 40.0, 0.10, 0.08, 2.0)
    sigma_log = 0.001
    direct = ModelParams("constant_log_futures", ou, (sigma_log,))
    smooth_near_zero = ModelParams("smooth_carry_log", ou, (1e-12, sigma_log))
    direct_ll = log_likelihood(make_dataset(panel, _gap_function), direct)
    smooth_ll = log_likelihood(make_dataset(panel, _gap_function), smooth_near_zero)
    assert np.isclose(direct_ll, smooth_ll, rtol=0.0, atol=1e-8)


def test_noise_functions_have_expected_maturity_behavior() -> None:
    tau = np.array([10.0, 80.0]) / 244.0
    sessions = np.array([10.0, 80.0])
    constant = carry_noise_sd("constant_carry", (0.01,), tau, sessions)
    bucket = carry_noise_sd("two_bucket_carry", (0.02, 0.005), tau, sessions)
    direct = carry_noise_sd("constant_log_futures", (0.001,), tau, sessions)
    assert np.allclose(constant, [0.01, 0.01])
    assert np.allclose(bucket, [0.02, 0.005])
    assert np.isclose(direct[0] / direct[1], 8.0)


def test_filter_outputs_one_prediction_per_observation() -> None:
    panel = small_panel()
    params = ModelParams(
        "two_bucket_carry",
        OUParams(1.5, 40.0, 0.10, 0.08, 2.0),
        (0.02, 0.01),
    )
    result = filter_panel(panel, params, gap_function=_gap_function)
    assert len(result.predictions) == len(panel)
    assert len(result.states) == panel["date"].nunique()
    assert np.isfinite(result.comparison_log_likelihood)


def test_common_split_has_no_overlap_and_expected_test_count() -> None:
    panel = small_panel()
    split = common_split_date(panel, 0.20)
    train_dates = set(panel.loc[panel["date"] <= split, "date"])
    test_dates = set(panel.loc[panel["date"] > split, "date"])
    assert train_dates.isdisjoint(test_dates)
    assert len(test_dates) == 8
