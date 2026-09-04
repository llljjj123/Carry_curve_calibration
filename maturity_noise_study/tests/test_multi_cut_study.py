"""Tests for multiple-cut-date window construction and paired inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from multi_cut_study import make_nonoverlapping_cut_windows, newey_west_mean_test


def test_cut_windows_are_equal_and_nonoverlapping() -> None:
    dates = pd.bdate_range("2020-01-01", periods=100)
    windows = make_nonoverlapping_cut_windows(
        dates,
        windows=3,
        test_dates_per_window=20,
        minimum_training_dates=40,
    )
    assert [window.training_dates for window in windows] == [40, 60, 80]
    assert all(
        len(dates[(dates >= window.test_start) & (dates <= window.test_end)]) == 20
        for window in windows
    )
    assert windows[0].test_end < windows[1].test_start
    assert windows[1].test_end < windows[2].test_start
    assert windows[-1].test_end == dates[-1]


def test_cut_windows_reject_insufficient_training_history() -> None:
    dates = pd.bdate_range("2020-01-01", periods=99)
    try:
        make_nonoverlapping_cut_windows(
            dates,
            windows=3,
            test_dates_per_window=20,
            minimum_training_dates=40,
        )
    except ValueError as error:
        assert "Not enough dates" in str(error)
    else:
        raise AssertionError("Expected insufficient history to raise ValueError")


def test_newey_west_mean_test_detects_positive_gain() -> None:
    values = np.linspace(0.5, 1.5, 200)
    result = newey_west_mean_test(values)
    assert result["n_dates"] == 200
    assert result["mean_gain"] > 0.0
    assert result["z_statistic"] > 0.0
    assert result["two_sided_p_value"] < 0.01
