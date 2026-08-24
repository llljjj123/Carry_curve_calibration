import numpy as np

from im_corr_ou_1factor.filtering import kalman_filter, smooth_states
from im_corr_ou_1factor.model import OUParams
from im_corr_ou_1factor.simulation import simulate_joint_panel


def gap(start, end):
    return np.busday_count(np.datetime64(start.date()), np.datetime64(end.date())) / 244


def test_filters_handle_ragged_curves_and_unequal_gaps() -> None:
    params = OUParams(3.0, 0.06, 0.18, -0.35, 0.003, 0.12)
    panel, truth = simulate_joint_panel(params, n_dates=180, seed=5, missing_probability=0.25)
    panel = panel.loc[~panel["date"].isin(panel["date"].drop_duplicates().iloc[30:33])]
    for mode in ("curve", "joint"):
        result = kalman_filter(panel, params, sigma=0.25, gap_function=gap, mode=mode)
        states = smooth_states(result.states)
        merged = states.merge(truth, on="date")
        assert np.isfinite(result.log_likelihood)
        assert (states["filtered_variance"] > 0).all()
        assert not states["smoothed_state"].isna().any()
        assert np.sqrt(np.mean((merged["filtered_state"] - merged["true_state"]) ** 2)) < 0.035


def test_filtered_and_smoothed_states_are_separate() -> None:
    params = OUParams(2.5, 0.05, 0.15, 0.3, 0.004, 0.10)
    panel, _ = simulate_joint_panel(params, n_dates=100, seed=9)
    result = kalman_filter(panel, params, sigma=0.25, gap_function=gap, mode="joint")
    original_filtered = result.states["filtered_state"].copy()
    smoothed = smooth_states(result.states)
    assert np.array_equal(result.states["filtered_state"], original_filtered)
    assert not np.allclose(smoothed["smoothed_state"].iloc[:-1], smoothed["filtered_state"].iloc[:-1])
    assert smoothed.iloc[-1]["smoothed_state"] == smoothed.iloc[-1]["filtered_state"]

