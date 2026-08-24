import numpy as np

from im_2factor_ou_carry.kalman import OUParams, kalman_filter, maturity_loading, transition_moments
from im_2factor_ou_carry.simulation import simulate_panel


def test_stable_loading_and_transition() -> None:
    assert maturity_loading(2.0, 0.0) == 1.0
    assert np.all(np.diff(maturity_loading(2.0, np.array([0.01, 0.1, 1.0]))) < 0)
    params = OUParams(2.0, 0.04, 0.08, 0.003)
    a, q = transition_moments(params, 0.0)
    assert a == 1.0
    assert q == 0.0


def test_filter_handles_ragged_multiple_contract_curves() -> None:
    params = OUParams(2.5, 0.04, 0.07, 0.0025)
    panel, truth = simulate_panel(params, n_dates=120, seed=4, missing_probability=0.25)
    def gap(start, end):
        return np.busday_count(np.datetime64(start.date()), np.datetime64(end.date())) / 244

    result = kalman_filter(panel, params, gap_function=gap)
    merged = result.states.merge(truth, on="date")
    assert np.isfinite(result.log_likelihood)
    assert len(result.states) == panel["date"].nunique()
    assert np.sqrt(np.mean((merged["filtered_state"] - merged["true_state"]) ** 2)) < 0.015
    assert (result.states["filtered_variance"] > 0).all()
