import numpy as np

from im_2factor_ou_carry.kalman import maturity_loading
from im_2factor_ou_carry.two_factor import (
    TwoFactorParams,
    transition,
    two_factor_kalman_filter,
)
from im_2factor_ou_carry.two_factor_simulation import simulate_two_factor_panel


def _gap(start, end):
    return np.busday_count(np.datetime64(start.date()), np.datetime64(end.date())) / 244


def test_exact_transition_and_ordered_parameters() -> None:
    params = TwoFactorParams(0.8, 12.0, 0.05, 0.08, 0.40, 0.003)
    a, q = transition(params, 0.0)
    assert np.allclose(a, np.eye(2))
    assert np.allclose(q, np.zeros((2, 2)))


def test_opposite_factor_states_can_create_a_hump() -> None:
    params = TwoFactorParams(0.8, 15.0, 0.05, 0.08, 0.40, 0.003)
    tau = np.linspace(1 / 244, 2.0, 300)
    carry = (
        params.theta
        + 0.05 * maturity_loading(params.kappa_slow, tau)
        - 0.08 * maturity_loading(params.kappa_fast, tau)
    )
    differences = np.diff(carry)
    assert np.any(differences > 0)
    assert np.any(differences < 0)


def test_two_factor_filter_recovers_latent_combined_state() -> None:
    params = TwoFactorParams(0.7, 14.0, 0.045, 0.06, 0.35, 0.0025)
    panel, truth = simulate_two_factor_panel(params, n_dates=350, seed=44, missing_probability=0.08)
    result = two_factor_kalman_filter(panel, params, gap_function=_gap)
    merged = result.states.merge(truth, on="date")
    fitted_combined = params.theta + merged["filtered_slow_state"] + merged["filtered_fast_state"]
    true_combined = params.theta + merged["true_slow_state"] + merged["true_fast_state"]
    assert np.isfinite(result.log_likelihood)
    assert np.sqrt(np.mean((fitted_combined - true_combined) ** 2)) < 0.02
    assert len(result.innovations) == len(panel)
    assert np.isfinite(result.innovations["standardized_innovation"]).all()

