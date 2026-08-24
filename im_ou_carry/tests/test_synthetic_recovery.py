import numpy as np

from im_ou_carry.estimation import estimate_ou
from im_ou_carry.kalman import OUParams
from im_ou_carry.simulation import simulate_panel


def test_estimator_approximately_recovers_known_ou_parameters() -> None:
    known = OUParams(kappa=3.0, theta=0.045, eta=0.075, sigma_epsilon=0.0025)
    panel, _ = simulate_panel(known, n_dates=700, seed=20260824, missing_probability=0.05)
    def gap(start, end):
        return np.busday_count(np.datetime64(start.date()), np.datetime64(end.date())) / 244

    result = estimate_ou(panel, gap_function=gap, starts=5, maxiter=800, seed=12, compute_standard_errors=False)
    fitted = result.params
    assert result.converged
    assert 0.40 * known.kappa < fitted.kappa < 2.5 * known.kappa
    assert abs(fitted.theta - known.theta) < 0.02
    assert abs(fitted.eta / known.eta - 1.0) < 0.45
    assert abs(fitted.sigma_epsilon / known.sigma_epsilon - 1.0) < 0.35
