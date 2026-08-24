from im_2factor_ou_carry.two_factor import TwoFactorParams
from im_2factor_ou_carry.two_factor_estimation import estimate_two_factor_ou
from im_2factor_ou_carry.two_factor_simulation import simulate_two_factor_panel


def _gap(start, end):
    import numpy as np

    return np.busday_count(np.datetime64(start.date()), np.datetime64(end.date())) / 244


def test_estimator_approximately_recovers_two_known_ou_factors() -> None:
    known = TwoFactorParams(
        kappa_slow=0.75,
        kappa_fast=14.0,
        theta=0.045,
        eta_slow=0.065,
        eta_fast=0.40,
        sigma_epsilon=0.0025,
    )
    panel, _ = simulate_two_factor_panel(
        known,
        n_dates=650,
        seed=20260824,
        missing_probability=0.03,
    )
    result = estimate_two_factor_ou(
        panel,
        gap_function=_gap,
        starts=5,
        maxiter=900,
        seed=17,
        compute_standard_errors=False,
    )
    fitted = result.params
    assert result.converged
    assert 0.25 * known.kappa_slow < fitted.kappa_slow < 4.0 * known.kappa_slow
    assert 0.45 * known.kappa_fast < fitted.kappa_fast < 2.2 * known.kappa_fast
    assert abs(fitted.theta - known.theta) < 0.025
    assert abs(fitted.eta_slow / known.eta_slow - 1.0) < 0.65
    assert abs(fitted.eta_fast / known.eta_fast - 1.0) < 0.45
    assert abs(fitted.sigma_epsilon / known.sigma_epsilon - 1.0) < 0.40
