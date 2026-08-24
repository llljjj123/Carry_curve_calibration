import numpy as np

from im_corr_ou_1factor.estimation import estimate_model
from im_corr_ou_1factor.model import OUParams
from im_corr_ou_1factor.simulation import simulate_joint_panel


def gap(start, end):
    return np.busday_count(np.datetime64(start.date()), np.datetime64(end.date())) / 244


def test_joint_estimator_approximately_recovers_parameters_with_fixed_sigma() -> None:
    known = OUParams(3.0, 0.055, 0.16, -0.45, 0.003, 0.11)
    panel, _ = simulate_joint_panel(known, sigma=0.25, n_dates=650, seed=20260824, missing_probability=0.04)
    result = estimate_model(
        panel,
        name="synthetic_joint",
        mode="joint",
        variant="exact",
        sigma=0.25,
        gap_function=gap,
        free_rho=True,
        starts=3,
        maxiter=650,
        seed=12,
        compute_standard_errors=False,
    )
    fitted = result.params
    assert result.converged
    assert 0.35 * known.kappa < fitted.kappa < 2.8 * known.kappa
    assert abs(fitted.theta - known.theta) < 0.025
    assert abs(fitted.eta / known.eta - 1) < 0.40
    assert abs(fitted.rho - known.rho) < 0.25
    assert abs(fitted.sigma_epsilon / known.sigma_epsilon - 1) < 0.30
    assert abs(fitted.mu - known.mu) < 0.12
    assert result.sigma == 0.25
