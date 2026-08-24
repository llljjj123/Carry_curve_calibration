import numpy as np
import pytest
from scipy.integrate import quad

from im_corr_ou_1factor.model import (
    OUParams,
    integral_b,
    integral_c,
    integral_d,
    integral_j,
    joint_interval_moments,
    log_futures_basis,
)


@pytest.mark.parametrize("kappa,tau", [(0.7, 0.01), (3.0, 0.4), (20.0, 1.2)])
def test_integrals_match_numerical_quadrature(kappa: float, tau: float) -> None:
    b = lambda u: (1 - np.exp(-kappa * u)) / kappa
    assert float(integral_b(kappa, tau)) == pytest.approx(b(tau), rel=1e-12)
    assert float(integral_c(kappa, tau)) == pytest.approx(quad(lambda u: b(u) ** 2, 0, tau)[0], rel=2e-10)
    assert float(integral_d(kappa, tau)) == pytest.approx(quad(b, 0, tau)[0], rel=2e-10)
    expected_j = quad(lambda u: np.exp(-kappa * u) * b(u), 0, tau)[0]
    assert float(integral_j(kappa, tau)) == pytest.approx(expected_j, rel=2e-10)


def test_small_kappa_tau_limits_are_stable() -> None:
    kappa, tau = 1e-8, 2e-5
    assert np.isfinite(integral_b(kappa, tau))
    assert float(integral_b(kappa, tau)) == pytest.approx(tau, rel=1e-12)
    assert float(integral_c(kappa, tau)) == pytest.approx(tau**3 / 3, rel=1e-12)
    assert float(integral_d(kappa, tau)) == pytest.approx(tau**2 / 2, rel=1e-12)
    assert float(integral_j(kappa, tau)) == pytest.approx(tau**2 / 2, rel=1e-12)


def test_futures_formula_reductions() -> None:
    base = OUParams(2.0, 0.06, 0.20, 0.0, 0.003, 0.10)
    tau, state, rate, sigma = 0.4, 0.09, 0.014, 0.25
    rho_zero = float(log_futures_basis(state, tau, base, sigma, rate))
    expected = (rate - base.theta) * tau - (state - base.theta) * float(integral_b(base.kappa, tau)) + 0.5 * base.eta**2 * float(integral_c(base.kappa, tau))
    assert rho_zero == pytest.approx(expected)
    tiny_eta = OUParams(base.kappa, base.theta, 1e-12, 0.7, base.sigma_epsilon, base.mu)
    no_eta = float(log_futures_basis(state, tau, tiny_eta, sigma, rate))
    expected_no_eta = (rate - base.theta) * tau - (state - base.theta) * float(integral_b(base.kappa, tau))
    assert no_eta == pytest.approx(expected_no_eta, abs=1e-12)


def test_exact_monte_carlo_futures_formula() -> None:
    params = OUParams(1.8, 0.07, 0.18, -0.45, 0.003, 0.10)
    tau, state, rate, sigma = 0.7, 0.11, 0.014, 0.25
    variance = sigma**2 * tau + params.eta**2 * float(integral_c(params.kappa, tau)) - 2 * params.rho * sigma * params.eta * float(integral_d(params.kappa, tau))
    deterministic = (rate - params.theta) * tau - (state - params.theta) * float(integral_b(params.kappa, tau)) - 0.5 * sigma**2 * tau
    rng = np.random.default_rng(2026)
    simulated_ratio = np.exp(deterministic + rng.normal(0, np.sqrt(variance), 300_000)).mean()
    analytical_ratio = np.exp(float(log_futures_basis(state, tau, params, sigma, rate)))
    assert simulated_ratio == pytest.approx(analytical_ratio, rel=2.5e-3)


@pytest.mark.parametrize("rho", [-0.95, -0.5, 0.0, 0.5, 0.95])
def test_joint_covariance_is_positive_semidefinite(rho: float) -> None:
    params = OUParams(3.0, 0.05, 0.25, rho, 0.003, 0.10)
    _, q, _, vr, g = joint_interval_moments(params, 0.25, 7 / 244)
    assert np.linalg.eigvalsh([[q, g], [g, vr]]).min() >= -1e-12

