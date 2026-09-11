"""Equivalence of the optional acceleration with the dense reference filter."""

import numpy as np
import pandas as pd
import pytest

from carry_put_backtest.engine import _gap
from im_2factor_ou_carry.fast_likelihood import make_fast_log_likelihood
from im_2factor_ou_carry.two_factor import TwoFactorParams, make_dataset, two_factor_log_likelihood


@pytest.mark.parametrize("model", ["constant_carry", "constant_log_futures"])
def test_compiled_likelihood_matches_dense_ragged_curves(model):
    rng = np.random.default_rng(852)
    rows = []
    for j, date in enumerate(pd.bdate_range("2025-03-03", periods=15)):
        for tau in np.linspace(.02, .7, 2+j%3):
            carry = .08 + rng.normal(0, .02)
            rows.append(dict(date=date, tau=tau, implied_carry=carry, spot=6000.,
                             futures_price=6000*np.exp((.014-carry)*tau), risk_free_rate=.014))
    dataset = make_dataset(pd.DataFrame(rows), _gap, model)
    fast = make_fast_log_likelihood(dataset)
    for _ in range(20):
        slow = rng.uniform(.02, 2.)
        params = TwoFactorParams(slow, slow+rng.uniform(2, 90), rng.uniform(-.1, .2),
                                 rng.uniform(.02, .2), rng.uniform(.1, 2.),
                                 .005 if model == "constant_log_futures" else .02)
        reference = two_factor_log_likelihood(dataset, params)
        assert fast(params) == pytest.approx(reference, rel=1e-10, abs=1e-8)
