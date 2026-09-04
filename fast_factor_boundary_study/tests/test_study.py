"""Fast unit tests for the boundary-study orchestration."""

from __future__ import annotations

import numpy as np

from configurable_estimation import pack, parameter_bounds, unpack
from im_2factor_ou_carry.two_factor import TwoFactorParams
from study import StudySettings, build_scenario_specs, prepare_sample


def test_pack_round_trip() -> None:
    params = TwoFactorParams(2.0, 102.0, 0.10, 0.20, 4.0, 0.01)
    recovered = unpack(pack(params))
    assert np.allclose(
        list(vars(recovered).values()),
        list(vars(params).values()),
    )


def test_configurable_gap_bound() -> None:
    bounds = parameter_bounds(6.0, 180.0)
    assert np.isclose(np.exp(bounds[1][1]), 180.0)


def test_default_design_has_ten_unique_scenarios() -> None:
    specs = build_scenario_specs(StudySettings())
    assert len(specs) == 10
    assert len({spec.scenario_id for spec in specs}) == 10


def test_strict_calendar_sample_and_short_end_filter() -> None:
    sample, _, _ = prepare_sample("2026-08-21", 244, 20)
    assert sample["date"].nunique() == 244
    assert sample["sessions_to_expiry"].min() > 20
    assert ((sample["date"] == sample["date"].max()) & (sample["contract"] == "IM2612")).sum() == 1
