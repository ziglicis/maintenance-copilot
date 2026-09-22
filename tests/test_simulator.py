"""The cost model, checked against a fleet small enough to compute by hand."""

from __future__ import annotations

import pandas as pd

from copilot import simulator


def _two_engine_fleet() -> pd.DataFrame:
    """Engine 1 fails at cycle 100 and is caught, engine 2 fails at 50 and is caught too late."""
    rows = []
    for cycle in range(1, 101):
        rows.append({"unit": 1, "cycle": cycle, "failure_cycle": 100, "lower": 100 if cycle < 85 else 10})
    for cycle in range(1, 51):
        rows.append({"unit": 2, "cycle": cycle, "failure_cycle": 50, "lower": 100 if cycle < 48 else 10})
    return pd.DataFrame(rows)


ASSUMPTIONS = simulator.Assumptions(
    cost_unscheduled=1000.0,
    cost_scheduled=100.0,
    downtime_unscheduled=10.0,
    downtime_scheduled=1.0,
    cost_per_wasted_cycle=1.0,
    fixed_interval=60,
    predictive_threshold=20,
    parts_lead_cycles=5,
)


def test_cost_model_matches_a_hand_computed_fleet():
    results = simulator.simulate(_two_engine_fleet(), ASSUMPTIONS)

    # Both engines run to failure: 2 x 1000, no life wasted.
    assert results.loc["Run to failure", "total_cost"] == 2000.0
    assert results.loc["Run to failure", "downtime_days"] == 20.0

    # Pull at cycle 60: engine 1 is scheduled with 40 cycles wasted, engine 2 has already failed.
    assert results.loc["Fixed interval", "total_cost"] == 1000.0 + 100.0 + 40.0
    assert results.loc["Fixed interval", "unscheduled_failures"] == 1

    # Engine 1 crosses at 85, acts at 90, wasting 10 cycles. Engine 2 crosses at 48 and the
    # 5 cycle lead time puts the slot at 53, past its failure at 50, so it is a failure.
    assert results.loc["Predictive", "total_cost"] == 1000.0 + 100.0 + 10.0
    assert results.loc["Predictive", "unscheduled_failures"] == 1
    assert results.loc["Predictive", "wasted_cycles"] == 10.0


def test_parts_lead_time_changes_whether_a_warning_is_useful():
    """With no lead time the late warning on engine 2 becomes actionable."""
    instant = simulator.simulate(_two_engine_fleet(), simulator.Assumptions(
        **{**ASSUMPTIONS.__dict__, "parts_lead_cycles": 0}
    ))
    assert instant.loc["Predictive", "unscheduled_failures"] == 0


def test_sensitivity_covers_every_threshold():
    curve = simulator.sensitivity(_two_engine_fleet(), ASSUMPTIONS, thresholds=range(5, 35, 5))
    assert list(curve.index) == [5, 10, 15, 20, 25, 30]
    assert curve["total_cost"].notna().all()
