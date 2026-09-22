"""Groundedness, the check that decides whether a work order can be trusted."""

from __future__ import annotations

import pandas as pd

from copilot import config, telemetry, workorder


def _order(**overrides) -> workorder.WorkOrder:
    payload = dict(
        engine_id=1,
        priority="Urgent",
        predicted_rul=40,
        interval=workorder.Interval(low=27, high=90),
        suspected_subsystem="HPC",
        evidence=[workorder.Evidence(sensor="s11", value=47.7, trend="rising", reference_range="47.0 to 47.4")],
        recommended_actions=["TSK-201 borescope, HPC stages 1 to 9"],
        parts_to_stage=["P-HPC-5051"],
        justification="HPC outlet pressure is above its healthy band and still climbing.",
        confidence_note="Interval is wide because degradation started recently.",
    )
    payload.update(overrides)
    return workorder.WorkOrder(**payload)


ROW = pd.Series({"unit": 1, "cycle": 133, "s11": 47.7, "s3": 1600.45, "point": 40.0})


def test_groundedness_accepts_a_faithful_work_order():
    assert workorder.check_grounded(_order(), ROW) == []


def test_groundedness_rejects_an_invented_sensor_value():
    """The failure mode that makes a system like this untrustworthy."""
    fabricated = _order(
        evidence=[workorder.Evidence(sensor="s11", value=52.9, trend="rising", reference_range="47.0 to 47.4")]
    )
    failures = workorder.check_grounded(fabricated, ROW)
    assert len(failures) == 1
    assert "s11" in failures[0] and "52.9" in failures[0]


def test_groundedness_rejects_a_changed_prediction():
    """The model may explain the number, never move it."""
    failures = workorder.check_grounded(_order(predicted_rul=65), ROW)
    assert any("RUL cited as 65" in f for f in failures)


def test_groundedness_rejects_a_sensor_that_is_not_in_the_data():
    failures = workorder.check_grounded(
        _order(evidence=[workorder.Evidence(sensor="s99", value=1.0, trend="stable", reference_range="n/a")]),
        ROW,
    )
    assert "s99" in failures[0]


def test_telemetry_round_trips_a_call(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TELEMETRY_DB", tmp_path / "telemetry.db")
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    telemetry.record(
        engine_id=7, cycle=133, model="claude-opus-5", latency_s=4.2, input_tokens=2000,
        output_tokens=400, cache_read_tokens=1800, cost_usd=0.02, grounded=False,
        failures=["s11 cited as 52.9 but the reading is 47.7000"], payload="{}",
    )
    stored = telemetry.read_all()
    assert len(stored) == 1
    assert stored.loc[0, "engine_id"] == 7
    assert stored.loc[0, "grounded"] == 0
    assert "s11" in stored.loc[0, "failures"]
    assert telemetry.summary_by_model(stored).loc["claude-opus-5", "calls"] == 1
