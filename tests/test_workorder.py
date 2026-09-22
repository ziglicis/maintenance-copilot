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

# A rising s11 whose healthy band is 47.0 to 47.4, matching the reference range in _order().
HISTORY = pd.DataFrame({
    "unit": [1] * 40,
    "cycle": range(94, 134),
    "s11": [47.0 + 0.4 * (i % 2) for i in range(20)] + [47.4 + 0.015 * i for i in range(20)],
    "s3": [1590.0] * 20 + [1600.45] * 20,
})


def _failures(order, row=ROW, history=HISTORY) -> list[str]:
    """Flatten the structured checks the way the UI's badge summary does."""
    return [f for c in workorder.check_grounded(order, row, history) for f in c.failures]


def test_groundedness_accepts_a_faithful_work_order():
    assert _failures(_order()) == []


def test_groundedness_rejects_an_invented_sensor_value():
    """The failure mode that makes a system like this untrustworthy."""
    fabricated = _order(
        evidence=[workorder.Evidence(sensor="s11", value=52.9, trend="rising", reference_range="47.0 to 47.4")]
    )
    failures = _failures(fabricated)
    assert len(failures) == 1
    assert "s11" in failures[0] and "52.9" in failures[0]


def test_groundedness_rejects_a_changed_prediction():
    """The model may explain the number, never move it."""
    checks = {c.name: c for c in workorder.check_grounded(_order(predicted_rul=65), ROW, HISTORY)}
    assert not checks["Prediction unchanged"].passed
    assert any("65" in f and "40" in f for f in checks["Prediction unchanged"].failures)


def test_groundedness_rejects_a_sensor_discussed_only_in_the_prose():
    """Free text can assert anything; if it names a sensor, that sensor must be verified."""
    order = _order(justification="s20 is below its healthy range, which fits HPT wear.")
    failures = _failures(order)
    assert any("s20" in f and "justification" in f for f in failures)


def test_groundedness_allows_prose_about_a_sensor_that_is_in_evidence():
    order = _order(justification="s11 is above its healthy band and still climbing.")
    assert _failures(order) == []


def test_groundedness_ignores_task_codes_that_are_not_sensor_names():
    """TSK-201 and 'stages 1 to 9' must not be read as sensor mentions."""
    order = _order(
        justification="Run TSK-201 borescope, HPC stages 1 to 9, port B2. s11 is rising.",
        confidence_note="Interval is wide. Stage P-HPC-5051 now.",
    )
    assert _failures(order) == []


def test_groundedness_rejects_a_sensor_that_is_not_in_the_data():
    failures = _failures(
        _order(evidence=[workorder.Evidence(sensor="s99", value=1.0, trend="stable", reference_range="n/a")])
    )
    assert "s99" in failures[0]


def test_telemetry_round_trips_a_call(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TELEMETRY_DB", tmp_path / "telemetry.db")
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    telemetry.record(
        engine_id=7, cycle=133, model="claude-opus-5", effort="medium", latency_s=4.2, input_tokens=2000,
        output_tokens=400, cache_read_tokens=1800, cost_usd=0.02, grounded=False,
        failures=["s11 cited as 52.9 but the reading is 47.7000"], payload="{}",
    )
    stored = telemetry.read_all()
    assert len(stored) == 1
    assert stored.loc[0, "engine_id"] == 7
    assert stored.loc[0, "grounded"] == 0
    assert "s11" in stored.loc[0, "failures"]
    assert telemetry.summary_by_model(stored).loc[("claude-opus-5", "medium"), "calls"] == 1


def test_groundedness_rejects_a_trend_that_contradicts_the_data():
    """The model asserts a direction; nothing used to check it against the sensor."""
    order = _order(evidence=[workorder.Evidence(
        sensor="s11", value=47.7, trend="falling", reference_range="47.0000 to 47.4000")])
    failures = _failures(order)
    assert any("falling" in f and "rising" in f for f in failures)


def test_groundedness_rejects_a_fabricated_healthy_range():
    order = _order(evidence=[workorder.Evidence(
        sensor="s11", value=47.7, trend="rising", reference_range="10.0 to 20.0")])
    assert any("healthy range" in f for f in _failures(order))


def test_observed_trend_calls_noise_stable():
    """Movement smaller than the sensor's own healthy spread is not a trend."""
    flat = pd.DataFrame({"unit": [1] * 40, "cycle": range(1, 41),
                         "s11": [47.0, 47.4] * 20})
    assert workorder.observed_trend(flat, "s11", 40) == "stable"


def test_evidence_audit_reports_every_claim():
    audit = workorder.evidence_audit(_order(), ROW, HISTORY)
    assert list(audit["sensor"]) == ["s11"]
    assert set(["cited value", "actual", "delta", "value ok", "trend ok", "range ok"]) <= set(audit.columns)


def test_groundedness_rejects_an_invented_task_code():
    """A plausible-looking code the manual has never heard of is the same as a made-up value."""
    order = _order(recommended_actions=["Run TSK-999 turbine teardown"])
    assert any("TSK-999" in f for f in _failures(order))


def test_groundedness_rejects_an_invented_part_number():
    order = _order(parts_to_stage=["P-HPC-9999"])
    assert any("P-HPC-9999" in f for f in _failures(order))


def test_groundedness_accepts_real_codes_and_prose_without_codes():
    order = _order(
        recommended_actions=["Ground the aircraft at next landing.", "Run TSK-201 borescope."],
        parts_to_stage=["P-HPC-5051"],
    )
    assert _failures(order) == []
