"""The checks that would catch a real regression. No fixtures, no mocking frameworks.

    uv run pytest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from copilot import config, data, evaluate, models, simulator, telemetry, workorder


def test_rul_labels_are_piecewise_capped():
    """A run-to-failure engine counts down to zero, flat above the cap."""
    frame = pd.DataFrame({"unit": [1] * 5, "cycle": [1, 2, 3, 4, 5]})
    labelled = data.add_rul(frame)
    assert labelled["rul"].tolist() == [4, 3, 2, 1, 0]
    assert labelled["rul_capped"].tolist() == [4, 3, 2, 1, 0]

    long_engine = pd.DataFrame({"unit": [1] * 200, "cycle": range(1, 201)})
    capped = data.add_rul(long_engine)["rul_capped"]
    assert capped.max() == config.RUL_CAP
    assert capped.iloc[-1] == 0
    assert (capped.iloc[:75] == config.RUL_CAP).all()


def test_test_engine_rul_is_offset_by_the_true_remaining_life():
    """Test trajectories stop early, so their labels have to carry the offset."""
    frame = pd.DataFrame({"unit": [7] * 3, "cycle": [1, 2, 3]})
    labelled = data.add_rul(frame, pd.Series({7: 50}))
    assert labelled["rul"].tolist() == [52, 51, 50]


def test_split_is_by_engine_not_by_row():
    """Splitting rows instead of engines would leak a failure trajectory across the split."""
    train = pd.DataFrame({"unit": np.repeat(np.arange(1, 101), 3), "cycle": [1, 2, 3] * 100})
    fit, val = data.split_engines(train)
    assert len(val) == config.VAL_ENGINES
    assert len(fit) + len(val) == 100
    assert not set(fit) & set(val)


def test_nasa_score_punishes_late_predictions_harder():
    """Being late means the engine fails before you pull it, so it must cost more."""
    true = np.array([50.0])
    early = models.nasa_score(true, np.array([40.0]))  # 10 cycles early
    late = models.nasa_score(true, np.array([60.0]))  # 10 cycles late
    assert late > early
    assert models.nasa_score(true, true) == pytest.approx(0.0)


def test_tiers_follow_the_configured_thresholds():
    assigned = models.tier(np.array([0, config.TIER_RED - 1, config.TIER_RED, config.TIER_AMBER + 1]))
    assert assigned.tolist() == ["Red", "Red", "Amber", "Green"]


def test_conformal_widening_restores_coverage():
    """An overconfident band around a noisy prediction should be widened until it holds."""
    rng = np.random.default_rng(0)
    true = rng.normal(50, 10, 500)
    point = true + rng.normal(0, 5, 500)  # the prediction is off, the band pretends it is not
    preds = pd.DataFrame({"point": point, "q10": point - 1, "q90": point + 1})

    assert ((true >= preds["q10"]) & (true <= preds["q90"])).mean() < 0.3

    width = models.conformal_width(true, preds, alpha=0.2)
    widened = models.apply_conformal(preds, width)
    coverage = ((true >= widened["lower"]) & (true <= widened["upper"])).mean()
    assert width > 0
    assert coverage >= 0.8


def test_conformal_shrinks_a_band_that_is_wider_than_it_needs_to_be():
    """Calibration corrects in both directions, so a cautious model is not left vague."""
    true = np.full(200, 50.0)
    preds = pd.DataFrame({"point": true, "q10": true - 30, "q90": true + 30})
    assert models.conformal_width(true, preds, alpha=0.2) < 0


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


def _replay_frame(rows: list[tuple[int, int, int, str]]) -> pd.DataFrame:
    """(unit, cycle, true_rul, tier) rows, which is all replay_metrics reads."""
    return pd.DataFrame(rows, columns=["unit", "cycle", "true_rul", "tier"])


def test_early_warning_only_scores_engines_that_reach_the_decision_point():
    """An engine whose data stops while it is healthy is not a missed warning."""
    frame = _replay_frame(
        # engine 1 runs down to 5 remaining and is flagged at 40: warned in time
        [(1, c, 45 - c, "Red" if 45 - c <= 40 else "Green") for c in range(1, 41)]
        # engine 2 stops at 80 remaining, never flagged, and must not be scored at all
        + [(2, c, 120 - c, "Green") for c in range(1, 41)]
    )
    result = evaluate.replay_metrics(frame)
    assert result["early_warning_eligible_engines"] == 1
    assert result["early_warning_rate"] == 1.0


def test_a_warning_that_arrives_too_late_is_not_an_early_warning():
    frame = _replay_frame([(1, c, 30 - c, "Red" if 30 - c <= 10 else "Green") for c in range(1, 30)])
    result = evaluate.replay_metrics(frame)
    assert result["early_warning_eligible_engines"] == 1
    assert result["early_warning_rate"] == 0.0
    assert result["median_lead_cycles"] == 10.0


def test_flagging_a_healthy_engine_counts_as_a_false_alarm():
    frame = _replay_frame([(1, c, 200 - c, "Red") for c in range(1, 30)])
    result = evaluate.replay_metrics(frame)
    assert result["false_alarm_rate"] == 1.0
    assert result["flagged_engines"] == 1


def test_validation_points_stay_inside_their_engines_and_are_warm():
    frame = pd.DataFrame({
        "unit": np.repeat([1, 2, 3], 100),
        "cycle": list(range(1, 101)) * 3,
    })
    picked = evaluate.validation_points(frame, per_engine=5)
    chosen = frame.loc[picked]
    assert len(picked) == 15
    assert chosen["cycle"].min() >= config.SLOPE_WINDOW
    assert chosen.groupby("unit").size().tolist() == [5, 5, 5]


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


def _render_view(name: str) -> None:
    """Render one view with real artifacts, the way the entrypoint wires it up."""
    # Imported inside: AppTest.from_function runs this source in a fresh namespace,
    # so nothing imported at module level in this test file is visible here.
    from copilot import config, shared, simulator
    from copilot.views import demo, evaluation, operations, overview

    predictions = shared.load_predictions()
    simulation = shared.load_simulation()
    metrics = shared.load_metrics()
    model = config.LLM_MODEL

    {
        "overview": lambda: overview.body(metrics, simulation, simulator.Assumptions()),
        "demo": lambda: demo.body(predictions, simulation, metrics, model),
        "evaluation": lambda: evaluation.body(predictions, metrics),
        "operations": lambda: operations.body(metrics, model),
    }[name]()


@pytest.mark.skipif(not config.PREDICTIONS.exists(), reason="run scripts/train.py first")
@pytest.mark.parametrize("view", ["overview", "demo", "evaluation", "operations"])
def test_each_view_renders_without_error(view):
    """Catches a broken chart or a renamed column, which data tests will not."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_render_view, kwargs={"name": view}, default_timeout=120)
    app.run()
    assert not app.exception, app.exception


@pytest.mark.skipif(not config.PREDICTIONS.exists(), reason="run scripts/train.py first")
def test_the_router_builds_every_page():
    """The entrypoint itself: navigation config, sidebar widgets and the default page."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.ROOT / "src" / "copilot" / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
