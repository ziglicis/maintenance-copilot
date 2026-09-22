"""Replay metrics and validation sampling. These decide two of the headline numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from copilot import config, evaluate


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
