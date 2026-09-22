"""Scoring functions, risk tiers and interval calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from copilot import config, models


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
