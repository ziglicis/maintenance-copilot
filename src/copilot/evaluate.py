"""Evaluation logic: how models are scored and how policies get their predictions.

This lives in the package rather than in the training script because it decides two of
the headline numbers. Anything that can change a reported metric should be importable
and testable, not buried in a one-shot script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from copilot import config, models


def validation_points(val: pd.DataFrame, per_engine: int = 5) -> pd.Index:
    """Sample truncation points so validation looks like the test set.

    Test engines are cut off at one unknown cycle, so scoring validation on every row
    would overweight healthy early life and flatter the model.
    """
    rng = np.random.default_rng(config.SEED)
    warm = val[val["cycle"] >= config.SLOPE_WINDOW]
    picks = [
        rng.choice(group.index, size=min(per_engine, len(group)), replace=False)
        for _, group in warm.groupby("unit")
    ]
    return pd.Index(np.concatenate(picks))


def out_of_fold(
    train_raw: pd.DataFrame,
    train_labelled: pd.DataFrame,
    train_x: pd.DataFrame,
    features: list[str],
    sensors: list[str],
    rounds: dict[str, int],
    width: float,
    folds: int = 5,
) -> pd.DataFrame:
    """Predictions for every training engine from models that never saw it.

    The policy simulator needs complete lives. Test trajectories are truncated at an
    unknown cycle, usually while the engine is still healthy, so a policy that acts
    near end of life cannot be measured on them at all. Training engines do run to
    failure, and cross-fitting by engine keeps every prediction out of sample.
    """
    units = np.sort(train_raw["unit"].unique())
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(units)

    frames = []
    for held in np.array_split(units, folds):
        held_rows = train_raw["unit"].isin(held).to_numpy()
        gbm = models.train_gbm(
            train_x.loc[~held_rows, features],
            train_labelled.loc[~held_rows, "rul_capped"],
            rounds=rounds,
        )
        # ponytail: the conformal width from the main validation split is reused here
        # rather than recalibrated per fold. Recalibrate if fold widths ever diverge.
        preds = models.apply_conformal(models.predict_gbm(gbm, train_x.loc[held_rows, features]), width)
        frames.append(
            pd.concat([train_labelled.loc[held_rows, ["unit", "cycle", "rul", *sensors]], preds], axis=1)
        )

    frame = pd.concat(frames).sort_values(["unit", "cycle"]).reset_index(drop=True)
    frame = frame.rename(columns={"rul": "true_rul"})
    frame["tier"] = models.tier(frame["point"])
    frame["last_cycle"] = frame.groupby("unit")["cycle"].transform("max")
    frame["failure_cycle"] = frame["last_cycle"]  # these engines run to failure
    return frame


def replay_metrics(
    frame: pd.DataFrame,
    lead_target: int = config.LEAD_TARGET_CYCLES,
    healthy: int = config.TIER_AMBER,
) -> dict:
    """Early warning and false alarms, measured by replaying every engine cycle by cycle.

    Only engines whose observed data actually reaches the `lead_target`-cycles-remaining
    point can be judged on warning lead time. Test trajectories are truncated at an
    unknown cycle, and most stop while the engine is still healthy: never flagging those
    is correct behaviour, not a miss, so scoring them would be meaningless.

    A false alarm is a Red flag raised while more than `healthy` cycles remain. It is
    expressed as a share of flags raised, because that is what a planner experiences.
    """
    red = frame[frame["tier"] == "Red"]
    first_flag = red.groupby("unit")[["cycle", "true_rul"]].first()

    final_rul = frame.groupby("unit")["true_rul"].min()
    eligible = final_rul[final_rul <= lead_target].index
    lead = first_flag["true_rul"].reindex(eligible)
    warned = (lead >= lead_target).sum()
    false_alarms = int((first_flag["true_rul"] > healthy).sum())

    return {
        "early_warning_rate": float(warned / len(eligible)) if len(eligible) else 0.0,
        "early_warning_eligible_engines": int(len(eligible)),
        "false_alarm_rate": float(false_alarms / len(first_flag)) if len(first_flag) else 0.0,
        "flagged_engines": int(len(first_flag)),
        "median_lead_cycles": float(lead.median()) if lead.notna().any() else 0.0,
    }
