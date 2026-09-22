"""Policy simulator: what each maintenance strategy costs over the test fleet.

Each engine gets one life and one maintenance event, so the comparison is "what
would have happened to these 100 engines under policy X". Every assumption is an
argument, nothing is hidden in the arithmetic.

Test trajectories are truncated at an unknown cycle, so a predictive policy can only
act on cycles it can actually see. An engine whose predictions never cross the
threshold before its data ends is scored as running to failure, which is the
pessimistic and honest reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from copilot import config


@dataclass(frozen=True)
class Assumptions:
    """Cost model inputs. Defaults come from config and are editable in the UI."""

    cost_unscheduled: float = config.COST_UNSCHEDULED_FAILURE
    cost_scheduled: float = config.COST_SCHEDULED_MAINTENANCE
    downtime_unscheduled: float = config.DOWNTIME_DAYS_UNSCHEDULED
    downtime_scheduled: float = config.DOWNTIME_DAYS_SCHEDULED
    cost_per_wasted_cycle: float = config.COST_PER_WASTED_CYCLE
    fixed_interval: int = config.FIXED_INTERVAL_CYCLES
    predictive_threshold: int = config.PREDICTIVE_THRESHOLD
    parts_lead_cycles: int = config.PARTS_LEAD_CYCLES


def engine_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: when it truly fails, and when each policy would act.

    `pull_cycle` is the first observed cycle at which the lower bound of the predicted
    RUL drops below the threshold. Using the lower bound rather than the point estimate
    is the risk-averse choice: it acts on the pessimistic end of the interval.
    """
    return predictions.groupby("unit").agg(
        last_cycle=("cycle", "max"),
        failure_cycle=("failure_cycle", "first"),
    )


def _pull_cycles(predictions: pd.DataFrame, threshold: float) -> pd.Series:
    crossed = predictions[predictions["lower"] < threshold]
    return crossed.groupby("unit")["cycle"].min()


def simulate(predictions: pd.DataFrame, assumptions: Assumptions) -> pd.DataFrame:
    """Total cost, failures, downtime and wasted life for all three policies."""
    engines = engine_summary(predictions)
    pulls = _pull_cycles(predictions, assumptions.predictive_threshold)

    rows = [
        _run_to_failure(engines, assumptions),
        _fixed_interval(engines, assumptions),
        _predictive(engines, pulls, assumptions),
    ]
    return pd.DataFrame(rows).set_index("policy")


def _cost(
    policy: str,
    unscheduled: pd.Series,
    wasted_cycles: pd.Series,
    assumptions: Assumptions,
) -> dict:
    n_unscheduled = int(unscheduled.sum())
    n_scheduled = int((~unscheduled).sum())
    wasted = float(wasted_cycles.sum())
    return {
        "policy": policy,
        "total_cost": (
            n_unscheduled * assumptions.cost_unscheduled
            + n_scheduled * assumptions.cost_scheduled
            + wasted * assumptions.cost_per_wasted_cycle
        ),
        "unscheduled_failures": n_unscheduled,
        "scheduled_events": n_scheduled,
        "downtime_days": (
            n_unscheduled * assumptions.downtime_unscheduled + n_scheduled * assumptions.downtime_scheduled
        ),
        "wasted_cycles": wasted,
    }


def _run_to_failure(engines: pd.DataFrame, assumptions: Assumptions) -> dict:
    unscheduled = pd.Series(True, index=engines.index)
    return _cost("Run to failure", unscheduled, pd.Series(0.0, index=engines.index), assumptions)


def _fixed_interval(engines: pd.DataFrame, assumptions: Assumptions) -> dict:
    """Pull every engine at a fixed cycle count, whatever its condition."""
    scheduled_at = assumptions.fixed_interval
    failed_first = engines["failure_cycle"] <= scheduled_at
    wasted = (engines["failure_cycle"] - scheduled_at).where(~failed_first, 0.0).clip(lower=0)
    return _cost("Fixed interval", failed_first, wasted, assumptions)


def _predictive(engines: pd.DataFrame, pulls: pd.Series, assumptions: Assumptions) -> dict:
    """Pull when the predicted lower bound crosses the threshold, if it ever does.

    Deciding to pull an engine is not the same as pulling it. Parts have to be ordered
    and a slot found, so the engine keeps flying for `parts_lead_cycles` after the
    decision. A warning that arrives inside that window is too late to be useful, and
    this is what stops the simulator from rewarding an ever-lower threshold.
    """
    decision_cycle = pulls.reindex(engines.index)
    action_cycle = decision_cycle + assumptions.parts_lead_cycles
    missed = action_cycle.isna() | (action_cycle >= engines["failure_cycle"])
    wasted = (engines["failure_cycle"] - action_cycle).where(~missed, 0.0).clip(lower=0)
    return _cost("Predictive", missed, wasted, assumptions)


def sensitivity(
    predictions: pd.DataFrame,
    assumptions: Assumptions,
    thresholds: range = range(5, 65, 5),
) -> pd.DataFrame:
    """Total cost of the predictive policy across threshold choices."""
    engines = engine_summary(predictions)
    rows = []
    for threshold in thresholds:
        pulls = _pull_cycles(predictions, threshold)
        row = _predictive(engines, pulls, assumptions)
        row["threshold"] = threshold
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")
