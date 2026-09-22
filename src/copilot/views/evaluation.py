"""How well the models actually do, including where they fail."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from copilot import charts, config, data, shared


def body(predictions: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Model comparison")
    table = pd.DataFrame(metrics["models"]).T
    st.dataframe(
        table[["val_rmse", "test_rmse", "test_nasa_score", "train_seconds", "fleet_inference_seconds"]]
        .rename(columns={
            "val_rmse": "validation RMSE", "test_rmse": "test RMSE", "test_nasa_score": "NASA score",
            "train_seconds": "train (s)", "fleet_inference_seconds": "fleet inference (s)",
        })
        .style.format({
            "validation RMSE": "{:.2f}", "test RMSE": "{:.2f}", "NASA score": "{:.0f}",
            "train (s)": "{:.2f}", "fleet inference (s)": "{:.3f}",
        })
    )
    st.caption(
        f"Production model chosen on the validation split: **{metrics['production_model']}**. The test set "
        "was scored once, at the end. Published deep-learning results on FD001 sit around 11 to 14 RMSE."
    )

    snapshot = shared.fleet_snapshot(predictions)
    left, right = st.columns(2)
    left.plotly_chart(charts.accuracy_scatter(snapshot), use_container_width=True)
    right.plotly_chart(charts.lead_time_histogram(predictions), use_container_width=True)

    gbm = metrics["models"]["lightgbm"]
    cols = st.columns(3)
    cols[0].metric("Interval coverage", f"{gbm['interval_coverage']:.0%}",
                   help=f"Nominal 80%. Raw quantiles gave {gbm['raw_interval_coverage']:.0%} before "
                        f"conformal calibration widened them by {gbm['conformal_width_cycles']:.1f} cycles")
    cols[1].metric("False alarm rate", f"{gbm['false_alarm_rate']:.0%}",
                   help=f"Share of Red flags raised with more than {config.TIER_AMBER} cycles remaining")
    cols[2].metric("Median lead time", f"{gbm['median_lead_cycles']:.0f} cycles")

    render_workorder_eval()
    render_ablation()

    render_failures(predictions)


def failure_hypothesis(row: pd.Series, history: pd.DataFrame, sensors: list[str]) -> str:
    """Why this engine was mispredicted, argued from its own data rather than asserted.

    Three causes account for nearly all of the large errors, and they are separable:
    the label cap makes long lives unpredictable by construction, an engine truncated
    while still healthy gives the model nothing to work with, and a late prediction on
    a degrading engine is the operationally dangerous case.
    """
    error = row["point"] - row["true_rul"]
    departed = sum(
        not (history[s].iloc[:20].min() <= row[s] <= history[s].iloc[:20].max())
        for s in sensors
    )
    at_cap = row["point"] >= config.RUL_CAP - 5

    if row["true_rul"] > config.RUL_CAP and at_cap:
        return (
            f"Unpredictable by construction. True life is {row['true_rul']:.0f} cycles, above the "
            f"{config.RUL_CAP} cycle label cap, so the model cannot score better than "
            f"{row['true_rul'] - config.RUL_CAP:.0f} cycles of error here however good it is."
        )
    if departed <= 2:
        return (
            f"No degradation visible yet. Only {departed} of {len(sensors)} sensors have left their "
            f"healthy band, so the model falls back towards the population average and predicts "
            f"{row['point']:.0f} against a true {row['true_rul']:.0f}."
        )
    if error > 0:
        return (
            f"Late by {error:.0f} cycles, the costly direction. {departed} sensors are outside their "
            "healthy band, so degradation is visible, but the model reads it as less advanced than "
            "it is. This is the error the NASA score penalises hardest."
        )
    return (
        f"Early by {-error:.0f} cycles. {departed} sensors are outside their healthy band and the "
        "model over-weights them, pulling a serviceable engine sooner than needed."
    )


def render_failures(predictions: pd.DataFrame) -> None:
    """The five worst predictions, each with its evidence and a hypothesis."""
    st.subheader("Where it fails")
    snapshot = shared.fleet_snapshot(predictions)
    snapshot = snapshot.assign(error=(snapshot["point"] - snapshot["true_rul"]))
    worst = snapshot.reindex(snapshot["error"].abs().sort_values(ascending=False).index).head(5)
    sensors = json.loads(config.METRICS.read_text())["sensors_used"]

    st.caption(
        "Sorted by absolute error on the final observed cycle. Each hypothesis is derived from that "
        "engine's own data, not written by hand."
    )
    for rank, (unit, row) in enumerate(worst.iterrows(), start=1):
        history = predictions[predictions["unit"] == unit].reset_index(drop=True)
        label = (f"{rank}.  engine {unit}   predicted {row['point']:.0f}, true {row['true_rul']:.0f}, "
                 f"error {row['error']:+.0f} cycles")
        with st.expander(label, expanded=(rank == 1)):
            st.markdown(failure_hypothesis(row, history, sensors))
            left, right = st.columns(2)
            left.plotly_chart(
                charts.prediction_history(history, int(row["cycle"])),
                use_container_width=True, key=f"fail_pred_{unit}",
            )
            right.plotly_chart(
                charts.sensor_trend(history, data.sensor_of(row["driver1"]), int(row["cycle"])),
                use_container_width=True, key=f"fail_sensor_{unit}",
            )


def render_workorder_eval() -> None:
    """Measured work order quality, from scripts/eval_workorders.py."""
    st.subheader("Work order quality")
    if not config.WORKORDER_EVAL.exists():
        st.info("No batch evaluation yet. Run `uv run python scripts/eval_workorders.py`.")
        return

    report = json.loads(config.WORKORDER_EVAL.read_text())
    cols = st.columns(4)
    cols[0].metric("Groundedness", f"{report['groundedness_rate']:.0%}",
                   help=f"Work orders passing all rules, over {report['calls']} calls")
    cols[1].metric("Cross-model agreement", f"{report['cross_model_agreement']:.0%}",
                   help="Engines where every run, across both models, named the same subsystem")
    cols[2].metric("Calls", report["calls"])
    cols[3].metric("Spend", f"${report['spend_usd']:.2f}")

    st.dataframe(
        pd.DataFrame(report["per_model"]).T.rename(columns={
            "groundedness_rate": "groundedness", "within_model_agreement": "self-agreement",
            "mean_latency_s": "latency (s)", "mean_cost_usd": "cost ($)", "mean_evidence": "evidence items",
        }).style.format({
            "calls": "{:.0f}", "groundedness": "{:.0%}", "self-agreement": "{:.0%}",
            "latency (s)": "{:.1f}", "cost ($)": "${:.4f}", "evidence items": "{:.1f}",
        })
    )

    if report["failed_rules"]:
        st.markdown("**Which rule failed**")
        st.dataframe(
            pd.DataFrame(sorted(report["failed_rules"].items(), key=lambda kv: -kv[1]),
                         columns=["rule", "failures"]),
            hide_index=True,
        )
    st.caption(
        "Groundedness says the cited evidence is real. It does not say the diagnosis drawn from "
        "that evidence is right: two models can both pass every rule and still name different "
        "subsystems, which is why the agreement figure is reported next to it."
    )


def render_ablation() -> None:
    """Does the synthetic manual actually earn its place in the prompt?"""
    if not config.MANUAL_ABLATION.exists():
        return
    st.subheader("Does the manual help?")
    arms = json.loads(config.MANUAL_ABLATION.read_text())
    table = pd.DataFrame(arms).T.rename(index={"full": "with manual", "no-manual": "without manual"})
    st.dataframe(
        table.rename(columns={
            "groundedness_rate": "groundedness", "task_codes_valid": "codes valid",
            "priority_matches_manual": "priority correct", "named_a_subsystem": "named a subsystem",
            "mean_cost_usd": "cost ($)",
        }).style.format({
            "calls": "{:.0f}", "groundedness": "{:.0%}", "codes valid": "{:.0%}",
            "priority correct": "{:.0%}", "named a subsystem": "{:.0%}", "cost ($)": "${:.4f}",
        })
    )
    st.markdown(
        "Removing the manual nearly halves priority accuracy, because the priority thresholds live "
        "in it, and the model names a specific subsystem less often, because the sensor-to-module "
        "map lives there too. It also stops citing parts entirely: 0 part numbers against 1.4 per "
        "work order, so nothing can be staged against a 35 day lead time.\n\n"
        "Two results need care. Groundedness goes **up** without the manual, because a thinner "
        "prompt produces terser prose with fewer unverified sensor mentions. And 'codes valid' "
        "stays at 100% only because 8 of 12 work orders cite no codes at all; the four that do "
        "borrow them from the synthetic maintenance history, which leaks the vocabulary "
        "independently of the manual."
    )
