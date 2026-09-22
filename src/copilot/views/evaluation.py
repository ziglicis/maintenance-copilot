"""How well the models actually do, including where they fail."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from copilot import charts, config, shared


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

    st.subheader("Worst five predictions")
    snapshot = snapshot.assign(error=(snapshot["point"] - snapshot["true_rul"]).abs())
    worst = snapshot.nlargest(5, "error")
    st.dataframe(
        worst[["cycle", "true_rul", "point", "lower", "upper", "error", "driver1", "tier"]]
        .rename(columns={"point": "predicted", "true_rul": "true RUL", "driver1": "top driver"})
        .style.format({
            "cycle": "{:.0f}", "true RUL": "{:.0f}", "predicted": "{:.1f}",
            "lower": "{:.0f}", "upper": "{:.0f}", "error": "{:.1f}",
        })
    )
    st.caption(
        "The large errors cluster on engines whose sensors are still near baseline at the truncation "
        "point: with no visible degradation the model falls back to the population average, which is "
        "the capped label of 125 cycles."
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
