"""What it costs to run, in latency and dollars."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from copilot import config, telemetry, workorder
from copilot.views import card


def body(predictions: pd.DataFrame, metrics: dict, llm_model: str, llm_effort: str) -> None:
    st.subheader("Inference")
    gbm = metrics["models"]["lightgbm"]
    cols = st.columns(3)
    cols[0].metric("Fleet inference", f"{gbm['fleet_inference_seconds'] * 1000:.0f} ms",
                   help=f"{metrics['n_test_engines']} engines, CPU only")
    cols[1].metric("Per engine", f"{gbm['fleet_inference_seconds'] / metrics['n_test_engines'] * 1000:.2f} ms")
    cols[2].metric("Training", f"{gbm['train_seconds']:.0f} s")

    st.subheader("Work order generation")
    calls = telemetry.read_all()
    if calls.empty:
        st.info("No work orders generated yet. Generate one in the Live demo tab to populate this tab.")
        return

    st.dataframe(
        telemetry.summary_by_model(calls).style.format({
            "calls": "{:.0f}", "p50_latency": "{:.1f} s", "p95_latency": "{:.1f} s",
            "mean_cost": "${:.4f}", "input_tokens": "{:,.0f}", "output_tokens": "{:,.0f}",
            "cache_reads": "{:,.0f}", "grounded": "{:.0%}",
        })
    )

    current = calls[calls["model"] == llm_model]
    mean_cost = current["cost_usd"].mean() if not current.empty else calls["cost_usd"].mean()
    monthly = mean_cost * config.FLEET_SIZE * config.REFRESHES_PER_MONTH
    cols = st.columns(3)
    cols[0].metric("Cost per work order", f"${mean_cost:.4f}")
    cols[1].metric(
        "Projected monthly cost",
        f"${monthly:,.0f}",
        help=f"{config.FLEET_SIZE} engines refreshed {config.REFRESHES_PER_MONTH} times a month on {llm_model}",
    )
    cols[2].metric("Groundedness pass rate", f"{calls['grounded'].mean():.0%}")

    st.dataframe(
        calls[["created_at", "engine_id", "model", "effort", "latency_s", "cost_usd", "grounded", "failures"]]
        .head(20)
        .style.format({"latency_s": "{:.1f} s", "cost_usd": "${:.4f}", "grounded": "{:.0f}"}),
        hide_index=True,
    )

    render_history(predictions, calls)

    st.subheader("Deployment notes")
    st.markdown(
        "- Inference is CPU only. The whole fleet scores in milliseconds, so there is no GPU in the "
        "serving path and no reason to batch.\n"
        "- The model artifacts are LightGBM text files. They load anywhere and diff in version control.\n"
        "- Work order generation is the only network call and the only variable cost. It is triggered per "
        "engine on demand, not on a schedule, which is why the monthly projection above is an upper bound.\n"
        "- The manual and system prompt are cached, so repeat calls within the cache window pay a tenth "
        "of the input rate for that prefix.\n"
        "- For an air-gapped site the generator swaps for a local instruct model behind the same schema. "
        "The groundedness check does not care which model wrote the work order."
    )


def render_history(predictions: pd.DataFrame, calls: pd.DataFrame) -> None:
    """Open any stored work order and re-run its audit against the source data.

    The audit is recomputed rather than stored, so a work order written weeks ago is
    still checked against the data as it stands now, and the check can be tightened
    without invalidating the history.
    """
    st.subheader("Work order history")
    labels = {
        int(r.id): f"#{int(r.id)}  engine {int(r.engine_id)} cycle {int(r.cycle)}  "
                   f"{r.model} {r.effort or 'default'}  ${r.cost_usd:.4f}  "
                   f"{'passed' if r.grounded else 'FAILED'}"
        for r in calls.itertuples()
    }
    chosen = st.selectbox(
        "Stored work order", options=list(labels), format_func=lambda i: labels[i], key="history_pick"
    )
    record = calls[calls["id"] == chosen].iloc[0]
    order = workorder.WorkOrder.model_validate_json(record.payload)

    history = predictions[predictions["unit"] == int(record.engine_id)].reset_index(drop=True)
    match = history[history["cycle"] == int(record.cycle)]
    if match.empty:
        st.warning("The engine and cycle behind this work order are not in the current predictions file.")
        return

    row = match.iloc[0]
    card.render_full(
        order, row, history, workorder.check_grounded(order, row, history),
        footer=f"{record.created_at} | {record.model} at {record.effort or 'default'} effort | "
               f"{record.latency_s:.1f} s | ${record.cost_usd:.4f}",
    )
