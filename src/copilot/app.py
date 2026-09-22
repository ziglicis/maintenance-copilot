"""Streamlit front end: four tabs over precomputed artifacts.

Run with:  uv run streamlit run src/copilot/app.py

No model is loaded here. Everything except work order generation reads parquet and
JSON written by scripts/train.py, which is what keeps the fleet view instant.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from copilot import charts, config, models, simulator, telemetry, theme, workorder

st.set_page_config(page_title="Predictive maintenance copilot", layout="wide")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(config.PREDICTIONS)


@st.cache_data
def load_simulation() -> pd.DataFrame:
    return pd.read_parquet(config.SIMULATION)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(config.METRICS.read_text())


def require_artifacts() -> None:
    missing = [p.name for p in (config.PREDICTIONS, config.SIMULATION, config.METRICS) if not p.exists()]
    if missing:
        st.error(f"Missing artifacts: {', '.join(missing)}. Run `uv run python scripts/train.py` first.")
        st.stop()


def fleet_snapshot(predictions: pd.DataFrame) -> pd.DataFrame:
    """The latest observed cycle for each engine, which is what a planner sees."""
    return predictions.loc[predictions.groupby("unit")["cycle"].idxmax()].set_index("unit")


def annual_factor(simulation: pd.DataFrame) -> float:
    """Convert one simulated fleet life into a per-year figure, visibly."""
    mean_life = simulation.groupby("unit")["cycle"].max().mean()
    return config.CYCLES_PER_ENGINE_PER_YEAR / mean_life


# --------------------------------------------------------------------------- sidebar

def assumption_form() -> simulator.Assumptions:
    """Every number behind the savings figure, editable."""
    st.sidebar.header("Cost assumptions")
    st.sidebar.caption("The savings figure is computed from these. Nothing is hidden.")
    return simulator.Assumptions(
        cost_unscheduled=st.sidebar.number_input(
            "Unscheduled failure ($)", value=float(config.COST_UNSCHEDULED_FAILURE), step=50_000.0
        ),
        cost_scheduled=st.sidebar.number_input(
            "Scheduled maintenance ($)", value=float(config.COST_SCHEDULED_MAINTENANCE), step=10_000.0
        ),
        downtime_unscheduled=st.sidebar.number_input(
            "Downtime, unscheduled (days)", value=float(config.DOWNTIME_DAYS_UNSCHEDULED), step=1.0
        ),
        downtime_scheduled=st.sidebar.number_input(
            "Downtime, scheduled (days)", value=float(config.DOWNTIME_DAYS_SCHEDULED), step=1.0
        ),
        cost_per_wasted_cycle=st.sidebar.number_input(
            "Value of a wasted cycle ($)", value=float(config.COST_PER_WASTED_CYCLE), step=50.0
        ),
        fixed_interval=st.sidebar.number_input(
            "Fixed interval (cycles)", value=int(config.FIXED_INTERVAL_CYCLES), step=10
        ),
        predictive_threshold=st.sidebar.number_input(
            "Predictive pull threshold (cycles)", value=int(config.PREDICTIVE_THRESHOLD), step=5
        ),
        parts_lead_cycles=st.sidebar.number_input(
            "Parts and slot lead time (cycles)", value=int(config.PARTS_LEAD_CYCLES), step=5
        ),
    )



# --------------------------------------------------------------------------- tabs

def tab_overview(metrics: dict, simulation: pd.DataFrame, assumptions: simulator.Assumptions) -> None:
    st.subheader("Fleet health for 100 turbofan engines")
    st.markdown(
        "Maintenance today is either time-based, which pulls healthy engines early, or reactive, "
        "which grounds aircraft without warning. Sensor data is collected every cycle and rarely "
        "used, because nobody has time to read 21 streams per engine.\n\n"
        "This system predicts remaining useful life per engine, ranks the fleet by risk, writes a "
        "work order that cites the sensor evidence behind each prediction, and prices the "
        "maintenance policy against the alternatives."
    )

    results = simulator.simulate(simulation, assumptions)
    savings = (results.loc["Fixed interval", "total_cost"] - results.loc["Predictive", "total_cost"])
    gbm = metrics["models"]["lightgbm"]

    left, middle, right = st.columns(3)
    left.metric("RUL error (RMSE)", f"{gbm['test_rmse']:.1f} cycles", help="Official FD001 test split")
    middle.metric(
        "Early warning rate",
        f"{gbm['early_warning_rate']:.0%}",
        help=f"Flagged at least {config.LEAD_TARGET_CYCLES} cycles before failure, across the {gbm['early_warning_eligible_engines']} "
        "test engines whose data reaches that point",
    )
    right.metric(
        "Modelled saving per year",
        f"${savings * annual_factor(simulation) / 1e6:.1f}M",
        help="Predictive against fixed-interval, using the sidebar assumptions",
    )

    st.graphviz_chart(
        """
        digraph {
            rankdir=LR; bgcolor="#fcfcfb";
            node [shape=box style="rounded,filled" fillcolor="#f0efec" color="#e8e7e3"
                  fontname="Helvetica" fontsize=10 fontcolor="#0b0b0b"];
            edge [color="#898781" arrowsize=0.6];
            raw [label="C-MAPSS\\nsensor data"];
            feat [label="Feature pipeline"];
            model [label="RUL models\\nRidge / LightGBM"];
            pred [label="Predictions\\n+ intervals\\n+ attributions"];
            rank [label="Fleet risk ranking"];
            wo [label="Work order\\ngenerator"];
            synth [label="Synthetic logs\\n+ manual" fillcolor="#fcfcfb"];
            check [label="Groundedness check"];
            sim [label="Policy simulator"];
            ui [label="Streamlit UI" fillcolor="#e8e7e3"];
            db [label="Telemetry\\nSQLite" fillcolor="#fcfcfb"];
            raw -> feat -> model -> pred;
            pred -> rank; pred -> wo; pred -> sim; synth -> wo; wo -> check;
            rank -> ui; check -> ui; sim -> ui; db -> ui [dir=none];
        }
        """
    )
    st.caption(
        "C-MAPSS is simulated engine data from NASA. The maintenance logs and the manual extract "
        "in this app are synthetic and were written for this project."
    )


def tab_demo(predictions: pd.DataFrame, simulation: pd.DataFrame, metrics: dict,
             assumptions: simulator.Assumptions, llm_model: str) -> None:
    snapshot = fleet_snapshot(predictions)
    sensors = metrics["sensors_used"]

    st.subheader("Fleet")
    counts = snapshot["tier"].value_counts()
    cols = st.columns(4)
    for col, tier in zip(cols, ("Red", "Amber", "Green")):
        col.metric(f"{theme.TIER_MARK[tier]} {tier}", int(counts.get(tier, 0)))
    cols[3].metric("Engines", len(snapshot))

    table = pd.DataFrame(
        {
            "cycle": snapshot["cycle"],
            "predicted RUL": snapshot["point"].round(1),
            "interval": [f"{lo:.0f} to {hi:.0f}" for lo, hi in zip(snapshot["lower"], snapshot["upper"])],
            "tier": snapshot["tier"],
            "top driver": snapshot["driver1"],
            "true RUL": snapshot["true_rul"],
        }
    ).sort_values("predicted RUL")

    left, right = st.columns([3, 2])
    left.dataframe(
        table.style.map(lambda t: f"color: {theme.TIER_COLOURS[t]}; font-weight: 600", subset=["tier"]),
        height=360,
    )
    right.plotly_chart(charts.rul_histogram(snapshot), use_container_width=True)

    st.divider()
    st.subheader("Engine detail and replay")
    engine = st.selectbox("Engine", options=sorted(predictions["unit"].unique()),
                          index=int(table.index[0]) - 1 if len(table) else 0)
    history = predictions[predictions["unit"] == engine].reset_index(drop=True)
    cycle = st.slider(
        "Replay cycle", int(history["cycle"].min()), int(history["cycle"].max()),
        int(history["cycle"].max()),
        help="Step the engine forward and watch the prediction and tier update",
    )
    row = history[history["cycle"] == cycle].iloc[0]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted RUL", f"{row['point']:.0f} cycles")
    metric_cols[1].metric("Interval", f"{row['lower']:.0f} to {row['upper']:.0f}")
    metric_cols[2].metric("Tier", f"{theme.TIER_MARK[row['tier']]} {row['tier']}")
    metric_cols[3].metric("True RUL", f"{row['true_rul']:.0f} cycles")

    st.plotly_chart(charts.prediction_history(history, cycle), use_container_width=True)
    st.plotly_chart(charts.attribution_chart(row), use_container_width=True)

    driver_sensors = list(dict.fromkeys(models.sensor_of(row[f"driver{i}"]) for i in (1, 2, 3)))
    for col, sensor in zip(st.columns(len(driver_sensors)), driver_sensors):
        col.plotly_chart(charts.sensor_trend(history, sensor, cycle), use_container_width=True)

    st.divider()
    st.subheader("Work order")
    st.caption(f"Generated by {llm_model}. The prediction is fixed input: the model explains it, never changes it.")
    if st.button("Generate work order", type="primary"):
        with st.spinner("Writing work order"):
            try:
                generated = workorder.generate(row, sensors, history, model=llm_model)
            except Exception as error:  # surfaced, never swallowed
                st.error(f"{type(error).__name__}: {error}")
                return
        st.session_state["work_order"] = generated

    if st.session_state.get("work_order") is not None:
        render_work_order(st.session_state["work_order"])

    st.divider()
    st.subheader("Policy simulator")
    st.caption(
        f"Replays {simulation['unit'].nunique()} engines with complete run-to-failure trajectories, using "
        "out-of-fold predictions so no engine is scored by a model that trained on it. The truncated test "
        "engines cannot be used here: most stop recording while still healthy, long before any pull decision."
    )
    results = simulator.simulate(simulation, assumptions)
    curve = simulator.sensitivity(simulation, assumptions)
    left, right = st.columns(2)
    left.plotly_chart(charts.policy_chart(results), use_container_width=True)
    right.plotly_chart(charts.sensitivity_chart(curve, assumptions.predictive_threshold), use_container_width=True)
    st.dataframe(
        results.assign(
            total_cost=results["total_cost"].map("${:,.0f}".format),
            downtime_days=results["downtime_days"].map("{:,.0f}".format),
            wasted_cycles=results["wasted_cycles"].map("{:,.0f}".format),
        )
    )


def render_work_order(generated: workorder.Generated) -> None:
    order = generated.work_order
    if generated.grounded:
        st.success("Groundedness check passed: every cited sensor value matches the data")
    else:
        st.error("Groundedness check failed: " + "; ".join(generated.failures))

    head = st.columns(4)
    head[0].metric("Priority", order.priority)
    head[1].metric("Subsystem", order.suspected_subsystem)
    head[2].metric("Predicted RUL", f"{order.predicted_rul} cycles")
    head[3].metric("Interval", f"{order.interval.low} to {order.interval.high}")

    left, right = st.columns(2)
    left.markdown("**Evidence**")
    left.dataframe(pd.DataFrame([e.model_dump() for e in order.evidence]), hide_index=True)
    right.markdown("**Recommended actions**")
    right.markdown("\n".join(f"1. {a}" for a in order.recommended_actions))
    if order.parts_to_stage:
        right.markdown("**Parts to stage**")
        right.markdown("\n".join(f"- {p}" for p in order.parts_to_stage))

    st.markdown(f"**Justification.** {order.justification}")
    st.markdown(f"**Confidence.** {order.confidence_note}")
    st.caption(
        f"{generated.latency_s:.1f} s, {generated.input_tokens} in / {generated.output_tokens} out, "
        f"${generated.cost_usd:.4f}"
    )
    with st.expander("Raw JSON"):
        st.json(order.model_dump())


def tab_evaluation(predictions: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Model comparison")
    table = pd.DataFrame(metrics["models"]).T
    st.dataframe(
        table[["val_rmse", "test_rmse", "test_nasa_score", "train_seconds", "fleet_inference_seconds"]]
        .rename(columns={
            "val_rmse": "validation RMSE", "test_rmse": "test RMSE", "test_nasa_score": "NASA score",
            "train_seconds": "train (s)", "fleet_inference_seconds": "fleet inference (s)",
        })
        .round(3)
    )
    st.caption(
        f"Production model chosen on the validation split: **{metrics['production_model']}**. The test set "
        "was scored once, at the end. Published deep-learning results on FD001 sit around 11 to 14 RMSE."
    )

    snapshot = fleet_snapshot(predictions)
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

    st.subheader("Worst five predictions")
    snapshot = snapshot.assign(error=(snapshot["point"] - snapshot["true_rul"]).abs())
    worst = snapshot.nlargest(5, "error")
    st.dataframe(
        worst[["cycle", "true_rul", "point", "lower", "upper", "error", "driver1", "tier"]]
        .rename(columns={"point": "predicted", "true_rul": "true RUL", "driver1": "top driver"})
        .round(1)
    )
    st.caption(
        "The large errors cluster on engines whose sensors are still near baseline at the truncation "
        "point: with no visible degradation the model falls back to the population average, which is "
        "the capped label of 125 cycles."
    )


def tab_operations(metrics: dict, llm_model: str) -> None:
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

    st.dataframe(telemetry.summary_by_model(calls).round(4))

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
        calls[["created_at", "engine_id", "cycle", "model", "latency_s", "cost_usd", "grounded", "failures"]]
        .head(20)
        .round(3),
        hide_index=True,
    )

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


def main() -> None:
    require_artifacts()
    predictions = load_predictions()
    simulation = load_simulation()
    metrics = load_metrics()

    st.title("Predictive maintenance copilot")
    assumptions = assumption_form()
    st.sidebar.header("Work order generator")
    llm_model = st.sidebar.selectbox("Model", options=config.LLM_MODELS, index=0)

    overview, demo, evaluation, operations = st.tabs(["Overview", "Live demo", "Evaluation", "Operations"])
    with overview:
        tab_overview(metrics, simulation, assumptions)
    with demo:
        tab_demo(predictions, simulation, metrics, assumptions, llm_model)
    with evaluation:
        tab_evaluation(predictions, metrics)
    with operations:
        tab_operations(metrics, llm_model)


main()
