"""What the system does and what it is worth."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from copilot import charts, config, shared, simulator


def body(metrics: dict, simulation: pd.DataFrame, assumptions: simulator.Assumptions) -> None:
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
        f"${savings * shared.annual_factor(simulation) / 1e6:.1f}M",
        help="Predictive against fixed-interval. Change the assumptions on the Live demo page",
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
