"""Streamlit entrypoint: shared controls, then routing.

Run with:  uv run streamlit run src/copilot/app.py

Each view gets its own URL, so /evaluation can be linked directly. The lambdas below
are deliberate: st.Page takes a zero-argument callable, and wrapping keeps each view's
body() an ordinary function with an explicit signature that a test can call on its own.
Without them, every view would reach into session state for its data.
"""

from __future__ import annotations

import streamlit as st

from copilot import shared
from copilot.views import business_case, engine, evaluation, fleet, operations, overview

st.set_page_config(page_title="Predictive maintenance copilot", layout="wide")


def main() -> None:
    shared.require_artifacts()
    st.sidebar.title("Maintenance copilot")

    predictions = shared.load_predictions()
    simulation = shared.load_simulation()
    metrics = shared.load_metrics()
    llm_model, llm_effort = shared.model_picker()

    st.navigation(
        {
            "Start here": [
                st.Page(
                    lambda: overview.body(metrics, simulation, shared.current_assumptions()),
                    title="Overview",
                    icon=":material/insights:",
                    url_path="overview",
                    default=True,
                ),
            ],
            "Operate": [
                st.Page(
                    lambda: fleet.body(predictions),
                    title="Fleet",
                    icon=":material/dashboard:",
                    url_path="fleet",
                ),
                st.Page(
                    lambda: engine.body(predictions, metrics, llm_model, llm_effort),
                    title="Engine detail",
                    icon=":material/precision_manufacturing:",
                    url_path="engine",
                ),
                st.Page(
                    lambda: business_case.body(simulation),
                    title="Business case",
                    icon=":material/payments:",
                    url_path="business-case",
                ),
            ],
            "Evidence": [
                st.Page(
                    lambda: evaluation.body(predictions, metrics),
                    title="Evaluation",
                    icon=":material/query_stats:",
                    url_path="evaluation",
                ),
                st.Page(
                    lambda: operations.body(predictions, metrics, llm_model, llm_effort),
                    title="Operations",
                    icon=":material/speed:",
                    url_path="operations",
                ),
            ],
        }
    ).run()


main()
