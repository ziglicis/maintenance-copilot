"""Streamlit entrypoint: sidebar controls, then routing.

Run with:  uv run streamlit run src/copilot/app.py

Each view gets its own URL, so /evaluation can be linked directly. The lambdas below
are deliberate: st.Page takes a zero-argument callable, and wrapping keeps each view's
body() an ordinary function with an explicit signature that a test can call on its own.
Without them, every view would reach into session state for its data.
"""

from __future__ import annotations

import streamlit as st

from copilot import shared
from copilot.views import demo, evaluation, operations, overview

st.set_page_config(page_title="Predictive maintenance copilot", layout="wide")


def main() -> None:
    shared.require_artifacts()
    st.sidebar.title("Maintenance copilot")

    predictions = shared.load_predictions()
    simulation = shared.load_simulation()
    metrics = shared.load_metrics()
    llm_model = shared.model_picker()

    st.navigation(
        [
            st.Page(
                lambda: overview.body(metrics, simulation, shared.current_assumptions()),
                title="Overview",
                icon=":material/insights:",
                url_path="overview",
                default=True,
            ),
            st.Page(
                lambda: demo.body(predictions, simulation, metrics, llm_model),
                title="Live demo",
                icon=":material/precision_manufacturing:",
                url_path="demo",
            ),
            st.Page(
                lambda: evaluation.body(predictions, metrics),
                title="Evaluation",
                icon=":material/query_stats:",
                url_path="evaluation",
            ),
            st.Page(
                lambda: operations.body(metrics, llm_model),
                title="Operations",
                icon=":material/speed:",
                url_path="operations",
            ),
        ]
    ).run()


main()
