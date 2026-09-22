"""What each maintenance policy costs, and what the assumptions behind that are."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from copilot import charts, shared, simulator, theme


def body(simulation: pd.DataFrame) -> None:
    assumptions = shared.assumption_form()

    results = simulator.simulate(simulation, assumptions)
    curve = simulator.sensitivity(simulation, assumptions)

    baseline = results.loc["Fixed interval", "total_cost"]
    predictive = results.loc["Predictive", "total_cost"]
    cols = st.columns(3)
    cols[0].metric("Predictive total", f"${predictive / 1e6:.1f}M")
    cols[1].metric(
        "Against fixed interval",
        f"${(baseline - predictive) / 1e6:.1f}M",
        delta=f"{(predictive - baseline) / baseline:.0%}",
        delta_color="inverse",
    )
    cols[2].metric(
        "Unscheduled failures avoided",
        int(results.loc["Fixed interval", "unscheduled_failures"] - results.loc["Predictive", "unscheduled_failures"]),
    )

    left, right = st.columns(2)
    left.plotly_chart(charts.policy_chart(results), use_container_width=True, config=theme.PLOTLY_CONFIG)
    right.plotly_chart(charts.sensitivity_chart(curve, assumptions.predictive_threshold), use_container_width=True, config=theme.PLOTLY_CONFIG)

    st.dataframe(
        results.assign(
            total_cost=results["total_cost"].map("${:,.0f}".format),
            downtime_days=results["downtime_days"].map("{:,.0f}".format),
            wasted_cycles=results["wasted_cycles"].map("{:,.0f}".format),
        )
    )
    st.caption(
        f"Replays {simulation['unit'].nunique()} engines with complete run-to-failure trajectories, using "
        "out-of-fold predictions so no engine is scored by a model that trained on it. The truncated test "
        "engines cannot be used here: most stop recording while still healthy, long before any pull decision."
    )
