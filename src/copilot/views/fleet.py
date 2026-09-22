"""Triage: which engines need attention, in what order."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from copilot import charts, shared, theme


def body(predictions: pd.DataFrame) -> None:
    snapshot = shared.fleet_snapshot(predictions)

    counts = snapshot["tier"].value_counts()
    cols = st.columns(4)
    for col, tier in zip(cols, ("Red", "Amber", "Green")):
        col.metric(f"{theme.TIER_MARK[tier]} {tier}", int(counts.get(tier, 0)))
    cols[3].metric("Engines", len(snapshot))

    # True RUL is the answer key. A planner would never have it, so it is off by
    # default: leaving it on makes an operational view look like an evaluation view.
    show_truth = st.toggle(
        "Show ground truth",
        value=False,
        key="show_ground_truth",
        help="Reveal each engine's actual remaining life. Available here only because this "
        "is a held-out test set, never in service.",
    )

    table = pd.DataFrame(
        {
            "cycle": snapshot["cycle"],
            "predicted RUL": snapshot["point"],
            "interval": [f"{lo:.0f} to {hi:.0f}" for lo, hi in zip(snapshot["lower"], snapshot["upper"])],
            "tier": snapshot["tier"],
            "top driver": snapshot["driver1"],
        }
    ).sort_values("predicted RUL")
    formats = {"predicted RUL": "{:.1f}"}
    if show_truth:
        table["true RUL"] = snapshot["true_rul"]
        table["error"] = snapshot["point"] - snapshot["true_rul"]
        formats |= {"true RUL": "{:.0f}", "error": "{:+.1f}"}

    left, right = st.columns([3, 2])
    left.dataframe(
        table.style
        .format(formats)
        .map(lambda t: f"color: {theme.TIER_COLOURS[t]}; font-weight: 600", subset=["tier"]),
        height=420,
    )
    right.plotly_chart(charts.rul_histogram(snapshot), use_container_width=True)
    st.caption(
        "Sorted by predicted remaining life, so the engines needing a decision are at the top. "
        "Open one on the Engine detail page."
    )
