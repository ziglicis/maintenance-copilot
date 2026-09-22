"""Rendering a work order and the audit behind it.

Shared by the Engine detail page, which generates work orders, and Operations, which
replays stored ones. The audit is the point: a badge nobody can interrogate is not
evidence, so every rule that was applied is shown with what it found.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from copilot import theme, workorder

PASS = "✓"
FAIL = "✗"


def render_checks(checks: list[workorder.Check]) -> None:
    """The groundedness result, rule by rule, rather than one coloured pill."""
    passed = [c for c in checks if c.passed]
    if len(passed) == len(checks):
        st.success(f"Groundedness check passed: {len(checks)} of {len(checks)} rules")
    else:
        failed = [c for c in checks if not c.passed]
        st.error(
            f"Groundedness check failed: {len(failed)} of {len(checks)} rules. "
            + "; ".join(f for c in failed for f in c.failures)
        )

    table = pd.DataFrame(
        [{"": PASS if c.passed else FAIL, "rule": c.name, "result": c.detail} for c in checks]
    )
    st.dataframe(
        table.style.map(
            lambda v: f"color: {theme.TIER_COLOURS['Green'] if v == PASS else theme.TIER_COLOURS['Red']}",
            subset=[""],
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_audit(order: workorder.WorkOrder, row: pd.Series, history: pd.DataFrame) -> None:
    """Every claim in evidence beside what the data says."""
    audit = workorder.evidence_audit(order, row, history)
    flags = ["value ok", "trend ok", "range ok"]
    audit[flags] = audit[flags].map(lambda ok: PASS if ok else FAIL)
    st.dataframe(
        audit.style.format({"cited value": "{:.4f}", "actual": "{:.4f}", "delta": "{:+.4f}"}).map(
            lambda v: f"color: {theme.TIER_COLOURS['Green'] if v == PASS else theme.TIER_COLOURS['Red']};"
                      " font-weight: 600",
            subset=flags,
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Cited values come from the model, actual values from the source data at this engine and "
        "cycle. Trend is recomputed over the last 20 cycles and counts as stable when the movement "
        "is smaller than the spread the sensor showed while healthy."
    )


def render_work_order(order: workorder.WorkOrder) -> None:
    """The work order itself, as a technician would read it."""
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


def render_full(
    order: workorder.WorkOrder,
    row: pd.Series,
    history: pd.DataFrame,
    checks: list[workorder.Check],
    footer: str = "",
) -> None:
    """Work order, groundedness rules, and the per-sensor audit behind them."""
    render_checks(checks)
    render_work_order(order)
    if footer:
        st.caption(footer)
    with st.expander("How this was checked"):
        render_audit(order, row, history)
    with st.expander("Raw JSON"):
        st.json(order.model_dump())
