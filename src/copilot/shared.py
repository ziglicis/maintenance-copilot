"""State every page needs: cached artifact loaders and the shared controls.

The loaders are cached once for the whole session, so switching pages re-reads nothing.
Cost assumptions live above the charts they drive rather than in the sidebar, and are
mirrored into session state so pages that show a result without owning the inputs can
still read them.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from copilot import config, simulator


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


# Distinct from the form's own widget key: Streamlit refuses to let you write a
# session key that a live widget already owns.
ASSUMPTIONS_KEY = "cost_assumptions"


def current_assumptions() -> simulator.Assumptions:
    """Whatever the user last submitted, or the config defaults.

    Stored under a plain session key rather than read off the widgets, because
    Streamlit clears widget state for widgets that did not render this run. The
    Overview page shows a savings figure without owning the inputs that drive it, so
    it has to survive navigating away from the page where those inputs live.
    """
    return st.session_state.get(ASSUMPTIONS_KEY, simulator.Assumptions())


def assumption_form() -> simulator.Assumptions:
    """Every number behind the savings figure, editable, and sitting above the charts.

    A form rather than loose inputs: eight number fields would otherwise rerun the
    simulator on every keystroke, so typing 750000 recalculates six times.
    """
    current = current_assumptions()
    with st.form("assumption_form"):
        st.caption("Every number behind the savings figure. Nothing is hidden in the arithmetic.")
        top = st.columns(4)
        cost_unscheduled = top[0].number_input(
            "Unscheduled failure ($)", value=float(current.cost_unscheduled), step=50_000.0
        )
        cost_scheduled = top[1].number_input(
            "Scheduled maintenance ($)", value=float(current.cost_scheduled), step=10_000.0
        )
        downtime_unscheduled = top[2].number_input(
            "Downtime, unscheduled (days)", value=float(current.downtime_unscheduled), step=1.0
        )
        downtime_scheduled = top[3].number_input(
            "Downtime, scheduled (days)", value=float(current.downtime_scheduled), step=1.0
        )

        bottom = st.columns(4)
        cost_per_wasted_cycle = bottom[0].number_input(
            "Value of a wasted cycle ($)", value=float(current.cost_per_wasted_cycle), step=50.0
        )
        fixed_interval = bottom[1].number_input(
            "Fixed interval (cycles)", value=int(current.fixed_interval), step=10
        )
        predictive_threshold = bottom[2].number_input(
            "Pull threshold (cycles)", value=int(current.predictive_threshold), step=5
        )
        parts_lead_cycles = bottom[3].number_input(
            "Parts and slot lead time (cycles)", value=int(current.parts_lead_cycles), step=5
        )
        st.form_submit_button("Recalculate", type="primary")

    assumptions = simulator.Assumptions(
        cost_unscheduled=cost_unscheduled,
        cost_scheduled=cost_scheduled,
        downtime_unscheduled=downtime_unscheduled,
        downtime_scheduled=downtime_scheduled,
        cost_per_wasted_cycle=cost_per_wasted_cycle,
        fixed_interval=fixed_interval,
        predictive_threshold=predictive_threshold,
        parts_lead_cycles=parts_lead_cycles,
    )
    st.session_state[ASSUMPTIONS_KEY] = assumptions
    return assumptions



def model_picker() -> str:
    """Which model writes the work orders. Lives in the sidebar so every page agrees."""
    st.sidebar.header("Work order generator")
    return st.sidebar.selectbox("Model", options=config.LLM_MODELS, index=0)
