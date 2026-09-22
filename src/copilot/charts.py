"""Every chart in the app, in one place.

Kept apart from app.py so the layout reads as layout and the chart definitions read as
chart definitions. Colours come from theme.py, never from a literal here.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from copilot import config, data, theme


def rul_histogram(snapshot: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for tier in ("Red", "Amber", "Green"):
        subset = snapshot[snapshot["tier"] == tier]
        fig.add_histogram(
            x=subset["point"],
            name=f"{tier} ({len(subset)})",
            marker_color=theme.TIER_COLOURS[tier],
            xbins=dict(size=10),
            hovertemplate="%{y} engines near %{x:.0f} cycles<extra></extra>",
        )
    fig.update_layout(barmode="stack", bargap=0.08, title="Predicted RUL across the fleet")
    fig.update_xaxes(title="predicted remaining cycles")
    return theme.style(fig, legend=True)


def sensor_trend(history: pd.DataFrame, sensor: str, current_cycle: int) -> go.Figure:
    """One sensor over the engine's life, with its own healthy band shaded."""
    healthy = history[sensor].iloc[:20]
    fig = go.Figure()
    fig.add_hrect(
        y0=healthy.min(), y1=healthy.max(), fillcolor=theme.GRID, opacity=0.9, line_width=0, layer="below"
    )
    fig.add_scatter(
        x=history["cycle"],
        y=history[sensor],
        mode="lines",
        line=dict(color=theme.SERIES[0], width=2),
        name=sensor,
        hovertemplate="cycle %{x}: %{y:.3f}<extra></extra>",
    )
    fig.add_vline(x=current_cycle, line=dict(color=theme.INK_MUTED, width=1, dash="dot"))
    fig.update_layout(title=f"{sensor}: {data.SENSOR_DESCRIPTIONS[sensor]}")
    fig.update_xaxes(title="cycle")
    return theme.style(fig, height=240)


def prediction_history(history: pd.DataFrame, current_cycle: int) -> go.Figure:
    shown = history[history["cycle"] <= current_cycle]
    fig = go.Figure()
    fig.add_scatter(
        x=list(shown["cycle"]) + list(shown["cycle"])[::-1],
        y=list(shown["upper"]) + list(shown["lower"])[::-1],
        fill="toself",
        fillcolor="rgba(42,120,214,0.12)",
        line=dict(width=0),
        name="prediction interval",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=shown["cycle"], y=shown["point"], mode="lines",
        line=dict(color=theme.SERIES[0], width=2), name="predicted RUL",
    )
    fig.add_scatter(
        x=shown["cycle"], y=shown["true_rul"], mode="lines",
        line=dict(color=theme.SERIES[1], width=2, dash="dash"), name="true RUL",
    )
    fig.add_hline(y=config.TIER_RED, line=dict(color=theme.TIER_COLOURS["Red"], width=1, dash="dot"))
    fig.update_layout(title="Prediction against truth, cycle by cycle")
    fig.update_xaxes(title="cycle")
    fig.update_yaxes(title="remaining cycles")
    return theme.style(fig, height=300, legend=True)


def attribution_chart(row: pd.Series) -> go.Figure:
    names = [row[f"driver{i}"] for i in (1, 2, 3)]
    values = [row[f"driver{i}_contrib"] for i in (1, 2, 3)]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[theme.POSITIVE if v >= 0 else theme.NEGATIVE for v in values],
            text=[f"{v:+.1f}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.1f} cycles<extra></extra>",
        )
    )
    fig.update_layout(title="What moved this prediction (cycles)")
    fig.update_yaxes(autorange="reversed")
    # Symmetric range around zero, with headroom so the outside value labels land
    # inside the plot rather than on top of the axis labels. Symmetry is also the
    # honest framing: it shows at a glance whether the drivers push life up or down.
    span = max(abs(v) for v in values) or 1.0
    fig.update_xaxes(range=[-span * 1.4, span * 1.4], zeroline=True, zerolinecolor=theme.GRID)
    return theme.style(fig, height=240)


def policy_chart(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=results.index,
            y=results["total_cost"],
            marker_color=theme.SERIES,
            text=[f"${v/1e6:.1f}M" for v in results["total_cost"]],
            textposition="outside",
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title="Total cost over the simulated fleet")
    # Headroom for the outside value labels, which otherwise clip on the tallest bar.
    fig.update_yaxes(title="dollars", range=[0, results["total_cost"].max() * 1.18])
    return theme.style(fig)


def sensitivity_chart(curve: pd.DataFrame, chosen: int) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=curve.index, y=curve["total_cost"], mode="lines+markers",
        line=dict(color=theme.SERIES[0], width=2), marker=dict(size=8),
        hovertemplate="threshold %{x}: $%{y:,.0f}<extra></extra>", name="predictive",
    )
    fig.add_vline(x=chosen, line=dict(color=theme.INK_MUTED, width=1, dash="dot"))
    best = curve["total_cost"].idxmin()
    fig.add_annotation(
        x=best, y=curve["total_cost"].min(), text=f"cheapest at {best}",
        showarrow=True, arrowhead=0, ax=0, ay=-28, font=dict(color=theme.INK_SECONDARY),
    )
    fig.update_layout(title="Cost against pull threshold")
    fig.update_xaxes(title="pull when predicted lower bound drops below (cycles)")
    # Room above the curve for the annotation pointing at the cheapest threshold.
    fig.update_yaxes(
        title="dollars",
        range=[curve["total_cost"].min() * 0.92, curve["total_cost"].max() * 1.12],
    )
    return theme.style(fig)


def accuracy_scatter(snapshot: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=snapshot["true_rul"], y=snapshot["point"], mode="markers",
        marker=dict(size=9, color=theme.SERIES[0], line=dict(width=2, color=theme.SURFACE)),
        hovertemplate="engine %{customdata}: true %{x}, predicted %{y:.0f}<extra></extra>",
        customdata=snapshot.index, name="engines",
    )
    limit = float(max(snapshot["true_rul"].max(), snapshot["point"].max()))
    fig.add_scatter(x=[0, limit], y=[0, limit], mode="lines",
                    line=dict(color=theme.INK_MUTED, width=1, dash="dot"), name="perfect")
    fig.update_layout(title="Predicted against true RUL, final cycle")
    fig.update_xaxes(title="true RUL")
    fig.update_yaxes(title="predicted RUL")
    return theme.style(fig, height=340)


def lead_time_histogram(predictions: pd.DataFrame) -> go.Figure:
    """How much warning each engine got, measured at its first Red flag."""
    lead = predictions[predictions["tier"] == "Red"].groupby("unit")["true_rul"].first()
    fig = go.Figure(
        go.Histogram(x=lead, marker_color=theme.SERIES[0], xbins=dict(size=10),
                     hovertemplate="%{y} engines at %{x} cycles lead<extra></extra>")
    )
    fig.add_vline(
        x=config.LEAD_TARGET_CYCLES, line=dict(color=theme.TIER_COLOURS["Red"], width=1, dash="dot")
    )
    fig.update_layout(title="Warning lead time at first Red flag")
    fig.update_xaxes(title="true cycles remaining when first flagged")
    return theme.style(fig, height=340)
