"""One palette for every chart, so the app reads as a single system.

Values come from a validated palette: the three policy colours clear the all-pairs
colour-vision and normal-vision separation floors on this surface. Risk tiers use the
reserved status colours, which are deliberately not part of the series palette. Two of
these sit under 3:1 contrast on a light surface by design, so every chart that uses
them also carries a visible text label, never colour alone.
"""

from __future__ import annotations

import plotly.graph_objects as go

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e8e7e3"

# Categorical series, assigned in fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

# Diverging, for attributions that push the prediction up or down.
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"

# Reserved status colours for risk tiers. Always shipped with the tier name.
TIER_COLOURS = {"Red": "#d03b3b", "Amber": "#fab219", "Green": "#0ca30c"}
TIER_MARK = {"Red": "●", "Amber": "●", "Green": "●"}


def style(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    """Recessive chrome, readable ink, no chartjunk."""
    # Title and legend both live in the top margin, so the margin has to be tall enough
    # for both or they overlap. The title is pinned to the top of the figure itself
    # (yref container) and the legend sits just above the plot area, below it.
    top_margin = 68 if legend else 34
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=top_margin, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK_SECONDARY, size=12),
        title=dict(
            font=dict(color=INK, size=14),
            x=0,
            xanchor="left",
            xref="container",
            y=1,
            yanchor="top",
            yref="container",
            pad=dict(t=8, l=8),
        ),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=INK_SECONDARY)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK_MUTED))
    return fig
