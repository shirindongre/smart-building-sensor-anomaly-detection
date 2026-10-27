"""Plotly figures for the demo (layout only; all numbers come from scripts/demo_core.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# One colour per meaning, used on every page.
ENERGY = "#C9CED6"  # light grey: energy and error lines
THRESHOLD = "#FF5A5A"  # bright red: threshold and drift gate
FLAG = "#5AB0FF"  # bright blue accent: flagged points and spans
INJECTED = "#FBBF24"  # bright amber: injected span
DRIFT = "#8A919B"  # mid grey: drift periods
GRID = "#4A4F57"
INK = "#ECEEF1"
SURFACE = "#363A41"

ENERGY_LABEL = "Energy (raw meter units per hour)"
ERROR_LABEL = "Reconstruction error (MSE)"
NORM_LABEL = "Normalized error (robust z-score)"


def _legend_swatch(fig, name, color, symbol="square"):
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, showlegend=True,
                             marker=dict(symbol=symbol, size=12, color=color), hoverinfo="skip"), row=1, col=1)


def _runs(mask: np.ndarray):
    padded = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(padded)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def timeline_figure(res: pd.DataFrame, *, normalized: bool = True, spans=None, ref_energy: pd.Series | None = None,
                    x_range=None, split_time=None, height: int | None = None, y_max: dict | None = None,
                    drift_gate: float = 5.0, split_label: str | None = None) -> go.Figure:
    """
    Stacked, linked panels: energy / reconstruction error with threshold / (optionally) normalized error.
    Flagged hours are blue markers, drift periods grey bands, ``spans`` (injected) amber bands.
    ``ref_energy`` draws the clean energy faintly behind the corrupted one.
    """
    rows = 3 if normalized else 2
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.4, 0.35, 0.25] if normalized else [0.55, 0.45])
    flagged = res[res["anomaly_flag"] == 1]
    x = res.index

    if ref_energy is not None:
        fig.add_trace(go.Scatter(x=x, y=ref_energy.reindex(x), mode="lines", name="Clean energy",
                                 line=dict(color="#8A919B", width=1.2, dash="dot"),
                                 hovertemplate="Clean: %{y:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=res["energy"], mode="lines", name="Energy", line=dict(color=ENERGY, width=1.3),
                             hovertemplate="Energy: %{y:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=flagged.index, y=flagged["energy"], mode="markers", name="Flagged hour",
                             marker=dict(color=FLAG, size=8, line=dict(color=SURFACE, width=1.5)),
                             hovertemplate="Flagged, energy: %{y:.1f}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=res["reconstruction_error"], mode="lines", name="Reconstruction error",
                             line=dict(color=ENERGY, width=1.3),
                             hovertemplate="Error: %{y:.4f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=res["threshold"], mode="lines", name="Threshold",
                             line=dict(color=THRESHOLD, width=2),
                             hovertemplate="Threshold: %{y:.4f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=flagged.index, y=flagged["reconstruction_error"], mode="markers", name="Flagged hour",
                             showlegend=False, marker=dict(color=FLAG, size=8, line=dict(color=SURFACE, width=1.5)),
                             hovertemplate="Flagged, error: %{y:.4f}<extra></extra>"), row=2, col=1)

    if normalized:
        fig.add_trace(go.Scatter(x=x, y=res["normalized_error"], mode="lines", name="Normalized error",
                                 line=dict(color=ENERGY, width=1.3),
                                 hovertemplate="Normalized: %{y:.1f}<extra></extra>"), row=3, col=1)
        fig.add_trace(go.Scatter(x=flagged.index, y=flagged["normalized_error"], mode="markers", name="Flagged hour",
                                 showlegend=False, marker=dict(color=FLAG, size=8, line=dict(color=SURFACE, width=1.5)),
                                 hovertemplate="Flagged, normalized: %{y:.1f}<extra></extra>"), row=3, col=1)
        fig.add_hline(y=drift_gate, line=dict(color=THRESHOLD, width=1.5, dash="dash"), row=3, col=1)

    drift = _runs(res["drift_flag"].to_numpy() == 1)
    for a, b in drift:
        fig.add_vrect(x0=x[a], x1=x[b], fillcolor=DRIFT, opacity=0.55, line_width=0, layer="below", row="all", col=1)
    if drift:
        _legend_swatch(fig, "Drift period", DRIFT)
    for t0, t1 in spans or []:
        fig.add_vrect(x0=t0, x1=t1 + pd.Timedelta(hours=1), fillcolor=INJECTED, opacity=0.28, line_width=0,
                      layer="below", row="all", col=1)
    if spans:
        _legend_swatch(fig, "Injected span", INJECTED)
    if split_time is not None:
        fig.add_vline(x=split_time, line=dict(color=INK, width=1.2, dash="dot"), row="all", col=1)
        fig.add_annotation(x=split_time, y=1, yref="y domain", xref="x", text=split_label or "Train / test split",
                           showarrow=False, xanchor="left", yanchor="top", xshift=6, font=dict(size=12, color=INK),
                           row=1, col=1)

    fig.update_yaxes(title_text=ENERGY_LABEL, row=1, col=1)
    fig.update_yaxes(title_text=ERROR_LABEL, row=2, col=1)
    if normalized:
        fig.update_yaxes(title_text=NORM_LABEL, row=3, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=rows, col=1)
    for r in range(1, rows + 1):
        fig.update_xaxes(showgrid=True, gridcolor=GRID, row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, row=r, col=1)
    if y_max:
        for r, v in y_max.items():
            fig.update_yaxes(range=[0, v], row=r, col=1)
    if x_range is not None:
        fig.update_xaxes(range=[x_range[0], x_range[1]])
    fig.update_layout(
        template="plotly_dark", height=height or (820 if normalized else 640), hovermode="x unified",
        font=dict(size=14, color=INK), margin=dict(l=70, r=20, t=40, b=60), dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=14)),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
    )
    return fig


def recall_bar_figure(long: pd.DataFrame, types: list[str]) -> go.Figure:
    """Grouped bars: recall by event type, one group per building. Four categorical colours (Okabe-Ito)."""
    palette = ["#56B4E9", "#E69F00", "#2FCB9B", "#E58FC1"]
    fig = go.Figure()
    for t, col in zip(types, palette):
        d = long[long["type"] == t]
        fig.add_trace(go.Bar(x=d["building"], y=d["recall"], name=t.replace("_", " "), marker=dict(color=col),
                             text=[f"{v:.0%}" for v in d["recall"]], textposition="outside",
                             hovertemplate="%{x}<br>" + t.replace("_", " ") + " recall: %{y:.0%}<extra></extra>"))
    fig.update_layout(template="plotly_dark", barmode="group", height=460, font=dict(size=14, color=INK),
                      yaxis=dict(title="Recall (share of scored events found)", tickformat=".0%", range=[0, 1.15],
                                 gridcolor=GRID),
                      xaxis=dict(title="Building"), legend=dict(orientation="h", y=1.1, x=0),
                      margin=dict(l=70, r=20, t=40, b=60), bargap=0.25, bargroupgap=0.08,
                      plot_bgcolor=SURFACE, paper_bgcolor=SURFACE)
    return fig
