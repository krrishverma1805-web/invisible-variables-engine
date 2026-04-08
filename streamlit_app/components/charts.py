"""
Streamlit Chart Components.

Reusable Plotly chart functions for the IVE results pages,
styled with the IBM Carbon Design System theme.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from components.theme import CARBON_CHART_LAYOUT


def shap_bar_chart(feature_importance: dict[str, float], title: str = "Feature Importance") -> None:
    """
    Render a horizontal bar chart of SHAP feature importance.

    Args:
        feature_importance: Dict mapping feature name to mean |SHAP| value.
        title: Chart title.
    """
    if not feature_importance:
        st.info("No feature importance data available.")
        return

    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20],
        columns=["Feature", "Importance"],
    )

    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=title,
    )
    fig.update_layout(
        **CARBON_CHART_LAYOUT,
        yaxis=dict(autorange="reversed"),  # highest importance at top
        height=max(400, len(df) * 28),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    fig.update_traces(marker_color="#0f62fe")  # IBM Blue 60
    st.plotly_chart(fig, use_container_width=True)


def residual_histogram(residuals: list[float], title: str = "Residual Distribution") -> None:
    """
    Render a histogram of residual values with normal distribution overlay.

    Args:
        residuals: List of residual values.
        title: Chart title.
    """
    if not residuals:
        st.info("No residual data available.")
        return

    import numpy as np
    import plotly.graph_objects as go

    residuals_arr = np.array(residuals)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals_arr,
        nbinsx=50,
        name="Residuals",
        marker_color="#0f62fe",
        opacity=0.7,
        histnorm="probability density",
    ))

    # Normal distribution overlay
    x_range = np.linspace(residuals_arr.min(), residuals_arr.max(), 200)
    mu, sigma = residuals_arr.mean(), residuals_arr.std()
    if sigma > 0:
        try:
            from scipy.stats import norm

            normal_curve = norm.pdf(x_range, mu, sigma)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_curve,
                mode="lines",
                name="Normal fit",
                line=dict(color="#da1e28", width=2),
            ))
        except ImportError:
            pass  # scipy not available; skip normal curve overlay

    fig.update_layout(
        **CARBON_CHART_LAYOUT,
        title=title,
        xaxis_title="Residual Value",
        yaxis_title="Density",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)


def coverage_gauge(coverage_pct: float, title: str = "Coverage") -> None:
    """Render a gauge chart showing the percentage coverage of a latent variable."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=coverage_pct,
        title={"text": title, "font": {"family": "IBM Plex Sans", "size": 16}},
        number={"suffix": "%", "font": {"family": "IBM Plex Sans", "size": 32}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#0f62fe"},
            "bgcolor": "#f4f4f4",
            "steps": [
                {"range": [0, 30], "color": "#fff1f1"},    # Red 10
                {"range": [30, 70], "color": "#fdf6dd"},   # Yellow 10
                {"range": [70, 100], "color": "#defbe6"},  # Green 10
            ],
            "threshold": {
                "line": {"color": "#161616", "width": 2},
                "thickness": 0.75,
                "value": coverage_pct,
            },
        },
    ))
    fig.update_layout(
        font={"family": "IBM Plex Sans, sans-serif", "color": "#161616"},
        paper_bgcolor="#ffffff",
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def latent_variable_table(lvs: list[dict[str, Any]]) -> None:
    """Render a sortable summary table of latent variables."""
    import pandas as pd

    if not lvs:
        st.info("No latent variables to display.")
        return

    # Build DataFrame, handling missing columns gracefully
    df = pd.DataFrame(lvs)

    # Select and rename columns that exist
    col_map = {
        "rank": "Rank",
        "name": "Name",
        "confidence_score": "Confidence",
        "effect_size": "Effect Size",
        "coverage_pct": "Coverage %",
        "status": "Status",
        "stability_score": "Stability",
    }
    display_cols = [k for k in col_map if k in df.columns]
    df = df[display_cols].rename(columns=col_map)

    # Column config for formatting
    column_config: dict[str, Any] = {}
    if "Confidence" in df.columns:
        column_config["Confidence"] = st.column_config.ProgressColumn(
            "Confidence", min_value=0, max_value=1, format="%.2f"
        )
    if "Stability" in df.columns:
        column_config["Stability"] = st.column_config.ProgressColumn(
            "Stability", min_value=0, max_value=1, format="%.2f"
        )

    st.dataframe(df, use_container_width=True, hide_index=True, column_config=column_config)
