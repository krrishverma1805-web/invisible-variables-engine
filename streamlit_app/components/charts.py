"""
Streamlit Chart Components.

Reusable Plotly/Altair chart functions for the IVE results pages.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def shap_bar_chart(feature_importance: dict[str, float], title: str = "Feature Importance") -> None:
    """
    Render a horizontal bar chart of SHAP feature importance.

    Args:
        feature_importance: Dict mapping feature name → mean |SHAP| value.
        title: Chart title.

    TODO:
        - import plotly.express as px
        - Sort by value descending, take top 20
        - Render with px.bar(..., orientation='h')
    """
    if not feature_importance:
        st.info("No feature importance data available.")
        return

    import pandas as pd

    df = pd.DataFrame(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20],
        columns=["Feature", "Importance"],
    )

    # TODO: Replace with plotly horizontal bar chart
    st.bar_chart(df.set_index("Feature")["Importance"])
    st.caption(title)


def residual_histogram(residuals: list[float], title: str = "Residual Distribution") -> None:
    """
    Render a histogram of residual values with normal distribution overlay.

    TODO:
        - import plotly.figure_factory as ff
        - Create distplot with curve_type='normal'
    """
    if not residuals:
        st.info("No residual data available.")
        return

    import pandas as pd

    df = pd.DataFrame({"Residual": residuals})
    st.subheader(title)
    # TODO: Replace with plotly distplot
    st.line_chart(df["Residual"].value_counts().sort_index())


def coverage_gauge(coverage_pct: float, title: str = "Coverage") -> None:
    """Render a gauge chart showing the percentage coverage of a latent variable."""
    # TODO: import plotly.graph_objects as go; render a go.Indicator gauge
    st.metric(title, f"{coverage_pct:.1f}%")


def latent_variable_table(lvs: list[dict[str, Any]]) -> None:
    """
    Render a sortable summary table of latent variables.

    TODO:
        - Build pd.DataFrame from lvs
        - Use st.dataframe with column_config for formatting
    """
    import pandas as pd

    if not lvs:
        st.info("No latent variables to display.")
        return

    df = pd.DataFrame(lvs)[
        ["rank", "name", "confidence_score", "effect_size", "coverage_pct"]
    ].rename(
        columns={
            "rank": "Rank",
            "name": "Name",
            "confidence_score": "Confidence",
            "effect_size": "Effect Size",
            "coverage_pct": "Coverage %",
        }
    )
    st.dataframe(df, use_container_width=True)
