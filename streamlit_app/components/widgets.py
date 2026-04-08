"""
Streamlit Widget Components.

Reusable UI widget helpers that wrap common Streamlit patterns for the IVE UI.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from components.theme import carbon_tag


def api_error_banner(error: str) -> None:
    """Display a standardised API error banner."""
    st.error(f"**API Error:** {error}", icon=":material/error:")


def loading_spinner(message: str = "Loading...") -> Any:
    """Return a st.spinner context manager with a standard message."""
    return st.spinner(message)


def confidence_badge(score: float) -> str:
    """Return a Carbon tag HTML string based on confidence score."""
    if score >= 0.8:
        return carbon_tag("High", "green")
    if score >= 0.6:
        return carbon_tag("Medium", "blue")
    return carbon_tag("Low", "yellow")


def paginator(items: list[Any], page_size: int = 10) -> list[Any]:
    """
    Simple stateful paginator for long lists.

    TODO:
        - Use st.session_state to track current page
        - Render Previous / Next buttons
        - Return the current page's items
    """
    total = len(items)
    n_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)
    start = (page - 1) * page_size
    return items[start : start + page_size]


def experiment_status_chip(status: str) -> None:
    """Display a Carbon-styled status tag for an experiment."""
    color_map = {
        "completed": "green",
        "running": "blue",
        "failed": "red",
        "cancelled": "yellow",
        "queued": "gray",
    }
    color = color_map.get(status.lower(), "gray")
    st.markdown(carbon_tag(status.upper(), color), unsafe_allow_html=True)
