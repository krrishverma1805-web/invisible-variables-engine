"""
IBM Carbon Design System Theme for Streamlit.

Provides CSS injection and helper utilities to apply the Carbon Design System
visual language across the IVE Streamlit application.

Heavy lifting (fonts, base colors, light/dark mode, sidebar theming,
border-radius) is handled by ``.streamlit/config.toml``.  This module adds
the component-level overrides that config.toml cannot express.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Plotly / chart layout constant (Carbon-aligned)
# ---------------------------------------------------------------------------
CARBON_CHART_LAYOUT: dict[str, object] = {
    "font": {
        "family": "IBM Plex Sans, Helvetica Neue, Arial, sans-serif",
        "color": "#161616",
    },
    "plot_bgcolor": "#ffffff",
    "paper_bgcolor": "#ffffff",
    "colorway": [
        "#0f62fe", "#da1e28", "#24a148", "#f1c21b",
        "#525252", "#0043ce", "#393939",
    ],
}

# ---------------------------------------------------------------------------
# CSS — supplements config.toml with things it can't express
#
# IMPORTANT: config.toml handles fonts, base colors, light/dark, sidebar
# background/text, border-radius (baseRadius=0), and status colors.
# This CSS ONLY adds: typography weight scale, bottom-border inputs,
# button sizing, component polish, and custom Carbon tag classes.
#
# CRITICAL: Do NOT use wildcard selectors like `[data-testid="stSidebar"] *`
# as they break Material icon rendering.
# ---------------------------------------------------------------------------
_CARBON_CSS = """
<style>
/* =================================================================
   A: Typography Weight Scale
   Carbon: h1=300 Light, h2=400 Regular, h3=600 Semibold
   ================================================================= */
h1 {
    font-weight: 300 !important;
    line-height: 1.19 !important;
    letter-spacing: 0 !important;
}

h2 {
    font-weight: 400 !important;
    line-height: 1.25 !important;
}

h3 {
    font-weight: 600 !important;
    line-height: 1.40 !important;
}

/* Caption / small — Carbon Caption 01: 12px, 0.32px tracking */
small {
    letter-spacing: 0.32px !important;
}

/* Code — letter-spacing 0.16px */
code, pre {
    letter-spacing: 0.16px;
}

/* =================================================================
   B: Buttons — 48px height, Carbon padding
   (border-radius=0 handled by config.toml buttonRadius=0)
   ================================================================= */
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    min-height: 48px !important;
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.16px !important;
    padding: 14px 24px !important;
    border: 1px solid transparent !important;
    transition: background-color 0.15s ease !important;
}

/* Download buttons — Carbon secondary style (Gray 80) */
.stDownloadButton > button {
    background-color: #393939 !important;
    color: #ffffff !important;
}
.stDownloadButton > button:hover {
    background-color: #4c4c4c !important;
    color: #ffffff !important;
}

/* =================================================================
   C: Inputs — bottom-border only (Carbon signature)
   (showWidgetBorder=false removes the box, we add bottom-border)
   ================================================================= */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    border: none !important;
    border-bottom: 2px solid var(--border-color, #c6c6c6) !important;
    min-height: 40px;
    padding: 0 16px !important;
    transition: border-color 0.15s ease !important;
}

.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    border-bottom-color: var(--primary-color, #0f62fe) !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Selectbox — bottom-border */
.stSelectbox > div > div {
    border: none !important;
    border-bottom: 2px solid var(--border-color, #c6c6c6) !important;
    min-height: 40px;
}

.stSelectbox > div > div:focus-within {
    border-bottom-color: var(--primary-color, #0f62fe) !important;
    box-shadow: none !important;
}

/* Input labels — Carbon Caption 01 */
.stTextInput label,
.stNumberInput label,
.stTextArea label,
.stSelectbox label,
.stMultiSelect label,
.stFileUploader label,
.stSlider label,
.stCheckbox label,
.stRadio label {
    font-size: 0.75rem !important;
    letter-spacing: 0.32px !important;
    font-weight: 400 !important;
}

/* =================================================================
   D: File Uploader — dashed border
   ================================================================= */
.stFileUploader section {
    border: 1px dashed var(--border-color, #c6c6c6) !important;
}

/* =================================================================
   E: Metrics — card padding
   ================================================================= */
.stMetric {
    padding: 16px !important;
}

.stMetric label {
    font-size: 0.75rem !important;
    letter-spacing: 0.32px !important;
}

/* =================================================================
   F: Progress Bar — flat
   ================================================================= */
.stProgress > div,
.stProgress > div > div {
    border-radius: 0px !important;
}

/* =================================================================
   G: Expanders — flat, no shadow
   ================================================================= */
.stExpander {
    box-shadow: none !important;
}

.stExpander summary {
    font-weight: 600 !important;
    font-size: 0.875rem !important;
}

/* =================================================================
   H: Sidebar — subtle tweaks only (bg/text from config.toml)
   DO NOT use wildcard * selectors — they break icon rendering
   ================================================================= */
section[data-testid="stSidebar"] hr {
    border-color: #393939 !important;
}

/* Sidebar heading color */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* =================================================================
   I: Hide Streamlit auto-generated page nav (we use custom st.page_link)
   ================================================================= */
section[data-testid="stSidebar"] ul[data-testid="stSidebarNavItems"] {
    display: none !important;
}

/* =================================================================
   I2: Dividers — opacity fix
   ================================================================= */
hr {
    opacity: 1 !important;
}

/* =================================================================
   J: Custom CSS Classes — Carbon tags, status dots
   ================================================================= */

/* Carbon pill tags (24px radius — only exception to 0px rule) */
.carbon-tag {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 24px;
    font-size: 12px;
    font-weight: 400;
    font-family: 'IBM Plex Sans', sans-serif;
    line-height: 1.33;
    letter-spacing: 0.32px;
    white-space: nowrap;
    vertical-align: middle;
}
.carbon-tag--blue   { background: #edf5ff; color: #0f62fe; }
.carbon-tag--red    { background: #fff1f1; color: #da1e28; }
.carbon-tag--green  { background: #defbe6; color: #24a148; }
.carbon-tag--yellow { background: #fdf6dd; color: #161616; }
.carbon-tag--gray   { background: #f4f4f4; color: #525252; }

/* Carbon status dots */
.carbon-status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}
.carbon-status-dot--ok      { background: #24a148; }
.carbon-status-dot--error   { background: #da1e28; }
.carbon-status-dot--warning { background: #f1c21b; }

/* Status row for sidebar */
.carbon-status-row {
    display: flex;
    align-items: center;
    padding: 6px 0;
    font-size: 14px;
    font-family: 'IBM Plex Sans', sans-serif;
    line-height: 1.29;
    letter-spacing: 0.16px;
}

/* =================================================================
   K: Dark-mode tag overrides
   ================================================================= */
@media (prefers-color-scheme: dark) {
    .carbon-tag--blue   { background: #001d6c; color: #78a9ff; }
    .carbon-tag--red    { background: #2d0709; color: #ff8389; }
    .carbon-tag--green  { background: #071908; color: #42be65; }
    .carbon-tag--yellow { background: #281e00; color: #f1c21b; }
    .carbon-tag--gray   { background: #262626; color: #c6c6c6; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_carbon_theme() -> None:
    """Inject IBM Carbon Design System CSS into the current Streamlit page.

    Call this immediately after ``st.set_page_config()`` on every page.
    """
    st.markdown(_CARBON_CSS, unsafe_allow_html=True)


def carbon_tag(text: str, color: str = "blue") -> str:
    """Return HTML for a Carbon pill-shaped tag.

    Args:
        text: Label text to display inside the tag.
        color: One of ``blue``, ``red``, ``green``, ``yellow``, ``gray``.
    """
    from html import escape

    safe_text = escape(text)
    safe_color = escape(color)
    return f'<span class="carbon-tag carbon-tag--{safe_color}">{safe_text}</span>'


def carbon_status_dot(status: str) -> str:
    """Return HTML for a small status-indicator dot.

    Args:
        status: One of ``ok``, ``error``, ``warning``.
    """
    from html import escape

    safe_status = escape(status)
    return f'<span class="carbon-status-dot carbon-status-dot--{safe_status}"></span>'
