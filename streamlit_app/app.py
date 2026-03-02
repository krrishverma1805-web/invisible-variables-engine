"""
IVE Streamlit Application — Entry Point.

This is the main Streamlit app file. It configures the app's global
settings, navigation, and shared API client used by all pages.

Run locally:
    streamlit run streamlit_app/app.py --server.port 8501
"""

from __future__ import annotations

import os

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Invisible Variables Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/invisible-variables-engine",
        "Report a bug": "https://github.com/yourusername/invisible-variables-engine/issues",
        "About": (
            "**Invisible Variables Engine** — Discovers hidden latent variables "
            "in your datasets by analysing systematic model prediction errors."
        ),
    },
)

# ---------------------------------------------------------------------------
# Shared API configuration (stored in session state)
# ---------------------------------------------------------------------------
if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = os.getenv("API_BASE_URL", "http://localhost:8000")

if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.getenv("IVE_API_KEY", "")

# ---------------------------------------------------------------------------
# Sidebar — global navigation and settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=IVE", use_column_width=True)
    st.markdown("---")
    st.markdown("### ⚙️ API Settings")

    api_url = st.text_input(
        "API Base URL",
        value=st.session_state["api_base_url"],
        help="URL of the IVE backend API",
    )
    api_key = st.text_input(
        "API Key",
        value=st.session_state["api_key"],
        type="password",
        help="Your X-API-Key header value",
    )

    if st.button("💾 Save Settings"):
        st.session_state["api_base_url"] = api_url
        st.session_state["api_key"] = api_key
        st.success("Settings saved!")

    st.markdown("---")
    st.markdown(
        "Pages:\n"
        "1. 📂 Upload Dataset\n"
        "2. ⚙️ Configure Experiment\n"
        "3. 📊 Monitor Progress\n"
        "4. 🔍 View Results"
    )

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------
st.title("🔍 Invisible Variables Engine")
st.markdown(
    """
    Welcome to the **Invisible Variables Engine** — a data science system that
    discovers hidden latent variables in your datasets by analysing systematic
    model prediction errors.

    ### How it works
    1. **Upload** your dataset (CSV or Parquet)
    2. **Configure** an experiment with your model preferences
    3. **Monitor** the four-phase analysis pipeline in real time
    4. **Review** the discovered latent variables with explanations

    ### Get started
    Use the sidebar to navigate between pages, or click one of the shortcuts below.
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    st.info("📂 **Upload a Dataset**\nStart by uploading your CSV or Parquet file.")
with col2:
    st.info("⚙️ **Configure & Run**\nSet up your experiment parameters.")
with col3:
    st.info("🔍 **View Results**\nExplore discovered latent variables.")

# ---------------------------------------------------------------------------
# API health check
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("API Status")

# TODO: Call GET /api/v1/health and display result
# import httpx
# try:
#     resp = httpx.get(f"{st.session_state['api_base_url']}/api/v1/health", timeout=3)
#     if resp.status_code == 200:
#         st.success("✅ API is reachable")
#     else:
#         st.error(f"❌ API returned {resp.status_code}")
# except Exception as e:
#     st.error(f"❌ Cannot reach API: {e}")

st.warning("⚠️ API health check not yet implemented — configure API settings in the sidebar.")
