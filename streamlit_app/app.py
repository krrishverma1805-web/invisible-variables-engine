import os

import requests
import streamlit as st
from components.sidebar import render_release_metadata
from components.theme import apply_carbon_theme

st.set_page_config(
    page_title="Invisible Variables Engine",
    page_icon=":material/search:",
    layout="wide",
)
apply_carbon_theme()

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
HEADERS = {"X-API-Key": "dev-key-1"}

# --- Sidebar ---
with st.sidebar:
    st.header("Navigation")
    st.page_link("app.py", label="Home", icon=":material/home:")
    st.page_link("pages/01_upload.py", label="1. Upload Dataset", icon=":material/folder:")
    st.page_link(
        "pages/02_configure.py", label="2. Configure Experiment", icon=":material/settings:"
    )
    st.page_link(
        "pages/03_monitor.py", label="3. Monitor Progress", icon=":material/hourglass_empty:"
    )
    st.page_link("pages/04_results.py", label="4. View Results", icon=":material/bar_chart:")

    render_release_metadata()

# --- Main Page ---
st.title("Invisible Variables Engine")
st.markdown(
    """
    **Discover hidden variables in your data that influence model predictions but aren't directly recorded.**

    The Invisible Variables Engine (IVE) uses advanced machine learning and statistical pattern
    recognition to find systematic errors in predictive models and construct new, meaningful
    features from them.
    """
)

st.divider()

# --- Quick Stats ---
st.subheader("System Overview")

col1, col2 = st.columns(2)

try:
    # Fetch datasets
    ds_resp = requests.get(f"{API_BASE}/api/v1/datasets/", headers=HEADERS, timeout=5)
    if ds_resp.ok:
        ds_data = ds_resp.json()
        n_datasets = ds_data.get("total", len(ds_data.get("datasets", [])))
    else:
        n_datasets = "Error"

    # Fetch experiments
    exp_resp = requests.get(f"{API_BASE}/api/v1/experiments/", headers=HEADERS, timeout=5)
    if exp_resp.ok:
        exp_data = exp_resp.json()
        n_experiments = exp_data.get("total", len(exp_data.get("experiments", [])))
    else:
        n_experiments = "Error"

except requests.RequestException:
    n_datasets = "Offline"
    n_experiments = "Offline"

with col1:
    st.metric("Datasets Uploaded", n_datasets)
with col2:
    st.metric("Experiments Run", n_experiments)

st.divider()
st.markdown("### Get Started")
st.markdown("Head over to the **[Upload Dataset](/01_upload)** page to begin your analysis.")
