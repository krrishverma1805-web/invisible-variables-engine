import os

import requests
import streamlit as st
from components.sidebar import render_sidebar
from components.theme import apply_carbon_theme, carbon_tag

st.set_page_config(
    page_title="Compare Experiments - IVE",
    page_icon=":material/compare_arrows:",
    layout="wide",
)
apply_carbon_theme()
render_sidebar()

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
HEADERS = {"X-API-Key": "dev-key-1"}

st.title("Compare Experiments")
st.markdown("Compare two experiments side-by-side to analyze configuration impact and pattern overlap.")

# --- Fetch completed experiments ---
experiments_dict: dict[str, str] = {}
try:
    response = requests.get(f"{API_BASE}/api/v1/experiments/", headers=HEADERS, timeout=5)
    if response.ok:
        data = response.json()
        for exp in data.get("experiments", []):
            if exp.get("status") == "completed":
                config = exp.get("config_json", {})
                mode = str(config.get("analysis_mode", "demo")).capitalize()
                label = f"{exp['id'][:8]}… — [{mode}]"
                experiments_dict[label] = exp["id"]
except requests.RequestException:
    st.error("Could not connect to the API.")

if len(experiments_dict) < 2:
    st.info("At least 2 completed experiments are needed for comparison.")
    st.stop()

# --- Experiment selectors ---
col1, col2 = st.columns(2)
labels = list(experiments_dict.keys())

with col1:
    exp_a_label = st.selectbox("Experiment A", labels, index=0, key="cmp_a")
with col2:
    default_b = min(1, len(labels) - 1)
    exp_b_label = st.selectbox("Experiment B", labels, index=default_b, key="cmp_b")

exp_a_id = experiments_dict[exp_a_label]
exp_b_id = experiments_dict[exp_b_label]

if exp_a_id == exp_b_id:
    st.warning("Please select two different experiments to compare.")
    st.stop()

st.divider()

# --- Fetch comparison ---
with st.spinner("Comparing experiments..."):
    try:
        resp = requests.get(
            f"{API_BASE}/api/v1/experiments/compare",
            headers=HEADERS,
            params={"ids": f"{exp_a_id},{exp_b_id}"},
            timeout=10,
        )
        if not resp.ok:
            st.error(f"Comparison failed: {resp.text}")
            st.stop()
        comparison = resp.json()
    except requests.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        st.stop()

experiments = comparison.get("experiments", [])
if len(experiments) < 2:
    st.error("Could not fetch both experiments.")
    st.stop()

exp_a = experiments[0]
exp_b = experiments[1]

# --- Summary metrics ---
st.subheader("Overview")
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Patterns (A)", exp_a.get("n_patterns", 0))
with metric_cols[1]:
    st.metric("Validated LVs (A)", exp_a.get("n_validated", 0))
with metric_cols[2]:
    st.metric("Patterns (B)", exp_b.get("n_patterns", 0))
with metric_cols[3]:
    st.metric("Validated LVs (B)", exp_b.get("n_validated", 0))

st.divider()

# --- Configuration Diff ---
config_diff = comparison.get("config_diff", {})
if config_diff:
    st.subheader("Configuration Differences")
    diff_rows = []
    for key, vals in config_diff.items():
        diff_rows.append({
            "Parameter": key,
            "Experiment A": str(vals.get("experiment_1", "—")),
            "Experiment B": str(vals.get("experiment_2", "—")),
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
else:
    st.info("Both experiments used identical configurations.")

st.divider()

# --- Latent Variable Overlap ---
st.subheader("Latent Variable Overlap")
overlap = comparison.get("latent_variable_overlap", {})
common = overlap.get("common", [])
unique_a = overlap.get("unique_to_first", [])
unique_b = overlap.get("unique_to_second", [])

ov_col1, ov_col2, ov_col3 = st.columns(3)
with ov_col1:
    st.markdown("**Found in Both**")
    if common:
        for name in common:
            st.markdown(f"- {carbon_tag(name, 'green')}", unsafe_allow_html=True)
    else:
        st.caption("No common latent variables.")

with ov_col2:
    st.markdown("**Unique to Experiment A**")
    if unique_a:
        for name in unique_a:
            st.markdown(f"- {carbon_tag(name, 'blue')}", unsafe_allow_html=True)
    else:
        st.caption("None.")

with ov_col3:
    st.markdown("**Unique to Experiment B**")
    if unique_b:
        for name in unique_b:
            st.markdown(f"- {carbon_tag(name, 'yellow')}", unsafe_allow_html=True)
    else:
        st.caption("None.")
