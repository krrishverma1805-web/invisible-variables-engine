import json
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Results Dashboard - IVE",
    page_icon="📊",
    layout="wide",
)

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
HEADERS = {"X-API-Key": "dev-key-1"}

st.title("📊 Results Dashboard")
st.markdown("View discovered error patterns and validated latent variables.")

# --- Fetch Completed Experiments ---
experiments_dict = {}
try:
    response = requests.get(f"{API_BASE}/api/v1/experiments/", headers=HEADERS, timeout=5)
    if response.ok:
        data = response.json()
        for exp in data.get("experiments", []):
            if exp.get("status") == "completed":
                label = f"{exp['id'][:8]}... - COMPLETED"
                experiments_dict[label] = exp["id"]
except requests.RequestException:
    st.error("Could not connect to the API to fetch experiments.")

if not experiments_dict:
    st.info("No completed experiments found. Please run an experiment first.")
    st.page_link("pages/02_configure.py", label="Run Experiment", icon="⚙️")
    st.stop()

# Pre-select based on session state if available
default_index = 0
active_exp_id = st.session_state.get("active_experiment_id")
if active_exp_id:
    for i, exp_id in enumerate(experiments_dict.values()):
        if exp_id == active_exp_id:
            default_index = i
            break

selected_label = st.selectbox(
    "Select Completed Experiment", list(experiments_dict.keys()), index=default_index
)
experiment_id = experiments_dict[selected_label]

st.divider()

# --- Fetch Data for Selected Experiment ---
patterns_data = []
lv_data = []

with st.spinner("Loading results..."):
    # Fetch Patterns
    try:
        p_resp = requests.get(
            f"{API_BASE}/api/v1/experiments/{experiment_id}/patterns", headers=HEADERS, timeout=5
        )
        if p_resp.ok:
            patterns_data = p_resp.json()
    except requests.RequestException:
        st.error("Failed to load error patterns.")

    # Fetch Latent Variables
    try:
        lv_resp = requests.get(
            f"{API_BASE}/api/v1/experiments/{experiment_id}/latent-variables",
            headers=HEADERS,
            timeout=5,
        )
        if lv_resp.ok:
            # Latent variable endpoint returns paginated response
            lv_data = lv_resp.json().get("variables", [])
    except requests.RequestException:
        st.error("Failed to load latent variables.")

# Calculate summary stats
total_patterns = len(patterns_data)
total_validated = sum(1 for v in lv_data if v.get("status") == "validated")
total_rejected = sum(1 for v in lv_data if v.get("status") == "rejected")

# --- UI Layout ---

# Section 3: Summary Statistics (Displayed at top for overview)
st.subheader("Summary Statistics")
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Total Error Patterns", total_patterns)
with metric_col2:
    st.metric("Validated Latent Variables", total_validated, delta="Stable", delta_color="normal")
with metric_col3:
    st.metric("Rejected Candidates", total_rejected, delta="Unstable", delta_color="inverse")

st.divider()

# Section 1: Discovered Patterns
st.subheader("Discovered Patterns")
if patterns_data:
    df_patterns = []
    for p in patterns_data:
        # Extract column/value info from subgroup_definition or use raw JSON string
        subgroup = p.get("subgroup_definition", {})
        col_name = subgroup.get("column_name", "N/A")

        df_patterns.append(
            {
                "Type": p.get("pattern_type").title(),
                "Column/Definition": col_name if col_name != "N/A" else str(subgroup),
                "Effect Size": f"{p.get('effect_size', 0):.3f}",
                "P-Value": f"{p.get('p_value', 1.0):.4f}",
                "Samples": p.get("sample_count", 0),
            }
        )
    st.dataframe(pd.DataFrame(df_patterns), use_container_width=True, hide_index=True)
else:
    st.info("No error patterns were discovered in the residuals.")

st.divider()

# Section 2: Validated Latent Variables
st.subheader("Validated Latent Variables")
validated_lvs = [v for v in lv_data if v.get("status") == "validated"]

if validated_lvs:
    for lv in validated_lvs:
        with st.expander(
            f"✨ {lv.get('name')} (Stability: {lv.get('stability_score', 0):.2f})", expanded=True
        ):
            col_info, col_metrics = st.columns([2, 1])

            with col_info:
                st.markdown(f"**Description:** {lv.get('description')}")
                st.markdown(f"**Explanation:** {lv.get('explanation_text', 'N/A')}")

                st.markdown("**Construction Rule:**")
                # Format JSON nicely
                rule = lv.get("construction_rule", {})
                st.code(json.dumps(rule, indent=2), language="json")

            with col_metrics:
                # Format Bootstrap Presence Rate as percentage
                presence_rate = lv.get("bootstrap_presence_rate", 0) * 100
                st.metric("Bootstrap Presence Rate", f"{presence_rate:.1f}%")
                st.metric("Importance Score", f"{lv.get('importance_score', 0):.3f}")

else:
    if total_patterns > 0:
        st.warning("Patterns were found, but none passed the bootstrap stability cross-validation.")
    else:
        st.info("No stable latent variables constructed.")
