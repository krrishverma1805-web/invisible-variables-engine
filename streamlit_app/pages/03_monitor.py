import os
import time

import requests
import streamlit as st

st.set_page_config(
    page_title="Monitor Progress - IVE",
    page_icon="⏳",
    layout="wide",
)

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
HEADERS = {"X-API-Key": "dev-key-1"}

st.title("⏳ Monitor Progress")
st.markdown("Track the execution status of your IVE experiments.")

# --- Fetch Experiments ---
experiments_dict = {}
try:
    response = requests.get(f"{API_BASE}/api/v1/experiments/", headers=HEADERS, timeout=5)
    if response.ok:
        data = response.json()
        for idx, exp in enumerate(data.get("experiments", [])):
            stage = exp.get("current_stage", "queued")
            # Create a label with UUID substring and status
            label = f"{exp['id'][:8]}... - {exp['status'].upper()} ({stage})"
            experiments_dict[label] = exp["id"]
except requests.RequestException:
    st.error("Could not connect to the API to fetch experiments.")

if not experiments_dict:
    st.info("No experiments found. Run an experiment first.")
    st.page_link("pages/02_configure.py", label="Go to Configure Experiment", icon="⚙️")
    st.stop()

# Pre-select active experiment if one just started
default_index = 0
active_exp_id = st.session_state.get("active_experiment_id")
if active_exp_id:
    # Find matching index
    for i, exp_id in enumerate(experiments_dict.values()):
        if exp_id == active_exp_id:
            default_index = i
            break

col1, col2 = st.columns([3, 1])
with col1:
    selected_label = st.selectbox(
        "Select Experiment", list(experiments_dict.keys()), index=default_index
    )
    experiment_id = experiments_dict[selected_label]

# Store the currently selected experiment back so auto-refresh maintains selection
st.session_state["active_experiment_id"] = experiment_id

st.divider()

# --- Monitor Active Experiment ---
try:
    # Use the lightweight progress endpoint
    response = requests.get(
        f"{API_BASE}/api/v1/experiments/{experiment_id}/progress", headers=HEADERS, timeout=2
    )
    if response.ok:
        exp_data = response.json()
        status = exp_data.get("status", "unknown").lower()
        progress = exp_data.get("progress_pct", 0)
        stage = exp_data.get("current_stage", "N/A")

        # Display Status Badge
        if status == "completed":
            st.success("✅ **Status: COMPLETED**")
        elif status == "failed":
            st.error("❌ **Status: FAILED**")
        elif status == "cancelled":
            st.warning("🛑 **Status: CANCELLED**")
        elif status == "running":
            st.info("🚀 **Status: RUNNING**")
        else:
            st.markdown(f"**Status:** {status.upper()}")

        # Display Progress Bar
        st.progress(
            progress / 100.0, text=f"{progress}% Complete — Current Stage: {str(stage).title()}"
        )

        # Display Details
        if status == "failed":
            # Fetch full details to get error message
            full_resp = requests.get(
                f"{API_BASE}/api/v1/experiments/{experiment_id}", headers=HEADERS, timeout=2
            )
            if full_resp.ok:
                full_data = full_resp.json()
                error_msg = full_data.get("error_message", "Unknown error")
                st.error(f"Error Details: {error_msg}")

        elif status == "completed":
            full_resp = requests.get(
                f"{API_BASE}/api/v1/experiments/{experiment_id}", headers=HEADERS, timeout=2
            )
            if full_resp.ok:
                full_data = full_resp.json()
                elapsed = "?"
                if full_data.get("started_at") and full_data.get("completed_at"):
                    # Basic calculation for display elapsed time could go here
                    pass
                st.success("Analysis complete! View the discovered patterns and latent variables.")

                # Check if results page exists to set up link properly, otherwise just use standard link
                st.page_link("pages/04_results.py", label="View Results Dashboard", icon="📊")

        # Auto-refresh if running or queued
        if status in ("running", "queued"):
            with st.spinner("Auto-refreshing status every 3 seconds..."):
                time.sleep(3)
                st.rerun()

    else:
        st.error(f"Failed to fetch progress: {response.text}")

except requests.RequestException:
    st.error("Lost connection to API while polling status.")
