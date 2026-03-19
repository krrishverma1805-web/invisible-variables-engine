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
st.markdown("Track the execution status of your IVE experiments in real time.")

# --- Fetch All Experiments ---
experiments_dict: dict[str, str] = {}
experiments_meta: dict[str, dict] = {}  # label → raw experiment dict
try:
    response = requests.get(f"{API_BASE}/api/v1/experiments/", headers=HEADERS, timeout=5)
    if response.ok:
        data = response.json()
        for exp in data.get("experiments", []):
            stage = exp.get("current_stage", "queued")
            label = f"{exp['id'][:8]}… — {exp['status'].upper()} ({stage})"
            experiments_dict[label] = exp["id"]
            experiments_meta[label] = exp
except requests.RequestException:
    st.error("Could not connect to the API to fetch experiments.")

if not experiments_dict:
    st.info("No experiments found. Run an experiment first.")
    st.page_link("pages/02_configure.py", label="Configure Experiment →", icon="⚙️")
    st.stop()

# Pre-select the active experiment from session state
default_index = 0
active_exp_id = st.session_state.get("active_experiment_id")
if active_exp_id:
    for i, exp_id in enumerate(experiments_dict.values()):
        if exp_id == active_exp_id:
            default_index = i
            break

col_sel, col_refresh = st.columns([4, 1])
with col_sel:
    selected_label = st.selectbox(
        "Select Experiment",
        list(experiments_dict.keys()),
        index=default_index,
        key="monitor_exp_select",
    )
experiment_id = experiments_dict[selected_label]

# Keep session state in sync
st.session_state["active_experiment_id"] = experiment_id

st.divider()

# --- Monitor Selected Experiment ---
try:
    progress_resp = requests.get(
        f"{API_BASE}/api/v1/experiments/{experiment_id}/progress",
        headers=HEADERS,
        timeout=5,
    )
    if progress_resp.ok:
        exp_data = progress_resp.json()
        status = exp_data.get("status", "unknown").lower()
        progress = exp_data.get("progress_pct", 0)
        stage = exp_data.get("current_stage", "N/A")

        # ── Status badge ───────────────────────────────────────────────
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

        # ── Progress bar ───────────────────────────────────────────────
        st.progress(
            progress / 100.0,
            text=f"{progress}% Complete — Current Stage: {str(stage).title()}",
        )

        # ── Fetch full experiment detail to show mode + config ─────────
        full_resp = requests.get(
            f"{API_BASE}/api/v1/experiments/{experiment_id}",
            headers=HEADERS,
            timeout=5,
        )
        full_data: dict = {}
        if full_resp.ok:
            full_data = full_resp.json()

        # Display analysis mode from stored config
        config_json: dict = full_data.get("config_json", {})
        analysis_mode: str = str(config_json.get("analysis_mode", "demo")).lower()
        mode_label = "🔬 Demo" if analysis_mode == "demo" else "🏭 Production"
        mode_desc = (
            "Permissive thresholds — optimised for exploration and demonstrations."
            if analysis_mode == "demo"
            else "Stricter thresholds — optimised to reduce false positives in production."
        )

        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.metric("Analysis Mode", mode_label)
        with meta_col2:
            cv = config_json.get("cv_folds", "—")
            st.metric("CV Folds", cv)
        with meta_col3:
            bs = config_json.get("bootstrap_iterations", "—")
            st.metric("Bootstrap Iterations", bs)

        st.caption(f"ℹ️ {mode_desc}")
        st.divider()

        # ── Error detail ───────────────────────────────────────────────
        if status == "failed":
            error_msg = full_data.get("error_message", "Unknown error")
            st.error(f"**Error Details:** {error_msg}")

        elif status == "completed":
            st.success(
                "Analysis complete. View the discovered patterns and validated latent variables."
            )
            st.page_link("pages/04_results.py", label="View Results Dashboard →", icon="📊")

        # ── Auto-refresh while running ─────────────────────────────────
        if status in ("running", "queued"):
            with st.spinner("Auto-refreshing every 3 seconds…"):
                time.sleep(3)
                st.rerun()

    else:
        st.error(f"Failed to fetch progress: {progress_resp.text}")

except requests.RequestException:
    st.error("Lost connection to the API while polling status.")
