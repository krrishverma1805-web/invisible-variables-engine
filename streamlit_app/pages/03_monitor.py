"""
Monitor Progress — Invisible Variables Engine.

Provides real-time experiment status tracking, configuration display,
and a chronological Execution Log sourced from the pipeline event audit table.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Phase badge colours (CSS-class-free — use inline markdown emoji)
# ---------------------------------------------------------------------------
_PHASE_ICONS: dict[str, str] = {
    "understand": "🔍",
    "model": "🤖",
    "detect": "🔎",
    "construct": "🏗️",
}

_EVENT_ICONS: dict[str, str] = {
    "experiment_started": "🚀",
    "dataset_loaded": "📂",
    "modeling_started": "⚙️",
    "modeling_completed": "✅",
    "detection_started": "🔎",
    "detection_completed": "✅",
    "construction_started": "🏗️",
    "construction_completed": "✅",
    "experiment_completed": "🎉",
    "experiment_failed": "❌",
}

st.title("⏳ Monitor Progress")
st.markdown("Track the execution status of your IVE experiments in real time.")

# ---------------------------------------------------------------------------
# Experiment selector
# ---------------------------------------------------------------------------
experiments_dict: dict[str, str] = {}
experiments_meta: dict[str, dict] = {}

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
st.session_state["active_experiment_id"] = experiment_id

st.divider()

# ---------------------------------------------------------------------------
# Live status + progress
# ---------------------------------------------------------------------------
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

        # ── Status badge ──────────────────────────────────────────────────
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

        # ── Progress bar ──────────────────────────────────────────────────
        st.progress(
            progress / 100.0,
            text=f"{progress}% Complete — Current Stage: {str(stage).title()}",
        )

        # ── Full experiment detail: mode + config ─────────────────────────
        full_resp = requests.get(
            f"{API_BASE}/api/v1/experiments/{experiment_id}",
            headers=HEADERS,
            timeout=5,
        )
        full_data: dict = {}
        if full_resp.ok:
            full_data = full_resp.json()

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

        # ── Error detail ──────────────────────────────────────────────────
        if status == "failed":
            error_msg = full_data.get("error_message", "Unknown error")
            st.error(f"**Error Details:** {error_msg}")

        elif status == "completed":
            st.success(
                "Analysis complete. View the discovered patterns and validated latent variables."
            )
            st.page_link("pages/04_results.py", label="View Results Dashboard →", icon="📊")

        # ── Execution Log ─────────────────────────────────────────────────
        st.subheader("🗂️ Execution Log")

        events_resp = requests.get(
            f"{API_BASE}/api/v1/experiments/{experiment_id}/events",
            headers=HEADERS,
            timeout=5,
        )

        if events_resp.ok:
            events_data = events_resp.json()
            events: list[dict] = events_data.get("events", [])

            if not events:
                st.info(
                    "No execution events recorded yet. "
                    "Events will appear here once the pipeline starts running."
                )
            else:
                # Determine whether the last event is a failure
                last_event = events[-1]
                last_is_failure = last_event.get("event_type") == "experiment_failed"

                with st.container():
                    for idx, event in enumerate(events):
                        ev_type: str = event.get("event_type", "unknown")
                        phase: str | None = event.get("phase")
                        payload: dict = event.get("payload") or {}
                        message: str = payload.get("message", ev_type)
                        created_at: str = event.get("created_at", "")

                        # Format timestamp (ISO → readable)
                        try:
                            from datetime import datetime

                            ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                            ts_label = ts.strftime("%H:%M:%S UTC")
                        except Exception:
                            ts_label = created_at[:19] if created_at else "—"

                        icon = _EVENT_ICONS.get(ev_type, "📋")
                        phase_icon = _PHASE_ICONS.get(phase or "", "")
                        phase_tag = f" `{phase_icon} {phase}`" if phase else ""

                        # Highlight the last event red on failure
                        is_last_failure = (
                            status == "failed" and idx == len(events) - 1 and last_is_failure
                        )

                        if is_last_failure:
                            st.error(
                                f"`{ts_label}` &nbsp; {icon} &nbsp; **{ev_type}**{phase_tag}\n\n"
                                f"{message}"
                            )
                        elif ev_type == "experiment_completed":
                            st.success(
                                f"`{ts_label}` &nbsp; {icon} &nbsp; **{ev_type}**{phase_tag}\n\n"
                                f"{message}"
                            )
                        else:
                            st.markdown(
                                f"`{ts_label}` &nbsp; {icon} &nbsp; **{ev_type}**{phase_tag}  \n"
                                f"<small>{message}</small>",
                                unsafe_allow_html=True,
                            )

                st.caption(f"Showing {len(events)} event(s) · oldest first")

        else:
            st.warning("Could not retrieve the execution log for this experiment.")

        # ── Auto-refresh while running ────────────────────────────────────
        if status in ("running", "queued"):
            with st.spinner("Auto-refreshing every 3 seconds…"):
                time.sleep(3)
                st.rerun()

    else:
        st.error(f"Failed to fetch progress: {progress_resp.text}")

except requests.RequestException:
    st.error("Lost connection to the API while polling status.")
