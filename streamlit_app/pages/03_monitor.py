"""
Page 3 — Monitor Progress.

Shows real-time experiment phase progress via polling or WebSocket.
"""

from __future__ import annotations

import time

import streamlit as st

st.set_page_config(page_title="Monitor Progress — IVE", layout="wide")
st.title("📊 Monitor Experiment Progress")

# TODO: Fetch experiments from GET /api/v1/experiments
experiment_id = st.text_input("Experiment ID", placeholder="Paste experiment UUID here")

auto_refresh = st.checkbox("Auto-refresh every 5 seconds", value=False)

PHASES = ["understand", "model", "detect", "construct"]
PHASE_LABELS = {
    "understand": "Phase 1 — Understand",
    "model": "Phase 2 — Model",
    "detect": "Phase 3 — Detect",
    "construct": "Phase 4 — Construct",
}

if experiment_id:
    st.markdown("---")
    # TODO: GET /api/v1/experiments/{experiment_id} and update these placeholders
    status = "running"  # placeholder
    current_phase = "model"  # placeholder

    st.metric("Status", status.upper())

    for phase in PHASES:
        label = PHASE_LABELS[phase]
        if phase == current_phase:
            st.progress(0.5, text=f"⏳ {label} — in progress")
        elif PHASES.index(phase) < PHASES.index(current_phase):
            st.progress(1.0, text=f"✅ {label} — complete")
        else:
            st.progress(0.0, text=f"⬜ {label} — pending")

    if auto_refresh:
        time.sleep(5)
        st.rerun()
else:
    st.info("Enter an Experiment ID above to monitor progress.")
