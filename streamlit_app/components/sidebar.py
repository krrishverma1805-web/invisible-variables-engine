import os

import requests
import streamlit as st
from components.theme import carbon_status_dot, carbon_tag


def render_release_metadata():
    """Renders release metadata and platform status in the sidebar."""

    st.divider()

    # Release Metadata
    st.caption("**Invisible Variables Engine (IVE)**")
    st.caption("Production-style latent variable discovery platform")
    st.caption("**Version:** 0.1.0")
    st.caption("**Environment:** Development / Demo")

    st.divider()

    # Platform Status
    st.subheader("Platform Status", divider=False)

    api_base = os.getenv("API_BASE_URL", "http://api:8000")
    headers = {"X-API-Key": "dev-key-1"}

    api_reachable = False
    db_ready = False
    redis_ready = False

    try:
        # Check liveness
        live_resp = requests.get(f"{api_base}/api/v1/health", headers=headers, timeout=5)
        if live_resp.status_code == 200:
            api_reachable = True

        # Check readiness (DB and Redis)
        # Even if 503 is returned, the JSON response contains the specific component status
        ready_resp = requests.get(f"{api_base}/api/v1/health/ready", headers=headers, timeout=5)

        # We parse JSON if we can, regardless of status_code, since 503 means partially degraded
        if ready_resp.headers.get("content-type", "").startswith("application/json"):
            data = ready_resp.json()
            checks = data.get("checks", {})
            db_ready = checks.get("database") == "healthy"
            redis_ready = checks.get("redis") == "healthy"

    except requests.RequestException:
        pass

    # Status Indicators (Carbon status dots + tags)
    def _status_row(label: str, is_ok: bool) -> str:
        dot = carbon_status_dot("ok" if is_ok else "error")
        tag = carbon_tag("OK", "green") if is_ok else carbon_tag("Error", "red")
        return f'<div class="carbon-status-row">{dot} {label}: {tag}</div>'

    st.markdown(
        _status_row("API Reachable", api_reachable)
        + _status_row("Database Sync", db_ready)
        + _status_row("Worker Queue (Redis)", redis_ready),
        unsafe_allow_html=True,
    )

    if api_reachable and db_ready and redis_ready:
        st.success("All systems operational", icon=":material/check_circle:")
    elif api_reachable:
        st.warning("System degraded (backend issues)", icon=":material/warning:")
    else:
        st.error("System offline", icon=":material/error:")
