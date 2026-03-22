import os

import requests
import streamlit as st


def render_release_metadata():
    """Renders release metadata and platform status in the sidebar."""

    st.divider()

    # Release Metadata
    st.caption("**Invisible Variables Engine (IVE)**")
    st.caption("Production-style latent variable discovery platform")

    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #888; margin-top: 0.5rem;">
            <div><b>Version:</b> 0.1.0</div>
            <div><b>Environment:</b> Development / Demo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # Status Indicators
    def status_icon(is_ok: bool) -> str:
        return "✅" if is_ok else "❌"

    st.markdown(
        f"""
        <div style="font-size: 0.85rem; line-height: 1.6;">
            <div>API Reachable: {status_icon(api_reachable)}</div>
            <div>Database Sync: {status_icon(db_ready)}</div>
            <div>Worker Queue (Redis): {status_icon(redis_ready)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if api_reachable and db_ready and redis_ready:
        st.success("All systems operational", icon="✅")
    elif api_reachable:
        st.warning("System degraded (backend issues)", icon="⚠️")
    else:
        st.error("System offline", icon="🔴")
