"""
Page 4 — Results.

Displays discovered latent variables with explanations, scores, and charts.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Results — IVE", layout="wide")
st.title("🔍 Latent Variable Results")

experiment_id = st.text_input("Experiment ID", placeholder="Paste completed experiment UUID")

if st.button("🔄 Load Results") and experiment_id:
    with st.spinner("Fetching latent variables..."):
        # TODO: GET /api/v1/experiments/{experiment_id}/latent-variables
        # lvs = httpx.get(...).json()["items"]
        lvs = [
            {
                "rank": 1,
                "name": "Neighbourhood Quality Factor",
                "confidence_score": 0.87,
                "effect_size": 0.34,
                "coverage_pct": 23.5,
                "candidate_features": ["zip_code", "avg_commute_mins"],
                "explanation": "The model consistently under-estimates for samples where "
                "neighbourhood quality (proxied by zip_code and commute time) is high.",
                "validation": {"bootstrap_stability": 0.91, "p_value": 0.0012},
            }
        ]  # placeholder

        if not lvs:
            st.warning("No latent variables found for this experiment.")
        else:
            for lv in lvs:
                with st.expander(
                    f"#{lv['rank']} — {lv['name']} (confidence: {lv['confidence_score']:.2f})",
                    expanded=lv["rank"] == 1,
                ):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Confidence", f"{lv['confidence_score']:.2f}")
                    col2.metric("Effect Size (d)", f"{lv['effect_size']:.2f}")
                    col3.metric("Coverage", f"{lv['coverage_pct']:.1f}%")

                    st.markdown(f"**Explanation:** {lv['explanation']}")
                    st.markdown(f"**Candidate Features:** `{', '.join(lv['candidate_features'])}`")

                    val = lv.get("validation", {})
                    st.markdown(
                        f"**Bootstrap Stability:** {val.get('bootstrap_stability', 'N/A')} | "
                        f"**p-value:** {val.get('p_value', 'N/A')}"
                    )

                    # TODO: Add SHAP bar chart using streamlit_app/components/charts.py
