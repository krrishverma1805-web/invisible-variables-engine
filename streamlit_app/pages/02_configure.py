"""
Page 2 — Configure Experiment.

Lets users select a dataset and configure experiment parameters before launching.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Configure Experiment — IVE", layout="wide")
st.title("⚙️ Configure Experiment")

# TODO: Fetch datasets from GET /api/v1/datasets and populate selector
dataset_id = st.selectbox("Select Dataset", options=["(No datasets loaded — upload one first)"])

st.markdown("---")
st.subheader("Model Configuration")

col1, col2, col3 = st.columns(3)
with col1:
    model_types = st.multiselect(
        "Model Types", options=["linear", "xgboost"], default=["linear", "xgboost"]
    )
    cv_folds = st.slider("CV Folds", min_value=2, max_value=10, value=5)
with col2:
    min_cluster_size = st.number_input("Min Cluster Size", min_value=5, value=10)
    shap_sample_size = st.number_input("SHAP Sample Size", min_value=50, value=500)
with col3:
    max_latent_vars = st.slider("Max Latent Variables", min_value=1, max_value=20, value=5)
    random_seed = st.number_input("Random Seed", value=42)

exp_name = st.text_input("Experiment Name", placeholder="e.g. Housing v1")

if st.button("▶️ Start Experiment", type="primary"):
    with st.spinner("Queuing experiment..."):
        # TODO: POST /api/v1/experiments with config payload
        st.success("Experiment queued! Navigate to **Monitor Progress** to track it.")
