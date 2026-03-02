"""
Page 1 — Upload Dataset.

Allows users to upload a CSV or Parquet file and initiate dataset profiling.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Upload Dataset — IVE", layout="wide")
st.title("📂 Upload Dataset")
st.markdown("Upload a CSV or Parquet file to begin analysis.")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "parquet"],
    help="Supported formats: CSV, Parquet. Max size: 500 MB.",
)

col1, col2 = st.columns(2)
with col1:
    dataset_name = st.text_input("Dataset Name", placeholder="e.g. Housing Prices Q1 2024")
with col2:
    target_column = st.text_input("Target Column", placeholder="e.g. price")

description = st.text_area("Description (optional)", height=80)

if st.button("🚀 Upload & Profile", type="primary", disabled=not uploaded_file or not dataset_name or not target_column):
    with st.spinner("Uploading and queuing profiling task..."):
        # TODO: POST to /api/v1/datasets with multipart form
        # import httpx
        # resp = httpx.post(
        #     f"{st.session_state['api_base_url']}/api/v1/datasets",
        #     headers={"X-API-Key": st.session_state["api_key"]},
        #     files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
        #     data={"name": dataset_name, "target_column": target_column, "description": description},
        # )
        st.success(f"✅ Dataset '{dataset_name}' uploaded successfully! (placeholder)")
        st.info("Navigate to **Configure Experiment** to start an analysis.")
