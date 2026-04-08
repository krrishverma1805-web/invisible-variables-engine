import os

import pandas as pd
import requests
import streamlit as st
from components.theme import apply_carbon_theme

st.set_page_config(
    page_title="Upload Dataset - IVE",
    page_icon=":material/folder:",
    layout="wide",
)
apply_carbon_theme()

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
HEADERS = {"X-API-Key": "dev-key-1"}

st.title("Upload Dataset")
st.markdown("Upload a CSV file to begin analysis.")

# --- Upload Form ---
with st.form("upload_form"):
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    col1, col2 = st.columns(2)
    with col1:
        target_column = st.text_input(
            "Target Column Name *", help="The name of the column you want to predict."
        )
    with col2:
        time_column = st.text_input(
            "Time Column Name (Optional)",
            help="If your data has a temporal component, specify the column name.",
        )

    submitted = st.form_submit_button("Upload Dataset", type="primary")

if submitted:
    if not uploaded_file:
        st.error("Please select a file to upload.")
    elif not target_column:
        st.error("Please specify the Target Column Name.")
    else:
        with st.spinner("Uploading and profiling dataset..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                data = {"target_column": target_column}
                if time_column:
                    data["time_column"] = time_column

                response = requests.post(
                    f"{API_BASE}/api/v1/datasets/",
                    headers=HEADERS,
                    files=files,
                    data=data,
                    timeout=30,
                )

                if response.ok:
                    res_data = response.json()
                    st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")

                    # Show Profile if available
                    st.subheader("Dataset Profile")

                    profile_cols = st.columns(4)
                    with profile_cols[0]:
                        st.metric("Rows", res_data.get("row_count", "N/A"))
                    with profile_cols[1]:
                        st.metric("Columns", res_data.get("col_count", "N/A"))
                    with profile_cols[2]:
                        # quality_score might be nested in schema_json if profile task completed synchronously
                        schema = res_data.get("schema_json", {})
                        qs = schema.get("quality_score", "N/A")
                        if isinstance(qs, float):
                            qs = f"{qs:.1f}/100"
                        st.metric("Quality Score", qs)

                else:
                    st.error(f"Upload failed: {response.text}")
            except requests.RequestException as e:
                st.error(f"Connection error: {str(e)}")

st.divider()

# --- List Uploaded Datasets ---
st.subheader("Previously Uploaded Datasets")

try:
    response = requests.get(f"{API_BASE}/api/v1/datasets/", headers=HEADERS, timeout=5)
    if response.ok:
        data = response.json()
        datasets = data.get("datasets", [])

        if datasets:
            # Flatten data for dataframe
            df_data = []
            for ds in datasets:
                schema = ds.get("schema_json", {})
                quality = schema.get("quality_score", "")
                if isinstance(quality, float):
                    quality = f"{quality:.1f}"

                df_data.append(
                    {
                        "Name": ds.get("name"),
                        "Target": ds.get("target_column"),
                        "Rows": ds.get("row_count"),
                        "Columns": ds.get("col_count"),
                        "Quality": quality,
                        "Uploaded At": pd.to_datetime(ds.get("created_at")).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        if ds.get("created_at")
                        else "",
                    }
                )

            st.dataframe(pd.DataFrame(df_data), use_container_width=True, hide_index=True)
        else:
            st.info("No datasets uploaded yet.")
    else:
        st.error(f"Failed to fetch datasets: {response.status_code}")
except requests.RequestException:
    st.error("Could not connect to the API to fetch datasets.")
