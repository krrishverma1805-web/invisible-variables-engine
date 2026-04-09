import os

import pandas as pd
import requests
import streamlit as st
from components.sidebar import render_sidebar
from components.theme import apply_carbon_theme

st.set_page_config(
    page_title="Upload Dataset - IVE",
    page_icon=":material/folder:",
    layout="wide",
)
apply_carbon_theme()
render_sidebar()

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

                    # Fetch full profile
                    ds_id = res_data.get("id")
                    if ds_id:
                        try:
                            profile_resp = requests.get(
                                f"{API_BASE}/api/v1/datasets/{ds_id}/profile",
                                headers=HEADERS,
                                timeout=10,
                            )
                            if profile_resp.ok:
                                profile = profile_resp.json()

                                # Quality Issues
                                quality_issues = profile.get("quality_issues", [])
                                if quality_issues:
                                    st.subheader("Data Quality Issues")
                                    for issue in quality_issues:
                                        severity = issue.get("severity", "low")
                                        if severity == "high":
                                            st.error(
                                                f"**{issue.get('column', 'N/A')}:** "
                                                f"{issue.get('message', '')}"
                                            )
                                        elif severity == "medium":
                                            st.warning(
                                                f"**{issue.get('column', 'N/A')}:** "
                                                f"{issue.get('message', '')}"
                                            )
                                        else:
                                            st.info(
                                                f"**{issue.get('column', 'N/A')}:** "
                                                f"{issue.get('message', '')}"
                                            )

                                # Recommendations
                                recommendations = profile.get("recommendations", [])
                                if recommendations:
                                    st.subheader("Recommendations")
                                    for rec in recommendations:
                                        st.markdown(f"- {rec}")

                                # Top Correlations
                                top_corrs = profile.get("top_correlations", [])
                                if top_corrs:
                                    st.subheader("Top Feature Correlations")
                                    corr_df = pd.DataFrame(
                                        [
                                            {
                                                "Feature A": c.get("feature_a", ""),
                                                "Feature B": c.get("feature_b", ""),
                                                "Correlation": f"{c.get('correlation', 0):.3f}",
                                            }
                                            for c in top_corrs[:10]
                                        ]
                                    )
                                    st.dataframe(
                                        corr_df, use_container_width=True, hide_index=True
                                    )
                        except requests.RequestException:
                            pass  # Profile is optional

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

            # Delete dataset
            st.caption("Select a dataset to delete:")
            delete_options = {ds.get("name", "Unknown"): ds.get("id") for ds in datasets}
            if delete_options:
                del_name = st.selectbox(
                    "Dataset to delete", list(delete_options.keys()), key="del_ds"
                )
                if st.button("Delete Dataset", type="secondary"):
                    del_id = delete_options[del_name]
                    # Check for associated experiments
                    try:
                        exp_resp = requests.get(
                            f"{API_BASE}/api/v1/experiments/",
                            headers=HEADERS,
                            params={"dataset_id": del_id},
                            timeout=5,
                        )
                        n_experiments = 0
                        if exp_resp.ok:
                            n_experiments = exp_resp.json().get("total", 0)
                    except requests.RequestException:
                        n_experiments = 0

                    if n_experiments > 0:
                        st.warning(
                            f"This dataset has **{n_experiments} experiment(s)**. "
                            "Deleting it will also remove all associated experiments "
                            "and results."
                        )

                    confirm_key = f"confirm_del_{del_id}"
                    if st.checkbox(
                        f"I confirm I want to delete '{del_name}'", key=confirm_key
                    ):
                        try:
                            del_resp = requests.delete(
                                f"{API_BASE}/api/v1/datasets/{del_id}",
                                headers=HEADERS,
                                timeout=10,
                            )
                            if del_resp.status_code == 204:
                                st.success(f"Dataset '{del_name}' deleted.")
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {del_resp.text}")
                        except requests.RequestException as e:
                            st.error(f"Connection error: {str(e)}")
        else:
            st.info("No datasets uploaded yet.")
    else:
        st.error(f"Failed to fetch datasets: {response.status_code}")
except requests.RequestException:
    st.error("Could not connect to the API to fetch datasets.")
