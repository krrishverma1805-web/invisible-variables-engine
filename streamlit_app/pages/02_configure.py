import os

import requests
import streamlit as st

st.set_page_config(
    page_title="Configure Experiment - IVE",
    page_icon="⚙️",
    layout="wide",
)

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
HEADERS = {"X-API-Key": "dev-key-1"}

st.title("⚙️ Configure Experiment")
st.markdown("Set up a new Invisible Variables Engine analysis.")

# --- Fetch Datasets ---
datasets_dict = {}
try:
    response = requests.get(f"{API_BASE}/api/v1/datasets/", headers=HEADERS, timeout=5)
    if response.ok:
        data = response.json()
        for ds in data.get("datasets", []):
            label = f"{ds['name']} (Target: {ds['target_column']})"
            datasets_dict[label] = ds["id"]
except requests.RequestException:
    st.error("Could not connect to the API to fetch datasets.")

if not datasets_dict:
    st.warning("No datasets available. Please upload a dataset first.")
    st.page_link("pages/01_upload.py", label="Go to Upload Dataset", icon="📂")
    st.stop()

# --- Configuration Form ---
with st.form("config_form"):
    st.subheader("Select Dataset")
    selected_label = st.selectbox("Dataset to analyze", list(datasets_dict.keys()))
    dataset_id = datasets_dict[selected_label]

    st.divider()

    st.subheader("Model Selection")
    col1, col2 = st.columns(2)
    with col1:
        use_linear = st.checkbox(
            "Linear Model", value=True, help="Train a Ridge regression/classification baseline."
        )
    with col2:
        use_xgboost = st.checkbox(
            "XGBoost Model", value=True, help="Train a non-linear XGBoost model."
        )

    st.divider()

    st.subheader("Analysis Parameters")
    col3, col4 = st.columns(2)
    with col3:
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of folds for out-of-fold residual generation.",
        )
    with col4:
        bootstrap_iters = st.slider(
            "Bootstrap Iterations",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Iterations for stability validation.",
        )

    submitted = st.form_submit_button("Run Experiment", type="primary")

if submitted:
    model_types = []
    if use_linear:
        model_types.append("linear")
    if use_xgboost:
        model_types.append("xgboost")

    if not model_types:
        st.error("Please select at least one model type.")
    else:
        with st.spinner("Queueing experiment..."):
            payload = {
                "dataset_id": dataset_id,
                "config": {
                    "model_types": model_types,
                    "cv_folds": cv_folds,
                    "bootstrap_iterations": bootstrap_iters,
                },
            }

            try:
                response = requests.post(
                    f"{API_BASE}/api/v1/experiments/", headers=HEADERS, json=payload, timeout=5
                )
                if response.ok:
                    res_data = response.json()
                    exp_id = res_data.get("id")
                    st.success(f"Experiment queued successfully! ID: {exp_id}")
                    # Store experiment ID in session state for the monitor page
                    st.session_state["active_experiment_id"] = exp_id
                    st.info("Head over to the Monitor page to see progress.")
                    st.page_link("pages/03_monitor.py", label="Monitor Progress", icon="⏳")
                else:
                    st.error(f"Failed to queue experiment: {response.text}")
            except requests.RequestException as e:
                st.error(f"Connection error: {str(e)}")
