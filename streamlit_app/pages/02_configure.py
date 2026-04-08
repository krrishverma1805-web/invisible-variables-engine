import os

import requests
import streamlit as st
from components.theme import apply_carbon_theme

st.set_page_config(
    page_title="Configure Experiment - IVE",
    page_icon=":material/settings:",
    layout="wide",
)
apply_carbon_theme()

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
HEADERS = {"X-API-Key": "dev-key-1"}

st.title("Configure Experiment")
st.markdown("Set up a new Invisible Variables Engine analysis run.")

# --- Fetch Datasets ---
datasets_dict: dict[str, str] = {}
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
    st.page_link("pages/01_upload.py", label="Go to Upload Dataset", icon=":material/folder:")
    st.stop()

# --- Configuration Form ---
with st.form("config_form"):
    st.subheader("Select Dataset")
    selected_label = st.selectbox(
        "Dataset to analyse", list(datasets_dict.keys()), key="cfg_dataset"
    )
    dataset_id = datasets_dict[selected_label]

    st.divider()

    # ── Analysis Mode ──────────────────────────────────────────────────
    st.subheader("Analysis Mode")
    mode_options = ["Demo", "Production"]
    selected_mode_label = st.selectbox(
        "Analysis Mode",
        mode_options,
        index=0,
        key="cfg_analysis_mode",
        help=(
            "**Demo** — permissive thresholds, suitable for demonstrations and exploratory analysis.\n\n"
            "**Production** — stricter thresholds, optimised to minimise false positives in live environments."
        ),
    )
    analysis_mode_value = selected_mode_label.lower()

    # Inline helper text beneath the selectbox
    if analysis_mode_value == "demo":
        st.caption(
            "**Demo mode** applies relaxed detection thresholds "
            "(effect size ≥ 0.15, bootstrap stability ≥ 0.60) "
            "to maximise discoverability on synthetic and exploratory datasets."
        )
    else:
        st.caption(
            "**Production mode** applies stricter thresholds "
            "(effect size ≥ 0.20, bootstrap stability ≥ 0.70) "
            "to reduce false positives and ensure only high-confidence variables are surfaced."
        )

    st.divider()

    # ── Model Selection ────────────────────────────────────────────────
    st.subheader("Model Selection")
    col1, col2 = st.columns(2)
    with col1:
        use_linear = st.checkbox(
            "Linear Model",
            value=True,
            help="Train a Ridge regression baseline to capture linear signal.",
        )
    with col2:
        use_xgboost = st.checkbox(
            "XGBoost Model",
            value=True,
            help="Train a non-linear gradient boosted model to capture complex interactions.",
        )

    st.divider()

    # ── Analysis Parameters ────────────────────────────────────────────
    st.subheader("Analysis Parameters")
    col3, col4 = st.columns(2)
    with col3:
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of K-folds for out-of-fold residual generation.",
        )
    with col4:
        bootstrap_iters = st.slider(
            "Bootstrap Iterations",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Iterations for latent variable stability validation.",
        )

    submitted = st.form_submit_button("Run Experiment", type="primary", use_container_width=True)

if submitted:
    model_types: list[str] = []
    if use_linear:
        model_types.append("linear")
    if use_xgboost:
        model_types.append("xgboost")

    if not model_types:
        st.error("Please select at least one model type.")
    else:
        with st.spinner("Queueing experiment…"):
            payload = {
                "dataset_id": dataset_id,
                "config": {
                    "model_types": model_types,
                    "cv_folds": cv_folds,
                    "bootstrap_iterations": bootstrap_iters,
                    "analysis_mode": analysis_mode_value,
                },
            }

            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/experiments/",
                    headers=HEADERS,
                    json=payload,
                    timeout=10,
                )
                if resp.ok:
                    res_data = resp.json()
                    exp_id = res_data.get("id")
                    mode_badge = "Demo" if analysis_mode_value == "demo" else "Production"
                    st.success(
                        f"Experiment queued successfully!\n\n"
                        f"**ID:** `{exp_id}`  |  **Mode:** {mode_badge}"
                    )
                    st.session_state["active_experiment_id"] = exp_id
                    st.session_state["active_analysis_mode"] = analysis_mode_value
                    st.info("Head over to the Monitor page to track progress.")
                    st.page_link(
                        "pages/03_monitor.py",
                        label="Monitor Progress →",
                        icon=":material/hourglass_empty:",
                    )
                else:
                    st.error(f"Failed to queue experiment: {resp.text}")
            except requests.RequestException as e:
                st.error(f"Connection error: {str(e)}")
