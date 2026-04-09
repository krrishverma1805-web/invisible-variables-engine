import json
import os

import pandas as pd
import requests
import streamlit as st
from components.charts import shap_bar_chart, residual_histogram, coverage_gauge
from components.sidebar import render_sidebar
from components.theme import apply_carbon_theme, carbon_tag

st.set_page_config(
    page_title="Results Dashboard - IVE",
    page_icon=":material/bar_chart:",
    layout="wide",
)
apply_carbon_theme()
render_sidebar()

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
HEADERS = {"X-API-Key": "dev-key-1"}

st.title("Results Dashboard")
st.markdown(
    "Review discovered error patterns, validated latent variables, and experiment insights."
)

# --- Fetch All Completed Experiments ---
experiments_dict: dict[str, str] = {}
experiments_full: dict[str, dict] = {}  # label → raw experiment dict (for config)
try:
    response = requests.get(f"{API_BASE}/api/v1/experiments/", headers=HEADERS, timeout=5)
    if response.ok:
        data = response.json()
        for exp in data.get("experiments", []):
            if exp.get("status") == "completed":
                config_json: dict = exp.get("config_json", {})
                mode = str(config_json.get("analysis_mode", "demo")).capitalize()
                label = f"{exp['id'][:8]}… — COMPLETED [{mode} Mode]"
                experiments_dict[label] = exp["id"]
                experiments_full[label] = exp
except requests.RequestException:
    st.error("Could not connect to the API to fetch experiments.")

if not experiments_dict:
    st.info("No completed experiments found. Please run an experiment first.")
    st.page_link("pages/02_configure.py", label="Run Experiment →", icon=":material/settings:")
    st.stop()

# Pre-select based on session state if available
default_index = 0
active_exp_id = st.session_state.get("active_experiment_id")
if active_exp_id:
    for i, exp_id in enumerate(experiments_dict.values()):
        if exp_id == active_exp_id:
            default_index = i
            break

selected_label = st.selectbox(
    "Select Completed Experiment",
    list(experiments_dict.keys()),
    index=default_index,
    key="results_exp_select",
)
experiment_id = experiments_dict[selected_label]
selected_exp_meta = experiments_full.get(selected_label, {})

st.divider()

# --- Derive analysis mode from the stored config ---
_cfg: dict = selected_exp_meta.get("config_json", {})
analysis_mode: str = str(_cfg.get("analysis_mode", "demo")).lower()
mode_label = "Demo" if analysis_mode == "demo" else "Production"
threshold_profile = (
    "Permissive (Demo) — effect size ≥ 0.15, bootstrap stability ≥ 0.60"
    if analysis_mode == "demo"
    else "Strict (Production) — effect size ≥ 0.20, bootstrap stability ≥ 0.70"
)

# --- Fetch all result data in parallel API calls ---
patterns_data: list[dict] = []
lv_data: list[dict] = []
summary_data: dict = {}
patterns_csv = ""
lv_csv = ""

with st.spinner("Loading results…"):
    try:
        p_resp = requests.get(
            f"{API_BASE}/api/v1/experiments/{experiment_id}/patterns",
            headers=HEADERS,
            timeout=10,
        )
        if p_resp.ok:
            patterns_data = p_resp.json()
    except requests.RequestException:
        st.error("Failed to load error patterns.")

    try:
        lv_resp = requests.get(
            f"{API_BASE}/api/v1/experiments/{experiment_id}/latent-variables",
            headers=HEADERS,
            timeout=10,
        )
        if lv_resp.ok:
            lv_data = lv_resp.json().get("variables", [])
    except requests.RequestException:
        st.error("Failed to load latent variables.")

    try:
        sum_resp = requests.get(
            f"{API_BASE}/api/v1/experiments/{experiment_id}/summary",
            headers=HEADERS,
            timeout=10,
        )
        if sum_resp.ok:
            summary_data = sum_resp.json()
    except requests.RequestException:
        st.error("Failed to load experiment summary.")

    try:
        p_csv_resp = requests.get(
            f"{API_BASE}/api/v1/experiments/{experiment_id}/patterns/export",
            headers=HEADERS,
            timeout=10,
        )
        if p_csv_resp.ok:
            patterns_csv = p_csv_resp.text
    except requests.RequestException:
        pass

    try:
        lv_csv_resp = requests.get(
            f"{API_BASE}/api/v1/experiments/{experiment_id}/latent-variables/export",
            headers=HEADERS,
            timeout=10,
        )
        if lv_csv_resp.ok:
            lv_csv = lv_csv_resp.text
    except requests.RequestException:
        pass

# --- Aggregate counts ---
total_patterns = len(patterns_data)
total_validated = sum(1 for v in lv_data if v.get("status") == "validated")
total_rejected = sum(1 for v in lv_data if v.get("status") == "rejected")

# ═══════════════════════════════════════════════════════════════════════
# Section: Analysis Mode Banner
# ═══════════════════════════════════════════════════════════════════════
mode_col1, mode_col2 = st.columns([1, 3])
with mode_col1:
    st.metric("Analysis Mode", mode_label)
with mode_col2:
    st.markdown(f"**Threshold Profile:** {threshold_profile}")

# Demo-mode zero-result advisory
if analysis_mode == "demo" and total_validated == 0 and total_patterns == 0:
    st.info(
        "**Demo mode** completed successfully, but no stable latent variables passed "
        "validation on this dataset. This typically indicates the dataset does not contain "
        "a strong enough hidden signal to satisfy the current thresholds. "
        "Try a dataset with a more pronounced latent structure, or switch to **Production** "
        "mode for production-scale data."
    )

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# Section: Export Actions
# ═══════════════════════════════════════════════════════════════════════
st.subheader("Export Results")
btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    if summary_data:
        st.download_button(
            label="Download Summary JSON",
            data=json.dumps(summary_data, indent=2),
            file_name=f"summary_{experiment_id}.json",
            mime="application/json",
            use_container_width=True,
        )

with btn_col2:
    if patterns_csv:
        st.download_button(
            label="Download Patterns CSV",
            data=patterns_csv,
            file_name=f"patterns_{experiment_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

with btn_col3:
    if lv_csv:
        st.download_button(
            label="Download Latent Variables CSV",
            data=lv_csv,
            file_name=f"latent_variables_{experiment_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# Section: Summary Statistics
# ═══════════════════════════════════════════════════════════════════════
st.subheader("Summary Statistics")
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Total Error Patterns", total_patterns)
with metric_col2:
    st.metric("Validated Latent Variables", total_validated, delta="Stable", delta_color="normal")
with metric_col3:
    st.metric("Rejected Candidates", total_rejected, delta="Unstable", delta_color="inverse")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# Section: Executive Summary
# ═══════════════════════════════════════════════════════════════════════
if summary_data:
    st.subheader("Executive Summary")

    # Analysis mode in summary
    summary_mode = summary_data.get("analysis_mode", analysis_mode)
    summary_profile = summary_data.get("threshold_profile", threshold_profile)
    st.caption(f"**Mode:** {summary_mode.capitalize()}  |  **Profile:** {summary_profile}")

    # Headline
    st.markdown(f"### {summary_data.get('headline', '')}")

    # Key insight (if available)
    key_insight = summary_data.get("key_insight", "")
    if key_insight:
        st.info(key_insight, icon=":material/lightbulb:")

    # Narrative summary (multi-paragraph)
    summary_text = summary_data.get("summary_text", "")
    if summary_text:
        for paragraph in summary_text.split("\n\n"):
            if paragraph.strip():
                st.markdown(paragraph.strip())

    # What this means (contextual interpretation)
    what_this_means = summary_data.get("what_this_means", "")
    if what_this_means:
        st.markdown(f"**What this means:** {what_this_means}")

    # Top findings
    findings = summary_data.get("top_findings", [])
    if findings:
        st.markdown("**Key Discoveries:**")
        for finding in findings:
            st.markdown(f"- {finding}")

    # Next steps
    next_steps = summary_data.get("next_steps", [])
    if next_steps:
        st.markdown("**Recommended Next Steps:**")
        for i, step in enumerate(next_steps, 1):
            st.markdown(f"{i}. {step}")

    # Recommendations
    recommendations = summary_data.get("recommendations", [])
    if recommendations and not next_steps:
        st.markdown("**Recommendations:**")
        for rec in recommendations:
            st.markdown(f"- {rec}")

    # Feature importance chart (if summary has feature data)
    feature_importance = summary_data.get("feature_importance", {})
    if feature_importance:
        shap_bar_chart(feature_importance, title="Feature Importance (SHAP)")

    st.divider()

# ═══════════════════════════════════════════════════════════════════════
# Section: Discovered Error Patterns
# ═══════════════════════════════════════════════════════════════════════
st.subheader("Discovered Error Patterns")
if patterns_data:
    df_patterns = []
    for p in patterns_data:
        subgroup = p.get("subgroup_definition", {})
        col_name = subgroup.get("column_name", "N/A")
        df_patterns.append(
            {
                "Type": p.get("pattern_type", "").title(),
                "Column / Definition": col_name if col_name != "N/A" else str(subgroup),
                "Effect Size": f"{p.get('effect_size', 0):.3f}",
                "P-Value": f"{p.get('p_value', 1.0):.4f}",
                "Samples": p.get("sample_count", 0),
            }
        )
    st.dataframe(pd.DataFrame(df_patterns), use_container_width=True, hide_index=True)
else:
    st.info("No error patterns were discovered in the residuals.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# Section: Validated Latent Variables
# ═══════════════════════════════════════════════════════════════════════
st.subheader("Latent Variables")
validated_lvs = [v for v in lv_data if v.get("status") == "validated"]
rejected_lvs = [v for v in lv_data if v.get("status") == "rejected"]

tab_validated, tab_rejected = st.tabs([
    f"Validated ({len(validated_lvs)})",
    f"Rejected ({len(rejected_lvs)})",
])

with tab_validated:
    if validated_lvs:
        for lv in validated_lvs:
            # Build expander title with improvement if available
            imp_data = lv.get("model_improvement_pct") or (lv.get("construction_rule") or {}).get("model_improvement_pct")
            if isinstance(imp_data, dict) and imp_data.get("improvement_pct"):
                expander_title = f"{lv.get('name')} — +{imp_data['improvement_pct']:.1f}% improvement (Stability: {lv.get('stability_score', 0):.2f})"
            else:
                expander_title = f"{lv.get('name')} (Stability: {lv.get('stability_score', 0):.2f})"

            with st.expander(expander_title, expanded=True):
                # Confidence badge
                stability = lv.get("stability_score", 0)
                holdout = (lv.get("construction_rule") or {}).get("holdout_validation", {}).get("holdout_validated")
                if stability >= 0.8 and holdout is True:
                    st.markdown(carbon_tag("High Confidence", "green"), unsafe_allow_html=True)
                elif stability >= 0.6:
                    st.markdown(carbon_tag("Medium Confidence", "blue"), unsafe_allow_html=True)
                else:
                    st.markdown(carbon_tag("Low Confidence", "yellow"), unsafe_allow_html=True)

                col_info, col_metrics = st.columns([2, 1])

                with col_info:
                    st.markdown(f"**Description:** {lv.get('description', 'N/A')}")
                    st.markdown(f"**Explanation:** {lv.get('explanation_text', 'N/A')}")
                    st.markdown("**Construction Rule:**")
                    rule = lv.get("construction_rule", {})
                    st.code(json.dumps(rule, indent=2), language="json")

                with col_metrics:
                    presence_rate = lv.get("bootstrap_presence_rate", 0) * 100
                    st.metric("Bootstrap Presence", f"{presence_rate:.1f}%")
                    st.metric("Importance Score", f"{lv.get('importance_score', 0):.3f}")
                    boot_mode = lv.get("bootstrap_mode", analysis_mode)
                    st.caption(f"Validated in **{boot_mode.capitalize()}** mode")

                    # Model improvement metrics
                    improvement = lv.get("model_improvement_pct")
                    if not improvement:
                        # Check inside construction_rule
                        rule = lv.get("construction_rule", {})
                        improvement = rule.get("model_improvement_pct")

                    if improvement and isinstance(improvement, dict):
                        metric_name = improvement.get("metric", "R²").upper()
                        baseline = improvement.get("baseline", 0)
                        augmented = improvement.get("augmented", 0)
                        imp_pct = improvement.get("improvement_pct", 0)
                        order = improvement.get("selection_order", "")

                        st.divider()
                        st.markdown(f"**Model Improvement**")
                        st.metric(
                            f"{metric_name} Change",
                            f"{augmented:.4f}",
                            delta=f"+{imp_pct:.1f}%",
                            delta_color="normal",
                        )
                        st.caption(f"Baseline {metric_name}: {baseline:.4f}")
                        if order:
                            st.caption(f"Selection order: #{order}")

    else:
        # Context-aware empty state
        if analysis_mode == "demo" and total_patterns > 0:
            st.info(
                "**Demo mode** — Patterns were detected but none passed stability validation. "
                "This dataset may require a stronger hidden signal or more rows to reach "
                "the stability threshold."
            )
        elif analysis_mode == "production" and total_patterns > 0:
            st.warning(
                "**Production mode** — Patterns were detected but none met the stricter "
                "stability requirements. Consider switching to **Demo** mode for exploratory analysis, "
                "or review the dataset for a clearer hidden-variable structure."
            )
        else:
            st.info("No stable latent variables were constructed in this experiment.")

with tab_rejected:
    if rejected_lvs:
        for lv in rejected_lvs:
            with st.expander(f"{lv.get('name')} — REJECTED", expanded=False):
                rejection_reason = lv.get("rejection_reason", "Unknown")
                st.error(f"**Rejection Reason:** {rejection_reason}")

                explanation = lv.get("explanation_text", "")
                if explanation:
                    st.markdown(f"**Explanation:** {explanation}")

                # Show causal warnings if present
                rule = lv.get("construction_rule", {})
                causal_info = rule.get("causal_warnings") or []
                causal_penalty = rule.get("causal_confidence_penalty", 1.0)
                if causal_info:
                    for warning in causal_info:
                        st.warning(f"**Causal Warning:** {warning}")
                    st.caption(f"Confidence penalty: ×{causal_penalty}")

                # Bootstrap stats
                presence = lv.get("bootstrap_presence_rate", 0)
                st.metric("Bootstrap Presence", f"{presence * 100:.1f}%")
    else:
        st.info("No rejected candidates for this experiment.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════
# Section: Apply Latent Variables to New Data
# ═══════════════════════════════════════════════════════════════════════
if validated_lvs:
    st.subheader("Apply Latent Variables to New Data")
    st.markdown("Upload a new CSV to compute latent variable scores on unseen data.")

    apply_file = st.file_uploader("Upload CSV for scoring", type=["csv"], key="apply_csv")

    if apply_file and st.button("Apply Latent Variables", type="primary"):
        with st.spinner("Applying construction rules to new data..."):
            try:
                files = {"file": (apply_file.name, apply_file.getvalue(), "text/csv")}
                data = {"experiment_id": experiment_id}
                apply_resp = requests.post(
                    f"{API_BASE}/api/v1/latent-variables/apply",
                    headers=HEADERS,
                    files=files,
                    data=data,
                    timeout=30,
                )

                if apply_resp.ok:
                    result = apply_resp.json()
                    n_applied = result.get("n_variables_applied", 0)
                    n_failed = result.get("n_variables_failed", 0)

                    if n_applied > 0:
                        st.success(f"Successfully applied {n_applied} latent variable(s) to {result.get('n_rows', 0)} rows.")
                    if n_failed > 0:
                        st.warning(f"{n_failed} variable(s) could not be applied.")

                    # Show results table
                    cols_added = result.get("columns_added", [])
                    if cols_added:
                        for col_info in cols_added:
                            if "error" in col_info:
                                st.error(f"**{col_info['name']}:** {col_info['error']}")
                            else:
                                st.markdown(
                                    f"**{col_info['name']}** — "
                                    f"Mean: {col_info.get('mean_score', 0):.4f}, "
                                    f"Non-zero: {col_info.get('non_zero_count', 0)}"
                                )
                elif apply_resp.status_code == 400:
                    error_data = apply_resp.json().get("error", {})
                    if error_data.get("code") == "SCHEMA_MISMATCH":
                        st.error(
                            f"**Schema Mismatch:** Your CSV is missing columns required "
                            f"by the construction rules.\n\n"
                            f"**Missing columns:** {error_data.get('message', '')}"
                        )
                    else:
                        st.error(f"Error: {error_data.get('message', apply_resp.text)}")
                else:
                    st.error(f"Failed to apply: {apply_resp.text}")
            except requests.RequestException as e:
                st.error(f"Connection error: {str(e)}")
