"""Dataset Settings — column sensitivity editor.

Per plan §142 / §174 / §202 / §203: users mark which columns are safe to
send to the LLM. Defaults to ``non_public`` (privacy-conservative); users
opt in by promoting columns to ``public``.

The page links from the dataset-level sensitivity banner on the Results
page (PR-6) so users can fix the limitation surfaced there.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from components.sidebar import render_sidebar
from components.theme import apply_carbon_theme, carbon_tag, explanation_source_badge

st.set_page_config(
    page_title="Dataset Settings - IVE",
    page_icon=":material/lock:",
    layout="wide",
)
apply_carbon_theme()
render_sidebar()

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
HEADERS = {"X-API-Key": "dev-key-1"}

st.title("Dataset Settings")
st.caption(
    "Mark which columns are safe to send to the AI explanation service. "
    "Non-public columns never leave the IVE perimeter; findings that reference "
    "them fall back to rule-based prose."
)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset picker
# ─────────────────────────────────────────────────────────────────────────────
try:
    ds_resp = requests.get(
        f"{API_BASE}/api/v1/datasets/", headers=HEADERS, timeout=5
    )
    ds_resp.raise_for_status()
    datasets = ds_resp.json().get("datasets", [])
except requests.RequestException:
    st.error("Could not connect to the API to fetch datasets.")
    st.stop()

if not datasets:
    st.info("No datasets uploaded yet. Visit the Upload page first.")
    st.stop()

dataset_options = {
    f"{d.get('name', '<unnamed>')} ({d['id'][:8]}…)": d["id"] for d in datasets
}
selected_label = st.selectbox(
    "Select a dataset",
    options=list(dataset_options.keys()),
    key="dataset_settings_picker",
)
dataset_id: str = dataset_options[selected_label]

# ─────────────────────────────────────────────────────────────────────────────
# Fetch column metadata
# ─────────────────────────────────────────────────────────────────────────────
try:
    col_resp = requests.get(
        f"{API_BASE}/api/v1/datasets/{dataset_id}/columns/",
        headers=HEADERS,
        timeout=10,
    )
    col_resp.raise_for_status()
    column_data = col_resp.json()
except requests.RequestException as exc:
    st.error(f"Could not load column metadata: {exc}")
    st.stop()

items: list[dict] = column_data.get("items", [])
total = column_data.get("total", len(items))
public_count = column_data.get("public_count", 0)

# ─────────────────────────────────────────────────────────────────────────────
# Header summary + AI-eligibility badge
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])
with col_left:
    st.metric(
        "Public columns",
        f"{public_count} of {total}",
        help="Only public columns appear in AI explanation payloads.",
    )

with col_right:
    if public_count == 0:
        st.markdown(
            explanation_source_badge("rule_based", status="disabled"),
            unsafe_allow_html=True,
        )
        st.caption("All findings will fall back to rule-based prose.")
    elif public_count < total:
        st.markdown(carbon_tag("Partial AI coverage", "yellow"), unsafe_allow_html=True)
        st.caption(
            "Findings referencing only public columns get AI-assisted prose. "
            "Findings touching any non-public column fall back."
        )
    else:
        st.markdown(carbon_tag("Full AI coverage", "green"), unsafe_allow_html=True)
        st.caption("Every finding is eligible for AI-assisted prose.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Bulk action helpers
# ─────────────────────────────────────────────────────────────────────────────
SESSION_KEY = f"dataset_settings::{dataset_id}::draft"

# Initialize a per-dataset draft in session_state so users can stage many
# changes before saving. Re-seeded whenever the dataset id changes.
if SESSION_KEY not in st.session_state or st.session_state.get(
    "_settings_loaded_for"
) != dataset_id:
    st.session_state[SESSION_KEY] = {
        item["column_name"]: item["sensitivity"] for item in items
    }
    st.session_state["_settings_loaded_for"] = dataset_id

draft: dict[str, str] = st.session_state[SESSION_KEY]
canonical: dict[str, str] = {
    item["column_name"]: item["sensitivity"] for item in items
}

bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
with bulk_col1:
    if st.button("Mark all public", use_container_width=True):
        for k in draft:
            draft[k] = "public"
        st.rerun()
with bulk_col2:
    if st.button("Mark all non-public", use_container_width=True):
        for k in draft:
            draft[k] = "non_public"
        st.rerun()
with bulk_col3:
    if st.button("Reset to saved", use_container_width=True):
        st.session_state[SESSION_KEY] = dict(canonical)
        st.rerun()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Per-column editor
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Per-column sensitivity")
st.caption(
    "Mark public only when the column name and aggregated values are safe "
    "to share with a third-party LLM. Examples of usually-safe: numeric metrics "
    "like `delivery_time`, low-cardinality categoricals like `region`. "
    "Examples of usually-not-safe: anything with `_id`, names, addresses, "
    "financial values, health attributes."
)

if not items:
    st.info("This dataset has no column metadata yet (re-upload to seed defaults).")
    st.stop()

header = st.columns([3, 2, 2])
with header[0]:
    st.caption("**Column**")
with header[1]:
    st.caption("**Sensitivity**")
with header[2]:
    st.caption("**Saved value**")

for column_name in sorted(draft.keys()):
    row = st.columns([3, 2, 2])
    saved_value = canonical.get(column_name, "non_public")
    with row[0]:
        # Visual signal: bold name, plus a hint when it's been edited.
        edited = draft[column_name] != saved_value
        marker = " ✱" if edited else ""
        st.markdown(f"`{column_name}`{marker}")
    with row[1]:
        new_value = st.radio(
            label=column_name,
            options=["public", "non_public"],
            index=0 if draft[column_name] == "public" else 1,
            horizontal=True,
            key=f"sensitivity::{dataset_id}::{column_name}",
            label_visibility="collapsed",
        )
        draft[column_name] = new_value
    with row[2]:
        color = "green" if saved_value == "public" else "gray"
        st.markdown(carbon_tag(saved_value.replace("_", " "), color), unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
diff = {col: val for col, val in draft.items() if canonical.get(col) != val}

st.markdown(
    f"**Pending changes:** {len(diff)} of {len(draft)} columns "
    + ("(no changes)" if not diff else "")
)

if diff:
    with st.expander("Preview pending changes", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "column": col,
                        "from": canonical.get(col, "non_public"),
                        "to": new,
                    }
                    for col, new in sorted(diff.items())
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

save_col1, save_col2 = st.columns([1, 4])
with save_col1:
    save_clicked = st.button(
        "Save changes",
        type="primary",
        use_container_width=True,
        disabled=not diff,
    )

if save_clicked and diff:
    payload = {
        "updates": [
            {"column_name": col, "sensitivity": new} for col, new in diff.items()
        ]
    }
    try:
        save_resp = requests.put(
            f"{API_BASE}/api/v1/datasets/{dataset_id}/columns/",
            headers=HEADERS,
            json=payload,
            timeout=15,
        )
    except requests.RequestException as exc:
        st.error(f"Save failed: {exc}")
    else:
        if save_resp.ok:
            st.success(
                f"Saved {len(diff)} change(s). AI explanation eligibility "
                "recomputes on the next experiment run.",
                icon=":material/check_circle:",
            )
            # Reset session draft so the page re-fetches canonical state.
            st.session_state.pop(SESSION_KEY, None)
            st.session_state.pop("_settings_loaded_for", None)
            st.rerun()
        else:
            st.error(
                f"Save failed (HTTP {save_resp.status_code}): {save_resp.text[:200]}"
            )
