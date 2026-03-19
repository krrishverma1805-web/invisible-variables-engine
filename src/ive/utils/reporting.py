"""
Reporting Utilities — Invisible Variables Engine.

Provides in-memory helpers for building CSV and full-report payloads
from experiment result objects.  All functions are pure (no I/O side
effects) so they are easy to test and compose.

Functions
---------
patterns_to_csv          — list[dict] → CSV string
latent_variables_to_csv  — list[dict] → CSV string
build_full_report        — assemble the complete report dict
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_PATTERN_COLUMNS = [
    "pattern_type",
    "column_name",
    "effect_size",
    "p_value",
    "adjusted_p_value",
    "sample_count",
    "mean_residual",
    "std_residual",
]

_LV_COLUMNS = [
    "name",
    "status",
    "stability_score",
    "bootstrap_presence_rate",
    "importance_score",
    "description",
    "explanation_text",
]


def patterns_to_csv(patterns: list[dict[str, Any]]) -> str:
    """Convert a list of error-pattern dicts to a CSV string.

    The ``column_name`` field is extracted from the nested
    ``subgroup_definition`` dict when not present at the top level —
    this handles both the raw Phase 3 format and the API response format.

    Args:
        patterns: List of pattern dicts (may come from DB or detection phase).

    Returns:
        UTF-8 CSV string with a header row.  Returns a header-only string
        when *patterns* is empty.
    """
    rows: list[dict[str, Any]] = []
    for p in patterns:
        # column_name may be nested inside subgroup_definition
        subgroup = p.get("subgroup_definition", {})
        col_name = p.get("column_name") or subgroup.get("column_name", "")

        rows.append(
            {
                "pattern_type": p.get("pattern_type", ""),
                "column_name": col_name,
                "effect_size": p.get("effect_size", ""),
                "p_value": p.get("p_value", ""),
                "adjusted_p_value": p.get("adjusted_p_value", ""),
                "sample_count": p.get("sample_count", ""),
                "mean_residual": p.get("mean_residual", ""),
                "std_residual": p.get("std_residual", ""),
            }
        )

    df = (
        pd.DataFrame(rows, columns=_PATTERN_COLUMNS)
        if rows
        else pd.DataFrame(columns=_PATTERN_COLUMNS)
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    log.debug("reporting.patterns_to_csv", n_rows=len(df))
    return buf.getvalue()


def latent_variables_to_csv(variables: list[dict[str, Any]]) -> str:
    """Convert a list of latent variable dicts to a CSV string.

    Args:
        variables: List of latent variable dicts (may come from DB ORM
                   model serialised to dict, or API response JSON).

    Returns:
        UTF-8 CSV string with a header row.  Returns a header-only string
        when *variables* is empty.
    """
    rows: list[dict[str, Any]] = []
    for v in variables:
        rows.append(
            {
                "name": v.get("name", ""),
                "status": v.get("status", ""),
                "stability_score": v.get("stability_score", ""),
                "bootstrap_presence_rate": v.get("bootstrap_presence_rate", ""),
                "importance_score": v.get("importance_score", ""),
                "description": v.get("description", ""),
                "explanation_text": v.get("explanation_text", ""),
            }
        )

    df = pd.DataFrame(rows, columns=_LV_COLUMNS) if rows else pd.DataFrame(columns=_LV_COLUMNS)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    log.debug("reporting.lv_to_csv", n_rows=len(df))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Full report builder
# ---------------------------------------------------------------------------


def build_full_report(
    experiment: dict[str, Any],
    dataset: dict[str, Any],
    patterns: list[dict[str, Any]],
    latent_variables: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the complete experiment report dict.

    Args:
        experiment:       Serialised experiment metadata.
        dataset:          Serialised dataset metadata.
        patterns:         List of error-pattern dicts.
        latent_variables: List of latent variable dicts.
        summary:          Experiment summary dict (from ExplanationGenerator
                          or a fallback).

    Returns:
        Full report dict suitable for JSON serialisation.
    """
    report = {
        "experiment": experiment,
        "dataset": dataset,
        "patterns": patterns,
        "latent_variables": latent_variables,
        "summary": summary,
    }
    log.info(
        "reporting.full_report_built",
        experiment_id=experiment.get("id"),
        n_patterns=len(patterns),
        n_latent_variables=len(latent_variables),
    )
    return report


# ---------------------------------------------------------------------------
# Fallback summary builder (when ExplanationGenerator is unavailable)
# ---------------------------------------------------------------------------


def build_fallback_summary(
    patterns: list[dict[str, Any]],
    latent_variables: list[dict[str, Any]],
    dataset_name: str,
    target_column: str,
) -> dict[str, Any]:
    """Construct a minimal summary dict when ExplanationGenerator output is not cached.

    Args:
        patterns:         Error-pattern dicts.
        latent_variables: Latent variable dicts.
        dataset_name:     Name of the dataset.
        target_column:    Name of the prediction target column.

    Returns:
        Summary dict with the same keys as ExplanationGenerator output.
    """
    validated = [v for v in latent_variables if v.get("status") == "validated"]
    rejected = [v for v in latent_variables if v.get("status") == "rejected"]

    n_p = len(patterns)
    n_v = len(validated)
    n_r = len(rejected)

    if n_v > 0:
        headline = f"{n_v} hidden variable{'s' if n_v != 1 else ''} confirmed in {dataset_name}"
    elif n_p > 0:
        headline = f"{n_p} patterns found but none passed stability validation"
    else:
        headline = f"No significant patterns detected in {dataset_name}"

    top_findings = [v.get("explanation_text", v.get("name", "")) for v in validated]
    recommendations = [v.get("description", "") for v in validated]

    return {
        "headline": headline,
        "patterns_found": n_p,
        "validated_variables": n_v,
        "rejected_variables": n_r,
        "summary_text": (
            f"The IVE analysis of {dataset_name} (target: {target_column}) "
            f"identified {n_p} error pattern{'s' if n_p != 1 else ''}, "
            f"of which {n_v} {'were' if n_v != 1 else 'was'} validated "
            f"and {n_r} {'were' if n_r != 1 else 'was'} rejected."
        ),
        "top_findings": top_findings,
        "recommendations": recommendations,
    }
