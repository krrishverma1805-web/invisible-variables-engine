"""Prompt-payload builders for the LLM enrichment task.

These are the *only* place in the codebase that converts ORM rows /
construction rules into the dict that ends up in ``INPUT_FACTS``.  Two
invariants the rest of the LLM package relies on:

1. **Numeric grounding** — every numeric field included is a verbatim
   stat the validator can ground against; nothing is rounded.
2. **Egress safety** — only column names listed in ``public_columns``
   appear in the payload.  Non-public column names never leave the
   process boundary.

Plan reference: §103 (batched calls — payload extraction shared with the
single-LV path), §107 (deterministic shape), §142 / §174 / §203 (egress).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ive.auth.egress import evaluate_lv_egress
from ive.db.models import Experiment, LatentVariable
from ive.llm.validators import sanitize_user_input


def _construction_rule_columns(rule: dict[str, Any] | None) -> list[str]:
    """Return the column names referenced by a construction rule.

    The rule schema varies by pattern type (subgroup, cluster, interaction,
    temporal). All known shapes carry their referenced columns under one
    of these keys; this helper unifies the lookup.
    """
    if not rule:
        return []
    cols: list[str] = []
    for key in ("source_columns", "feature", "features", "columns"):
        v = rule.get(key)
        if isinstance(v, str):
            cols.append(v)
        elif isinstance(v, list):
            cols.extend(c for c in v if isinstance(c, str))
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def build_lv_payload(
    lv: LatentVariable,
    *,
    public_columns: Iterable[str],
    target_column: str | None,
) -> tuple[dict[str, Any], list[str]] | None:
    """Build the prompt payload for a single latent variable.

    Returns:
        ``(payload, blocked_columns)`` — when egress is allowed, ``payload``
        is the dict to feed ``render('lv_explanation', 'v1', payload)`` and
        ``blocked_columns`` is empty.  When egress is blocked, returns
        ``None``; the caller should mark the LV ``disabled`` with reason
        ``pii_protection_per_column``.
    """
    referenced = _construction_rule_columns(lv.construction_rule)
    decision = evaluate_lv_egress(referenced, public_columns)
    if not decision.allowed:
        return None

    payload: dict[str, Any] = {
        "name": sanitize_user_input(lv.name),
        "status": lv.status,
        "effect_size": float(lv.importance_score),
        "presence_rate": float(lv.bootstrap_presence_rate),
        "stability_score": float(lv.stability_score),
    }

    if target_column and target_column in set(public_columns):
        payload["target_column"] = sanitize_user_input(target_column)

    if lv.model_improvement_pct is not None:
        payload["model_improvement_pct"] = float(lv.model_improvement_pct)

    if lv.confidence_interval_lower is not None and lv.confidence_interval_upper is not None:
        payload["effect_size_ci_lower"] = float(lv.confidence_interval_lower)
        payload["effect_size_ci_upper"] = float(lv.confidence_interval_upper)

    description = sanitize_user_input(lv.description or "")
    if description:
        payload["segment_human"] = description

    return payload, []


def build_experiment_payload(
    experiment: Experiment,
    *,
    lvs: list[LatentVariable],
    public_columns: Iterable[str],
    target_column: str | None,
    dataset_name: str | None,
) -> dict[str, Any]:
    """Build the prompt payload for the experiment-level narrative.

    Aggregates only LVs whose construction rule references exclusively
    public columns; non-eligible LVs are summarized as a count, never
    by name.
    """
    public_set = set(public_columns)
    eligible: list[LatentVariable] = []
    blocked = 0
    for lv in lvs:
        cols = _construction_rule_columns(lv.construction_rule)
        if all(c in public_set for c in cols):
            eligible.append(lv)
        else:
            blocked += 1

    top = sorted(
        eligible,
        key=lambda x: (x.importance_score or 0.0),
        reverse=True,
    )[:5]

    headline_stats: list[str] = []
    findings: list[dict[str, Any]] = []
    for lv in top:
        stat = f"{float(lv.importance_score):.2f}"
        headline_stats.append(stat)
        findings.append(
            {
                "name": sanitize_user_input(lv.name),
                "effect_size": float(lv.importance_score),
                "presence_rate": float(lv.bootstrap_presence_rate),
            }
        )

    payload: dict[str, Any] = {
        "n_findings": len(eligible),
        "n_blocked_for_pii": blocked,
        "headline_stats": headline_stats,
        "top_findings": findings,
        "experiment_status": experiment.status,
    }
    if target_column and target_column in public_set:
        payload["target_column"] = sanitize_user_input(target_column)
    if dataset_name:
        payload["dataset_name"] = sanitize_user_input(dataset_name)
    return payload
