"""
Latent Variable API Schemas — Invisible Variables Engine.

Pydantic v2 schemas for latent variable request/response models.
All ORM-backed schemas use ``model_config = ConfigDict(from_attributes=True)``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ExplanationSource = Literal["llm", "rule_based"]


class LatentVariableResponse(BaseModel):  # type: ignore[misc]
    """Single latent variable — maps directly from ``LatentVariable`` ORM model.

    The ``explanation_text`` field is the *active* explanation the UI should
    render — it surfaces the LLM-generated prose when
    ``llm_explanation_status='ready'``, otherwise the rule-based prose.

    ``explanation_source`` and ``llm_explanation_pending`` give the UI
    enough context to render the AI-assisted badge and a "generating…"
    indicator (per plan §A1, §107, §174).
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    experiment_id: UUID
    name: str
    description: str
    construction_rule: dict[str, Any]
    importance_score: float
    stability_score: float
    bootstrap_presence_rate: float
    explanation_text: str
    status: str
    created_at: datetime

    # ── LLM enrichment surface (per plan §A1) ──────────────────────────────
    explanation_source: ExplanationSource = Field(
        default="rule_based",
        description="Which generator produced explanation_text.",
    )
    llm_explanation_pending: bool = Field(
        default=False,
        description=(
            "True when llm_explanation_status='pending' — UI should poll. "
            "Always false in flag-off / disabled / failed states."
        ),
    )
    llm_explanation_status: str = Field(
        default="pending",
        description="Lifecycle: pending | ready | failed | disabled.",
    )

    # ── Confidence interval (Phase B4) + selective inference (plan §96) ───
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None
    cross_fit_splits_supporting: int | None = Field(
        default=None,
        description="Splits (out of K) in which this LV was discovered.",
    )
    selection_corrected: bool = Field(
        default=False,
        description="True when CI was computed via cross-fit (selection-aware).",
    )

    # ── Lineage / apply compatibility (plan §157 + §197 / RC §19) ─────────
    apply_compatibility: Literal["ok", "requires_review", "incompatible"] = Field(
        default="ok",
        description=(
            "Whether this LV can be re-applied to the dataset's current "
            "version: ok / requires_review / incompatible."
        ),
    )


class LatentVariableListResponse(BaseModel):  # type: ignore[misc]
    """Paginated list of latent variables."""

    variables: list[LatentVariableResponse]
    total: int
    skip: int
    limit: int


# Legacy aliases used by the existing latent_variables.py stub
LatentVariableDetail = LatentVariableResponse


def serialize_lv(lv: Any) -> LatentVariableResponse:
    """Build a ``LatentVariableResponse`` with LLM-prefer logic.

    Rules (per plan §A1 endpoint behavior):
        - status=='ready' AND llm_explanation present → use LLM prose,
          source='llm', pending=False.
        - status=='pending' → use rule-based prose, source='rule_based',
          pending=True so the UI can poll.
        - status in {'failed','disabled'} → use rule-based prose,
          source='rule_based', pending=False.
    """
    raw_status = getattr(lv, "llm_explanation_status", "pending") or "pending"
    llm_text = getattr(lv, "llm_explanation", None)
    rule_text = lv.explanation_text

    # apply_compatibility is a Literal in the response schema; coerce any
    # unexpected sentinel (e.g. MagicMock from older tests) to "ok".
    raw_apply_compat = getattr(lv, "apply_compatibility", "ok")
    if raw_apply_compat not in ("ok", "requires_review", "incompatible"):
        raw_apply_compat = "ok"

    # Optional numeric fields that older tests may not set.
    def _opt_float(name: str) -> float | None:
        v = getattr(lv, name, None)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _opt_int(name: str) -> int | None:
        v = getattr(lv, name, None)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _opt_bool(name: str) -> bool:
        v = getattr(lv, name, False)
        try:
            return bool(v)
        except Exception:
            return False

    if raw_status == "ready" and llm_text:
        active = llm_text
        source: ExplanationSource = "llm"
        pending = False
    else:
        active = rule_text
        source = "rule_based"
        pending = raw_status == "pending"

    return LatentVariableResponse(
        id=lv.id,
        experiment_id=lv.experiment_id,
        name=lv.name,
        description=lv.description,
        construction_rule=lv.construction_rule,
        importance_score=lv.importance_score,
        stability_score=lv.stability_score,
        bootstrap_presence_rate=lv.bootstrap_presence_rate,
        explanation_text=active,
        status=lv.status,
        created_at=lv.created_at,
        explanation_source=source,
        llm_explanation_pending=pending,
        llm_explanation_status=raw_status,
        confidence_interval_lower=_opt_float("confidence_interval_lower"),
        confidence_interval_upper=_opt_float("confidence_interval_upper"),
        cross_fit_splits_supporting=_opt_int("cross_fit_splits_supporting"),
        selection_corrected=_opt_bool("selection_corrected"),
        apply_compatibility=raw_apply_compat,  # type: ignore[arg-type]
    )
