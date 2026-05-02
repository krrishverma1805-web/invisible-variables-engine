"""Pydantic types for the LLM enrichment layer.

Plan reference: §A1, §107.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    """Input to a single LLM generation call.

    ``facts`` is the canonicalized, sanitized payload the prompt template
    will serialize into ``INPUT_FACTS``.  ``allowed_columns`` lists every
    column-like identifier the LLM is permitted to mention; the validator
    rejects mentions of any other column name.
    """

    model_config = ConfigDict(extra="forbid")

    function: str = Field(description="Prompt-registry name, e.g. 'lv_explanation'.")
    facts: dict[str, Any] = Field(description="Sanitized fact payload.")
    allowed_columns: set[str] = Field(
        default_factory=set,
        description="Column names the LLM is permitted to mention.",
    )
    allow_causal: bool = Field(
        default=False,
        description="When False, causal verbs trigger validation failure.",
    )


class ValidationReport(BaseModel):
    """Outcome of running ``composite_validate`` against an LLM output."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    failures: list[str] = Field(
        default_factory=list,
        description="Human-readable failure reasons; empty when passed=True.",
    )
    rule: str | None = Field(
        default=None,
        description="First failing rule name (None when passed=True).",
    )


class GenerationResult(BaseModel):
    """Outcome of a generation attempt with fallback awareness."""

    model_config = ConfigDict(extra="forbid")

    text: str
    source: Literal["llm", "fallback", "cache"]
    validation: ValidationReport | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    cache_status: Literal["hit", "miss", "bypass"] = "miss"
    failure_reason: str | None = None
