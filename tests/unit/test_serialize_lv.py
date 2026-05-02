"""Unit tests for the LV serialization helper (LLM-prefer logic).

Per plan §A1 endpoint behavior + §174 (status semantics):
    - status=='ready' AND llm_explanation present → use LLM, source='llm'.
    - status=='pending' → rule-based, source='rule_based', pending=True.
    - status in {'failed','disabled'} → rule-based, source='rule_based', pending=False.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from ive.api.v1.schemas.latent_variable_schemas import serialize_lv

pytestmark = pytest.mark.unit


def _lv(
    *,
    explanation_text: str = "rule-based prose",
    llm_text: str | None = None,
    llm_status: str = "pending",
):
    lv = MagicMock()
    lv.id = uuid.uuid4()
    lv.experiment_id = uuid.uuid4()
    lv.name = "lv1"
    lv.description = "users where x > 5"
    lv.construction_rule = {"source_columns": ["x"]}
    lv.importance_score = 0.42
    lv.stability_score = 0.85
    lv.bootstrap_presence_rate = 0.85
    lv.explanation_text = explanation_text
    lv.status = "validated"
    lv.created_at = datetime.now(UTC)
    lv.llm_explanation = llm_text
    lv.llm_explanation_status = llm_status
    return lv


class TestSerializeLv:
    def test_ready_with_llm_text_uses_llm(self):
        lv = _lv(llm_text="LLM-polished prose", llm_status="ready")
        out = serialize_lv(lv)
        assert out.explanation_text == "LLM-polished prose"
        assert out.explanation_source == "llm"
        assert out.llm_explanation_pending is False
        assert out.llm_explanation_status == "ready"

    def test_ready_but_no_llm_text_falls_back(self):
        # Edge case: status='ready' set but llm_explanation column is None.
        lv = _lv(llm_text=None, llm_status="ready")
        out = serialize_lv(lv)
        assert out.explanation_text == "rule-based prose"
        assert out.explanation_source == "rule_based"
        assert out.llm_explanation_pending is False

    def test_pending_status_marks_pending_true(self):
        lv = _lv(llm_text=None, llm_status="pending")
        out = serialize_lv(lv)
        assert out.explanation_text == "rule-based prose"
        assert out.explanation_source == "rule_based"
        assert out.llm_explanation_pending is True
        assert out.llm_explanation_status == "pending"

    def test_failed_status_no_pending(self):
        lv = _lv(llm_text=None, llm_status="failed")
        out = serialize_lv(lv)
        assert out.explanation_text == "rule-based prose"
        assert out.explanation_source == "rule_based"
        assert out.llm_explanation_pending is False
        assert out.llm_explanation_status == "failed"

    def test_disabled_status_no_pending(self):
        lv = _lv(llm_text=None, llm_status="disabled")
        out = serialize_lv(lv)
        assert out.explanation_source == "rule_based"
        assert out.llm_explanation_pending is False
        assert out.llm_explanation_status == "disabled"

    def test_default_status_when_none(self):
        # Pre-PR-1 rows might have None for llm_explanation_status; the
        # serializer must treat that as 'pending' (its declared default).
        lv = _lv(llm_text=None, llm_status=None)
        out = serialize_lv(lv)
        assert out.explanation_source == "rule_based"
        assert out.llm_explanation_pending is True

    def test_preserves_core_fields(self):
        lv = _lv(llm_text="x", llm_status="ready")
        out = serialize_lv(lv)
        assert out.id == lv.id
        assert out.experiment_id == lv.experiment_id
        assert out.name == "lv1"
        assert out.importance_score == 0.42
