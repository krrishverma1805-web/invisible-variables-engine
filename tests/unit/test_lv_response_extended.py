"""Tests for the extended LatentVariableResponse fields (Wave 3).

Confirms that ``serialize_lv`` propagates:

* ``confidence_interval_lower / _upper`` (Phase B4)
* ``cross_fit_splits_supporting`` (plan §96)
* ``selection_corrected`` (plan §96)
* ``apply_compatibility`` (plan §157 / §197)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

from ive.api.v1.schemas.latent_variable_schemas import serialize_lv


def _make_lv(**overrides):
    base = dict(
        id=uuid.uuid4(),
        experiment_id=uuid.uuid4(),
        name="weekend_pattern",
        description="weekend traffic spike",
        construction_rule={"feature": "is_weekend"},
        importance_score=0.42,
        stability_score=0.91,
        bootstrap_presence_rate=0.83,
        explanation_text="Rule-based prose.",
        status="validated",
        created_at=datetime.now(UTC),
        llm_explanation=None,
        llm_explanation_status="pending",
        confidence_interval_lower=0.31,
        confidence_interval_upper=0.53,
        cross_fit_splits_supporting=4,
        selection_corrected=True,
        apply_compatibility="ok",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestExtendedFields:
    def test_pending_lv_carries_ci_and_selective_inference_flags(self):
        lv = _make_lv()
        out = serialize_lv(lv)
        assert out.confidence_interval_lower == 0.31
        assert out.confidence_interval_upper == 0.53
        assert out.cross_fit_splits_supporting == 4
        assert out.selection_corrected is True
        assert out.apply_compatibility == "ok"
        # Pending status -> rule-based prose, pending=True.
        assert out.explanation_source == "rule_based"
        assert out.llm_explanation_pending is True

    def test_ready_lv_with_llm_text_uses_llm_source(self):
        lv = _make_lv(
            llm_explanation="LLM prose with hedged phrasing.",
            llm_explanation_status="ready",
        )
        out = serialize_lv(lv)
        assert out.explanation_source == "llm"
        assert out.explanation_text == "LLM prose with hedged phrasing."
        # Selective-inference fields still propagated.
        assert out.cross_fit_splits_supporting == 4
        assert out.selection_corrected is True

    def test_apply_compatibility_propagates_requires_review(self):
        lv = _make_lv(apply_compatibility="requires_review")
        out = serialize_lv(lv)
        assert out.apply_compatibility == "requires_review"

    def test_apply_compatibility_propagates_incompatible(self):
        lv = _make_lv(apply_compatibility="incompatible")
        out = serialize_lv(lv)
        assert out.apply_compatibility == "incompatible"

    def test_legacy_lv_without_new_fields_defaults_correctly(self):
        # Mimic an LV from a pre-Wave-3 codepath (no new attrs at all).
        lv = SimpleNamespace(
            id=uuid.uuid4(),
            experiment_id=uuid.uuid4(),
            name="x",
            description="x",
            construction_rule={},
            importance_score=0.0,
            stability_score=0.0,
            bootstrap_presence_rate=0.0,
            explanation_text="rule-based",
            status="rejected",
            created_at=datetime.now(UTC),
            llm_explanation=None,
            llm_explanation_status="disabled",
        )
        out = serialize_lv(lv)
        assert out.confidence_interval_lower is None
        assert out.confidence_interval_upper is None
        assert out.cross_fit_splits_supporting is None
        assert out.selection_corrected is False
        assert out.apply_compatibility == "ok"
        assert out.explanation_source == "rule_based"
        assert out.llm_explanation_pending is False
