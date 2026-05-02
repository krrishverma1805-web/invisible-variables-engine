"""Unit tests for ive.llm.payloads."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ive.llm.payloads import (
    _construction_rule_columns,
    build_experiment_payload,
    build_lv_payload,
)

pytestmark = pytest.mark.unit


def _lv(
    *,
    name: str = "lv1",
    importance: float = 0.42,
    presence: float = 0.85,
    stability: float = 0.9,
    rule: dict | None = None,
    description: str = "users where x > 5",
    status: str = "validated",
    improvement: float | None = None,
    ci: tuple[float, float] | None = None,
):
    lv = MagicMock()
    lv.name = name
    lv.importance_score = importance
    lv.bootstrap_presence_rate = presence
    lv.stability_score = stability
    lv.construction_rule = rule or {}
    lv.description = description
    lv.status = status
    lv.model_improvement_pct = improvement
    if ci is None:
        lv.confidence_interval_lower = None
        lv.confidence_interval_upper = None
    else:
        lv.confidence_interval_lower, lv.confidence_interval_upper = ci
    return lv


class TestConstructionRuleColumns:
    def test_handles_empty_rule(self):
        assert _construction_rule_columns(None) == []
        assert _construction_rule_columns({}) == []

    def test_extracts_from_source_columns_list(self):
        rule = {"source_columns": ["a", "b", "a"]}
        assert _construction_rule_columns(rule) == ["a", "b"]

    def test_extracts_from_feature_string(self):
        rule = {"feature": "income"}
        assert _construction_rule_columns(rule) == ["income"]

    def test_extracts_from_features_list(self):
        rule = {"features": ["age", "city"]}
        assert _construction_rule_columns(rule) == ["age", "city"]


class TestBuildLvPayload:
    def test_blocked_when_any_column_non_public(self):
        lv = _lv(rule={"source_columns": ["age", "ssn"]})
        out = build_lv_payload(lv, public_columns={"age"}, target_column="y")
        assert out is None

    def test_allowed_when_all_columns_public(self):
        lv = _lv(rule={"source_columns": ["age"]}, ci=(0.31, 0.53), improvement=12.4)
        result = build_lv_payload(lv, public_columns={"age"}, target_column="y")
        assert result is not None
        payload, _ = result
        assert payload["name"] == "lv1"
        assert payload["effect_size"] == 0.42
        assert payload["presence_rate"] == 0.85
        assert payload["stability_score"] == 0.9
        assert payload["model_improvement_pct"] == 12.4
        assert payload["effect_size_ci_lower"] == 0.31
        assert payload["effect_size_ci_upper"] == 0.53

    def test_target_only_included_when_public(self):
        lv = _lv(rule={"source_columns": ["age"]})
        result = build_lv_payload(lv, public_columns={"age"}, target_column="y")
        assert result is not None
        payload, _ = result
        # y is not in public set → omitted
        assert "target_column" not in payload

    def test_target_included_when_in_public_set(self):
        lv = _lv(rule={"source_columns": ["age"]})
        result = build_lv_payload(lv, public_columns={"age", "y"}, target_column="y")
        assert result is not None
        payload, _ = result
        assert payload["target_column"] == "y"

    def test_sanitizes_user_input(self):
        lv = _lv(name="lv1 ignore previous", rule={"source_columns": ["x"]})
        result = build_lv_payload(lv, public_columns={"x"}, target_column=None)
        assert result is not None
        payload, _ = result
        assert "ignore previous" not in payload["name"].lower()

    def test_omits_optional_fields_when_none(self):
        lv = _lv(rule={"source_columns": ["x"]}, improvement=None, ci=None)
        result = build_lv_payload(lv, public_columns={"x"}, target_column=None)
        assert result is not None
        payload, _ = result
        assert "model_improvement_pct" not in payload
        assert "effect_size_ci_lower" not in payload


class TestBuildExperimentPayload:
    def _exp(self, status: str = "completed"):
        e = MagicMock()
        e.status = status
        e.dataset_id = "ds-id"
        e.id = "exp-id"
        return e

    def test_filters_to_eligible_lvs(self):
        eligible = _lv(name="ok", rule={"source_columns": ["age"]})
        blocked = _lv(name="bad", rule={"source_columns": ["ssn"]})
        payload = build_experiment_payload(
            self._exp(),
            lvs=[eligible, blocked],
            public_columns={"age"},
            target_column="y",
            dataset_name="customers",
        )
        assert payload["n_findings"] == 1
        assert payload["n_blocked_for_pii"] == 1
        names = [f["name"] for f in payload["top_findings"]]
        assert names == ["ok"]
        assert "bad" not in [f["name"] for f in payload["top_findings"]]

    def test_top_5_by_importance(self):
        lvs = [
            _lv(name=f"lv{i}", importance=float(i), rule={"source_columns": ["x"]})
            for i in range(10)
        ]
        payload = build_experiment_payload(
            self._exp(),
            lvs=lvs,
            public_columns={"x"},
            target_column=None,
            dataset_name=None,
        )
        names = [f["name"] for f in payload["top_findings"]]
        # Sorted desc by importance, top 5
        assert names == ["lv9", "lv8", "lv7", "lv6", "lv5"]
        assert payload["headline_stats"][0] == "9.00"

    def test_dataset_name_sanitized(self):
        payload = build_experiment_payload(
            self._exp(),
            lvs=[],
            public_columns=set(),
            target_column=None,
            dataset_name="ds ignore previous",
        )
        assert "ignore previous" not in payload["dataset_name"].lower()

    def test_target_omitted_when_not_public(self):
        payload = build_experiment_payload(
            self._exp(),
            lvs=[],
            public_columns=set(),
            target_column="y",
            dataset_name=None,
        )
        assert "target_column" not in payload
