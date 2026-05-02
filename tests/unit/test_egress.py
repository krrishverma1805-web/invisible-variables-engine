"""Unit tests for ive.auth.egress."""

from __future__ import annotations

import pytest

from ive.auth.egress import (
    EgressDecision,
    evaluate_lv_egress,
    filter_payload_columns,
)

pytestmark = pytest.mark.unit


class TestEvaluateLvEgress:
    def test_all_columns_public_allows(self):
        d = evaluate_lv_egress(["age", "income"], {"age", "income", "ssn"})
        assert d.allowed
        assert d.reason is None
        assert d.blocked_columns == ()

    def test_any_non_public_blocks(self):
        d = evaluate_lv_egress(["age", "income", "ssn"], {"age", "income"})
        assert not d.allowed
        assert d.reason == "pii_protection_per_column"
        assert d.blocked_columns == ("ssn",)

    def test_empty_referenced_allows(self):
        d = evaluate_lv_egress([], {"any"})
        assert d.allowed

    def test_no_public_columns_blocks_everything(self):
        d = evaluate_lv_egress(["a", "b"], set())
        assert not d.allowed
        assert set(d.blocked_columns) == {"a", "b"}

    def test_duplicate_referenced_columns_dedupes(self):
        d = evaluate_lv_egress(["a", "a", "b"], {"a"})
        assert not d.allowed
        assert d.blocked_columns == ("b",)

    def test_decision_is_immutable(self):
        d = EgressDecision(allowed=True)
        with pytest.raises(Exception):  # noqa: B017 - dataclass(frozen=True)
            d.allowed = False  # type: ignore[misc]


class TestFilterPayloadColumns:
    def test_strips_non_public_from_allowed_columns(self):
        payload = {"allowed_columns": ["age", "ssn", "income"], "stat": 0.42}
        out = filter_payload_columns(payload, {"age", "income"})
        assert out["allowed_columns"] == ["age", "income"]

    def test_no_allowed_columns_field_passes_through(self):
        payload = {"name": "lv1", "stat": 0.42}
        out = filter_payload_columns(payload, {"a"})
        assert out == payload

    def test_empty_public_set_strips_all(self):
        payload = {"allowed_columns": ["a", "b"]}
        out = filter_payload_columns(payload, set())
        assert out["allowed_columns"] == []

    def test_does_not_mutate_input(self):
        payload = {"allowed_columns": ["a", "b"]}
        original = list(payload["allowed_columns"])
        filter_payload_columns(payload, {"a"})
        assert payload["allowed_columns"] == original
