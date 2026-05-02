"""Tests for the Prometheus metrics registry (plan §C4)."""

from __future__ import annotations

import pytest

from ive.observability import metrics as metrics_module
from ive.observability.metrics import (
    MetricsRegistry,
    record_fpr_sentinel,
    record_llm_call,
    record_phase_duration,
    record_validation_failure,
    reset_registry_for_tests,
    set_circuit_breaker_state,
)


@pytest.fixture(autouse=True)
def _reset_registry_singleton():
    reset_registry_for_tests()
    yield
    reset_registry_for_tests()


class TestRegistryDisabled:
    def test_no_op_when_disabled(self):
        reg = MetricsRegistry(enabled=False)
        # All calls are safe and emit nothing.
        assert reg.expose() == b""
        assert reg.content_type == "text/plain"


class TestRegistryEnabled:
    def test_phase_duration_recorded(self):
        reg = MetricsRegistry(enabled=True)
        # Substitute the singleton so the helper functions hit this registry.
        metrics_module._registry = reg
        record_phase_duration(phase="phase_2", subphase="hpo", duration_seconds=2.7)
        body = reg.expose().decode("utf-8")
        assert "ive_phase_duration_seconds" in body
        assert 'phase="phase_2"' in body
        assert 'subphase="hpo"' in body

    def test_llm_call_records_latency_and_tokens(self):
        reg = MetricsRegistry(enabled=True)
        metrics_module._registry = reg
        record_llm_call(
            function="lv_explanation",
            outcome="success",
            latency_ms=145.0,
            tokens_in=300,
            tokens_out=120,
        )
        body = reg.expose().decode("utf-8")
        assert 'ive_llm_request_total{function="lv_explanation",outcome="success"} 1.0' in body
        assert 'ive_llm_tokens_total{kind="prompt"} 300.0' in body
        assert 'ive_llm_tokens_total{kind="completion"} 120.0' in body
        assert 'ive_llm_tokens_total{kind="total"} 420.0' in body

    def test_validation_failure_increments(self):
        reg = MetricsRegistry(enabled=True)
        metrics_module._registry = reg
        record_validation_failure(reason="numeric_grounding")
        record_validation_failure(reason="numeric_grounding")
        record_validation_failure(reason="banned_phrase")
        body = reg.expose().decode("utf-8")
        assert 'ive_llm_validation_failed_total{reason="numeric_grounding"} 2.0' in body
        assert 'ive_llm_validation_failed_total{reason="banned_phrase"} 1.0' in body

    def test_circuit_breaker_state_gauge(self):
        reg = MetricsRegistry(enabled=True)
        metrics_module._registry = reg
        set_circuit_breaker_state(service="groq", state="open")
        body = reg.expose().decode("utf-8")
        assert 'ive_llm_circuit_breaker_state{service="groq"} 1.0' in body
        # Unknown state names map to closed=0 (defensive).
        set_circuit_breaker_state(service="groq", state="totally_invalid")
        body = reg.expose().decode("utf-8")
        assert 'ive_llm_circuit_breaker_state{service="groq"} 0.0' in body

    def test_fpr_sentinel(self):
        reg = MetricsRegistry(enabled=True)
        metrics_module._registry = reg
        record_fpr_sentinel(fpr=0.045, status="pass")
        body = reg.expose().decode("utf-8")
        assert "ive_fpr_sentinel_value 0.045" in body
        assert 'ive_fpr_sentinel_runs_total{status="pass"} 1.0' in body


class TestSingletonRespectsSetting:
    def test_get_registry_disabled_by_default(self, monkeypatch):
        # No metrics envvar -> defaults to disabled.
        from ive.config import get_settings

        get_settings.cache_clear()
        reset_registry_for_tests()
        # Ensure the env doesn't accidentally enable.
        monkeypatch.delenv("ENABLE_METRICS", raising=False)
        reg = metrics_module.get_registry()
        assert reg.enabled is False
        assert reg.expose() == b""

    def test_get_registry_enabled_via_env(self, monkeypatch):
        from ive.config import get_settings

        monkeypatch.setenv("ENABLE_METRICS", "true")
        get_settings.cache_clear()
        reset_registry_for_tests()
        reg = metrics_module.get_registry()
        assert reg.enabled is True
        # And it survives subsequent get_registry calls.
        assert metrics_module.get_registry() is reg
