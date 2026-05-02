"""Verify production code paths actually call the metrics helpers (Phase C4).

Each test installs a real enabled MetricsRegistry via the singleton, exercises
the instrumented call site through a thin spy, and asserts the expected
metric line appears in the exposed output.
"""

from __future__ import annotations

import asyncio

import pytest

from ive.observability import metrics as metrics_module
from ive.observability.metrics import (
    MetricsRegistry,
    reset_registry_for_tests,
)


@pytest.fixture
def enabled_registry():
    reset_registry_for_tests()
    reg = MetricsRegistry(enabled=True)
    metrics_module._registry = reg
    yield reg
    reset_registry_for_tests()


class _FakeRedis:
    """Minimal in-memory async Redis surface for the cache + breaker tests."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    async def get(self, key):
        v = self.store.get(key)
        return v

    async def set(self, key, value, *, ex=None):  # noqa: ARG002
        self.store[key] = value

    async def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                n += 1
            if k in self.sets:
                del self.sets[k]
                n += 1
        return n

    async def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(values)
        return len(values)

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def incr(self, key):
        v = int(self.store.get(key, "0")) + 1
        self.store[key] = str(v)
        return v

    async def expire(self, key, time):  # noqa: ARG002
        return True


class TestCacheMetrics:
    def test_hit_and_miss_recorded(self, enabled_registry):
        from ive.llm.cache import RedisLLMCache

        cache = RedisLLMCache(_FakeRedis(), ttl_seconds=10)

        async def run():
            assert await cache.get("missing") is None
            await cache.set("k", "v")
            assert await cache.get("k") == "v"

        asyncio.run(run())
        body = enabled_registry.expose().decode("utf-8")
        assert 'ive_llm_cache_total{outcome="miss"} 1.0' in body
        assert 'ive_llm_cache_total{outcome="hit"} 1.0' in body


class TestCircuitBreakerMetrics:
    def test_open_and_close_state_recorded(self, enabled_registry):
        from ive.llm.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(
            _FakeRedis(), scope="groq", threshold=2, cooldown_seconds=10
        )

        async def run():
            opened = await breaker.record_failure()
            assert opened is False
            opened = await breaker.record_failure()
            assert opened is True
            await breaker.record_success()

        asyncio.run(run())
        body = enabled_registry.expose().decode("utf-8")
        # Final state after record_success is closed.
        assert 'ive_llm_circuit_breaker_state{service="groq"} 0.0' in body


class TestFallbackMetrics:
    def test_fallback_emits_reason(self, enabled_registry):
        from ive.llm.fallback import _fallback

        result = _fallback(lambda: "rule-based output", reason="circuit_breaker_open")
        assert result.source == "fallback"
        body = enabled_registry.expose().decode("utf-8")
        assert (
            'ive_llm_fallback_total{reason="circuit_breaker_open"} 1.0' in body
        )

    def test_validation_failure_recorded(self, enabled_registry):
        from ive.llm.fallback import _fallback
        from ive.llm.types import ValidationReport

        report = ValidationReport(
            passed=False,
            failures=["banned_phrase", "numeric_grounding"],
            rule="banned_phrase",
        )
        _fallback(lambda: "x", reason="validation_failed", validation=report)
        body = enabled_registry.expose().decode("utf-8")
        assert (
            'ive_llm_validation_failed_total{reason="banned_phrase"} 1.0' in body
        )
        assert (
            'ive_llm_validation_failed_total{reason="numeric_grounding"} 1.0'
            in body
        )


class TestLLMClientMetrics:
    def test_success_records_tokens_and_latency(self, enabled_registry):
        # Hit the helper directly — full client path requires a respx mock.
        from ive.llm.client import _emit_metric

        _emit_metric(
            function="lv_explanation",
            outcome="success",
            latency_ms=120,
            tokens_in=240,
            tokens_out=80,
        )
        body = enabled_registry.expose().decode("utf-8")
        assert (
            'ive_llm_request_total{function="lv_explanation",outcome="success"} 1.0'
            in body
        )
        assert 'ive_llm_tokens_total{kind="prompt"} 240.0' in body
        assert 'ive_llm_tokens_total{kind="completion"} 80.0' in body
        assert 'ive_llm_tokens_total{kind="total"} 320.0' in body


class TestPipelinePhaseMetricsHelper:
    def test_phase_duration_helper_records(self, enabled_registry):
        from ive.observability.metrics import record_phase_duration

        record_phase_duration(phase="model", duration_seconds=2.5)
        body = enabled_registry.expose().decode("utf-8")
        assert 'phase="model"' in body
        # bucket count incremented at >= 5s? we observed 2.5 so bucket 5.0 +.
        assert "ive_phase_duration_seconds_count" in body
