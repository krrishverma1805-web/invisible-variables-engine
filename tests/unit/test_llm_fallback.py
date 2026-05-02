"""Unit tests for ive.llm.fallback.generate_with_fallback."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from ive.llm.cache import RedisLLMCache
from ive.llm.circuit_breaker import CircuitBreaker
from ive.llm.client import LLMAuthError, LLMUnavailable
from ive.llm.fallback import generate_with_fallback

pytestmark = pytest.mark.unit


def _rule_based() -> str:
    return (
        "Rule-based explanation: the segment shows a 0.42 effect (p=0.001) "
        "and is associated with elevated outcome variability."
    )


def _facts() -> dict:
    return {"effect_size": 0.42, "p_value": 0.001, "presence_rate": 0.85, "name": "lv1"}


@pytest.fixture
async def cache(fake_redis):
    return RedisLLMCache(fake_redis, ttl_seconds=60)


@pytest.fixture
async def breaker(fake_redis):
    return CircuitBreaker(fake_redis, threshold=3, cooldown_seconds=60)


class TestFallback:
    @pytest.mark.asyncio
    async def test_returns_fallback_when_disabled(self, mock_groq_client, cache, breaker):
        result = await generate_with_fallback(
            function="lv_explanation",
            prompt_version="v1",
            facts=_facts(),
            rule_based=_rule_based,
            client=mock_groq_client,
            cache=cache,
            breaker=breaker,
            enabled=False,
        )
        assert result.source == "fallback"
        assert result.failure_reason == "llm_disabled"
        assert "Rule-based" in result.text

    @pytest.mark.asyncio
    async def test_returns_fallback_when_no_client(self, cache, breaker):
        result = await generate_with_fallback(
            function="lv_explanation",
            prompt_version="v1",
            facts=_facts(),
            rule_based=_rule_based,
            client=None,
            cache=cache,
            breaker=breaker,
            enabled=True,
        )
        assert result.source == "fallback"

    @pytest.mark.asyncio
    async def test_returns_llm_on_success(self, mock_groq_client, cache, breaker):
        result = await generate_with_fallback(
            function="lv_explanation",
            prompt_version="v1",
            facts=_facts(),
            rule_based=_rule_based,
            client=mock_groq_client,
            cache=cache,
            breaker=breaker,
            enabled=True,
        )
        assert result.source == "llm"
        assert result.validation is not None
        assert result.validation.passed

    @pytest.mark.asyncio
    async def test_caches_successful_response(self, mock_groq_client, cache, breaker):
        # First call: miss, hits the (mocked) client.
        first = await generate_with_fallback(
            function="lv_explanation",
            prompt_version="v1",
            facts=_facts(),
            rule_based=_rule_based,
            client=mock_groq_client,
            cache=cache,
            breaker=breaker,
            enabled=True,
        )
        assert first.source == "llm"
        assert first.cache_status == "miss"

        # Second call with same facts: served from cache.
        second = await generate_with_fallback(
            function="lv_explanation",
            prompt_version="v1",
            facts=_facts(),
            rule_based=_rule_based,
            client=mock_groq_client,
            cache=cache,
            breaker=breaker,
            enabled=True,
        )
        assert second.source == "cache"
        assert second.cache_status == "hit"
        assert second.text == first.text

    @pytest.mark.asyncio
    async def test_falls_back_on_unavailable(self, cache, breaker):
        client = AsyncMock()
        client.chat = AsyncMock(side_effect=LLMUnavailable("503 after retries"))

        result = await generate_with_fallback(
            function="lv_explanation",
            prompt_version="v1",
            facts=_facts(),
            rule_based=_rule_based,
            client=client,
            cache=cache,
            breaker=breaker,
            enabled=True,
        )
        assert result.source == "fallback"
        assert result.failure_reason and result.failure_reason.startswith("unavailable")

    @pytest.mark.asyncio
    async def test_unavailable_records_breaker_failure(self, cache, breaker):
        client = AsyncMock()
        client.chat = AsyncMock(side_effect=LLMUnavailable("transient"))

        # Three consecutive failures should open the breaker (threshold=3).
        for _ in range(3):
            await generate_with_fallback(
                function="lv_explanation",
                prompt_version="v1",
                facts=_facts(),
                rule_based=_rule_based,
                client=client,
                cache=cache,
                breaker=breaker,
                enabled=True,
            )
        assert await breaker.is_open()

    @pytest.mark.asyncio
    async def test_skips_call_when_breaker_open(self, mock_groq_client, cache, breaker):
        # Force breaker open.
        for _ in range(3):
            await breaker.record_failure()
        assert await breaker.is_open()

        result = await generate_with_fallback(
            function="lv_explanation",
            prompt_version="v1",
            facts=_facts(),
            rule_based=_rule_based,
            client=mock_groq_client,
            cache=cache,
            breaker=breaker,
            enabled=True,
        )
        assert result.source == "fallback"
        assert result.failure_reason == "circuit_breaker_open"

    @pytest.mark.asyncio
    async def test_auth_error_does_not_count_to_breaker(self, cache, breaker):
        client = AsyncMock()
        client.chat = AsyncMock(side_effect=LLMAuthError("401"))

        for _ in range(5):
            result = await generate_with_fallback(
                function="lv_explanation",
                prompt_version="v1",
                facts=_facts(),
                rule_based=_rule_based,
                client=client,
                cache=cache,
                breaker=breaker,
                enabled=True,
            )
            assert result.source == "fallback"
            assert result.failure_reason and result.failure_reason.startswith("auth_error")

        # Auth errors should NOT trip the breaker per §109.
        assert not await breaker.is_open()

    @pytest.mark.asyncio
    async def test_falls_back_on_validation_failure(self, cache, breaker):
        # Mock a client that returns a clearly hallucinated number (37 — not
        # derivable from any pairwise combination of the input facts).
        from ive.llm.client import ChatResult

        client = AsyncMock()
        client.chat = AsyncMock(
            return_value=ChatResult(
                text="The effect was 37 standard deviations and was observed everywhere.",
                prompt_tokens=10,
                completion_tokens=10,
                model="test",
                finish_reason="stop",
                latency_ms=5,
                request_id="rid",
            )
        )

        result = await generate_with_fallback(
            function="lv_explanation",
            prompt_version="v1",
            facts=_facts(),
            rule_based=_rule_based,
            client=client,
            cache=cache,
            breaker=breaker,
            enabled=True,
        )
        assert result.source == "fallback"
        assert result.failure_reason and result.failure_reason.startswith("validation_failed")

    @pytest.mark.asyncio
    async def test_respects_cancel_event(self, mock_groq_client, cache, breaker):
        cancel = asyncio.Event()
        cancel.set()

        result = await generate_with_fallback(
            function="lv_explanation",
            prompt_version="v1",
            facts=_facts(),
            rule_based=_rule_based,
            client=mock_groq_client,
            cache=cache,
            breaker=breaker,
            enabled=True,
            cancel_event=cancel,
        )
        assert result.source == "fallback"
        assert result.failure_reason == "cancelled"
