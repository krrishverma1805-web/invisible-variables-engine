"""Unit tests for ive.llm.circuit_breaker."""

from __future__ import annotations

import pytest

from ive.llm.circuit_breaker import CircuitBreaker

pytestmark = pytest.mark.unit


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, fake_redis):
        breaker = CircuitBreaker(fake_redis, threshold=3, cooldown_seconds=10)
        assert not await breaker.is_open()

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, fake_redis):
        breaker = CircuitBreaker(fake_redis, threshold=3, cooldown_seconds=10)

        await breaker.record_failure()
        assert not await breaker.is_open()
        await breaker.record_failure()
        assert not await breaker.is_open()
        await breaker.record_failure()
        assert await breaker.is_open()

    @pytest.mark.asyncio
    async def test_success_resets_counter(self, fake_redis):
        breaker = CircuitBreaker(fake_redis, threshold=3, cooldown_seconds=10)
        await breaker.record_failure()
        await breaker.record_failure()
        await breaker.record_success()
        # next two failures should not open
        await breaker.record_failure()
        await breaker.record_failure()
        assert not await breaker.is_open()

    @pytest.mark.asyncio
    async def test_separate_scopes_are_independent(self, fake_redis):
        a = CircuitBreaker(fake_redis, scope="a", threshold=2, cooldown_seconds=10)
        b = CircuitBreaker(fake_redis, scope="b", threshold=2, cooldown_seconds=10)
        await a.record_failure()
        await a.record_failure()
        assert await a.is_open()
        assert not await b.is_open()
