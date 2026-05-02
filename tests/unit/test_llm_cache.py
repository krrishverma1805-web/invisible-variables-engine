"""Unit tests for ive.llm.cache."""

from __future__ import annotations

import pytest

from ive.llm.cache import (
    RedisLLMCache,
    entity_index_key,
    facts_hash,
    make_key,
    payload_schema_hash,
)

pytestmark = pytest.mark.unit


class TestKeyConstruction:
    def test_key_is_stable_across_calls(self):
        payload = {"name": "lv1", "effect_size": 0.42, "p_value": 0.001}
        a = make_key("lv_explanation", "v1", payload)
        b = make_key("lv_explanation", "v1", payload)
        assert a == b

    def test_key_changes_with_facts(self):
        a = make_key("lv_explanation", "v1", {"effect": 0.42})
        b = make_key("lv_explanation", "v1", {"effect": 0.43})
        assert a != b

    def test_key_changes_with_function(self):
        payload = {"effect": 0.42}
        a = make_key("lv_explanation", "v1", payload)
        b = make_key("pattern_summary", "v1", payload)
        assert a != b

    def test_key_changes_with_version(self):
        payload = {"effect": 0.42}
        a = make_key("lv_explanation", "v1", payload)
        b = make_key("lv_explanation", "v2", payload)
        # v2 isn't registered so template_sha returns empty SHA — still distinct prefix
        assert a != b


class TestSchemaHash:
    def test_same_keys_same_hash(self):
        a = payload_schema_hash({"a": 1, "b": 2})
        b = payload_schema_hash({"b": 99, "a": 0})
        assert a == b

    def test_added_field_changes_hash(self):
        a = payload_schema_hash({"a": 1})
        b = payload_schema_hash({"a": 1, "effect_size_ci": [0.3, 0.5]})
        assert a != b


class TestFactsHash:
    def test_value_change_changes_hash(self):
        a = facts_hash({"x": 0.42})
        b = facts_hash({"x": 0.43})
        assert a != b

    def test_key_order_does_not_change_hash(self):
        a = facts_hash({"x": 1, "y": 2})
        b = facts_hash({"y": 2, "x": 1})
        assert a == b

    def test_truncated_to_32_chars(self):
        assert len(facts_hash({"x": 1})) == 32


class TestRedisLLMCache:
    @pytest.mark.asyncio
    async def test_get_miss_returns_none(self, fake_redis):
        cache = RedisLLMCache(fake_redis, ttl_seconds=60)
        assert await cache.get("missing") is None

    @pytest.mark.asyncio
    async def test_set_then_get_roundtrip(self, fake_redis):
        cache = RedisLLMCache(fake_redis, ttl_seconds=60)
        await cache.set("k1", "value1")
        assert await cache.get("k1") == "value1"

    @pytest.mark.asyncio
    async def test_entity_index_tracks_keys(self, fake_redis):
        cache = RedisLLMCache(fake_redis, ttl_seconds=60)
        await cache.set("k1", "v1", entity_index=("dataset", "ds1"))
        await cache.set("k2", "v2", entity_index=("dataset", "ds1"))
        members = await fake_redis.smembers(entity_index_key("dataset", "ds1"))
        assert len(members) == 2

    @pytest.mark.asyncio
    async def test_delete_for_entity_clears_indexed_keys(self, fake_redis):
        cache = RedisLLMCache(fake_redis, ttl_seconds=60)
        await cache.set("k1", "v1", entity_index=("dataset", "ds1"))
        await cache.set("k2", "v2", entity_index=("dataset", "ds1"))
        await cache.set("k3", "v3", entity_index=("dataset", "ds2"))

        removed = await cache.delete_for_entity("dataset", "ds1")
        assert removed == 2

        # ds1's keys are gone, ds2's remain
        assert await cache.get("k1") is None
        assert await cache.get("k2") is None
        assert await cache.get("k3") == "v3"

    @pytest.mark.asyncio
    async def test_delete_for_unknown_entity_is_noop(self, fake_redis):
        cache = RedisLLMCache(fake_redis, ttl_seconds=60)
        removed = await cache.delete_for_entity("dataset", "never_existed")
        assert removed == 0
